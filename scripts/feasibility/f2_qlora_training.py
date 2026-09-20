"""
F2 — QLoRA training feasibility test.

Runs ONE configuration end to end and records a PASS/FAIL/OOM result.
Does NOT auto-retry with reduced settings on failure -- per project rules,
a failure is a valid, recorded scientific result, not something to paper
over by silently shrinking the experiment until it passes.

Usage:
    python -m scripts.feasibility.f2_qlora_training --config configs/feasibility.yaml
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from PIL import Image
from pathlib import Path

import torch

from src.data.core50 import scan_core50
from src.data.splits import select_deterministic_subset
from src.telemetry.logger import TelemetryLogger
from src.vlm.loader import audit_trainable_parameters, load_quantized_vlm_for_training
from src.vlm.quantization import apply_vram_cap
from src.vlm.training import (
    _caption_for_sample,
    build_supervised_example,
    evaluate_validation_loss,
    prepare_supervised_batches,
    run_training_loop,
    stream_supervised_batches,
    evaluate_validation_loss_by_category,
)

DEFAULT_CONFIG = {
    "model": {"id": "HuggingFaceTB/SmolVLM-256M-Instruct"},
    "dataset": {
        "path": "data/raw/core50/core50_128x128",
        "object_ids": [1, 2],
        "train_session": 1,
        "val_session": 3,
        "num_train": 32,
        "num_val": 12,
        "seed": 42,
    },
    "qlora": {"r": 8, "alpha": 16, "dropout": 0.05, "compute_dtype": "float16"},
    "training": {"learning_rate": 1e-4, "vram_fraction": 0.2, "device_index": 0},
}


def load_config(config_path: str | None) -> dict:
    if config_path is None:
        return DEFAULT_CONFIG
    try:
        import yaml
    except ImportError:
        print(
            f"WARNING: PyYAML not installed, ignoring --config {config_path} "
            "and using built-in defaults. Install with: pip install pyyaml",
            file=sys.stderr,
        )
        return DEFAULT_CONFIG

    with open(config_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024  # Linux reports ru_maxrss in KB


def resolve_compute_dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported compute_dtype '{name}', expected one of {list(mapping)}")
    return mapping[name]


def coerce_numeric_config_fields(training_cfg: dict) -> dict:
    """
    Defensive type coercion for numeric config fields.

    PyYAML's default resolver has a well-known footgun: scientific
    notation without an explicit decimal point (e.g. "1e-4") is parsed
    as a STRING, not a float, because the implicit-float regex requires
    a "." in the mantissa. "1.0e-4" parses correctly; "1e-4" silently
    becomes the string "1e-4".

    Rather than rely on every config file being written correctly
    forever, coerce and validate these fields explicitly here, so a bad
    value fails loudly with a clear message at config-load time -- not
    three stack frames deep inside torch.optim's internal validation,
    which is what happens if this is skipped.
    """
    numeric_fields = ["learning_rate", "vram_fraction"]
    coerced = dict(training_cfg)
    for field in numeric_fields:
        value = coerced.get(field)
        try:
            coerced[field] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"config['training']['{field}'] = {value!r} is not a valid "
                f"number (got type {type(value).__name__}). If this is "
                "scientific notation in YAML, make sure it includes a "
                "decimal point (e.g. 1.0e-4, not 1e-4) -- PyYAML parses "
                "the latter as a string."
            ) from exc
    return coerced


def run_f2(config: dict, output_path: Path) -> dict:
    model_id = config["model"]["id"]
    dataset_cfg = config["dataset"]
    qlora_cfg = config["qlora"]
    training_cfg = coerce_numeric_config_fields(config["training"])

    result: dict = {
        "model": model_id,
        "quantization": "nf4_double_quant",
        "lora_configuration": {
            "r": qlora_cfg["r"],
            "alpha": qlora_cfg["alpha"],
            "dropout": qlora_cfg["dropout"],
            "target_modules": "model.text_model.layers.*.self_attn.{q_proj,v_proj}",
        },
        "gpu": None,
        "peak_vram_mb": None,
        "peak_ram_mb": None,
        "model_loading_time_seconds": None,
        "training_time_seconds": None,
        "time_per_step_seconds": None,
        "throughput_samples_per_sec": None,
        "initial_loss": None,
        "final_loss": None,
        "loss_reduction": None,
        "validation_loss": None,
        "parameter_audit": None,
        "vram_cap_info": None,
        "dataset_split": None,
        "input_shapes": None,
        "status": None,
        "error_message": None,
    }

    try:
        # 1. Enforce the VRAM cap BEFORE any CUDA allocation.
        vram_cap_info = apply_vram_cap(
            fraction=training_cfg["vram_fraction"],
            device_index=training_cfg["device_index"],
        )
        result["vram_cap_info"] = vram_cap_info
        result["gpu"] = vram_cap_info["device_name"]

        # 2. Deterministic, session-based (non-leaky) data split.
        samples = scan_core50(dataset_cfg["path"])
        split = select_deterministic_subset(
            samples=samples,
            object_ids=dataset_cfg["object_ids"],
            train_session=dataset_cfg["train_session"],
            val_session=dataset_cfg["val_session"],
            num_train=dataset_cfg["num_train"],
            num_val=dataset_cfg["num_val"],
            seed=dataset_cfg["seed"],
        )
        result["dataset_split"] = {
            "object_ids": split.object_ids,
            "train_session": split.train_session,
            "val_session": split.val_session,
            "num_train": len(split.train),
            "num_val": len(split.val),
            "seed": split.seed,
        }

        # 3. Load quantized model + attach scoped LoRA.
        compute_dtype = resolve_compute_dtype(qlora_cfg["compute_dtype"])
        start_load = time.perf_counter()
        model, processor = load_quantized_vlm_for_training(
            model_id=model_id,
            lora_r=qlora_cfg["r"],
            lora_alpha=qlora_cfg["alpha"],
            lora_dropout=qlora_cfg["dropout"],
            compute_dtype=compute_dtype,
            device_index=training_cfg["device_index"],
        )
        load_time = time.perf_counter() - start_load
        result["model_loading_time_seconds"] = round(load_time, 3)

        # 4. Parameter audit -- raises if anything is wrong, does not
        #    silently continue with a misconfigured LoRA attachment.
        result["parameter_audit"] = audit_trainable_parameters(model)

        # 5. Build supervised batches. Training streams one sample at a
        #    time instead of preloading everything onto the GPU up front,
        #    so VRAM usage reflects per-step cost, not dataset size.
        device = torch.device(f"cuda:{training_cfg['device_index']}")

        first_sample = split.train[0]
        peek_image = Image.open(first_sample.path).convert("RGB")
        peek_caption = _caption_for_sample(first_sample)
        peek_batch = build_supervised_example(
            processor=processor, image=peek_image, caption=peek_caption, device=device
        )
        result["input_shapes"] = {
            "pixel_values": list(peek_batch["pixel_values"].shape),
            "input_ids": list(peek_batch["input_ids"].shape),
        }

        train_batches = lambda: stream_supervised_batches(
            processor, split.train, device, shuffle=True, seed=dataset_cfg["seed"]
        )
        val_batches = prepare_supervised_batches(processor, split.val, device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=training_cfg["learning_rate"])

        # 6. Train.
        with TelemetryLogger(output_path="results/telemetry/f2_run.csv", episode=output_path.stem):
            train_result = run_training_loop(
                model, optimizer, train_batches, epochs=training_cfg.get("epochs", 3)
            )
        result["training_time_seconds"] = round(train_result.train_seconds, 3)
        result["time_per_step_seconds"] = round(train_result.avg_step_seconds, 4)
        result["throughput_samples_per_sec"] = (
            round(len(split.train) / train_result.train_seconds, 3)
            if train_result.train_seconds > 0
            else 0.0
        )
        result["initial_loss"] = round(train_result.initial_loss, 4)
        result["final_loss"] = round(train_result.final_loss, 4)
        result["loss_reduction"] = round(train_result.loss_reduction, 4)

        # 7. Validate on a DIFFERENT session (no leakage).
        result["validation_loss"] = round(evaluate_validation_loss(model, val_batches), 4)
        result["validation_loss_by_category"] = evaluate_validation_loss_by_category(
            model, val_batches, [s.category_name for s in split.val]
        )

        # 8. Save the trained LoRA adapter to disk. F1 needs an actual
        #    checkpoint to load -- without this, the trained weights are
        #    discarded when this process exits.
        adapter_output_dir = output_path.parent / "adapter" / output_path.stem
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_output_dir))
        processor.save_pretrained(str(adapter_output_dir))
        result["adapter_path"] = str(adapter_output_dir.resolve())

        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(training_cfg["device_index"]) / (1024 * 1024), 2)
        result["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(training_cfg["device_index"]) / (1024 * 1024), 2)
        result["peak_ram_mb"] = round(get_peak_rss_mb(), 2)

        result["status"] = "PASS" if train_result.loss_reduction > 0 else "FAIL_NO_LOSS_REDUCTION"

    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "OOM"
        result["error_message"] = str(exc)
        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(training_cfg.get("device_index", 0)) / (1024 * 1024), 2)
        result["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(training_cfg.get("device_index", 0)) / (1024 * 1024), 2)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must still be recorded as JSON, not crash silently
        result["status"] = "ERROR"
        result["error_message"] = f"{type(exc).__name__}: {exc}\n\nFull traceback:\n{traceback.format_exc()}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="F2: QLoRA training feasibility test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/f2/qlora_training.json")
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=None,
        help="Override training.vram_fraction from the config, without editing the YAML file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.vram_fraction is not None:
        config["training"]["vram_fraction"] = args.vram_fraction

    result = run_f2(config, Path(args.output))

    print(json.dumps(result, indent=2, default=str))
    print(f"\nStatus: {result['status']}")
    if result["error_message"]:
        print(f"Error: {result['error_message']}")


if __name__ == "__main__":
    main()
