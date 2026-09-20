"""
Thermal transient probe.

Not an F1-F4 feasibility test -- produces no PASS/FAIL result. Its only
job is to generate one continuous telemetry episode containing several
repeated load-then-idle cycles, so we can check whether GPU temperature
actually rises and falls fast enough, and often enough, during real
retraining bursts to justify a learned forecaster over a fixed
threshold rule.

Usage:
    python -m scripts.feasibility.thermal_probe --config configs/feasibility.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from scripts.feasibility.f2_qlora_training import (
    coerce_numeric_config_fields,
    load_config,
    resolve_compute_dtype,
)
from src.data.core50 import scan_core50
from src.data.splits import select_deterministic_subset
from src.telemetry.logger import TelemetryLogger
from src.vlm.loader import load_quantized_vlm_for_training
from src.vlm.quantization import apply_vram_cap
from src.vlm.training import run_training_loop, stream_supervised_batches


def run_probe(
    config: dict,
    num_bursts: int,
    idle_seconds: float,
    samples_per_burst: int,
    output_csv: str,
) -> None:
    dataset_cfg = config["dataset"]
    qlora_cfg = config["qlora"]
    training_cfg = coerce_numeric_config_fields(config["training"])

    apply_vram_cap(
        fraction=training_cfg["vram_fraction"],
        device_index=training_cfg["device_index"],
    )

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
    burst_samples = split.train[:samples_per_burst]

    compute_dtype = resolve_compute_dtype(qlora_cfg["compute_dtype"])
    model, processor = load_quantized_vlm_for_training(
        model_id=config["model"]["id"],
        lora_r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["alpha"],
        lora_dropout=qlora_cfg["dropout"],
        compute_dtype=compute_dtype,
        device_index=training_cfg["device_index"],
    )

    device = torch.device(f"cuda:{training_cfg['device_index']}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_cfg["learning_rate"])

    episode_name = Path(output_csv).stem
    with TelemetryLogger(output_path=output_csv, episode=episode_name) as logger:
        for burst_index in range(num_bursts):
            logger.set_pipeline_state(queue_size=burst_index, latency_ms=1.0)
            print(f"burst {burst_index}: training on {len(burst_samples)} samples")

            train_batches = lambda: stream_supervised_batches(
                processor,
                burst_samples,
                device,
                shuffle=True,
                seed=dataset_cfg["seed"] + burst_index,
            )
            run_training_loop(model, optimizer, train_batches, epochs=1)

            print(f"burst {burst_index}: idling for {idle_seconds}s")
            logger.set_pipeline_state(queue_size=burst_index, latency_ms=0.0)
            time.sleep(idle_seconds)

    print(f"done, telemetry saved to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated load/idle thermal transient probe"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num-bursts", type=int, default=6)
    parser.add_argument("--idle-seconds", type=float, default=45.0)
    parser.add_argument("--samples-per-burst", type=int, default=150)
    parser.add_argument(
        "--output", type=str, default="results/telemetry/thermal_transient_probe.csv"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_probe(
        config, args.num_bursts, args.idle_seconds, args.samples_per_burst, args.output
    )


if __name__ == "__main__":
    main()
