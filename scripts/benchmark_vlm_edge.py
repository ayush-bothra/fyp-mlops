from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from transformers import AutoProcessor, BitsAndBytesConfig


@dataclass
class HardwareProfile:
    name: str
    ram_limit_label: str
    memory_max: str
    cpu_quota: str
    cpu_cores: int


EDGE_PROFILES: list[HardwareProfile] = [
    HardwareProfile(
        name="Pi 5-A",
        ram_limit_label="4 GB",
        memory_max="4G",
        cpu_quota="400%",
        cpu_cores=4,
    ),
    HardwareProfile(
        name="Pi 5-B",
        ram_limit_label="2 GB",
        memory_max="2G",
        cpu_quota="400%",
        cpu_cores=4,
    ),
    HardwareProfile(
        name="Pi 5-C",
        ram_limit_label="1 GB",
        memory_max="1G",
        cpu_quota="400%",
        cpu_cores=4,
    ),
    HardwareProfile(
        name="Zero 2 W",
        ram_limit_label="512 MB",
        memory_max="512M",
        cpu_quota="100%",
        cpu_cores=1,
    ),
]


@dataclass
class BenchmarkResult:
    profile: str
    ram_limit: str
    cpu_quota: str
    device: str
    quantization: str
    mode: str
    status: str
    peak_rss_mb: float
    peak_vram_mb: float
    load_seconds: float
    execution_seconds: float
    total_seconds: float
    initial_loss: float | None
    final_loss: float | None
    loss_reduction: float | None
    throughput_fps: float
    error_message: str | None


def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def load_core50_sample_frames(
    dataset_dir: str | Path,
    num_frames: int,
) -> list[Image.Image]:
    dataset_path = Path(dataset_dir)
    image_paths = sorted(dataset_path.glob("s1/o*/*.png"))
    if not image_paths:
        image_paths = sorted(dataset_path.glob("**/*.png"))

    if image_paths:
        selected = image_paths[:num_frames]
        return [Image.open(path).convert("RGB") for path in selected]

    return [
        Image.new("RGB", (128, 128), color=(30 * i % 255, 60 * i % 255, 90 * i % 255))
        for i in range(num_frames)
    ]


def prepare_core50_training_batches(
    processor: Any,
    dataset_dir: str | Path,
    num_samples: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    frames = load_core50_sample_frames(dataset_dir, num_samples)
    text_prompt = "<image>Describe the object in this image."
    target_device = torch.device(device)
    batch_list: list[dict[str, torch.Tensor]] = []

    for frame in frames:
        inputs = processor(
            text=text_prompt,
            images=frame,
            return_tensors="pt",
        )
        batch_inputs = {key: value.to(target_device) for key, value in inputs.items()}
        batch_list.append(batch_inputs)

    return batch_list


def load_cpu_lora_vlm(
    model_id: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> tuple[Any, Any]:
    if get_peft_model is None or LoraConfig is None:
        raise ImportError(
            "peft is required for LoRA training. Install with: pip install peft"
        )

    processor = AutoProcessor.from_pretrained(model_id)
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    base_model.to("cpu")

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()
    return peft_model, processor


def load_cuda_quantized_lora_vlm(
    model_id: str,
    quantization: str = "4bit",
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> tuple[Any, Any]:
    if get_peft_model is None or LoraConfig is None:
        raise ImportError(
            "peft is required for LoRA training. Install with: pip install peft"
        )

    processor = AutoProcessor.from_pretrained(model_id)
    compute_dtype = torch.float16

    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()
    return peft_model, processor


def execute_training_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch_inputs: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad()
    outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training_worker(
    model_id: str,
    device: str,
    quantization: str,
    num_steps: int,
    dataset_dir: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    output_json_path: Path,
) -> None:
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_total = time.perf_counter()

    start_load = time.perf_counter()
    if device == "cpu":
        model, processor = load_cpu_lora_vlm(
            model_id=model_id,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
    else:
        model, processor = load_cuda_quantized_lora_vlm(
            model_id=model_id,
            quantization=quantization,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
    load_time = time.perf_counter() - start_load

    batches = prepare_core50_training_batches(
        processor=processor,
        dataset_dir=dataset_dir,
        num_samples=num_steps,
        device=device,
    )

    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    step_losses: list[float] = []
    start_train = time.perf_counter()

    for step_index, batch in enumerate(batches):
        loss_val = execute_training_step(
            model=model,
            optimizer=optimizer,
            batch_inputs=batch,
        )
        step_losses.append(loss_val)

    train_time = time.perf_counter() - start_train
    total_time = time.perf_counter() - start_total

    peak_rss = get_peak_memory_mb()
    peak_vram = get_peak_vram_mb()
    initial_loss = step_losses[0] if step_losses else 0.0
    final_loss = step_losses[-1] if step_losses else 0.0
    loss_reduction = round(initial_loss - final_loss, 4)
    fps = round(len(batches) / train_time, 2) if train_time > 0 else 0.0

    payload = {
        "status": "PASS",
        "peak_rss_mb": round(peak_rss, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "load_seconds": round(load_time, 3),
        "execution_seconds": round(train_time, 3),
        "total_seconds": round(total_time, 3),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_reduction": loss_reduction,
        "throughput_fps": fps,
        "error_message": None,
    }

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def run_scoring_worker(
    model_id: str,
    device: str,
    quantization: str,
    num_frames: int,
    dataset_dir: str,
    output_json_path: Path,
) -> None:
    from src.vlm.scorer import VLMScorer

    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_total = time.perf_counter()

    start_load = time.perf_counter()
    scorer = VLMScorer(
        model_id=model_id,
        device=device,
        quantization=quantization,
    )
    load_time = time.perf_counter() - start_load

    frames = load_core50_sample_frames(
        dataset_dir=dataset_dir,
        num_frames=num_frames,
    )

    start_infer = time.perf_counter()
    scores = scorer.batch_score(frames)
    infer_time = time.perf_counter() - start_infer

    total_time = time.perf_counter() - start_total
    peak_rss = get_peak_memory_mb()
    peak_vram = get_peak_vram_mb()
    avg_score = float(np.mean([s.usefulness_score for s in scores])) if scores else 0.0
    fps = round(len(frames) / infer_time, 2) if infer_time > 0 else 0.0

    payload = {
        "status": "PASS",
        "peak_rss_mb": round(peak_rss, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "load_seconds": round(load_time, 3),
        "execution_seconds": round(infer_time, 3),
        "total_seconds": round(total_time, 3),
        "initial_loss": None,
        "final_loss": None,
        "loss_reduction": round(avg_score, 4),
        "throughput_fps": fps,
        "error_message": None,
    }

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def is_systemd_run_available() -> bool:
    return shutil.which("systemd-run") is not None


def execute_profile_benchmark(
    profile: HardwareProfile,
    mode: str,
    device: str,
    quantization: str,
    num_items: int,
    dataset_dir: str,
    lora_r: int,
    lora_alpha: int,
    results_dir: Path,
    use_systemd: bool,
) -> BenchmarkResult:
    results_dir.mkdir(parents=True, exist_ok=True)
    worker_output_file = (
        results_dir / f"worker_{profile.name.replace(' ', '_')}_{device}_{mode}.json"
    )

    if worker_output_file.exists():
        worker_output_file.unlink()

    worker_cmd = [
        sys.executable,
        "-m",
        "scripts.benchmark_vlm_edge",
        "--worker",
        "--mode",
        mode,
        "--device",
        device,
        "--quantization",
        quantization,
        "--num-items",
        str(num_items),
        "--dataset-dir",
        dataset_dir,
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
        "--output-file",
        str(worker_output_file),
    ]

    if use_systemd and is_systemd_run_available():
        full_command = [
            "systemd-run",
            "--user",
            "--scope",
            f"--property=MemoryMax={profile.memory_max}",
            "--property=MemorySwapMax=0M",
            f"--property=CPUQuota={profile.cpu_quota}",
            *worker_cmd,
        ]
    else:
        full_command = worker_cmd

    try:
        worker_env = os.environ.copy()
        if device == "cpu":
            worker_env["CUDA_VISIBLE_DEVICES"] = ""

        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False,
            env=worker_env,
        )

        if process.returncode in (137, -9):
            return BenchmarkResult(
                profile=profile.name,
                ram_limit=profile.ram_limit_label,
                cpu_quota=profile.cpu_quota,
                device=device,
                quantization=quantization,
                mode=mode,
                status="OOM_KILLED",
                peak_rss_mb=0.0,
                peak_vram_mb=0.0,
                load_seconds=0.0,
                execution_seconds=0.0,
                total_seconds=0.0,
                initial_loss=None,
                final_loss=None,
                loss_reduction=None,
                throughput_fps=0.0,
                error_message="Process killed by cgroup MemoryMax limit (Out of Memory)",
            )

        if process.returncode != 0:
            error_details = process.stderr.strip() or process.stdout.strip()
            last_error_line = (
                error_details.splitlines()[-1]
                if error_details
                else f"Exited with code {process.returncode}"
            )
            return BenchmarkResult(
                profile=profile.name,
                ram_limit=profile.ram_limit_label,
                cpu_quota=profile.cpu_quota,
                device=device,
                quantization=quantization,
                mode=mode,
                status="FAILED",
                peak_rss_mb=0.0,
                peak_vram_mb=0.0,
                load_seconds=0.0,
                execution_seconds=0.0,
                total_seconds=0.0,
                initial_loss=None,
                final_loss=None,
                loss_reduction=None,
                throughput_fps=0.0,
                error_message=last_error_line,
            )

        if worker_output_file.exists():
            with open(worker_output_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            return BenchmarkResult(
                profile=profile.name,
                ram_limit=profile.ram_limit_label,
                cpu_quota=profile.cpu_quota,
                device=device,
                quantization=quantization,
                mode=mode,
                status=data.get("status", "PASS"),
                peak_rss_mb=data.get("peak_rss_mb", 0.0),
                peak_vram_mb=data.get("peak_vram_mb", 0.0),
                load_seconds=data.get("load_seconds", 0.0),
                execution_seconds=data.get("execution_seconds", 0.0),
                total_seconds=data.get("total_seconds", 0.0),
                initial_loss=data.get("initial_loss"),
                final_loss=data.get("final_loss"),
                loss_reduction=data.get("loss_reduction"),
                throughput_fps=data.get("throughput_fps", 0.0),
                error_message=data.get("error_message"),
            )

        return BenchmarkResult(
            profile=profile.name,
            ram_limit=profile.ram_limit_label,
            cpu_quota=profile.cpu_quota,
            device=device,
            quantization=quantization,
            mode=mode,
            status="FAILED",
            peak_rss_mb=0.0,
            peak_vram_mb=0.0,
            load_seconds=0.0,
            execution_seconds=0.0,
            total_seconds=0.0,
            initial_loss=None,
            final_loss=None,
            loss_reduction=None,
            throughput_fps=0.0,
            error_message="Worker output payload missing",
        )

    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        return BenchmarkResult(
            profile=profile.name,
            ram_limit=profile.ram_limit_label,
            cpu_quota=profile.cpu_quota,
            device=device,
            quantization=quantization,
            mode=mode,
            status="ERROR",
            peak_rss_mb=0.0,
            peak_vram_mb=0.0,
            load_seconds=0.0,
            execution_seconds=0.0,
            total_seconds=0.0,
            initial_loss=None,
            final_loss=None,
            loss_reduction=None,
            throughput_fps=0.0,
            error_message=str(exc),
        )


def print_results_table(results: list[BenchmarkResult]) -> None:
    header = f"{'Profile':<10} | {'RAM Limit':<10} | {'Device':<6} | {'Mode':<6} | {'Status':<11} | {'Peak RAM':<10} | {'Peak VRAM':<10} | {'Exec (s)':<9} | {'Init Loss':<9} | {'End Loss':<9}"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for r in results:
        ram_str = f"{r.peak_rss_mb:.1f} MB" if r.peak_rss_mb > 0 else "N/A"
        vram_str = f"{r.peak_vram_mb:.1f} MB" if r.peak_vram_mb > 0 else "N/A"
        exec_str = f"{r.execution_seconds:.2f}" if r.execution_seconds > 0 else "N/A"
        init_loss_str = f"{r.initial_loss:.3f}" if r.initial_loss is not None else "N/A"
        end_loss_str = f"{r.final_loss:.3f}" if r.final_loss is not None else "N/A"
        print(
            f"{r.profile:<10} | {r.ram_limit:<10} | {r.device:<6} | {r.mode:<6} | {r.status:<11} | {ram_str:<10} | {vram_str:<10} | {exec_str:<9} | {init_loss_str:<9} | {end_loss_str:<9}"
        )

    print(divider)


def save_reports(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "edge_vlm_benchmark.json"
    markdown_path = output_dir / "edge_vlm_benchmark.md"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump([asdict(r) for r in results], file, indent=2)

    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write("# Edge VLM Hardware Benchmark Report\n\n")
        file.write(
            "| Profile | RAM Limit | Device | Mode | Status | Peak RAM | Peak VRAM | Load (s) | Exec (s) | Init Loss | Final Loss |\n"
        )
        file.write(
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        )
        for r in results:
            ram_str = f"{r.peak_rss_mb:.1f} MB" if r.peak_rss_mb > 0 else "N/A"
            vram_str = f"{r.peak_vram_mb:.1f} MB" if r.peak_vram_mb > 0 else "N/A"
            load_str = f"{r.load_seconds:.2f}" if r.load_seconds > 0 else "N/A"
            exec_str = (
                f"{r.execution_seconds:.2f}" if r.execution_seconds > 0 else "N/A"
            )
            init_loss_str = (
                f"{r.initial_loss:.3f}" if r.initial_loss is not None else "N/A"
            )
            end_loss_str = f"{r.final_loss:.3f}" if r.final_loss is not None else "N/A"
            file.write(
                f"| {r.profile} | {r.ram_limit} | {r.device} | {r.mode} | {r.status} | {ram_str} | {vram_str} | {load_str} | {exec_str} | {init_loss_str} | {end_loss_str} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge hardware VLM benchmark harness")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "score"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument("--num-items", type=int, default=3)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw/core50/core50_128x128",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/benchmarks")
    parser.add_argument("--no-systemd", action="store_true")

    args = parser.parse_args()

    if args.worker:
        if not args.output_file:
            raise ValueError("Worker mode requires --output-file")

        if args.mode == "train":
            run_training_worker(
                model_id=args.model_id,
                device=args.device,
                quantization=args.quantization,
                num_steps=args.num_items,
                dataset_dir=args.dataset_dir,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                learning_rate=args.lr,
                output_json_path=Path(args.output_file),
            )
        else:
            run_scoring_worker(
                model_id=args.model_id,
                device=args.device,
                quantization=args.quantization,
                num_frames=args.num_items,
                dataset_dir=args.dataset_dir,
                output_json_path=Path(args.output_file),
            )
        return

    results_dir = Path(args.output_dir)
    results: list[BenchmarkResult] = []

    print(
        f"Starting Edge VLM Benchmark (Mode: {args.mode.upper()}, Device: {args.device.upper()}, Quant: {args.quantization})...\n"
    )

    for profile in EDGE_PROFILES:
        print(
            f"--> Running profile: {profile.name} (RAM: {profile.ram_limit_label}, CPU: {profile.cpu_quota})..."
        )
        res = execute_profile_benchmark(
            profile=profile,
            mode=args.mode,
            device=args.device,
            quantization=args.quantization,
            num_items=args.num_items,
            dataset_dir=args.dataset_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            results_dir=results_dir,
            use_systemd=not args.no_systemd,
        )
        if args.mode == "train":
            loss_info = (
                f"Loss: {res.initial_loss:.3f} -> {res.final_loss:.3f}"
                if res.initial_loss is not None
                else "No loss recorded"
            )
            print(
                f"    Result: {res.status} (Peak RAM: {res.peak_rss_mb:.1f} MB, {loss_info})"
            )
        else:
            print(
                f"    Result: {res.status} (Peak RAM: {res.peak_rss_mb:.1f} MB, FPS: {res.throughput_fps:.1f})"
            )
        results.append(res)

    print_results_table(results)
    save_reports(results, results_dir)
    print(f"\nBenchmark reports saved in: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
