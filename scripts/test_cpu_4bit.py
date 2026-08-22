from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from PIL import Image
from transformers import (
    AutoModelForImageTextToText as AutoModelForVision2Seq,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET_DIR = "data/raw/core50/core50_128x128"


def get_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        return usage / (1024 * 1024)

    return usage / 1024


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

    raise FileNotFoundError(f"No CORe50 images found in {dataset_path}")


def load_cpu_4bit_vlm(
    model_id: str,
    compute_dtype: torch.dtype,
) -> tuple[Any, Any]:

    processor = AutoProcessor.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": "cpu"},
    )

    return model, processor


def inspect_quantization(model: Any) -> None:

    total_modules = 0
    quantized_modules = 0

    for name, module in model.named_modules():
        total_modules += 1

        if isinstance(module, bnb.nn.Linear4bit):
            quantized_modules += 1

    print(f"\nTotal modules: {total_modules}")

    print(f"4-bit modules: {quantized_modules}")


def prepare_training_batch(
    processor: Any,
    image: Image.Image,
) -> dict[str, torch.Tensor]:

    text_prompt = "<image>Describe the object in this image."

    inputs = processor(
        text=text_prompt,
        images=image,
        return_tensors="pt",
    )

    return inputs


def setup_qlora(
    model: Any,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> Any:

    print("\nPreparing model for k-bit training...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.train()

    return model


def print_memory_checkpoint(
    label: str,
) -> None:

    print(f"[MEMORY] {label}: {get_peak_memory_mb():.2f} MB")


def run_qlora_training_step(
    model: Any,
    processor: Any,
    image: Image.Image,
    learning_rate: float = 1e-4,
) -> dict[str, float]:

    print("\n" + "=" * 60)
    print("QLoRA TRAINING TEST")
    print("=" * 60)

    timings: dict[str, float] = {}

    # --------------------------------------------------
    # Prepare input
    # --------------------------------------------------

    start = time.perf_counter()

    inputs = prepare_training_batch(
        processor=processor,
        image=image,
    )

    timings["preprocessing_seconds"] = time.perf_counter() - start

    print_memory_checkpoint("after preprocessing")

    print("\nInput tensors:")

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: dtype={value.dtype}, shape={tuple(value.shape)}")

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]

    trainable_count = sum(param.numel() for param in trainable_parameters)

    total_count = sum(param.numel() for param in model.parameters())

    print(f"\nTrainable parameters: {trainable_count:,}")

    print(f"Total parameters: {total_count:,}")

    print(f"Trainable percentage: {100 * trainable_count / total_count:.4f}%")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
    )

    print_memory_checkpoint("after optimizer creation")

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    optimizer.zero_grad()

    print("\nStarting forward pass...")

    start = time.perf_counter()

    outputs = model(
        **inputs,
        labels=inputs["input_ids"],
        use_cache=False,
    )

    timings["forward_seconds"] = time.perf_counter() - start

    loss = outputs.loss

    print(f"Forward completed in {timings['forward_seconds']:.3f} s")

    print(f"Loss: {loss.item():.6f}")

    print_memory_checkpoint("after forward")

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    print("\nStarting backward pass...")

    start = time.perf_counter()

    loss.backward()

    timings["backward_seconds"] = time.perf_counter() - start

    print(f"Backward completed in {timings['backward_seconds']:.3f} s")

    print_memory_checkpoint("after backward")

    # --------------------------------------------------
    # Optimizer step
    # --------------------------------------------------

    print("\nStarting optimizer step...")

    start = time.perf_counter()

    optimizer.step()

    timings["optimizer_seconds"] = time.perf_counter() - start

    print(f"Optimizer step completed in {timings['optimizer_seconds']:.3f} s")

    print_memory_checkpoint("after optimizer step")

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    timings["total_training_seconds"] = sum(timings.values())

    print("\n" + "=" * 60)
    print("QLoRA TEST RESULTS")
    print("=" * 60)

    print(f"Preprocessing: {timings['preprocessing_seconds']:.3f} s")

    print(f"Forward:       {timings['forward_seconds']:.3f} s")

    print(f"Backward:      {timings['backward_seconds']:.3f} s")

    print(f"Optimizer:     {timings['optimizer_seconds']:.3f} s")

    print(f"Total:         {timings['total_training_seconds']:.3f} s")

    print(f"\nPeak RSS: {get_peak_memory_mb():.2f} MB")

    return timings


def main() -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"PyTorch: {torch.__version__}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"\nInitial peak RSS: {get_peak_memory_mb():.2f} MB")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading 4-bit CPU model...")

    start_load = time.perf_counter()

    model, processor = load_cpu_4bit_vlm(
        MODEL_ID,
        torch.bfloat16,
    )

    load_time = time.perf_counter() - start_load

    print(f"Model loaded in {load_time:.3f} s")

    print_memory_checkpoint("after model load")

    inspect_quantization(model)

    # --------------------------------------------------
    # Add QLoRA
    # --------------------------------------------------

    start_lora = time.perf_counter()

    model = setup_qlora(
        model,
        lora_r=8,
        lora_alpha=16,
    )

    lora_time = time.perf_counter() - start_lora

    print(f"\nLoRA setup completed in {lora_time:.3f} s")

    print_memory_checkpoint("after LoRA setup")

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Load one real CORe50 image
    # --------------------------------------------------

    print("\nLoading one CORe50 image...")

    frames = load_core50_sample_frames(
        DATASET_DIR,
        num_frames=1,
    )

    print(f"Loaded {len(frames)} image.")

    print_memory_checkpoint("after image load")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    run_qlora_training_step(
        model=model,
        processor=processor,
        image=frames[0],
    )


if __name__ == "__main__":
    main()
