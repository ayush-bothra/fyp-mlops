"""
Merge a trained LoRA adapter into the base SmolVLM model.

Must load the base model WITHOUT 4-bit quantization here -- PEFT's
merge_and_unload() computes base_weight + (lora_B @ lora_A) * scaling,
which requires real floating-point base weights. A 4-bit quantized
weight can't be merged into directly; it has to be dequantized first.
CPU-side 4-bit quantization (for F1) happens as a SEPARATE later step,
by loading this merged fp16 checkpoint with a BitsAndBytesConfig at
that point -- not here.

Usage:
    python -m scripts.feasibility.merge_lora_adapter \
        --adapter-path results/f2/adapter/qlora_training_final \
        --output-path results/f2/merged_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def merge_adapter(adapter_path: str, output_path: str, base_model_id: str) -> None:
    print(f"Loading base model '{base_model_id}' in fp16 (no quantization)...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    processor = AutoProcessor.from_pretrained(adapter_path, do_image_splitting=False)

    print(f"Attaching adapter from '{adapter_path}'...")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model...")
    merged_model = merged_model.merge_and_unload()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Merged fp16 model saved to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a trained LoRA adapter into the base model")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="results/f2/merged_model")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    args = parser.parse_args()

    merge_adapter(args.adapter_path, args.output_path, args.base_model_id)


if __name__ == "__main__":
    main()
