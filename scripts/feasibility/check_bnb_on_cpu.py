"""
One-off check: does bitsandbytes support 4-bit inference on CPU?

Not assumed -- verified empirically per project rules. This does NOT
use the LoRA-adapted checkpoint; it loads the plain base model in 4-bit
straight to CPU and tries a single forward pass. If this fails, F1's
"4-bit on CPU" plan is not viable as designed and needs a different
quantization strategy before any further F1 code is written.

Usage:
    python scripts/feasibility/check_bnb_cpu_support.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    print(f"Loading merged model from '{args.model_path}' in 4-bit on CPU...")
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": "cpu"},
    )
    load_time = time.perf_counter() - load_start
    peak_ram_after_load_mb = get_peak_rss_mb()

    print(f"Model loaded in {load_time:.2f}s")
    print(f"Peak RAM after load: {peak_ram_after_load_mb:.2f} MB")

    print("Running one inference pass...")
    dummy_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the object in this image."},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[dummy_image], return_tensors="pt")

    infer_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_time = time.perf_counter() - infer_start
    peak_ram_after_inference_mb = get_peak_rss_mb()

    print(f"Inference completed in {infer_time:.3f}s")
    print(f"Peak RAM after inference: {peak_ram_after_inference_mb:.2f} MB")
    print(f"Output logits shape: {outputs.logits.shape}")

    print("\n--- SUMMARY ---")
    print(f"Peak RAM during/after model load: {peak_ram_after_load_mb:.2f} MB")
    print(f"Peak RAM during/after inference:  {peak_ram_after_inference_mb:.2f} MB")
    print(f"Target budget (Pi 5): ~2048 MB")
    fits = peak_ram_after_inference_mb < 2048
    print(f"Fits under 2GB target: {'YES' if fits else 'NO'}")


if __name__ == "__main__":
    main()