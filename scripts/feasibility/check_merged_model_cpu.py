"""
Confirm the merged, CPU-quantized checkpoint loads and INFERS CORRECTLY
on REAL CORe50 validation images, run repeatedly, and measure peak RAM
plus latency for load / prefill / decode separately.

Correctness check: decodes the model's generated text and does a simple
substring match against the CORe50 ground-truth category_name. This is a
coarse check (not exact-match classification) since the model outputs
free text, not a class label — treat "pass" as "the category word
appears in the description," not as classification accuracy.

Usage:
    python scripts/feasibility/check_merged_model_cpu.py \
        --model-path results/f2/merged_model \
        --dataset-path data/raw/core50/core50_128x128 \
        --object-ids 1 2 3 4 5 6 7 8 9 10 \
        --session 3 \
        --num-samples 50 \
        --max-new-tokens 30
"""

from __future__ import annotations

import argparse
import re
import resource
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.data.core50 import scan_core50


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def build_prompt(processor, image: Image.Image):
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
    return processor(text=prompt_text, images=[image], return_tensors="pt")


def normalize(text: str) -> str:
    # lowercase, strip punctuation, collapse whitespace -> for loose substring matching
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def category_matches(generated_text: str, category_name: str) -> bool:
    """
    Loose correctness check: CORe50 category names are underscore_separated
    (e.g. 'plug_adapter'). Split into words and require ALL words to appear
    somewhere in the normalized generated text. This is intentionally coarse
    -- it does not verify grammatical correctness or full description quality,
    only that the model is naming the right object.
    """
    gen_norm = normalize(generated_text)
    category_words = category_name.lower().split("_")
    return all(word in gen_norm for word in category_words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--object-ids", type=int, nargs="+", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Cap on generated tokens. This directly sets your decode-cost "
        "budget -- pick the smallest value that still lets the model "
        "produce a usable answer for your downstream task.",
    )
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
    model.eval()
    load_time = time.perf_counter() - load_start
    peak_ram_after_load_mb = get_peak_rss_mb()
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Peak RAM after load: {peak_ram_after_load_mb:.2f} MB")

    print(f"Scanning real CORe50 data from '{args.dataset_path}'...")
    all_samples = scan_core50(args.dataset_path)
    eval_pool = sorted(
        (
            s
            for s in all_samples
            if s.session_id == args.session and s.object_id in args.object_ids
        ),
        key=lambda s: (s.object_id, s.frame_id),
    )
    if len(eval_pool) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} eval samples but only "
            f"{len(eval_pool)} available for objects {args.object_ids} in "
            f"session {args.session}."
        )
    eval_samples = eval_pool[: args.num_samples]
    print(f"Running generation on {len(eval_samples)} real CORe50 images "
          f"(max_new_tokens={args.max_new_tokens})...")

    total_latencies = []
    prefill_latencies = []
    decode_latencies = []
    tokens_generated_list = []
    correct_count = 0

    for i, sample in enumerate(eval_samples):
        image = Image.open(sample.path).convert("RGB")
        inputs = build_prompt(processor, image)
        input_len = inputs["input_ids"].shape[1]

        # Prefill-only timing: one forward pass over the input, no generation.
        # This double-runs work generate() will also do internally, but it's
        # the only way to isolate prefill cost with the public generate() API
        # -- there's no hook to grab "time to first token" from generate()
        # without reaching into StoppingCriteria/StreamerCallback. Accept the
        # ~1x extra prefill cost here since this is a feasibility script, not
        # the production path.
        prefill_start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        prefill_time = time.perf_counter() - prefill_start

        gen_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy -- deterministic, reproducible correctness checks
            )
        total_time = time.perf_counter() - gen_start

        new_tokens = generated_ids[:, input_len:]
        tokens_generated = new_tokens.shape[1]
        decode_time = total_time - prefill_time  # approx: generate() re-does prefill internally
        per_token_decode = decode_time / max(tokens_generated - 1, 1) if tokens_generated > 1 else float("nan")

        generated_text = processor.decode(new_tokens[0], skip_special_tokens=True)
        is_correct = category_matches(generated_text, sample.category_name)
        correct_count += int(is_correct)

        total_latencies.append(total_time)
        prefill_latencies.append(prefill_time)
        decode_latencies.append(decode_time)
        tokens_generated_list.append(tokens_generated)

        status = "PASS" if is_correct else "FAIL"
        print(
            f"  [{i + 1}/{len(eval_samples)}] object={sample.object_id} "
            f"category={sample.category_name} frame={sample.frame_id} "
            f"-> total={total_time:.3f}s prefill={prefill_time:.3f}s "
            f"decode={decode_time:.3f}s ({tokens_generated} tok, "
            f"{per_token_decode * 1000:.1f} ms/tok) [{status}]"
        )
        print(f"       generated: {generated_text!r}")

    peak_ram_after_inference_mb = get_peak_rss_mb()
    n = len(eval_samples)
    avg_total = sum(total_latencies) / n
    avg_prefill = sum(prefill_latencies) / n
    avg_decode = sum(decode_latencies) / n
    avg_tokens = sum(tokens_generated_list) / n
    accuracy = correct_count / n

    print("\n--- SUMMARY ---")
    print(f"Samples run: {n}")
    print(f"Peak RAM during/after model load: {peak_ram_after_load_mb:.2f} MB")
    print(f"Peak RAM during/after inference:  {peak_ram_after_inference_mb:.2f} MB")
    print(f"Avg total latency (generate):  {avg_total:.3f}s "
          f"(min {min(total_latencies):.3f}s, max {max(total_latencies):.3f}s)")
    print(f"Avg prefill latency:           {avg_prefill:.3f}s")
    print(f"Avg decode latency:            {avg_decode:.3f}s "
          f"({avg_tokens:.1f} tokens/sample avg)")
    print(f"Correctness (category word in generated text): "
          f"{correct_count}/{n} = {accuracy * 100:.1f}%")
    print("Target budget (Pi 5): ~2048 MB")
    fits = peak_ram_after_inference_mb < 2048
    print(f"Fits under 2GB target: {'YES' if fits else 'NO'}")


if __name__ == "__main__":
    main()
