"""
Isolation check: does the model's output change when the INPUT IMAGE changes,
holding the text prompt fixed? If output is identical across visually distinct
images, the image pathway is broken (merge dropped vision weights, or
pixel_values aren't reaching the model) -- this must be ruled out BEFORE
concluding anything about training quality.

Usage:
    python scripts/feasibility/check_image_conditioning.py \
        --model-path results/f2/merged_model \
        --image-paths path/to/imgA.png path/to/imgB.png path/to/imgC.png
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-paths", type=str, nargs="+", required=True)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, quantization_config=bnb_config, device_map={"": "cpu"},
    )
    model.eval()

    outputs_seen = []
    pixel_value_hashes = []

    for path in args.image_paths:
        image = Image.open(path).convert("RGB")
        inputs = build_prompt(processor, image)

        # Sanity check #1: are pixel_values actually different per image?
        # If this hash collides across images, the bug is upstream of the
        # model entirely (image loading/preprocessing), not the model itself.
        pv_hash = hash(inputs["pixel_values"].numpy().tobytes())
        pixel_value_hashes.append(pv_hash)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
        outputs_seen.append(text)

        print(f"{path}\n  pixel_values hash: {pv_hash}\n  generated: {text!r}\n")

    print("--- DIAGNOSIS ---")
    if len(set(pixel_value_hashes)) == 1:
        print("FAIL: pixel_values are IDENTICAL across different images.")
        print("  -> Bug is in image loading/preprocessing, upstream of the model.")
        print("  -> Check build_prompt / processor call, not the model or training.")
    elif len(set(outputs_seen)) == 1:
        print("FAIL: pixel_values differ, but generated text is IDENTICAL.")
        print("  -> Model is not conditioning on the image at all.")
        print("  -> Suspect: merge script dropped/mis-merged vision-tower adapter weights,")
        print("     or image token embedding is broken in the merged checkpoint.")
        print("  -> Compare against the UN-merged base model + PEFT adapter loaded")
        print("     separately (not merged) to see if the bug is merge-specific.")
    else:
        print("Output varies with image -- image pathway is at least partially live.")
        print("  -> Low accuracy is then a training/data question, not a pipeline bug.")
        print("  -> Proceed to inspect training data quality, LR, epochs, LoRA rank.")


if __name__ == "__main__":
    main()
