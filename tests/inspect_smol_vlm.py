"""
One-off inspection script: dump the real module names of
HuggingFaceTB/SmolVLM-256M-Instruct so LoRA target_modules can be chosen
based on actual architecture, not assumption.

Run this once, share the output, then this script can be discarded --
it is not part of the F2 pipeline.

Usage:
    python inspect_smolvlm_modules.py
"""

from transformers import AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> None:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)

    print("=" * 70)
    print("TOP-LEVEL MODULE TREE (depth 2)")
    print("=" * 70)
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
        for sub_name, sub_module in module.named_children():
            print(f"  {sub_name}: {sub_module.__class__.__name__}")

    print()
    print("=" * 70)
    print("ALL LINEAR LAYER NAMES (these are LoRA target_modules candidates)")
    print("=" * 70)
    linear_suffixes: set[str] = set()
    for full_name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            suffix = full_name.rsplit(".", 1)[-1]
            linear_suffixes.add(suffix)

    for suffix in sorted(linear_suffixes):
        print(f"  {suffix}")

    print()
    print("=" * 70)
    print("SAMPLE FULL DOTTED NAMES (first 5 linear layers, to see nesting)")
    print("=" * 70)
    count = 0
    for full_name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            print(f"  {full_name}")
            count += 1
            if count >= 5:
                break

    print()
    print("=" * 70)
    print(f"Text backbone class: {model.config.text_config.model_type if hasattr(model.config, 'text_config') else 'N/A'}")
    print(f"Vision backbone class: {model.config.vision_config.model_type if hasattr(model.config, 'vision_config') else 'N/A'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
