from transformers import AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> None:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)

    print("TEXT_MODEL linear layer full dotted names (first 10):")
    count = 0
    for full_name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            if "text_model" in full_name:
                print(f"  {full_name}")
                count += 1
                if count >= 10:
                    break

    print()
    print("VISION_MODEL linear layer full dotted names (first 5, for contrast):")
    count = 0
    for full_name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            if "vision_model" in full_name:
                print(f"  {full_name}")
                count += 1
                if count >= 5:
                    break

    print()
    print("CONNECTOR linear layer full dotted names:")
    for full_name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            if "connector" in full_name:
                print(f"  {full_name}")


if __name__ == "__main__":
    main()
