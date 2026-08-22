"""
VLM Scorer Module for Edge Sample Selection.

Uses SmolVLM (or another Hugging Face Vision-Language Model) to evaluate
incoming visual stream samples and estimate their learning value / informativeness.
Supports INT8 (8-bit) quantization via bitsandbytes for reduced memory footprint on CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from transformers import AutoProcessor, BitsAndBytesConfig


@dataclass
class SampleScore:
    """Represents the evaluation output for a single visual sample."""

    usefulness_score: float  # Normalized in [0.0, 1.0], higher = more informative/novel
    confidence: float  # Mean top-1 confidence from token distribution
    entropy: float  # Mean predictive entropy
    description: str | None = None


class VLMScorer:
    """
    Evaluates visual samples using a Vision-Language Model (e.g. SmolVLM-256M-Instruct).
    Derives uncertainty and novelty metrics to determine sample usefulness for retraining.
    """

    DEFAULT_MODEL_ID: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    DEFAULT_PROMPT: str = (
        "<image>Describe the main object and its attributes in this image."
    )

    def __init__(
        self,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        quantization: str = "4bit",
    ):
        self.model_id: str = model_id
        self.cache_dir: str | None = cache_dir
        self.quantization: str = quantization.lower()
        if device is not None:
            self.device: torch.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if torch_dtype is not None:
            self.torch_dtype: torch.dtype = torch_dtype
        else:
            self.torch_dtype = (
                torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
            )

        # Configure INT8 quantization (active on CUDA)
        quantization_config: BitsAndBytesConfig | None = None
        compute_dtype = (
            torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        )
        if self.quantization == "4bit" and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            print(f"Loading VLM '{self.model_id}' in NF4 (4-bit) on {self.device}...")
        elif self.quantization == "8bit" and self.device.type == "cuda":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"Loading VLM '{self.model_id}' in 8-bit on {self.device}...")
        else:
            print(
                f"Loading VLM '{self.model_id}' on {self.device} (dtype={self.torch_dtype})..."
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype if quantization_config is None else None,
            quantization_config=quantization_config,
            cache_dir=self.cache_dir,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        if self.device.type != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"VLM '{self.model_id}' loaded successfully.")

    def _prepare_image(self, image_input: str | Path | Image.Image) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        return image_input.convert("RGB")

    def compute_usefulness(
        self,
        image_input: str | Path | Image.Image,
        prompt: str | None = None,
        max_new_tokens: int = 40,
        generate_text: bool = False,
    ) -> SampleScore:
        """
        Evaluates an image sample and computes its informativeness / usefulness score.

        Args:
            image_input: PIL Image or file path to an image.
            prompt: Text prompt to condition the VLM.
            max_new_tokens: Number of tokens to generate if text is requested.
            generate_text: If True, decodes the text description.

        Returns:
            SampleScore containing usefulness_score (0.0 - 1.0), confidence, and entropy.
        """
        image = self._prepare_image(image_input)
        prompt_text = prompt or self.DEFAULT_PROMPT

        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits: torch.Tensor = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Focus on the last token prediction logits for uncertainty calculation
            last_token_logits = logits[:, -1, :].float()
            probs = torch.softmax(last_token_logits, dim=-1)

            # 1. Top-1 Confidence
            top1_conf = probs.max(dim=-1).values.item()

            # 2. Predictive Entropy: H(p) = - sum(p * log(p + eps))
            log_probs = torch.log(probs + 1e-12)
            entropy_val = -torch.sum(probs * log_probs, dim=-1).item()

            # Normalize entropy (log(vocab_size) is max entropy)
            vocab_size = logits.size(-1)
            max_entropy = math.log(vocab_size) if vocab_size > 0 else 1.0
            norm_entropy = min(max(entropy_val / max_entropy, 0.0), 1.0)

            # Combined usefulness score: higher entropy & lower confidence -> higher learning value
            usefulness = 0.5 * (1.0 - top1_conf) + 0.5 * norm_entropy
            usefulness = round(min(max(usefulness, 0.0), 1.0), 4)

            description = None
            if generate_text:
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                description = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

        return SampleScore(
            usefulness_score=usefulness,
            confidence=round(top1_conf, 4),
            entropy=round(entropy_val, 4),
            description=description,
        )

    def batch_score(
        self,
        images: list[str | Path | Image.Image],
        prompt: str | None = None,
    ) -> list[SampleScore]:
        """Convenience method to score a sequence of incoming frames."""
        return [self.compute_usefulness(img, prompt=prompt) for img in images]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VLM Scorer standalone.")
    parser.add_argument(
        "--image", type=str, default=None, help="Path to test image file"
    )
    parser.add_argument(
        "--model-id", type=str, default=VLMScorer.DEFAULT_MODEL_ID, help="HF model ID"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="quantization mode: 4bit (NF4), 8bit (INT8) or none",
    )
    parser.add_argument(
        "--generate-text", action="store_true", help="Generate text description"
    )

    args = parser.parse_args()

    scorer = VLMScorer(
        model_id=args.model_id,
        device=args.device,
        quantization=args.quantization,
    )

    if args.image and Path(args.image).exists():
        test_img = Path(args.image)
        print(f"Scoring provided image: {test_img}")
    else:
        print("No image provided. Creating a synthetic test image...")
        test_img = Image.new("RGB", (224, 224), color=(80, 120, 160))

    result = scorer.compute_usefulness(test_img, generate_text=args.generate_text)
    print("\n--- Evaluation Result ---")
    print(
        f"Usefulness Score: {result.usefulness_score} (0=redundant, 1=highly informative)"
    )
    print(f"Confidence:       {result.confidence}")
    print(f"Entropy:          {result.entropy}")
    if result.description:
        print(f"VLM Description:  {result.description}")
