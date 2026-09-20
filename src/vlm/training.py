"""
Training utilities for F2 QLoRA feasibility.

Fixes relative to the old benchmark_vlm_edge.py:

1. Real supervision target. The old script used a fixed prompt with no
   answer appended and trained with labels=input_ids over the WHOLE
   sequence -- including the image tokens and the prompt text itself.
   That trains the model to reconstruct its own fixed prompt template,
   which is a task any model does immediately from initialization; "loss
   decreases" would have been true for reasons having nothing to do with
   whether QLoRA actually taught the model anything about the image.

2. Label masking. Only the answer span (the CORe50 category caption) is
   supervised. Everything before it -- image tokens, prompt text,
   chat-template scaffolding -- is masked with -100 so it doesn't
   contribute to the loss.

Fixes relative to the first version of this file:

3. initial_loss/final_loss used to be the loss on the first and last
   training example only, not an aggregate. That told us almost nothing
   about overall learning, only about whichever single sample happened
   to be first or last in an unshuffled, class-grouped ordering.
   TrainingRunResult now reports per-epoch mean loss instead.

4. Training ran a single unshuffled pass over class-grouped data, which
   biases the final weights toward whichever classes were trained last.
   Training now runs multiple epochs with a freshly shuffled order each
   epoch.

5. Every image of a given class got the exact same caption string,
   which makes it easy for the model to memorize a short fixed target
   instead of learning visual features. Captions are now drawn from a
   small set of equivalent phrasings.

Boundary detection: SmolVLM's processor expands a single <image>
placeholder into many tokens (patch-dependent), so the prompt's token
length can't be predicted from text alone. We process the prompt-only
text WITH the image to get its true expanded token length, then process
prompt+answer together and mask everything before that boundary.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
from PIL import Image

from src.data.core50 import Core50Sample

CAPTION_TEMPLATES = [
    "a photo of a {name}.",
    "an image showing a {name}.",
    "this is a {name}.",
    "a picture of a {name}.",
]


def _caption_for_sample(sample: Core50Sample, rng: random.Random | None = None) -> str:
    readable_name = sample.category_name.replace("_", " ")
    template = rng.choice(CAPTION_TEMPLATES) if rng is not None else CAPTION_TEMPLATES[0]
    return template.format(name=readable_name)


def build_supervised_example(
    processor: Any,
    image: Image.Image,
    caption: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build one training example with labels masked so only the caption
    span contributes to the loss.
    """
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

    prompt_only_inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    prompt_len = prompt_only_inputs["input_ids"].shape[1]

    eos_token = processor.tokenizer.eos_token or ""
    full_text = f"{prompt_text}{caption}{eos_token}"
    full_inputs = processor(text=full_text, images=[image], return_tensors="pt")
    full_len = full_inputs["input_ids"].shape[1]

    if full_len <= prompt_len:
        raise RuntimeError(
            f"Full sequence length ({full_len}) did not grow beyond the "
            f"prompt-only length ({prompt_len}) after appending the caption. "
            "Label masking would supervise nothing -- refusing to silently "
            "produce a zero-length training target. Check the caption text "
            "and chat template output."
        )

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    batch = {key: value.to(device) for key, value in full_inputs.items()}
    batch["labels"] = labels.to(device)
    return batch


def _ordered_samples(
    samples: list[Core50Sample], rng: random.Random, shuffle: bool
) -> list[Core50Sample]:
    if not shuffle:
        return samples
    ordered = list(samples)
    rng.shuffle(ordered)
    return ordered


def prepare_supervised_batches(
    processor: Any,
    samples: list[Core50Sample],
    device: torch.device,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    rng = random.Random(seed)
    batches: list[dict[str, torch.Tensor]] = []
    for sample in _ordered_samples(samples, rng, shuffle):
        image = Image.open(sample.path).convert("RGB")
        caption = _caption_for_sample(sample, rng)
        batches.append(
            build_supervised_example(
                processor=processor, image=image, caption=caption, device=device
            )
        )
    return batches


def stream_supervised_batches(
    processor: Any,
    samples: list[Core50Sample],
    device: torch.device,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    rng = random.Random(seed)
    for sample in _ordered_samples(samples, rng, shuffle):
        image = Image.open(sample.path).convert("RGB")
        caption = _caption_for_sample(sample, rng)
        yield build_supervised_example(
            processor=processor, image=image, caption=caption, device=device
        )


def execute_training_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch_inputs: dict[str, torch.Tensor],
) -> float:
    optimizer.zero_grad()
    outputs = model(**batch_inputs)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError(
            "Model forward pass returned no loss despite labels being present "
            "in the batch. Check that 'labels' key survived any batch "
            "filtering/collation upstream."
        )
    loss.backward()
    optimizer.step()
    return float(loss.item())


@dataclass
class TrainingRunResult:
    epoch_losses: list[float]
    step_losses: list[float]
    step_times_seconds: list[float]
    train_seconds: float

    @property
    def initial_loss(self) -> float:
        return self.epoch_losses[0] if self.epoch_losses else float("nan")

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1] if self.epoch_losses else float("nan")

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def avg_step_seconds(self) -> float:
        return sum(self.step_times_seconds) / len(self.step_times_seconds) if self.step_times_seconds else 0.0


def run_training_loop(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch_factory: Callable[[], Iterator[dict[str, torch.Tensor]]],
    epochs: int = 1,
) -> TrainingRunResult:
    epoch_losses: list[float] = []
    step_losses: list[float] = []
    step_times: list[float] = []

    start = time.perf_counter()
    for _ in range(epochs):
        losses_this_epoch: list[float] = []
        for batch in batch_factory():
            step_start = time.perf_counter()
            loss_val = execute_training_step(model=model, optimizer=optimizer, batch_inputs=batch)
            step_times.append(time.perf_counter() - step_start)
            step_losses.append(loss_val)
            losses_this_epoch.append(loss_val)
        epoch_mean = sum(losses_this_epoch) / len(losses_this_epoch) if losses_this_epoch else float("nan")
        epoch_losses.append(epoch_mean)
    total_train_seconds = time.perf_counter() - start

    return TrainingRunResult(
        epoch_losses=epoch_losses,
        step_losses=step_losses,
        step_times_seconds=step_times,
        train_seconds=total_train_seconds,
    )


def evaluate_validation_loss(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
) -> float:
    """Forward-only pass over validation batches, no grad, no optimizer step."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in batches:
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
    model.train()
    return total_loss / len(batches) if batches else float("nan")


def evaluate_validation_loss_by_category(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
    categories: list[str],
) -> dict[str, float]:
    model.eval()
    losses_by_category: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch, category in zip(batches, categories):
            outputs = model(**batch)
            losses_by_category.setdefault(category, []).append(float(outputs.loss.item()))
    model.train()
    return {
        category: sum(losses) / len(losses)
        for category, losses in losses_by_category.items()
    }
