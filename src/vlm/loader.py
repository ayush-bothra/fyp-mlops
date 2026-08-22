"""
QLoRA model loading for F2 feasibility testing.

Corrections relative to the old benchmark_vlm_edge.py CUDA path:

1. target_modules is a REGEX STRING scoped to the text decoder only,
   not a plain list like ["q_proj", "v_proj"]. A plain list matches by
   suffix across the ENTIRE model tree -- SmolVLM's vision tower
   (Idefics3VisionTransformer / SigLIP) also has q_proj/v_proj/k_proj
   modules under model.vision_model.encoder.layers.N.self_attn.*, so the
   naive list form would silently LoRA-adapt the vision encoder too.
   Confirmed via runtime introspection of the actual model, not assumed.

2. prepare_model_for_kbit_training() is called before attaching LoRA.
   Without it, training against a 4-bit quantized base can fail at
   backward() (frozen embeddings not marked to propagate grad to the
   adapters) or silently misbehave (norm layers left in reduced
   precision). This was never exercised by the old script because its
   CUDA path had never actually been run against 4-bit end to end with
   a real backward pass and label supervision.

3. device_map={"": 0} instead of "auto" -- deterministic single-GPU
   placement for a controlled feasibility experiment, not a heuristic
   placement decision.
"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.vlm.quantization import get_qlora_bnb_config

# Confirmed via runtime introspection (scripts/inspect_smolvlm_text_paths.py)
# against the real HuggingFaceTB/SmolVLM-256M-Instruct module tree.
# Anchored to text_model only -- deliberately excludes
# model.vision_model.* and model.connector.* subtrees.
TEXT_ATTENTION_TARGET_REGEX = r"^model\.text_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"

FORBIDDEN_TRAINABLE_SUBSTRINGS = ("vision_model", "connector")


def build_lora_config(
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TEXT_ATTENTION_TARGET_REGEX,
        bias="none",
        task_type=None,  # SmolVLM/Idefics3 is not a standard CAUSAL_LM task-type shape at the PEFT wrapper level for image-text models; leave unset and rely on target_modules regex rather than a task-type-driven auto module map.
    )


def load_quantized_vlm_for_training(
    model_id: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    compute_dtype: torch.dtype = torch.float16,
    device_index: int = 0,
) -> tuple[Any, Any]:
    """
    Loads SmolVLM in 4-bit NF4, freezes the base model, attaches LoRA
    scoped to the text decoder's attention q/v projections only.

    device_index MUST match whatever device apply_vram_cap() was called
    with -- a mismatch here would mean the cap is enforced on one device
    while the model loads onto another, silently defeating the whole
    point of the experiment.

    Caller is responsible for calling
    src.vlm.quantization.apply_vram_cap() BEFORE this function, since the
    cap must be in effect before any CUDA allocation happens.
    """
    bnb_config = get_qlora_bnb_config(compute_dtype=compute_dtype)

    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map={"": device_index},
    )

    # Required for stable QLoRA training against a 4-bit frozen base:
    # upcasts norm layers to fp32, enables input-embedding grad flow so
    # gradients actually reach the LoRA adapters during backward().
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    base_model.enable_input_require_grads()

    lora_config = build_lora_config(
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()

    return peft_model, processor


def audit_trainable_parameters(model: Any) -> dict:
    """
    Not just a report -- an ASSERTION. Raises RuntimeError if:
      - zero parameters are trainable (LoRA didn't attach to anything --
        e.g. the regex matched nothing, silently producing a model that
        "trains" but never updates)
      - any trainable parameter's name contains "vision_model" or
        "connector" (the scoping regex failed and vision/connector
        layers got adapted too)
      - any trainable parameter is NOT a lora_A/lora_B weight (something
        other than LoRA got unfrozen, e.g. prepare_model_for_kbit_training
        or a library version change unfroze more than expected)

    This function is the actual "verify LoRA parameters are trainable,
    verify base parameters are frozen" step the spec requires -- it must
    fail loudly rather than let a broken run report PASS.
    """
    total_params = 0
    trainable_params = 0
    trainable_names: list[str] = []
    dtypes_seen: set[str] = set()

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        dtypes_seen.add(str(param.dtype))
        if param.requires_grad:
            trainable_params += numel
            trainable_names.append(name)

    if trainable_params == 0:
        raise RuntimeError(
            "Zero trainable parameters after LoRA attachment. The "
            "target_modules regex almost certainly matched nothing -- "
            "check TEXT_ATTENTION_TARGET_REGEX against the model's real "
            "module names before trusting any training result."
        )

    for name in trainable_names:
        if not any(marker in name for marker in ("lora_A", "lora_B")):
            raise RuntimeError(
                f"Trainable parameter '{name}' is not a LoRA weight. "
                "Something other than the intended LoRA adapters is "
                "unfrozen -- do not trust this run's results."
            )
        if any(forbidden in name for forbidden in FORBIDDEN_TRAINABLE_SUBSTRINGS):
            raise RuntimeError(
                f"Trainable parameter '{name}' lives under a forbidden "
                f"subtree {FORBIDDEN_TRAINABLE_SUBSTRINGS}. The LoRA "
                "target_modules regex incorrectly matched the vision "
                "tower or connector -- this invalidates F2's premise of "
                "text-only adaptation."
            )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        "num_trainable_tensors": len(trainable_names),
        "trainable_tensor_names_sample": trainable_names[:6],
        "param_dtypes_observed": sorted(dtypes_seen),
    }
