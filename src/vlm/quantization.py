"""
Quantization + VRAM-capping utilities for F2 QLoRA feasibility testing.

IMPORTANT DIFFERENCE FROM THE OLD benchmark_vlm_edge.py:

The old script called `device_map="auto"` and *measured* peak VRAM after
the fact. That tells you how much VRAM training happened to use on a card
with 4GB physically available -- it does NOT tell you whether training
fits inside a simulated 2GB budget. Measuring without constraining is a
different experiment than the one F2 asks for.

There is no hardware-level VRAM partition on a consumer 3050 (no MIG,
no vGPU) -- so the only realistic simulation mechanism from within
PyTorch is `torch.cuda.set_per_process_memory_fraction()`. This makes the
CUDA caching allocator raise OOM if the process tries to exceed the
fraction of total device memory, even though more is physically present.

Caveat worth knowing, not hiding: this is a soft allocator-level cap, not
a kernel/driver-level partition. It will not catch every possible way a
process could touch VRAM (e.g. it does not affect other processes, and
extremely low-level CUDA API misuse outside the caching allocator could
theoretically bypass it), but for a standard HF + bitsandbytes + PEFT
training loop -- which allocates exclusively through the PyTorch caching
allocator -- it is an accurate simulation of a hard 2GB ceiling.
"""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig


def get_qlora_bnb_config(compute_dtype: torch.dtype = torch.float16) -> BitsAndBytesConfig:
    """
    NF4 double-quantization config for genuine QLoRA-style loading.

    - load_in_4bit: base weights stored in 4-bit NF4
    - bnb_4bit_use_double_quant: quantizes the quantization constants
      themselves for a further ~0.4 bit/param savings -- standard QLoRA
    - bnb_4bit_compute_dtype: dequantized on-the-fly to this dtype for
      the actual matmuls. fp16 chosen over bf16 because the 3050 (Ampere,
      compute capability 8.6) supports bf16, but fp16 has broader
      operator coverage in bitsandbytes at time of writing -- flag if you
      want to switch to bf16 and test empirically, the spec asks for
      compute dtype to be justified, not just picked.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def apply_vram_cap(fraction: float, device_index: int = 0) -> dict:
    """
    Actually enforce a VRAM ceiling on the current process, rather than
    just measuring usage after the fact.

    fraction: e.g. 0.5 on a 4GB card simulates a ~2GB budget.

    Returns a dict of the effective cap in MB for logging, so the F2
    results JSON can record what was actually enforced, not just what
    was requested (protects against silently running unconstrained if
    CUDA isn't available at all).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available -- cannot apply a VRAM cap or run F2 "
            "QLoRA training. F2 explicitly targets a GPU (RTX 3050 class); "
            "this is not a CPU experiment. Refusing to silently continue "
            "with an unconstrained/fake result."
        )

    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    torch.cuda.reset_peak_memory_stats(device=device_index)

    total_device_mb = torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024)
    effective_cap_mb = total_device_mb * fraction

    return {
        "device_name": torch.cuda.get_device_properties(device_index).name,
        "total_device_mb": round(total_device_mb, 1),
        "requested_fraction": fraction,
        "effective_cap_mb": round(effective_cap_mb, 1),
    }
