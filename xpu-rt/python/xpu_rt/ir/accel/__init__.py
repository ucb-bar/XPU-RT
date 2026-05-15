"""Custom accelerator dialect for hardware-specific operations.

Represents operations that are specific to custom accelerators where
Triton does not fit: tile load/store, DMA, matrix engine launches,
barriers, packed layouts, quantized accumulation.

This is a **target dialect**, not a replacement for Payload IR. Selected
regions lower from Payload IR into this dialect when the target profile
specifies a custom accelerator backend.

The accelerator dialect models **hardware semantics**, not implementation
details. LLVM intrinsics are a late lowering target, not the user-facing API.
"""

from __future__ import annotations

__all__: list[str] = []
