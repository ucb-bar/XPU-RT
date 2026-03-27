"""IR subsystem -- three-layer IR stack for CompGen.

CompGen uses a three-layer IR architecture:

Layer 1: **Payload IR** (``ir.payload``)
    The canonical computational IR from FX→xDSL. Represents model structure,
    tensors, control flow, layouts, effects. Optimized for progressive lowering,
    typed transforms, and verification. The LLM does NOT edit this directly.

Layer 2: **Recipe IR** (``ir.recipe``)
    The LLM-facing control IR. Encodes optimization decisions: which regions
    to tile/fuse, which device to target, where copies go, kernel strategy,
    verification obligations. Recipe IR lowers to Transform Dialect scripts +
    kernel search jobs + execution plan fragments + verification obligations.

Layer 3: **Semantic IR** (``ir.semantic``)
    The verification/trust layer. Dialect semantics lowered into semantic
    dialects for translation validation, peephole rewrite verification, and
    dataflow analysis verification. Inspired by "First-Class Verification
    Dialects for MLIR."

Additional dialect sublayers:

    ``ir.accel``   -- Custom accelerator dialect for hardware-specific ops
    ``ir.ukernel`` -- Stable leaf-call boundary for all kernel backends

Cross-cutting:

    ``ir.checks``  -- FileCheck-style IR structural assertions (all layers)
"""

from __future__ import annotations

__all__: list[str] = []
