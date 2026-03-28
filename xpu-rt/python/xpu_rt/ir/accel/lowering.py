"""Lowering from accelerator dialect to backend targets.

Lowers accelerator dialect ops to:
    - LLVM dialect + target intrinsics (for LLVM-based backends)
    - Vendor runtime API calls (for vendor-specific backends)
    - Binary/firmware commands (for direct hardware interfaces)

Invariants:
    - Lowering is target-profile-driven.
    - LLVM intrinsics appear only at this stage, not earlier.
    - Lowering failures produce diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.ir.accel.ops import (
    BarrierOp,
    DMAStartOp,
    DMAWaitOp,
    MatrixEngineOp,
    TileLoadOp,
    TileStoreOp,
)


@dataclass
class LoweringOutput:
    """Result of lowering accelerator dialect ops.

    Attributes:
        lowered_ops: List of lowered op descriptors (dicts).
        diagnostics: List of warning/info messages produced during lowering.
    """

    lowered_ops: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def lower_accel_to_llvm(module: Any, target_triple: str = "") -> LoweringOutput:
    """Lower accelerator dialect ops to LLVM dialect descriptors.

    Args:
        module: A single op, or a list/tuple of ops to lower.
        target_triple: LLVM target triple (e.g. ``"x86_64-unknown-linux-gnu"``).

    Returns:
        A :class:`LoweringOutput` with lowered op descriptors and diagnostics.
    """
    result = LoweringOutput()
    ops = module if isinstance(module, (list, tuple)) else [module]

    for op in ops:
        if isinstance(op, TileLoadOp):
            result.lowered_ops.append({
                "type": "memcpy",
                "src": op.src_memref,
                "dst": op.dst_memref,
                "shape": list(op.shape),
                "dtype": op.dtype,
                "target_triple": target_triple,
            })
        elif isinstance(op, TileStoreOp):
            result.lowered_ops.append({
                "type": "memcpy",
                "src": op.src_memref,
                "dst": op.dst_memref,
                "shape": list(op.shape),
                "dtype": op.dtype,
                "target_triple": target_triple,
            })
        elif isinstance(op, DMAStartOp):
            result.lowered_ops.append({
                "type": "dma_start",
                "src": op.src,
                "dst": op.dst,
                "size_bytes": op.size_bytes,
                "event": op.event,
                "target_triple": target_triple,
            })
        elif isinstance(op, DMAWaitOp):
            result.lowered_ops.append({
                "type": "dma_wait",
                "event": op.event,
                "target_triple": target_triple,
            })
        elif isinstance(op, MatrixEngineOp):
            result.lowered_ops.append({
                "type": "matrix_engine",
                "op_kind": op.op_kind,
                "a_ref": op.a_ref,
                "b_ref": op.b_ref,
                "c_ref": op.c_ref,
                "config": dict(op.config),
                "target_triple": target_triple,
            })
        elif isinstance(op, BarrierOp):
            result.lowered_ops.append({
                "type": "barrier",
                "scope": op.scope,
                "target_triple": target_triple,
            })
        else:
            result.diagnostics.append(f"Unsupported op type: {type(op).__name__}")

    return result


__all__ = ["LoweringOutput", "lower_accel_to_llvm"]
