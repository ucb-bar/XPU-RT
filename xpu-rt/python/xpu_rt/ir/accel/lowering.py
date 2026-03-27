"""Lowering from accelerator dialect to backend targets.

Lowers accelerator dialect ops to:
    - LLVM dialect + target intrinsics (for LLVM-based backends)
    - Vendor runtime API calls (for vendor-specific backends)
    - Binary/firmware commands (for direct hardware interfaces)

Invariants:
    - Lowering is target-profile-driven.
    - LLVM intrinsics appear only at this stage, not earlier.
    - Lowering failures produce diagnostics.

TODO: Implement lower_accel_to_llvm() for LLVM-based targets.
TODO: Implement lower_accel_to_runtime() for runtime API targets.
"""

from __future__ import annotations

from typing import Any


def lower_accel_to_llvm(module: Any, target_triple: str = "") -> Any:
    """Lower accelerator dialect ops to LLVM dialect.

    Args:
        module: xDSL module with accelerator dialect ops.
        target_triple: LLVM target triple.

    Returns:
        Module with LLVM dialect ops.

    TODO: Map each accel op to corresponding LLVM intrinsics/ops.
    """
    raise NotImplementedError("lower_accel_to_llvm is not yet implemented")


__all__ = ["lower_accel_to_llvm"]
