"""Lowering from ukernel dialect to actual kernel calls.

Converts UkernelCallOp into concrete function calls appropriate for
the kernel backend (Triton, C, CUDA, NKI, vendor library).

Invariants:
    - Lowering is backend-specific (selected by calling_convention).
    - Lowered calls include workspace allocation.
    - Lowering preserves scheduling metadata for the planner.

TODO: Implement lower_ukernel_to_call() dispatching on calling_convention.
TODO: Implement per-backend lowering (Triton, C, CUDA).
"""

from __future__ import annotations

from typing import Any


def lower_ukernel_to_call(module: Any, backend: str = "c") -> Any:
    """Lower ukernel ops to concrete function calls.

    Args:
        module: xDSL module with ukernel ops.
        backend: Target backend for lowering.

    Returns:
        Module with concrete function calls.

    TODO: Walk ukernel ops, dispatch on calling_convention.
    TODO: Insert workspace allocation.
    TODO: Preserve scheduling metadata as attributes.
    """
    raise NotImplementedError("lower_ukernel_to_call is not yet implemented")


__all__ = ["lower_ukernel_to_call"]
