"""Ukernel interface contracts at IR level.

Defines the contract between the IR and external kernel implementations.
Contracts ensure that the caller and kernel agree on types, layouts,
side effects, and performance bounds.

Invariants:
    - Every UkernelCallOp must have a matching UkernelContract.
    - Contracts are checked at verification time.
    - Contracts are serializable for LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UkernelContract:
    """Contract for a microkernel interface.

    Attributes:
        kernel_name: Kernel identifier (must match UkernelDeclOp).
        input_layouts: Layout requirements per input.
        output_layouts: Layout requirements per output.
        supported_dtypes: Set of supported data types.
        max_workspace_bytes: Maximum scratch workspace the kernel may use.
        perf_bound_us: Performance upper bound in microseconds (if known).
        side_effects: Side effect classification.
    """

    kernel_name: str
    input_layouts: list[str] = field(default_factory=list)
    output_layouts: list[str] = field(default_factory=list)
    supported_dtypes: set[str] = field(default_factory=lambda: {"float32"})
    max_workspace_bytes: int = 0
    perf_bound_us: float | None = None
    side_effects: str = "readwrite"
    # Layout bridge fields (additive, backward-compatible)
    accepted_input_layouts: tuple[str, ...] = ()
    preferred_input_layouts: tuple[str, ...] = ()
    can_absorb_transpose: bool = False
    supports_prepacked_rhs: bool = False
    supports_prepacked_lhs: bool = False
    tile_layout_family: str = ""


def check_contract(call: Any, contract: Any) -> bool:
    """Check that a UkernelCallOp satisfies its contract.

    Args:
        call: A UkernelCallOp instance.
        contract: A UkernelContract instance.

    Returns:
        True if the call satisfies all contract requirements.
    """
    # Name must match
    if call.kernel_name != contract.kernel_name:
        return False

    # Operand count must match input layout count (if layouts specified)
    if contract.input_layouts and len(call.operands) != len(contract.input_layouts):
        return False

    # Result count must match output layout count (if layouts specified)
    if contract.output_layouts and len(call.results) != len(contract.output_layouts):
        return False

    # Workspace must not exceed contract maximum
    if call.workspace_bytes > contract.max_workspace_bytes:
        return False

    # Dtype check: if call metadata specifies a dtype, it must be supported
    call_dtype = call.metadata.get("dtype", None) if hasattr(call, "metadata") else None
    if call_dtype is not None and call_dtype not in contract.supported_dtypes:
        return False

    return True


__all__ = ["UkernelContract", "check_contract"]
