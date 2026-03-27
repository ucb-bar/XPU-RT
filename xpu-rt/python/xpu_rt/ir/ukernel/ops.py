"""Ukernel dialect operations.

Invariants:
    - UkernelCallOp is the only way to invoke an external kernel in the IR.
    - Every call carries metadata sufficient for the planner to schedule it.
    - Declarations and calls must match (checked by ukernel verification).

TODO: Implement as xDSL Operation subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UkernelDeclOp:
    """Declare a microkernel interface.

    Attributes:
        kernel_name: Unique kernel identifier.
        input_types: List of input type descriptions.
        output_types: List of output type descriptions.
        side_effects: Side effect classification ("none", "read", "write", "readwrite").
        calling_convention: Calling convention ("c", "triton", "nki", "cuda").
    """

    kernel_name: str
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    side_effects: str = "readwrite"
    calling_convention: str = "c"


@dataclass(frozen=True)
class UkernelCallOp:
    """Call a declared microkernel.

    Attributes:
        kernel_name: Name of the declared kernel.
        operands: List of operand names/references.
        results: List of result names/references.
        workspace_bytes: Scratch workspace required.
        metadata: Additional call metadata (device affinity, perf hints).
    """

    kernel_name: str
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    workspace_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["UkernelCallOp", "UkernelDeclOp"]
