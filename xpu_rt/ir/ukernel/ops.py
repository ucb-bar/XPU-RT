"""Ukernel dialect operations.

Four operation types form the ukernel contract layer:

    - ``UkernelDeclOp``: semantic + layout contract (what it computes,
      what layouts it accepts, transparency level).
    - ``UkernelMatchOp``: selection constraints (when is this kernel
      legal/profitable for a given target context).
    - ``UkernelBodyOp``: implementation (transparent MLIR/xDSL body,
      or opaque C/Triton/library/binary reference).
    - ``UkernelCallOp``: stable call boundary in the graph.

Two execution classes share one unified contract:

    1. **Transparent ukernels** -- MLIR/xDSL bodies that stay compiler-visible
       for fusion, layout propagation, prepacking, and target-aware lowering.
    2. **Opaque ukernels** -- C/Triton/library/binary bodies that satisfy the
       same contract but are black boxes to the compiler.

Invariants:
    - UkernelCallOp is the only way to invoke an external kernel in the IR.
    - Every call carries metadata sufficient for the planner to schedule it.
    - Declarations and calls must match (checked by ukernel verification).
    - One UkernelDeclOp may have multiple UkernelMatchOp (different targets)
      and multiple UkernelBodyOp (different implementations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UkernelDeclOp:
    """Declare a microkernel interface with semantic and layout contract.

    Attributes:
        kernel_name: Unique kernel identifier.
        input_types: List of input type descriptions.
        output_types: List of output type descriptions.
        side_effects: Side effect classification ("none", "read", "write", "readwrite").
        calling_convention: Calling convention ("c", "triton", "nki", "cuda").
        transparency: "transparent" (compiler-visible body) or "opaque" (black box).
        body_kind: Body format ("mlir", "xdsl", "triton", "c", "cpp", "python", "library", "binary").
        accepted_layouts: Layout strings this kernel can accept (empty = any).
        preferred_layouts: Preferred layouts for optimal performance.
        output_layout: Output layout produced (empty = unspecified).
        supports_prepacked_rhs: Can consume prepacked RHS operand.
        supports_transpose_absorption: Can absorb input transposes internally.
        tile_family: Tile/blocking family name (e.g., "mma", "rvv_vlmul", "npu_tile").
    """

    kernel_name: str
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    side_effects: str = "readwrite"
    calling_convention: str = "c"
    # Transparency and body
    transparency: str = "opaque"
    body_kind: str = "library"
    # Layout contract
    accepted_layouts: tuple[str, ...] = ()
    preferred_layouts: tuple[str, ...] = ()
    output_layout: str = ""
    supports_prepacked_rhs: bool = False
    supports_transpose_absorption: bool = False
    tile_family: str = ""


@dataclass(frozen=True)
class UkernelMatchOp:
    """Declarative selection/matching constraints for a ukernel.

    One UkernelDeclOp may have multiple UkernelMatchOp entries for
    different target contexts. The registry evaluates all constraints
    and selects the highest-priority match.

    Constraints are declarative strings evaluated by the constraint
    evaluator (no eval(), no code execution). Examples:
        - Shape: "M%16==0", "K>=32"
        - Feature: "has_tensor_core", "has_rvv"
        - Device: "device_type==gpu", "device_type==npu"
        - Dtype: "dtype==float32", "dtype_in(float16,bfloat16)"
        - Layout: "lhs_rowmajor", "rhs_prepacked"

    Attributes:
        kernel_name: Links to UkernelDeclOp.
        op_family: Op pattern this matches ("matmul", "matmul_bias", "attention_score").
        dtype_constraints: Required dtype predicates.
        shape_constraints: Shape predicates.
        target_constraints: Target capability predicates.
        layout_constraints: Layout predicates.
        priority: Higher = preferred when multiple match.
    """

    kernel_name: str
    op_family: str = ""
    dtype_constraints: tuple[str, ...] = ()
    shape_constraints: tuple[str, ...] = ()
    target_constraints: tuple[str, ...] = ()
    layout_constraints: tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class UkernelBodyOp:
    """Implementation body for a ukernel.

    One UkernelDeclOp may have multiple bodies for different targets
    or body kinds. The registry selects the best body based on
    target_family.

    Attributes:
        kernel_name: Links to UkernelDeclOp.
        body_kind: Implementation format ("mlir", "xdsl", "triton", "c", "library", etc.).
        transparency: "transparent" (compiler can see through) or "opaque" (black box).
        source_ref: Path to source file for opaque/external bodies.
        inline_body: Inline implementation code for transparent or small bodies.
        target_family: Which target family this body is for, or "any".
        body_metadata: Extra metadata (compiler flags, tile sizes, etc.).
    """

    kernel_name: str
    body_kind: str = "library"
    transparency: str = "opaque"
    source_ref: str = ""
    inline_body: str = ""
    target_family: str = "any"
    body_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UkernelCallOp:
    """Call a declared microkernel.

    Attributes:
        kernel_name: Name of the declared kernel.
        operands: List of operand names/references.
        results: List of result names/references.
        workspace_bytes: Scratch workspace required.
        metadata: Additional call metadata (device affinity, perf hints).
        selected_body: Which body was selected during matching (empty = not yet selected).
    """

    kernel_name: str
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    workspace_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    selected_body: str = ""


__all__ = ["UkernelBodyOp", "UkernelCallOp", "UkernelDeclOp", "UkernelMatchOp"]
