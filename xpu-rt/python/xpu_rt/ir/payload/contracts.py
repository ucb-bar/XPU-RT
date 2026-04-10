"""Kernel and op contracts.

Contracts define the interface between the IR and kernel selection/generation.
Each op or subgraph in the payload IR has a contract specifying:

- Required input/output layouts (strides, alignment)
- Aliasing constraints (which buffers may alias)
- Cost estimates (compute, memory bandwidth, latency)
- Supported dtypes and shape constraints
- Fusion boundary preferences

Contracts are extracted from the canonical IR and used by:
- The kernel selector (to choose native/library/autocomp/fallback)
- The LLM (as context for transform/kernel generation)
- The verifier (to check kernel compliance)

Invariants:
    - Every op in the IR must have a contract (even if it's "unknown").
    - Contracts are serializable to YAML for LLM context injection.
    - Layout requirements are explicit, not inferred from conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import GenericOp, MatmulOp, TransposeOp
from xdsl.ir import Operation


class LayoutKind(Enum):
    """Memory layout specification."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    CUSTOM_STRIDES = "custom_strides"
    ANY = "any"


@dataclass(frozen=True)
class LayoutRequirement:
    """Layout requirement for a tensor operand.

    Attributes:
        kind: The required layout kind.
        strides: Explicit strides (if kind is CUSTOM_STRIDES).
        alignment: Required alignment in bytes.
    """

    kind: LayoutKind = LayoutKind.ANY
    strides: tuple[int, ...] | None = None
    alignment: int = 1


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of an operation.

    Attributes:
        flops: Estimated floating-point operations.
        bytes_read: Estimated bytes read from memory.
        bytes_written: Estimated bytes written to memory.
        latency_us: Estimated latency in microseconds (if known).
    """

    flops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    latency_us: float | None = None


@dataclass(frozen=True)
class KernelContract:
    """Contract for an op or subgraph in the payload IR.

    Attributes:
        op_name: Name of the op or subgraph.
        input_layouts: Layout requirements per input.
        output_layouts: Layout requirements per output.
        supported_dtypes: Set of supported dtype strings.
        aliasing: Pairs of (input_idx, output_idx) that may alias.
        cost: Estimated cost.
        fusable: Whether this op can be fused with neighbors.
        metadata: Additional op-specific metadata.
    """

    op_name: str
    input_layouts: list[LayoutRequirement] = field(default_factory=list)
    output_layouts: list[LayoutRequirement] = field(default_factory=list)
    supported_dtypes: set[str] = field(default_factory=lambda: {"float32"})
    aliasing: list[tuple[int, int]] = field(default_factory=list)
    cost: CostEstimate = field(default_factory=CostEstimate)
    fusable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Layout bridge fields (additive, backward-compatible)
    accepted_input_layouts: list[list[LayoutRequirement]] = field(default_factory=list)
    preferred_input_layouts: list[LayoutRequirement] = field(default_factory=list)
    can_absorb_transpose: bool = False
    supports_prepacked_lhs: bool = False
    supports_prepacked_rhs: bool = False
    tile_layout_family: str | None = None
    materialization_cost_hint: float = 0.0


def _estimate_matmul_flops(op: MatmulOp) -> int:
    """Estimate FLOPs for a matmul operation from its operand types."""
    for operand in op.operands:
        if isinstance(operand.type, TensorType):
            shape = operand.type.get_shape()
            if len(shape) >= 2:
                # M×K × K×N matmul: 2*M*N*K FLOPs
                dim_m = shape[0] if shape[0] != -1 else 1
                dim_k = shape[1] if shape[1] != -1 else 1
                # Get N from the second operand
                for other in op.operands:
                    if other is not operand and isinstance(other.type, TensorType):
                        other_shape = other.type.get_shape()
                        if len(other_shape) >= 2:
                            dim_n = other_shape[1] if other_shape[1] != -1 else 1
                            return 2 * dim_m * dim_n * dim_k
    return 0


def _dtype_bytes(dtype_name: str) -> int:
    """Bytes per element for common dtypes."""
    return {"f32": 4, "f64": 8, "f16": 2, "bf16": 2, "f8e4m3": 1, "i32": 4, "i8": 1}.get(dtype_name, 4)


def _extract_op_contract(op: Operation) -> KernelContract | None:
    """Extract a KernelContract from a single xDSL operation."""
    if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
        return None
    if not op.results:
        return None

    op_name = op.name
    input_layouts: list[LayoutRequirement] = []
    output_layouts: list[LayoutRequirement] = []
    dtypes: set[str] = set()
    flops = 0
    bytes_read = 0
    bytes_written = 0
    fusable = True

    # Extract input info
    for operand in op.operands:
        input_layouts.append(LayoutRequirement(kind=LayoutKind.ROW_MAJOR))
        if isinstance(operand.type, TensorType):
            shape = operand.type.get_shape()
            dtype_str = str(operand.type.element_type)
            dtypes.add(dtype_str)
            elem_bytes = _dtype_bytes(dtype_str)
            num_elements = 1
            for dim in shape:
                if dim > 0:
                    num_elements *= dim
            bytes_read += num_elements * elem_bytes

    # Extract output info
    for result in op.results:
        output_layouts.append(LayoutRequirement(kind=LayoutKind.ROW_MAJOR))
        if isinstance(result.type, TensorType):
            shape = result.type.get_shape()
            dtype_str = str(result.type.element_type)
            dtypes.add(dtype_str)
            elem_bytes = _dtype_bytes(dtype_str)
            num_elements = 1
            for dim in shape:
                if dim > 0:
                    num_elements *= dim
            bytes_written += num_elements * elem_bytes

    # Estimate FLOPs and layout semantics
    can_absorb_transpose = False
    supports_prepacked_lhs = False
    supports_prepacked_rhs = False
    tile_layout_family: str | None = None

    if isinstance(op, MatmulOp):
        flops = _estimate_matmul_flops(op)
        fusable = False  # Matmul is typically a kernel boundary
        can_absorb_transpose = True
        supports_prepacked_rhs = True
        tile_layout_family = "mma"
    elif isinstance(op, GenericOp):
        # Generic ops: estimate from output size
        for result in op.results:
            if isinstance(result.type, TensorType):
                shape = result.type.get_shape()
                size = 1
                for dim in shape:
                    if dim > 0:
                        size *= dim
                flops = size  # 1 FLOP per element (rough)
    elif isinstance(op, TransposeOp):
        flops = 0  # Transpose is just a layout change
        fusable = True
        can_absorb_transpose = True

    if not dtypes:
        dtypes = {"float32"}

    return KernelContract(
        op_name=op_name,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        supported_dtypes=dtypes,
        cost=CostEstimate(
            flops=flops,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        ),
        fusable=fusable,
        can_absorb_transpose=can_absorb_transpose,
        supports_prepacked_lhs=supports_prepacked_lhs,
        supports_prepacked_rhs=supports_prepacked_rhs,
        tile_layout_family=tile_layout_family,
    )


def extract_contracts(module: ModuleOp) -> list[KernelContract]:
    """Extract kernel contracts from a canonical xDSL module.

    Args:
        module: A canonicalized xDSL module.

    Returns:
        List of KernelContract, one per significant op.
    """
    contracts: list[KernelContract] = []
    for op in module.walk():
        contract = _extract_op_contract(op)
        if contract is not None:
            contracts.append(contract)
    return contracts


__all__ = ["CostEstimate", "KernelContract", "LayoutKind", "LayoutRequirement", "extract_contracts"]
