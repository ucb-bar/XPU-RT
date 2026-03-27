"""ATen to xDSL decomposition table.

Maps PyTorch ATen operator targets to functions that produce real xDSL
linalg/arith/tensor operations. This replaces the opaque ``func.call``
approach with structured IR the agent can reason about.

Each decomposition function takes the FX node's args (as xDSL SSAValues)
and metadata, and returns a list of xDSL Operations to insert into the block.

Ops without decompositions fall back to ``func.call`` (flagged as opaque).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import (
    DenseArrayBase,
    Float32Type,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import MatmulOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Operation, SSAValue


@dataclass
class DecompResult:
    """Result of decomposing one FX node into xDSL ops.

    Attributes:
        ops: xDSL operations to insert into the block.
        result: The SSAValue that represents this node's output.
        region_ids: region_id labels attached to linalg ops.
    """

    ops: list[Operation] = field(default_factory=list)
    result: SSAValue | None = None
    region_ids: list[str] = field(default_factory=list)


# Type for decomposition functions
DecompFn = Callable[
    [
        list[SSAValue],       # positional operands (resolved FX args)
        dict[str, Any],       # FX node metadata (shapes, dtypes)
        str,                  # node name (for region_id generation)
    ],
    DecompResult,
]


# ============================================================================
# Counters for unique region IDs
# ============================================================================

_region_counters: dict[str, int] = {}


def _next_region_id(prefix: str) -> str:
    """Generate a unique region ID like 'matmul_0', 'matmul_1'."""
    count = _region_counters.get(prefix, 0)
    _region_counters[prefix] = count + 1
    return f"{prefix}_{count}"


def reset_region_counters() -> None:
    """Reset counters between imports."""
    _region_counters.clear()


def _make_empty(result_type: TensorType) -> EmptyOp:
    """Create a tensor.empty for an output tensor."""
    return EmptyOp([], result_type)


def _attach_region_id(op: Operation, region_id: str) -> None:
    """Attach a compgen.region_id attribute to an operation."""
    op.attributes["compgen.region_id"] = StringAttr(region_id)


# ============================================================================
# Decomposition functions
# ============================================================================


def decompose_linear(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.linear.default(input, weight, bias?) -> matmul + bias add.

    linear(x, w, b) = x @ w^T + b
    - x: [M, K], w: [N, K] (note: weight is transposed), b: [N]
    - output: [M, N]
    """
    ops: list[Operation] = []
    region_ids: list[str] = []

    x = operands[0]  # input: [M, K]
    w = operands[1]  # weight: [N, K]

    # Get result type from metadata
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    # Step 1: Transpose weight [N, K] -> [K, N]
    w_type = w.type
    assert isinstance(w_type, TensorType)
    w_shape = w_type.get_shape()
    wt_type = TensorType(Float32Type(), [w_shape[1], w_shape[0]])
    wt_empty = _make_empty(wt_type)
    ops.append(wt_empty)

    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=w,
        init=wt_empty.results[0],
        permutation=perm,
        result=wt_type,
    )
    ops.append(transpose)

    # Step 2: Matmul: x [M, K] @ w^T [K, N] -> [M, N]
    mm_empty = _make_empty(result_type)
    ops.append(mm_empty)

    matmul = MatmulOp(
        inputs=[x, transpose.results[0]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)
    region_ids.append(rid)
    ops.append(matmul)

    result = matmul.results[0]

    # Step 3: Add bias if present
    if len(operands) >= 3:
        # TODO: Implement proper bias broadcast + linalg.add
        # For now the matmul result IS the output (bias addition deferred)
        pass

    return DecompResult(ops=ops, result=result, region_ids=region_ids)


def decompose_gelu(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.gelu.default(input) -> element-wise GELU.

    For MVP, represent as a func.call to @aten_gelu (element-wise ops in
    linalg.generic require indexing_maps and a body region, which we'll
    add in a later phase). The region_id is still attached.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    # Create external function declaration for gelu
    # (In later phase, this becomes a linalg.generic with the GELU body)
    rid = _next_region_id("gelu")
    call = CallOp("aten_gelu", [operands[0]], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[call], result=call.res[0], region_ids=[rid])


def decompose_add_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.add.Tensor(a, b) -> element-wise add."""
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    rid = _next_region_id("add")
    call = CallOp("aten_add", [operands[0], operands[1]], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[call], result=call.res[0], region_ids=[rid])


def decompose_mul_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mul.Tensor(a, b) -> element-wise mul."""
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    rid = _next_region_id("mul")
    call = CallOp("aten_mul", [operands[0], operands[1]], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[call], result=call.res[0], region_ids=[rid])


def decompose_mm(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mm.default(a, b) -> linalg.matmul."""
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    mm_empty = _make_empty(result_type)
    matmul = MatmulOp(
        inputs=[operands[0], operands[1]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)

    return DecompResult(ops=[mm_empty, matmul], result=matmul.results[0], region_ids=[rid])


def decompose_transpose(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.t.default(input) -> linalg.transpose."""
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), list(val.shape))

    t_empty = _make_empty(result_type)
    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=operands[0],
        init=t_empty.results[0],
        permutation=perm,
        result=result_type,
    )
    rid = _next_region_id("transpose")
    _attach_region_id(transpose, rid)

    return DecompResult(ops=[t_empty, transpose], result=transpose.results[0], region_ids=[rid])


# ============================================================================
# Decomposition table
# ============================================================================

DECOMPOSITION_TABLE: dict[str, DecompFn] = {
    "aten.linear.default": decompose_linear,
    "aten.gelu.default": decompose_gelu,
    "aten.add.Tensor": decompose_add_tensor,
    "aten.mul.Tensor": decompose_mul_tensor,
    "aten.mm.default": decompose_mm,
    "aten.t.default": decompose_transpose,
}


__all__ = [
    "DECOMPOSITION_TABLE",
    "DecompFn",
    "DecompResult",
    "reset_region_counters",
]
