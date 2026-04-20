"""Construction helpers for the Wave 1+ pattern rewrites.

Each rewrite in :mod:`compgen.ir.payload.passes.rewrites` needs a
small, repetitive set of IR constructions: build a ``linalg.generic``
with an elementwise body, build an indexing map that transposes two
dims, or insert an ``arith.truncf``/``extf`` pair. Re-deriving these
inline makes every pass 30+ lines longer than it needs to be and
introduces subtle indexing_map bugs. This module centralises them.

Public entry points:

- :func:`linalg_generic_elementwise` -- an elementwise
  ``linalg.generic`` over N input tensors + 1 init tensor with a
  caller-supplied body function.
- :func:`linalg_generic_reduction` -- a reduction ``linalg.generic``
  over specified dims.
- :func:`linalg_generic_matmul_like` -- a matmul-shaped
  ``linalg.generic`` parametrised by indexing maps (so callers can
  pass ``matmul``, ``matmul_transpose_a``, ``matmul_transpose_b``
  shapes).
- :func:`affine_map_identity(rank)`,
  :func:`affine_map_transpose(rank, perm)`,
  :func:`affine_map_broadcast(rank, bcast_dims)` -- plain AffineMap
  constructors.
- :func:`insert_arith_cast(value, target_elem_type, rewriter)` --
  inserts ``arith.truncf`` or ``arith.extf`` (or returns ``value``
  unchanged when ``target_elem_type`` already matches).
"""

from __future__ import annotations

from collections.abc import Callable

from xdsl.dialects.arith import ExtFOp, TruncFOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    TensorType,
)
from xdsl.dialects.linalg import GenericOp, IteratorTypeAttr, YieldOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import PatternRewriter

# --- AffineMap builders ------------------------------------------------------


def affine_map_identity(rank: int) -> AffineMap:
    """``(d0, d1, ..., d_{rank-1}) -> (d0, d1, ..., d_{rank-1})``."""
    if rank < 0:
        raise ValueError(f"affine_map_identity: rank must be non-negative, got {rank}")
    return AffineMap.identity(rank)


def affine_map_transpose(rank: int, perm: list[int]) -> AffineMap:
    """Permute dims by ``perm``.

    ``affine_map_transpose(3, [1, 0, 2])`` -> ``(d0, d1, d2) -> (d1, d0, d2)``.

    ``perm`` must be a permutation of ``range(rank)``.
    """
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"affine_map_transpose: perm {perm} must be a permutation of range({rank})")
    exprs = tuple(AffineExpr.dimension(perm[i]) for i in range(rank))
    return AffineMap(rank, 0, exprs)


def affine_map_broadcast(rank: int, bcast_dims: list[int]) -> AffineMap:
    """Identity map with ``bcast_dims`` collapsed to 0.

    Example: ``affine_map_broadcast(3, [1])`` ->
    ``(d0, d1, d2) -> (d0, 0, d2)``. Used to broadcast a 1-D bias
    into a matmul output in ``linalg.generic`` bodies.

    ``bcast_dims`` entries must be unique and in ``[0, rank)``.
    """
    if any(d < 0 or d >= rank for d in bcast_dims):
        raise ValueError(f"affine_map_broadcast: bcast_dims {bcast_dims} out of range for rank {rank}")
    if len(set(bcast_dims)) != len(bcast_dims):
        raise ValueError(f"affine_map_broadcast: bcast_dims {bcast_dims} must be unique")
    bset = set(bcast_dims)
    exprs = tuple(AffineExpr.constant(0) if i in bset else AffineExpr.dimension(i) for i in range(rank))
    return AffineMap(rank, 0, exprs)


# --- linalg.generic builders --------------------------------------------------


def _iterator_array(kinds: list[str]) -> ArrayAttr:
    from xdsl.dialects.linalg import IteratorType

    valid = {
        "parallel": IteratorType.PARALLEL,
        "reduction": IteratorType.REDUCTION,
        "window": IteratorType.WINDOW,
    }
    attrs = []
    for k in kinds:
        if k not in valid:
            raise ValueError(f"iterator kind {k!r} must be one of {sorted(valid)}")
        attrs.append(IteratorTypeAttr(valid[k]))
    return ArrayAttr(attrs)


def _maps_array(maps: list[AffineMap]) -> ArrayAttr:
    return ArrayAttr([AffineMapAttr(m) for m in maps])


def linalg_generic_elementwise(
    inputs: list[SSAValue],
    init: SSAValue,
    result_type: Attribute,
    body: Callable[[list[SSAValue], Block], None],
) -> GenericOp:
    """Build an elementwise ``linalg.generic`` over ``inputs`` + ``init``.

    All operands are assumed to have the same shape (the shape of
    ``init``). ``body`` is a callable ``(block_args, block) ->
    None`` that appends scalar ops to ``block`` and ends with a
    ``linalg.yield``; ``block_args`` are the scalar args matching
    ``inputs + [init]``.
    """
    init_type = init.type
    if not isinstance(init_type, TensorType):
        raise TypeError(f"linalg_generic_elementwise: init must have TensorType, got {init_type}")
    rank = len(init_type.get_shape())

    maps = [AffineMapAttr(affine_map_identity(rank))] * (len(inputs) + 1)
    iterators = _iterator_array(["parallel"] * rank)

    scalar_arg_types: list[Attribute] = []
    for v in inputs:
        vt = v.type
        if isinstance(vt, TensorType):
            scalar_arg_types.append(vt.get_element_type())
        else:
            scalar_arg_types.append(vt)
    scalar_arg_types.append(init_type.get_element_type())

    block = Block(arg_types=scalar_arg_types)
    body(list(block.args), block)
    # Defensive check: user body must terminate with linalg.yield.
    if not block.ops or not isinstance(block.last_op, YieldOp):
        raise ValueError("linalg_generic_elementwise: body must end with linalg.yield")

    region = Region([block])
    return GenericOp(
        inputs=list(inputs),
        outputs=[init],
        body=region,
        indexing_maps=maps,
        iterator_types=iterators,
        result_types=[result_type],
    )


def linalg_generic_reduction(
    input: SSAValue,
    init: SSAValue,
    result_type: Attribute,
    reduction_dims: list[int],
    body: Callable[[list[SSAValue], Block], None],
) -> GenericOp:
    """Build a reduction ``linalg.generic`` over ``reduction_dims``.

    The result rank equals the input rank minus the number of
    ``reduction_dims`` -- matching ``tensor.dim(input, d)`` for
    every ``d not in reduction_dims``.

    ``body(args, block)``: ``args == [input_scalar, init_scalar]``.
    """
    in_type = input.type
    if not isinstance(in_type, TensorType):
        raise TypeError("linalg_generic_reduction: input must have TensorType")
    in_rank = len(in_type.get_shape())

    init_type = init.type
    if not isinstance(init_type, TensorType):
        raise TypeError("linalg_generic_reduction: init must have TensorType")

    for d in reduction_dims:
        if d < 0 or d >= in_rank:
            raise ValueError(f"reduction_dim {d} out of range for input rank {in_rank}")
    if len(set(reduction_dims)) != len(reduction_dims):
        raise ValueError(f"reduction_dims {reduction_dims} must be unique")

    # Input map: identity over all dims.
    input_map = affine_map_identity(in_rank)
    # Output map: skip reduction dims.
    kept = [i for i in range(in_rank) if i not in reduction_dims]
    output_exprs = tuple(AffineExpr.dimension(i) for i in kept)
    output_map = AffineMap(in_rank, 0, output_exprs)

    iterator_kinds = ["reduction" if i in reduction_dims else "parallel" for i in range(in_rank)]
    iterators = _iterator_array(iterator_kinds)

    block = Block(arg_types=[in_type.get_element_type(), init_type.get_element_type()])
    body(list(block.args), block)
    if not block.ops or not isinstance(block.last_op, YieldOp):
        raise ValueError("linalg_generic_reduction: body must end with linalg.yield")

    region = Region([block])
    return GenericOp(
        inputs=[input],
        outputs=[init],
        body=region,
        indexing_maps=[AffineMapAttr(input_map), AffineMapAttr(output_map)],
        iterator_types=iterators,
        result_types=[result_type],
    )


def linalg_generic_matmul_like(
    lhs: SSAValue,
    rhs: SSAValue,
    init: SSAValue,
    result_type: Attribute,
    lhs_map: AffineMap,
    rhs_map: AffineMap,
    output_map: AffineMap,
    body: Callable[[list[SSAValue], Block], None],
) -> GenericOp:
    """Build a matmul-shaped ``linalg.generic``.

    Callers supply the three indexing maps explicitly so this handles
    ``matmul`` (``[d0,k]`` x ``[k,d1]`` -> ``[d0,d1]``),
    ``matmul_transpose_a`` (``[k,d0]`` x ``[k,d1]`` -> ``[d0,d1]``),
    and ``matmul_transpose_b`` (``[d0,k]`` x ``[d1,k]`` -> ``[d0,d1]``)
    without bespoke code.

    ``body(args, block)``: ``args == [lhs_scalar, rhs_scalar, init_scalar]``.
    """
    lhs_type = lhs.type
    rhs_type = rhs.type
    init_type = init.type
    for label, t in (("lhs", lhs_type), ("rhs", rhs_type), ("init", init_type)):
        if not isinstance(t, TensorType):
            raise TypeError(f"linalg_generic_matmul_like: {label} must have TensorType")

    dim_count = max(m.num_dims for m in (lhs_map, rhs_map, output_map))
    # Two parallel output dims followed by ``dim_count - 2`` reduction
    # dims. Callers that need a more exotic ordering should use
    # ``linalg_generic_reduction`` directly.
    iterator_kinds = ["parallel", "parallel"] + ["reduction"] * (dim_count - 2)
    iterators = _iterator_array(iterator_kinds)

    block = Block(
        arg_types=[
            lhs_type.get_element_type(),
            rhs_type.get_element_type(),
            init_type.get_element_type(),
        ]
    )
    body(list(block.args), block)
    if not block.ops or not isinstance(block.last_op, YieldOp):
        raise ValueError("linalg_generic_matmul_like: body must end with linalg.yield")

    region = Region([block])
    return GenericOp(
        inputs=[lhs, rhs],
        outputs=[init],
        body=region,
        indexing_maps=[
            AffineMapAttr(lhs_map),
            AffineMapAttr(rhs_map),
            AffineMapAttr(output_map),
        ],
        iterator_types=iterators,
        result_types=[result_type],
    )


# --- arith cast insertion -----------------------------------------------------


def insert_arith_cast(
    value: SSAValue,
    target_elem_type: Attribute,
    rewriter: PatternRewriter | None = None,
    *,
    insert_point_op: Operation | None = None,
) -> SSAValue:
    """Insert ``arith.truncf``/``arith.extf`` to cast ``value`` to
    ``target_elem_type`` (applied elementwise over a tensor).

    Chooses direction by comparing bitwidths when both source and
    target are ``FixedBitwidthType``. Returns ``value`` unchanged
    when the types already match (no-op cast).

    When ``rewriter`` is provided the cast op is inserted via
    ``rewriter.insert_op_before_matched_op`` (so pattern rewrites
    stay legal). Otherwise the cast op is returned dangling; the
    caller is responsible for ``block.add_op`` / ``insert_op_after``
    placement.
    """
    value_type = value.type
    if not isinstance(value_type, TensorType):
        raise TypeError(f"insert_arith_cast expects a tensor-typed value, got {value_type}")
    src_elem = value_type.get_element_type()
    if src_elem == target_elem_type:
        return value

    dst_type = TensorType(target_elem_type, list(value_type.get_shape()))

    # Pick truncf (narrowing) vs extf (widening) by bitwidth.
    src_bits = getattr(src_elem, "bitwidth", None)
    dst_bits = getattr(target_elem_type, "bitwidth", None)
    if src_bits is not None and dst_bits is not None:
        if dst_bits < src_bits:
            op: Operation = TruncFOp(value, dst_type)
        else:
            op = ExtFOp(value, dst_type)
    else:
        # When bitwidths are unknown (e.g. abstract float types)
        # default to extf -- safe widening, symmetrical to how the
        # xDSL arith dialect handles equal-bitwidth casts.
        op = ExtFOp(value, dst_type)

    if rewriter is not None:
        rewriter.insert_op_before_matched_op(op)
    elif insert_point_op is not None:
        insert_point_op.parent_block().insert_op_before(op, insert_point_op)
    return op.result


__all__ = [
    "affine_map_broadcast",
    "affine_map_identity",
    "affine_map_transpose",
    "insert_arith_cast",
    "linalg_generic_elementwise",
    "linalg_generic_matmul_like",
    "linalg_generic_reduction",
]
