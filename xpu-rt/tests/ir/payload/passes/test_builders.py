"""Tests for W0.5 pattern-construction helpers."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes._builders import (
    affine_map_broadcast,
    affine_map_identity,
    affine_map_transpose,
    insert_arith_cast,
    linalg_generic_elementwise,
    linalg_generic_matmul_like,
    linalg_generic_reduction,
)
from xdsl.dialects.arith import AddfOp, MulfOp
from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    TensorType,
)
from xdsl.dialects.linalg import GenericOp, YieldOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir.affine import AffineExpr, AffineMap


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _value(shape):
    return EmptyOp([], _ft(shape)).results[0]


# --- AffineMap builders -------------------------------------------------------


def test_affine_map_identity_rank_0():
    assert affine_map_identity(0).num_dims == 0


def test_affine_map_identity_rank_3_is_identity():
    m = affine_map_identity(3)
    assert m.num_dims == 3


def test_affine_map_identity_rejects_negative_rank():
    with pytest.raises(ValueError):
        affine_map_identity(-1)


def test_affine_map_transpose_swaps():
    m = affine_map_transpose(2, [1, 0])
    assert m.num_dims == 2


def test_affine_map_transpose_three_way():
    m = affine_map_transpose(3, [2, 0, 1])
    assert m.num_dims == 3


def test_affine_map_transpose_rejects_non_permutation():
    with pytest.raises(ValueError, match="permutation"):
        affine_map_transpose(3, [0, 0, 2])


def test_affine_map_broadcast_simple():
    m = affine_map_broadcast(3, [1])
    assert m.num_dims == 3


def test_affine_map_broadcast_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        affine_map_broadcast(2, [5])


def test_affine_map_broadcast_rejects_duplicate():
    with pytest.raises(ValueError, match="unique"):
        affine_map_broadcast(3, [1, 1])


# --- linalg_generic_elementwise ----------------------------------------------


def _ew_add_body(args, block):
    add = AddfOp(args[0], args[1])
    block.add_op(add)
    block.add_op(YieldOp(add.result))


def test_linalg_generic_elementwise_verifies():
    t = _ft([4, 8])
    a = _value([4, 8])
    b = _value([4, 8])
    init = _value([4, 8])
    op = linalg_generic_elementwise([a, b], init, t, _ew_add_body)
    op.verify()
    assert isinstance(op, GenericOp)
    assert len(op.inputs) == 2
    assert op.result_types[0] == t


def test_linalg_generic_elementwise_rejects_non_tensor_init():
    # init must be TensorType.
    with pytest.raises(TypeError):

        class Fake:
            type = Float32Type()

        linalg_generic_elementwise([_value([4])], Fake(), _ft([4]), _ew_add_body)


def test_linalg_generic_elementwise_body_must_yield():
    def broken_body(args, block):
        # no YieldOp
        add = AddfOp(args[0], args[1])
        block.add_op(add)

    with pytest.raises(ValueError, match="linalg.yield"):
        linalg_generic_elementwise(
            [_value([4, 8]), _value([4, 8])],
            _value([4, 8]),
            _ft([4, 8]),
            broken_body,
        )


# --- linalg_generic_reduction ------------------------------------------------


def _red_sum_body(args, block):
    add = AddfOp(args[0], args[1])
    block.add_op(add)
    block.add_op(YieldOp(add.result))


def test_linalg_generic_reduction_drops_dim():
    t = _ft([4])
    a = _value([4, 8])
    init = _value([4])
    op = linalg_generic_reduction(a, init, t, reduction_dims=[1], body=_red_sum_body)
    op.verify()


def test_linalg_generic_reduction_rejects_out_of_range():
    a = _value([4, 8])
    init = _value([4])
    with pytest.raises(ValueError, match="out of range"):
        linalg_generic_reduction(a, init, _ft([4]), [5], body=_red_sum_body)


def test_linalg_generic_reduction_rejects_duplicates():
    a = _value([4, 8])
    init = _value([])
    with pytest.raises(ValueError, match="unique"):
        linalg_generic_reduction(a, init, _ft([]), [0, 0], body=_red_sum_body)


# --- linalg_generic_matmul_like ----------------------------------------------


def _mm_body(args, block):
    mul = MulfOp(args[0], args[1])
    block.add_op(mul)
    add = AddfOp(mul.result, args[2])
    block.add_op(add)
    block.add_op(YieldOp(add.result))


def test_matmul_like_basic_matmul_shape():
    M, K, N = 4, 8, 16
    lhs = _value([M, K])
    rhs = _value([K, N])
    out = _value([M, N])

    d0, d1, d2 = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    lhs_map = AffineMap(3, 0, (d0, d2))
    rhs_map = AffineMap(3, 0, (d2, d1))
    out_map = AffineMap(3, 0, (d0, d1))

    op = linalg_generic_matmul_like(
        lhs,
        rhs,
        out,
        _ft([M, N]),
        lhs_map,
        rhs_map,
        out_map,
        _mm_body,
    )
    op.verify()


def test_matmul_like_transpose_a_shape():
    # lhs [K, M], rhs [K, N] -> out [M, N] with lhs_map (d0, d2) -> (d2, d0).
    M, K, N = 4, 8, 16
    lhs = _value([K, M])
    rhs = _value([K, N])
    out = _value([M, N])

    d0, d1, d2 = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    lhs_map = AffineMap(3, 0, (d2, d0))  # transposed
    rhs_map = AffineMap(3, 0, (d2, d1))
    out_map = AffineMap(3, 0, (d0, d1))

    op = linalg_generic_matmul_like(
        lhs,
        rhs,
        out,
        _ft([M, N]),
        lhs_map,
        rhs_map,
        out_map,
        _mm_body,
    )
    op.verify()


# --- insert_arith_cast -------------------------------------------------------


def test_insert_arith_cast_truncates_when_narrowing():
    a = _value([4, 8])
    out = insert_arith_cast(a, Float16Type())
    assert out is not a
    assert out.type == TensorType(Float16Type(), [4, 8])


def test_insert_arith_cast_extends_when_widening():
    a = _value([4, 8])
    out = insert_arith_cast(a, Float64Type())
    assert out is not a
    assert out.type == TensorType(Float64Type(), [4, 8])


def test_insert_arith_cast_is_noop_on_same_type():
    a = _value([4, 8])
    assert insert_arith_cast(a, Float32Type()) is a


def test_insert_arith_cast_rejects_non_tensor():
    class Fake:
        type = Float32Type()

    with pytest.raises(TypeError):
        insert_arith_cast(Fake(), Float16Type())
