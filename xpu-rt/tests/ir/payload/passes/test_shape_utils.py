"""Tests for W0.5 shape utilities."""

from __future__ import annotations

from compgen.ir.payload.passes._shape_utils import (
    common_element_type,
    infer_result_shape,
    is_static_shape,
    rank_of,
    static_shape_or_none,
)
from xdsl.dialects.builtin import Float16Type, Float32Type, TensorType
from xdsl.dialects.tensor import EmptyOp


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def test_static_shape_recovers_tuple():
    assert static_shape_or_none(_ft([4, 8])) == (4, 8)


def test_static_shape_returns_none_on_dynamic_dim():
    assert static_shape_or_none(_ft([4, -1])) is None


def test_static_shape_on_ssa_value():
    v = EmptyOp([], _ft([2, 3])).results[0]
    assert static_shape_or_none(v) == (2, 3)


def test_static_shape_on_non_tensor():
    assert static_shape_or_none(Float32Type()) is None


def test_is_static_shape_bool():
    assert is_static_shape(_ft([4, 8]))
    assert not is_static_shape(_ft([-1, 8]))


def test_rank_of():
    assert rank_of(_ft([4, 8])) == 2
    assert rank_of(Float32Type()) is None


# --- infer_result_shape -------------------------------------------------------


def test_infer_elementwise():
    assert infer_result_shape("elementwise", [(4, 8), (4, 8)]) == (4, 8)


def test_infer_elementwise_mismatch_returns_none():
    assert infer_result_shape("elementwise", [(4, 8), (4, 9)]) is None


def test_infer_matmul():
    assert infer_result_shape("matmul", [(4, 8), (8, 16)]) == (4, 16)


def test_infer_matmul_k_mismatch_returns_none():
    assert infer_result_shape("matmul", [(4, 8), (7, 16)]) is None


def test_infer_matmul_dynamic_k_allowed():
    assert infer_result_shape("matmul", [(4, -1), (8, 16)]) == (4, 16)
    assert infer_result_shape("matmul", [(4, 8), (-1, 16)]) == (4, 16)


def test_infer_matmul_wrong_rank_returns_none():
    assert infer_result_shape("matmul", [(4, 8, 2), (8, 16)]) is None


def test_infer_concat_basic():
    assert infer_result_shape("concat", [(2, 4), (3, 4)], axis=0) == (5, 4)


def test_infer_concat_three_way():
    assert infer_result_shape("concat", [(2, 4), (3, 4), (5, 4)], axis=0) == (10, 4)


def test_infer_concat_axis_mismatch_returns_none():
    assert infer_result_shape("concat", [(2, 4), (3, 5)], axis=0) is None


def test_infer_concat_dynamic_dim_propagates():
    assert infer_result_shape("concat", [(-1, 4), (3, 4)], axis=0) == (-1, 4)


def test_infer_reduction():
    assert infer_result_shape("reduction", [(4, 8, 16)], reduction_dims=[1]) == (4, 16)


def test_infer_reduction_rejects_bad_dim():
    assert infer_result_shape("reduction", [(4, 8)], reduction_dims=[5]) is None


def test_infer_transpose():
    assert infer_result_shape("transpose", [(4, 8, 16)], perm=[2, 0, 1]) == (16, 4, 8)


def test_infer_transpose_rejects_bad_perm():
    assert infer_result_shape("transpose", [(4, 8)], perm=[0, 0]) is None


def test_infer_unknown_kind_returns_none():
    assert infer_result_shape("not_a_real_op", [(4,)]) is None


# --- common_element_type -----------------------------------------------------


def test_common_element_type_agreement():
    assert common_element_type([_ft([4, 8]), _ft([2, 2])]) == Float32Type()


def test_common_element_type_disagreement():
    a = TensorType(Float32Type(), [4])
    b = TensorType(Float16Type(), [4])
    assert common_element_type([a, b]) is None


def test_common_element_type_with_non_tensor():
    assert common_element_type([_ft([4]), Float32Type()]) is None
