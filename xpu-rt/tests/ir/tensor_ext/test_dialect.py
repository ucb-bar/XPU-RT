"""Tests for the ``compgen.tensor_ext`` dialect."""

from __future__ import annotations

import pytest
from compgen.ir.tensor_ext import (
    ALL_OPS,
    ConcatOp,
    PackOp,
    TensorExt,
    UnpackOp,
)
from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Dialect
from xdsl.utils.exceptions import VerifyException


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _value(shape):
    return EmptyOp([], _ft(shape)).results[0]


# --- registration -------------------------------------------------------------


def test_dialect_is_registered():
    assert isinstance(TensorExt, Dialect)
    assert TensorExt.name == "compgen.tensor_ext"


def test_dialect_has_three_ops():
    assert len(ALL_OPS) == 3


def test_op_names_registered():
    names = {op.name for op in TensorExt._operations}
    assert names == {
        "compgen.tensor_ext.concat",
        "compgen.tensor_ext.pack",
        "compgen.tensor_ext.unpack",
    }


# --- ConcatOp -----------------------------------------------------------------


def test_concat_basic_outer_dim():
    a, b = _value([2, 4]), _value([3, 4])
    op = ConcatOp([a, b], dim=0, result_type=_ft([5, 4]))
    op.verify()
    assert op.dim.value.data == 0
    assert len(op.inputs) == 2


def test_concat_rejects_empty_inputs():
    op = ConcatOp([], dim=0, result_type=_ft([0, 4]))
    with pytest.raises(VerifyException, match="at least one input"):
        op.verify()


def test_concat_rejects_out_of_range_dim():
    a, b = _value([2, 4]), _value([3, 4])
    op = ConcatOp([a, b], dim=5, result_type=_ft([5, 4]))
    with pytest.raises(VerifyException, match="out of range"):
        op.verify()


def test_concat_rejects_mismatched_non_concat_dim():
    a, b = _value([2, 4]), _value([3, 8])
    op = ConcatOp([a, b], dim=0, result_type=_ft([5, 4]))
    with pytest.raises(VerifyException, match="mismatched extent"):
        op.verify()


def test_concat_rejects_mismatched_rank():
    a, b = _value([2, 4]), _value([3, 4, 1])
    op = ConcatOp([a, b], dim=0, result_type=_ft([5, 4]))
    with pytest.raises(VerifyException, match="must have the same rank"):
        op.verify()


def test_concat_three_way_middle_dim():
    a, b, c = _value([2, 3, 5]), _value([2, 4, 5]), _value([2, 1, 5])
    op = ConcatOp([a, b, c], dim=1, result_type=_ft([2, 8, 5]))
    op.verify()


# --- PackOp -------------------------------------------------------------------


def test_pack_basic_two_tile():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[1, 0],
        inner_tiles=[32, 16],
        result_type=_ft([8, 4, 32, 16]),
    )
    op.verify()


def test_pack_rejects_duplicate_inner_dims_pos():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[0, 0],
        inner_tiles=[16, 32],
        result_type=_ft([8, 8, 16, 32]),
    )
    with pytest.raises(VerifyException, match="must be unique"):
        op.verify()


def test_pack_rejects_mismatched_len():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16, 32],
        result_type=_ft([8, 128, 16, 32]),
    )
    with pytest.raises(VerifyException, match="same length"):
        op.verify()


def test_pack_rejects_non_positive_tile():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[0],
        result_type=_ft([0, 128, 0]),
    )
    with pytest.raises(VerifyException, match="strictly positive"):
        op.verify()


def test_pack_rejects_out_of_range_pos():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[5],
        inner_tiles=[16],
        result_type=_ft([128, 128, 16]),
    )
    with pytest.raises(VerifyException, match="out of range"):
        op.verify()


def test_pack_rejects_bad_rank():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        result_type=_ft([128, 128]),  # wrong rank
    )
    with pytest.raises(VerifyException, match="result rank"):
        op.verify()


def test_pack_outer_dims_perm_permutation_check():
    src = _value([128, 128])
    op = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        outer_dims_perm=[1, 0],
        result_type=_ft([128, 8, 16]),
    )
    op.verify()

    bad = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        outer_dims_perm=[0, 0],
        result_type=_ft([128, 8, 16]),
    )
    with pytest.raises(VerifyException, match="permutation"):
        bad.verify()


def test_pack_accepts_padding_value_operand():
    src = _value([120, 128])
    pad = _value([])  # scalar
    op = PackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        result_type=_ft([8, 128, 16]),
        padding_value=pad,
    )
    op.verify()
    assert op.padding_value is not None


# --- UnpackOp -----------------------------------------------------------------


def test_unpack_basic():
    src = _value([8, 4, 32, 16])
    op = UnpackOp(
        src,
        inner_dims_pos=[1, 0],
        inner_tiles=[32, 16],
        result_type=_ft([128, 128]),
    )
    op.verify()


def test_unpack_rejects_mismatched_len():
    src = _value([8, 4, 32, 16])
    op = UnpackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[32, 16],
        result_type=_ft([128, 128]),
    )
    with pytest.raises(VerifyException, match="same length"):
        op.verify()


def test_unpack_rejects_bad_rank():
    src = _value([128, 128])
    op = UnpackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        result_type=_ft([128, 128]),  # wrong rank relation
    )
    with pytest.raises(VerifyException, match="source rank"):
        op.verify()


def test_unpack_outer_dims_perm_permutation_check():
    src = _value([4, 8, 16])
    op = UnpackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        outer_dims_perm=[1, 0],
        result_type=_ft([128, 8]),
    )
    op.verify()

    bad = UnpackOp(
        src,
        inner_dims_pos=[0],
        inner_tiles=[16],
        outer_dims_perm=[1, 1],
        result_type=_ft([128, 8]),
    )
    with pytest.raises(VerifyException, match="permutation"):
        bad.verify()


# --- Pure trait ---------------------------------------------------------------


@pytest.mark.parametrize("op_cls", [ConcatOp, PackOp, UnpackOp])
def test_ops_are_pure(op_cls):
    from xdsl.traits import Pure

    assert any(isinstance(t, Pure) for t in op_cls.traits.traits)
