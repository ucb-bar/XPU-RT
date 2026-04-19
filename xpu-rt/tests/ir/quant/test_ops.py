"""Tests for ``compgen.quant`` operations."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    Float32Type,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.utils.exceptions import VerifyException

from compgen.ir.quant import (
    AffineQuantizedTensorType,
    ChooseQParamsPerChannelOp,
    ChooseQParamsPerTensorOp,
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
    FakeQuantOp,
    QuantizePerChannelOp,
    QuantizePerGroupOp,
    QuantizePerTensorOp,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)


# --- shared fixtures ----------------------------------------------------------


def _f32(shape):
    return TensorType(Float32Type(), list(shape))


def _i8(shape):
    return TensorType(IntegerType(8), list(shape))


def _i32(shape):
    return TensorType(IntegerType(32), list(shape))


@pytest.fixture
def scalar_f32():
    return EmptyOp([], _f32([])).results[0]


@pytest.fixture
def scalar_i32():
    return EmptyOp([], _i32([])).results[0]


@pytest.fixture
def x_f32():
    return EmptyOp([], _f32([4, 8])).results[0]


@pytest.fixture
def x_i8():
    return EmptyOp([], _i8([4, 8])).results[0]


def _ia(v, width=64):
    return IntegerAttr(v, IntegerType(width))


# --- QuantizePerTensor --------------------------------------------------------


def test_quantize_per_tensor_builds_and_verifies(x_f32, scalar_f32, scalar_i32):
    op = QuantizePerTensorOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_i8([4, 8])],
        properties={
            "quant_min": _ia(-128),
            "quant_max": _ia(127),
            "output_dtype": StringAttr("int8"),
        },
    )
    op.verify()
    assert op.result.type == _i8([4, 8])


def test_quantize_per_tensor_rejects_inverted_range(x_f32, scalar_f32, scalar_i32):
    op = QuantizePerTensorOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_i8([4, 8])],
        properties={"quant_min": _ia(100), "quant_max": _ia(-10)},
    )
    with pytest.raises(VerifyException, match="quant_min"):
        op.verify()


def test_quantize_per_tensor_accepts_affine_qtype_prop(x_f32, scalar_f32, scalar_i32):
    qtype = AffineQuantizedTensorType(IntegerType(8), Float32Type())
    op = QuantizePerTensorOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_i8([4, 8])],
        properties={"qtype": qtype},
    )
    op.verify()
    assert op.qtype == qtype


# --- DequantizePerTensor ------------------------------------------------------


def test_dequantize_per_tensor_builds_and_verifies(x_i8, scalar_f32, scalar_i32):
    op = DequantizePerTensorOp(
        operands=[x_i8, scalar_f32, scalar_i32],
        result_types=[_f32([4, 8])],
    )
    op.verify()


# --- Per-channel --------------------------------------------------------------


def test_quantize_per_channel_requires_axis(x_f32):
    scales = EmptyOp([], _f32([8])).results[0]
    zps = EmptyOp([], _i32([8])).results[0]
    op = QuantizePerChannelOp(
        operands=[x_f32, scales, zps],
        result_types=[_i8([4, 8])],
        properties={"axis": _ia(1)},
    )
    op.verify()
    assert op.axis.value.data == 1


def test_dequantize_per_channel_verifies(x_i8):
    scales = EmptyOp([], _f32([8])).results[0]
    zps = EmptyOp([], _i32([8])).results[0]
    op = DequantizePerChannelOp(
        operands=[x_i8, scales, zps],
        result_types=[_f32([4, 8])],
        properties={"axis": _ia(1)},
    )
    op.verify()


# --- Per-group ----------------------------------------------------------------


def test_quantize_per_group_requires_positive_group_size(x_f32, scalar_f32, scalar_i32):
    op = QuantizePerGroupOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_i8([4, 8])],
        properties={"group_size": _ia(64)},
    )
    op.verify()


def test_quantize_per_group_rejects_zero_group_size(x_f32, scalar_f32, scalar_i32):
    op = QuantizePerGroupOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_i8([4, 8])],
        properties={"group_size": _ia(0)},
    )
    with pytest.raises(VerifyException, match="group_size must be positive"):
        op.verify()


def test_dequantize_per_group_verifies(x_i8, scalar_f32, scalar_i32):
    op = DequantizePerGroupOp(
        operands=[x_i8, scalar_f32, scalar_i32],
        result_types=[_f32([4, 8])],
        properties={"group_size": _ia(128)},
    )
    op.verify()


# --- Packed GEMMs -------------------------------------------------------------


def test_weight_int8pack_mm(x_f32):
    w = EmptyOp([], _i8([16, 8])).results[0]
    scales = EmptyOp([], _f32([16])).results[0]
    op = WeightInt8PackMMOp(
        operands=[x_f32, w, scales],
        result_types=[_f32([4, 16])],
    )
    op.verify()


@pytest.mark.parametrize("gs", [32, 64, 128, 256])
def test_weight_int4pack_mm_valid_group_sizes(x_f32, gs):
    w = EmptyOp([], _i8([16, 4])).results[0]
    sz = EmptyOp([], _f32([16, 2])).results[0]
    op = WeightInt4PackMMOp(
        operands=[x_f32, w, sz],
        result_types=[_f32([4, 16])],
        properties={"group_size": _ia(gs)},
    )
    op.verify()


def test_weight_int4pack_mm_rejects_bad_group_size(x_f32):
    w = EmptyOp([], _i8([16, 4])).results[0]
    sz = EmptyOp([], _f32([16, 2])).results[0]
    op = WeightInt4PackMMOp(
        operands=[x_f32, w, sz],
        result_types=[_f32([4, 16])],
        properties={"group_size": _ia(17)},
    )
    with pytest.raises(VerifyException, match="group_size must be one of"):
        op.verify()


def test_weight_int4pack_qm_verifies(x_f32):
    w = EmptyOp([], _i8([2, 16, 4])).results[0]
    sz = EmptyOp([], _f32([2, 16, 2])).results[0]
    op = WeightInt4PackQMOp(
        operands=[x_f32, w, sz],
        result_types=[_f32([2, 4, 16])],
        properties={"group_size": _ia(128)},
    )
    op.verify()


# --- choose_qparams -----------------------------------------------------------


def test_choose_qparams_per_tensor_yields_two_results(x_f32):
    op = ChooseQParamsPerTensorOp(
        operands=[x_f32],
        result_types=[_f32([]), TensorType(IntegerType(64), [])],
    )
    op.verify()
    assert len(op.results) == 2


def test_choose_qparams_per_channel_requires_axis(x_f32):
    op = ChooseQParamsPerChannelOp(
        operands=[x_f32],
        result_types=[_f32([8]), TensorType(IntegerType(64), [8])],
        properties={"axis": _ia(1)},
    )
    op.verify()
    assert op.axis.value.data == 1


# --- FakeQuant ----------------------------------------------------------------


def test_fake_quantize_builds(x_f32, scalar_f32, scalar_i32):
    op = FakeQuantOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_f32([4, 8])],
        properties={
            "quant_min": _ia(-128),
            "quant_max": _ia(127),
            "granularity": StringAttr("per_tensor"),
        },
    )
    op.verify()


def test_fake_quantize_rejects_inverted_range(x_f32, scalar_f32, scalar_i32):
    op = FakeQuantOp(
        operands=[x_f32, scalar_f32, scalar_i32],
        result_types=[_f32([4, 8])],
        properties={"quant_min": _ia(10), "quant_max": _ia(-10)},
    )
    with pytest.raises(VerifyException, match="quant_min"):
        op.verify()


# --- purity trait -------------------------------------------------------------


@pytest.mark.parametrize(
    "op_cls",
    [
        QuantizePerTensorOp,
        DequantizePerTensorOp,
        QuantizePerChannelOp,
        DequantizePerChannelOp,
        QuantizePerGroupOp,
        DequantizePerGroupOp,
        WeightInt8PackMMOp,
        WeightInt4PackMMOp,
        WeightInt4PackQMOp,
        ChooseQParamsPerTensorOp,
        ChooseQParamsPerChannelOp,
        FakeQuantOp,
    ],
)
def test_ops_are_pure(op_cls):
    from xdsl.traits import Pure
    trait_classes = {type(t) for t in op_cls.traits.traits}
    assert Pure in trait_classes, f"{op_cls.__name__} missing Pure trait"
