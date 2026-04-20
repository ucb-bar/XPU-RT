"""Tests for the ``compgen.quant`` dialect registration + types."""

from __future__ import annotations

import pytest
from compgen.ir.payload.types import Float8E4M3FNType
from compgen.ir.quant import (
    ALL_ATTRS,
    ALL_OPS,
    AffineQuantizedTensorType,
    MXQuantizedTensorType,
    NVFP4TensorType,
    PackedIntTensorType,
    Quant,
)
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float32Type,
    IntegerType,
)
from xdsl.ir import Dialect

# --- registration -------------------------------------------------------------


def test_quant_is_dialect_instance():
    assert isinstance(Quant, Dialect)
    assert Quant.name == "compgen.quant"


def test_all_ops_registered_with_dialect():
    dialect_op_names = {op.name for op in Quant._operations}
    exported_names = {op.name for op in ALL_OPS}
    assert dialect_op_names == exported_names


def test_all_attrs_registered_with_dialect():
    dialect_attr_names = {attr.name for attr in Quant._attributes}
    exported_names = {attr.name for attr in ALL_ATTRS}
    assert dialect_attr_names == exported_names


def test_dialect_has_at_least_twelve_ops():
    # 6 quant/dequant + 3 packed GEMMs + 2 choose_qparams + 1 fake_quant
    assert len(ALL_OPS) >= 12


def test_dialect_has_four_attr_types():
    assert len(ALL_ATTRS) == 4


# --- AffineQuantizedTensorType ------------------------------------------------


def test_affine_tensor_per_tensor_default():
    a = AffineQuantizedTensorType(IntegerType(8), Float32Type())
    assert a.storage_type == IntegerType(8)
    assert a.scale_dtype == Float32Type()
    assert a.granularity.data == "per_tensor"
    assert a.layout.data == "plain"
    assert len(a.block_size.data) == 0


def test_affine_tensor_per_group_carries_block_size():
    a = AffineQuantizedTensorType(
        IntegerType(4),
        BFloat16Type(),
        zero_point_dtype=IntegerType(32),
        granularity="per_group",
        block_size=[1, 32],
        layout="tensor_core_tiled",
    )
    assert a.granularity.data == "per_group"
    assert a.layout.data == "tensor_core_tiled"
    assert [d.value.data for d in a.block_size.data] == [1, 32]


def test_affine_tensor_fp8_storage():
    a = AffineQuantizedTensorType(Float8E4M3FNType(), BFloat16Type(), granularity="per_token")
    assert a.storage_type == Float8E4M3FNType()
    assert a.granularity.data == "per_token"


def test_affine_tensor_rejects_bad_granularity():
    with pytest.raises(ValueError, match="Invalid granularity"):
        AffineQuantizedTensorType(IntegerType(8), Float32Type(), granularity="per_everything")


def test_affine_tensor_rejects_bad_layout():
    with pytest.raises(ValueError, match="Invalid layout"):
        AffineQuantizedTensorType(IntegerType(8), Float32Type(), layout="made_up")


# --- PackedIntTensorType ------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_packed_int_accepts_valid_bitwidths(bits):
    p = PackedIntTensorType(bits, pack_dim=0)
    assert p.bit_width.value.data == bits
    assert p.pack_dim.value.data == 0


def test_packed_int_rejects_invalid_bitwidth():
    with pytest.raises(ValueError, match="bit_width must be one of"):
        PackedIntTensorType(5, pack_dim=0)


def test_packed_int_default_storage_is_int8():
    p = PackedIntTensorType(4, pack_dim=1)
    assert p.storage_type == IntegerType(8)


# --- MXQuantizedTensorType ----------------------------------------------------


@pytest.mark.parametrize("bits", [4, 6, 8, 9])
def test_mx_accepts_valid_element_widths(bits):
    m = MXQuantizedTensorType(bits)
    assert m.element_bit_width.value.data == bits
    assert m.block_size.value.data == 32
    assert m.scale_bit_width.value.data == 8
    assert m.scale_kind.data == "e8m0"


def test_mx_rejects_invalid_element_bitwidth():
    with pytest.raises(ValueError, match="element_bit_width must be one of"):
        MXQuantizedTensorType(5)


def test_mx_rejects_unknown_scale_kind():
    with pytest.raises(ValueError, match="scale_kind must be one of"):
        MXQuantizedTensorType(4, scale_kind="fp16")


# --- NVFP4TensorType ----------------------------------------------------------


def test_nvfp4_defaults():
    n = NVFP4TensorType()
    assert n.block_size.value.data == 16
    assert n.scale_dtype == Float32Type()


def test_nvfp4_custom_block_size():
    n = NVFP4TensorType(block_size=8)
    assert n.block_size.value.data == 8
