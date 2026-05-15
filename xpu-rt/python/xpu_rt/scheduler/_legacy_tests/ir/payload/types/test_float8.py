"""Tests for W0.1 FP8 types (``Float8E4M3FNType`` + ``Float8E5M2Type``)."""

from __future__ import annotations

import pytest
from compgen.ir.payload.types import Float8E4M3FNType, Float8E5M2Type
from xdsl.dialects.builtin import Float16Type, TensorType

# --- bitwidth + size ----------------------------------------------------------


@pytest.mark.parametrize("cls", [Float8E4M3FNType, Float8E5M2Type])
def test_bitwidth_is_eight(cls):
    assert cls().bitwidth == 8


@pytest.mark.parametrize("cls", [Float8E4M3FNType, Float8E5M2Type])
def test_compile_time_and_runtime_size_is_one_byte(cls):
    t = cls()
    assert t.compile_time_size == 1
    assert t.size == 1


# --- format semantics ---------------------------------------------------------


def test_e4m3fn_semantics_match_mlir():
    t = Float8E4M3FNType()
    assert t.exponent_bits == 4
    assert t.mantissa_bits == 3
    assert t.exponent_bias == 7
    assert t.has_infinity is False
    assert t.has_nan is True
    assert t.max_finite == 448.0


def test_e5m2_semantics_match_mlir():
    t = Float8E5M2Type()
    assert t.exponent_bits == 5
    assert t.mantissa_bits == 2
    assert t.exponent_bias == 15
    assert t.has_infinity is True
    assert t.has_nan is True
    assert t.max_finite == 57344.0


# --- equality / identity ------------------------------------------------------


@pytest.mark.parametrize("cls", [Float8E4M3FNType, Float8E5M2Type])
def test_parameterless_instances_compare_equal(cls):
    assert cls() == cls()


def test_e4m3fn_and_e5m2_are_distinct():
    assert Float8E4M3FNType() != Float8E5M2Type()


def test_e4m3fn_is_not_float16():
    assert Float8E4M3FNType() != Float16Type()


# --- tensor-legality ----------------------------------------------------------


@pytest.mark.parametrize("cls", [Float8E4M3FNType, Float8E5M2Type])
def test_tensor_accepts_fp8_element_type(cls):
    tt = TensorType(cls(), [16, 16])
    assert tt.get_element_type() == cls()
    assert list(tt.get_shape()) == [16, 16]


def test_tensor_with_fp8_prints_readably():
    tt = TensorType(Float8E4M3FNType(), [2, 3])
    assert "float8_e4m3fn" in str(tt)


# --- integration: import_fx respects the new types ---------------------------


def test_import_fx_maps_float8_e4m3fn_to_new_type():
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build lacks float8_e4m3fn")

    from compgen.ir.payload.import_fx import _torch_dtype_to_xdsl

    out = _torch_dtype_to_xdsl(torch.float8_e4m3fn)
    assert isinstance(out, Float8E4M3FNType)


def test_import_fx_maps_float8_e5m2_to_new_type():
    import torch

    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("torch build lacks float8_e5m2")

    from compgen.ir.payload.import_fx import _torch_dtype_to_xdsl

    out = _torch_dtype_to_xdsl(torch.float8_e5m2)
    assert isinstance(out, Float8E5M2Type)
