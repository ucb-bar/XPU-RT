"""Wave 0.1 — quantization decompositions emit real ``compgen.quant`` ops.

These tests replace the prior opaque-``func.call`` expectations: after
W0.1, every entry in the TorchAO / quantized_decomposed family lowers
to a native dialect op that carries scalars as properties rather than
as unused kwargs.
"""

from __future__ import annotations

import pytest
from compgen.ir.payload.decompositions import (
    decompose_choose_qparams_per_channel,
    decompose_choose_qparams_per_tensor,
    decompose_dequantize_per_channel,
    decompose_dequantize_per_group,
    decompose_dequantize_per_tensor,
    decompose_quantize_per_channel,
    decompose_quantize_per_group,
    decompose_quantize_per_tensor,
    decompose_weight_int4pack_mm,
    decompose_weight_int4pack_qm,
    decompose_weight_int8pack_mm,
    reset_region_counters,
)
from compgen.ir.quant import (
    ChooseQParamsPerChannelOp,
    ChooseQParamsPerTensorOp,
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
    QuantizePerChannelOp,
    QuantizePerGroupOp,
    QuantizePerTensorOp,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)
from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.dialects.tensor import EmptyOp


@pytest.fixture(autouse=True)
def _reset():
    reset_region_counters()


def _fake_operand(shape=(4, 8)):
    t = TensorType(Float32Type(), list(shape))
    return EmptyOp([], t).results[0]


def _fake_meta(shape=(4, 8)):
    return {"val": type("V", (), {"shape": shape})()}


# --- per-tensor ---------------------------------------------------------------


def test_quantize_per_tensor_emits_real_op():
    r = decompose_quantize_per_tensor(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "q0",
    )
    assert len(r.ops) == 1
    assert isinstance(r.ops[0], QuantizePerTensorOp)
    assert r.pattern_hint == "quantize_per_tensor"
    assert r.ops[0].attributes["compgen.region_id"].data.startswith("quantize_")


def test_dequantize_per_tensor_emits_real_op():
    r = decompose_dequantize_per_tensor(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "dq0",
    )
    assert isinstance(r.ops[0], DequantizePerTensorOp)


def test_per_tensor_reads_fx_args_for_quant_range():
    # args: (input, scale, zp, quant_min, quant_max, dtype)
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", "scale", "zp", -128, 127, "torch.int8")
    r = decompose_quantize_per_tensor(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        meta,
        "q0",
    )
    op = r.ops[0]
    assert op.quant_min.value.data == -128
    assert op.quant_max.value.data == 127


# --- per-channel --------------------------------------------------------------


def test_quantize_per_channel_carries_axis():
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", "scale", "zp", 1, -128, 127, "torch.int8")
    r = decompose_quantize_per_channel(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        meta,
        "qc",
    )
    op = r.ops[0]
    assert isinstance(op, QuantizePerChannelOp)
    assert op.axis.value.data == 1


def test_dequantize_per_channel_emits_real_op():
    r = decompose_dequantize_per_channel(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "dqc",
    )
    assert isinstance(r.ops[0], DequantizePerChannelOp)


# --- per-group ----------------------------------------------------------------


def test_quantize_per_group_reads_group_size_from_fx_args():
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", "scale", "zp", 64, -128, 127, "torch.int8")
    r = decompose_quantize_per_group(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        meta,
        "qg",
    )
    op = r.ops[0]
    assert isinstance(op, QuantizePerGroupOp)
    assert op.group_size.value.data == 64


def test_quantize_per_group_defaults_to_128_when_absent():
    r = decompose_quantize_per_group(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "qg0",
    )
    assert r.ops[0].group_size.value.data == 128


def test_dequantize_per_group_emits_real_op():
    r = decompose_dequantize_per_group(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "dqg",
    )
    assert isinstance(r.ops[0], DequantizePerGroupOp)


# --- packed GEMMs -------------------------------------------------------------


def test_weight_int8pack_mm_emits_real_op():
    r = decompose_weight_int8pack_mm(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "w8",
    )
    assert isinstance(r.ops[0], WeightInt8PackMMOp)
    # region_id prefix stays ``quantized_matmul`` so downstream
    # Recipe IR passes continue to scope on it.
    assert r.ops[0].attributes["compgen.region_id"].data.startswith("quantized_matmul_")


def test_weight_int4pack_mm_snaps_group_size_to_valid_set():
    # FX arg 2 is group_size = 17; should snap to the default (128).
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", "weight", 17, "scales_and_zeros")
    r = decompose_weight_int4pack_mm(
        [_fake_operand(), _fake_operand(), _fake_operand(), _fake_operand()],
        meta,
        "w4",
    )
    op = r.ops[0]
    assert isinstance(op, WeightInt4PackMMOp)
    assert op.group_size.value.data == 128
    op.verify()


def test_weight_int4pack_mm_accepts_valid_group_size():
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", "weight", 64, "scales_and_zeros")
    r = decompose_weight_int4pack_mm(
        [_fake_operand(), _fake_operand(), _fake_operand(), _fake_operand()],
        meta,
        "w4",
    )
    assert r.ops[0].group_size.value.data == 64


def test_weight_int4pack_qm_emits_real_op():
    r = decompose_weight_int4pack_qm(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "w4q",
    )
    assert isinstance(r.ops[0], WeightInt4PackQMOp)


# --- choose_qparams -----------------------------------------------------------


def test_choose_qparams_per_tensor_returns_two_results():
    r = decompose_choose_qparams_per_tensor(
        [_fake_operand()],
        _fake_meta(),
        "cq",
    )
    op = r.ops[0]
    assert isinstance(op, ChooseQParamsPerTensorOp)
    assert len(op.results) == 2


def test_choose_qparams_per_channel_carries_axis():
    meta = _fake_meta()
    meta["_fx_args"] = ("inp", 1)
    r = decompose_choose_qparams_per_channel(
        [_fake_operand()],
        meta,
        "cqc",
    )
    op = r.ops[0]
    assert isinstance(op, ChooseQParamsPerChannelOp)
    assert op.axis.value.data == 1
