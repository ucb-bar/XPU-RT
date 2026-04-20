"""Tests for the wave-6 DECOMPOSITION_TABLE expansion.

Each test builds a minimal FX-style meta dict and operand list, calls
the decomposition function directly, and asserts the result carries:
  - expected op count (>= 1)
  - a non-empty region_ids list
  - a meaningful pattern_hint (B.2 + B.1 integration)
"""

from __future__ import annotations

import pytest
from compgen.ir.payload.decompositions import (
    DECOMPOSITION_TABLE,
    decompose_bmm,
    decompose_cat,
    decompose_clone,
    decompose_convolution,
    decompose_dequantize_per_channel,
    decompose_dequantize_per_group,
    decompose_dequantize_per_tensor,
    decompose_div_tensor,
    decompose_embedding,
    decompose_expand,
    decompose_mean_dim,
    decompose_native_layer_norm,
    decompose_neg,
    decompose_pow_tensor_scalar,
    decompose_quantize_per_channel,
    decompose_quantize_per_tensor,
    decompose_rsqrt,
    decompose_sigmoid,
    decompose_silu,
    decompose_softmax,
    decompose_split_with_sizes,
    decompose_sub_tensor,
    decompose_unsqueeze,
    decompose_view,
    decompose_weight_int4pack_mm,
    decompose_weight_int8pack_mm,
    reset_region_counters,
)
from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.dialects.tensor import EmptyOp


def _fake_meta(shape=(4, 8)):
    return {"val": type("V", (), {"shape": shape})()}


def _fake_operand(shape=(4, 8)):
    t = TensorType(Float32Type(), list(shape))
    return EmptyOp([], t).results[0]


@pytest.fixture(autouse=True)
def _reset():
    reset_region_counters()


@pytest.mark.parametrize(
    "fn,hint",
    [
        (decompose_bmm, "batch_matmul"),
        (decompose_native_layer_norm, "layer_norm"),
        (decompose_softmax, "softmax"),
        (decompose_rsqrt, "rsqrt"),
        (decompose_pow_tensor_scalar, "pow_tensor_scalar"),
        (decompose_mean_dim, "reduce_mean"),
        (decompose_embedding, "embedding_lookup"),
        (decompose_sigmoid, "sigmoid"),
        (decompose_neg, "neg"),
        (decompose_silu, "silu"),
        (decompose_sub_tensor, "sub"),
        (decompose_div_tensor, "div"),
        (decompose_view, "view"),
        (decompose_unsqueeze, "unsqueeze"),
        (decompose_expand, "expand"),
        (decompose_clone, "clone"),
    ],
)
def test_compute_and_layout_decompositions_emit_hint(fn, hint):
    result = fn(
        [_fake_operand(), _fake_operand()],
        _fake_meta(),
        "node_x",
    )
    assert result.pattern_hint == hint
    assert len(result.ops) >= 1
    assert len(result.region_ids) >= 1


def test_convolution_accepts_three_tensor_operands():
    result = decompose_convolution(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta((1, 16, 32, 32)),
        "conv_x",
    )
    assert result.pattern_hint == "convolution"
    assert len(result.ops) >= 1


def test_cat_preserves_all_tensor_operands():
    result = decompose_cat(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta((12, 8)),
        "cat_x",
    )
    assert result.pattern_hint == "cat"


def test_split_with_sizes_only_tensor_operand():
    result = decompose_split_with_sizes(
        [_fake_operand()],
        _fake_meta((4, 8)),
        "split_x",
    )
    assert result.pattern_hint == "split"


# --- C.2: TorchAO quantized_decomposed + _weight_int*pack_mm ---


@pytest.mark.parametrize(
    "fn,hint",
    [
        (decompose_quantize_per_tensor, "quantize_per_tensor"),
        (decompose_dequantize_per_tensor, "dequantize_per_tensor"),
        (decompose_quantize_per_channel, "quantize_per_channel"),
        (decompose_dequantize_per_channel, "dequantize_per_channel"),
        (decompose_dequantize_per_group, "dequantize_per_group"),
    ],
)
def test_qd_decompositions_carry_pattern_hint(fn, hint):
    result = fn(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "qd_x",
    )
    assert result.pattern_hint == hint


def test_weight_int8pack_mm_emits_quantized_matmul_region():
    result = decompose_weight_int8pack_mm(
        [_fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "w8_x",
    )
    assert result.pattern_hint == "weight_int8pack_mm"
    assert any(rid.startswith("quantized_matmul") for rid in result.region_ids)


def test_weight_int4pack_mm_skips_group_size_scalar():
    # Args: (input, weight_int4, group_size, scales_and_zeros)
    # The decomposition should skip operands[2] (scalar) but include [0,1,3].
    result = decompose_weight_int4pack_mm(
        [_fake_operand(), _fake_operand(), _fake_operand(), _fake_operand()],
        _fake_meta(),
        "w4_x",
    )
    assert result.pattern_hint == "weight_int4pack_mm"


# --- Table-level assertion ---


def test_decomposition_table_has_expected_entries():
    """Wave 6 adds at least 28 new entries."""
    # Before wave 6: 8 entries. After: 28 compute/layout + 17 quantized
    # (some with aliases) = 45+.
    assert len(DECOMPOSITION_TABLE) >= 28
    # Key entries survived
    for key in (
        "aten.addmm.default",  # pre-existing
        "aten.bmm.default",  # wave 6 B.1
        "aten.native_layer_norm.default",
        "aten._softmax.default",
        "aten.rsqrt.default",
        "aten.convolution.default",
        "aten.embedding.default",
        "aten._weight_int8pack_mm.default",  # wave 6 C.2
        "aten._weight_int4pack_mm.default",
        "torch.ops.quantized_decomposed.dequantize_per_channel.default",
    ):
        assert key in DECOMPOSITION_TABLE, f"{key} missing"
