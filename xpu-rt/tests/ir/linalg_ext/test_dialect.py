"""Tests for the ``compgen.linalg_ext`` dialect."""

from __future__ import annotations

import pytest
from compgen.ir.linalg_ext import (
    ALL_OPS,
    GeluOp,
    LayerNormOp,
    LinalgExt,
    RMSNormOp,
    RoPEOp,
    SiluOp,
    SoftmaxOp,
    SwiGLUOp,
)
from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Dialect
from xdsl.utils.exceptions import VerifyException


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _value(shape):
    return EmptyOp([], _ft(shape)).results[0]


def test_dialect_registered():
    assert isinstance(LinalgExt, Dialect)
    assert LinalgExt.name == "compgen.linalg_ext"


def test_dialect_has_seven_ops():
    assert len(ALL_OPS) == 7


def test_op_names():
    names = {op.name for op in LinalgExt._operations}
    assert names == {
        "compgen.linalg_ext.softmax",
        "compgen.linalg_ext.layer_norm",
        "compgen.linalg_ext.rms_norm",
        "compgen.linalg_ext.rope",
        "compgen.linalg_ext.swiglu",
        "compgen.linalg_ext.gelu",
        "compgen.linalg_ext.silu",
    }


# --- Softmax ------------------------------------------------------------------


def test_softmax_builds_and_verifies():
    x = _value([4, 8])
    op = SoftmaxOp(x, dim=1, result_type=_ft([4, 8]))
    op.verify()
    assert op.dim.value.data == 1


def test_softmax_rejects_negative_dim():
    x = _value([4, 8])
    op = SoftmaxOp(x, dim=-1, result_type=_ft([4, 8]))
    with pytest.raises(VerifyException, match="non-negative"):
        op.verify()


# --- LayerNorm ----------------------------------------------------------------


def test_layer_norm_full_affine():
    x = _value([4, 8])
    w = _value([8])
    b = _value([8])
    op = LayerNormOp(x, _ft([4, 8]), weight=w, bias=b, eps=1e-5)
    op.verify()
    assert op.weight is not None
    assert op.bias is not None


def test_layer_norm_without_affine_operands():
    x = _value([4, 8])
    op = LayerNormOp(x, _ft([4, 8]), eps=1e-5)
    op.verify()


def test_layer_norm_weight_only():
    x = _value([4, 8])
    w = _value([8])
    op = LayerNormOp(x, _ft([4, 8]), weight=w, eps=1e-5)
    op.verify()


def test_layer_norm_rejects_zero_eps():
    x = _value([4, 8])
    op = LayerNormOp(x, _ft([4, 8]), eps=0.0)
    with pytest.raises(VerifyException, match="strictly positive"):
        op.verify()


# --- RMSNorm ------------------------------------------------------------------


def test_rms_norm_builds():
    x = _value([4, 8])
    w = _value([8])
    op = RMSNormOp(x, _ft([4, 8]), weight=w, eps=1e-6)
    op.verify()


def test_rms_norm_optional_weight():
    x = _value([4, 8])
    op = RMSNormOp(x, _ft([4, 8]), eps=1e-6)
    op.verify()


def test_rms_norm_rejects_negative_eps():
    x = _value([4, 8])
    op = RMSNormOp(x, _ft([4, 8]), eps=-1e-6)
    with pytest.raises(VerifyException, match="strictly positive"):
        op.verify()


# --- RoPE ---------------------------------------------------------------------


def test_rope_two_results():
    t = _ft([4, 8])
    q, k, cos, sin = _value([4, 8]), _value([4, 8]), _value([4, 8]), _value([4, 8])
    op = RoPEOp(q, k, cos, sin, t, t, feature_dim=1, variant="llama")
    op.verify()
    assert len(op.results) == 2


def test_rope_rejects_unknown_variant():
    t = _ft([4, 8])
    q, k, cos, sin = _value([4, 8]), _value([4, 8]), _value([4, 8]), _value([4, 8])
    op = RoPEOp(q, k, cos, sin, t, t, variant="made_up")
    with pytest.raises(VerifyException, match="variant must be one of"):
        op.verify()


def test_rope_defaults_no_variant():
    t = _ft([4, 8])
    q, k, cos, sin = _value([4, 8]), _value([4, 8]), _value([4, 8]), _value([4, 8])
    op = RoPEOp(q, k, cos, sin, t, t)
    op.verify()


# --- SwiGLU -------------------------------------------------------------------


def test_swiglu_builds():
    t = _ft([4, 8])
    gate, up = _value([4, 8]), _value([4, 8])
    op = SwiGLUOp(gate, up, t)
    op.verify()


# --- GELU ---------------------------------------------------------------------


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_valid_approximations(approximate):
    x = _value([4, 8])
    op = GeluOp(x, _ft([4, 8]), approximate=approximate)
    op.verify()
    assert op.approximate.data == approximate


def test_gelu_rejects_unknown_approximation():
    x = _value([4, 8])
    op = GeluOp(x, _ft([4, 8]), approximate="sigmoid")
    with pytest.raises(VerifyException, match="approximate must be"):
        op.verify()


# --- SiLU ---------------------------------------------------------------------


def test_silu_builds():
    x = _value([4, 8])
    op = SiluOp(x, _ft([4, 8]))
    op.verify()


# --- Pure trait ---------------------------------------------------------------


@pytest.mark.parametrize(
    "op_cls",
    [SoftmaxOp, LayerNormOp, RMSNormOp, RoPEOp, SwiGLUOp, GeluOp, SiluOp],
)
def test_ops_are_pure(op_cls):
    from xdsl.traits import Pure

    assert any(isinstance(t, Pure) for t in op_cls.traits.traits)
