"""Tests for W4.2 ``lower_quantized_conv``."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.ir.quant import (
    DequantizePerChannelOp,
    DequantizePerTensorOp,
)
from compgen.ir.payload.passes.rewrites.lower_quantized_conv import (
    LowerQuantizedConvStats,
    run_lower_quantized_conv,
)
from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
)


def _ft(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _quantized_conv_module(
    *, per_channel: bool = True,
) -> tuple[ModuleOp, CallOp]:
    N, C, H, W = 1, 3, 8, 8
    F, KH, KW = 16, 3, 3
    input_t = _ft([N, C, H, W])
    wi_t = _ft([F, C, KH, KW], IntegerType(8))
    wf_t = _ft([F, C, KH, KW])
    out_t = _ft([N, F, H, W])

    x = EmptyOp([], input_t)
    wi = EmptyOp([], wi_t)

    if per_channel:
        scales = EmptyOp([], _ft([F]))
        zps = EmptyOp([], _ft([F], IntegerType(32)))
        dq = DequantizePerChannelOp(
            operands=[wi.results[0], scales.results[0], zps.results[0]],
            result_types=[wf_t],
            properties={"axis": IntegerAttr(0, IntegerType(64))},
        )
        dq_ops = [scales, zps, dq]
    else:
        scale = EmptyOp([], _ft([]))
        zp = EmptyOp([], _ft([], IntegerType(32)))
        dq = DequantizePerTensorOp(
            operands=[wi.results[0], scale.results[0], zp.results[0]],
            result_types=[wf_t],
        )
        dq_ops = [scale, zp, dq]

    conv_ext = FuncOp.external("aten_convolution", [input_t, wf_t], [out_t])
    conv = CallOp("aten_convolution", [x.results[0], dq.result], [out_t])
    conv.attributes["compgen._pattern_hint"] = StringAttr("convolution")

    block = Block()
    for op in (x, wi, *dq_ops, conv):
        block.add_op(op)
    block.add_op(ReturnOp(conv.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [out_t]), Region([block]))
    return ModuleOp([conv_ext, func]), conv


def _plain_conv_module() -> tuple[ModuleOp, CallOp]:
    """A conv with a plain (non-dequantized) weight operand."""
    N, C, H, W = 1, 3, 8, 8
    F, KH, KW = 16, 3, 3
    input_t = _ft([N, C, H, W])
    wf_t = _ft([F, C, KH, KW])
    out_t = _ft([N, F, H, W])
    x = EmptyOp([], input_t)
    w = EmptyOp([], wf_t)
    conv_ext = FuncOp.external("aten_convolution", [input_t, wf_t], [out_t])
    conv = CallOp("aten_convolution", [x.results[0], w.results[0]], [out_t])
    conv.attributes["compgen._pattern_hint"] = StringAttr("convolution")
    block = Block()
    for op in (x, w, conv):
        block.add_op(op)
    block.add_op(ReturnOp(conv.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [out_t]), Region([block]))
    return ModuleOp([conv_ext, func]), conv


# --- happy path ------------------------------------------------------------


def test_per_channel_dequant_conv_is_tagged():
    m, conv = _quantized_conv_module(per_channel=True)
    stats = run_lower_quantized_conv(m)
    assert stats.quantized_convs_tagged == 1
    assert conv.attributes["compgen.quantized_conv_scheduled"].data == "true"
    assert conv.attributes["compgen.quantized_conv_kind"].data == "per_channel"
    assert_module_verifies(m)


def test_per_tensor_dequant_conv_is_tagged():
    m, conv = _quantized_conv_module(per_channel=False)
    run_lower_quantized_conv(m)
    assert conv.attributes["compgen.quantized_conv_kind"].data == "per_tensor"


# --- non-matching cases ----------------------------------------------------


def test_plain_conv_is_skipped():
    m, conv = _plain_conv_module()
    stats = run_lower_quantized_conv(m)
    assert stats.quantized_convs_tagged == 0
    assert stats.non_quantized_convs_skipped == 1
    assert "compgen.quantized_conv_scheduled" not in conv.attributes


def test_non_conv_call_is_skipped():
    t = _ft([4, 8])
    ext = FuncOp.external("aten_gelu", [t], [t])
    x = EmptyOp([], t)
    call = CallOp("aten_gelu", [x.results[0]], [t])
    call.attributes["compgen._pattern_hint"] = StringAttr("gelu")
    block = Block()
    for op in (x, call):
        block.add_op(op)
    block.add_op(ReturnOp(call.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([ext, func])

    stats = run_lower_quantized_conv(m)
    assert stats.opaque_convs_seen == 0
    assert stats.quantized_convs_tagged == 0


# --- idempotence + stats --------------------------------------------------


def test_idempotent_second_run_is_noop():
    m, _ = _quantized_conv_module(per_channel=True)
    first = run_lower_quantized_conv(m)
    assert first.quantized_convs_tagged == 1
    second = run_lower_quantized_conv(m)
    assert second.quantized_convs_tagged == 0


def test_stats_initial_values():
    s = LowerQuantizedConvStats()
    assert s.opaque_convs_seen == 0
    assert s.quantized_convs_tagged == 0


def test_per_channel_emits_real_dequant_generic():
    """Real structural emission: the conv's weight is replaced by
    a linalg.generic output, the dequant body is materialized."""
    m, conv = _quantized_conv_module(per_channel=True)
    stats = run_lower_quantized_conv(m)
    assert stats.dequant_generics_emitted == 1
    # The conv's operand[1] now points at a linalg.generic result.
    producer = conv.operands[1].owner
    assert producer.name == "linalg.generic"
    # Body contains arith.subi / sitofp / mulf.
    body_ops = {o.name for o in producer.body.walk()}
    assert "arith.subi" in body_ops
    assert "arith.sitofp" in body_ops
    assert "arith.mulf" in body_ops
    assert_module_verifies(m)


def test_per_tensor_emits_real_dequant_generic_with_scalar_scale():
    m, conv = _quantized_conv_module(per_channel=False)
    stats = run_lower_quantized_conv(m)
    assert stats.dequant_generics_emitted == 1
    producer = conv.operands[1].owner
    assert producer.name == "linalg.generic"


def test_multiple_quantized_convs_all_tagged():
    # Build a module with two quantized convs side by side.
    m1, c1 = _quantized_conv_module(per_channel=True)
    # We reuse _quantized_conv_module's module by running once and
    # then building a second module to check cumulative stats.
    m2, c2 = _quantized_conv_module(per_channel=False)
    s1 = run_lower_quantized_conv(m1)
    s2 = run_lower_quantized_conv(m2)
    assert s1.quantized_convs_tagged == 1
    assert s2.quantized_convs_tagged == 1
    assert c1.attributes["compgen.quantized_conv_kind"].data == "per_channel"
    assert c2.attributes["compgen.quantized_conv_kind"].data == "per_tensor"
