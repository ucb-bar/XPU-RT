"""Tests for W5.1 ``lower_conv_to_img2col``."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.ir.payload.passes.rewrites.lower_conv_to_img2col import (
    LowerConvToImg2ColConfig,
    LowerConvToImg2ColStats,
    run_lower_conv_to_img2col,
)
from tests.ir.payload.passes._pattern_test_helpers import assert_module_verifies


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _conv_module(
    input_shape=(1, 3, 8, 8),
    filter_shape=(16, 3, 3, 3),
    output_shape=(1, 16, 8, 8),
) -> tuple[ModuleOp, CallOp]:
    it = _ft(input_shape)
    ft = _ft(filter_shape)
    ot = _ft(output_shape)
    x = EmptyOp([], it)
    w = EmptyOp([], ft)
    ext = FuncOp.external("aten_convolution", [it, ft], [ot])
    call = CallOp("aten_convolution", [x.results[0], w.results[0]], [ot])
    call.attributes["compgen._pattern_hint"] = StringAttr("convolution")
    block = Block()
    for op in (x, w, call):
        block.add_op(op)
    block.add_op(ReturnOp(call.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [ot]), Region([block]))
    return ModuleOp([ext, func]), call


# --- happy path ------------------------------------------------------------


def test_static_conv_gets_scheduled():
    m, call = _conv_module()
    stats = run_lower_conv_to_img2col(m)
    assert stats.convs_scheduled == 1
    assert call.attributes["compgen.img2col_scheduled"].data == "true"
    assert_module_verifies(m)


def test_shape_metadata_is_recorded():
    m, call = _conv_module(
        input_shape=(2, 3, 16, 16), filter_shape=(32, 3, 5, 5),
        output_shape=(2, 32, 12, 12),
    )
    run_lower_conv_to_img2col(m)
    assert call.attributes["compgen.img2col_input_shape"].data == "2,3,16,16"
    assert call.attributes["compgen.img2col_filter_shape"].data == "32,3,5,5"
    assert call.attributes["compgen.img2col_output_shape"].data == "2,32,12,12"


# --- gates ----------------------------------------------------------------


def test_dynamic_input_shape_is_skipped():
    m, call = _conv_module(input_shape=(1, 3, -1, 8))
    stats = run_lower_conv_to_img2col(m)
    assert stats.convs_scheduled == 0
    assert stats.convs_skipped_dynamic == 1


def test_dynamic_can_be_allowed_via_config():
    m, call = _conv_module(input_shape=(1, 3, -1, 8))
    cfg = LowerConvToImg2ColConfig(require_static_shapes=False)
    stats = run_lower_conv_to_img2col(m, config=cfg)
    assert stats.convs_scheduled == 1


def test_wrong_rank_is_skipped():
    # 3D conv (rank-3 input) -> skip.
    t_in = _ft([3, 8, 8])
    t_w = _ft([16, 3, 3])
    t_out = _ft([16, 8, 8])
    x = EmptyOp([], t_in); w = EmptyOp([], t_w)
    ext = FuncOp.external("aten_convolution", [t_in, t_w], [t_out])
    call = CallOp("aten_convolution", [x.results[0], w.results[0]], [t_out])
    call.attributes["compgen._pattern_hint"] = StringAttr("convolution")
    block = Block()
    for op in (x, w, call):
        block.add_op(op)
    block.add_op(ReturnOp(call.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t_out]), Region([block]))
    m = ModuleOp([ext, func])

    stats = run_lower_conv_to_img2col(m)
    assert stats.convs_skipped_wrong_rank == 1


def test_too_small_output_is_skipped():
    m, _ = _conv_module(
        input_shape=(1, 1, 2, 2), filter_shape=(1, 1, 1, 1),
        output_shape=(1, 1, 2, 2),
    )
    cfg = LowerConvToImg2ColConfig(min_output_elements=64)
    stats = run_lower_conv_to_img2col(m, config=cfg)
    assert stats.convs_skipped_too_small == 1


# --- non-matching cases ---------------------------------------------------


def test_non_convolution_call_is_ignored():
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

    stats = run_lower_conv_to_img2col(m)
    assert stats.convs_seen == 0
    assert stats.convs_scheduled == 0


# --- idempotence + stats --------------------------------------------------


def test_idempotent_second_run_is_noop():
    m, _ = _conv_module()
    first = run_lower_conv_to_img2col(m)
    assert first.convs_scheduled == 1
    second = run_lower_conv_to_img2col(m)
    assert second.convs_scheduled == 0


def test_stats_initial_values():
    s = LowerConvToImg2ColStats()
    assert s.convs_seen == 0
    assert s.convs_scheduled == 0


# --- real-workload (no convs -> no-op) ------------------------------------


def test_emits_real_pack_op_when_tiles_divide_evenly():
    """Real structural emission: compgen.tensor_ext.pack is emitted
    when H/KH and W/KW divide evenly."""
    # Input 1,3,8,8 with kernel 2,2 → H=W=8 divisible by KH=KW=2.
    m, _ = _conv_module(
        input_shape=(1, 3, 8, 8),
        filter_shape=(16, 3, 2, 2),
        output_shape=(1, 16, 4, 4),
    )
    stats = run_lower_conv_to_img2col(m)
    assert stats.pack_ops_emitted == 1
    packs = [op for op in m.walk() if op.name == "compgen.tensor_ext.pack"]
    assert len(packs) == 1
    result_shape = list(packs[0].result.type.get_shape())
    # Original [1, 3, 8, 8] → [1, 3, 4, 4, 2, 2] (spatial dims tiled).
    assert result_shape == [1, 3, 4, 4, 2, 2]


def test_does_not_emit_pack_when_tiles_dont_divide():
    # 8 % 3 != 0 → no pack emitted; conv still tagged.
    m, conv = _conv_module(
        input_shape=(1, 3, 8, 8),
        filter_shape=(16, 3, 3, 3),
        output_shape=(1, 16, 6, 6),
    )
    stats = run_lower_conv_to_img2col(m)
    assert stats.convs_scheduled == 1
    assert stats.pack_ops_emitted == 0
    assert "compgen.img2col_scheduled" in conv.attributes


def test_pack_tile_sizes_recorded_on_conv():
    m, conv = _conv_module(
        input_shape=(1, 3, 8, 8),
        filter_shape=(16, 3, 2, 2),
        output_shape=(1, 16, 4, 4),
    )
    run_lower_conv_to_img2col(m)
    assert conv.attributes["compgen.img2col_pack_tile_kh"].value.data == 2
    assert conv.attributes["compgen.img2col_pack_tile_kw"].value.data == 2


def test_noop_on_attention_mlp_tiny():
    """attention_mlp_tiny has no conv ops -> pass is a no-op."""
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None
    stats = run_lower_conv_to_img2col(result.module)
    assert stats.convs_seen == 0
    assert_module_verifies(result.module)
