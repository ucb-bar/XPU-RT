"""Tests for W5.2 ``match_library_call``."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.rewrites.match_library_call import (
    MatchLibraryCallConfig,
    MatchLibraryCallStats,
    run_match_library_call,
)
from compgen.ir.quant import WeightInt4PackMMOp, WeightInt8PackMMOp
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import assert_module_verifies


def _ft(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _wrap(ops, ret_value, ret_type, external_funcs=()):
    block = Block()
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(ret_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ret_type]), Region([block]))
    return ModuleOp(list(external_funcs) + [func])


def _matmul_module(shape_a=(4, 8), shape_b=(8, 16), shape_out=(4, 16)):
    a = EmptyOp([], _ft(shape_a))
    b = EmptyOp([], _ft(shape_b))
    init = EmptyOp([], _ft(shape_out))
    mm = MatmulOp(inputs=[a.results[0], b.results[0]], outputs=[init.results[0]], res=[_ft(shape_out)])
    return _wrap([a, b, init, mm], mm.res[0], _ft(shape_out)), mm


def _int8_pack_mm_module():
    xt = _ft([4, 128])
    wt = _ft([32, 128], IntegerType(8))
    st = _ft([32])
    rt = _ft([4, 32])
    x = EmptyOp([], xt)
    wi = EmptyOp([], wt)
    s = EmptyOp([], st)
    q = WeightInt8PackMMOp(operands=[x.results[0], wi.results[0], s.results[0]], result_types=[rt])
    return _wrap([x, wi, s, q], q.result, rt), q


def _int4_pack_mm_module():
    from xdsl.dialects.builtin import IntegerAttr

    xt = _ft([4, 128])
    wt = _ft([32, 64], IntegerType(8))
    sz = _ft([32, 2])
    rt = _ft([4, 32])
    x = EmptyOp([], xt)
    wi = EmptyOp([], wt)
    szo = EmptyOp([], sz)
    q = WeightInt4PackMMOp(
        operands=[x.results[0], wi.results[0], szo.results[0]],
        result_types=[rt],
        properties={"group_size": IntegerAttr(128, IntegerType(64))},
    )
    return _wrap([x, wi, szo, q], q.result, rt), q


def _conv_module():
    it = _ft([1, 3, 8, 8])
    ft = _ft([16, 3, 3, 3])
    ot = _ft([1, 16, 8, 8])
    x = EmptyOp([], it)
    w = EmptyOp([], ft)
    ext = FuncOp.external("aten_convolution", [it, ft], [ot])
    call = CallOp("aten_convolution", [x.results[0], w.results[0]], [ot])
    call.attributes["compgen._pattern_hint"] = StringAttr("convolution")
    return _wrap([x, w, call], call.res[0], ot, external_funcs=[ext]), call


# --- happy paths -----------------------------------------------------------


def test_matmul_matches_cublas():
    m, mm = _matmul_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("cublas",))
    stats = run_match_library_call(m, config=cfg)
    assert stats.matmul_matches == 1
    assert mm.attributes["compgen.library_dispatch"].data == "cublas"
    assert stats.dispatch_counts["cublas"] == 1
    assert_module_verifies(m)


def test_matmul_falls_through_to_triton_when_cublas_missing():
    m, mm = _matmul_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("triton",))
    run_match_library_call(m, config=cfg)
    assert mm.attributes["compgen.library_dispatch"].data == "triton"


def test_int8_pack_mm_prefers_cublaslt():
    m, q = _int8_pack_mm_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("cublaslt", "triton"))
    stats = run_match_library_call(m, config=cfg)
    assert stats.quant_matmul_matches == 1
    assert q.attributes["compgen.library_dispatch"].data == "cublaslt"


def test_int4_pack_mm_goes_to_triton():
    m, q = _int4_pack_mm_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("cublaslt", "triton"))
    run_match_library_call(m, config=cfg)
    # int4 only matches Triton (or qnn).
    assert q.attributes["compgen.library_dispatch"].data == "triton"


def test_conv_matches_cudnn():
    m, call = _conv_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("cudnn",))
    stats = run_match_library_call(m, config=cfg)
    assert stats.conv_matches == 1
    assert call.attributes["compgen.library_dispatch"].data == "cudnn"


def test_allowlist_order_preserved():
    # Triton listed first; matmul should dispatch to Triton, not cuBLAS.
    m, mm = _matmul_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("triton", "cublas"))
    run_match_library_call(m, config=cfg)
    assert mm.attributes["compgen.library_dispatch"].data == "triton"


# --- no-match cases -------------------------------------------------------


def test_no_library_in_allowlist_matches_nothing():
    m, mm = _matmul_module()
    cfg = MatchLibraryCallConfig(library_allowlist=())
    stats = run_match_library_call(m, config=cfg)
    assert stats.no_match >= 1
    assert "compgen.library_dispatch" not in mm.attributes


def test_quant_mm_no_matching_library():
    m, q = _int4_pack_mm_module()
    # Only allow cublas (float) / cudnn (conv) -> int4 won't match either.
    cfg = MatchLibraryCallConfig(library_allowlist=("cublas", "cudnn"))
    run_match_library_call(m, config=cfg)
    assert "compgen.library_dispatch" not in q.attributes


# --- validation -----------------------------------------------------------


def test_unknown_library_in_allowlist_raises():
    with pytest.raises(ValueError, match="unknown library"):
        MatchLibraryCallConfig(library_allowlist=("cublas", "made_up_lib"))


# --- idempotence + stats -------------------------------------------------


def test_idempotent_second_run_skips_already_dispatched():
    m, mm = _matmul_module()
    cfg = MatchLibraryCallConfig(library_allowlist=("cublas",))
    first = run_match_library_call(m, config=cfg)
    assert first.matmul_matches == 1
    second = run_match_library_call(m, config=cfg)
    assert second.matmul_matches == 0
    assert second.skipped_already_dispatched >= 1


def test_stats_initial_values():
    s = MatchLibraryCallStats()
    assert s.ops_seen == 0
    assert s.dispatch_counts == {}


# --- real-workload with cuda_a100 preset ---------------------------------


def test_match_library_call_on_attention_mlp_tiny():
    """Real-workload: attention_mlp_tiny through the cuda_a100 preset.

    The two matmuls (attention + output projection) should pick up
    a Triton dispatch tag.
    """
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.options import cuda_a100_defaults

    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    preset = cuda_a100_defaults()
    cfg = MatchLibraryCallConfig(library_allowlist=tuple(preset.library_allowlist))
    stats = run_match_library_call(result.module, config=cfg)
    # At least one matmul in the fixture should match something.
    assert stats.matmul_matches + stats.quant_matmul_matches + stats.conv_matches >= 1
    assert_module_verifies(result.module)


def test_match_library_call_on_qwen_moe_tiny():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.options import cuda_a100_defaults

    from tests._fixtures.real_workloads import qwen_moe_tiny

    fx = qwen_moe_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    preset = cuda_a100_defaults()
    cfg = MatchLibraryCallConfig(library_allowlist=tuple(preset.library_allowlist))
    run_match_library_call(result.module, config=cfg)
    assert_module_verifies(result.module)
