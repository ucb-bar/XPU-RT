"""Tests for W2.2 ``fuse_softmax_to_triton``."""

from __future__ import annotations

from compgen.ir.linalg_ext import SoftmaxOp
from compgen.ir.payload.passes.rewrites.fuse_softmax_to_triton import (
    FuseSoftmaxToTritonConfig,
    FuseSoftmaxToTritonStats,
    run_fuse_softmax_to_triton,
)
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _softmax_module(shape=(4, 8), dim: int = 1) -> ModuleOp:
    t = _ft(shape)
    e = EmptyOp([], t)
    sm = SoftmaxOp(e.results[0], dim=dim, result_type=t)
    block = Block()
    for op in (e, sm):
        block.add_op(op)
    block.add_op(ReturnOp(sm.result))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    return ModuleOp([func])


def _triton_config() -> FuseSoftmaxToTritonConfig:
    return FuseSoftmaxToTritonConfig(
        kernel_family_allowlist=frozenset({"triton"}),
    )


# --- policy gating ----------------------------------------------------------


def test_default_config_is_noop():
    m = _softmax_module()
    stats = run_fuse_softmax_to_triton(m)
    assert stats.softmaxes_annotated == 0
    assert stats.softmaxes_skipped_policy == 1


def test_no_triton_in_allowlist_is_noop():
    m = _softmax_module()
    cfg = FuseSoftmaxToTritonConfig(kernel_family_allowlist=frozenset({"cublas"}))
    stats = run_fuse_softmax_to_triton(m, config=cfg)
    assert stats.softmaxes_annotated == 0


# --- happy path: annotation-only when triton-shared unavailable ------------


def test_triton_in_allowlist_annotates_softmax():
    m = _softmax_module()
    stats = run_fuse_softmax_to_triton(m, config=_triton_config())
    assert stats.softmaxes_annotated == 1
    assert count_ops(m, "compgen.linalg_ext.softmax") == 1
    assert_module_verifies(m)


def test_annotations_include_kernel_name_and_source():
    m = _softmax_module(shape=(4, 16))
    run_fuse_softmax_to_triton(m, config=_triton_config())
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    kname = sm.attributes["compgen.triton_kernel_call"].data
    assert "4x16" in kname
    source = sm.attributes["compgen.triton_source"].data
    assert "@triton.jit" in source
    assert kname in source


def test_triton_status_is_source_only_without_triton_shared_opt():
    m = _softmax_module()
    cfg = _triton_config()
    run_fuse_softmax_to_triton(m, config=cfg)
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm.attributes["compgen.triton_status"].data == "source_only"


# --- shape gates ------------------------------------------------------------


def test_higher_rank_softmax_on_last_axis_is_annotated():
    """3-D softmax over the last dim is legal (MoE router pattern).

    The pass flattens the leading dims into the kernel's row dim.
    """
    m = _softmax_module(shape=(2, 4, 8), dim=2)
    stats = run_fuse_softmax_to_triton(m, config=_triton_config())
    assert stats.softmaxes_annotated == 1
    # kernel name should reflect the flattened row count (2*4=8) x N (8).
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert "8x8" in sm.attributes["compgen.triton_kernel_call"].data


def test_softmax_not_on_last_axis_is_skipped():
    """The template only handles softmax over the last axis."""
    m = _softmax_module(shape=(4, 8), dim=0)
    stats = run_fuse_softmax_to_triton(m, config=_triton_config())
    assert stats.softmaxes_annotated == 0
    assert stats.softmaxes_skipped_rank == 1


def test_dynamic_shape_softmax_is_skipped():
    m = _softmax_module(shape=(-1, 8))
    stats = run_fuse_softmax_to_triton(m, config=_triton_config())
    assert stats.softmaxes_annotated == 0
    assert stats.softmaxes_skipped_dynamic == 1


# --- idempotence + stats ---------------------------------------------------


def test_idempotent_second_run_is_noop():
    m = _softmax_module()
    cfg = _triton_config()
    first = run_fuse_softmax_to_triton(m, config=cfg)
    assert first.softmaxes_annotated == 1
    second = run_fuse_softmax_to_triton(m, config=cfg)
    assert second.softmaxes_annotated == 0


def test_stats_initial_values():
    s = FuseSoftmaxToTritonStats()
    assert s.softmaxes_seen == 0
    assert s.softmaxes_annotated == 0


# --- attribute preservation ------------------------------------------------


def test_region_id_preserved_through_rewrite():
    m = _softmax_module()
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    sm.attributes["compgen.region_id"] = StringAttr("softmax_abc")
    run_fuse_softmax_to_triton(m, config=_triton_config())
    sm2 = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm2.attributes["compgen.region_id"].data == "softmax_abc"


# --- real-workload integration ---------------------------------------------


def test_real_workload_qwen_moe_softmax_gets_triton_annotation():
    """Real-workload test: qwen_moe_tiny has a softmax in the router.

    Pipeline: bridge FX -> raise_special_ops -> fuse_softmax_to_triton.
    The resulting module must still verify and the softmax op(s) must
    carry ``compgen.triton_kernel_call`` attributes.
    """
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.ir.payload.passes.rewrites.raise_special_ops import (
        run_raise_special_ops,
    )

    from tests._fixtures.real_workloads import qwen_moe_tiny

    fx = qwen_moe_tiny()
    bridge = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridge.module is not None

    run_raise_special_ops(bridge.module)
    stats = run_fuse_softmax_to_triton(bridge.module, config=_triton_config())
    assert stats.softmaxes_annotated >= 1

    annotated = [
        op
        for op in bridge.module.walk()
        if op.name == "compgen.linalg_ext.softmax" and "compgen.triton_kernel_call" in op.attributes
    ]
    assert len(annotated) >= 1
    assert_module_verifies(bridge.module)


def test_real_workload_attention_mlp_softmax_gets_annotation():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.ir.payload.passes.rewrites.raise_special_ops import (
        run_raise_special_ops,
    )

    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    bridge = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridge.module is not None

    run_raise_special_ops(bridge.module)
    stats = run_fuse_softmax_to_triton(bridge.module, config=_triton_config())
    # attention_mlp_tiny's softmax is 3-D over the last axis -> legal.
    assert stats.softmaxes_annotated >= 1
    assert_module_verifies(bridge.module)
