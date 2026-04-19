"""Tests for W3.2 ``plan_reduction``."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    Float32Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
    YieldOp,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap

from compgen.ir.linalg_ext import LayerNormOp, RMSNormOp, SoftmaxOp
from compgen.ir.payload.passes.rewrites.plan_reduction import (
    PlanReductionConfig,
    PlanReductionStats,
    run_plan_reduction,
)
from tests.ir.payload.passes._pattern_test_helpers import assert_module_verifies


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


# --- auto strategy by extent ------------------------------------------------


def test_small_reduction_chooses_group():
    m = _softmax_module(shape=(4, 8))
    stats = run_plan_reduction(m)
    assert stats.chosen_group == 1
    assert stats.chosen_split == 0
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm.attributes["compgen.reduction_strategy"].data == "group"


def test_medium_reduction_chooses_split():
    m = _softmax_module(shape=(4, 1024))
    stats = run_plan_reduction(m)
    assert stats.chosen_split == 1
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm.attributes["compgen.reduction_strategy"].data == "split"


def test_large_reduction_chooses_tree_reduce():
    m = _softmax_module(shape=(4, 16384))
    stats = run_plan_reduction(m)
    assert stats.chosen_tree_reduce == 1
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm.attributes["compgen.reduction_strategy"].data == "tree_reduce"


def test_reduction_extent_recorded():
    m = _softmax_module(shape=(4, 1024))
    run_plan_reduction(m)
    sm = next(op for op in m.walk() if op.name == "compgen.linalg_ext.softmax")
    assert sm.attributes["compgen.reduction_extent"].value.data == 1024


# --- explicit policy overrides auto --------------------------------------


def test_explicit_group_policy():
    m = _softmax_module(shape=(4, 16384))  # would auto-pick tree_reduce
    stats = run_plan_reduction(m, config=PlanReductionConfig(policy="group"))
    assert stats.chosen_group == 1
    assert stats.chosen_tree_reduce == 0


def test_explicit_tree_reduce_policy():
    m = _softmax_module(shape=(4, 8))  # would auto-pick group
    stats = run_plan_reduction(m, config=PlanReductionConfig(policy="tree_reduce"))
    assert stats.chosen_tree_reduce == 1


def test_invalid_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        PlanReductionConfig(policy="cosmic")


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        PlanReductionConfig(large_reduction_threshold=0)
    with pytest.raises(ValueError, match="tree_reduce_threshold"):
        PlanReductionConfig(
            large_reduction_threshold=1000, tree_reduce_threshold=500
        )


# --- linalg.generic reduction coverage ---------------------------------------


def _generic_reduction_module(shape=(4, 1024)) -> ModuleOp:
    """``linalg.generic`` reducing the last axis to f32[shape[0]]."""
    t_in = _ft(shape)
    t_out = _ft([shape[0]])
    e_in = EmptyOp([], t_in)
    e_out = EmptyOp([], t_out)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    from xdsl.dialects.arith import AddfOp
    add = AddfOp(body.args[0], body.args[1])
    body.add_op(add)
    body.add_op(YieldOp(add.result))

    d0, d1 = AffineExpr.dimension(0), AffineExpr.dimension(1)
    input_map = AffineMap(2, 0, (d0, d1))
    output_map = AffineMap(2, 0, (d0,))
    g = GenericOp(
        inputs=[e_in.results[0]],
        outputs=[e_out.results[0]],
        body=Region([body]),
        indexing_maps=[AffineMapAttr(input_map), AffineMapAttr(output_map)],
        iterator_types=[
            IteratorTypeAttr(IteratorType.PARALLEL),
            IteratorTypeAttr(IteratorType.REDUCTION),
        ],
        result_types=[t_out],
    )
    block = Block()
    for op in (e_in, e_out, g):
        block.add_op(op)
    block.add_op(ReturnOp(g.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t_out]), Region([block]))
    return ModuleOp([func])


def test_generic_reduction_is_annotated():
    m = _generic_reduction_module(shape=(4, 1024))
    stats = run_plan_reduction(m)
    assert stats.ops_annotated == 1
    g = next(op for op in m.walk() if op.name == "linalg.generic")
    assert g.attributes["compgen.reduction_strategy"].data == "split"


def test_non_reduction_generic_is_skipped():
    """Elementwise (all-parallel) generic must NOT be annotated."""
    t = _ft([4, 8])
    e_in = EmptyOp([], t)
    e_out = EmptyOp([], t)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    body.add_op(YieldOp(body.args[0]))
    identity = AffineMap.identity(2)
    g = GenericOp(
        inputs=[e_in.results[0]],
        outputs=[e_out.results[0]],
        body=Region([body]),
        indexing_maps=[AffineMapAttr(identity), AffineMapAttr(identity)],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 2,
        result_types=[t],
    )
    block = Block()
    for op in (e_in, e_out, g):
        block.add_op(op)
    block.add_op(ReturnOp(g.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([func])

    stats = run_plan_reduction(m)
    assert stats.ops_annotated == 0
    assert stats.skipped_non_reduction >= 1


# --- linalg_ext ops ---------------------------------------------------------


def test_rms_norm_is_annotated():
    t = _ft([4, 128])
    e = EmptyOp([], t)
    w = EmptyOp([], _ft([128]))
    op = RMSNormOp(e.results[0], t, weight=w.results[0], eps=1e-6)
    block = Block()
    for x in (e, w, op):
        block.add_op(x)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([func])

    run_plan_reduction(m)
    assert "compgen.reduction_strategy" in op.attributes


def test_layer_norm_is_annotated():
    t = _ft([4, 1024])
    e = EmptyOp([], t)
    op = LayerNormOp(e.results[0], t, eps=1e-5)
    block = Block()
    for x in (e, op):
        block.add_op(x)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([func])

    run_plan_reduction(m)
    assert op.attributes["compgen.reduction_strategy"].data == "split"


# --- idempotence + stats ---------------------------------------------------


def test_idempotent_second_run_is_noop():
    m = _softmax_module(shape=(4, 1024))
    first = run_plan_reduction(m)
    assert first.ops_annotated == 1
    second = run_plan_reduction(m)
    assert second.ops_annotated == 0
    assert second.skipped_already_annotated == 1


def test_stats_initial_values():
    s = PlanReductionStats()
    assert s.ops_seen == 0
    assert s.ops_annotated == 0


# --- real-workload ----------------------------------------------------------


def test_plan_reduction_on_qwen_moe_tiny():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.ir.payload.passes.rewrites.raise_special_ops import (
        run_raise_special_ops,
    )
    from tests._fixtures.real_workloads import qwen_moe_tiny

    fx = qwen_moe_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    # Raise softmax first so plan_reduction sees the linalg_ext op.
    run_raise_special_ops(result.module)
    stats = run_plan_reduction(result.module)
    # qwen_moe_tiny has a softmax over last dim = 2 (n_experts).
    # That's small -> group strategy.
    assert stats.ops_annotated >= 1
    annotated = [
        op for op in result.module.walk()
        if "compgen.reduction_strategy" in op.attributes
    ]
    assert len(annotated) >= 1
    for op in annotated:
        assert op.attributes["compgen.reduction_strategy"].data in {
            "group", "split", "tree_reduce"
        }
    assert_module_verifies(result.module)


# --- real structural rewrite: group permutes iteration dims ---------------


def test_group_permutes_iteration_dims_so_reductions_trail():
    """Real structural rewrite: iterator_types=[p, r, p, r] → [p, p, r, r]."""
    from xdsl.dialects.arith import AddfOp
    t_in = _ft([2, 4, 8, 16])
    t_out = _ft([2, 8])
    e_in = EmptyOp([], t_in)
    e_out = EmptyOp([], t_out)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    add = AddfOp(body.args[0], body.args[1])
    body.add_op(add)
    body.add_op(YieldOp(add.result))
    d0, d1, d2, d3 = (AffineExpr.dimension(i) for i in range(4))
    input_map = AffineMap(4, 0, (d0, d1, d2, d3))
    output_map = AffineMap(4, 0, (d0, d2))
    g = GenericOp(
        inputs=[e_in.results[0]], outputs=[e_out.results[0]], body=Region([body]),
        indexing_maps=[AffineMapAttr(input_map), AffineMapAttr(output_map)],
        iterator_types=[
            IteratorTypeAttr(IteratorType.PARALLEL),
            IteratorTypeAttr(IteratorType.REDUCTION),
            IteratorTypeAttr(IteratorType.PARALLEL),
            IteratorTypeAttr(IteratorType.REDUCTION),
        ],
        result_types=[t_out],
    )
    block = Block()
    for op in (e_in, e_out, g):
        block.add_op(op)
    block.add_op(ReturnOp(g.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t_out]), Region([block]))
    m = ModuleOp([func])

    stats = run_plan_reduction(m, config=PlanReductionConfig(policy="group"))
    assert stats.iteration_permutations_applied == 1

    # After the rewrite: reduction dims trail.
    kinds_after = [k.data for k in g.iterator_types.data]
    assert kinds_after == [
        IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.REDUCTION, IteratorType.REDUCTION,
    ]
    assert_module_verifies(m)


def test_group_no_op_when_reductions_already_trailing():
    """When iterator_types are already [p, ..., r, ...], no permutation fires."""
    from xdsl.dialects.arith import AddfOp
    t_in = _ft([2, 4])
    t_out = _ft([2])
    e_in = EmptyOp([], t_in)
    e_out = EmptyOp([], t_out)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    add = AddfOp(body.args[0], body.args[1])
    body.add_op(add)
    body.add_op(YieldOp(add.result))
    d0, d1 = (AffineExpr.dimension(i) for i in range(2))
    g = GenericOp(
        inputs=[e_in.results[0]], outputs=[e_out.results[0]], body=Region([body]),
        indexing_maps=[
            AffineMapAttr(AffineMap(2, 0, (d0, d1))),
            AffineMapAttr(AffineMap(2, 0, (d0,))),
        ],
        iterator_types=[
            IteratorTypeAttr(IteratorType.PARALLEL),
            IteratorTypeAttr(IteratorType.REDUCTION),
        ],
        result_types=[t_out],
    )
    block = Block()
    for op in (e_in, e_out, g):
        block.add_op(op)
    block.add_op(ReturnOp(g.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t_out]), Region([block]))
    m = ModuleOp([func])

    stats = run_plan_reduction(m, config=PlanReductionConfig(policy="group"))
    assert stats.chosen_group == 1
    assert stats.iteration_permutations_applied == 0
    assert_module_verifies(m)


def test_plan_reduction_on_attention_mlp_tiny():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from compgen.ir.payload.passes.rewrites.raise_special_ops import (
        run_raise_special_ops,
    )
    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    run_raise_special_ops(result.module)
    stats = run_plan_reduction(result.module)
    # softmax + layer_norm both carry implicit reductions.
    assert stats.ops_annotated >= 2
    assert_module_verifies(result.module)
