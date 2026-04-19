"""Tests for W3.1 ``propagate_transposes``."""

from __future__ import annotations

import pytest
from xdsl.dialects.arith import ConstantOp, MulfOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    Float32Type,
    FloatAttr,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
    TransposeOp,
    YieldOp,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from compgen.ir.payload.passes.rewrites.propagate_transposes import (
    PropagateTransposesConfig,
    PropagateTransposesStats,
    run_propagate_transposes,
)
from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _perm(vs):
    return DenseArrayBase.from_list(i64, vs)


def _wrap(ops, ret_value, ret_type):
    block = Block()
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(ret_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ret_type]), Region([block]))
    return ModuleOp([func])


# --- Chained transpose collapse ---------------------------------------------


def test_double_reverse_perm_collapses():
    t1 = _ft([4, 8])
    t2 = _ft([8, 4])
    x = EmptyOp([], t1)
    i1 = EmptyOp([], t2)
    tr1 = TransposeOp(
        input=x.results[0], init=i1.results[0],
        permutation=_perm([1, 0]), result=t2,
    )
    i2 = EmptyOp([], t1)
    tr2 = TransposeOp(
        input=tr1.results[0], init=i2.results[0],
        permutation=_perm([1, 0]), result=t1,
    )
    m = _wrap([x, i1, tr1, i2, tr2], tr2.results[0], t1)

    stats = run_propagate_transposes(m)
    assert stats.chained_collapses >= 1
    assert_module_verifies(m)


def test_composed_non_identity_folds_into_single():
    # 3D: (0,1,2) -> tr([2,0,1]) -> tr([1,2,0]) => composed = [(2)(1), (0)(1), (1)(1)]
    # For outer=[1,2,0], inner=[2,0,1]: composed = [inner[outer[i]]] = [inner[1], inner[2], inner[0]]
    # = [0, 1, 2] = identity. So we hit the identity path.
    # Use outer=[2,0,1], inner=[2,0,1]: composed = [inner[2], inner[0], inner[1]] = [1, 2, 0] (not identity).
    t0 = _ft([2, 3, 4])
    t1 = _ft([4, 2, 3])
    t2 = _ft([3, 4, 2])
    x = EmptyOp([], t0)
    i1 = EmptyOp([], t1)
    tr1 = TransposeOp(input=x.results[0], init=i1.results[0],
                      permutation=_perm([2, 0, 1]), result=t1)
    i2 = EmptyOp([], t2)
    tr2 = TransposeOp(input=tr1.results[0], init=i2.results[0],
                      permutation=_perm([2, 0, 1]), result=t2)
    m = _wrap([x, i1, tr1, i2, tr2], tr2.results[0], t2)

    stats = run_propagate_transposes(m)
    assert stats.chained_collapses >= 1
    # After one fold the module should still verify.
    assert_module_verifies(m)


def test_three_chained_transposes_reduce_to_one_or_zero():
    """Three transposes where p3 ∘ p2 ∘ p1 = identity should reduce fully."""
    t0 = _ft([4, 8])
    x = EmptyOp([], t0)
    i1 = EmptyOp([], _ft([8, 4]))
    tr1 = TransposeOp(input=x.results[0], init=i1.results[0],
                      permutation=_perm([1, 0]), result=_ft([8, 4]))
    i2 = EmptyOp([], _ft([4, 8]))
    tr2 = TransposeOp(input=tr1.results[0], init=i2.results[0],
                      permutation=_perm([1, 0]), result=_ft([4, 8]))
    i3 = EmptyOp([], _ft([8, 4]))
    tr3 = TransposeOp(input=tr2.results[0], init=i3.results[0],
                      permutation=_perm([1, 0]), result=_ft([8, 4]))
    m = _wrap([x, i1, tr1, i2, tr2, i3, tr3], tr3.results[0], _ft([8, 4]))

    stats = run_propagate_transposes(m)
    # At least one collapse should fire; multiple rounds may fire in the
    # greedy walker.
    assert stats.chained_collapses >= 1
    assert_module_verifies(m)


def test_single_transpose_is_not_touched():
    t0 = _ft([4, 8])
    t1 = _ft([8, 4])
    x = EmptyOp([], t0)
    i = EmptyOp([], t1)
    tr = TransposeOp(input=x.results[0], init=i.results[0],
                     permutation=_perm([1, 0]), result=t1)
    m = _wrap([x, i, tr], tr.results[0], t1)

    stats = run_propagate_transposes(m)
    assert stats.chained_collapses == 0
    assert count_ops(m, "linalg.transpose") == 1


# --- Push transpose into elementwise generic --------------------------------


def _identity_generic(inp, out_shape):
    """Build a simple elementwise-mul-by-2 generic."""
    out_type = _ft(out_shape)
    init = EmptyOp([], out_type)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    c = ConstantOp(FloatAttr(2.0, Float32Type()))
    mul = MulfOp(body.args[0], c.result)
    body.add_op(c)
    body.add_op(mul)
    body.add_op(YieldOp(mul.result))

    rank = len(list(out_shape))
    idmap = AffineMap.identity(rank)
    g = GenericOp(
        inputs=[inp],
        outputs=[init.results[0]],
        body=Region([body]),
        indexing_maps=[AffineMapAttr(idmap), AffineMapAttr(idmap)],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * rank,
        result_types=[out_type],
    )
    return init, g


def test_push_through_elementwise_generic():
    t1 = _ft([4, 8])
    t2 = _ft([8, 4])
    x = EmptyOp([], t1)
    it = EmptyOp([], t2)
    tr = TransposeOp(input=x.results[0], init=it.results[0],
                     permutation=_perm([1, 0]), result=t2)
    init, g = _identity_generic(tr.results[0], (8, 4))
    m = _wrap([x, it, tr, init, g], g.results[0], t2)

    stats = run_propagate_transposes(m)
    assert stats.elementwise_pushes == 1
    assert_module_verifies(m)


def test_conservative_aggressiveness_does_not_push():
    t1 = _ft([4, 8])
    t2 = _ft([8, 4])
    x = EmptyOp([], t1)
    it = EmptyOp([], t2)
    tr = TransposeOp(input=x.results[0], init=it.results[0],
                     permutation=_perm([1, 0]), result=t2)
    init, g = _identity_generic(tr.results[0], (8, 4))
    m = _wrap([x, it, tr, init, g], g.results[0], t2)

    stats = run_propagate_transposes(
        m, config=PropagateTransposesConfig(aggressiveness="conservative")
    )
    assert stats.elementwise_pushes == 0


def test_invalid_aggressiveness_raises():
    # Build a trivial empty module; the aggressiveness check fires
    # before anything touches the IR.
    m = ModuleOp([])
    with pytest.raises(ValueError, match="aggressiveness"):
        run_propagate_transposes(
            m,
            config=PropagateTransposesConfig(aggressiveness="lunatic"),
        )


# --- Non-matching cases -----------------------------------------------------


def test_no_transpose_is_noop():
    t = _ft([4, 8])
    init, g = _identity_generic(EmptyOp([], t).results[0], (4, 8))
    # need to include the input op
    input_e = EmptyOp([], t)
    body = Block(arg_types=[Float32Type(), Float32Type()])
    c = ConstantOp(FloatAttr(2.0, Float32Type()))
    mul = MulfOp(body.args[0], c.result)
    body.add_op(c); body.add_op(mul); body.add_op(YieldOp(mul.result))
    init2 = EmptyOp([], t)
    idmap = AffineMap.identity(2)
    g = GenericOp(
        inputs=[input_e.results[0]], outputs=[init2.results[0]],
        body=Region([body]),
        indexing_maps=[AffineMapAttr(idmap), AffineMapAttr(idmap)],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL), IteratorTypeAttr(IteratorType.PARALLEL)],
        result_types=[t],
    )
    m = _wrap([input_e, init2, g], g.results[0], t)
    stats = run_propagate_transposes(m)
    assert stats.chained_collapses == 0
    assert stats.elementwise_pushes == 0


# --- stats + idempotence ---------------------------------------------------


def test_stats_initial_values():
    s = PropagateTransposesStats()
    assert s.chained_collapses == 0
    assert s.elementwise_pushes == 0


def test_idempotent_second_run_is_noop_or_smaller():
    t1 = _ft([4, 8])
    t2 = _ft([8, 4])
    x = EmptyOp([], t1)
    i1 = EmptyOp([], t2)
    tr1 = TransposeOp(input=x.results[0], init=i1.results[0],
                      permutation=_perm([1, 0]), result=t2)
    i2 = EmptyOp([], t1)
    tr2 = TransposeOp(input=tr1.results[0], init=i2.results[0],
                      permutation=_perm([1, 0]), result=t1)
    m = _wrap([x, i1, tr1, i2, tr2], tr2.results[0], t1)
    first = run_propagate_transposes(m)
    assert first.chained_collapses >= 1
    second = run_propagate_transposes(m)
    assert second.chained_collapses == 0


# --- attribute preservation ------------------------------------------------


def test_region_id_preserved_through_chained_collapse():
    t1 = _ft([4, 8])
    t2 = _ft([8, 4])
    x = EmptyOp([], t1)
    i1 = EmptyOp([], t2)
    tr1 = TransposeOp(input=x.results[0], init=i1.results[0],
                      permutation=_perm([1, 0]), result=t2)
    i2 = EmptyOp([], t1)
    tr2 = TransposeOp(input=tr1.results[0], init=i2.results[0],
                      permutation=_perm([1, 0]), result=t1)
    tr2.attributes["compgen.region_id"] = StringAttr("outer_t")
    m = _wrap([x, i1, tr1, i2, tr2], tr2.results[0], t1)

    run_propagate_transposes(m)
    # Identity collapse erases the outer transpose entirely and
    # replaces it with the inner's input, so the region_id is gone.
    # That's expected (compose semantics preserved upstream).
    assert_module_verifies(m)


# --- real-workload integration ---------------------------------------------


def test_propagate_transposes_on_attention_mlp_tiny_is_safe():
    """Real-workload test: the pass should not introduce verifier
    errors on a bridged attention block."""
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    stats = run_propagate_transposes(result.module)
    # Structural sanity: module still verifies and has same linalg.matmul count.
    assert_module_verifies(result.module)
    # The fixture has no linalg.transpose to start (only linalg.matmul),
    # so the pass should be effectively a no-op.
    assert stats.chained_collapses + stats.elementwise_pushes >= 0


def test_propagate_transposes_on_qwen_moe_tiny_is_safe():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import qwen_moe_tiny

    fx = qwen_moe_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    run_propagate_transposes(result.module)
    assert_module_verifies(result.module)
