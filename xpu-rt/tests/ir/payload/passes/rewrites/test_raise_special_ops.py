"""Tests for W2.1 ``raise_special_ops``.

Includes both structural tests (synthetic fixtures built inline) and
real-workload tests that bridge ``attention_mlp_tiny`` /
``qwen_moe_tiny`` through the FX importer and run the pass.
"""

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

from compgen.ir.payload.passes.rewrites.raise_special_ops import (
    RaiseSpecialOpsStats,
    run_raise_special_ops,
)
from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


# --- synthetic fixture helpers -----------------------------------------------


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _wrap_single_hinted_call(
    callee: str,
    hint: str,
    *,
    n_operands: int = 1,
    shape: tuple[int, ...] = (4, 8),
) -> tuple[ModuleOp, CallOp]:
    """Build a module with one ``func.call`` carrying a pattern hint."""
    t = _ft(shape)
    empties = [EmptyOp([], t) for _ in range(n_operands)]
    call = CallOp(callee, [e.results[0] for e in empties], [t])
    call.attributes["compgen._pattern_hint"] = StringAttr(hint)
    call.attributes["compgen.region_id"] = StringAttr(f"{hint}_0")

    block = Block()
    for op in (*empties, call):
        block.add_op(op)
    block.add_op(ReturnOp(call.res[0]))
    # Declare the external callee so the module verifies.
    ext = FuncOp.external(callee, [t] * n_operands, [t])
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    return ModuleOp([ext, func]), call


# --- softmax ----------------------------------------------------------------


def test_softmax_hint_raises_to_linalg_ext_softmax():
    m, _ = _wrap_single_hinted_call("sm", "softmax", shape=(4, 8))
    stats = run_raise_special_ops(m)
    assert stats.raised_by_hint["softmax"] == 1
    assert count_ops(m, "compgen.linalg_ext.softmax") == 1


def test_softmax_dim_defaults_to_last_axis():
    m, _ = _wrap_single_hinted_call("sm", "softmax", shape=(4, 8, 16))
    run_raise_special_ops(m)
    for op in m.walk():
        if op.name == "compgen.linalg_ext.softmax":
            assert op.dim.value.data == 2
            break


# --- layer_norm -------------------------------------------------------------


def test_layer_norm_hint_raises_single_operand():
    m, _ = _wrap_single_hinted_call("ln", "layer_norm", n_operands=1)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.layer_norm") == 1


def test_layer_norm_hint_raises_with_weight_and_bias():
    m, _ = _wrap_single_hinted_call("ln", "layer_norm", n_operands=3)
    run_raise_special_ops(m)
    ops = [op for op in m.walk() if op.name == "compgen.linalg_ext.layer_norm"]
    assert len(ops) == 1
    assert ops[0].weight is not None
    assert ops[0].bias is not None


def test_native_layer_norm_alias_also_raises():
    # Some decomp paths emit ``native_layer_norm`` directly.
    m, _ = _wrap_single_hinted_call("nln", "native_layer_norm", n_operands=1)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.layer_norm") == 1


# --- rms_norm ---------------------------------------------------------------


def test_rms_norm_hint_raises_with_weight():
    m, _ = _wrap_single_hinted_call("rms", "rms_norm", n_operands=2)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.rms_norm") == 1


# --- silu / gelu / swiglu --------------------------------------------------


def test_silu_hint_raises():
    m, _ = _wrap_single_hinted_call("silu", "silu", n_operands=1)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.silu") == 1


def test_gelu_hint_raises():
    m, _ = _wrap_single_hinted_call("gelu", "gelu", n_operands=1)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.gelu") == 1


def test_swiglu_hint_raises_with_two_operands():
    m, _ = _wrap_single_hinted_call("swiglu", "swiglu", n_operands=2)
    run_raise_special_ops(m)
    assert count_ops(m, "compgen.linalg_ext.swiglu") == 1


# --- gates + preservation ---------------------------------------------------


def test_ops_without_hint_are_untouched():
    t = _ft([4, 8])
    e = EmptyOp([], t)
    block = Block()
    block.add_op(e)
    block.add_op(ReturnOp(e.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([func])
    stats = run_raise_special_ops(m)
    assert stats.total_raised == 0


def test_unknown_hint_is_ignored():
    m, _ = _wrap_single_hinted_call("x", "some_unknown_pattern")
    stats = run_raise_special_ops(m)
    assert stats.total_raised == 0


def test_region_id_preserved_through_raise():
    m, _ = _wrap_single_hinted_call("sm", "softmax")
    run_raise_special_ops(m)
    for op in m.walk():
        if op.name == "compgen.linalg_ext.softmax":
            assert op.attributes["compgen.region_id"].data == "softmax_0"
            break
    else:
        pytest.fail("softmax not raised")


# --- idempotence + stats ---------------------------------------------------


def test_idempotent_second_run_is_noop():
    m, _ = _wrap_single_hinted_call("sm", "softmax")
    first = run_raise_special_ops(m)
    assert first.total_raised == 1
    second = run_raise_special_ops(m)
    assert second.total_raised == 0


def test_stats_initial_values():
    s = RaiseSpecialOpsStats()
    assert s.hinted_ops_seen == 0
    assert s.total_raised == 0


# --- real-workload integration ---------------------------------------------


def test_raise_special_ops_on_attention_mlp_tiny():
    """Real-workload test per no-stubs-real-examples memory.

    Bridges ``attention_mlp_tiny`` through the FX importer and
    verifies softmax + layer_norm + silu are all raised, and the
    module still verifies.
    """
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None, "bridge_fx_graph must succeed on this fixture"

    stats = run_raise_special_ops(result.module)
    # attention_mlp_tiny has one layer_norm + one softmax + one silu.
    assert stats.raised_by_hint.get("layer_norm", 0) == 1
    assert stats.raised_by_hint.get("softmax", 0) == 1
    assert stats.raised_by_hint.get("silu", 0) == 1
    assert_module_verifies(result.module)


def test_raise_special_ops_on_qwen_moe_tiny():
    """Real-workload test: qwen_moe_tiny has a softmax in the router."""
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import qwen_moe_tiny

    fx = qwen_moe_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    assert result.module is not None

    stats = run_raise_special_ops(result.module)
    # Router softmax + silu in each of the 2 experts.
    assert stats.raised_by_hint.get("softmax", 0) >= 1
    assert stats.raised_by_hint.get("silu", 0) >= 1
    assert_module_verifies(result.module)


def test_raise_special_ops_is_attribute_preserving_on_real_workload():
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    from tests._fixtures.real_workloads import attention_mlp_tiny

    fx = attention_mlp_tiny()
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    # Collect region_ids before the raise.
    hinted_region_ids = set()
    for op in result.module.walk():
        if "compgen._pattern_hint" in op.attributes:
            attr = op.attributes.get("compgen.region_id")
            if attr is not None:
                hinted_region_ids.add(attr.data)

    run_raise_special_ops(result.module)

    # Every region_id should still be present on *some* op after the rewrite.
    ids_after = set()
    for op in result.module.walk():
        attr = op.attributes.get("compgen.region_id")
        if attr is not None:
            ids_after.add(attr.data)
    missing = hinted_region_ids - ids_after
    assert not missing, f"region_ids dropped during raise: {missing}"
