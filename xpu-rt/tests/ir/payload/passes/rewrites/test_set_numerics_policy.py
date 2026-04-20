"""Tests for W1.4 ``set_numerics_policy``."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.rewrites.set_numerics_policy import (
    NumericsPolicy,
    SetNumericsPolicyStats,
    run_set_numerics_policy,
)
from xdsl.dialects.arith import (
    AddfOp,
    ConstantOp,
    DivfOp,
    MaximumfOp,
    MinimumfOp,
    MulfOp,
    SubfOp,
)
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    FloatAttr,
    FunctionType,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _wrap(ops, return_value, ret_type):
    block = Block()
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(return_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ret_type]), Region([block]))
    return ModuleOp([func])


def _bf_add_module() -> ModuleOp:
    bf = BFloat16Type()
    c1 = ConstantOp(FloatAttr(1.0, bf))
    c2 = ConstantOp(FloatAttr(2.0, bf))
    add = AddfOp(c1.result, c2.result)
    return _wrap([c1, c2, add], add.result, bf)


# --- basic promotion --------------------------------------------------------


def test_bf16_addf_promoted_to_f32_when_not_supported():
    m = _bf_add_module()
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_add": frozenset({Float32Type})},
    )
    stats = run_set_numerics_policy(m, policy=policy)
    assert stats.ops_promoted == 1
    assert stats.operands_extf_inserted == 2
    assert stats.results_truncf_inserted == 1
    assert count_ops(m, "arith.extf") == 2
    assert count_ops(m, "arith.truncf") == 1
    assert_module_verifies(m)


def test_supported_elem_type_is_not_touched():
    m = _bf_add_module()
    policy = NumericsPolicy(
        supported_per_kind={
            "elementwise_add": frozenset({Float32Type, BFloat16Type}),
        },
    )
    stats = run_set_numerics_policy(m, policy=policy)
    assert stats.ops_promoted == 0
    assert stats.ops_skipped_legal == 1
    assert count_ops(m, "arith.extf") == 0


def test_default_policy_is_permissive():
    # The default policy has no per-kind restrictions -> nothing promoted.
    m = _bf_add_module()
    stats = run_set_numerics_policy(m)
    assert stats.ops_promoted == 0


# --- stats ------------------------------------------------------------------


def test_stats_initial_values():
    s = SetNumericsPolicyStats()
    assert s.ops_seen == 0
    assert s.ops_promoted == 0


# --- multiple op kinds ------------------------------------------------------


def test_mulf_promotion():
    bf = BFloat16Type()
    c1 = ConstantOp(FloatAttr(1.0, bf))
    c2 = ConstantOp(FloatAttr(2.0, bf))
    mul = MulfOp(c1.result, c2.result)
    m = _wrap([c1, c2, mul], mul.result, bf)
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_mul": frozenset({Float32Type})},
    )
    run_set_numerics_policy(m, policy=policy)
    assert count_ops(m, "arith.mulf") == 1
    assert count_ops(m, "arith.extf") == 2
    assert count_ops(m, "arith.truncf") == 1


def test_divf_promotion():
    f16 = Float16Type()
    c1 = ConstantOp(FloatAttr(3.0, f16))
    c2 = ConstantOp(FloatAttr(2.0, f16))
    div = DivfOp(c1.result, c2.result)
    m = _wrap([c1, c2, div], div.result, f16)
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_div": frozenset({Float32Type})},
    )
    stats = run_set_numerics_policy(m, policy=policy)
    assert stats.ops_promoted == 1


def test_subf_maxf_minf_each_promote_independently():
    bf = BFloat16Type()
    c1 = ConstantOp(FloatAttr(1.0, bf))
    c2 = ConstantOp(FloatAttr(2.0, bf))
    sub = SubfOp(c1.result, c2.result)
    mx = MaximumfOp(sub.result, c2.result)
    mn = MinimumfOp(mx.result, c1.result)
    m = _wrap([c1, c2, sub, mx, mn], mn.result, bf)
    policy = NumericsPolicy(
        supported_per_kind={
            "elementwise_sub": frozenset({Float32Type}),
            "elementwise_max": frozenset({Float32Type}),
            "elementwise_min": frozenset({Float32Type}),
        },
    )
    stats = run_set_numerics_policy(m, policy=policy)
    assert stats.ops_promoted == 3


# --- idempotence ------------------------------------------------------------


def test_idempotent_second_run_is_noop():
    m = _bf_add_module()
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_add": frozenset({Float32Type})},
    )
    first = run_set_numerics_policy(m, policy=policy)
    assert first.ops_promoted == 1

    # After the first pass the addf operates on f32 (promoted) -- which
    # is in the allowlist so the second pass shouldn't rewrite.
    second = run_set_numerics_policy(m, policy=policy)
    assert second.ops_promoted == 0


# --- kind not listed => unrestricted ---------------------------------------


def test_unlisted_kind_is_unrestricted():
    m = _bf_add_module()
    # Policy restricts MUL but not ADD -- our bf16 addf should not be
    # touched.
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_mul": frozenset({Float32Type})},
    )
    stats = run_set_numerics_policy(m, policy=policy)
    assert stats.ops_promoted == 0


# --- region-id / pattern-hint preservation ----------------------------------


def test_attributes_preserved_across_promotion():
    m = _bf_add_module()
    for op in m.walk():
        if op.name == "arith.addf":
            op.attributes["compgen.region_id"] = StringAttr("add_0")
            op.attributes["compgen._pattern_hint"] = StringAttr("bias")
            break
    policy = NumericsPolicy(
        supported_per_kind={"elementwise_add": frozenset({Float32Type})},
    )
    run_set_numerics_policy(m, policy=policy)
    for op in m.walk():
        if op.name == "arith.addf":
            assert op.attributes["compgen.region_id"].data == "add_0"
            assert op.attributes["compgen._pattern_hint"].data == "bias"
            break
    else:
        pytest.fail("no promoted addf found after run")


# --- nothing to do ---------------------------------------------------------


def test_empty_module_is_safe():
    block = Block()
    block.add_op(ReturnOp())
    func = FuncOp("noop", FunctionType.from_lists([], []), Region([block]))
    m = ModuleOp([func])
    stats = run_set_numerics_policy(m)
    assert stats.ops_seen == 0
    assert_module_verifies(m)
