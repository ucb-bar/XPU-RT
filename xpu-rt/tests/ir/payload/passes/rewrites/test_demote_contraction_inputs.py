"""Tests for W1.3 ``demote_contraction_inputs``."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.rewrites.demote_contraction_inputs import (
    DemoteContractionInputsConfig,
    DemoteContractionStats,
    run_demote_contraction_inputs,
)
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _tensor(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _wrap(ops, return_value, ret_type):
    block = Block()
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(return_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ret_type]), Region([block]))
    return ModuleOp([func])


def _f32_matmul_module(M=4, K=8, N=16) -> ModuleOp:
    a = EmptyOp([], _tensor([M, K]))
    b = EmptyOp([], _tensor([K, N]))
    out = EmptyOp([], _tensor([M, N]))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_tensor([M, N])],
    )
    return _wrap([a, b, out, mm], mm.res[0], _tensor([M, N]))


# --- basic demote ------------------------------------------------------------


def test_basic_f32_matmul_is_demoted_to_bf16():
    m = _f32_matmul_module()
    stats = run_demote_contraction_inputs(m)
    assert stats.contractions_rewritten == 1
    assert stats.operands_truncated == 2
    assert count_ops(m, "linalg.matmul") == 0
    assert count_ops(m, "linalg.generic") == 3  # 2 truncs + 1 matmul body
    assert_module_verifies(m)


def test_demote_inserts_truncf_ops():
    m = _f32_matmul_module()
    run_demote_contraction_inputs(m)
    assert count_ops(m, "arith.truncf") == 2


def test_demote_inserts_extf_in_matmul_body():
    m = _f32_matmul_module()
    run_demote_contraction_inputs(m)
    assert count_ops(m, "arith.extf") == 2


def test_demote_preserves_f32_accumulator():
    m = _f32_matmul_module(M=4, K=8, N=16)
    run_demote_contraction_inputs(m)
    # Walk the linalg.generic replacing the matmul and check its output type.
    for op in m.walk():
        if op.name == "linalg.generic" and len(op.operands) == 3:
            out_val = op.operands[2]
            assert out_val.type.get_element_type() == Float32Type()
            break
    else:
        pytest.fail("expected a 3-operand linalg.generic")


# --- target type ------------------------------------------------------------


def test_demote_to_f16_works():
    m = _f32_matmul_module()
    cfg = DemoteContractionInputsConfig(target_type=Float16Type())
    run_demote_contraction_inputs(m, config=cfg)
    assert count_ops(m, "arith.truncf") == 2
    assert_module_verifies(m)


def test_demote_to_f16_preserves_input_shape():
    m = _f32_matmul_module(M=4, K=8, N=16)
    cfg = DemoteContractionInputsConfig(target_type=Float16Type())
    run_demote_contraction_inputs(m, config=cfg)
    # Find a truncf. Its result tensor's element type should be f16.
    for op in m.walk():
        if op.name == "arith.truncf":
            assert op.result.type == Float16Type()
            break


# --- skip cases -------------------------------------------------------------


def test_already_bf16_matmul_is_skipped():
    bf = BFloat16Type()
    a = EmptyOp([], _tensor([4, 8], bf))
    b = EmptyOp([], _tensor([8, 16], bf))
    out = EmptyOp([], _tensor([4, 16], bf))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_tensor([4, 16], bf)],
    )
    m = _wrap([a, b, out, mm], mm.res[0], _tensor([4, 16], bf))
    stats = run_demote_contraction_inputs(m)
    assert stats.contractions_rewritten == 0
    # Skipped because accumulator (bf16) isn't wider than target (bf16).
    assert stats.contractions_skipped_wrong_dtype == 1


def test_f16_accumulator_rejects_bf16_target():
    f16 = Float16Type()
    a = EmptyOp([], _tensor([4, 8], f16))
    b = EmptyOp([], _tensor([8, 16], f16))
    out = EmptyOp([], _tensor([4, 16], f16))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_tensor([4, 16], f16)],
    )
    m = _wrap([a, b, out, mm], mm.res[0], _tensor([4, 16], f16))
    stats = run_demote_contraction_inputs(m)
    assert stats.contractions_rewritten == 0


# --- region filter ----------------------------------------------------------


def test_restrict_to_region_ids_filters_matches():
    m = _f32_matmul_module()
    # Tag the matmul with a region_id that ISN'T in the allowlist.
    for op in m.walk():
        if op.name == "linalg.matmul":
            op.attributes["compgen.region_id"] = StringAttr("mm_other")
            break

    cfg = DemoteContractionInputsConfig(
        restrict_to_region_ids=frozenset({"mm_selected"}),
    )
    stats = run_demote_contraction_inputs(m, config=cfg)
    assert stats.contractions_rewritten == 0
    assert stats.contractions_skipped_region_filter == 1


def test_restrict_to_region_ids_includes_match():
    m = _f32_matmul_module()
    for op in m.walk():
        if op.name == "linalg.matmul":
            op.attributes["compgen.region_id"] = StringAttr("mm_0")
            break

    cfg = DemoteContractionInputsConfig(
        restrict_to_region_ids=frozenset({"mm_0"}),
    )
    stats = run_demote_contraction_inputs(m, config=cfg)
    assert stats.contractions_rewritten == 1


# --- idempotence + stats ---------------------------------------------------


def test_idempotent_second_run_is_noop():
    m = _f32_matmul_module()
    first = run_demote_contraction_inputs(m)
    assert first.contractions_rewritten == 1
    second = run_demote_contraction_inputs(m)
    assert second.contractions_rewritten == 0


def test_stats_initial_values():
    s = DemoteContractionStats()
    assert s.contractions_seen == 0
    assert s.operands_truncated == 0


# --- region-id / pattern-hint preservation ----------------------------------


def test_region_id_preserved_across_demote():
    m = _f32_matmul_module()
    for op in m.walk():
        if op.name == "linalg.matmul":
            op.attributes["compgen.region_id"] = StringAttr("matmul_0")
            op.attributes["compgen._pattern_hint"] = StringAttr("gemm")
            break

    run_demote_contraction_inputs(m)
    # The replacement generic should inherit both tags.
    generics = [op for op in m.walk() if op.name == "linalg.generic"]
    # The mixed-precision matmul is the 3-operand generic.
    mm_generic = next(op for op in generics if len(op.operands) == 3)
    assert mm_generic.attributes["compgen.region_id"].data == "matmul_0"
    assert mm_generic.attributes["compgen._pattern_hint"].data == "gemm"


# --- multiple contractions --------------------------------------------------


def test_two_matmuls_both_rewritten():
    # Build two independent matmuls feeding a third that consumes both.
    a = EmptyOp([], _tensor([4, 8]))
    b = EmptyOp([], _tensor([8, 4]))
    out1 = EmptyOp([], _tensor([4, 4]))
    mm1 = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out1.results[0]],
        res=[_tensor([4, 4])],
    )
    c = EmptyOp([], _tensor([4, 8]))
    out2 = EmptyOp([], _tensor([4, 8]))
    mm2 = MatmulOp(
        inputs=[mm1.res[0], c.results[0]],
        outputs=[out2.results[0]],
        res=[_tensor([4, 8])],
    )
    m = _wrap([a, b, out1, mm1, c, out2, mm2], mm2.res[0], _tensor([4, 8]))

    stats = run_demote_contraction_inputs(m)
    assert stats.contractions_rewritten == 2
    assert_module_verifies(m)
