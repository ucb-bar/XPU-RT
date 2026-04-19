"""Tests for the shared pattern-test helpers."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.dialects.tensor import EmptyOp

from tests.ir.payload.passes._pattern_test_helpers import (
    all_ops,
    apply_pattern,
    assert_module_verifies,
    assert_op_count,
    assert_smt_equivalent,
    build_concat_module,
    build_linalg_matmul_module,
    build_quantized_matmul_module,
    count_ops,
    find_op_by_region_id,
)


# --- Module builders ---------------------------------------------------------


def test_build_linalg_matmul_module_verifies():
    m = build_linalg_matmul_module()
    assert_module_verifies(m)
    assert count_ops(m, "linalg.matmul") == 1


def test_build_linalg_matmul_module_tags_region_id():
    m = build_linalg_matmul_module(region_id="mm_xyz")
    op = find_op_by_region_id(m, "mm_xyz")
    assert op is not None
    assert op.name == "linalg.matmul"


def test_build_concat_module_verifies_and_has_concat():
    m = build_concat_module([(2, 4), (3, 4), (5, 4)], dim=0)
    assert_module_verifies(m)
    assert count_ops(m, "compgen.tensor_ext.concat") == 1


def test_build_quantized_matmul_module_int8():
    m = build_quantized_matmul_module(bits=8)
    assert_module_verifies(m)
    assert count_ops(m, "compgen.quant.weight_int8pack_mm") == 1


def test_build_quantized_matmul_module_int4():
    m = build_quantized_matmul_module(bits=4, group_size=128)
    assert_module_verifies(m)
    assert count_ops(m, "compgen.quant.weight_int4pack_mm") == 1


def test_build_quantized_matmul_rejects_bad_bits():
    with pytest.raises(ValueError):
        build_quantized_matmul_module(bits=6)


# --- Counting / listing ------------------------------------------------------


def test_count_ops_handles_zero():
    m = build_linalg_matmul_module()
    assert count_ops(m, "does.not.exist") == 0


def test_assert_op_count_passes_and_fails():
    m = build_linalg_matmul_module()
    assert_op_count(m, "linalg.matmul", 1)
    with pytest.raises(AssertionError):
        assert_op_count(m, "linalg.matmul", 5)


def test_all_ops_includes_tensor_empty():
    m = build_linalg_matmul_module()
    names = all_ops(m)
    assert "tensor.empty" in names


def test_find_op_by_region_id_returns_none_when_missing():
    m = build_linalg_matmul_module(region_id="mm_0")
    assert find_op_by_region_id(m, "ghost") is None


# --- Applicator --------------------------------------------------------------


def test_apply_pattern_noop_when_no_match():
    class _NoMatch(RewritePattern):
        def match_and_rewrite(self, op, rewriter):
            pass

    m = build_linalg_matmul_module()
    before_names = all_ops(m)
    apply_pattern(m, _NoMatch())
    after_names = all_ops(m)
    assert before_names == after_names


def test_apply_pattern_can_replace_ops():
    """Confirm the applicator actually mutates a module when the pattern fires."""

    class _ErasesAllMatmuls(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter):
            # Replace linalg.matmul with its ``outs`` operand so the module
            # still verifies but has no matmul op.
            rewriter.replace_matched_op([], new_results=[op.outputs[0]])

    m = build_linalg_matmul_module()
    apply_pattern(m, _ErasesAllMatmuls())
    assert_module_verifies(m)
    assert count_ops(m, "linalg.matmul") == 0


# --- SMT equivalence wrapper -------------------------------------------------


def test_assert_smt_equivalent_identical_module_passes():
    m = build_linalg_matmul_module()
    # Structural identity (same ModuleOp) -> fast-path ``valid``.
    assert_smt_equivalent(m, m)


def test_assert_smt_equivalent_two_module_instances_with_identical_text_pass():
    m1 = build_linalg_matmul_module()
    m2 = build_linalg_matmul_module()
    # The pretty-printed text matches exactly -> translation_validation
    # takes its text-equality fast path.
    assert_smt_equivalent(m1, m2)
