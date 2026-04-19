"""Tests for W1.2 ``fold_transposes_into_dots``."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    DenseArrayBase,
    Float32Type,
    FunctionType,
    ModuleOp,
    TensorType,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.ir.payload.passes.rewrites.fold_transposes_into_dots import (
    FoldTransposesStats,
    run_fold_transposes_into_dots,
)
from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    assert_op_count,
    count_ops,
)


def _f32(shape):
    return TensorType(Float32Type(), list(shape))


def _perm(values):
    return DenseArrayBase.from_list(i64, values)


def _wrap(ops, return_value, ret_type):
    block = Block()
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(return_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ret_type]), Region([block]))
    return ModuleOp([func])


# --- matmul(transpose(a), b) --------------------------------------------------


def _build_transpose_a_module(M=4, K=8, N=16) -> ModuleOp:
    a_src = EmptyOp([], _f32([K, M]))
    a_init = EmptyOp([], _f32([M, K]))
    tr = TransposeOp(
        input=a_src.results[0],
        init=a_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([M, K]),
    )
    b = EmptyOp([], _f32([K, N]))
    out = EmptyOp([], _f32([M, N]))
    mm = MatmulOp(
        inputs=[tr.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_f32([M, N])],
    )
    return _wrap([a_src, a_init, tr, b, out, mm], mm.res[0], _f32([M, N]))


def test_lhs_transpose_folds():
    m = _build_transpose_a_module()
    stats = run_fold_transposes_into_dots(m)
    assert stats.transpose_a_folds == 1
    assert stats.transpose_b_folds == 0
    assert_module_verifies(m)


def test_lhs_transpose_produces_correct_indexing_maps():
    m = _build_transpose_a_module()
    run_fold_transposes_into_dots(m)
    mm_ops = [op for op in m.walk() if op.name == "linalg.matmul"]
    assert len(mm_ops) == 1
    maps = mm_ops[0].properties["indexing_maps"]
    # First map should read (k, i), not (i, k).
    assert "d2, d0" in str(maps.data[0]) or "d2, d0)" in str(maps.data[0])


def test_lhs_transpose_matmul_input_becomes_untransposed_a():
    m = _build_transpose_a_module(M=4, K=8, N=16)
    run_fold_transposes_into_dots(m)
    mm_ops = [op for op in m.walk() if op.name == "linalg.matmul"]
    assert len(mm_ops) == 1
    # The matmul's lhs should now be the transpose-op's SOURCE,
    # i.e. tensor<8x4xf32>, not tensor<4x8xf32>.
    assert list(mm_ops[0].inputs[0].type.get_shape()) == [8, 4]


# --- matmul(a, transpose(b)) --------------------------------------------------


def _build_transpose_b_module(M=4, K=8, N=16) -> ModuleOp:
    a = EmptyOp([], _f32([M, K]))
    b_src = EmptyOp([], _f32([N, K]))
    b_init = EmptyOp([], _f32([K, N]))
    tr = TransposeOp(
        input=b_src.results[0],
        init=b_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([K, N]),
    )
    out = EmptyOp([], _f32([M, N]))
    mm = MatmulOp(
        inputs=[a.results[0], tr.results[0]],
        outputs=[out.results[0]],
        res=[_f32([M, N])],
    )
    return _wrap([a, b_src, b_init, tr, out, mm], mm.res[0], _f32([M, N]))


def test_rhs_transpose_folds():
    m = _build_transpose_b_module()
    stats = run_fold_transposes_into_dots(m)
    assert stats.transpose_a_folds == 0
    assert stats.transpose_b_folds == 1
    assert_module_verifies(m)


# --- both sides transposed ---------------------------------------------------


def _build_both_transposes_module(M=4, K=8, N=16) -> ModuleOp:
    a_src = EmptyOp([], _f32([K, M]))
    a_init = EmptyOp([], _f32([M, K]))
    tr_a = TransposeOp(
        input=a_src.results[0],
        init=a_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([M, K]),
    )
    b_src = EmptyOp([], _f32([N, K]))
    b_init = EmptyOp([], _f32([K, N]))
    tr_b = TransposeOp(
        input=b_src.results[0],
        init=b_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([K, N]),
    )
    out = EmptyOp([], _f32([M, N]))
    mm = MatmulOp(
        inputs=[tr_a.results[0], tr_b.results[0]],
        outputs=[out.results[0]],
        res=[_f32([M, N])],
    )
    ops = [a_src, a_init, tr_a, b_src, b_init, tr_b, out, mm]
    return _wrap(ops, mm.res[0], _f32([M, N]))


def test_both_sides_transposed_folds_once():
    m = _build_both_transposes_module()
    stats = run_fold_transposes_into_dots(m)
    assert stats.transpose_both_folds == 1
    assert stats.transpose_a_folds == 0
    assert stats.transpose_b_folds == 0
    assert_module_verifies(m)


# --- non-matching cases ------------------------------------------------------


def test_plain_matmul_is_not_touched():
    a = EmptyOp([], _f32([4, 8]))
    b = EmptyOp([], _f32([8, 16]))
    out = EmptyOp([], _f32([4, 16]))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_f32([4, 16])],
    )
    m = _wrap([a, b, out, mm], mm.res[0], _f32([4, 16]))

    stats = run_fold_transposes_into_dots(m)
    assert stats.transpose_a_folds == 0
    assert stats.transpose_b_folds == 0


def test_transpose_with_multiple_uses_is_not_folded():
    # The transpose feeds both a matmul and a return -> folding
    # would remove a still-needed value. The pattern must skip.
    a_src = EmptyOp([], _f32([8, 4]))
    a_init = EmptyOp([], _f32([4, 8]))
    tr = TransposeOp(
        input=a_src.results[0],
        init=a_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([4, 8]),
    )
    b = EmptyOp([], _f32([8, 16]))
    out = EmptyOp([], _f32([4, 16]))
    mm = MatmulOp(
        inputs=[tr.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_f32([4, 16])],
    )
    # Returns the transpose so it has TWO users (matmul + return).
    m = _wrap([a_src, a_init, tr, b, out, mm], tr.results[0], _f32([4, 8]))

    stats = run_fold_transposes_into_dots(m)
    assert stats.transpose_a_folds == 0
    assert count_ops(m, "linalg.transpose") == 1
    assert count_ops(m, "linalg.matmul") == 1


# --- Double transpose elimination --------------------------------------------


def test_double_transpose_collapses_to_source():
    # transpose(transpose(x, [1, 0]), [1, 0]) -> x
    x = EmptyOp([], _f32([8, 4]))
    inner_init = EmptyOp([], _f32([4, 8]))
    inner = TransposeOp(
        input=x.results[0],
        init=inner_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([4, 8]),
    )
    outer_init = EmptyOp([], _f32([8, 4]))
    outer = TransposeOp(
        input=inner.results[0],
        init=outer_init.results[0],
        permutation=_perm([1, 0]),
        result=_f32([8, 4]),
    )
    m = _wrap([x, inner_init, inner, outer_init, outer], outer.results[0], _f32([8, 4]))

    stats = run_fold_transposes_into_dots(m)
    assert stats.double_transpose_eliminations >= 1
    assert_module_verifies(m)


def test_non_inverse_double_transpose_is_not_collapsed():
    # 3D: outer uses perm [0, 1, 2] (identity) -- but inner uses [2, 0, 1].
    # Composed identity ∘ [2, 0, 1] = [2, 0, 1] != identity -> do not fold.
    x = EmptyOp([], _f32([2, 3, 4]))
    inner_init = EmptyOp([], _f32([4, 2, 3]))
    inner = TransposeOp(
        input=x.results[0],
        init=inner_init.results[0],
        permutation=_perm([2, 0, 1]),
        result=_f32([4, 2, 3]),
    )
    outer_init = EmptyOp([], _f32([4, 2, 3]))
    outer = TransposeOp(
        input=inner.results[0],
        init=outer_init.results[0],
        permutation=_perm([0, 1, 2]),
        result=_f32([4, 2, 3]),
    )
    m = _wrap([x, inner_init, inner, outer_init, outer], outer.results[0], _f32([4, 2, 3]))

    stats = run_fold_transposes_into_dots(m)
    assert stats.double_transpose_eliminations == 0


# --- stats + idempotence -----------------------------------------------------


def test_stats_initial_values():
    s = FoldTransposesStats()
    assert s.matmuls_seen == 0
    assert s.transpose_a_folds == 0


def test_idempotent_second_run_is_noop():
    m = _build_transpose_a_module()
    first = run_fold_transposes_into_dots(m)
    assert first.transpose_a_folds == 1
    second = run_fold_transposes_into_dots(m)
    assert second.transpose_a_folds == 0


def test_region_id_and_pattern_hint_preserved_across_fold():
    from xdsl.dialects.builtin import StringAttr

    m = _build_transpose_a_module()
    mm_ops = [op for op in m.walk() if op.name == "linalg.matmul"]
    mm_ops[0].attributes["compgen.region_id"] = StringAttr("mm_xyz")
    mm_ops[0].attributes["compgen._pattern_hint"] = StringAttr("matmul_hint")

    run_fold_transposes_into_dots(m)
    mm_ops_after = [op for op in m.walk() if op.name == "linalg.matmul"]
    assert len(mm_ops_after) == 1
    assert mm_ops_after[0].attributes["compgen.region_id"].data == "mm_xyz"
    assert mm_ops_after[0].attributes["compgen._pattern_hint"].data == "matmul_hint"
