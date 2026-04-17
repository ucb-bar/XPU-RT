"""Tests for the synthesized guard DSL."""

from __future__ import annotations

from compgen.semantic.synthesis.guard_lang import (
    Cmp,
    CmpOp,
    Const,
    ModEq,
    Var,
    and_,
    eval_guard,
    expr_from_json,
    expr_to_json,
    or_,
)


def test_eval_guard_conjunction() -> None:
    expr = and_(
        Cmp(CmpOp.EQ, Var("backend_triton"), Const(True)),
        Cmp(CmpOp.GE, Var("estimated_flops"), Const(1024)),
    )
    assert eval_guard(expr, {"backend_triton": True, "estimated_flops": 2048}) is True
    assert eval_guard(expr, {"backend_triton": False, "estimated_flops": 2048}) is False


def test_eval_guard_mod_eq() -> None:
    expr = ModEq(Var("tile_k"), 32, 0)
    assert eval_guard(expr, {"tile_k": 64}) is True
    assert eval_guard(expr, {"tile_k": 48}) is False


def test_expr_json_round_trip() -> None:
    expr = or_(
        Cmp(CmpOp.EQ, Var("backend_triton"), Const(True)),
        ModEq(Var("M"), 64, 0),
    )
    restored = expr_from_json(expr_to_json(expr))
    assert eval_guard(restored, {"backend_triton": False, "M": 128}) is True
