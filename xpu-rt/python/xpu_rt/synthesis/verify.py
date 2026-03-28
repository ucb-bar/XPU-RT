"""SMT-backed proof checks for synthesized guards."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from compgen.synthesis.guard_lang import (
    Add,
    BoolN,
    BoolOp,
    Cmp,
    CmpOp,
    Const,
    Div,
    Expr,
    ModEq,
    Mul,
    Not,
    Sub,
    Var,
    and_,
)
from compgen.solve.backends.smt import SMTSolver


class SoundnessFormulaSpec(Protocol):
    """Family-specific sufficient-condition proof spec."""

    def build_vars(self) -> dict[str, Any]:
        ...

    def sound_formula(self, vars: dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class GuardProofResult:
    """Outcome of proving a synthesized guard sound."""

    proved_sound: bool
    status: str
    verification_time_ms: float = 0.0
    counterexample: dict[str, Any] | None = None


def lower_expr_to_solver(expr: Expr, vars: dict[str, Any]) -> Any:
    """Lower a guard expression to a Z3 formula."""

    try:
        import z3
    except ImportError as exc:  # pragma: no cover - dependent on optional z3
        raise RuntimeError("z3 is required for SMT lowering") from exc

    if isinstance(expr, Var):
        return vars[expr.name]
    if isinstance(expr, Const):
        value = expr.value
        if isinstance(value, bool):
            return z3.BoolVal(value)
        if isinstance(value, int):
            return z3.IntVal(value)
        if isinstance(value, float):
            return z3.RealVal(value)
        raise TypeError(f"unsupported const type for SMT lowering: {type(value)!r}")
    if isinstance(expr, Add):
        return lower_expr_to_solver(expr.lhs, vars) + lower_expr_to_solver(expr.rhs, vars)
    if isinstance(expr, Sub):
        return lower_expr_to_solver(expr.lhs, vars) - lower_expr_to_solver(expr.rhs, vars)
    if isinstance(expr, Mul):
        return lower_expr_to_solver(expr.lhs, vars) * lower_expr_to_solver(expr.rhs, vars)
    if isinstance(expr, Div):
        return lower_expr_to_solver(expr.lhs, vars) / lower_expr_to_solver(expr.rhs, vars)
    if isinstance(expr, ModEq):
        return z3.Mod(lower_expr_to_solver(expr.lhs, vars), expr.mod) == expr.rem
    if isinstance(expr, Not):
        return z3.Not(lower_expr_to_solver(expr.term, vars))
    if isinstance(expr, BoolN):
        parts = [lower_expr_to_solver(term, vars) for term in expr.terms]
        return z3.And(*parts) if expr.op == BoolOp.AND else z3.Or(*parts)
    if isinstance(expr, Cmp):
        lhs = lower_expr_to_solver(expr.lhs, vars)
        rhs = lower_expr_to_solver(expr.rhs, vars)
        return {
            CmpOp.LT: lhs < rhs,
            CmpOp.LE: lhs <= rhs,
            CmpOp.EQ: lhs == rhs,
            CmpOp.NE: lhs != rhs,
            CmpOp.GE: lhs >= rhs,
            CmpOp.GT: lhs > rhs,
        }[expr.op]
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def prove_guard_soundness(
    fragments: tuple[Expr, ...] | list[Expr],
    spec: SoundnessFormulaSpec,
    *,
    timeout_ms: int = 5000,
) -> GuardProofResult:
    """Prove that the conjunction of fragments implies the family soundness spec."""

    start = time.perf_counter()
    try:
        import z3
    except ImportError:
        return GuardProofResult(
            proved_sound=False,
            status="unavailable",
            verification_time_ms=(time.perf_counter() - start) * 1000,
        )

    vars = spec.build_vars()
    solver = SMTSolver(timeout_ms=timeout_ms)
    guard_expr = and_(*tuple(fragments))
    implication = z3.Implies(lower_expr_to_solver(guard_expr, vars), spec.sound_formula(vars))
    status = solver.prove(implication)
    counterexample = None
    if status == "invalid":
        counterexample = solver.get_model(z3.And(lower_expr_to_solver(guard_expr, vars), z3.Not(spec.sound_formula(vars))))
    return GuardProofResult(
        proved_sound=status == "valid",
        status=status,
        verification_time_ms=(time.perf_counter() - start) * 1000,
        counterexample=counterexample,
    )


__all__ = ["GuardProofResult", "SoundnessFormulaSpec", "lower_expr_to_solver", "prove_guard_soundness"]
