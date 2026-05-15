"""Tests for cost_model_gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from compgen.agent.gates import cost_model_gate


@dataclass
class _FakePlacement:
    feasible: bool = True
    objective_value: float = 0.0


@dataclass
class _FakeSolveResult:
    feasible: bool = True
    solve_time_ms: float = 1.0
    placement: Any = field(default_factory=_FakePlacement)


class _FakeSolver:
    def __init__(self, result: _FakeSolveResult) -> None:
        self._result = result

    def solve(self, problem: Any) -> _FakeSolveResult:
        return self._result


def test_deferred_without_problem_or_module() -> None:
    r = cost_model_gate({})
    assert r["status"] == "deferred"


def test_deferred_with_module_but_no_target() -> None:
    # Only module, no target → still deferred
    r = cost_model_gate({}, module=object())
    assert r["status"] == "deferred"


def test_accepts_when_solver_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    from compgen.agent.gates import cost_model as cm

    fake = _FakeSolveResult(feasible=True, placement=_FakePlacement(feasible=True, objective_value=42.0))
    # Patch CPSatSolver to our fake
    monkeypatch.setattr(cm, "__name__", cm.__name__)  # no-op

    # Inject fake via the problem path (no extract_solver_problem needed)
    def _patched_solver_class(*a: Any, **k: Any) -> _FakeSolver:
        return _FakeSolver(fake)

    monkeypatch.setattr("compgen.solve.backends.cp_sat.CPSatSolver", _patched_solver_class)

    r = cost_model_gate({}, problem=object())
    assert r["status"] == "accepted"
    assert r["details"]["feasible"] is True
    assert r["details"]["objective_value"] == 42.0


def test_rejects_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSolveResult(feasible=False)

    def _patched(*a: Any, **k: Any) -> _FakeSolver:
        return _FakeSolver(fake)

    monkeypatch.setattr("compgen.solve.backends.cp_sat.CPSatSolver", _patched)
    r = cost_model_gate({}, problem=object())
    assert r["status"] == "rejected"


def test_rejects_on_baseline_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSolveResult(
        feasible=True,
        placement=_FakePlacement(feasible=True, objective_value=200.0),
    )
    monkeypatch.setattr(
        "compgen.solve.backends.cp_sat.CPSatSolver",
        lambda *a, **k: _FakeSolver(fake),
    )

    r = cost_model_gate({}, problem=object(), baseline_cost=100.0, tolerance_ratio=1.0)
    assert r["status"] == "rejected"
    assert "regressed" in r["details"]["reason"]


def test_accepts_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSolveResult(
        feasible=True,
        placement=_FakePlacement(feasible=True, objective_value=110.0),
    )
    monkeypatch.setattr(
        "compgen.solve.backends.cp_sat.CPSatSolver",
        lambda *a, **k: _FakeSolver(fake),
    )

    r = cost_model_gate({}, problem=object(), baseline_cost=100.0, tolerance_ratio=1.2)
    assert r["status"] == "accepted"
