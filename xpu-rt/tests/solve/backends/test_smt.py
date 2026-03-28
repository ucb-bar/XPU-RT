"""Tests for solve/backends/smt.py -- SMT solver backend."""

from __future__ import annotations

import pytest
from compgen.solve.backends.smt import SMTSolver


def test_smt_solver_defaults() -> None:
    solver = SMTSolver()
    assert solver.timeout_ms == 30000


def test_smt_solver_custom_timeout() -> None:
    solver = SMTSolver(timeout_ms=10000)
    assert solver.timeout_ms == 10000


def test_smt_check_sat() -> None:
    """SMTSolver.check_sat should return 'sat', 'unsat', or 'unknown'."""
    z3 = pytest.importorskip("z3")

    solver = SMTSolver(timeout_ms=5000)

    # Satisfiable: x > 0
    x = z3.Int("x")
    result = solver.check_sat(x > 0)
    assert result == "sat"

    # Unsatisfiable: x > 0 AND x < 0
    result = solver.check_sat(z3.And(x > 0, x < 0))
    assert result == "unsat"


def test_smt_prove() -> None:
    """SMTSolver.prove should return 'valid', 'invalid', or 'unknown'."""
    z3 = pytest.importorskip("z3")

    solver = SMTSolver(timeout_ms=5000)

    # Tautology: x == x is always valid
    x = z3.Int("x")
    result = solver.prove(x == x)
    assert result == "valid"

    # Not a tautology: x > 0 is not always true
    result = solver.prove(x > 0)
    assert result == "invalid"
