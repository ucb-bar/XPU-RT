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


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_smt_check_sat() -> None:
    """SMTSolver.check_sat should return 'sat', 'unsat', or 'unknown'."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_smt_prove() -> None:
    """SMTSolver.prove should return 'valid', 'invalid', or 'unknown'."""
