"""Tests for solve/backends/milp.py -- MILP solver backend."""

from __future__ import annotations

import pytest
from compgen.solve.backends.milp import MILPSolver


def test_milp_solver_defaults() -> None:
    solver = MILPSolver()
    assert solver.timeout_ms == 30000
    assert solver.gap_tolerance == 0.01


def test_milp_solver_custom_params() -> None:
    solver = MILPSolver(timeout_ms=5000, gap_tolerance=0.05)
    assert solver.timeout_ms == 5000
    assert solver.gap_tolerance == 0.05


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_milp_solver_solve() -> None:
    """MILPSolver.solve should return a solution for a MILP problem."""
