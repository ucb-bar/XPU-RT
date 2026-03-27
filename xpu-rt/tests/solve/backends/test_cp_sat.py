"""Tests for CP-SAT solver backend."""

from __future__ import annotations

import pytest
from compgen.solve.backends.cp_sat import CPSatSolver


def test_cp_sat_instantiation() -> None:
    solver = CPSatSolver(timeout_ms=5000)
    assert solver.timeout_ms == 5000


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_cp_sat_solve() -> None:
    """CPSatSolver.solve should return a solution."""
