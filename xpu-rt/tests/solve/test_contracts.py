"""Tests for solve/contracts.py -- solver problem extraction."""

from __future__ import annotations

import pytest
from compgen.solve.contracts import SolverProblem


def test_solver_problem_defaults() -> None:
    sp = SolverProblem()
    assert sp.partitions == []
    assert sp.placement_constraints == []
    assert sp.schedule_constraints == []
    assert sp.device_capacities == {}
    assert sp.transfer_costs == {}
    assert sp.target_name == ""


def test_solver_problem_with_target_name() -> None:
    sp = SolverProblem(target_name="cuda_a100")
    assert sp.target_name == "cuda_a100"


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_extract_solver_problem() -> None:
    """extract_solver_problem should build a SolverProblem from Recipe IR + target."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_extract_solver_problem_with_cost_data() -> None:
    """extract_solver_problem should incorporate profiled cost data."""
