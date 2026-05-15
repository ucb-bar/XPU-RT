"""Tests for solve/backends/milp.py -- MILP solver backend."""

from __future__ import annotations

from compgen.solve.backends.milp import MILPResult, MILPSolver
from compgen.solve.contracts import SolverProblem
from compgen.solve.partition import Partition


def test_milp_solver_defaults() -> None:
    solver = MILPSolver()
    assert solver.timeout_ms == 30000
    assert solver.gap_tolerance == 0.01


def test_milp_solver_custom_params() -> None:
    solver = MILPSolver(timeout_ms=5000, gap_tolerance=0.05)
    assert solver.timeout_ms == 5000
    assert solver.gap_tolerance == 0.05


def test_milp_solver_solve_empty() -> None:
    """MILPSolver.solve returns a feasible empty result for 0 partitions."""
    solver = MILPSolver(timeout_ms=5000)
    problem = SolverProblem(partitions=[], device_capacities={0: 1024, 1: 1024})
    result = solver.solve(problem)
    assert isinstance(result, MILPResult)
    assert result.feasible is True
    assert result.objective_value == 0.0


def test_milp_solver_solve() -> None:
    """MILPSolver.solve returns a feasible solution for 2 partitions on 2 devices."""
    solver = MILPSolver(timeout_ms=5000)

    p0 = Partition(partition_id="p_0", op_names=["matmul"], estimated_cost_us=10.0, memory_bytes=1024)
    p1 = Partition(
        partition_id="p_1",
        op_names=["relu"],
        dependencies=["p_0"],
        estimated_cost_us=2.0,
        memory_bytes=512,
    )

    problem = SolverProblem(
        partitions=[p0, p1],
        device_capacities={0: 4096, 1: 4096},
        target_name="test_target",
    )

    result = solver.solve(problem)

    assert isinstance(result, MILPResult)
    assert result.feasible is True
    assert result.solve_time_ms > 0.0
    assert result.objective_value < float("inf")

    # Both partitions should be assigned
    assert "p_0" in result.placement
    assert "p_1" in result.placement

    # Assignments should be valid device indices
    for pid, dev in result.placement.items():
        assert dev in (0, 1), f"Partition {pid} assigned to invalid device {dev}"
