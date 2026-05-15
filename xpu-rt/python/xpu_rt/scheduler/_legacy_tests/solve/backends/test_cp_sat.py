"""Tests for CP-SAT solver backend."""

from __future__ import annotations

from compgen.solve.backends.cp_sat import CPSatResult, CPSatSolver
from compgen.solve.contracts import SolverProblem
from compgen.solve.partition import Partition


def test_cp_sat_instantiation() -> None:
    solver = CPSatSolver(timeout_ms=5000)
    assert solver.timeout_ms == 5000


def test_cp_sat_solve_empty() -> None:
    """CPSatSolver.solve returns a feasible empty result for 0 partitions."""
    solver = CPSatSolver(timeout_ms=5000)
    problem = SolverProblem(partitions=[], device_capacities={0: 1024, 1: 1024})
    result = solver.solve(problem)
    assert isinstance(result, CPSatResult)
    assert result.feasible is True


def test_cp_sat_solve() -> None:
    """CPSatSolver.solve returns a feasible solution for 2 partitions on 2 devices."""
    solver = CPSatSolver(timeout_ms=5000)

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

    assert isinstance(result, CPSatResult)
    assert result.feasible is True
    assert result.placement is not None
    assert result.placement.feasible is True
    assert result.schedule is not None
    assert result.schedule.feasible is True
    assert result.memory is not None
    assert result.memory.feasible is True
    assert result.solve_time_ms > 0.0

    # Both partitions should be assigned
    assert "p_0" in result.placement.assignments
    assert "p_1" in result.placement.assignments
