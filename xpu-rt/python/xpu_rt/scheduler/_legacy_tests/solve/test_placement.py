"""Tests for CP-SAT placement solver."""

from __future__ import annotations

from compgen.solve.partition import Partition
from compgen.solve.placement import PlacementConstraint, PlacementSolution, solve_placement


def test_placement_solution_defaults() -> None:
    sol = PlacementSolution()
    assert sol.feasible is False
    assert sol.objective_value == float("inf")


def test_placement_constraint() -> None:
    c = PlacementConstraint(partition_id="p0", allowed_devices={0, 1}, reason="GPU only")
    assert 1 in c.allowed_devices


def test_solve_empty_partitions() -> None:
    """Empty partition list should return feasible with zero cost."""
    sol = solve_placement([], num_devices=2)
    assert sol.feasible
    assert sol.objective_value == 0.0


def test_solve_single_partition_two_devices() -> None:
    """Single partition should be placed on the fastest device."""
    partitions = [
        Partition(partition_id="matmul_0", op_names=["matmul_0"], estimated_cost_us=100.0, memory_bytes=4096),
    ]
    sol = solve_placement(
        partitions,
        num_devices=2,
        device_compute_rates=[1.0, 10.0],  # device 1 is 10x faster
    )
    assert sol.feasible
    assert sol.assignments["matmul_0"] == 1  # should go to faster device


def test_solve_respects_forced_constraint() -> None:
    """Force constraint should be respected."""
    partitions = [
        Partition(partition_id="matmul_0", op_names=["matmul_0"], estimated_cost_us=100.0, memory_bytes=4096),
    ]
    constraints = [PlacementConstraint(partition_id="matmul_0", required_device=0)]
    sol = solve_placement(
        partitions,
        num_devices=2,
        device_compute_rates=[1.0, 10.0],
        constraints=constraints,
    )
    assert sol.feasible
    assert sol.assignments["matmul_0"] == 0  # forced to device 0


def test_solve_memory_constraint() -> None:
    """Partitions should not exceed device memory."""
    partitions = [
        Partition(partition_id="p0", op_names=["p0"], estimated_cost_us=10.0, memory_bytes=8000),
        Partition(partition_id="p1", op_names=["p1"], estimated_cost_us=10.0, memory_bytes=8000),
    ]
    sol = solve_placement(
        partitions,
        num_devices=2,
        device_memory_caps=[10000, 10000],  # each can hold only 1 partition
    )
    assert sol.feasible
    # Partitions should be on different devices
    assert sol.assignments["p0"] != sol.assignments["p1"]


def test_solve_reports_time() -> None:
    """Solver should report solve time."""
    partitions = [
        Partition(partition_id="p0", op_names=["p0"], estimated_cost_us=10.0, memory_bytes=100),
    ]
    sol = solve_placement(partitions, num_devices=2)
    assert sol.feasible
    assert sol.solve_time_ms > 0
