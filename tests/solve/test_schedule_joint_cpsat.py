"""Tests for the joint CP-SAT placement+ordering scheduler."""

from __future__ import annotations

from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint


def test_two_op_chain_two_devices_picks_cheaper() -> None:
    """Single chain on two devices: cheaper device gets both ops, transfer avoided."""
    sol = solve_schedule_joint(
        partition_ids=["a", "b"],
        durations_us_by_device={"a": [10.0, 5.0], "b": [10.0, 5.0]},
        dependencies={"b": ["a"]},
        num_devices=2,
        transfer_us=[[0.0, 1.0], [1.0, 0.0]],
        timeout_ms=5000,
    )
    assert sol.feasible
    assert sol.device_assignments["a"] == 1
    assert sol.device_assignments["b"] == 1
    assert abs(sol.makespan_us - 10.0) < 0.05


def test_parallel_branches_use_both_devices() -> None:
    """Two independent ops with equal cost should run in parallel, not stack."""
    sol = solve_schedule_joint(
        partition_ids=["a", "b"],
        durations_us_by_device={"a": [10.0, 10.0], "b": [10.0, 10.0]},
        dependencies={},
        num_devices=2,
        transfer_us=[[0.0, 0.0], [0.0, 0.0]],
        timeout_ms=5000,
    )
    assert sol.feasible
    assert sol.device_assignments["a"] != sol.device_assignments["b"]
    assert abs(sol.makespan_us - 10.0) < 0.05


def test_transfer_penalty_keeps_chain_on_one_device() -> None:
    """Punitive transfer cost should keep dependent ops co-located."""
    sol = solve_schedule_joint(
        partition_ids=["a", "b"],
        durations_us_by_device={"a": [10.0, 9.0], "b": [10.0, 9.0]},
        dependencies={"b": ["a"]},
        num_devices=2,
        transfer_us=[[0.0, 1000.0], [1000.0, 0.0]],
        timeout_ms=5000,
    )
    assert sol.feasible
    assert sol.device_assignments["a"] == sol.device_assignments["b"]


def test_infeasible_op_skipped_on_first_device() -> None:
    """An op marked infeasible (None) on device 0 must execute on device 1."""
    sol = solve_schedule_joint(
        partition_ids=["a", "b"],
        durations_us_by_device={"a": [None, 7.0], "b": [5.0, 5.0]},
        dependencies={},
        num_devices=2,
        transfer_us=[[0.0, 0.0], [0.0, 0.0]],
        timeout_ms=5000,
    )
    assert sol.feasible
    assert sol.device_assignments["a"] == 1


def test_timeout_returns_status_field() -> None:
    """Aggressive timeout on a trivial workload must not crash."""
    sol = solve_schedule_joint(
        partition_ids=["a"],
        durations_us_by_device={"a": [1.0]},
        dependencies={},
        num_devices=1,
        timeout_ms=1,
    )
    assert sol.status in {"optimal", "feasible", "timeout"}


def test_empty_workload_returns_zero_makespan() -> None:
    sol = solve_schedule_joint(
        partition_ids=[],
        durations_us_by_device={},
        dependencies={},
        num_devices=2,
    )
    assert sol.feasible
    assert sol.makespan_us == 0.0
