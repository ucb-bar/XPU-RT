"""Tests for CP-SAT scheduling solver."""

from __future__ import annotations

from compgen.solve.schedule import ScheduleConstraint, ScheduleSolution, solve_schedule


def test_schedule_solution_defaults() -> None:
    sol = ScheduleSolution()
    assert sol.feasible is False


def test_solve_empty() -> None:
    sol = solve_schedule([], {}, {}, {}, num_devices=1)
    assert sol.feasible
    assert sol.makespan_us == 0.0


def test_solve_two_independent_tasks_one_device() -> None:
    """Two tasks on one device should run sequentially."""
    sol = solve_schedule(
        partition_ids=["a", "b"],
        durations_us={"a": 10.0, "b": 20.0},
        device_assignments={"a": 0, "b": 0},
        dependencies={},
        num_devices=1,
    )
    assert sol.feasible
    assert sol.makespan_us >= 30.0  # must be sequential


def test_solve_two_tasks_two_devices_parallel() -> None:
    """Two independent tasks on different devices should run in parallel."""
    sol = solve_schedule(
        partition_ids=["a", "b"],
        durations_us={"a": 10.0, "b": 10.0},
        device_assignments={"a": 0, "b": 1},
        dependencies={},
        num_devices=2,
    )
    assert sol.feasible
    # Parallel: makespan should be ~10, not 20
    assert sol.makespan_us <= 11.0


def test_solve_dependency_ordering() -> None:
    """Dependent tasks should respect precedence."""
    sol = solve_schedule(
        partition_ids=["a", "b"],
        durations_us={"a": 10.0, "b": 5.0},
        device_assignments={"a": 0, "b": 0},
        dependencies={"b": ["a"]},  # b depends on a
        num_devices=1,
    )
    assert sol.feasible
    assert sol.start_times["b"] >= sol.end_times["a"] - 0.001


def test_solve_deadline_constraint() -> None:
    sol = solve_schedule(
        partition_ids=["a"],
        durations_us={"a": 10.0},
        device_assignments={"a": 0},
        dependencies={},
        num_devices=1,
        constraints=[ScheduleConstraint(partition_id="a", deadline_us=100.0)],
    )
    assert sol.feasible
    assert sol.end_times["a"] <= 100.0


def test_solve_reports_time() -> None:
    sol = solve_schedule(
        partition_ids=["a"],
        durations_us={"a": 5.0},
        device_assignments={"a": 0},
        dependencies={},
        num_devices=1,
    )
    assert sol.feasible
    assert sol.solve_time_ms > 0
