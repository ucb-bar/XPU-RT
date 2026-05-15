"""Temporal scheduling via CP-SAT interval variables.

Given a placement (which device each region runs on), solves for start times
minimizing makespan, respecting dependencies and per-device no-overlap.

Uses OR-Tools CP-SAT interval scheduling — the solver handles the
combinatorics of ordering tasks on shared resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScheduleConstraint:
    """A scheduling constraint.

    Attributes:
        partition_id: Partition this constrains.
        earliest_start_us: Earliest allowed start time.
        deadline_us: Latest completion time (None = no deadline).
    """

    partition_id: str
    earliest_start_us: float = 0.0
    deadline_us: float | None = None


@dataclass(frozen=True)
class ScheduleSolution:
    """Solution from the scheduling solver.

    Attributes:
        start_times: Dict mapping partition_id -> start time (microseconds).
        end_times: Dict mapping partition_id -> end time (microseconds).
        makespan_us: Total makespan.
        feasible: Whether a feasible schedule was found.
        deadline_met: Whether all deadlines are met.
        solve_time_ms: Solver wall-clock time.
    """

    start_times: dict[str, float] = field(default_factory=dict)
    end_times: dict[str, float] = field(default_factory=dict)
    makespan_us: float = float("inf")
    feasible: bool = False
    deadline_met: bool = True
    solve_time_ms: float = 0.0


def solve_schedule(
    partition_ids: list[str],
    durations_us: dict[str, float],
    device_assignments: dict[str, int],
    dependencies: dict[str, list[str]],
    num_devices: int,
    constraints: list[ScheduleConstraint] | None = None,
    timeout_ms: int = 10000,
) -> ScheduleSolution:
    """Solve temporal scheduling using CP-SAT interval variables.

    Each partition is an interval task with a fixed duration, assigned to a device.
    Tasks on the same device cannot overlap. Dependencies enforce ordering.

    Args:
        partition_ids: IDs of all partitions to schedule.
        durations_us: Duration per partition in microseconds.
        device_assignments: Device index per partition (from placement solver).
        dependencies: For each partition, list of partitions that must finish first.
        num_devices: Number of devices.
        constraints: Optional scheduling constraints (deadlines, earliest start).
        timeout_ms: Solver timeout.

    Returns:
        ScheduleSolution with start times and makespan.
    """
    import time

    from ortools.sat.python import cp_model

    if not partition_ids:
        return ScheduleSolution(feasible=True, makespan_us=0.0)

    t0 = time.perf_counter()
    scale = 1000  # microseconds → solver units (integer)

    model = cp_model.CpModel()

    # Horizon: sum of all durations (upper bound on makespan)
    max_duration = sum(durations_us.get(pid, 1.0) for pid in partition_ids)
    horizon = int(max_duration * scale) + 1

    # Create interval variables for each partition
    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    intervals: dict[str, cp_model.IntervalVar] = {}

    for pid in partition_ids:
        dur = int(durations_us.get(pid, 1.0) * scale)
        dur = max(dur, 1)  # minimum duration 1 unit

        start = model.new_int_var(0, horizon, f"start_{pid}")
        end = model.new_int_var(0, horizon, f"end_{pid}")
        interval = model.new_interval_var(start, dur, end, f"interval_{pid}")

        starts[pid] = start
        ends[pid] = end
        intervals[pid] = interval

    # Dependencies: predecessor must finish before successor starts
    deps = dependencies or {}
    for pid in partition_ids:
        for dep_id in deps.get(pid, []):
            if dep_id in ends and pid in starts:
                model.add(starts[pid] >= ends[dep_id])

    # Per-device no-overlap: tasks on the same device cannot overlap
    for d in range(num_devices):
        device_intervals = [
            intervals[pid] for pid in partition_ids if device_assignments.get(pid) == d and pid in intervals
        ]
        if len(device_intervals) > 1:
            model.add_no_overlap(device_intervals)

    # Agent constraints
    for c in constraints or []:
        if c.partition_id in starts:
            if c.earliest_start_us > 0:
                model.add(starts[c.partition_id] >= int(c.earliest_start_us * scale))
            if c.deadline_us is not None:
                model.add(ends[c.partition_id] <= int(c.deadline_us * scale))

    # Minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    for pid in partition_ids:
        model.add(makespan >= ends[pid])
    model.minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0

    status = solver.solve(model)
    solve_time = (time.perf_counter() - t0) * 1000

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        start_times = {pid: solver.value(starts[pid]) / scale for pid in partition_ids}
        end_times = {pid: solver.value(ends[pid]) / scale for pid in partition_ids}
        ms = solver.value(makespan) / scale

        return ScheduleSolution(
            start_times=start_times,
            end_times=end_times,
            makespan_us=ms,
            feasible=True,
            deadline_met=True,  # if infeasible due to deadline, status would be INFEASIBLE
            solve_time_ms=solve_time,
        )

    return ScheduleSolution(feasible=False, solve_time_ms=solve_time)


__all__ = ["ScheduleConstraint", "ScheduleSolution", "solve_schedule"]
