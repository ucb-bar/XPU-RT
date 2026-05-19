"""Joint CP-SAT placement + ordering scheduler.

Unlike :mod:`xpu_rt.solve.schedule`, which takes a fixed device assignment
as input and only orders the resulting intervals, this module solves the
combined problem: it picks both the device for each partition and the
start time, minimising end-to-end makespan.

The model uses one optional interval per ``(partition, device)`` pair,
with presence linked to a Boolean ``assign[p, d]``. Cross-device
dependency arcs are activated only when both endpoints' presence
literals are true, so the transfer penalty is paid exactly once per
realised arc.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JointScheduleSolution:
    """Solution returned by :func:`solve_schedule_joint`.

    Attributes:
        start_times: Realised start time per partition (microseconds).
        end_times: Realised end time per partition (microseconds).
        device_assignments: Chosen device index per partition.
        makespan_us: Total makespan in microseconds.
        feasible: Whether a feasible (or optimal) assignment was found.
        solve_time_ms: Wall-clock solver time.
        status: One of ``"optimal" | "feasible" | "infeasible" | "timeout" | "model_invalid"``.
    """

    start_times: dict[str, float] = field(default_factory=dict)
    end_times: dict[str, float] = field(default_factory=dict)
    device_assignments: dict[str, int] = field(default_factory=dict)
    makespan_us: float = float("inf")
    feasible: bool = False
    solve_time_ms: float = 0.0
    status: str = "infeasible"


def _is_feasible_duration(value: float | None) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return False
    return value >= 0.0


def solve_schedule_joint(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float | None]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    transfer_us: list[list[float]] | None = None,
    timeout_ms: int = 30000,
) -> JointScheduleSolution:
    """Solve joint placement + ordering minimising makespan.

    Args:
        partition_ids: Stable ordering of partitions to schedule.
        durations_us_by_device: ``partition_id -> [duration on device 0, ...]``.
            Entries that are ``None`` or ``inf`` mark the partition as
            infeasible on that device and are skipped.
        dependencies: ``partition_id -> [predecessor_id, ...]``.
        num_devices: Number of devices in the topology.
        transfer_us: Optional ``num_devices x num_devices`` transfer
            penalty matrix in microseconds (zero on diagonal). Defaults
            to all-zero.
        timeout_ms: Hard solver timeout.

    Returns:
        A :class:`JointScheduleSolution`. On infeasibility / timeout the
        ``feasible`` flag is ``False`` and ``status`` distinguishes the
        cause.
    """
    from ortools.sat.python import cp_model

    if not partition_ids:
        return JointScheduleSolution(feasible=True, makespan_us=0.0, status="optimal")

    if num_devices <= 0:
        return JointScheduleSolution(status="model_invalid")

    t0 = time.perf_counter()
    scale = 1000  # microseconds → integer solver units

    if transfer_us is None:
        transfer_us = [[0.0] * num_devices for _ in range(num_devices)]

    feasible_dev: dict[str, list[int]] = {}
    int_durations: dict[tuple[str, int], int] = {}
    for pid in partition_ids:
        per_dev = durations_us_by_device.get(pid, [])
        if len(per_dev) != num_devices:
            return JointScheduleSolution(
                status="model_invalid",
                solve_time_ms=(time.perf_counter() - t0) * 1000,
            )
        feas: list[int] = []
        for d, val in enumerate(per_dev):
            if not _is_feasible_duration(val):
                continue
            dur_units = max(int(round(float(val) * scale)), 1)
            int_durations[(pid, d)] = dur_units
            feas.append(d)
        if not feas:
            return JointScheduleSolution(
                status="infeasible",
                solve_time_ms=(time.perf_counter() - t0) * 1000,
            )
        feasible_dev[pid] = feas

    horizon_us = sum(
        max((durations_us_by_device[pid][d] or 0.0) for d in feasible_dev[pid])
        for pid in partition_ids
    )
    horizon_us += sum(max(row) for row in transfer_us) * len(partition_ids)
    horizon = int(round(horizon_us * scale)) + 1

    model = cp_model.CpModel()

    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    presence: dict[tuple[str, int], cp_model.IntVar] = {}
    intervals_by_device: dict[int, list[cp_model.IntervalVar]] = {d: [] for d in range(num_devices)}
    end_vars_per_pd: dict[tuple[str, int], cp_model.IntVar] = {}

    for pid in partition_ids:
        starts[pid] = model.new_int_var(0, horizon, f"start_{pid}")
        ends[pid] = model.new_int_var(0, horizon, f"end_{pid}")
        present_lits: list[cp_model.IntVar] = []
        for d in feasible_dev[pid]:
            presence[(pid, d)] = model.new_bool_var(f"assign_{pid}_d{d}")
            present_lits.append(presence[(pid, d)])
            dur = int_durations[(pid, d)]
            start_pd = model.new_int_var(0, horizon, f"start_{pid}_d{d}")
            end_pd = model.new_int_var(0, horizon, f"end_{pid}_d{d}")
            interval = model.new_optional_interval_var(
                start_pd, dur, end_pd, presence[(pid, d)], f"iv_{pid}_d{d}"
            )
            intervals_by_device[d].append(interval)
            end_vars_per_pd[(pid, d)] = end_pd
            model.add(start_pd == starts[pid]).only_enforce_if(presence[(pid, d)])
            model.add(end_pd == ends[pid]).only_enforce_if(presence[(pid, d)])
        model.add_exactly_one(present_lits)

    for d, ivs in intervals_by_device.items():
        if len(ivs) > 1:
            model.add_no_overlap(ivs)

    deps = dependencies or {}
    for succ in partition_ids:
        for pred in deps.get(succ, []):
            if pred not in starts:
                continue
            for dp in feasible_dev[pred]:
                for ds in feasible_dev[succ]:
                    transfer_units = int(round(float(transfer_us[dp][ds]) * scale))
                    model.add(starts[succ] >= ends[pred] + transfer_units).only_enforce_if(
                        [presence[(pred, dp)], presence[(succ, ds)]]
                    )

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[pid] for pid in partition_ids])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(timeout_ms / 1000.0, 0.001)
    status_int = solver.solve(model)
    solve_time_ms = (time.perf_counter() - t0) * 1000

    if status_int == cp_model.OPTIMAL:
        status = "optimal"
    elif status_int == cp_model.FEASIBLE:
        status = "feasible"
    elif status_int == cp_model.INFEASIBLE:
        return JointScheduleSolution(
            feasible=False, status="infeasible", solve_time_ms=solve_time_ms
        )
    else:
        return JointScheduleSolution(
            feasible=False, status="timeout", solve_time_ms=solve_time_ms
        )

    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    device_assignments: dict[str, int] = {}
    for pid in partition_ids:
        start_times[pid] = solver.value(starts[pid]) / scale
        end_times[pid] = solver.value(ends[pid]) / scale
        chosen = -1
        for d in feasible_dev[pid]:
            if solver.value(presence[(pid, d)]) == 1:
                chosen = d
                break
        device_assignments[pid] = chosen

    return JointScheduleSolution(
        start_times=start_times,
        end_times=end_times,
        device_assignments=device_assignments,
        makespan_us=solver.value(makespan) / scale,
        feasible=True,
        solve_time_ms=solve_time_ms,
        status=status,
    )


__all__ = ["JointScheduleSolution", "solve_schedule_joint"]
