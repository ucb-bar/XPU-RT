"""
Decision-driving metrics for an XPU-RT schedule.

Every scheduler in the registry must produce ``(t, alpha)`` that this module can
consume; new metric fields land here (not in the report) so all callers see the
same dict shape.

Fields that depend on later milestones (memory planner, multi-instance jitter
under periodic execution) are populated as ``None`` until the data is available.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload


# Floating-point slack: a finish time reconstructed as t[i] + duration can
# land a few ulps past an exactly-met deadline. Without this, exact hits are
# reported as misses.
_DEADLINE_EPS = 1e-9


def compute_metrics(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    *,
    scheduler_name: str,
    solver_wall_time_s: Optional[float] = None,
    memory_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the full decision-driving metric set from (workload, t, alpha)."""
    machine_combinations = workload.get_machine_combinations()
    machines = workload.machines
    ops = workload.operations
    n = len(ops)

    durations: List[float] = []
    finish_times: List[float] = []
    nonperiodic_finish: List[float] = []
    deadline_miss = 0
    total_lateness = 0.0
    max_lateness = 0.0
    per_machine_busy: Dict[str, float] = {m: 0.0 for m in machines}
    cross_device_transitions = 0

    op_to_idx = {id(op): i for i, op in enumerate(ops)}

    for i, op in enumerate(ops):
        combo_idx = int(np.argmax(alpha[i]))
        dur = float(op.get_duration_for_combination(combo_idx, machine_combinations, machines))
        finish = float(t[i]) + dur
        durations.append(dur)
        finish_times.append(finish)

        is_periodic = (op.min_start_t is not None) or (op.max_end_t is not None)
        if not is_periodic:
            nonperiodic_finish.append(finish)

        # A periodic instance's deadline lives in max_end_t (release + window),
        # set by workload_factory when `period`/`window_duration` are expanded.
        # deadline_us is only populated by the per-dispatch path in
        # create_workload_from_dependencies, so counting misses against it alone
        # made every periodic miss invisible: a schedule where all 10 dronet
        # instances overran their 33.3 ms window by ~80 ms reported
        # deadline_miss_count = 0. Take whichever bound is tighter.
        bounds = [b for b in (op.deadline_us, op.max_end_t) if b is not None]
        if bounds:
            late = finish - float(min(bounds))
            if late > _DEADLINE_EPS:
                deadline_miss += 1
                total_lateness += late
                if late > max_lateness:
                    max_lateness = late

        for m in machine_combinations[combo_idx]:
            per_machine_busy[m] += dur

        for pred in op.get_predecessors():
            pi = op_to_idx.get(id(pred))
            if pi is None:
                continue
            pred_combo = set(machine_combinations[int(np.argmax(alpha[pi]))])
            cur_combo = set(machine_combinations[combo_idx])
            if not (pred_combo & cur_combo):
                cross_device_transitions += 1

    makespan = max(finish_times) if finish_times else 0.0
    nonperiodic_makespan = max(nonperiodic_finish) if nonperiodic_finish else 0.0

    critical_path_us = _critical_path_length(workload, durations, op_to_idx)

    utilization = {
        m: (per_machine_busy[m] / makespan) if makespan > 0 else 0.0
        for m in machines
    }
    idle_time = {
        m: max(0.0, makespan - per_machine_busy[m]) for m in machines
    }

    metrics: Dict[str, Any] = {
        "scheduler": scheduler_name,
        "num_operations": n,
        "makespan_us": float(makespan),
        "nonperiodic_makespan_us": float(nonperiodic_makespan),
        "mean_op_duration_us": float(np.mean(durations)) if durations else 0.0,
        "p95_op_duration_us": float(np.percentile(durations, 95)) if durations else 0.0,
        "p99_op_duration_us": float(np.percentile(durations, 99)) if durations else 0.0,
        "deadline_miss_count": int(deadline_miss),
        "deadline_miss_ratio": (deadline_miss / n) if n else 0.0,
        "total_lateness_us": float(total_lateness),
        "max_lateness_us": float(max_lateness),
        "per_machine_utilization": utilization,
        "per_machine_idle_us": idle_time,
        "cross_device_transitions": int(cross_device_transitions),
        "critical_path_us": float(critical_path_us),
        "solver_wall_time_s": solver_wall_time_s,
        # Memory fields are populated when a memory plan is supplied (milestone 7+).
        "peak_dram_bytes": None,
        "peak_scratchpad_bytes": None,
        "buffer_reuse_count": None,
        # Jitter requires multi-instance periodic execution; filled in milestone 4.
        "jitter_us": None,
        # Packing capacity is computed at the sweep level, not per-schedule.
        "packing_capacity": None,
    }

    if memory_plan is not None:
        metrics["peak_dram_bytes"] = memory_plan.get("peak_dram_bytes")
        metrics["peak_scratchpad_bytes"] = memory_plan.get("peak_scratchpad_bytes")
        metrics["buffer_reuse_count"] = memory_plan.get("buffer_reuse_count")

    return metrics


def _critical_path_length(
    workload: Workload,
    durations: List[float],
    op_to_idx: Dict[int, int],
) -> float:
    """Longest-path length over the DAG using the assigned per-op durations."""
    ops = workload.operations
    n = len(ops)
    cp = [0.0] * n
    # ops in `workload.operations` are not guaranteed topo-sorted; do a Kahn pass.
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_to_idx.get(id(pred))
            if pi is None:
                continue
            indeg[i] += 1
            succ[pi].append(i)

    queue = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while queue:
        u = queue.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) < n:
        # Cycle — fall back to sum of durations as a conservative upper bound.
        return float(sum(durations))

    for u in order:
        best_pred = 0.0
        for pred in ops[u].get_predecessors():
            pi = op_to_idx.get(id(pred))
            if pi is None:
                continue
            if cp[pi] > best_pred:
                best_pred = cp[pi]
        cp[u] = best_pred + durations[u]

    return max(cp) if cp else 0.0
