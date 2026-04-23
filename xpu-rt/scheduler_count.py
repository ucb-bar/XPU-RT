"""
Count-based scheduler: MILP only distinguishes by (type, count), then a
greedy pass assigns which specific cores each op runs on.

Key difference from scheduler.schedule():
- Non-overlap constraints use CAPACITY (size_i + size_j > type_capacity)
  instead of per-combination core-sharing. Two ops each wanting 1 P-core
  no longer have to serialize on a 4-P-core machine.

This is a pairwise capacity relaxation: it handles all pairs exactly but
only approximates cumulative triple-and-higher conflicts. If the greedy
core assignment after the MILP fails, that signals the relaxation missed
a conflict (see `assign_specific_cores` in core_count_flow.py).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload
from workload_factory import machine_type_prefix
from scheduler import _compute_dependency_descendants_bitset
from core_count_flow import (
    assign_specific_cores,
    build_all_subset_combinations,
    build_workload_with_subset_combinations,
    _combo_index_for_cores,
)


def _combos_conflict_by_capacity(
    combos, k1: int, k2: int, capacities: Dict[str, int],
) -> bool:
    """True iff combo_k1 and combo_k2 are same type and their combined count
    exceeds that type's core capacity — i.e. they really do contend."""
    type1 = machine_type_prefix(combos[k1][0])
    type2 = machine_type_prefix(combos[k2][0])
    if type1 != type2:
        return False
    total = len(combos[k1]) + len(combos[k2])
    return total > capacities.get(type1, 0)


def schedule_by_count(
    workload: Workload,
    machine_core_counts: Dict[str, int],
    solver_verbosity: int = 0,
    time_limit: Optional[float] = None,
    restrict_makespan_to_nonperiodic: bool = True,
    prune_cross_period_constraints: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """MILP that picks a cumulative-prefix combo per op and schedules start
    times. Non-overlap uses capacity-based conflict, not specific-core sharing."""
    operations = workload.get_operations()
    n = len(operations)
    combos = workload.get_machine_combinations()
    C = len(combos)
    transfer_times = workload.get_transfer_times()
    capacities = dict(machine_core_counts)

    alpha = cp.Variable((n, C), boolean=True)
    beta = cp.Variable((n, n), boolean=True)
    t = cp.Variable(n)
    C_max = cp.Variable()
    H = 5000
    constraints = []

    # (2) each op assigned to exactly one combo
    for i in range(n):
        constraints.append(cp.sum(alpha[i, :]) == 1)

    # (3) precedence
    max_tt = float(np.max(transfer_times)) if transfer_times is not None else 0.0
    for i in range(n):
        op = operations[i]
        for pred in op.get_predecessors():
            try:
                i_pred = operations.index(pred)
            except ValueError:
                continue
            if prune_cross_period_constraints:
                pe = getattr(pred, "max_end_t", None)
                ss = getattr(op, "min_start_t", None)
                if pe is not None and ss is not None and pe <= ss:
                    continue
            dur_vec_pred = [
                operations[i_pred].get_duration_for_combination(k, combos, workload.machines)
                for k in range(C)
            ]
            constraints.append(
                t[i] >= t[i_pred] + cp.sum(cp.multiply(dur_vec_pred, alpha[i_pred, :])) + max_tt
            )

    # time windows
    for i in range(n):
        op = operations[i]
        if op.min_start_t is not None:
            constraints.append(t[i] >= op.min_start_t)
        if op.max_end_t is not None:
            dur_vec = [op.get_duration_for_combination(k, combos, workload.machines) for k in range(C)]
            constraints.append(t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])) <= op.max_end_t)

    # (4)(5) non-overlap — ONLY when capacity would be exceeded
    def _periods_overlap(a, b):
        if (a.min_start_t is None or a.max_end_t is None
                or b.min_start_t is None or b.max_end_t is None):
            return True
        return a.min_start_t < b.max_end_t and b.min_start_t < a.max_end_t

    dep_desc = _compute_dependency_descendants_bitset(operations)
    overlap_constraints = 0
    overlap_pairs_considered = 0
    for i in range(n):
        for j in range(i + 1, n):
            if prune_cross_period_constraints and not _periods_overlap(operations[i], operations[j]):
                continue
            if dep_desc is not None:
                if ((dep_desc[i] >> j) & 1) or ((dep_desc[j] >> i) & 1):
                    continue
            overlap_pairs_considered += 1
            for k1 in range(C):
                for k2 in range(C):
                    if not _combos_conflict_by_capacity(combos, k1, k2, capacities):
                        continue
                    dur_j_k2 = operations[j].get_duration_for_combination(k2, combos, workload.machines)
                    dur_i_k1 = operations[i].get_duration_for_combination(k1, combos, workload.machines)
                    constraints.append(
                        t[i] >= t[j] + dur_j_k2
                        - (2 - alpha[i, k1] - alpha[j, k2] + beta[i, j]) * H
                    )
                    constraints.append(
                        t[j] >= t[i] + dur_i_k1
                        - (3 - alpha[i, k1] - alpha[j, k2] - beta[i, j]) * H
                    )
                    overlap_constraints += 2

    # (6) makespan
    if restrict_makespan_to_nonperiodic:
        any_np = False
        for i in range(n):
            op = operations[i]
            if op.min_start_t is not None or op.max_end_t is not None:
                continue
            any_np = True
            dur_vec = [op.get_duration_for_combination(k, combos, workload.machines) for k in range(C)]
            constraints.append(C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])))
        if not any_np:
            constraints.append(C_max >= 0)
    else:
        for i in range(n):
            dur_vec = [operations[i].get_duration_for_combination(k, combos, workload.machines) for k in range(C)]
            constraints.append(C_max >= t[i] + cp.sum(cp.multiply(dur_vec, alpha[i, :])))

    for i in range(n):
        constraints.append(t[i] >= 0)

    problem = cp.Problem(cp.Minimize(C_max), constraints)
    mosek_params = {}
    if time_limit is not None and time_limit > 0:
        mosek_params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = float(time_limit)

    print(f"[schedule_by_count] ops={n} combos={C} "
          f"pairs_considered={overlap_pairs_considered} "
          f"non-overlap_constraints={overlap_constraints} "
          f"total_constraints={len(constraints)}")

    t0 = time.perf_counter()
    if mosek_params:
        problem.solve(solver=cp.MOSEK, verbose=solver_verbosity > 0, mosek_params=mosek_params)
    else:
        problem.solve(solver=cp.MOSEK, verbose=solver_verbosity > 0)
    elapsed = time.perf_counter() - t0

    print(f"[schedule_by_count] status={problem.status} "
          f"value={problem.value} elapsed={elapsed:.2f}s")

    if t.value is None or alpha.value is None:
        return None, None
    mask = alpha.value == alpha.value.max(axis=1, keepdims=True)
    return t.value, mask.astype(int)


def schedule_with_count_and_assignment(
    workload: Workload,
    machine_core_counts: Dict[str, int],
    solver_verbosity: int = 0,
    time_limit: Optional[float] = None,
    restrict_makespan_to_nonperiodic: bool = True,
    prune_cross_period_constraints: bool = True,
) -> Tuple[Workload, np.ndarray, np.ndarray]:
    """Run count-only MILP, then assign specific cores greedily.
    Returns (expanded_workload on all-subset combos, t, alpha)."""
    print("[count_flow] MILP on cumulative combos (count-only)")
    t_count, alpha_count = schedule_by_count(
        workload, machine_core_counts,
        solver_verbosity=solver_verbosity, time_limit=time_limit,
        restrict_makespan_to_nonperiodic=restrict_makespan_to_nonperiodic,
        prune_cross_period_constraints=prune_cross_period_constraints,
    )
    if t_count is None or alpha_count is None:
        raise RuntimeError("Count-only MILP failed to find a solution.")

    old_combos = workload.get_machine_combinations()
    decisions = {}
    for i, op in enumerate(workload.operations):
        k = int(np.argmax(alpha_count[i]))
        type_ = machine_type_prefix(old_combos[k][0])
        count = len(old_combos[k])
        dur = float(op.processing_times[k])
        start = float(t_count[i])
        decisions[op] = (type_, count, start, dur)

    print("[count_flow] greedy specific-core assignment")
    try:
        cores_per_op = assign_specific_cores(decisions, machine_core_counts)
    except RuntimeError as e:
        raise RuntimeError(
            f"Specific-core assignment failed after count MILP — this means "
            f"the pairwise capacity relaxation missed a higher-order conflict. "
            f"Details: {e}"
        ) from e

    new_machines, new_combos = build_all_subset_combinations(machine_core_counts)
    new_workload, old_to_new = build_workload_with_subset_combinations(
        workload, old_combos, new_combos, new_machines,
    )

    n = len(new_workload.operations)
    C = len(new_combos)
    t_out = np.zeros(n)
    alpha_out = np.zeros((n, C), dtype=int)
    for old_op, new_op in old_to_new.items():
        i = new_workload.operations.index(new_op)
        _, _, start, _ = decisions[old_op]
        t_out[i] = start
        cores = cores_per_op[old_op]
        k = _combo_index_for_cores(new_combos, cores)
        alpha_out[i, k] = 1

    return new_workload, t_out, alpha_out
