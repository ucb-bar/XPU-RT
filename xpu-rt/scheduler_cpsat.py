"""
CP-SAT scheduler for XPU-RT.

Each operation is modelled with one optional interval per machine combination;
exactly one is present. Constraints:

  - exclusive resources: ``NoOverlap`` over all optional intervals whose
    combination uses each machine (a machine can appear in multiple
    combinations; intervals from any of those must not overlap on that machine)
  - precedence: ``pred_end + transfer_cost <= succ_start``, transfer cost
    derived from the workload's transfer_times[pred_machine, succ_machine]
  - release time: ``op.min_start_t <= start``
  - deadline: ``end <= deadline + lateness``, with lateness >= 0; lateness
    above zero counts as a deadline miss

Objective (lexicographic, encoded via large coefficients):

  1. deadline_miss_count          (weight 10^12)
  2. total_lateness               (weight 10^8)
  3. makespan                     (weight 1)
  4. cross-device transfer cost   (weight 1)  (small tiebreak)

OR-Tools requires integer values; durations and transfers are scaled to
integer microseconds (rounded; min 1).

The MOSEK MILP signature (solver_verbosity, time_limit, etc.) is accepted so
the scheduler is a drop-in registry entry.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload


def _lazy_cp_model():
    try:
        from ortools.sat.python import cp_model  # noqa
        return cp_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ortools is required for the CP-SAT scheduler. "
            "Install with `pip install ortools` or add to env.yml."
        ) from exc


def _to_int_us(x: float) -> int:
    if x is None or x <= 0:
        return 0
    v = int(round(x))
    return max(1, v)


def cpsat_schedule(
    workload: Workload,
    *,
    time_limit: Optional[float] = 30.0,
    solver_verbosity: int = 0,
    warm_start: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    deadline_weight: int = 10_000_000_000,
    lateness_weight: int = 100_000,
    transfer_weight: int = 1,
    makespan_weight: int = 1,
    memory_aware: bool = False,
    region_capacities: Optional[Dict[str, int]] = None,
    memory_weight: int = 10,
    **_unused,
) -> Tuple[np.ndarray, np.ndarray, None, None]:
    cp_model = _lazy_cp_model()
    model = cp_model.CpModel()

    ops = workload.operations
    n = len(ops)
    combos = workload.get_machine_combinations()
    n_combos = len(combos)
    machines = list(workload.machines)
    name_to_idx = {m: i for i, m in enumerate(machines)}
    transfer = workload.get_transfer_times()
    if transfer is None or len(transfer) == 0:
        transfer = np.zeros((len(machines), len(machines)))

    # Horizon = sum of max per-op duration across *feasible* combos.
    # Infeasible combos get a placeholder large duration (won't be chosen), but
    # we exclude them from horizon arithmetic so horizon stays tight.
    horizon = 0
    durations_int: List[List[int]] = []
    for op in ops:
        per_combo: List[int] = []
        feasible_durs: List[int] = []
        for k in range(n_combos):
            if k in op.infeasible_combinations:
                per_combo.append(_to_int_us(1e9))
            else:
                d = _to_int_us(float(op.get_duration_for_combination(
                    k, combos, machines)))
                per_combo.append(d)
                feasible_durs.append(d)
        durations_int.append(per_combo)
        horizon += (max(feasible_durs) if feasible_durs else _to_int_us(1e6)) + 1

    # Horizon must encompass deadlines and release windows of all ops.
    max_release = max((int(op.min_start_t) for op in ops if op.min_start_t is not None),
                      default=0)
    max_deadline = max((int(op.deadline_us) for op in ops if op.deadline_us is not None),
                       default=0)
    horizon += max_release
    horizon = max(horizon, max_deadline + 1000, 1000)

    # Decision variables.
    starts: List[Any] = []
    ends: List[Any] = []
    intervals_per_machine: Dict[str, List[Any]] = {m: [] for m in machines}
    presence: List[List[Any]] = []  # presence[i][k]
    chosen_start: List[Any] = []     # consolidated start (the chosen interval's start)
    chosen_end: List[Any] = []       # consolidated end

    for i, op in enumerate(ops):
        feasible_combos = [k for k in range(n_combos) if k not in op.infeasible_combinations]
        if not feasible_combos:
            # Fall back to any combo with the smallest duration to keep model feasible.
            feasible_combos = [int(np.argmin(durations_int[i]))]

        # Per-(op, combo) optional interval.
        per_k_start: List[Any] = []
        per_k_end: List[Any] = []
        per_k_pres: List[Any] = []
        for k in range(n_combos):
            pres = model.NewBoolVar(f"pres_{i}_{k}")
            per_k_pres.append(pres)
            if k not in feasible_combos:
                model.Add(pres == 0)
                # Dummy start/end placeholders.
                per_k_start.append(model.NewConstant(0))
                per_k_end.append(model.NewConstant(0))
                continue
            s = model.NewIntVar(0, horizon, f"s_{i}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{i}_{k}")
            d = durations_int[i][k]
            ivar = model.NewOptionalIntervalVar(s, d, e, pres, f"iv_{i}_{k}")
            for m in combos[k]:
                intervals_per_machine[m].append(ivar)
            per_k_start.append(s)
            per_k_end.append(e)
        presence.append(per_k_pres)
        # Exactly one combo chosen.
        model.AddExactlyOne(per_k_pres[k] for k in feasible_combos)

        # Consolidated start/end (the chosen-combo's start/end).
        chosen_s = model.NewIntVar(0, horizon, f"cs_{i}")
        chosen_e = model.NewIntVar(0, horizon, f"ce_{i}")
        for k in feasible_combos:
            # When pres[i][k] is true, chosen_s == per_k_start[k] etc.
            model.Add(chosen_s == per_k_start[k]).OnlyEnforceIf(per_k_pres[k])
            model.Add(chosen_e == per_k_end[k]).OnlyEnforceIf(per_k_pres[k])
        chosen_start.append(chosen_s)
        chosen_end.append(chosen_e)

        # Release time.
        if op.min_start_t is not None and op.min_start_t > 0:
            model.Add(chosen_s >= _to_int_us(float(op.min_start_t)))

    # Per-machine NoOverlap.
    for m, ivars in intervals_per_machine.items():
        if len(ivars) > 1:
            model.AddNoOverlap(ivars)

    # Precedence with transfer cost.
    op_idx = {id(op): i for i, op in enumerate(ops)}
    transfer_terms = []  # accumulate for the small transfer objective term

    for i, op in enumerate(ops):
        feasible_i = [k for k in range(n_combos) if k not in op.infeasible_combinations]
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is None:
                continue
            feasible_p = [k for k in range(n_combos) if k not in pred.infeasible_combinations]
            for kp in feasible_p:
                for ki in feasible_i:
                    # Transfer cost: worst-case between any (pred_machine, curr_machine)
                    # pair in the two combos. Use the cheaper-direction (or zero if same).
                    cost = 0
                    for mp in combos[kp]:
                        for mi in combos[ki]:
                            mp_idx = name_to_idx.get(mp)
                            mi_idx = name_to_idx.get(mi)
                            if mp_idx is None or mi_idx is None or mp_idx == mi_idx:
                                continue
                            cost = max(cost, int(transfer[mp_idx][mi_idx]))
                    pair = model.NewBoolVar(f"pair_{pi}_{kp}__{i}_{ki}")
                    # Pair is true iff both presences are true.
                    model.AddBoolAnd([presence[pi][kp], presence[i][ki]]).OnlyEnforceIf(pair)
                    model.AddBoolOr([presence[pi][kp].Not(), presence[i][ki].Not()]).OnlyEnforceIf(pair.Not())
                    # If pair is true, enforce precedence.
                    model.Add(chosen_start[i] >= chosen_end[pi] + cost).OnlyEnforceIf(pair)
                    if cost > 0:
                        transfer_terms.append((pair, cost))

    # Deadlines with lateness.
    deadline_vars: List[Any] = []  # boolean miss flags
    lateness_vars: List[Any] = []
    for i, op in enumerate(ops):
        if op.deadline_us is None:
            continue
        d = _to_int_us(float(op.deadline_us))
        lat = model.NewIntVar(0, horizon, f"lat_{i}")
        # lateness = max(0, chosen_end[i] - d)
        diff = model.NewIntVar(-horizon, horizon, f"diff_{i}")
        model.Add(diff == chosen_end[i] - d)
        model.AddMaxEquality(lat, [diff, model.NewConstant(0)])
        lateness_vars.append(lat)
        miss = model.NewBoolVar(f"miss_{i}")
        # miss = (lat > 0)
        model.Add(lat >= 1).OnlyEnforceIf(miss)
        model.Add(lat == 0).OnlyEnforceIf(miss.Not())
        deadline_vars.append(miss)

    # Makespan.
    makespan = model.NewIntVar(0, horizon, "makespan")
    if chosen_end:
        model.AddMaxEquality(makespan, chosen_end)

    # Memory-aware cumulative constraint over buffer live intervals.
    peak_mem_var = None
    if memory_aware:
        # Find consumers per producer (graph adjacency we computed via predecessors).
        consumers_of: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, op in enumerate(ops):
            for pred in op.get_predecessors():
                pi = op_idx.get(id(pred))
                if pi is not None:
                    consumers_of[pi].append(i)

        capacity = int(max((region_capacities or {}).values(), default=10**12))
        peak_mem_var = model.NewIntVar(0, capacity, "peak_mem")

        buf_intervals = []
        buf_demands = []
        for i, op in enumerate(ops):
            size = int(getattr(op, "output_bytes", 0))
            if size <= 0:
                continue
            cons = consumers_of[i]
            buf_start = chosen_end[i]
            if cons:
                # buf_end = max(chosen_start[c] for c in cons)
                buf_end = model.NewIntVar(0, horizon, f"bufend_{i}")
                model.AddMaxEquality(buf_end, [chosen_start[c] for c in cons])
            else:
                # sink: kept until makespan
                buf_end = makespan
            buf_size = model.NewIntVar(0, horizon, f"bufdur_{i}")
            model.Add(buf_size == buf_end - buf_start)
            iv = model.NewIntervalVar(buf_start, buf_size, buf_end, f"bufiv_{i}")
            buf_intervals.append(iv)
            buf_demands.append(size)

        if buf_intervals:
            model.AddCumulative(buf_intervals, buf_demands, capacity)

    # Objective.
    obj_terms: List[Any] = []
    if deadline_vars:
        obj_terms.append(deadline_weight * sum(deadline_vars))
    if lateness_vars:
        obj_terms.append(lateness_weight * sum(lateness_vars))
    obj_terms.append(makespan_weight * makespan)
    if transfer_terms:
        obj_terms.append(transfer_weight * sum(p * c for p, c in transfer_terms))
    model.Minimize(sum(obj_terms))

    # Warm start (HEFT).
    if warm_start is not None:
        ws_t, ws_alpha = warm_start
        try:
            for i in range(n):
                k = int(np.argmax(ws_alpha[i]))
                model.AddHint(presence[i][k], 1)
                model.AddHint(chosen_start[i], _to_int_us(float(ws_t[i])))
        except Exception:
            pass

    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 4
    if solver_verbosity >= 2:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Return None to signal infeasible / no solution.
        return None, None, None, None  # type: ignore[return-value]

    t = np.zeros(n)
    alpha = np.zeros((n, n_combos))
    for i in range(n):
        t[i] = float(solver.Value(chosen_start[i]))
        for k in range(n_combos):
            if solver.Value(presence[i][k]) == 1:
                alpha[i, k] = 1.0
                break
    return t, alpha, None, None


def cpsat_with_heft_warm_start(workload, **kwargs):
    """Convenience: HEFT first, then CP-SAT seeded by HEFT placement."""
    from scheduler_heft import heft
    try:
        warm_t, warm_alpha, _, _ = heft(workload)
        kwargs["warm_start"] = (warm_t, warm_alpha)
    except Exception:
        pass
    return cpsat_schedule(workload, **kwargs)


def cpsat_memory_aware(workload, *, scratchpad_bytes: int = 16 * 1024 * 1024, **kwargs):
    """Memory-aware CP-SAT: cumulative constraint over buffer live intervals
    enforces ``sum(active output_bytes) <= scratchpad_bytes`` at every instant.

    Defaults the capacity to 16 MB. ``output_bytes`` is read from
    ``op.output_bytes`` (set by realistic_workloads / pack_periodic_workload).
    """
    kwargs.setdefault("memory_aware", True)
    kwargs.setdefault("region_capacities", {"scratchpad": int(scratchpad_bytes)})
    return cpsat_with_heft_warm_start(workload, **kwargs)
