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

Objective (lexicographic, solved in sequential phases):

  1. deadline_miss_count
  2. total_lateness
  3. makespan + cross-device transfer cost (small tiebreak)

Each phase is solved to optimality and fixed before the next objective is
installed. This is both an exact priority order and avoids the integer overflow
that a big weighted sum causes once times are represented in microseconds.

OR-Tools requires integer values; the workload uses milliseconds, so durations
and transfers are scaled to integer microseconds (rounded up; min 1) and the
returned start times are converted back to milliseconds.

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
import job_names


_US_PER_MS = 1000


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
    # A modeled interval must never be shorter than the measured duration that
    # postprocessing serializes. Rounding to the nearest millisecond used to
    # turn 1.292625 ms into one solver tick, then emit a successor at 1.000 ms
    # while retaining the predecessor's exact duration: a real overlap. The
    # microsecond ceiling is conservative by less than one microsecond.
    v = int(math.ceil(float(x) * _US_PER_MS))
    return max(1, v)


def _from_int_us(x: int) -> float:
    return float(x) / _US_PER_MS


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
    objective_mode: str = "legacy",
    critical_models: Optional[List[str]] = None,
    heavy_model: Optional[str] = None,
    objective_stop_after: Optional[str] = None,
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
    max_release = max((_to_int_us(float(op.min_start_t)) for op in ops
                       if op.min_start_t is not None), default=0)
    # Periodic jobs carry their deadline in max_end_t, not deadline_us. The
    # response/lateness variables range over the horizon, so omitting those
    # deadlines can make a perfectly feasible short workload contradictory
    # during presolve (e.g. an 11 ms chain with a 20 ms periodic deadline).
    max_deadline = max((
        _to_int_us(float(
            op.deadline_us if op.deadline_us is not None else op.max_end_t))
        for op in ops
        if op.deadline_us is not None or op.max_end_t is not None
    ), default=0)
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
                            cost = max(
                                cost,
                                _to_int_us(float(transfer[mp_idx][mi_idx])))
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
        # A PERIODIC op carries its deadline as max_end_t (= release + window),
        # while deadline_us is a separate optional robotics hard deadline. Use
        # whichever is set (deadline_us wins) so the deadline-miss / lateness
        # objective (weights 10^12 / 10^8) is ACTIVE for periodic workloads too
        # — otherwise a windowed spec falls through to makespan-only and the
        # solver trades deadlines away (the greedy-beats-CP-SAT gap).
        dl = op.deadline_us if op.deadline_us is not None \
            else getattr(op, "max_end_t", None)
        if dl is None:
            continue
        d = _to_int_us(float(dl))
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

    # Exact-cycle response objective. The legacy model counts late DISPATCHES
    # and then minimizes their summed lateness. That is useful as a generic
    # scheduler objective, but it is not the quantity the feedback experiment
    # reports: real-time compliance and response are defined per released MODEL
    # INSTANCE. Build those variables explicitly so the solver and evaluator
    # optimize the same lexicographic vector.
    exact_objectives: List[Tuple[str, Any, str]] = []
    exact_job_count = 0
    if objective_mode == "exact_cycle_worst_response":
        critical = set(critical_models or ())
        known = set(critical)
        if heavy_model:
            known.add(heavy_model)
        if not known:
            raise ValueError(
                "exact_cycle_worst_response requires critical_models and/or "
                "heavy_model")

        by_job: Dict[int, List[int]] = {}
        for i, op in enumerate(ops):
            if op.job_id is not None:
                by_job.setdefault(int(op.job_id), []).append(i)

        job_misses: List[Any] = []
        job_lateness: List[Any] = []
        critical_responses: List[Any] = []
        heavy_responses: List[Any] = []
        for job_id, indices in sorted(by_job.items()):
            if job_id < 0 or job_id >= len(workload.job_names):
                continue
            name = str(workload.job_names[job_id])
            model_name = job_names.model_of(name, known)
            if model_name not in known:
                continue

            release_values = [
                _to_int_us(float(ops[i].min_start_t))
                for i in indices if ops[i].min_start_t is not None
            ]
            deadline_values = [
                _to_int_us(float(
                    ops[i].deadline_us if ops[i].deadline_us is not None
                    else ops[i].max_end_t))
                for i in indices
                if (ops[i].deadline_us is not None
                    or ops[i].max_end_t is not None)
            ]
            if not release_values or not deadline_values:
                raise ValueError(
                    f"exact-cycle job {name!r} lacks a release or deadline")
            release = min(release_values)
            deadline = min(deadline_values)

            completion = model.NewIntVar(0, horizon, f"job_end_{job_id}")
            model.AddMaxEquality(completion, [chosen_end[i] for i in indices])
            response = model.NewIntVar(0, horizon, f"job_resp_{job_id}")
            model.Add(response == completion - release)
            late_diff = model.NewIntVar(-horizon, horizon,
                                        f"job_late_diff_{job_id}")
            model.Add(late_diff == completion - deadline)
            late = model.NewIntVar(0, horizon, f"job_late_{job_id}")
            model.AddMaxEquality(late, [late_diff, model.NewConstant(0)])
            miss = model.NewBoolVar(f"job_miss_{job_id}")
            model.Add(late >= 1).OnlyEnforceIf(miss)
            model.Add(late == 0).OnlyEnforceIf(miss.Not())

            job_misses.append(miss)
            job_lateness.append(late)
            if model_name in critical:
                critical_responses.append(response)
            if heavy_model and model_name == heavy_model:
                heavy_responses.append(response)
            exact_job_count += 1

        if not critical_responses:
            raise ValueError("no critical-model instances found in workload")

        max_lateness = model.NewIntVar(0, horizon, "max_job_lateness")
        model.AddMaxEquality(max_lateness, job_lateness)
        worst_critical = model.NewIntVar(0, horizon,
                                         "worst_critical_response")
        model.AddMaxEquality(worst_critical, critical_responses)
        exact_objectives.extend([
            ("job_deadline_misses", sum(job_misses), "instances"),
            ("max_job_lateness", max_lateness, "us"),
            ("worst_critical_response", worst_critical, "us"),
        ])
        if heavy_responses:
            heavy_max = model.NewIntVar(0, horizon, "heavy_max_response")
            model.AddMaxEquality(heavy_max, heavy_responses)
            exact_objectives.append(("heavy_max_response", heavy_max, "us"))
    elif objective_mode != "legacy":
        raise ValueError(f"unknown CP-SAT objective_mode {objective_mode!r}")

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

    # Lowest-priority objective. Deadline misses and lateness are optimized in
    # separate phases below, then fixed at their proven optima before this is
    # installed. Keep the public weight arguments for registry compatibility;
    # deadline/lateness weights are no longer needed when each is isolated.
    lower_obj = makespan_weight * makespan
    if transfer_terms:
        lower_obj += transfer_weight * sum(p * c for p, c in transfer_terms)

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
    if time_limit is not None and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    # Phase Q-rerun: cold-rerun gate requires reproducibility. CPSAT's
    # parallel search and worker-stealing produce non-deterministic
    # solutions when the optimizer is time-limited. We pin both:
    #   num_search_workers = 1     → no parallel race
    #   random_seed         = 42   → deterministic branching
    # The cost: ~1.5-2× wall-clock vs 4-worker parallel. The benefit:
    # the cold rerun matches the warm run bit-exactly.
    # num_search_workers=1 pins reproducibility (bit-exact cold rerun) but CRIPPLES
    # CP-SAT's parallel portfolio search -- the single biggest reason it gets stuck
    # above greedy's makespan. Set XPURT_CPSAT_WORKERS=8 (or 0=auto) to unleash the
    # full portfolio when beating greedy matters more than bit-exact reruns.
    _nw = os.environ.get("XPURT_CPSAT_WORKERS", "")
    solver.parameters.num_search_workers = int(_nw) if _nw.strip() else 1
    solver.parameters.random_seed = 42
    if solver_verbosity >= 2:
        solver.parameters.log_search_progress = True

    # True lexicographic optimization. If a configured time limit yields only
    # FEASIBLE rather than OPTIMAL in an early phase, return that incumbent:
    # fixing an unproven value and optimizing a lower-priority term would be a
    # false lexicographic claim. With no limit, each phase runs to proof.
    if exact_objectives:
        objectives = list(exact_objectives)
        objectives.append(("makespan_plus_transfer", lower_obj, "us"))
    else:
        objectives = []
        if deadline_vars:
            objectives.append(("dispatch_deadline_misses",
                               sum(deadline_vars), "dispatches"))
        if lateness_vars:
            objectives.append(("total_dispatch_lateness",
                               sum(lateness_vars), "us"))
        objectives.append(("makespan_plus_transfer", lower_obj, "us"))

    status = None
    phase_reports = []
    for phase, (phase_name, phase_obj, phase_unit) in enumerate(objectives):
        model.Minimize(phase_obj)
        status = solver.Solve(model)
        phase_reports.append({
            "name": phase_name,
            "unit": phase_unit,
            "status": solver.StatusName(status),
            "objective": (float(solver.ObjectiveValue())
                          if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                          else None),
            "best_bound": (float(solver.BestObjectiveBound())
                           if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                           else None),
            "wall_s": float(solver.WallTime()),
        })
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            workload.solver_certificate = {
                "schema_version": 1,
                "solver": "cpsat",
                "objective_mode": objective_mode,
                "time_unit": "microseconds",
                "phases": phase_reports,
                "certified": False,
            }
            return None, None, None, None  # type: ignore[return-value]
        if objective_stop_after and phase_name == objective_stop_after:
            break
        if status != cp_model.OPTIMAL or phase == len(objectives) - 1:
            break
        optimum = int(round(solver.ObjectiveValue()))
        model.Add(phase_obj == optimum)

    assert status is not None

    workload.solver_certificate = {
        "schema_version": 1,
        "solver": "cpsat",
        "objective_mode": objective_mode,
        "time_unit": "microseconds",
        "critical_models": sorted(critical_models or ()),
        "heavy_model": heavy_model,
        "jobs_modeled": exact_job_count if exact_objectives else None,
        "stop_after": objective_stop_after,
        "phases": phase_reports,
        "certified": all(p["status"] == "OPTIMAL" for p in phase_reports),
        "certified_through": phase_reports[-1]["name"],
    }

    t = np.zeros(n)
    alpha = np.zeros((n, n_combos))
    for i in range(n):
        t[i] = _from_int_us(solver.Value(chosen_start[i]))
        for k in range(n_combos):
            if solver.Value(presence[i][k]) == 1:
                alpha[i, k] = 1.0
                break
    return t, alpha, None, None


def cpsat_with_heft_warm_start(workload, **kwargs):
    """Convenience: HEFT first, then CP-SAT seeded by HEFT placement.

    NOTE: HEFT's (t, alpha) share CP-SAT's machine-combination indexing, so its
    hints are consistent. The greedy list-scheduler's alpha uses a DIFFERENT
    combination indexing, so seeding CP-SAT with it produces inconsistent hints
    that MISLEAD the solver (observed: makespan 86->153ms, 0->38 deadline misses).
    If a greedy seed is ever wanted, its placement must first be remapped into
    CP-SAT's combo indices; do not feed greedy_schedule's alpha raw.
    """
    # NOTE: warm-starting from the GREEDY list-scheduler was tried twice (raw, and
    # with an infeasible-combo filter) and both TRAP CP-SAT at 86->153ms / 38 misses:
    # greedy's (t, alpha) is inconsistent with CP-SAT's timing/precedence/stateful
    # semantics, so its hints mislead rather than help. Seeding from greedy would
    # require re-deriving its schedule in CP-SAT's exact variables -- a real project.
    # HEFT's hints ARE consistent (same model lineage), so we use HEFT.
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
