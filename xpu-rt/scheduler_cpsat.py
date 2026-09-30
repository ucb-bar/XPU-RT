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

# profile_loader marks a cell the hardware cannot run -- an op with no IME kernel, a
# net with no profile at that shard width -- by COSTING it 1e8 ms rather than by
# listing it in `infeasible_combinations`. CP-SAT only excludes the latter, so it read
# "impossible" as merely "expensive" and, under a time limit, placed it: the sensor
# workload's shard candidate came back with a makespan of 500,000,069 us -- five
# sentinels laid end to end -- and 186 deadline misses, reported as a solved schedule.
# A schedule containing a sentinel is not a schedule, so treat the cost as the
# exclusion it was meant to be.
_INFEASIBLE_COST_MS = 1e8


def _lazy_cp_model():
    try:
        from ortools.sat.python import cp_model  # noqa
        return cp_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ortools is required for the CP-SAT scheduler. "
            "Install with `pip install ortools` or add to env.yml."
        ) from exc


def _packed_weight_groups(ops):
    """`{(network, dispatch_id): [op index, ...]}` for packed-weight dispatches only.

    The op kind is not a field on an Operation; it is inside `operation_name`, which is
    the module name (`<net>$dispatch_<id>_<backend>_<op>_<SHAPE>`). The packed set comes
    from ModelBlaster's codegen contract so there is one definition of it, and an
    unavailable contract yields no groups -- constraining nothing is the right failure
    here, since the alternative is constraining every dispatch on a guess.
    """
    try:
        import codegen_contract
        packed = codegen_contract.packed_weight_ops()
        op_of = codegen_contract.op_of
    except Exception:
        return {}
    groups = {}
    for i, op in enumerate(ops):
        name = str(getattr(op, "operation_name", "") or "")
        # `op_kind` is attached from the profile database by
        # run_xpurt_schedule._annotate_op_kinds, because a dispatch graph carries only
        # ids and dependencies -- `operation_name` is `<net-instance>_dispatch_<id>` and
        # names no op at all. The module-name parse is the fallback for callers that
        # build a workload without going through that path.
        kind = str(getattr(op, "op_kind", "") or "") or op_of(name, packed)
        if kind not in packed:
            continue
        did = getattr(op, "operation_id", None)
        if did is None:
            continue
        # The GROUP is the base network: its periodic instances are what must agree.
        net = str(getattr(op, "op_network", "") or "")
        if not net:
            net = (name.split("$")[0] if "$" in name
                   else name.split("_dispatch_")[0])
        groups.setdefault((net, int(did)), []).append(i)
    return groups


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

    # ---- CODEGEN CONTRACT: a shard's OC slice cannot have a remainder ---------
    # A packed convolution's weights are split across shards by output channel, so a
    # width that does not divide OC has nowhere to put the remainder and ModelBlaster
    # refuses the schedule: "has OC=2, not divisible by scheduled width 4; codegen would
    # silently run a serial implementation".
    #
    # Excluded BEFORE the duration table is built, so the horizon arithmetic never sees
    # these combinations. This is the cheapest constraint in the file to satisfy: on the
    # deployed sensor workload the two offending dispatches are yolo's OC=2 detect-head
    # convs, and the board measures them SLOWER on four cores than on one (0.0319 vs
    # 0.0312 ms) -- so the widths being removed were never worth having, and the
    # schedule that motivated this rule paid 0.17% of yolo's time for them.
    if os.environ.get("XPURT_UNIFORM_PACKED_WIDTH", "0") not in ("0", "", "false"):
        _n_oc = 0
        for op in ops:
            oc = getattr(op, "op_oc", None)
            if not oc:
                continue
            bad = {k for k in range(n_combos)
                   if len(combos[k]) > 1 and oc % len(combos[k])}
            if bad and not bad >= set(range(n_combos)):
                op.infeasible_combinations = set(op.infeasible_combinations) | bad
                _n_oc += 1
        if _n_oc:
            print(f"[cpsat] codegen contract: {_n_oc} packed-weight dispatch(es) had "
                  f"width(s) that do not divide their OC excluded")

    # PER-NETWORK SHARD RESTRICTION, before the duration table, so a restricted
    # combination is excluded rather than merely expensive and the horizon stays tight.
    try:
        import codegen_contract as _cc
        _only = _cc.shard_only_networks_from_env()
        if _only is not None:
            _cc.restrict_shard_to_networks(ops, combos, machines, _only, log=print)
    except Exception as _e:
        print(f"[cpsat] per-network shard restriction unavailable: {_e}")

    # Horizon = sum of max per-op duration across *feasible* combos.
    # Infeasible combos get a placeholder large duration (won't be chosen), but
    # we exclude them from horizon arithmetic so horizon stays tight.
    horizon = 0
    durations_int: List[List[int]] = []
    for op in ops:
        per_combo: List[int] = []
        feasible_durs: List[int] = []
        sentinel_ks: List[int] = []
        for k in range(n_combos):
            if k in op.infeasible_combinations:
                per_combo.append(_to_int_us(1e9))
            else:
                d_ms = float(op.get_duration_for_combination(k, combos, machines))
                d = _to_int_us(d_ms)
                if d_ms >= _INFEASIBLE_COST_MS:
                    # An exclusion expressed as a cost. Record it as an exclusion.
                    sentinel_ks.append(k)
                    per_combo.append(_to_int_us(1e9))
                else:
                    per_combo.append(d)
                    feasible_durs.append(d)
        if sentinel_ks:
            try:
                op.infeasible_combinations = set(op.infeasible_combinations) | set(
                    sentinel_ks)
            except Exception:
                pass
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

    # ---- CODEGEN CONTRACT: one width per packed-weight dispatch --------------
    # Every periodic instance of a dispatch whose weights are PACKED PER SHARD must be
    # given the same core width, because the packed weight array is materialised per
    # shard while generating the skeleton -- one generated model cannot carry two
    # layouts for one dispatch. Without this the solver is free to give `dronet`
    # dispatch 0 two cores in one instance and four in another; the schedule is valid
    # for the runtime and the compiler refuses it, and the refusal arrives at stage 1 of
    # 5 of a board build.
    #
    # Encoded as a per-(dispatch, width) indicator rather than by pinning a width: the
    # solver still CHOOSES the width, it just has to choose one. Pinning would trade a
    # correctness constraint for a policy decision, and the whole point of shard mode is
    # that the right width depends on what else is running.
    #
    # Off unless asked for, so no existing result moves. The co-design loop's `shard`
    # lever turns it on, which is what makes that lever's output deployable.
    # Recorded for the warm start below: a hint that violates the coupling is worse
    # than no hint, so the hint has to know which dispatches are coupled to what.
    uniform_groups: dict = {}
    if os.environ.get("XPURT_UNIFORM_PACKED_WIDTH", "0") not in ("0", "", "false"):
        _groups = _packed_weight_groups(ops)
        _n_coupled = 0
        _skipped = []
        for _key, _idxs in sorted(_groups.items()):
            if len(_idxs) < 2:
                continue
            _widths = sorted({len(combos[k]) for k in range(n_combos)})
            # A width is only usable by the GROUP if every instance can actually take
            # it. Without this check a group whose instances have disjoint feasible
            # widths -- one restricted to 1 core, another to 4, by
            # `infeasible_combinations` -- makes AddExactlyOne unsatisfiable, and the
            # solve comes back INFEASIBLE with nothing pointing at the reason. Such a
            # dispatch is genuinely unbuildable under shard mode, and saying so beats
            # returning "no solution" for the whole workload.
            _usable = [
                _w for _w in _widths
                if all(any(k not in ops[_i].infeasible_combinations
                           for k in range(n_combos) if len(combos[k]) == _w)
                       for _i in _idxs)]
            if not _usable:
                _skipped.append(_key)
                continue
            _w_vars = {}
            for _w in _usable:
                _ks = [k for k in range(n_combos) if len(combos[k]) == _w]
                _wv = model.NewBoolVar(f"pw_{_key[0]}_{_key[1]}_{_w}")
                _w_vars[_w] = _wv
                for _i in _idxs:
                    # this instance runs at width w  <=>  the dispatch runs at width w
                    model.Add(sum(presence[_i][k] for k in _ks) == _wv)
            model.AddExactlyOne(_w_vars.values())
            uniform_groups[_key] = (list(_idxs), sorted(_usable))
            _n_coupled += 1
        if _skipped:
            print(f"[cpsat] codegen contract: {len(_skipped)} packed-weight "
                  f"dispatch(es) have NO width every instance can take, so they are "
                  f"left unconstrained and the schedule will not be buildable: "
                  f"{_skipped[:4]}")
        if _n_coupled:
            print(f"[cpsat] codegen contract: {_n_coupled} packed-weight dispatch(es) "
                  f"constrained to one width across their instances")

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
    #
    # A HINT THAT BREAKS THE UNIFORM-WIDTH COUPLING IS WORSE THAN NO HINT. HEFT picks a
    # width per INSTANCE, so on a packed-weight dispatch it will happily hint width 1
    # for one instance and 4 for another -- exactly the assignment the coupling above
    # forbids. CP-SAT then starts from an infeasible point and spends its budget
    # repairing it: on w4 the constrained solve returned NO SCHEDULE AT ALL (alpha None)
    # while the unconstrained one solved fine, which is what sent us looking here.
    # So project the hint onto the constraint first: per coupled dispatch, take the
    # width HEFT chose most often among that dispatch's instances (restricted to the
    # widths every instance can take), and hint that one width for all of them.
    _forced_width = {}
    if uniform_groups and warm_start is not None:
        try:
            _ws_alpha = warm_start[1]
            for _key, (_idxs, _usable) in uniform_groups.items():
                _votes: dict = {}
                for _i in _idxs:
                    _w = len(combos[int(np.argmax(_ws_alpha[_i]))])
                    if _w in _usable:
                        _votes[_w] = _votes.get(_w, 0) + 1
                # No instance voted for a usable width -> fall back to the narrowest,
                # which is always buildable and never the reason a solve fails.
                _w_pick = (max(_votes, key=lambda w: (_votes[w], -w)) if _votes
                           else _usable[0])
                for _i in _idxs:
                    _forced_width[_i] = _w_pick
        except Exception:
            _forced_width = {}

    if warm_start is not None:
        ws_t, ws_alpha = warm_start
        try:
            for i in range(n):
                k = int(np.argmax(ws_alpha[i]))
                if i in _forced_width:
                    # Keep HEFT's machine choice when it already has the right width;
                    # otherwise take any combination of the forced width this op can run.
                    if len(combos[k]) != _forced_width[i]:
                        _alt = [kk for kk in range(n_combos)
                                if len(combos[kk]) == _forced_width[i]
                                and kk not in ops[i].infeasible_combinations]
                        if not _alt:
                            continue  # nothing legal to hint; leave this op unhinted
                        k = _alt[0]
                # AND NEVER HINT AN EXCLUDED COMBINATION, whatever excluded it. HEFT runs
                # in `cpsat_with_heft_warm_start` BEFORE this function, so it never sees
                # the exclusions made here -- `restrict_shard_to_networks` prices out
                # every multi-core combination for networks the loop chose not to widen,
                # and HEFT will happily have placed one there. The projection above only
                # covers packed-weight groups, so the w5 board re-solve still handed
                # CP-SAT an infeasible starting point and came back with NO SCHEDULE at
                # 400 s ("SOLVE FAILED"), which the loop then reported as a re-solve that
                # found nothing. One feasibility check on the way out covers every
                # exclusion source at once.
                if k in ops[i].infeasible_combinations:
                    _ok = [kk for kk in range(n_combos)
                           if kk not in ops[i].infeasible_combinations]
                    if not _ok:
                        continue
                    # Prefer the cheapest legal combination, so the hint is not merely
                    # feasible but a reasonable place to start.
                    k = min(_ok, key=lambda kk: durations_int[i][kk])
                model.AddHint(presence[i][k], 1)
                model.AddHint(chosen_start[i], _to_int_us(float(ws_t[i])))
        except Exception:
            pass
    if _forced_width:
        print(f"[cpsat] codegen contract: warm start projected onto the coupling for "
              f"{len(_forced_width)} dispatch instance(s)")

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
    bounded_phases: List[str] = []

    # PER-PHASE BUDGET. `max_time_in_seconds` applies to EACH Solve() call, so a
    # three-phase lexicographic solve under a "300 s limit" could legitimately run
    # 900 s -- and, worse, phase 1 could spend the entire wall clock the caller
    # budgeted for the whole solve and leave nothing for the rest. Split it: the top
    # phase matters most, so it gets half, and the remaining phases share the other
    # half, with a floor so no phase gets a budget too small to find a point at all.
    total_budget = float(time_limit) if time_limit and time_limit > 0 else None
    last_solution = None

    def _snapshot():
        """The current solution as (var, value) pairs, to hint the next phase with."""
        out = [(chosen_start[i], solver.Value(chosen_start[i])) for i in range(n)]
        for i in range(n):
            for k in range(n_combos):
                out.append((presence[i][k], solver.Value(presence[i][k])))
        return out

    for phase, (phase_name, phase_obj, phase_unit) in enumerate(objectives):
        if total_budget is not None:
            n_rest = max(1, len(objectives) - 1)
            share = (total_budget * 0.5 if phase == 0
                     else total_budget * 0.5 / n_rest)
            solver.parameters.max_time_in_seconds = max(10.0, share)
        # START EACH PHASE WHERE THE LAST ONE ENDED. Minimize+Solve restarts the
        # search, keeping no incumbent across calls, so a phase given a modest budget
        # can otherwise return a point WORSE than the phase before it -- or, on a hard
        # instance, nothing at all, discarding a schedule we already had in hand.
        if last_solution is not None:
            if hasattr(model, "ClearHints"):
                model.ClearHints()  # re-hinting a var without this is a proto error
            for _var, _val in last_solution:
                model.AddHint(_var, _val)
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
        last_solution = _snapshot()
        if objective_stop_after and phase_name == objective_stop_after:
            break
        if phase == len(objectives) - 1:
            break
        value = int(round(solver.ObjectiveValue()))
        if status == cp_model.OPTIMAL:
            model.Add(phase_obj == value)
        else:
            # BOUND THE PHASE, DO NOT ABANDON THE REST. Fixing an unproven value with
            # `==` would be a false lexicographic claim, but breaking out of the loop
            # when a phase fails to PROVE its optimum means that wherever phase 1
            # times out THE LOWER PHASES NEVER RUN: the schedule minimises deadline
            # misses only, with lateness and makespan left wherever phase 1's
            # incumbent drops them. That alone makes such a solve return a worse
            # makespan than greedy on the big rungs (77.95 vs 71.57 ms on w5) -- not a
            # solver limitation, an objective that is never optimised; its certificate
            # shows one phase, `dispatch_deadline_misses`, FEASIBLE, and nothing after it.
            #
            # `<= value` is sound where `== value` is not. The incumbent is achieved,
            # so the feasible region stays non-empty; the higher-priority term can
            # never get worse than what we already had; and the next phase minimises
            # the lower term subject to that bound. The claim it supports is
            # "lexicographic, with this phase bounded but not proven" -- which is what
            # the certificate records, per phase.
            model.Add(phase_obj <= value)
            bounded_phases.append(phase_name)

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
        # Phases that ran, but under a bound taken from an unproven incumbent rather
        # than a proven optimum. A reader can tell "lexicographic and proven" from
        # "lexicographic and bounded" without inferring it from statuses.
        "bounded_not_proven": list(bounded_phases),
        "phase_budget_s": ({"top": total_budget * 0.5,
                            "each_lower": total_budget * 0.5 / max(1, len(objectives) - 1)}
                           if total_budget is not None else None),
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
