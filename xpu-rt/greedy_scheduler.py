"""
List-scheduling greedy heuristic for the XPU-RT scheduler.

Companion to `scheduler.py` (the MILP-based optimal scheduler). Given a
:class:`Workload`, picks the (op, machine_combination) pair that gives
the earliest completion time, respecting:

  - intra-job data deps (op.predecessors + transfer_times),
  - machine-combination conflicts (combinations_overlap),
  - periodic / windowed time-bounds (op.min_start_t / op.max_end_t).

Three flavors are exported:

  - :func:`greedy_schedule` — vanilla list-scheduler. Picks the (op,
    combo) with the lowest completion time among ALL ready ops.
    Periodic and non-periodic ops compete on equal footing. Pre-2026-05
    behavior.

  - :func:`greedy_periodic_schedule` — non-periodic-priority variant.
    Among ready ops, schedules the lowest-completion non-periodic op
    first; periodic ops only get scheduled when no non-periodic is
    ready, with one exception: if delaying a periodic op would push it
    past its `max_end_t` window, it gets emergency-promoted.

    The motivation: with both yolov8 (non-periodic) and dronet
    (periodic 50ms) sharing a heterogeneous bitstream, vanilla greedy
    interleaves dronet-instance dispatches between yolov8 dispatches —
    every dronet dispatch displaces a yolov8 dispatch on its target
    core, growing yolov8's makespan. Non-periodic priority pins
    dronet-instances to "fill the gaps" left by yolov8, shrinking
    yolov8's makespan at the cost of slightly higher dronet jitter.

  - :func:`greedy_reserved_schedule` — same op ordering as
    `greedy_periodic`, but a periodic op picks the *least contended*
    combination that still meets its deadline instead of the one that
    finishes it soonest. Keeps the lanes the non-periodic critical path
    needs free; see that function's docstring for the 2.7x QRB5165 case.

This module used to live inline in `scripts/run_greedy_schedule.py`.
It moved here when run_greedy was folded into
`run_xpurt_schedule.py --solver {greedy,greedy_periodic}`, so both
schedulers ship as siblings under `xpu-rt/`.
"""

from __future__ import annotations

import numpy as np

from workload import Workload

# --- co-runner contention (additive, OFF by default) ---------------------
#
# `_CONTENTION` is None unless someone explicitly installs a measured model via
# `configure_contention()` or passes `contention=` to a scheduler entry point.
# While it is None, `_duration()` is exactly `op.get_duration_for_combination`,
# so a repo with no contention.json behaves bit-identically to before.
#
# The multiplier is applied at lookup time and never written back into the
# profile: a solo service time stays a solo service time on disk.
_CONTENTION = None
_CONTENTION_PLACEMENT = None


def configure_contention(model=None, placement=None) -> None:
    """Install (or clear) the contention model used by the duration lookup.

    `model=None` disables contention entirely — the default state.
    `placement` forces every lookup to use one placement; when left None the
    placement is derived per-combination by the model
    (`placement_for_combination`), which returns None for combinations it has
    no measurement for, leaving those durations untouched.
    """
    global _CONTENTION, _CONTENTION_PLACEMENT
    _CONTENTION = model
    _CONTENTION_PLACEMENT = placement


def get_contention():
    """Return (model, forced_placement). Mostly for tests and reporting."""
    return _CONTENTION, _CONTENTION_PLACEMENT


class _contention_scope:
    """Context manager so a per-call `contention=` argument cannot leak into
    the next caller's schedule."""

    def __init__(self, model, placement=None):
        self._model = model
        self._placement = placement

    def __enter__(self):
        self._prev = get_contention()
        if self._model is not None:
            configure_contention(self._model, self._placement)
        return self

    def __exit__(self, *exc):
        configure_contention(*self._prev)
        return False


def _duration(op, combo_idx, machine_combinations, machines) -> float:
    """Solo duration for (op, combination), scaled by the measured co-runner
    contention factor when a contention model is installed.

    Off by default: with `_CONTENTION is None` this is a straight passthrough.
    """
    base = op.get_duration_for_combination(combo_idx, machine_combinations, machines)
    model = _CONTENTION
    if model is None:
        return base
    placement = _CONTENTION_PLACEMENT
    if placement is None:
        try:
            placement = model.placement_for_combination(
                machine_combinations[combo_idx]
            )
        except Exception:
            placement = None
    if placement is None:
        return base
    try:
        factor = float(model.contention_factor(op, placement))
    except Exception:
        return base
    if factor <= 0:
        return base
    return base * factor

# Slack allowed when asking "does this placement still meet the deadline?".
# Durations are in ms, so this is a nanosecond-scale guard against a
# placement being rejected purely by floating-point round-off.
_EPS = 1e-9

# Default for how much slower than its own fastest combination a periodic op
# may run in exchange for staying off a lane the non-periodic jobs need
# (`greedy_reserved`). The premise of that trade — "finishing a periodic op
# early buys nothing, it only has to fit its window" — stops holding once the
# alternative lane is far slower: the op still fits, but it burns so much more
# machine time that the schedule as a whole gets worse. Unbounded, the spike
# mlp+dronet+yolov8 workload moves dronet from the RVV lane (28 ms) to the
# scalar lane (346 ms, 12.3x) and stretches the schedule from 694 to 967 ms.
#
# Swept over {1, 2, 4, 8, unbounded} on eleven spike/FireSim workloads: 2.0 is
# best-or-tied on ten of them (it is worth 30 ms on FireSim dronet@20ms+yolov8
# and 8 ms on dronet50+yolov8 static against a 4x cap), and it is the smallest
# cap that still reaches the valid 293.23 ms schedule on the FireSim
# mlp_control+dronet heterogeneous workload, where a 1.0 cap collapses to
# 1163.46 ms and doesn't even cover its own periodic demand.
#
# This is workload-family dependent, not universal: on the QRB5165
# 2x-resnet50 workload nothing below 8 finds the 21.81 ms schedule. Hence the
# per-run override (`scheduler.reserved_max_slowdown` in the workload spec, or
# --reserved-max-slowdown) rather than a constant everyone has to live with.
_RESERVED_MAX_SLOWDOWN = 2.0


def _op_indices(workload) -> dict[int, int]:
    """`id(op) -> position` map for ``workload.operations``, cached on the
    workload.

    ``workload.operations.index(pred)`` is a linear scan, and it sits in the
    innermost loop of every scheduling pass here (once per predecessor, per
    candidate op, per iteration). ``Operation`` defines no ``__eq__``, so
    ``list.index`` compares by identity — an ``id()`` keyed dict is exactly
    equivalent and O(1). On the 733-dispatch spike workload this is the
    difference between a quadratic and a linear inner loop.
    """
    cache = getattr(workload, "_op_index_cache", None)
    ops = workload.operations
    if cache is not None and cache[0] is ops and cache[1] == len(ops):
        return cache[2]
    idx = {id(op): i for i, op in enumerate(ops)}
    workload._op_index_cache = (ops, len(ops), idx)
    return idx


def _is_periodic_op(op) -> bool:
    """A periodic op is one whose `max_end_t` window-bound was set by
    `create_workload_from_network_hierarchy`. Non-periodic ops have
    `min_start_t = max_end_t = None`."""
    return op.max_end_t is not None


def _compute_alap_deadlines(workload, machine_combinations) -> dict:
    """Backward-propagate the per-instance window deadline (`op.max_end_t`)
    through the dependency DAG to give each periodic op its own
    chain-aware *effective deadline*.

    Why we need this: `op.max_end_t` is set by the workload constructor
    to a single window value per *instance* (e.g., dronet0 has
    max_end_t=20ms on every one of its 30 ops). Used directly, the
    slack of the first op is `20 - 1ms_op_duration ≈ 19ms`, which makes
    the picker think there's plenty of time — but the rest of the
    chain (29 more ops, ~9ms of work) means the first op actually has
    to *complete by t=11ms*, not t=20ms, for the instance to fit.

    Strategy: for each op, compute
        ALAP_completion[op] = max_end_t                       (sink)
                            = min over successors s of
                               (ALAP_completion[s] - duration(s))
    The op's effective deadline is then
        effective_max_end[op] = ALAP_completion[op]
    Use the *minimum* duration across machine combinations so we don't
    artificially tighten the deadline based on a slow combo we wouldn't
    actually pick.

    Returns: {op_idx: effective_max_end_t} for every periodic op.
    """
    n = len(workload.operations)
    # Min duration of each op across combos — used as the propagated
    # subtractive cost. Faster combo = tighter (more pessimistic)
    # deadline propagation, but only by however much that combo would
    # actually run faster, which is the right thing.
    min_dur = _min_durations(workload, machine_combinations)

    # Build successor index (op -> [op_idx of successors that depend on op])
    op_idx_of = _op_indices(workload)
    succ = [[] for _ in range(n)]
    for i, op in enumerate(workload.operations):
        for pred in op.predecessors:
            pred_idx = op_idx_of.get(id(pred))
            if pred_idx is not None:
                succ[pred_idx].append(i)

    # Topological order via DFS. Then process in reverse for ALAP propagation.
    visited = [False] * n
    topo = []
    def _dfs(u):
        if visited[u]:
            return
        visited[u] = True
        for v in succ[u]:
            _dfs(v)
        topo.append(u)
    for i in range(n):
        _dfs(i)
    # `topo` is now reverse-topological (sinks first when iterating forward).

    alap = {}
    for i in topo:
        op = workload.operations[i]
        if op.max_end_t is None:
            continue  # non-periodic
        if not succ[i]:
            # Sink — deadline is the instance's window end.
            alap[i] = float(op.max_end_t)
        else:
            # Tightest constraint: each successor's ALAP-completion must
            # leave room for that successor's own duration.
            tightest = float(op.max_end_t)
            for v in succ[i]:
                if v in alap:
                    candidate = alap[v] - min_dur[v]
                    if candidate < tightest:
                        tightest = candidate
            alap[i] = tightest
    return alap


def _min_durations(workload, machine_combinations) -> list[float]:
    """Per-op fastest duration across machine combinations (0 when the op
    has no positive duration anywhere)."""
    machines = workload.machines
    out = [0.0] * len(workload.operations)
    for i, op in enumerate(workload.operations):
        best = None
        for c in range(len(machine_combinations)):
            try:
                d = float(_duration(op, c, machine_combinations, machines))
            except Exception:
                continue
            if d > 0 and (best is None or d < best):
                best = d
        out[i] = best if best is not None else 0.0
    return out


def _nonperiodic_lane_demand(workload, machine_combinations) -> list[float]:
    """How much non-periodic work each machine combination would attract if
    every non-periodic op were free to pick its own fastest combination.

    This is the "comparative advantage" signal `greedy_reserved` places
    periodic ops against: a combination with a large demand here is one the
    makespan-critical (non-periodic) jobs genuinely need, so a periodic op
    that has any deadline-feasible alternative should stay off it.
    """
    demand = [0.0] * len(machine_combinations)
    machines = workload.machines
    for op in workload.operations:
        if _is_periodic_op(op):
            continue
        best_c, best_d = None, float("inf")
        for c in range(len(machine_combinations)):
            try:
                d = float(_duration(op, c, machine_combinations, machines))
            except Exception:
                continue
            if d > 0 and d < best_d:
                best_c, best_d = c, d
        if best_c is not None:
            demand[best_c] += best_d
    return demand


def _earliest_start_for(
    workload, op_idx, combo_idx, t, alpha, scheduled,
    combination_available_time, machines, machine_combinations,
    transfer_times,
):
    """Compute the earliest completion time for placing op_idx on combo_idx
    given the current schedule state. Returns (earliest_start, duration).
    Pure function — no side effects."""
    op = workload.operations[op_idx]
    # `combination_available_time[c]` is already the max end time over every
    # scheduled op sitting on a combination that overlaps `c` — the commit
    # paths below fan each placement out to all overlapping combinations. The
    # old code re-derived that same maximum by scanning every scheduled op on
    # every (op, combo) probe, which made each probe O(len(operations)); the
    # array lookup is equivalent and O(1).
    earliest_start = combination_available_time[combo_idx]

    # Wait for predecessors + their transfer cost into this combination's
    # first machine.
    op_idx_of = _op_indices(workload)
    for pred in op.predecessors:
        pred_idx = op_idx_of[id(pred)]
        pred_combo_idx = int(np.argmax(alpha[pred_idx, :]))
        pred_dur = _duration(workload.operations[pred_idx],
            pred_combo_idx, machine_combinations, machines
        )
        pred_end = t[pred_idx] + pred_dur
        pred_combo = machine_combinations[pred_combo_idx]
        cand_combo = machine_combinations[combo_idx]
        pred_machine_idx = machines.index(pred_combo[0])
        cand_machine_idx = machines.index(cand_combo[0])
        transfer = transfer_times[pred_machine_idx, cand_machine_idx]
        earliest_start = max(earliest_start, pred_end + transfer)

    # Honor periodic time-bounds (min_start_t = start + i*period for
    # instance i, max_end_t = min_start_t + window_duration).
    if op.min_start_t is not None:
        earliest_start = max(earliest_start, float(op.min_start_t))

    duration = _duration(op,
        combo_idx, machine_combinations, machines
    )
    return earliest_start, duration


def _schedule_loop(workload, mode: str,
                   max_slowdown: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shared scheduling loop. `mode` selects the priority discipline:

      - "greedy"          — earliest-completion across all ready ops.
      - "greedy_periodic" — earliest-completion among non-periodic ready
        ops; periodic only when no non-periodic is ready or when its
        max_end_t window is about to close.
      - "greedy_reserved" — the "greedy_periodic" op ordering, plus a
        different *combination* choice for periodic ops: instead of the
        combination that finishes them soonest, the one that leaves the
        makespan-critical lanes alone, among those that still meet the
        op's chain-aware deadline. See `greedy_reserved_schedule`.
    """
    assert mode in ("greedy", "greedy_periodic", "greedy_reserved"), (
        f"unknown mode {mode!r}"
    )

    num_operations = len(workload.operations)
    machines = workload.machines
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()
    op_idx_of = _op_indices(workload)

    # `greedy_reserved` needs, per periodic op, (a) how much the
    # non-periodic jobs want each lane and (b) the chain-aware deadline
    # that says which lanes are still fast enough. Both are static for the
    # whole pass, so compute them once.
    if mode == "greedy_reserved":
        lane_demand = _nonperiodic_lane_demand(workload, machine_combinations)
        alap = _compute_alap_deadlines(workload, machine_combinations)
        min_dur = _min_durations(workload, machine_combinations)
        if max_slowdown is None:
            max_slowdown = _RESERVED_MAX_SLOWDOWN
    else:
        lane_demand = None
        alap = None
        min_dur = None

    t = np.zeros(num_operations)
    alpha = np.zeros((num_operations, num_combinations))
    combination_available_time = np.zeros(num_combinations)
    scheduled = [False] * num_operations


    while not all(scheduled):
        # Best non-periodic candidate (always preferred under greedy_periodic).
        best_np_op = best_np_combo = None
        best_np_completion = float("inf")
        best_np_start = 0.0
        # Best periodic candidate (kept separately so we can check
        # "would deferring miss the window?").
        best_p_op = best_p_combo = None
        best_p_completion = float("inf")
        best_p_start = 0.0
        best_p_max_end = float("inf")

        # In greedy mode we only track one bucket — collapse both into a
        # single best_*_op stream by treating everything as non-periodic.
        treat_all_as_nonperiodic = (mode == "greedy")

        for i in range(num_operations):
            if scheduled[i]:
                continue
            op = workload.operations[i]

            # Predecessors must all be scheduled.
            can_schedule = True
            for pred in op.predecessors:
                pred_idx = op_idx_of.get(id(pred))
                if pred_idx is not None and not scheduled[pred_idx]:
                    can_schedule = False
                    break
            if not can_schedule:
                continue

            is_periodic = (not treat_all_as_nonperiodic) and _is_periodic_op(op)

            # Under greedy_reserved a periodic op picks its combination by
            # "cheapest lane that still makes the deadline" rather than
            # "soonest finish"; `op_best` collects that choice so the
            # cross-op comparison below still ranks by completion time.
            reserved = is_periodic and mode == "greedy_reserved"
            op_best_key = None
            op_best = None

            for combo_idx in range(num_combinations):
                earliest_start, duration = _earliest_start_for(
                    workload, i, combo_idx, t, alpha, scheduled,
                    combination_available_time, machines, machine_combinations,
                    transfer_times,
                )
                completion_time = earliest_start + duration

                if reserved:
                    deadline = alap.get(i, float("inf"))
                    # Bucket 0 = meets the chain-aware deadline *and* isn't
                    # a runaway slowdown against this op's own fastest
                    # combination; bucket 1 = neither, and only competes
                    # when bucket 0 is empty. Within a bucket, give up the
                    # lane the non-periodic jobs want most; break ties on
                    # finish time.
                    too_slow = duration > max_slowdown * min_dur[i]
                    misses = completion_time > deadline + _EPS
                    key = (1 if (misses or too_slow) else 0,
                           lane_demand[combo_idx], completion_time)
                    if op_best_key is None or key < op_best_key:
                        op_best_key = key
                        op_best = (completion_time, combo_idx, earliest_start)
                    continue

                # Window-miss candidates still compete by completion
                # time — same as the original greedy behavior. A
                # downstream validator surfaces any final miss.
                if is_periodic:
                    if completion_time < best_p_completion:
                        best_p_completion = completion_time
                        best_p_op = i
                        best_p_combo = combo_idx
                        best_p_start = earliest_start
                        best_p_max_end = (float(op.max_end_t)
                                          if op.max_end_t is not None
                                          else float("inf"))
                else:
                    if completion_time < best_np_completion:
                        best_np_completion = completion_time
                        best_np_op = i
                        best_np_combo = combo_idx
                        best_np_start = earliest_start

            if reserved and op_best is not None:
                completion_time, combo_idx, earliest_start = op_best
                if completion_time < best_p_completion:
                    best_p_completion = completion_time
                    best_p_op = i
                    best_p_combo = combo_idx
                    best_p_start = earliest_start
                    best_p_max_end = (float(op.max_end_t)
                                      if op.max_end_t is not None
                                      else float("inf"))

        # Decide what to schedule this iteration.
        chosen_op = chosen_combo = None
        chosen_start = 0.0

        if mode == "greedy":
            # All candidates already lumped into best_np_*.
            chosen_op, chosen_combo, chosen_start = (
                best_np_op, best_np_combo, best_np_start
            )
        else:  # greedy_periodic / greedy_reserved
            np_ready = best_np_op is not None
            p_ready = best_p_op is not None
            if not np_ready and not p_ready:
                chosen_op = None
            elif np_ready and not p_ready:
                chosen_op, chosen_combo, chosen_start = (
                    best_np_op, best_np_combo, best_np_start
                )
            elif p_ready and not np_ready:
                chosen_op, chosen_combo, chosen_start = (
                    best_p_op, best_p_combo, best_p_start
                )
            else:
                # Both ready. Default: schedule the non-periodic one and
                # defer the periodic. EXCEPT: if scheduling the
                # non-periodic first would push the periodic past its
                # max_end_t window, schedule the periodic now to avoid a
                # window miss.
                #
                # Estimate: if we schedule np now, then the periodic's
                # earliest_start can grow by at most the np's duration
                # (when both target overlapping combos). Conservatively
                # check whether the periodic's existing latest-start
                # already clears its window: if best_p_completion is
                # already past max_end_t, scheduling it now is the
                # least-bad option; otherwise we have slack to defer.
                np_duration = best_np_completion - best_np_start
                period_post_np_completion = best_p_completion + np_duration
                if period_post_np_completion > best_p_max_end:
                    chosen_op, chosen_combo, chosen_start = (
                        best_p_op, best_p_combo, best_p_start
                    )
                else:
                    chosen_op, chosen_combo, chosen_start = (
                        best_np_op, best_np_combo, best_np_start
                    )

        if chosen_op is None:
            # Cycle in the dep DAG (shouldn't happen if validation runs).
            # Force-place the first unscheduled op onto combo 0 so we make
            # progress instead of looping forever.
            for i in range(num_operations):
                if not scheduled[i]:
                    chosen_op = i
                    chosen_combo = 0
                    chosen_start = combination_available_time[0]
                    break

        # Commit the chosen (op, combo).
        t[chosen_op] = chosen_start
        alpha[chosen_op, chosen_combo] = 1.0
        scheduled[chosen_op] = True

        duration = _duration(workload.operations[chosen_op],
            chosen_combo, machine_combinations, machines
        )
        op_end = chosen_start + duration

        combination_available_time[chosen_combo] = op_end
        for combo_idx in range(num_combinations):
            if combo_idx == chosen_combo:
                continue
            if workload.combinations_overlap(chosen_combo, combo_idx):
                combination_available_time[combo_idx] = max(
                    combination_available_time[combo_idx], op_end
                )

    return t, alpha


def greedy_schedule(workload: Workload, contention=None, contention_placement=None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """List-scheduling greedy. Each iteration picks the (op, combo) with
    the lowest completion time among ready ops. Ties broken by the
    order operations appear in ``workload.operations``.

    `contention` optionally takes a `contention_model.ContentionModel`; with
    the default None the durations are the plain solo profile."""
    with _contention_scope(contention, contention_placement):
        return _schedule_loop(workload, mode="greedy")


def greedy_periodic_schedule(workload: Workload, contention=None,
                             contention_placement=None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Non-periodic-priority list-scheduling greedy. Among ready ops,
    schedules the lowest-completion non-periodic op first; periodic
    ops only get scheduled when no non-periodic is ready, with one
    exception: if scheduling the non-periodic first would push a
    periodic op past its max_end_t window, the periodic gets
    emergency-promoted.

    Use case: heterogeneous schedule where one network is non-periodic
    (e.g. yolov8) and one is periodic (e.g. dronet at 50ms). Vanilla
    greedy interleaves dronet dispatches between yolov8 dispatches,
    growing yolov8's makespan. This variant pins dronet to "fill the
    gaps" left by yolov8, shrinking yolov8's makespan at the cost of
    slightly higher dronet jitter (still within the periodic window).

    `contention` optionally takes a `contention_model.ContentionModel`; with
    the default None the durations are the plain solo profile."""
    with _contention_scope(contention, contention_placement):
        return _schedule_loop(workload, mode="greedy_periodic")


def greedy_reserved_schedule(workload: Workload,
                             max_slowdown: float | None = None,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Lane-reserving list scheduler: `greedy_periodic`'s op ordering with a
    contention-aware *combination* choice for periodic ops.

    The problem it fixes. `greedy`/`greedy_periodic`/`decomposed` all place
    every op on the combination that finishes it soonest. For a periodic op
    that is the wrong objective: finishing a 5 ms-period dronet instance in
    0.92 ms instead of 2.65 ms buys nothing (either fits the window), but it
    costs the DSP lane — which may be the only lane where the non-periodic
    job that actually defines the makespan runs well. On the QRB5165 3-way
    workload (dronet @5 ms + mlp_control @2 ms + non-periodic yolov8n) that
    single decision is worth 2.7x: dronet takes DSP because it is fastest
    there, so yolov8n is pushed off its own best lane and the makespan grows
    from 33.57 ms to 90.13 ms. Here dronet runs happily on the otherwise
    idle HTA lane at 2.65 ms — comfortably inside its 5 ms window — and
    yolov8n keeps the DSP to itself.

    The rule. A periodic op ranks its candidate combinations by

        (misses its deadline, or is more than `max_slowdown` times
         slower than this op's own fastest lane?,
         how much the non-periodic jobs want this lane,
         completion time)

    so it takes the least contended lane that still meets its chain-aware
    ALAP deadline without running absurdly slower there, and falls back to
    the rejected lanes only when nothing else qualifies. Non-periodic ops
    keep pure earliest-completion — their lanes are the objective. "How much
    the non-periodic jobs want this lane" is `_nonperiodic_lane_demand`: the
    total non-periodic work that would land there under free choice.

    Cost: one extra pass over the ops for the demand vector, the ALAP map
    and the per-op minimum durations, all O(ops x combinations). The
    scheduling loop itself is unchanged.
    """
    return _schedule_loop(workload, mode="greedy_reserved",
                          max_slowdown=max_slowdown)


def _per_instance_independent_makespan(
    workload, machine_combinations, instance_op_idxs: list[int]
) -> float:
    """Schedule a single instance (set of ops belonging to one network
    invocation, e.g. dronet0's 30 ops) on its own — no contention from
    other instances — and return the resulting makespan.

    Used by `decomposed_schedule` for two purposes:
      (a) sanity-check feasibility: is the per-instance chain even
          short enough to fit its `window_duration`?
      (b) compute non-periodic critical-path makespan for ALAP
          deadlines.

    The mini-schedule uses the same `_schedule_loop` machinery but on
    a slice of ops with all dependencies preserved. We don't actually
    use the resulting (t, alpha) — only the makespan."""
    if not instance_op_idxs:
        return 0.0
    op_set = set(instance_op_idxs)
    machines = workload.machines
    transfer_times = workload.get_transfer_times()
    num_combinations = len(machine_combinations)
    op_idx_of = _op_indices(workload)

    # Local schedule state restricted to the instance's ops.
    local_t = {}
    local_combo = {}
    combo_avail = [0.0] * num_combinations
    remaining = set(instance_op_idxs)

    while remaining:
        best = None  # (completion, op_idx, combo_idx, start)
        for op_idx in remaining:
            op = workload.operations[op_idx]
            # Check predecessors: only those within the instance count.
            ok = True
            for pred in op.predecessors:
                pred_idx = op_idx_of.get(id(pred))
                if pred_idx is None:
                    continue
                if pred_idx in op_set and pred_idx not in local_t:
                    ok = False
                    break
            if not ok:
                continue

            for c in range(num_combinations):
                start = combo_avail[c]
                # Predecessor-end constraints (within instance).
                for pred in op.predecessors:
                    pred_idx = op_idx_of.get(id(pred))
                    if pred_idx is None or pred_idx not in op_set:
                        continue
                    pred_dur = _duration(workload.operations[pred_idx],
                        local_combo[pred_idx], machine_combinations, machines)
                    pred_end = local_t[pred_idx] + pred_dur
                    pc = machine_combinations[local_combo[pred_idx]]
                    cc = machine_combinations[c]
                    transfer = transfer_times[
                        machines.index(pc[0]), machines.index(cc[0])]
                    start = max(start, pred_end + transfer)
                if op.min_start_t is not None:
                    start = max(start, float(op.min_start_t))
                try:
                    dur = _duration(op, c, machine_combinations, machines)
                except Exception:
                    continue
                completion = start + dur
                if best is None or completion < best[0]:
                    best = (completion, op_idx, c, start)
        if best is None:
            # Cycle in the local DAG; bail.
            break
        completion, op_idx, c, start = best
        local_t[op_idx] = start
        local_combo[op_idx] = c
        # Mark this combo + overlapping combos busy until completion.
        combo_avail[c] = completion
        for c2 in range(num_combinations):
            if c2 != c and workload.combinations_overlap(c, c2):
                combo_avail[c2] = max(combo_avail[c2], completion)
        remaining.remove(op_idx)

    if not local_t:
        return 0.0
    return max(
        local_t[i] + _duration(workload.operations[i],
            local_combo[i], machine_combinations, machines)
        for i in local_t
    )


def _ops_by_job(workload) -> dict[str, list[int]]:
    """Group op indices by job_name (e.g., {'yolov8_nano': [...], 'dronet0': [...], ...}).

    The Operation class stores job_id (an int); the human-readable
    `job_name` lives in `workload.job_names` indexed by job_id."""
    job_names = getattr(workload, "job_names", []) or []
    by_job = {}
    for i, op in enumerate(workload.operations):
        jid = getattr(op, "job_id", None)
        if jid is not None and 0 <= jid < len(job_names) and job_names[jid]:
            name = job_names[jid]
        elif jid is not None:
            name = f"job_{jid}"
        else:
            name = "?"
        by_job.setdefault(name, []).append(i)
    return by_job


def _compute_alap_deadlines_with_np(
    workload, machine_combinations,
    np_makespan_target: float,
) -> dict[int, float]:
    """ALAP deadline propagation that *also* assigns deadlines to
    non-periodic ops, treating their critical-path makespan as a soft
    `max_end_t`.

    Periodic ops keep their windowed `max_end_t` (per-instance); non-
    periodic ops get `np_makespan_target` as their deadline anchor and
    propagate it backwards through their chain just like periodic ops
    do. This lets EDF reason about both kinds of ops in the same way:
    the op with the *earliest* effective deadline wins.

    Why this works for the user's "push yolov8 back" preference:
    periodic deadlines (e.g. 20ms) are typically much smaller than
    yolov8's makespan target (e.g. 115ms), so periodic ops naturally
    win EDF when they're ready — but yolov8 still has a defined
    deadline so it doesn't starve forever in pathological cases.
    """
    n = len(workload.operations)

    # Min duration across combos.
    min_dur = _min_durations(workload, machine_combinations)

    # Successor index.
    op_idx_of = _op_indices(workload)
    succ = [[] for _ in range(n)]
    for i, op in enumerate(workload.operations):
        for pred in op.predecessors:
            pred_idx = op_idx_of.get(id(pred))
            if pred_idx is not None:
                succ[pred_idx].append(i)

    # Reverse-topological order via iterative DFS (avoids recursion-depth
    # issues on large graphs).
    visited = [False] * n
    topo = []
    for root in range(n):
        if visited[root]:
            continue
        stack = [(root, iter(succ[root]))]
        visited[root] = True
        while stack:
            u, it = stack[-1]
            try:
                v = next(it)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, iter(succ[v])))
            except StopIteration:
                topo.append(u)
                stack.pop()
    # `topo` is post-order: sinks appear first when iterating forward.

    alap = {}
    for i in topo:
        op = workload.operations[i]
        if op.max_end_t is not None:
            base = float(op.max_end_t)
        else:
            base = float(np_makespan_target)
        if not succ[i]:
            alap[i] = base
        else:
            tightest = base
            for v in succ[i]:
                if v in alap:
                    candidate = alap[v] - min_dur[v]
                    if candidate < tightest:
                        tightest = candidate
            alap[i] = tightest
    return alap


def decomposed_schedule(workload: Workload, contention=None,
                        contention_placement=None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase decomposed scheduling, optionally with measured co-runner
    contention applied to the duration lookup (`contention=` a
    `contention_model.ContentionModel`; None = plain solo profile)."""
    with _contention_scope(contention, contention_placement):
        return _decomposed_schedule(workload)


def _decomposed_schedule(workload: Workload) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase decomposed scheduling: periodic first, non-periodic backfills.

    Phase 1 — periodic placement (EDF over chain-aware deadlines):
        Schedule every periodic op into its window using EDF list
        scheduling. Each op gets its ALAP-propagated max_end_t as
        priority, so the first op of a chain has a tighter effective
        deadline than the last op (accounting for downstream chain work).
        Periodic ops freely pick whichever (combo, time) gives the
        earliest completion — this is the HW-relaxation step the user
        asked for: an op nominally suited to gemmini may end up on the
        RVV hart when gemmini's slot is contested.

    Phase 2 — non-periodic backfill:
        Schedule the remaining (non-periodic) ops using plain list
        scheduling with earliest-completion priority. Non-periodic
        ops have no hard deadline; they take whatever slots the
        periodic placement left. yolov8's makespan is therefore an
        *output* of the schedule, not a constraint, and grows by
        however much periodic load it has to absorb.

    Why two-phase instead of single-pass EDF:
        A single-pass EDF that *also* assigns deadlines to non-periodic
        ops via critical-path ALAP gives yolov8's first op an
        effective deadline near zero (since the chain that follows it
        is ~its full standalone makespan). That makes yolov8 win EDF
        unfairly at t=0, defeating the periodic-first intent. Phase 1
        scheduling periodic in isolation cleanly avoids this.

    Closest names in the literature: *slack stealing* and *server-based
    scheduling* (real-time systems), *cyclic executive + backfill*
    (avionics), *list scheduling with reservation* (HPC).
    """
    num_operations = len(workload.operations)
    machines = workload.machines
    machine_combinations = workload.get_machine_combinations()
    num_combinations = len(machine_combinations)
    transfer_times = workload.get_transfer_times()
    op_idx_of = _op_indices(workload)

    # Per-job makespan (diagnostic) and ALAP-propagated deadlines.
    by_job = _ops_by_job(workload)
    np_jobs = [
        name for name, op_idxs in by_job.items()
        if not any(workload.operations[i].max_end_t is not None for i in op_idxs)
    ]
    p_jobs = [name for name in by_job if name not in np_jobs]
    print(
        f"decomposed_schedule: {len(by_job)} jobs total "
        f"({len(p_jobs)} periodic, {len(np_jobs)} non-periodic)"
    )

    # ALAP deadlines for periodic ops only. Non-periodic ops are backfilled
    # in Phase 2 with no deadline (priority = earliest completion).
    alap = _compute_alap_deadlines(workload, machine_combinations)

    t = np.zeros(num_operations)
    alpha = np.zeros((num_operations, num_combinations))
    combination_available_time = np.zeros(num_combinations)
    scheduled = [False] * num_operations

    def _ready(op_idx: int) -> bool:
        """All predecessors of op_idx already scheduled?"""
        op = workload.operations[op_idx]
        for pred in op.predecessors:
            pred_idx = op_idx_of.get(id(pred))
            if pred_idx is None:
                continue
            if not scheduled[pred_idx]:
                return False
        return True

    def _commit(op_idx: int, combo_idx: int, start: float) -> None:
        t[op_idx] = start
        alpha[op_idx, combo_idx] = 1.0
        scheduled[op_idx] = True
        duration = _duration(workload.operations[op_idx],
            combo_idx, machine_combinations, machines)
        end = start + duration
        combination_available_time[combo_idx] = end
        for c2 in range(num_combinations):
            if c2 != combo_idx and workload.combinations_overlap(combo_idx, c2):
                combination_available_time[c2] = max(
                    combination_available_time[c2], end)

    # ---- Phase 1: periodic ops, EDF over ALAP deadlines. ----
    periodic_idxs = {i for i, op in enumerate(workload.operations)
                     if op.max_end_t is not None}
    while True:
        # Pick the unscheduled periodic op with smallest ALAP deadline,
        # among those whose predecessors are all scheduled.
        best = None  # (deadline, completion, op_idx, combo_idx, start)
        for i in periodic_idxs:
            if scheduled[i] or not _ready(i):
                continue
            for c in range(num_combinations):
                start, duration = _earliest_start_for(
                    workload, i, c, t, alpha, scheduled,
                    combination_available_time, machines, machine_combinations,
                    transfer_times,
                )
                completion = start + duration
                deadline = alap.get(i, float("inf"))
                key = (deadline, completion)
                if best is None or key < best[0]:
                    best = (key, i, c, start)
        if best is None:
            break  # all periodic scheduled (or none ready — should not happen)
        _, i, c, start = best
        _commit(i, c, start)

    # ---- Phase 2: non-periodic ops, with interval-based backfill. ----
    # Build per-combo free-interval lists from the periodic placement.
    # Each combo has a list of [start, end) intervals where it's busy
    # (collected from Phase 1's scheduled ops). Yolov8 ops slot into
    # the gaps in earliest-completion order. This is the "squeeze yolov8
    # into the holes between periodic" behavior the user asked for.
    busy = [[] for _ in range(num_combinations)]
    for i in range(num_operations):
        if not scheduled[i]:
            continue
        c = int(np.argmax(alpha[i]))
        dur = _duration(workload.operations[i],
            c, machine_combinations, machines)
        busy[c].append((float(t[i]), float(t[i]) + float(dur)))
        # Reflect onto overlapping combos so they see the same busy windows.
        for c2 in range(num_combinations):
            if c2 != c and workload.combinations_overlap(c, c2):
                busy[c2].append((float(t[i]), float(t[i]) + float(dur)))
    for c in range(num_combinations):
        busy[c].sort()

    def _earliest_free_slot(combo_idx: int, after: float, duration: float) -> float:
        """Find the earliest start time on `combo_idx` that is >= `after`
        and where a duration-long window fits before the next busy
        interval. busy[combo_idx] is sorted by start. Returns the slot's
        start time."""
        cur = after
        for s, e in busy[combo_idx]:
            if e <= cur:
                continue  # busy interval entirely before us
            if s >= cur + duration:
                return cur  # fits entirely before this busy interval
            cur = max(cur, e)
        return cur

    def _np_start_for(op_idx: int, combo_idx: int) -> tuple[float, float]:
        """Compute earliest_start for a non-periodic op + combo,
        considering both predecessors AND backfill into Phase 1's gaps."""
        op = workload.operations[op_idx]
        # Predecessor end + transfer.
        pred_floor = 0.0
        for pred in op.predecessors:
            pred_idx = op_idx_of.get(id(pred))
            if pred_idx is None:
                continue
            if not scheduled[pred_idx]:
                return float("inf"), 0.0
            pc = int(np.argmax(alpha[pred_idx]))
            pred_dur = _duration(workload.operations[pred_idx],
                pc, machine_combinations, machines)
            pred_end = float(t[pred_idx]) + float(pred_dur)
            transfer = transfer_times[
                machines.index(machine_combinations[pc][0]),
                machines.index(machine_combinations[combo_idx][0])]
            pred_floor = max(pred_floor, pred_end + float(transfer))
        try:
            duration = float(_duration(op,
                combo_idx, machine_combinations, machines))
        except Exception:
            return float("inf"), 0.0
        # duration==0 is a legitimate "no-op" / structural op (e.g., the
        # zero-duration sync ops yolov8 emits between blocks). Schedule them
        # without backfilling — they don't consume time.
        if duration < 0:
            return float("inf"), 0.0
        if duration == 0:
            return pred_floor, 0.0
        start = _earliest_free_slot(combo_idx, pred_floor, duration)
        return start, duration

    def _commit_np(op_idx: int, combo_idx: int, start: float, duration: float) -> None:
        """Like _commit but also updates busy-interval lists for backfill."""
        t[op_idx] = start
        alpha[op_idx, combo_idx] = 1.0
        scheduled[op_idx] = True
        end = start + duration
        # Update busy intervals on this combo + overlapping combos.
        for c2 in range(num_combinations):
            if c2 == combo_idx or workload.combinations_overlap(combo_idx, c2):
                # Insert (start, end) keeping sort order.
                lst = busy[c2]
                idx = 0
                while idx < len(lst) and lst[idx][0] < start:
                    idx += 1
                lst.insert(idx, (start, end))
        # Keep combination_available_time loosely up to date so postprocessing
        # downstream remains consistent.
        combination_available_time[combo_idx] = max(
            combination_available_time[combo_idx], end)

    np_idxs = {i for i in range(num_operations) if i not in periodic_idxs}
    while True:
        best = None  # (completion, op_idx, combo_idx, start, duration)
        for i in np_idxs:
            if scheduled[i] or not _ready(i):
                continue
            for c in range(num_combinations):
                start, duration = _np_start_for(i, c)
                completion = start + duration
                if best is None or completion < best[0]:
                    best = (completion, i, c, start, duration)
        if best is None:
            break
        _, i, c, start, duration = best
        _commit_np(i, c, start, duration)

    # Defensive: any unscheduled ops? (cycles, missed cases)
    for i in range(num_operations):
        if not scheduled[i]:
            _commit(i, 0, combination_available_time[0])

    return t, alpha
