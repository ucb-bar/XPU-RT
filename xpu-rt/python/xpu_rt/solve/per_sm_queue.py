"""Per-SM execution-queue solver for static megakernel scheduling.

Implements the planner half of Algorithm 1 in the Event Tensor Compiler
paper (Jin et al., MLSys '26).  Given a tile-task DAG with event-tensor
edges and an SM count, decide:

    1. Which SM owns each task (assignment).
    2. The execution order on each SM (queue).

while honouring all event-tensor dependencies (producer must finish before
consumer starts) and minimising the persistent megakernel's wall-clock
makespan.

We reuse OR-Tools CP-SAT (already a dependency of ``compgen.solve``) and
follow the interval-variable pattern already established in
``compgen.solve.schedule.solve_schedule``.

The solver is *compile-time*; the resulting per-SM queues are baked into
the persistent kernel as a pre-computed dispatch table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TileTask:
    """A single CTA-level task scheduled onto an SM.

    Attributes:
        task_id:     Stable identifier (typically ``"<device_func>:<i,j,...>"``).
        device_func: Name of the ``func.func`` body executed by this task.
        duration_us: Estimated wall-clock duration on a single SM.
        affinity_sm: Optional SM hint (preferred SM index, ``None`` = any).
    """

    task_id: str
    device_func: str
    duration_us: float = 1.0
    affinity_sm: int | None = None


@dataclass(frozen=True)
class EventEdge:
    """A producer -> consumer edge derived from an Event Tensor coord.

    Attributes:
        producer: Task id that emits ``event.notify``.
        consumer: Task id that emits ``event.wait``.
        event:    Event tensor name (informational, useful for debugging).
    """

    producer: str
    consumer: str
    event: str = ""


@dataclass(frozen=True)
class PerSMSchedule:
    """A solved per-SM execution plan.

    Attributes:
        assignment:    task_id -> SM index.
        per_sm_order:  SM index -> ordered list of task ids on that SM.
        start_times:   task_id -> compile-time relative start time (us).
        makespan_us:   total persistent-kernel wall time (us).
        feasible:      whether the solver returned a feasible plan.
        solve_time_ms: solver wall-clock.
    """

    assignment: dict[str, int] = field(default_factory=dict)
    per_sm_order: dict[int, list[str]] = field(default_factory=dict)
    start_times: dict[str, float] = field(default_factory=dict)
    makespan_us: float = float("inf")
    feasible: bool = False
    solve_time_ms: float = 0.0


def solve_per_sm_queue(
    tasks: list[TileTask],
    edges: list[EventEdge],
    sm_count: int,
    timeout_ms: int = 10000,
) -> PerSMSchedule:
    """Solve per-SM assignment + ordering for a static megakernel.

    Each task is an interval variable; tasks on the same SM cannot overlap;
    every ``EventEdge(producer, consumer)`` enforces ``end[producer] <=
    start[consumer]``; ``affinity_sm`` (when set) pins a task to an SM.

    Returns a PerSMSchedule with assignment + per-SM ordered queue.  When
    no feasible plan is found within ``timeout_ms`` the schedule comes back
    with ``feasible=False`` and a round-robin fallback assignment so the
    static-schedule pass can still emit *something* and surface the
    planner failure to the verification gate rather than crashing.
    """
    from ortools.sat.python import cp_model

    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")

    if not tasks:
        return PerSMSchedule(feasible=True, makespan_us=0.0)

    t0 = time.perf_counter()
    scale = 1000  # microseconds -> integer solver units

    horizon = int(sum(max(t.duration_us, 1e-3) for t in tasks) * scale) + 1
    model = cp_model.CpModel()

    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    intervals: dict[str, cp_model.IntervalVar] = {}
    sm_assign: dict[str, cp_model.IntVar] = {}

    for t in tasks:
        dur = max(int(t.duration_us * scale), 1)
        s = model.new_int_var(0, horizon, f"s_{t.task_id}")
        e = model.new_int_var(0, horizon, f"e_{t.task_id}")
        iv = model.new_interval_var(s, dur, e, f"iv_{t.task_id}")
        starts[t.task_id] = s
        ends[t.task_id] = e
        intervals[t.task_id] = iv

        if t.affinity_sm is not None:
            if not 0 <= t.affinity_sm < sm_count:
                raise ValueError(
                    f"task {t.task_id} affinity_sm={t.affinity_sm} out of range "
                    f"[0, {sm_count})"
                )
            sm = model.new_int_var(t.affinity_sm, t.affinity_sm, f"sm_{t.task_id}")
        else:
            sm = model.new_int_var(0, sm_count - 1, f"sm_{t.task_id}")
        sm_assign[t.task_id] = sm

    # Event-tensor dependency edges -> end[producer] <= start[consumer].
    for edge in edges:
        if edge.producer in ends and edge.consumer in starts:
            model.add(starts[edge.consumer] >= ends[edge.producer])

    # Per-SM no-overlap, expressed via *optional* intervals indexed by SM.
    for sm_idx in range(sm_count):
        sm_intervals: list[cp_model.IntervalVar] = []
        for t in tasks:
            on_this_sm = model.new_bool_var(f"on_{t.task_id}_sm{sm_idx}")
            model.add(sm_assign[t.task_id] == sm_idx).only_enforce_if(on_this_sm)
            model.add(sm_assign[t.task_id] != sm_idx).only_enforce_if(on_this_sm.Not())
            opt = model.new_optional_interval_var(
                starts[t.task_id],
                max(int(t.duration_us * scale), 1),
                ends[t.task_id],
                on_this_sm,
                f"opt_{t.task_id}_sm{sm_idx}",
            )
            sm_intervals.append(opt)
        if sm_intervals:
            model.add_no_overlap(sm_intervals)

    # Minimise makespan.
    makespan = model.new_int_var(0, horizon, "makespan")
    for t in tasks:
        model.add(makespan >= ends[t.task_id])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0
    status = solver.solve(model)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = {t.task_id: int(solver.value(sm_assign[t.task_id])) for t in tasks}
        st = {t.task_id: solver.value(starts[t.task_id]) / scale for t in tasks}
        per_sm: dict[int, list[str]] = {sm: [] for sm in range(sm_count)}
        for t in tasks:
            per_sm[assignment[t.task_id]].append(t.task_id)
        for sm in per_sm:
            per_sm[sm].sort(key=lambda tid: st[tid])
        return PerSMSchedule(
            assignment=assignment,
            per_sm_order={sm: q for sm, q in per_sm.items() if q},
            start_times=st,
            makespan_us=solver.value(makespan) / scale,
            feasible=True,
            solve_time_ms=solve_time_ms,
        )

    # Infeasible / timed out -> round-robin fallback so the caller can
    # still emit a kernel and the verification gate sees the failure.
    fallback_assign = {t.task_id: i % sm_count for i, t in enumerate(tasks)}
    fallback_per_sm: dict[int, list[str]] = {sm: [] for sm in range(sm_count)}
    for t in tasks:
        fallback_per_sm[fallback_assign[t.task_id]].append(t.task_id)
    return PerSMSchedule(
        assignment=fallback_assign,
        per_sm_order={sm: q for sm, q in fallback_per_sm.items() if q},
        feasible=False,
        solve_time_ms=solve_time_ms,
    )


__all__ = ["EventEdge", "PerSMSchedule", "TileTask", "solve_per_sm_queue"]
