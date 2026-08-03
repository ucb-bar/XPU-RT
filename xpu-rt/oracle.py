"""Analytic lower bounds on schedule makespan ("oracle floor").

A scheduler's reported makespan is only meaningful next to the
theoretical floor it could conceivably achieve. We compute the floor
from the workload alone, with no solver in the loop, and surface it
on every SchedulerReport so users can read the gap directly:

  gap_pct = (makespan_us - oracle_floor_us) / oracle_floor_us * 100

The floor is the max of three well-known bounds:

1. **Critical-path bound.** Longest chain of `min(processing_times)`
   per op (each op evaluated on its fastest-affinity hardware combo).
   Already implemented as `_lower_bound_makespan` in
   `scheduler_ml.py:71` — we reuse it directly.

2. **Load bound.** For each machine, sum the min-duration of every op
   that can only run on that machine; divide by parallelism factor
   (always 1 for fixed hetero — we don't split a single op across
   cores). The floor is the max across machines.

3. **Release-time bound.** For periodic ops with `min_start_t`, the
   floor cannot be earlier than `max(min_start_t + own_duration)`.

The combined floor is `max(critical_path, load, release)`. Compaction
+ MOSEK on a tight workload should reach (load, release) within
floating-point error; a wide gap signals either (a) a hard-to-pack
hetero workload, (b) a solver that gave up early, or (c) a bound that
isn't tight for this shape (rare).
"""

from __future__ import annotations

from typing import Any


def compute_floor(workload) -> dict[str, float]:
    """Return {critical_path_us, load_us, release_us, oracle_floor_us}.

    Pure-function: takes only a Workload, returns numbers. Safe to
    call before any solver runs.
    """
    cp = _critical_path_us(workload)
    ld = _load_us(workload)
    rl = _release_us(workload)
    return {
        "critical_path_us": float(cp),
        "load_us": float(ld),
        "release_us": float(rl),
        "oracle_floor_us": float(max(cp, ld, rl)),
    }


def _critical_path_us(workload) -> float:
    """Reuse scheduler_ml._lower_bound_makespan (critical path with
    fastest-device-per-op). Returns 0.0 on empty workloads."""
    ops = workload.get_operations()
    if not ops:
        return 0.0
    try:
        from scheduler_ml import _lower_bound_makespan
        return float(_lower_bound_makespan(workload))
    except Exception:
        # Self-contained fallback if scheduler_ml import fails.
        return _critical_path_fallback(workload)


def _critical_path_fallback(workload) -> float:
    """Pure-Python topological critical path. Used only if
    scheduler_ml._lower_bound_makespan can't be imported (e.g. headless
    test envs without numpy linkage)."""
    ops = workload.get_operations()
    n = len(ops)
    if n == 0:
        return 0.0
    idx = {id(op): i for i, op in enumerate(ops)}
    cp = [0.0] * n
    # Topo via DFS — small graphs only.
    visited = [False] * n
    order: list[int] = []

    def dfs(u: int) -> None:
        if visited[u]:
            return
        visited[u] = True
        for p in ops[u].get_predecessors():
            pi = idx.get(id(p))
            if pi is not None:
                dfs(pi)
        order.append(u)

    for u in range(n):
        dfs(u)

    for u in order:
        own_feasible = [c for c in ops[u].processing_times if c < 1e8]
        own = min(own_feasible) if own_feasible else 0.0
        max_pred = 0.0
        for p in ops[u].get_predecessors():
            pi = idx.get(id(p))
            if pi is not None and cp[pi] > max_pred:
                max_pred = cp[pi]
        cp[u] = max_pred + own
    return max(cp) if cp else 0.0


def _load_us(workload) -> float:
    """Per-machine load floor.

    For each machine, we sum the minimum duration of every op whose
    feasible combinations all include that machine. Returns the max
    across machines. This is a strict lower bound because, regardless
    of dependency structure, a machine must be busy at least that long
    to clear its own work."""
    ops = workload.get_operations()
    if not ops:
        return 0.0
    machines = list(workload.machines)
    combos = workload.get_machine_combinations()
    if not machines or not combos:
        return 0.0

    # For each op, pick its feasible-min duration combo and credit
    # every machine in that combo. This is the tightest single-op
    # contribution — placing the op anywhere else costs at least this
    # much wall-clock on this machine (since the op cannot run faster
    # than its min duration anywhere).
    load_by_machine: dict[str, float] = {m: 0.0 for m in machines}
    for op in ops:
        durs = list(op.processing_times)
        # Filter infeasible combos.
        feasible = [(k, d) for k, d in enumerate(durs) if d < 1e8]
        # Also drop hard-excluded combos.
        excluded = getattr(op, "infeasible_combinations", set())
        feasible = [(k, d) for k, d in feasible if k not in excluded]
        if not feasible:
            continue
        k_min, d_min = min(feasible, key=lambda kd: kd[1])
        if k_min >= len(combos):
            continue
        combo = combos[k_min]
        combo = combo if isinstance(combo, list) else [combo]
        for m in combo:
            if m in load_by_machine:
                load_by_machine[m] += d_min

    return max(load_by_machine.values()) if load_by_machine else 0.0


def _release_us(workload) -> float:
    """Largest (min_start_t + own_min_duration) across all ops that
    declare a release time. Periodic ops with min_start_t in the
    future force the makespan to be at least that late."""
    ops = workload.get_operations()
    if not ops:
        return 0.0
    best = 0.0
    for op in ops:
        rel = getattr(op, "min_start_t", None)
        if rel is None:
            continue
        feasible = [d for d in op.processing_times if d < 1e8]
        own = float(min(feasible)) if feasible else 0.0
        finish = float(rel) + own
        if finish > best:
            best = finish
    return best


def oracle_gap_pct(makespan_us: float, oracle_floor_us: float) -> float:
    """Convenience: (makespan - floor) / floor * 100. Returns Inf if
    floor == 0 (degenerate empty workload)."""
    if oracle_floor_us <= 0.0:
        return float("inf")
    return 100.0 * (makespan_us - oracle_floor_us) / oracle_floor_us
