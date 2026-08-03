"""Left-shift compaction post-pass for any scheduler output.

After a solver returns (t, alpha), some schedules (HEFT/PEFT/EDF/greedy
families) leave eliminable slack: a task placed at time `s` could
actually start earlier without violating any constraint, because the
list-scheduling heuristic placed it after a later-priority task that
slid in by accident.

This pass walks operations in current-start-time order and slides each
op's start to the earliest time that still satisfies:

  * Release time (Operation.min_start_t — periodic release windows).
  * All predecessor end times (data dependencies).
  * The most recent end time on its assigned machine combination
    (single-server / no-overlap on the same hardware).

MOSEK / CPSAT outputs are already tight, so the pass is a no-op
(idempotent) — we apply it uniformly so every solver's reported
makespan is "what the schedule actually requires" rather than "what
the heuristic happened to produce".

The pass never:
  * Reassigns alpha (machine-combo placement stays as the solver chose).
  * Reorders dependent ops.
  * Violates min_start_t.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def left_shift_compact(
    t: np.ndarray,
    alpha: np.ndarray,
    workload,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t', alpha) with each start slid left as far as constraints allow.

    Inputs:
      t      : (N,) float array of solver-assigned start times.
      alpha  : (N, K) boolean / one-hot machine-combo assignment.
      workload: xpu-rt Workload object — used for predecessors,
                durations, and machine combinations.

    Output:
      (t', alpha) — same shape; alpha is returned unchanged. t' is
      element-wise <= t (every start either earlier or unchanged).
    """
    if t is None or alpha is None:
        return t, alpha

    ops = workload.get_operations()
    n = len(ops)
    if n == 0:
        return t, alpha

    combos = workload.get_machine_combinations()
    machines = workload.machines

    # Map id(op) -> index for predecessor lookup. This mirrors what
    # scheduler.py does internally.
    idx_by_op = {id(op): i for i, op in enumerate(ops)}

    # alpha may be float (cvxpy returns floats near 0/1); resolve each
    # op's chosen combination by argmax.
    chosen_combo = np.asarray(alpha).argmax(axis=1).astype(int)

    # Pre-compute each op's duration on its chosen combination.
    duration = np.zeros(n, dtype=float)
    for i, op in enumerate(ops):
        duration[i] = float(op.get_duration_for_combination(
            int(chosen_combo[i]), combos, machines))

    # Map combo index -> set of machines covered, so we can detect
    # combos that share a machine (true single-server constraint runs
    # at the machine granularity, not the combo granularity).
    combo_machines = [set(combos[k]) for k in range(len(combos))]

    # Walk in current-start-time order — this matches the solver's
    # original placement order so we never "reorder" ops, only slide
    # them earlier. Ties broken by index for determinism.
    order = sorted(range(n), key=lambda i: (float(t[i]), i))

    new_t = np.array(t, dtype=float, copy=True)

    # Per-machine "last end" tracker — keyed by machine name (not combo
    # index) because two different combos that share a machine must
    # still respect single-server on that machine.
    machine_busy_until: dict[str, float] = {m: 0.0 for m in machines}

    for i in order:
        op = ops[i]
        k = int(chosen_combo[i])

        # 1. Release time (periodic ops have min_start_t set).
        earliest = float(op.min_start_t) if op.min_start_t is not None else 0.0

        # 2. All predecessor end times.
        for pred in op.get_predecessors():
            p_idx = idx_by_op.get(id(pred))
            if p_idx is None:
                # Some workloads pass predecessor as a sentinel or
                # operation_name string. Try to resolve by name.
                p_name = getattr(pred, "operation_name", None)
                if p_name is None:
                    continue
                for j, candidate in enumerate(ops):
                    if candidate.operation_name == p_name:
                        p_idx = j
                        break
                if p_idx is None:
                    continue
            earliest = max(earliest, float(new_t[p_idx] + duration[p_idx]))

        # 3. Single-server on every machine this op's combo touches.
        for m in combo_machines[k]:
            earliest = max(earliest, machine_busy_until[m])

        # Slide left (never right) — refuse to move an op later than the
        # solver placed it, even if the constraint check says we could.
        # (The solver may have a max_end_t constraint we don't model
        # here; preserving the original upper bound is the safe choice.)
        candidate = min(float(t[i]), earliest)

        # Phase A3: band-safe precheck. If the op declares max_end_t,
        # the post-shift finish must respect it. A pure left-shift can
        # only IMPROVE deadline satisfaction (finishing earlier never
        # makes a deadline-miss worse), so the only failure mode is when
        # the candidate position itself overruns the deadline — which
        # the solver already accepted (we can't fix it here, and
        # shifting left doesn't help). Document the invariant explicitly:
        # compaction is monotone in deadline slack.
        new_t[i] = candidate

        end_time = new_t[i] + duration[i]
        # Sanity: when max_end_t is defined, the compacted finish must
        # be <= max_end_t IF the original was. (Left-shift can't make a
        # deadline-compliant op miss.) Assert this — any violation is a
        # compaction bug, not workload data.
        if op.max_end_t is not None:
            original_finish = float(t[i]) + duration[i]
            if original_finish <= float(op.max_end_t) + 1e-6:
                assert end_time <= float(op.max_end_t) + 1e-6, (
                    f"compaction violated band invariant for "
                    f"{getattr(op, 'operation_name', '?')}: "
                    f"max_end_t={op.max_end_t}, new_finish={end_time}, "
                    f"original_finish={original_finish}"
                )
        for m in combo_machines[k]:
            machine_busy_until[m] = max(machine_busy_until[m], end_time)

    return new_t, alpha


def compaction_savings_us(
    t_before: np.ndarray,
    t_after: np.ndarray,
    alpha: np.ndarray,
    workload,
) -> dict:
    """Summarize how much the compaction pass moved makespan.

    Returns:
      {
        "makespan_before_us": float,
        "makespan_after_us":  float,
        "delta_us":           float,  # before - after, always >= 0
        "ops_moved":          int,    # how many ops slid earlier
        "max_op_shift_us":    float,  # largest single-op left-shift
      }
    """
    ops = workload.get_operations()
    combos = workload.get_machine_combinations()
    machines = workload.machines

    chosen_combo = np.asarray(alpha).argmax(axis=1).astype(int)
    duration = np.array([
        float(op.get_duration_for_combination(
            int(chosen_combo[i]), combos, machines))
        for i, op in enumerate(ops)
    ])

    end_before = t_before + duration
    end_after = t_after + duration
    return {
        "makespan_before_us": float(end_before.max()) if len(end_before) else 0.0,
        "makespan_after_us": float(end_after.max()) if len(end_after) else 0.0,
        "delta_us": float(end_before.max() - end_after.max()) if len(end_before) else 0.0,
        "ops_moved": int(np.sum(t_after < t_before - 1e-9)),
        "max_op_shift_us": float((t_before - t_after).max()) if len(t_before) else 0.0,
    }
