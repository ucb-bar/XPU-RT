"""Sender's delay: re-time a finished schedule for the cost of waking another worker.

On the board every core has its own worker thread. When a layer finishes and a layer on ANOTHER
core is asleep waiting for its result, the finishing worker has to wake that thread before it can
go on: its own next layer starts late (about 8 us on the K3; waking two threads costs about the
same as one). The receiver's wait is charged by the transfer table (`scheduler.handoff_ms`); this
module charges the sender, with `scheduler.sender_ms` = {core kind: ms}.

`apply_sender_delay` keeps every placement and the order on every core, and only ever moves starts
later:
  * a layer is a SENDER if one of its successors sits on another leading core and, in the given
    schedule, starts as soon as the sender's result can reach it (so its thread was waiting);
  * the next layer LED BY the sender's leading core cannot start before sender end + delay. A layer
    led by another core that merely uses that core is not held up (the core is unlocked first);
  * the shift then ripples through successors and through whatever follows on the same cores.
Zero-cost layers take no core time, as in the planner.

Limits: the solver does not see this cost when it places layers; it only makes the predicted times
honest. Hooked for the constructive solvers (greedy family, decomposed) in run_xpurt_schedule.py.
"""
from __future__ import annotations

import numpy as np

from workload_factory import machine_type_prefix

#: slack when testing "starts as soon as the result arrives" (ms); far below any real duration
_BOUND_EPS_MS = 1e-6


def _sender_ms_by_machine(machines: list[str], sender_ms: dict) -> list[float]:
    if not isinstance(sender_ms, dict):
        raise ValueError(f"scheduler.sender_ms must be {{core kind: ms}}, got {sender_ms!r}")
    by_kind = {str(k).strip().lower(): v for k, v in sender_ms.items()}
    out = []
    for name in machines:
        kind = machine_type_prefix(name).lower()
        value = by_kind.get(kind)
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok or not np.isfinite(value) or value < 0:
            raise ValueError(f"scheduler.sender_ms needs a number >= 0 (ms) for {kind}, got {value!r}")
        out.append(float(value))
    return out


def _topological_rank(preds: list[list[int]]) -> list[int]:
    """Rank such that every predecessor has a smaller rank than its successors (iterative DFS)."""
    n = len(preds)
    rank = [-1] * n
    counter = 0
    for root in range(n):
        if rank[root] >= 0:
            continue
        stack = [(root, 0)]
        while stack:
            node, k = stack.pop()
            if rank[node] >= 0:
                continue
            if k < len(preds[node]):
                stack.append((node, k + 1))
                if rank[preds[node][k]] < 0:
                    stack.append((preds[node][k], 0))
            else:
                rank[node] = counter
                counter += 1
    return rank


def apply_sender_delay(workload, t, alpha, sender_ms: dict | None) -> np.ndarray:
    """Return new start times with the sender's delay applied; `t` and `alpha` are not modified.

    `sender_ms` maps a core kind (case-insensitive) to the delay in ms for a sender led by a core
    of that kind. None or {} returns a copy of `t`.
    """
    t = np.asarray(t, dtype=float)
    if not sender_ms:
        return t.copy()
    ops = workload.operations
    machines = workload.machines
    combos = workload.get_machine_combinations()
    transfer = workload.get_transfer_times()
    delay_of_machine = _sender_ms_by_machine(machines, sender_ms)
    machine_idx = {name: k for k, name in enumerate(machines)}
    n = len(ops)

    index_of = {id(op): k for k, op in enumerate(ops)}
    combo = [int(np.argmax(alpha[k])) for k in range(n)]
    dur = [float(ops[k].get_duration_for_combination(combo[k], combos, machines)) for k in range(n)]
    cores = [[machine_idx[m] for m in combos[combo[k]]] for k in range(n)]
    lead = [c[0] for c in cores]
    preds = [[index_of[id(p)] for p in ops[k].predecessors if id(p) in index_of] for k in range(n)]
    succs: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        for p in preds[k]:
            succs[p].append(k)

    def wakes_another_thread(k: int) -> bool:
        end = t[k] + dur[k]
        return any(lead[s] != lead[k] and t[s] <= end + transfer[lead[k], lead[s]] + _BOUND_EPS_MS
                   for s in succs[k])

    is_sender = [wakes_another_thread(k) for k in range(n)]
    rank = _topological_rank(preds)
    new_t = t.copy()
    new_end = [0.0] * n
    core_free = [0.0] * len(machines)       # when each core's last layer ended
    worker_free = [0.0] * len(machines)     # when each core's own worker can start its next layer
    for k in sorted(range(n), key=lambda i: (t[i], rank[i])):
        start = float(t[k])
        for p in preds[k]:
            start = max(start, new_end[p] + float(transfer[lead[p], lead[k]]))
        if dur[k] > 0:
            start = max(start, worker_free[lead[k]], max(core_free[c] for c in cores[k]))
        new_t[k] = start
        new_end[k] = start + dur[k]
        if dur[k] > 0:
            for c in cores[k]:
                core_free[c] = max(core_free[c], new_end[k])
            extra = delay_of_machine[lead[k]] if is_sender[k] else 0.0
            worker_free[lead[k]] = max(worker_free[lead[k]], new_end[k] + extra)
    return new_t
