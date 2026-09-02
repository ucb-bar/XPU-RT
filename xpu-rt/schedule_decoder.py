"""Serial schedule generation scheme (SGS) shared by every search-based solver.

The greedy pickers in `greedy_scheduler` each hard-code their own priority
rule *and* their own placement loop, so a new rule means a new copy of the
placement logic. Everything in here separates the two:

    (priority key per op, machine-combination choice per op)  ->  (t, alpha)

`decode` walks the ops in priority order and places each one at the earliest
feasible start on its chosen combination, respecting data dependencies plus
transfer cost, machine-combination conflicts, and periodic `min_start_t`
windows. Placement is *insertion-based*: an op may drop into an idle gap left
by an earlier placement rather than only appending to the end of a lane, which
is what lets a search over priorities actually explore anything.

That makes the encoding a "random key" representation in the RCPSP sense
(Bean 1994; Kolisch & Hartmann's SGS comparison), so any optimiser that can
propose real-valued vectors — particle swarm, GA, simulated annealing — can
drive it without knowing anything about scheduling.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


class DecoderContext:
    """Immutable, workload-derived tables the decoder needs.

    Built once and reused across the thousands of decodes a metaheuristic
    performs; rebuilding per evaluation dominated the runtime otherwise.
    """

    def __init__(self, workload):
        self.workload = workload
        self.ops = workload.operations
        self.n = len(self.ops)
        self.machines = workload.machines
        self.combos = workload.get_machine_combinations()
        self.n_combos = len(self.combos)
        self.transfer = workload.get_transfer_times()

        idx = {id(op): i for i, op in enumerate(self.ops)}
        self.pred = [[idx[id(p)] for p in op.predecessors if id(p) in idx]
                     for op in self.ops]
        self.succ = [[] for _ in range(self.n)]
        for i, preds in enumerate(self.pred):
            for p in preds:
                self.succ[p].append(i)

        # durations[i, c]; +inf marks a combination the op cannot use.
        self.dur = np.full((self.n, self.n_combos), np.inf)
        for i, op in enumerate(self.ops):
            for c in range(self.n_combos):
                try:
                    self.dur[i, c] = float(
                        op.get_duration_for_combination(c, self.combos, self.machines))
                except Exception:
                    pass
        finite = np.where(np.isfinite(self.dur), self.dur, np.inf)
        self.min_dur = np.min(np.where(finite > 0, finite, np.inf), axis=1)
        self.min_dur[~np.isfinite(self.min_dur)] = 0.0

        # Which combinations conflict (share a machine), and the first machine
        # of each combination, for transfer-time lookups.
        self.conflict = [[workload.combinations_overlap(a, b)
                          for b in range(self.n_combos)]
                         for a in range(self.n_combos)]
        self.first_machine = [self.machines.index(self.combos[c][0])
                              for c in range(self.n_combos)]

        self.min_start = np.array(
            [float(op.min_start_t) if op.min_start_t is not None else 0.0
             for op in self.ops])
        self.max_end = np.array(
            [float(op.max_end_t) if op.max_end_t is not None else np.inf
             for op in self.ops])
        self.periodic = np.array([op.max_end_t is not None for op in self.ops])

        self.topo = self._topological_order()

    def _topological_order(self) -> list[int]:
        indeg = [len(p) for p in self.pred]
        stack = [i for i in range(self.n) if indeg[i] == 0]
        out = []
        while stack:
            u = stack.pop()
            out.append(u)
            for v in self.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if len(out) != self.n:            # cycle: fall back to input order
            return list(range(self.n))
        return out

    def upward_rank(self) -> np.ndarray:
        """HEFT's upward rank: the longest path from each op to a sink, using
        mean duration across usable combinations. Ops on the critical path get
        the largest values and so are scheduled first."""
        mean = np.where(np.isfinite(self.dur), self.dur, np.nan)
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(mean, axis=1)
        avg = np.nan_to_num(avg, nan=0.0)
        rank = np.zeros(self.n)
        for u in reversed(self.topo):
            best = 0.0
            for v in self.succ[u]:
                if rank[v] > best:
                    best = rank[v]
            rank[u] = avg[u] + best
        return rank


def decode(ctx: DecoderContext, priority: np.ndarray,
           combo_choice: np.ndarray | None = None,
           insertion: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Turn (priority, combination choice) into a concrete schedule.

    `combo_choice[i] < 0` (or a None vector) means "pick the combination that
    finishes this op earliest given what is already placed" — the HEFT rule.
    Otherwise the given combination is used, which is what lets a search
    optimise placement as well as ordering.
    """
    n, n_combos = ctx.n, ctx.n_combos
    t = np.zeros(n)
    alpha = np.zeros((n, n_combos))
    chosen = np.full(n, -1, dtype=int)
    # busy[c] holds (start, end) intervals already committed on combination c,
    # kept sorted so the gap scan below is a single pass.
    busy: list[list[tuple[float, float]]] = [[] for _ in range(n_combos)]

    order = _priority_order(ctx, priority)

    for i in order:
        pred_ready = 0.0
        for p in ctx.pred[i]:
            pc = chosen[p]
            end = t[p] + ctx.dur[p, pc]
            pred_ready = max(pred_ready, end)
        floor_base = max(pred_ready, ctx.min_start[i])

        best = None                       # (finish, combo, start)
        candidates = (range(n_combos) if combo_choice is None or combo_choice[i] < 0
                      else [int(combo_choice[i]) % n_combos])
        for c in candidates:
            d = ctx.dur[i, c]
            if not np.isfinite(d):
                continue
            floor = floor_base
            for p in ctx.pred[i]:
                pc = chosen[p]
                floor = max(floor, t[p] + ctx.dur[p, pc]
                            + ctx.transfer[ctx.first_machine[pc], ctx.first_machine[c]])
            start = _earliest_slot(ctx, busy, c, floor, d, insertion)
            finish = start + d
            if best is None or finish < best[0]:
                best = (finish, c, start)

        if best is None:                  # no usable combination; force one
            c = 0
            d = ctx.dur[i, c] if np.isfinite(ctx.dur[i, c]) else 0.0
            best = (floor_base + d, c, floor_base)

        _, c, start = best
        t[i] = start
        chosen[i] = c
        alpha[i, c] = 1.0
        d = ctx.dur[i, c]
        if np.isfinite(d) and d > 0:
            _commit(ctx, busy, c, start, start + d)

    return t, alpha


def _priority_order(ctx: DecoderContext, priority: np.ndarray) -> list[int]:
    """Precedence-feasible order, highest priority first among ready ops.

    A raw sort by priority would place ops before their predecessors; this
    walks the DAG so the order is always feasible and the priority vector only
    decides between ops that are *simultaneously* eligible.
    """
    indeg = [len(p) for p in ctx.pred]
    ready = [i for i in range(ctx.n) if indeg[i] == 0]
    order = []
    while ready:
        # argmax over the small ready set beats maintaining a heap here.
        pick = max(range(len(ready)), key=lambda k: priority[ready[k]])
        u = ready.pop(pick)
        order.append(u)
        for v in ctx.succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) != ctx.n:               # cycle: append the stragglers
        order.extend(i for i in range(ctx.n) if i not in set(order))
    return order


def _earliest_slot(ctx: DecoderContext, busy, combo: int, after: float,
                   duration: float, insertion: bool) -> float:
    """Earliest start >= `after` on `combo` where `duration` fits without
    overlapping anything already placed on a conflicting combination."""
    intervals = []
    for c2 in range(ctx.n_combos):
        if ctx.conflict[combo][c2]:
            intervals.extend(busy[c2])
    if not intervals:
        return after
    intervals.sort()
    if not insertion:
        return max(after, max(e for _s, e in intervals))
    cur = after
    for s, e in intervals:
        if e <= cur + _EPS:
            continue
        if s >= cur + duration - _EPS:
            return cur
        cur = max(cur, e)
    return cur


def _commit(ctx: DecoderContext, busy, combo: int, start: float, end: float) -> None:
    lst = busy[combo]
    k = 0
    while k < len(lst) and lst[k][0] < start:
        k += 1
    lst.insert(k, (start, end))


def evaluate(ctx: DecoderContext, t: np.ndarray, alpha: np.ndarray,
             restrict_to_nonperiodic: bool = True) -> tuple[float, int, float]:
    """(objective makespan, missed periodic windows, all-operations end).

    The objective mirrors the rest of the pipeline: when periodic ops have
    their own windows, the makespan being minimised is the non-periodic one.
    """
    combo = np.argmax(alpha, axis=1)
    finish = t + ctx.dur[np.arange(ctx.n), combo]
    finish = np.where(np.isfinite(finish), finish, t)
    all_end = float(finish.max()) if ctx.n else 0.0
    misses = int(np.sum(finish > ctx.max_end + 1e-6))
    if restrict_to_nonperiodic and np.any(~ctx.periodic):
        obj = float(finish[~ctx.periodic].max())
    else:
        obj = all_end
    return obj, misses, all_end
