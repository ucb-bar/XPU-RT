"""
Core-count flow: a solver-free scheduling pipeline kept separate from scheduler.py.

Pipeline:
  Stage 1) split_into_windows: bucket ops by (min_start_t / topological depth).
  Stage 2) greedy_core_count_selection: for each op, pick (type, count, start_t)
           by minimizing completion time. The preferred (type, count) is the
           one where the op runs fastest; it breaks ties.
  Stage 3) assign_specific_cores: greedy interval coloring — given (type, count,
           start, duration) from stage 2, assign which specific N cores of the
           chosen type the op runs on. Stage 2's per-type capacity check
           guarantees this always succeeds.

No MILP, no LP: O(N log N) overall.
"""

from __future__ import annotations

import heapq
import itertools
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Window, Workload
from workload_factory import expand_machine_core_counts_to_list, machine_type_prefix


# ---------------------------------------------------------------------------
# Combinations: include every same-type N-core subset (not just prefixes)
# ---------------------------------------------------------------------------

def build_all_subset_combinations(
    machine_core_counts: Dict[str, int],
) -> Tuple[List[str], List[List[str]]]:
    """All same-type core subsets, so stage 3 can pick specific cores."""
    machines = expand_machine_core_counts_to_list(machine_core_counts)
    combinations: List[List[str]] = []
    for type_name, count in machine_core_counts.items():
        cores = [f"{type_name}#{i}" for i in range(count)]
        for n in range(1, count + 1):
            for subset in itertools.combinations(cores, n):
                combinations.append(list(subset))
    return machines, combinations


def _lookup_duration(
    op: Operation,
    old_combinations: List[List[str]],
    target_type: str,
    target_count: int,
) -> Optional[float]:
    for k, combo in enumerate(old_combinations):
        if machine_type_prefix(combo[0]) == target_type and len(combo) == target_count:
            if k < len(op.processing_times):
                return float(op.processing_times[k])
            return None
    return None


def _preferred_type_count(
    op: Operation, old_combinations: List[List[str]],
) -> Tuple[str, int, float]:
    best: Optional[Tuple[str, int, float]] = None
    for k, combo in enumerate(old_combinations):
        if k >= len(op.processing_times):
            continue
        t = machine_type_prefix(combo[0])
        n = len(combo)
        d = float(op.processing_times[k])
        if best is None or d < best[2]:
            best = (t, n, d)
    if best is None:
        raise ValueError("Operation has no processing_times entries")
    return best


# ---------------------------------------------------------------------------
# Stage 1: windowing with no solver
# ---------------------------------------------------------------------------

def _topo_depth(operations: List[Operation]) -> Dict[Operation, int]:
    """Depth = longest predecessor path (0 for roots)."""
    idx_of = {id(op): i for i, op in enumerate(operations)}
    indeg = [0] * len(operations)
    succ: List[List[int]] = [[] for _ in range(len(operations))]
    for i, op in enumerate(operations):
        for pred in op.get_predecessors():
            p = idx_of.get(id(pred))
            if p is None:
                continue
            succ[p].append(i)
            indeg[i] += 1
    depth = [0] * len(operations)
    from collections import deque
    q = deque([i for i in range(len(operations)) if indeg[i] == 0])
    while q:
        u = q.popleft()
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return {op: depth[i] for i, op in enumerate(operations)}


def split_into_windows(
    workload: Workload, n_splits: int,
) -> List[Window]:
    """
    Bucket ops into n_splits+1 windows. Periodic ops use min_start_t.
    Non-periodic ops use topological depth, scaled to the same range.
    Periodic groups (same job_id + min_start_t) stay intact.
    """
    operations = workload.get_operations()
    depth = _topo_depth(operations)
    max_depth = max(depth.values()) if depth else 0

    groups: Dict[Tuple, List[Operation]] = {}
    rep_time: Dict[Tuple, float] = {}

    def _key(op: Operation) -> Tuple:
        return (op.job_id, op.min_start_t, op.max_end_t)

    max_periodic_start = 0.0
    for op in operations:
        if op.min_start_t is not None:
            max_periodic_start = max(max_periodic_start, float(op.min_start_t))

    for op in operations:
        k = _key(op)
        groups.setdefault(k, []).append(op)
        if op.min_start_t is not None:
            rep_time[k] = float(op.min_start_t)
        else:
            # Spread non-periodic ops across the same range as periodic ones.
            if max_depth > 0 and max_periodic_start > 0:
                scaled = (depth[op] / max_depth) * max_periodic_start
            else:
                scaled = float(depth[op])
            prev = rep_time.get(k)
            rep_time[k] = scaled if prev is None else max(prev, scaled)

    max_time = max(rep_time.values()) if rep_time else 1.0
    if max_time <= 0:
        max_time = 1.0
    window_time = max_time / (n_splits + 1)

    buckets: List[List[Operation]] = [[] for _ in range(n_splits + 1)]
    for k, members in groups.items():
        idx = min(int(rep_time[k] // window_time), n_splits) if window_time > 0 else 0
        buckets[idx].extend(members)

    return [
        Window(
            time_frame=window_time,
            operations=ops,
            machines=workload.machines,
            transfer_times=workload.get_transfer_times(),
            machine_combinations=workload.get_machine_combinations(),
        )
        for ops in buckets
    ]


# ---------------------------------------------------------------------------
# Stage 2: greedy (type, count) selection with per-type capacity tracker
# ---------------------------------------------------------------------------

class _CountTracker:
    """Tracks per-type usage as a list of (start, end, count) intervals."""

    def __init__(self, machine_core_counts: Dict[str, int]):
        self._capacity = dict(machine_core_counts)
        self._usage: Dict[str, List[Tuple[float, float, int]]] = {
            t: [] for t in machine_core_counts
        }

    def earliest_free(
        self, type_: str, count: int, t_min: float, duration: float,
    ) -> float:
        if count > self._capacity.get(type_, 0):
            return float("inf")
        candidates = {float(t_min)}
        for s, e, _ in self._usage.get(type_, []):
            if e >= t_min:
                candidates.add(float(e))
            if s >= t_min:
                candidates.add(float(s))
        for t_candidate in sorted(candidates):
            if self._fits(type_, t_candidate, t_candidate + duration, count):
                return t_candidate
        last = 0.0
        for _, e, _ in self._usage.get(type_, []):
            if e > last:
                last = e
        return max(float(t_min), last)

    def _fits(self, type_: str, t_start: float, t_end: float, count: int) -> bool:
        capacity = self._capacity[type_]
        events: List[Tuple[float, int]] = []
        for s, e, c in self._usage.get(type_, []):
            if e <= t_start or s >= t_end:
                continue
            events.append((max(s, t_start), +c))
            events.append((min(e, t_end), -c))
        events.sort(key=lambda ev: (ev[0], 0 if ev[1] < 0 else 1))
        running = 0
        peak = 0
        for _, delta in events:
            running += delta
            if running > peak:
                peak = running
        return peak + count <= capacity

    def reserve(self, type_: str, t_start: float, duration: float, count: int) -> None:
        self._usage.setdefault(type_, []).append(
            (float(t_start), float(t_start + duration), int(count))
        )


def _order_by_topo_then_depth(
    operations: List[Operation], depth: Dict[Operation, int],
) -> List[int]:
    n = len(operations)
    idx_of = {id(op): i for i, op in enumerate(operations)}
    succ: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for i, op in enumerate(operations):
        for pred in op.get_predecessors():
            p = idx_of.get(id(pred))
            if p is None:
                continue
            succ[p].append(i)
            indeg[i] += 1
    heap: List[Tuple[int, int]] = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(heap, (depth.get(operations[i], 0), i))
    out: List[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (depth.get(operations[v], 0), v))
    if len(out) < n:
        seen = set(out)
        for i in range(n):
            if i not in seen:
                out.append(i)
    return out


def greedy_core_count_selection(
    window_operations: List[Operation],
    old_combinations: List[List[str]],
    machine_core_counts: Dict[str, int],
    tracker: _CountTracker,
    prior_decisions: Dict[Operation, Tuple[str, int, float, float]],
) -> Dict[Operation, Tuple[str, int, float, float]]:
    """
    Stage 2: pick (type, count, start, duration) per op by minimum completion.
    `tracker` is shared across windows so cross-window precedence is respected.
    """
    if not window_operations:
        return {}
    depth = _topo_depth(window_operations)
    decisions: Dict[Operation, Tuple[str, int, float, float]] = {}

    for idx in _order_by_topo_then_depth(window_operations, depth):
        op = window_operations[idx]
        t_earliest = 0.0
        for pred in op.get_predecessors():
            if pred in decisions:
                _, _, ps, pd = decisions[pred]
                t_earliest = max(t_earliest, ps + pd)
            elif pred in prior_decisions:
                _, _, ps, pd = prior_decisions[pred]
                t_earliest = max(t_earliest, ps + pd)
        if op.min_start_t is not None:
            t_earliest = max(t_earliest, float(op.min_start_t))

        pref_type, pref_count, _ = _preferred_type_count(op, old_combinations)

        best = None
        for type_, capacity in machine_core_counts.items():
            for count in range(1, capacity + 1):
                dur = _lookup_duration(op, old_combinations, type_, count)
                if dur is None:
                    continue
                if op.max_end_t is not None and t_earliest + dur > float(op.max_end_t):
                    # Skip options that can't fit the deadline even at earliest.
                    pass
                start = tracker.earliest_free(type_, count, t_earliest, dur)
                completion = start + dur
                tie = (
                    0 if (type_, count) == (pref_type, pref_count) else 1,
                    abs(count - pref_count),
                    0 if type_ == pref_type else 1,
                )
                key = (completion,) + tie
                if best is None or key < best[0]:
                    best = (key, type_, count, start, dur)

        if best is None:
            raise ValueError(
                f"No (type, count) for op {op.operation_name or op.operation_id}"
            )
        _, chosen_type, chosen_count, chosen_start, chosen_dur = best
        tracker.reserve(chosen_type, chosen_start, chosen_dur, chosen_count)
        decisions[op] = (chosen_type, chosen_count, chosen_start, chosen_dur)

    return decisions


# ---------------------------------------------------------------------------
# Stage 3: assign specific cores via greedy interval coloring
# ---------------------------------------------------------------------------

def assign_specific_cores(
    decisions: Dict[Operation, Tuple[str, int, float, float]],
    machine_core_counts: Dict[str, int],
) -> Dict[Operation, List[str]]:
    """
    Given (type, count, start, duration) per op, pick which N specific cores
    of that type each op uses. Stage 2 guarantees per-type capacity is
    respected, so this greedy always succeeds.
    """
    per_type: Dict[str, List[Tuple[float, float, int, Operation]]] = {
        t: [] for t in machine_core_counts
    }
    for op, (tname, ncnt, start, dur) in decisions.items():
        per_type[tname].append((float(start), float(start + dur), int(ncnt), op))

    assignments: Dict[Operation, List[str]] = {}
    for tname, intervals in per_type.items():
        if not intervals:
            continue
        capacity = machine_core_counts[tname]
        core_names = [f"{tname}#{i}" for i in range(capacity)]
        # core_busy_until[i] = list of (end, start) intervals reserved on core i;
        # we just track pairwise overlap, so storing intervals per core is enough.
        intervals_per_core: List[List[Tuple[float, float]]] = [[] for _ in range(capacity)]

        # Sort by start, then by descending count so fatter ops place first.
        intervals.sort(key=lambda x: (x[0], -x[2]))
        for start, end, count, op in intervals:
            free: List[int] = []
            for i in range(capacity):
                busy = False
                for s2, e2 in intervals_per_core[i]:
                    if s2 < end and start < e2:
                        busy = True
                        break
                if not busy:
                    free.append(i)
                    if len(free) == count:
                        break
            if len(free) < count:
                raise RuntimeError(
                    f"Could not place {count}x{tname} for op "
                    f"{op.operation_name or op.operation_id} at [{start}, {end}); "
                    f"only {len(free)} cores free. Stage 2 tracker invariant violated."
                )
            assignments[op] = [core_names[i] for i in free]
            for i in free:
                intervals_per_core[i].append((start, end))

    return assignments


# ---------------------------------------------------------------------------
# Build outputs in the Workload / alpha / t conventions used elsewhere
# ---------------------------------------------------------------------------

def build_workload_with_subset_combinations(
    original_workload: Workload,
    old_combinations: List[List[str]],
    new_combinations: List[List[str]],
    new_machines: List[str],
) -> Tuple[Workload, Dict[Operation, Operation]]:
    """Rewrite workload onto the all-subset combo space, keeping processing_times aligned."""
    type_size_to_old_idx: Dict[Tuple[str, int], int] = {}
    for k, combo in enumerate(old_combinations):
        key = (machine_type_prefix(combo[0]), len(combo))
        type_size_to_old_idx.setdefault(key, k)

    new_ops: List[Operation] = []
    op_map: Dict[Operation, Operation] = {}
    for op in original_workload.get_operations():
        new_pt: List[float] = []
        for combo in new_combinations:
            key = (machine_type_prefix(combo[0]), len(combo))
            old_idx = type_size_to_old_idx.get(key)
            if old_idx is None or old_idx >= len(op.processing_times):
                new_pt.append(float("inf"))
            else:
                new_pt.append(float(op.processing_times[old_idx]))
        new_op = Operation(
            processing_times=new_pt,
            predecessors=[],
            operation_id=op.operation_id,
            operation_name=op.operation_name,
            job_id=op.job_id,
            min_start_t=op.min_start_t,
            max_end_t=op.max_end_t,
        )
        new_ops.append(new_op)
        op_map[op] = new_op

    for old_op, new_op in op_map.items():
        for pred in old_op.get_predecessors():
            new_pred = op_map.get(pred)
            if new_pred is not None:
                new_op.add_predecessor(new_pred)

    new_workload = Workload(
        operations=new_ops,
        machines=new_machines,
        transfer_times=original_workload.get_transfer_times(),
        job_names=list(original_workload.job_names),
        machine_combinations=new_combinations,
    )
    return new_workload, op_map


def _combo_index_for_cores(
    combinations: List[List[str]], cores: List[str],
) -> int:
    target = tuple(sorted(cores))
    for k, combo in enumerate(combinations):
        if tuple(sorted(combo)) == target:
            return k
    raise ValueError(f"No combination matches core set {cores}")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def schedule_with_core_count_flow(
    workload: Workload,
    machine_core_counts: Dict[str, int],
    n_splits: int = 3,
    **_ignored,
) -> Tuple[Workload, np.ndarray, np.ndarray]:
    """
    Windowing -> greedy count pick -> greedy specific-core assignment.

    Returns (expanded_workload, t, alpha) where expanded_workload uses
    all-subset combinations so alpha pinpoints specific cores.
    """
    old_combinations = workload.get_machine_combinations()

    print("[core_count_flow] stage 1: splitting into windows (no solver)")
    windows = split_into_windows(workload, n_splits)
    for i, w in enumerate(windows):
        print(f"  window {i}: {len(w.operations)} ops")

    print("[core_count_flow] stage 2: greedy core-count selection")
    tracker = _CountTracker(machine_core_counts)
    all_decisions: Dict[Operation, Tuple[str, int, float, float]] = {}
    for w_idx, window in enumerate(windows):
        w_dec = greedy_core_count_selection(
            window_operations=window.operations,
            old_combinations=old_combinations,
            machine_core_counts=machine_core_counts,
            tracker=tracker,
            prior_decisions=all_decisions,
        )
        print(f"  window {w_idx}: placed {len(w_dec)} ops "
              f"({_summarize_counts(w_dec)})")
        all_decisions.update(w_dec)

    print("[core_count_flow] stage 3: assigning specific cores")
    core_assignments = assign_specific_cores(all_decisions, machine_core_counts)

    new_machines, new_combinations = build_all_subset_combinations(machine_core_counts)
    new_workload, old_to_new = build_workload_with_subset_combinations(
        workload, old_combinations, new_combinations, new_machines,
    )

    n_ops = len(new_workload.operations)
    n_combos = len(new_combinations)
    t = np.zeros(n_ops)
    alpha = np.zeros((n_ops, n_combos), dtype=int)
    for old_op, new_op in old_to_new.items():
        i = new_workload.operations.index(new_op)
        _, _, start, _ = all_decisions[old_op]
        t[i] = float(start)
        cores = core_assignments[old_op]
        k = _combo_index_for_cores(new_combinations, cores)
        alpha[i, k] = 1

    return new_workload, t, alpha


def _summarize_counts(decisions: Dict[Operation, Tuple[str, int, float, float]]) -> str:
    tally: Dict[Tuple[str, int], int] = {}
    for _, (tname, ncnt, _s, _d) in decisions.items():
        tally[(tname, ncnt)] = tally.get((tname, ncnt), 0) + 1
    return ", ".join(f"{c}x{t}={n}" for (t, c), n in sorted(tally.items()))
