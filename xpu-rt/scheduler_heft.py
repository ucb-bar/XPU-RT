"""
Fast list-scheduling baselines for XPU-RT.

Six schedulers share a single list-scheduling engine, differing only in their
priority and placement policies:

    name             priority             placement
    --------------   -----------------    ------------------
    heft             upward rank          earliest-finish-time
    critical_path    upward rank          first-available
    edf              earliest deadline    first-available
    fastest_device   topological order    locally fastest combo
    fifo             topological order    first-available
    random_list      shuffled order       first-available

All entries return ``(t, alpha, None, None)`` to match the existing
``scheduler.schedule`` contract. Predecessor constraints are honored, machine
combinations are respected (members of the same combination must not overlap),
release-time windows (``min_start_t``) act as earliest-start, and infeasible
combinations are excluded.

The schedulers do not perform fusion. Memory-aware variants live in milestone 7+.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


# ---------------------------------------------------------------------------
# Common list-scheduling engine
# ---------------------------------------------------------------------------


def _build_topo_order(ops: List[Operation]) -> List[int]:
    """Kahn topological sort over the operation predecessor DAG."""
    n = len(ops)
    op_idx = {id(op): i for i, op in enumerate(ops)}
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is None:
                continue
            indeg[i] += 1
            succ[pi].append(i)

    ready = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) < n:
        raise ValueError("Workload contains a cycle in the predecessor graph.")
    return order


def _mean_duration(op: Operation) -> float:
    pts = op.processing_times
    return float(np.mean(pts)) if len(pts) else 0.0


def _upward_rank(workload: Workload) -> List[float]:
    """HEFT upward-rank (avg duration + max successor rank)."""
    ops = workload.operations
    n = len(ops)
    op_idx = {id(op): i for i, op in enumerate(ops)}
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is not None:
                succ[pi].append(i)

    topo = _build_topo_order(ops)
    rank = [0.0] * n
    for u in reversed(topo):
        base = _mean_duration(ops[u])
        succ_max = 0.0
        for v in succ[u]:
            if rank[v] > succ_max:
                succ_max = rank[v]
        rank[u] = base + succ_max
    return rank


def _feasible_combinations(op: Operation, n_combos: int) -> List[int]:
    return [k for k in range(n_combos) if k not in op.infeasible_combinations]


def _transfer_us(workload: Workload, machines_a: List[str], machines_b: List[str]) -> float:
    """Worst-case transfer between two combinations (per existing convention)."""
    tt = workload.get_transfer_times()
    if tt is None or len(tt) == 0:
        return 0.0
    name_to_idx = {m: i for i, m in enumerate(workload.machines)}
    worst = 0.0
    for a in machines_a:
        ia = name_to_idx.get(a)
        if ia is None:
            continue
        for b in machines_b:
            ib = name_to_idx.get(b)
            if ib is None:
                continue
            if ia != ib:
                v = float(tt[ia][ib])
                if v > worst:
                    worst = v
    return worst


def _earliest_start_on_combo(
    workload: Workload,
    op: Operation,
    combo_idx: int,
    pred_finish: Dict[int, float],
    pred_combo: Dict[int, int],
    machine_busy_until: Dict[str, float],
) -> float:
    """Earliest legal start of ``op`` on ``combo_idx`` given the partial schedule."""
    combos = workload.get_machine_combinations()
    machines_here = combos[combo_idx]

    # Predecessor + transfer-cost ready time.
    op_idx_map = {id(o): i for i, o in enumerate(workload.operations)}
    ready = 0.0
    for pred in op.get_predecessors():
        pi = op_idx_map.get(id(pred))
        if pi is None:
            continue
        pf = pred_finish.get(pi, 0.0)
        pred_machines = combos[pred_combo[pi]]
        tx = _transfer_us(workload, pred_machines, machines_here)
        cand = pf + tx
        if cand > ready:
            ready = cand

    # Release-time window.
    if op.min_start_t is not None:
        if op.min_start_t > ready:
            ready = float(op.min_start_t)

    # Machine-busy time: every machine in the combination must be free.
    busy = max((machine_busy_until.get(m, 0.0) for m in machines_here), default=0.0)
    return max(ready, busy)


def _list_schedule(
    workload: Workload,
    *,
    priority: Callable[[Workload], List[float]],
    placement: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, None, None]:
    """Generic list scheduler shared by all six baselines.

    ``priority`` returns a list of floats; ops with HIGHER priority are scheduled
    earlier (after topo sort filters to ready ops).

    ``placement`` ∈ {"eft", "first_available", "fastest_local"}.
    """
    ops = workload.operations
    n = len(ops)
    combos = workload.get_machine_combinations()
    n_combos = len(combos)

    if n == 0:
        return np.zeros((0,)), np.zeros((0, n_combos)), None, None

    rng = np.random.default_rng(seed)
    prio = priority(workload)

    op_idx = {id(op): i for i, op in enumerate(ops)}
    succ: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is None:
                continue
            indeg[i] += 1
            succ[pi].append(i)

    ready = {i for i in range(n) if indeg[i] == 0}

    t = np.zeros(n)
    alpha = np.zeros((n, n_combos))
    pred_finish: Dict[int, float] = {}
    pred_combo: Dict[int, int] = {}
    machine_busy_until: Dict[str, float] = {m: 0.0 for m in workload.machines}

    while ready:
        ready_list = list(ready)
        # Tie-break by a stable secondary key so test runs are deterministic.
        if placement == "first_available" and seed is not None:
            rng.shuffle(ready_list)
            pick = ready_list[0]
        else:
            ready_list.sort(key=lambda i: (-prio[i], i))
            pick = ready_list[0]

        op = ops[pick]
        feasible = _feasible_combinations(op, n_combos)
        if not feasible:
            raise ValueError(f"Operation {pick} ({op.operation_name}) has no feasible combinations.")

        best_combo = feasible[0]
        best_eft = float("inf")
        best_start = 0.0

        for k in feasible:
            est = _earliest_start_on_combo(
                workload, op, k, pred_finish, pred_combo, machine_busy_until
            )
            dur = float(op.get_duration_for_combination(k, combos, workload.machines))
            eft = est + dur

            if placement == "eft":
                key = eft
            elif placement == "fastest_local":
                # Pick the combination with the lowest *duration*, breaking ties by EFT.
                key = (dur, eft)
            else:  # first_available
                key = (est, eft)

            if not isinstance(key, tuple):
                key = (key,)
            if not isinstance(best_eft, tuple) or best_eft == float("inf"):
                best_key = (float("inf"),)
            else:
                best_key = best_eft

            if key < best_key:
                best_eft = key
                best_combo = k
                best_start = est

        t[pick] = best_start
        alpha[pick, best_combo] = 1.0
        finish = best_start + float(op.get_duration_for_combination(best_combo, combos, workload.machines))
        pred_finish[pick] = finish
        pred_combo[pick] = best_combo
        for m in combos[best_combo]:
            machine_busy_until[m] = finish

        ready.remove(pick)
        for v in succ[pick]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.add(v)

    return t, alpha, None, None


# ---------------------------------------------------------------------------
# Public scheduler entries
# ---------------------------------------------------------------------------


def _topo_priority(workload: Workload) -> List[float]:
    """FIFO / fastest_device priority: topological position (smaller = earlier)."""
    order = _build_topo_order(workload.operations)
    prio = [0.0] * len(workload.operations)
    n = len(order)
    for rank, idx in enumerate(order):
        prio[idx] = float(n - rank)  # higher = earlier
    return prio


def _deadline_priority(workload: Workload) -> List[float]:
    """EDF priority: earlier deadline → higher priority (negative deadline as score).

    Operations without a deadline fall back to upward rank so the algorithm is
    well-defined on workloads with partial deadline annotations.
    """
    fallback = _upward_rank(workload)
    out: List[float] = []
    for i, op in enumerate(workload.operations):
        if op.deadline_us is not None:
            out.append(1e12 - float(op.deadline_us))
        else:
            out.append(fallback[i])
    return out


def _random_priority_factory(seed: Optional[int]) -> Callable[[Workload], List[float]]:
    def _prio(workload: Workload) -> List[float]:
        rng = np.random.default_rng(seed)
        return list(rng.random(len(workload.operations)))
    return _prio


def heft(workload: Workload, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(workload, priority=_upward_rank, placement="eft")


def critical_path(workload: Workload, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(workload, priority=_upward_rank, placement="first_available")


def edf(workload: Workload, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(workload, priority=_deadline_priority, placement="first_available")


def fastest_device(workload: Workload, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(workload, priority=_topo_priority, placement="fastest_local")


def fifo(workload: Workload, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(workload, priority=_topo_priority, placement="first_available")


def random_list(workload: Workload, *, random_seed: Optional[int] = 0, **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    return _list_schedule(
        workload,
        priority=_random_priority_factory(random_seed),
        placement="first_available",
        seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Extra list-scheduling variants and a simple simulated-annealing wrapper
# ---------------------------------------------------------------------------


def _round_robin_assign(workload: Workload) -> Tuple[np.ndarray, np.ndarray, None, None]:
    """Round-robin: assign ops to machine combinations cyclically in
    topological order, ignoring feasibility/cost. Trivial baseline."""
    ops = workload.operations
    n = len(ops)
    combos = workload.get_machine_combinations()
    order = _build_topo_order(ops)

    t = np.zeros(n)
    alpha = np.zeros((n, len(combos)))
    machine_busy: Dict[str, float] = {m: 0.0 for m in workload.machines}
    pred_finish: Dict[int, float] = {}
    pred_combo: Dict[int, int] = {}

    for rank_i, i in enumerate(order):
        op = ops[i]
        feasible = _feasible_combinations(op, len(combos))
        # Cyclic pick among feasible, biased by rank.
        k = feasible[rank_i % len(feasible)]
        est = _earliest_start_on_combo(workload, op, k, pred_finish, pred_combo, machine_busy)
        dur = float(op.get_duration_for_combination(k, combos, workload.machines))
        t[i] = est
        alpha[i, k] = 1.0
        finish = est + dur
        pred_finish[i] = finish
        pred_combo[i] = k
        for m in combos[k]:
            machine_busy[m] = finish

    return t, alpha, None, None


def round_robin(workload, **_):
    return _round_robin_assign(workload)


def _peft_priority(workload: Workload) -> List[float]:
    """PEFT optimistic cost table priority. For each (op, machine_combination):
       OCT[i, k] = max over successor j of (min over k': OCT[j, k'] + dur[j, k'] + transfer(k, k'))
    Then priority(op) = mean over k of (OCT[i, k] + dur[i, k]).

    Larger -> earlier in schedule.
    """
    ops = workload.operations
    n = len(ops)
    combos = workload.get_machine_combinations()
    n_combos = len(combos)
    machines = list(workload.machines)
    name_to_idx = {m: i for i, m in enumerate(machines)}
    transfer = workload.get_transfer_times()

    op_idx = {id(op): i for i, op in enumerate(ops)}
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is not None:
                succ[pi].append(i)

    topo = _build_topo_order(ops)

    # Per-op per-combo duration matrix.
    dur = [[float(ops[i].get_duration_for_combination(k, combos, machines))
            if k not in ops[i].infeasible_combinations else float("inf")
            for k in range(n_combos)] for i in range(n)]

    def _xfer(k_a: int, k_b: int) -> float:
        if k_a == k_b:
            return 0.0
        worst = 0.0
        for ma in combos[k_a]:
            for mb in combos[k_b]:
                ia, ib = name_to_idx.get(ma), name_to_idx.get(mb)
                if ia is None or ib is None or ia == ib:
                    continue
                v = float(transfer[ia][ib])
                if v > worst:
                    worst = v
        return worst

    # OCT[i, k] computed bottom-up.
    oct_table = [[0.0] * n_combos for _ in range(n)]
    for i in reversed(topo):
        for k in range(n_combos):
            if not succ[i]:
                oct_table[i][k] = 0.0
                continue
            best = 0.0
            for j in succ[i]:
                # min over k' of (OCT[j, k'] + dur[j, k'] + transfer)
                min_term = float("inf")
                for k_prime in range(n_combos):
                    if dur[j][k_prime] == float("inf"):
                        continue
                    cand = oct_table[j][k_prime] + dur[j][k_prime] + _xfer(k, k_prime)
                    if cand < min_term:
                        min_term = cand
                if min_term == float("inf"):
                    continue
                if min_term > best:
                    best = min_term
            oct_table[i][k] = best

    # Priority = mean(OCT[i, k] + dur[i, k]) across feasible k.
    out: List[float] = []
    for i in range(n):
        vals = [oct_table[i][k] + dur[i][k]
                for k in range(n_combos) if dur[i][k] != float("inf")]
        out.append(float(np.mean(vals)) if vals else 0.0)
    return out


def peft(workload, **_):
    return _list_schedule(workload, priority=_peft_priority, placement="eft")


def _min_min_priority(workload: Workload) -> List[float]:
    """min-min priority: prefer ops with the shortest min-over-machines duration."""
    return [-min(op.processing_times) for op in workload.operations]


def min_min(workload, **_):
    return _list_schedule(workload, priority=_min_min_priority, placement="fastest_local")


def _max_min_priority(workload: Workload) -> List[float]:
    """max-min priority: prefer ops with the LONGEST min-over-machines duration."""
    return [min(op.processing_times) for op in workload.operations]


def max_min(workload, **_):
    return _list_schedule(workload, priority=_max_min_priority, placement="fastest_local")


def simulated_annealing(workload, *, n_iters: int = 200,
                        T0: float = 100.0, alpha: float = 0.97,
                        random_seed: Optional[int] = 0, **_):
    """Simulated annealing starting from HEFT. Each iteration perturbs the
    placement of a randomly-chosen op to a different feasible combo; accepts
    if measured makespan improves or by Metropolis criterion at temperature T.

    Returns the best-found schedule under the search.
    """
    rng = np.random.default_rng(random_seed)
    # Start from HEFT.
    t_cur, a_cur, _, _ = heft(workload)
    combos = workload.get_machine_combinations()
    machines = list(workload.machines)
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}

    def _makespan(t: np.ndarray, alpha: np.ndarray) -> float:
        worst = 0.0
        for i, op in enumerate(workload.operations):
            k = int(np.argmax(alpha[i]))
            d = float(op.get_duration_for_combination(k, combos, machines))
            f = float(t[i]) + d
            if f > worst:
                worst = f
        return worst

    def _resimulate_with_assignment(alpha: np.ndarray) -> Optional[np.ndarray]:
        """Given a fixed (op -> combo) assignment, recompute valid start times
        by topo + per-combo earliest-start. Returns None if infeasible."""
        ops = workload.operations
        n = len(ops)
        try:
            order = _build_topo_order(ops)
        except ValueError:
            return None
        t_new = np.zeros(n)
        machine_busy: Dict[str, float] = {m: 0.0 for m in machines}
        pred_finish: Dict[int, float] = {}
        pred_combo: Dict[int, int] = {}
        for i in order:
            op = ops[i]
            k = int(np.argmax(alpha[i]))
            if k in op.infeasible_combinations:
                return None
            est = _earliest_start_on_combo(workload, op, k, pred_finish, pred_combo, machine_busy)
            t_new[i] = est
            dur = float(op.get_duration_for_combination(k, combos, machines))
            pred_finish[i] = est + dur
            pred_combo[i] = k
            for m in combos[k]:
                machine_busy[m] = est + dur
        return t_new

    best_t = t_cur.copy()
    best_a = a_cur.copy()
    best_ms = _makespan(best_t, best_a)
    cur_ms = best_ms
    T = T0

    for _ in range(n_iters):
        # Pick a random op and a different feasible combo.
        i = int(rng.integers(0, len(workload.operations)))
        op = workload.operations[i]
        feasible = _feasible_combinations(op, len(combos))
        if len(feasible) <= 1:
            continue
        cur_k = int(np.argmax(a_cur[i]))
        choices = [k for k in feasible if k != cur_k]
        new_k = int(rng.choice(choices))
        a_new = a_cur.copy()
        a_new[i] = 0
        a_new[i, new_k] = 1.0
        t_new = _resimulate_with_assignment(a_new)
        if t_new is None:
            continue
        new_ms = _makespan(t_new, a_new)
        if new_ms < cur_ms or rng.random() < float(np.exp((cur_ms - new_ms) / max(T, 1e-3))):
            t_cur, a_cur, cur_ms = t_new, a_new, new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_t = t_new.copy()
                best_a = a_new.copy()
        T *= alpha

    return best_t, best_a, None, None
