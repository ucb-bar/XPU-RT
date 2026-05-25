"""
M22 — DAG analysis to identify fusion / split opportunities in a Workload.

Used to construct M22's ``realworld_fusion_opportunity`` and
``realworld_split_opportunity`` scenarios from analysing the real
dronet / mlp_wide / yolov8n graphs (rather than inventing the scenarios
by hand). Also exposed for the closed-loop optimizer to seed its candidate
generator (M9 ``rewrite.generate_candidates`` uses simpler heuristics; this
module is the more thorough analyser).

Two analyses:

  ``find_fusion_opportunities(workload, min_chain_len=3, transfer_threshold_us=10)``
    Returns linear chains of ops where total transfer cost across the chain
    boundary exceeds the chain's total compute on its locally-fastest
    machine. These are the "fuse-me" candidates — schedulers that keep the
    chain on one machine will see large transfer savings.

  ``find_split_opportunities(workload, top_k=3, dominance_ratio=2.0)``
    Returns the top-K ops that:
      (a) sit on the critical path, AND
      (b) have differential per-machine costs spanning >= dominance_ratio
          (so splitting could expose multi-device parallelism).
    These are the "split-me" candidates — schedulers can rewrite them into
    finer dispatches across multiple devices.

Both return structured records that downstream code can consume to (i)
construct synthetic-but-real-anchored scenarios, (ii) score real rewrite
candidates against an oracle.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class FusionOpportunity:
    chain_op_ids: List[int]            # indices into workload.operations
    chain_op_names: List[str]
    dominant_machine_idx: int          # the locally-fastest machine for the whole chain
    chain_compute_us: float            # sum of per-op cost on dominant machine
    saved_transfer_us: float           # mean inter-machine transfer cost * (chain_len - 1)
    score: float                       # saved_transfer - compute_overhead heuristic


@dataclass
class SplitOpportunity:
    op_id: int                         # index into workload.operations
    op_name: str
    on_critical_path: bool
    cost_min_us: float                 # cost on locally-fastest machine
    cost_max_us: float                 # cost on locally-slowest machine
    cost_mean_us: float
    dominance_ratio: float             # cost_max / cost_min
    estimated_parallelism_gain_us: float  # naive: cost_mean / 2 if split into 2 sub-ops
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topo_order(workload: Workload) -> List[int]:
    n = len(workload.operations)
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(workload.operations):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                indeg[i] += 1
                succ[pi].append(i)
    queue = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return order


def _successors(workload: Workload) -> Dict[int, List[int]]:
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}
    out: Dict[int, List[int]] = {i: [] for i in range(len(workload.operations))}
    for i, op in enumerate(workload.operations):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                out[pi].append(i)
    return out


def _critical_path_ops(workload: Workload) -> set:
    """Set of op indices on the critical path using fastest-feasible per-op cost."""
    order = _topo_order(workload)
    n = len(workload.operations)
    if not order:
        return set()
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}

    def _own_cost(op):
        feas = [c for c in op.processing_times if c < 1e8]
        return float(min(feas)) if feas else 0.0

    cp = [0.0] * n
    best_pred = [-1] * n
    for u in order:
        op = workload.operations[u]
        own = _own_cost(op)
        max_pred_v = 0.0
        max_pred_idx = -1
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is None:
                continue
            if cp[pi] >= max_pred_v:
                max_pred_v = cp[pi]
                max_pred_idx = pi
        cp[u] = max_pred_v + own
        best_pred[u] = max_pred_idx

    sink = max(range(n), key=lambda i: cp[i])
    path = set()
    cur = sink
    while cur != -1:
        path.add(cur)
        cur = best_pred[cur]
    return path


def _mean_off_diag_transfer_us(workload: Workload) -> float:
    """Average transfer cost between distinct machines."""
    tt = np.asarray(workload.get_transfer_times())
    n = len(workload.machines)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(np.mean(tt[iu]))


# ---------------------------------------------------------------------------
# Fusion-opportunity finder
# ---------------------------------------------------------------------------


def find_fusion_opportunities(
    workload: Workload,
    min_chain_len: int = 3,
    max_chain_len: int = 12,
    transfer_threshold_us: float = 10.0,
) -> List[FusionOpportunity]:
    """Walk the DAG; identify maximal single-pred / single-succ chains where
    fusing eliminates inter-machine transfer cost > chain compute."""
    ops = workload.operations
    n = len(ops)
    if n == 0:
        return []
    succ = _successors(workload)
    machines = list(workload.machines)
    mean_xfer = _mean_off_diag_transfer_us(workload)

    visited = set()
    out: List[FusionOpportunity] = []
    for start in range(n):
        if start in visited:
            continue
        chain = [start]
        cur = start
        while (len(succ[cur]) == 1
               and len(chain) < max_chain_len
               and len(ops[succ[cur][0]].get_predecessors()) == 1):
            cur = succ[cur][0]
            chain.append(cur)
        if len(chain) < min_chain_len:
            continue
        for idx in chain:
            visited.add(idx)

        # Pick the locally-fastest machine for the chain — the one whose
        # SUM of per-op costs across the chain is minimum AND is feasible
        # for every op in the chain.
        n_combos = len(workload.get_machine_combinations())
        best_m = 0
        best_sum = float("inf")
        for k in range(n_combos):
            if any(k in ops[i].infeasible_combinations for i in chain):
                continue
            s = sum(float(ops[i].processing_times[k]) for i in chain
                    if k < len(ops[i].processing_times) and ops[i].processing_times[k] < 1e8)
            if s < best_sum:
                best_sum = s
                best_m = k
        if not np.isfinite(best_sum):
            continue

        saved_xfer = mean_xfer * (len(chain) - 1)
        if saved_xfer < transfer_threshold_us:
            continue
        score = saved_xfer - 0.1 * best_sum  # heuristic; reward big transfer-save,
                                              # discount large compute (less to gain)

        out.append(FusionOpportunity(
            chain_op_ids=list(chain),
            chain_op_names=[ops[i].operation_name or f"op_{i}" for i in chain],
            dominant_machine_idx=best_m,
            chain_compute_us=float(best_sum),
            saved_transfer_us=float(saved_xfer),
            score=float(score),
        ))
    out.sort(key=lambda f: -f.score)
    return out


# ---------------------------------------------------------------------------
# Split-opportunity finder
# ---------------------------------------------------------------------------


def find_split_opportunities(
    workload: Workload,
    top_k: int = 3,
    dominance_ratio: float = 2.0,
) -> List[SplitOpportunity]:
    """Find heavy ops on the critical path with high per-machine cost spread."""
    ops = workload.operations
    if not ops:
        return []
    cp = _critical_path_ops(workload)
    out: List[SplitOpportunity] = []
    for i, op in enumerate(ops):
        feas_costs = [float(c) for k, c in enumerate(op.processing_times)
                      if c < 1e8 and k not in op.infeasible_combinations]
        if not feas_costs:
            continue
        c_min = min(feas_costs)
        c_max = max(feas_costs)
        if c_min <= 0:
            continue
        ratio = c_max / c_min
        if ratio < dominance_ratio:
            continue
        gain = float(np.mean(feas_costs)) / 2.0
        score = gain * (2.0 if i in cp else 0.7)
        out.append(SplitOpportunity(
            op_id=i,
            op_name=op.operation_name or f"op_{i}",
            on_critical_path=(i in cp),
            cost_min_us=float(c_min),
            cost_max_us=float(c_max),
            cost_mean_us=float(np.mean(feas_costs)),
            dominance_ratio=float(ratio),
            estimated_parallelism_gain_us=float(gain),
            score=float(score),
        ))
    out.sort(key=lambda s: -s.score)
    return out[:top_k]


# ---------------------------------------------------------------------------
# Summary helpers (used by scenarios.py)
# ---------------------------------------------------------------------------


def summarise(workload: Workload) -> Dict[str, object]:
    """One-shot summary for a real-anchored scenario builder."""
    fusion = find_fusion_opportunities(workload)
    split = find_split_opportunities(workload)
    return {
        "n_ops": len(workload.operations),
        "n_machines": len(workload.machines),
        "n_fusion_chains": len(fusion),
        "top_fusion": {
            "chain_op_names": fusion[0].chain_op_names if fusion else [],
            "chain_compute_us": fusion[0].chain_compute_us if fusion else 0.0,
            "saved_transfer_us": fusion[0].saved_transfer_us if fusion else 0.0,
            "score": fusion[0].score if fusion else 0.0,
        },
        "top_splits": [
            {
                "op_name": s.op_name,
                "dominance_ratio": s.dominance_ratio,
                "on_critical_path": s.on_critical_path,
                "estimated_parallelism_gain_us": s.estimated_parallelism_gain_us,
            }
            for s in split
        ],
    }
