"""
Fusion / split candidate generator and deterministic scorer.

generate_candidates(workload, schedule_t, schedule_alpha) returns a list of
Candidate dicts. Each candidate is a small, legal graph rewrite paired with
an *expected_benefit* (heuristic prediction, no compilation) and an
*expected_risk*.

score_candidates(candidates, workload, scheduler_fn) re-schedules the
workload with each candidate applied and records the MEASURED benefit
(makespan and dispatch-count deltas vs the unmodified baseline). The
returned list is sorted by measured benefit so the caller can pick a top-K.

apply_candidate(workload, candidate) returns a NEW Workload with the
rewrite applied. Implemented rewrites:

  fuse_producer_consumer  — fuse a single producer→consumer pair into one op
                            whose processing_times equal the sum of the two
                            ops' costs (per machine). Transfer is eliminated
                            because both stages run on the same machine.
  fuse_linear_chain       — fuse a maximal segment of single-predecessor /
                            single-successor ops into one op.
  split_heavy_dispatch    — split one heavy op into N equal sub-ops chained
                            sequentially (caller picks N).

The scorer's correlation with measured improvement is what closes the loop
in M10.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: str
    type: str
    affected_ops: List[str]
    expected_benefit: Dict[str, float] = field(default_factory=dict)
    expected_risk: Dict[str, Any] = field(default_factory=dict)
    # Internal: enough info to apply.
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "type": self.type,
            "affected_ops": list(self.affected_ops),
            "expected_benefit": dict(self.expected_benefit),
            "expected_risk": dict(self.expected_risk),
        }


# ---------------------------------------------------------------------------
# Helpers — graph manipulation on a Workload
# ---------------------------------------------------------------------------


def _copy_op(op: Operation) -> Operation:
    new = Operation(
        processing_times=list(op.processing_times),
        operation_name=op.operation_name,
        operation_id=op.operation_id,
        job_id=op.job_id,
        min_start_t=op.min_start_t,
        max_end_t=op.max_end_t,
        deadline_us=op.deadline_us,
        skip_allowed=op.skip_allowed,
        infeasible_combinations=set(op.infeasible_combinations),
    )
    new.output_bytes = getattr(op, "output_bytes", 0)
    new.memory_region_preference = getattr(op, "memory_region_preference", None)
    return new


def _copy_workload(workload: Workload) -> Tuple[Workload, Dict[int, Operation]]:
    """Return a fresh workload with deep-copied ops AND a mapping from
    OLD-op id → NEW-op so callers can find equivalents."""
    new_ops: List[Operation] = []
    old_to_new: Dict[int, Operation] = {}
    for op in workload.operations:
        n = _copy_op(op)
        new_ops.append(n)
        old_to_new[id(op)] = n
    # Wire predecessors using the mapping.
    for old, new in zip(workload.operations, new_ops):
        for pred in old.get_predecessors():
            new.add_predecessor(old_to_new[id(pred)])
    wl = Workload(
        new_ops, list(workload.machines),
        np.array(workload.get_transfer_times()),
        job_names=list(workload.job_names),
        machine_combinations=[list(c) for c in workload.machine_combinations],
    )
    return wl, old_to_new


def _consumers_map(workload: Workload) -> Dict[int, List[int]]:
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}
    out: Dict[int, List[int]] = {i: [] for i in range(len(workload.operations))}
    for i, op in enumerate(workload.operations):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is not None:
                out[pi].append(i)
    return out


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_candidates(
    workload: Workload,
    schedule_t: Optional[np.ndarray] = None,
    schedule_alpha: Optional[np.ndarray] = None,
    *,
    transfer_threshold_us: float = 5.0,
    max_chain_len: int = 6,
) -> List[Candidate]:
    """Walk the DAG, propose fuse and split candidates.

    Without a baseline schedule, fuse benefit is estimated as:
        saved_transfer = transfer_cost_estimate(producer, consumer)
        saved_dispatch = small constant per fused op (modelled per-target)
    Larger ops are NOT fused (split candidates are emitted instead).
    """
    candidates: List[Candidate] = []
    ops = workload.operations
    consumers = _consumers_map(workload)
    op_idx = {id(op): i for i, op in enumerate(ops)}
    transfer = workload.get_transfer_times()
    n_machines = len(workload.machines)
    mean_transfer = float(np.mean(transfer[np.triu_indices(n_machines, k=1)])) if n_machines > 1 else 0.0

    # 1. Fuse producer-consumer pairs.
    for i, op in enumerate(ops):
        if len(consumers[i]) != 1:
            continue
        j = consumers[i][0]
        cons = ops[j]
        # Don't fuse if consumer has other predecessors (would force them all to one machine)
        if len(cons.get_predecessors()) != 1:
            continue
        # Estimated benefit.
        saved_transfer = mean_transfer
        # If transfer is small relative to op cost, fusion's value drops.
        mean_op_cost = float(np.mean(op.processing_times + cons.processing_times))
        if saved_transfer < transfer_threshold_us and mean_op_cost > 100:
            continue
        cand = Candidate(
            candidate_id=f"fuse_{op.operation_name}__{cons.operation_name}",
            type="fuse_producer_consumer",
            affected_ops=[op.operation_name, cons.operation_name],
            expected_benefit={
                "saved_transfer_us": saved_transfer,
                "predicted_makespan_delta": -saved_transfer,
            },
            expected_risk={
                "lost_device_flexibility": True,
                "scratchpad_pressure_increase": 0,
            },
            payload={"producer_idx": i, "consumer_idx": j},
        )
        candidates.append(cand)

    # 2. Fuse maximal linear chains (>= 3 ops in a row).
    visited = set()
    for i, op in enumerate(ops):
        if i in visited:
            continue
        # Extend forward as long as the chain has single-in/single-out.
        chain = [i]
        cur = i
        while (len(consumers[cur]) == 1 and len(ops[consumers[cur][0]].get_predecessors()) == 1
               and len(chain) < max_chain_len):
            nxt = consumers[cur][0]
            chain.append(nxt)
            cur = nxt
        if len(chain) >= 3:
            for c in chain:
                visited.add(c)
            total_saved = mean_transfer * (len(chain) - 1)
            cand = Candidate(
                candidate_id=f"fuse_chain_{ops[chain[0]].operation_name}_{ops[chain[-1]].operation_name}",
                type="fuse_linear_chain",
                affected_ops=[ops[c].operation_name for c in chain],
                expected_benefit={
                    "saved_transfer_us": total_saved,
                    "predicted_makespan_delta": -total_saved,
                    "dispatch_count_delta": -(len(chain) - 1),
                },
                expected_risk={"lost_device_flexibility": True},
                payload={"chain_indices": list(chain)},
            )
            candidates.append(cand)

    # 3. Split heavy ops (where one op's cost > 3x the average op cost).
    avg = float(np.mean([float(np.mean(o.processing_times)) for o in ops])) if ops else 0.0
    for i, op in enumerate(ops):
        op_cost = float(np.mean(op.processing_times))
        if avg > 0 and op_cost > 3 * avg:
            cand = Candidate(
                candidate_id=f"split_{op.operation_name}",
                type="split_heavy_dispatch",
                affected_ops=[op.operation_name],
                expected_benefit={
                    "exposed_parallelism": op_cost,
                    "predicted_makespan_delta": -op_cost * 0.3,
                },
                expected_risk={"extra_dispatch_overhead": True,
                               "extra_transfer_cost": True},
                payload={"op_idx": i, "n_splits": 2},
            )
            candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_candidate(workload: Workload, cand: Candidate) -> Workload:
    """Return a NEW workload with ``cand`` applied. The input workload is
    not modified."""
    if cand.type == "fuse_producer_consumer":
        return _apply_fuse_pair(workload, cand.payload["producer_idx"],
                                cand.payload["consumer_idx"])
    if cand.type == "fuse_linear_chain":
        return _apply_fuse_chain(workload, cand.payload["chain_indices"])
    if cand.type == "split_heavy_dispatch":
        return _apply_split(workload, cand.payload["op_idx"],
                            cand.payload["n_splits"])
    raise ValueError(f"unknown candidate type: {cand.type}")


def _apply_fuse_pair(workload, prod_idx, cons_idx):
    wl, old_to_new = _copy_workload(workload)
    ops = wl.operations
    prod_old = workload.operations[prod_idx]
    cons_old = workload.operations[cons_idx]
    prod_new = old_to_new[id(prod_old)]
    cons_new = old_to_new[id(cons_old)]

    # Fused op: processing_times = sum per machine, infeasible if either is.
    fused = _copy_op(prod_new)
    fused.operation_name = f"{prod_new.operation_name}+{cons_new.operation_name}"
    fused.processing_times = [
        prod_new.processing_times[k] + cons_new.processing_times[k]
        for k in range(len(prod_new.processing_times))
    ]
    fused.infeasible_combinations = (
        prod_new.infeasible_combinations | cons_new.infeasible_combinations
    )
    # Predecessors of fused = predecessors of producer.
    fused.predecessors = list(prod_new.predecessors)
    # Inherit deadline / window from CONSUMER (the sink of the merged pair).
    fused.deadline_us = cons_new.deadline_us
    fused.max_end_t = cons_new.max_end_t
    fused.output_bytes = getattr(cons_new, "output_bytes", 0)
    # Swap.
    new_ops = []
    for op in ops:
        if op is prod_new:
            new_ops.append(fused)
        elif op is cons_new:
            continue  # drop
        else:
            # Rewire predecessors that point to producer/consumer -> fused.
            op.predecessors = [
                fused if (p is prod_new or p is cons_new) else p
                for p in op.predecessors
            ]
            new_ops.append(op)
    wl.operations = new_ops
    return wl


def _apply_fuse_chain(workload, chain_indices):
    """Apply pairwise fuses along the chain to collapse it into one op."""
    wl = workload
    # Apply iteratively: at each step, fuse (chain[0], chain[1]) and rebuild.
    # We use the operation_name to track which op to fuse next.
    chain_names = [workload.operations[i].operation_name for i in chain_indices]
    for k in range(len(chain_names) - 1):
        cur_a = chain_names[0] if k == 0 else f"{'+'.join(chain_names[:k+1])}"
        b = chain_names[k+1]
        # Find the indices in current wl.
        idx_a = next(i for i, op in enumerate(wl.operations) if op.operation_name == cur_a)
        idx_b = next(i for i, op in enumerate(wl.operations) if op.operation_name == b)
        wl = _apply_fuse_pair(wl, idx_a, idx_b)
    return wl


def _apply_split(workload, op_idx, n_splits):
    wl, old_to_new = _copy_workload(workload)
    old = workload.operations[op_idx]
    target = old_to_new[id(old)]
    # Replace target with n_splits chained sub-ops sharing 1/n of the cost.
    parts: List[Operation] = []
    for s in range(n_splits):
        part = _copy_op(target)
        part.operation_name = f"{target.operation_name}_p{s}"
        part.processing_times = [c / n_splits for c in target.processing_times]
        part.predecessors = list(target.predecessors) if s == 0 else [parts[-1]]
        # Only last part inherits deadline and consumers.
        if s == n_splits - 1:
            part.deadline_us = target.deadline_us
            part.max_end_t = target.max_end_t
            part.output_bytes = getattr(target, "output_bytes", 0)
        else:
            part.deadline_us = None
            part.max_end_t = None
            part.output_bytes = 0
        parts.append(part)
    # Rewire all consumers of target to last part.
    new_ops: List[Operation] = []
    for op in wl.operations:
        if op is target:
            new_ops.extend(parts)
            continue
        op.predecessors = [parts[-1] if p is target else p for p in op.predecessors]
        new_ops.append(op)
    wl.operations = new_ops
    return wl


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidates(
    candidates: List[Candidate],
    workload: Workload,
    scheduler_fn: Callable,
    *,
    metric: str = "makespan_us",
    fast_scorer: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Apply each candidate, run scheduler_fn, return rows with both predicted
    and measured ``metric`` delta vs the unmodified baseline.

    ``fast_scorer(workload, t, alpha) -> predicted_metric_value`` (optional):
    when supplied, replaces the per-candidate re-schedule with a single
    forward-pass prediction. Used by M11's learned cost model to avoid
    re-scheduling every candidate.
    """
    # Baseline.
    base_t, base_alpha, _, _ = scheduler_fn(workload)
    if base_t is None:
        raise RuntimeError("baseline scheduler returned no solution")
    base = _summarize(workload, base_t, base_alpha)

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        try:
            new_wl = apply_candidate(workload, cand)
            if fast_scorer is not None:
                # Use HEFT placement for the rewritten workload (cost model
                # needs SOME placement to score). HEFT is fast.
                from scheduler_heft import heft as _heft
                t, alpha, _, _ = _heft(new_wl)
                if t is None:
                    results.append({
                        "candidate": cand.to_dict(),
                        "applied": False, "error": "heft_failed_on_rewrite",
                    })
                    continue
                predicted_value = float(fast_scorer(new_wl, t, alpha))
                summ = {"makespan_us": predicted_value,
                        "dispatch_count": len(new_wl.operations)}
            else:
                t, alpha, _, _ = scheduler_fn(new_wl)
                if t is None:
                    results.append({
                        "candidate": cand.to_dict(),
                        "applied": False, "error": "no_schedule",
                    })
                    continue
                summ = _summarize(new_wl, t, alpha)
            measured_delta = summ[metric] - base[metric]
            predicted_delta = cand.expected_benefit.get("predicted_makespan_delta", 0.0)
            results.append({
                "candidate": cand.to_dict(),
                "applied": True,
                "baseline_metric": base[metric],
                "new_metric": summ[metric],
                "measured_delta": measured_delta,
                "predicted_delta": predicted_delta,
                "baseline_dispatch_count": base["dispatch_count"],
                "new_dispatch_count": summ["dispatch_count"],
                "summary": summ,
            })
        except Exception as exc:
            results.append({
                "candidate": cand.to_dict(),
                "applied": False, "error": str(exc),
            })
    # Sort: most improvement first (most negative measured_delta wins).
    results.sort(key=lambda r: r.get("measured_delta", 0.0))
    return results


def _summarize(workload, t, alpha):
    combos = workload.get_machine_combinations()
    finish = []
    for i, op in enumerate(workload.operations):
        k = int(np.argmax(alpha[i]))
        dur = float(op.get_duration_for_combination(k, combos, workload.machines))
        finish.append(float(t[i]) + dur)
    return {
        "makespan_us": max(finish) if finish else 0.0,
        "dispatch_count": len(workload.operations),
    }


# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------


def spearman_correlation(predicted: List[float], measured: List[float]) -> float:
    """Rank-correlation between predicted_delta and measured_delta."""
    if len(predicted) < 2:
        return 0.0
    p = np.array(predicted)
    m = np.array(measured)
    # Argsort gives ranks.
    pr = np.argsort(np.argsort(p))
    mr = np.argsort(np.argsort(m))
    pr_d = pr - pr.mean()
    mr_d = mr - mr.mean()
    denom = float(np.sqrt((pr_d ** 2).sum() * (mr_d ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((pr_d * mr_d).sum() / denom)
