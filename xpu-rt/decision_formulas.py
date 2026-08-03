"""Analytical, deterministic decision formulas for fuse/split/shard.

The granularity advisor (`xpu-rt/rewrite.py:generate_candidates` +
`score_candidates`) currently ranks candidates by re-running the
scheduler. That's a predict-by-resolve heuristic, not an analytical
tool. This module provides closed-form, O(1) (per-op) formulas the
agent can call BEFORE building/measuring anything, so it has a reason
in addition to a score.

Formulas:

- `B1 frequency_feasibility(network_load, period, machines)` — gate
  before scheduling any sweep cell.
- `B2 shard_benefit(op_costs, machine_busy_at)` — closed-form best
  asymmetric tile fraction; harmonic-mean.
- `B3 fuse_benefit(op1, op2, placement)` — dispatch + reuse savings
  minus parallelism opportunity cost.
- `B4 unfuse_benefit(fused_op, placement)` — mirror of B3.
- `B5 compaction_eligible(op, machines, deps)` — per-op precondition
  for left-shift compaction without band violation.
- `B6 critical_path(ops, durations, edges)` — longest-path on the DAG
  using placed durations; memoized.

All inputs are plain numbers / dicts / lists. No xpu-rt Workload
object dependency, so this module is unit-testable without standing
up a scheduler. Adapters in policies/ translate Workload → these
inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================================================================
# B1. frequency_feasibility
# =========================================================================

@dataclass
class FeasibilityReport:
    feasible: bool
    per_class_load: Dict[str, float]
    min_class_load: float
    bottleneck_class: str
    slack: float          # period - min_class_load. Negative => infeasible.
    multiclass_partition_load: Optional[float] = None
    multiclass_partition: Optional[Dict[str, str]] = None


def frequency_feasibility(network_op_costs: Dict[str, Dict[str, float]],
                          period: float,
                          machine_classes: Sequence[str],
                          ) -> FeasibilityReport:
    """B1. Can one instance of `network` fit inside `period`?

    Args:
        network_op_costs: {op_name: {machine_class: cost_us}}.
            Each entry gives the cost of running `op_name` on
            `machine_class` (e.g. {"conv1": {"gemmini": 2.1, "rvv": 8.6}}).
            Missing keys mean the op can't run on that class.
        period: the period of the network (any time unit; must match
            cost units).
        machine_classes: classes considered (subset of those that appear
            in the cost map).

    Returns:
        FeasibilityReport:
        - per_class_load[c] = sum of op-min cost when ALL ops are
          assigned to class c (single-class scheduling on c).
        - min_class_load = the smallest of those — the best a
          single-class schedule could do.
        - feasible = min_class_load ≤ period (i.e. there's at least one
          class that can run the whole network within period).
        - multiclass_partition_load = sum of cheapest-class cost for
          each op (perfect-split, no contention). This is a strict
          lower bound on parallel multi-class scheduling.
    """
    per_class_load: Dict[str, float] = {c: 0.0 for c in machine_classes}
    multiclass_load = 0.0
    multiclass_partition: Dict[str, str] = {}

    for op_name, costs in network_op_costs.items():
        # Single-class accumulation: every op contributes to every class
        # that supports it. Class that can't run an op gets +inf (infeasible
        # single-class schedule).
        for c in machine_classes:
            if c in costs:
                per_class_load[c] += float(costs[c])
            else:
                per_class_load[c] = float("inf")

        # Multi-class: this op's cheapest-class cost.
        feasible_costs = {c: costs[c] for c in machine_classes if c in costs}
        if not feasible_costs:
            multiclass_load = float("inf")
            multiclass_partition[op_name] = "INFEASIBLE"
            continue
        best_c = min(feasible_costs, key=lambda c: feasible_costs[c])
        multiclass_load += feasible_costs[best_c]
        multiclass_partition[op_name] = best_c

    min_class_load = min(per_class_load.values()) if per_class_load else float("inf")
    bottleneck_class = (
        min(per_class_load, key=lambda c: per_class_load[c])
        if per_class_load else ""
    )
    feasible = min_class_load <= period + 1e-9
    slack = period - min_class_load

    return FeasibilityReport(
        feasible=feasible,
        per_class_load=per_class_load,
        min_class_load=min_class_load,
        bottleneck_class=bottleneck_class,
        slack=slack,
        multiclass_partition_load=multiclass_load,
        multiclass_partition=multiclass_partition,
    )


# =========================================================================
# B2. shard_benefit
# =========================================================================

@dataclass
class ShardReport:
    best_fraction: float                # f in (0,1): fraction on alt
    expected_delta: float               # negative = improvement (faster)
    home_cost: float
    alt_cost: float
    alt_machine: str
    optimal_finish_no_contention: float # harmonic-mean balanced finish
    contention_delay_on_alt: float      # slack-window block on alt before op ready
    on_critical_path: bool
    realizable: bool
    rejection_reason: Optional[str] = None


def shard_benefit(home_cost: float, alt_cost: float, alt_machine: str,
                  *, alt_soonest_free: float = 0.0,
                  op_ready: float = 0.0,
                  on_critical_path: bool = False,
                  realizable: bool = True,
                  ) -> ShardReport:
    """B2. Closed-form asymmetric tile placement.

    Splitting an op into fraction f on alt_machine and (1-f) on home:

        parallel_finish(f) = max((1-f) * home_cost, f * alt_cost
                                                  + contention_delay)
        optimal f          = home_cost / (home_cost + alt_cost)   # if no contention
        optimal finish     = home_cost * alt_cost / (home_cost + alt_cost)

    With contention on alt (alt becomes free at `alt_soonest_free` while
    the op becomes ready at `op_ready`), the alt-tile actually starts at
    `max(op_ready, alt_soonest_free)` and finishes
    `(start + f*alt_cost)`. We pick f to balance the two finishes:

        (1-f)*home_cost  =  contention_delay + f*alt_cost
        =>  f  =  (home_cost - contention_delay) / (home_cost + alt_cost)

    Clamped to [0, 1]. Negative f means alt is already too contended —
    splitting hurts; we report f=0 (no shard) and a non-negative
    expected_delta.

    expected_delta = parallel_finish_at_f - home_cost   (negative = win)
    """
    contention_delay = max(0.0, alt_soonest_free - op_ready)

    if home_cost <= 0 or alt_cost <= 0:
        return ShardReport(
            best_fraction=0.0,
            expected_delta=0.0,
            home_cost=home_cost,
            alt_cost=alt_cost,
            alt_machine=alt_machine,
            optimal_finish_no_contention=0.0,
            contention_delay_on_alt=contention_delay,
            on_critical_path=on_critical_path,
            realizable=realizable,
            rejection_reason="zero or negative cost",
        )

    # Optimum: balance two parallel branches.
    # (1-f)*home  =  contention + f*alt
    # 1 - f = (contention + f*alt) / home
    # home - f*home = contention + f*alt
    # f * (home + alt) = home - contention
    # f = (home - contention) / (home + alt)
    f_star = (home_cost - contention_delay) / (home_cost + alt_cost)

    # Clamp + harmonic-mean reference for the contention-free case.
    optimal_finish = (home_cost * alt_cost) / (home_cost + alt_cost)

    if f_star <= 0:
        # Alt is too contended; keep it all on home.
        return ShardReport(
            best_fraction=0.0,
            expected_delta=0.0,
            home_cost=home_cost,
            alt_cost=alt_cost,
            alt_machine=alt_machine,
            optimal_finish_no_contention=optimal_finish,
            contention_delay_on_alt=contention_delay,
            on_critical_path=on_critical_path,
            realizable=realizable,
            rejection_reason="alt contended past breakeven",
        )

    f = min(f_star, 1.0)
    # parallel_finish at this f:
    home_branch = (1.0 - f) * home_cost
    alt_branch = contention_delay + f * alt_cost
    parallel_finish = max(home_branch, alt_branch)
    expected_delta = parallel_finish - home_cost  # negative = good

    return ShardReport(
        best_fraction=f,
        expected_delta=expected_delta,
        home_cost=home_cost,
        alt_cost=alt_cost,
        alt_machine=alt_machine,
        optimal_finish_no_contention=optimal_finish,
        contention_delay_on_alt=contention_delay,
        on_critical_path=on_critical_path,
        realizable=realizable,
    )


# =========================================================================
# B3 / B4. fuse_benefit / unfuse_benefit
# =========================================================================

@dataclass
class FuseReport:
    expected_delta: float       # negative = improvement
    dispatch_save: float
    data_reuse_save: float
    parallelism_cost: float     # >= 0; only nonzero when ops were running in parallel
    realizable: bool
    reason: str = ""


def fuse_benefit(*,
                 op1_cost: float, op2_cost: float,
                 op1_machine: str, op2_machine: str,
                 fused_cost: Optional[float] = None,
                 dispatch_overhead: float = 5.0,
                 intermediate_bytes: float = 0.0,
                 mem_bw_per_us: float = 0.0,
                 realizable: bool = True,
                 ) -> FuseReport:
    """B3. Should we fuse op1 and op2?

    Three contributions:
      dispatch_save = dispatch_overhead   (one launch instead of two)
      data_reuse_save = intermediate_bytes / mem_bw_per_us  (the
        intermediate buffer that previously went through memory now
        stays in registers/cache; saved bw * size = time).
      parallelism_cost = if op1 was on machine A and op2 on B, fusing
        forces both onto one machine. The lost parallelism is the
        smaller op's duration (the larger was on the critical path
        anyway).

    expected_delta = fused_cost - (op1_cost + op2_cost)
                     - dispatch_save - data_reuse_save
                     + parallelism_cost

    If fused_cost is unknown, the caller can pass None and we treat it
    as `op1_cost + op2_cost` (serial-fused, no in-kernel speedup —
    the conservative assumption).
    """
    if fused_cost is None:
        fused_cost_eff = op1_cost + op2_cost
    else:
        fused_cost_eff = fused_cost

    dispatch_save = max(0.0, dispatch_overhead)
    data_reuse_save = (
        intermediate_bytes / mem_bw_per_us
        if mem_bw_per_us > 0 else 0.0
    )

    if op1_machine != op2_machine:
        # ops were in parallel; fusing serializes them.
        parallelism_cost = min(op1_cost, op2_cost)
    else:
        parallelism_cost = 0.0

    expected_delta = (
        fused_cost_eff - (op1_cost + op2_cost)
        - dispatch_save - data_reuse_save
        + parallelism_cost
    )

    return FuseReport(
        expected_delta=expected_delta,
        dispatch_save=dispatch_save,
        data_reuse_save=data_reuse_save,
        parallelism_cost=parallelism_cost,
        realizable=realizable,
        reason="" if realizable else "no registered fused kernel",
    )


def unfuse_benefit(*,
                   fused_cost: float,
                   op1_cost: float, op2_cost: float,
                   on_same_machine_now: bool,
                   alt_machine_for_op2_cost: Optional[float] = None,
                   dispatch_overhead: float = 5.0,
                   intermediate_bytes: float = 0.0,
                   mem_bw_per_us: float = 0.0,
                   realizable: bool = True,
                   ) -> FuseReport:
    """B4. Should we unfuse a previously-fused op?

    Inverse of B3. The wins come from:
      - parallelism_gain = if a fast alt machine is available for op2,
        we can run op1 on home and op2 on alt in parallel, finish in
        max(op1_cost, alt_machine_for_op2_cost).
        The save is fused_cost - max(...).
      - dispatch_cost = one extra launch (small loss).
      - data_reuse_loss = intermediate now goes through memory.

    expected_delta = max(op1_cost, alt_cost) - fused_cost
                     + dispatch_overhead + data_reuse_loss   (>= 0 = no win)
    """
    if alt_machine_for_op2_cost is None or alt_machine_for_op2_cost <= 0:
        # No useful alternative -> unfusing only adds overhead.
        return FuseReport(
            expected_delta=dispatch_overhead,
            dispatch_save=-dispatch_overhead,
            data_reuse_save=-(
                intermediate_bytes / mem_bw_per_us if mem_bw_per_us > 0 else 0.0
            ),
            parallelism_cost=0.0,
            realizable=realizable,
            reason="no alt machine for op2",
        )

    parallel_finish = max(op1_cost, alt_machine_for_op2_cost)
    parallelism_gain = fused_cost - parallel_finish  # positive = parallel wins

    data_reuse_loss = (
        intermediate_bytes / mem_bw_per_us
        if mem_bw_per_us > 0 else 0.0
    )
    expected_delta = -parallelism_gain + dispatch_overhead + data_reuse_loss

    return FuseReport(
        expected_delta=expected_delta,
        dispatch_save=-dispatch_overhead,
        data_reuse_save=-data_reuse_loss,
        parallelism_cost=parallelism_gain,  # >0 means there IS a win
        realizable=realizable,
    )


# =========================================================================
# B5. compaction_eligible
# =========================================================================

@dataclass
class CompactReport:
    applicable: bool
    gap: float
    blocked_by: str  # "" if applicable; else "release", "dep:<name>",
                     # "machine_busy", "downstream_deadline"


def compaction_eligible(*,
                        op_start: float,
                        op_duration: float,
                        op_release: float = 0.0,
                        op_max_end: Optional[float] = None,
                        dep_finishes: Sequence[float] = (),
                        dep_names: Sequence[str] = (),
                        machine_last_busy_before: float = 0.0,
                        downstream_max_ends: Sequence[float] = (),
                        downstream_offsets: Sequence[float] = (),
                        ) -> CompactReport:
    """B5. Per-op compaction precondition.

    Compaction tries to shift `op` left by `gap`. Returns whether the
    full gap can be reclaimed without violating:
      - op's own release time,
      - op's downstream deadline (if shifting were to ripple),
      - any predecessor's finish (we cannot start before its finish),
      - the machine's last-busy time before op_start (we cannot start
        before the machine is free).

    gap = op_start - max(op_release,
                         max(dep_finishes) if any,
                         machine_last_busy_before)

    applicable = gap > 0 AND moving op left by gap keeps every
                 downstream consumer within [downstream_max_end -
                 downstream_offset, ...]. In practice the existing
                 compaction loop iterates; B5 just predicts whether the
                 single move is band-safe for op's own band.
    """
    earliest = op_release
    blocked_by = ""

    if dep_finishes:
        max_dep_finish = max(dep_finishes)
        if max_dep_finish > earliest:
            earliest = max_dep_finish
            # locate the binding dep for the reason field.
            for f, n in zip(dep_finishes, dep_names or [""]*len(dep_finishes)):
                if abs(f - max_dep_finish) < 1e-9:
                    blocked_by = f"dep:{n}" if n else "dep"
                    break

    if machine_last_busy_before > earliest:
        earliest = machine_last_busy_before
        blocked_by = "machine_busy"

    if op_release >= earliest - 1e-9 and not blocked_by:
        blocked_by = "release"

    gap = op_start - earliest

    if gap <= 1e-9:
        return CompactReport(applicable=False, gap=max(gap, 0.0),
                             blocked_by=blocked_by or "no gap")

    # Check downstream max_end_t — if shifting op left by `gap` lets
    # downstream consumer finish earlier (good), or doesn't move it at
    # all (also fine). Only flag a violation when a downstream consumer
    # ALREADY overruns its deadline and shifting op left by gap would
    # not relax it. (A pure left-shift of `op` cannot push downstream
    # ops *later* — they only depend on `op` finishing earlier, which
    # is monotone-relaxing. We include the check for completeness
    # against future band-coupling extensions.)
    for d_max_end, d_offset in zip(downstream_max_ends, downstream_offsets):
        # downstream_offset is the consumer's own start-time relative
        # to op's finish. After left-shift by gap, the consumer can
        # start at max(consumer.release, op.finish - gap + d_offset).
        # That's earlier or equal, so it's never worse.
        # No regression possible from op-only left-shift; placeholder
        # left for symmetry.
        _ = (d_max_end, d_offset)

    if op_max_end is not None:
        new_finish = op_start - gap + op_duration
        if new_finish > op_max_end + 1e-9:
            return CompactReport(applicable=False, gap=gap,
                                 blocked_by="own_deadline")

    return CompactReport(applicable=True, gap=gap, blocked_by="")


# =========================================================================
# B6. critical_path
# =========================================================================

@dataclass
class CriticalPathReport:
    path: List[str]                  # op names in order along critical path
    length: float                    # sum of durations on path
    op_to_rank: Dict[str, float]     # earliest-finish rank per op
    on_path: Dict[str, bool]


def critical_path(*,
                  ops: Sequence[str],
                  durations: Dict[str, float],
                  edges: Sequence[Tuple[str, str]],
                  ) -> CriticalPathReport:
    """B6. Longest-path on the DAG using placed durations.

    Topological order: Kahn's algorithm with stable tie-breaking
    (input order). For each node, earliest-finish = max over preds
    of pred.earliest_finish + node.duration. The sink with the largest
    earliest-finish is on the critical path; walk back to source via
    predecessor with matching earliest-finish.

    Tie-breaking on the back-walk: pick the predecessor with the
    largest earliest-finish; if ties, the one whose op_name sorts
    first (deterministic).
    """
    preds: Dict[str, List[str]] = {o: [] for o in ops}
    succs: Dict[str, List[str]] = {o: [] for o in ops}
    indeg: Dict[str, int] = {o: 0 for o in ops}
    for u, v in edges:
        if u not in preds or v not in preds:
            continue
        preds[v].append(u)
        succs[u].append(v)
        indeg[v] += 1

    order: List[str] = []
    queue = [o for o in ops if indeg[o] == 0]
    # FIFO queue; stable wrt input order.
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) != len(ops):
        # Cycle — return empty critical path (caller should error).
        return CriticalPathReport(path=[], length=0.0,
                                  op_to_rank={o: 0.0 for o in ops},
                                  on_path={o: False for o in ops})

    earliest_finish: Dict[str, float] = {}
    for u in order:
        max_pred = 0.0
        for p in preds[u]:
            if earliest_finish[p] > max_pred:
                max_pred = earliest_finish[p]
        earliest_finish[u] = max_pred + float(durations.get(u, 0.0))

    if not earliest_finish:
        return CriticalPathReport(path=[], length=0.0, op_to_rank={}, on_path={})

    sink = max(earliest_finish, key=lambda o: earliest_finish[o])
    length = earliest_finish[sink]

    # Walk back along longest-incoming edges.
    path_rev = [sink]
    cur = sink
    while preds[cur]:
        # Pick pred whose earliest_finish + own duration matches
        # earliest_finish[cur] (within fp tolerance).
        best = None
        best_ef = -math.inf
        for p in preds[cur]:
            if abs((earliest_finish[p] + float(durations.get(cur, 0.0))) -
                   earliest_finish[cur]) < 1e-6:
                if earliest_finish[p] > best_ef or (
                    earliest_finish[p] == best_ef and (best is None or p < best)
                ):
                    best = p
                    best_ef = earliest_finish[p]
        if best is None:
            break
        path_rev.append(best)
        cur = best

    path = list(reversed(path_rev))
    on_path = {o: (o in set(path)) for o in ops}
    return CriticalPathReport(
        path=path,
        length=length,
        op_to_rank=earliest_finish,
        on_path=on_path,
    )
