"""Deadline-aware scheduler advisor.

Turns a ``SchedulerReport`` (see profiling.py) into an actionable diagnosis:
are we meeting the deadline, which backend is the bottleneck, is dispatch
granularity too fine/coarse, and — concretely — what to try, with an estimated
saving and a projected makespan.

It composes the existing analysis pieces rather than duplicating them:
  - ``fusion_advisor.advise`` for granularity->fusion ("coarsen") recommendations
  - ``dag_analysis.find_split_opportunities`` for "finer" (split) recommendations
  - the report's own deadline_miss_count / utilization / granularity aggregates

and adds the one thing none of them do: a feasibility- and contention-aware
**rebalance** recommendation that spreads mutually-independent ops off an
overloaded backend onto idle, *feasible* backends (parallelism beats per-op
speed), plus a projected-makespan check.

Imports are kept cvxpy-free (metrics/dag_analysis/fusion_advisor/workload only),
so the CLI runs in minimal environments.

CLI (mirrors fusion_advisor):
    python3 xpu-rt/advisor.py --report scheduled_..._report.json [--deadline-us N]
        [--top-k 5] [--emit out.json] [--json] [--gantt]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fusion_advisor

IDLE_FRAC_THRESHOLD = 0.55        # frac_busy below this => backend has headroom
BOTTLENECK_FRAC_THRESHOLD = 0.70  # frac_busy above this => backend is a bottleneck
TINY_FRACTION_THRESHOLD = 0.30
MAX_REBALANCE_CANDIDATES = 256


@dataclass
class Recommendation:
    kind: str                       # "rebalance" | "coarsen" | "finer" | "none"
    target: str
    expected_savings_us: float
    confidence: str                 # "high" | "medium" | "low"
    rationale: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    network: Optional[str]
    solver_name: Optional[str]
    deadline_us: Optional[float]
    makespan_us: float
    meets_deadline: Optional[bool]
    margin_us: Optional[float]
    margin_pct: Optional[float]
    deadline_miss_count: int
    bottleneck_backend: Optional[str]
    idle_backends: List[str]
    granularity_verdict: str
    recommendations: List[Recommendation]
    projected_makespan_us: float
    projected_meets_deadline: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommendations"] = [r.to_dict() if isinstance(r, Recommendation) else r
                                for r in self.recommendations]
        return d


def _as_dict(report: Any) -> Dict[str, Any]:
    if hasattr(report, "to_dict"):
        return report.to_dict()
    if isinstance(report, dict):
        return report
    return asdict(report)


def _machines_of(target: str) -> List[str]:
    return [m for m in str(target).split("+") if m]


def _makespan(d: Dict[str, Any], dispatches: List[Dict[str, Any]]) -> float:
    if dispatches:
        return max((float(x.get("finish_us", 0.0)) for x in dispatches), default=0.0)
    return float(d.get("makespan_cycles", d.get("makespan_us", 0.0)) or 0.0)


def _levels(dispatches: List[Dict[str, Any]]) -> Dict[int, int]:
    """Longest-dependency-chain depth per dispatch id. Two ops at the same depth
    cannot be ancestor/descendant of each other, so a depth level is a valid
    antichain (set of mutually-independent ops)."""
    by_id = {x["id"]: x for x in dispatches if "id" in x}
    memo: Dict[int, int] = {}

    def depth(i: int, seen: frozenset = frozenset()) -> int:
        if i in memo:
            return memo[i]
        deps = [dp for dp in by_id.get(i, {}).get("deps", []) if dp in by_id and dp not in seen]
        lvl = 0 if not deps else 1 + max(depth(dp, seen | {i}) for dp in deps)
        memo[i] = lvl
        return lvl

    return {x["id"]: depth(x["id"]) for x in dispatches if "id" in x}


def _granularity_verdict(buckets: Dict[str, int], n: int) -> str:
    if not n:
        return "unknown"
    fine = buckets.get("lt_1k", 0) + buckets.get("lt_10k", 0)
    coarse = buckets.get("lt_1M", 0) + buckets.get("ge_1M", 0)
    if fine / n > 0.5:
        return "too_fine"
    if coarse / n > 0.5:
        return "coarse"
    return "balanced"


def _rebalance_rec(d: Dict[str, Any], dispatches: List[Dict[str, Any]],
                   util: Dict[str, Dict[str, float]], makespan: float,
                   bottleneck: str, idle: List[str]) -> Optional[Recommendation]:
    if not dispatches or not bottleneck or not idle:
        return None
    # candidates: ops on the bottleneck that are feasible on some idle backend
    cand = [x for x in dispatches
            if bottleneck in _machines_of(x.get("target", ""))
            and any(t in idle for t in x.get("feasible_targets", []))]
    if len(cand) < 2:
        return None
    cand = cand[:MAX_REBALANCE_CANDIDATES]
    levels = _levels(dispatches)
    by_level: Dict[int, List[Dict[str, Any]]] = {}
    for x in cand:
        by_level.setdefault(levels.get(x["id"], 0), []).append(x)
    # the largest depth-level is the biggest provably-independent (parallel) set
    antichain = max(by_level.values(), key=lambda g: (len(g), sum(o.get("duration_us", 0) for o in g)))
    if len(antichain) < 2:
        return None
    durs = sorted((float(o.get("duration_us", 0.0)) for o in antichain), reverse=True)
    saving = sum(durs) - durs[0]   # keep the largest on the bottleneck; others overlap
    tgts = sorted({t for o in antichain for t in o.get("feasible_targets", []) if t in idle})
    names = ", ".join(o.get("name", str(o.get("id"))) for o in antichain[:4])
    if len(antichain) > 4:
        names += ", ..."
    conf = "high" if (len(antichain) >= 3 and min(util.get(t, {}).get("frac_busy", 1.0) for t in tgts) < 0.3) else "medium"
    return Recommendation(
        kind="rebalance",
        target=bottleneck,
        expected_savings_us=round(saving, 3),
        confidence=conf,
        rationale=(
            f"{bottleneck} is {util.get(bottleneck, {}).get('frac_busy', 0) * 100:.0f}% "
            f"utilised and runs {len(antichain)} mutually-independent ops back-to-back "
            f"while {tgts} sit idle. These ops are feasible there, so spreading them lets "
            f"the branches overlap — parallelism beats per-op speed. (Try: rerun with "
            f"load-aware/EFT placement, e.g. --scheduler heft/peft.)"
        ),
        detail={"op_ids": [o["id"] for o in antichain], "moved": names,
                "from": bottleneck, "to_candidates": tgts},
    )


def _coarsen_recs(d: Dict[str, Any], makespan: float, top_k: int) -> List[Recommendation]:
    # fusion_advisor.expected_savings_pct is the % of *dispatches* collapsed, not
    # a makespan-saving %. Fusing reduces per-dispatch overhead, which we can't
    # convert to microseconds without an overhead model — so surface it
    # qualitatively (carry the collapsed-% in detail) and don't claim a us saving
    # (which would otherwise dwarf the real, makespan-level rebalance saving).
    out: List[Recommendation] = []
    try:
        for fr in fusion_advisor.advise(d, top_k=top_k):
            out.append(Recommendation(
                kind="coarsen",
                target=getattr(fr, "target", ""),
                expected_savings_us=0.0,
                confidence=getattr(fr, "confidence", "medium"),
                rationale=getattr(fr, "rationale", ""),
                detail={"fusion_kind": getattr(fr, "kind", ""),
                        "dispatches_collapsed_pct": getattr(fr, "expected_savings_pct", 0.0)},
            ))
    except Exception:
        pass
    return out


def _finer_recs(workload: Any, makespan: float) -> List[Recommendation]:
    if workload is None:
        return []
    try:
        import dag_analysis
        opps = dag_analysis.find_split_opportunities(workload)
    except Exception:
        return []
    out: List[Recommendation] = []
    for so in opps or []:
        gain = float(getattr(so, "estimated_parallelism_gain_us", 0.0) or 0.0)
        out.append(Recommendation(
            kind="finer",
            target=str(getattr(so, "op_name", getattr(so, "op_id", "?"))),
            expected_savings_us=round(gain, 3),
            confidence="medium" if getattr(so, "on_critical_path", False) else "low",
            rationale=(getattr(so, "rationale", "") or
                       "Split a heavy serial op to expose parallelism across devices."),
            detail={"op_id": getattr(so, "op_id", None)},
        ))
    return out


def advise_schedule(report: Any, *, deadline_us: Optional[float] = None,
                    workload: Any = None, top_k: int = 5) -> Diagnosis:
    d = _as_dict(report)
    dispatches = d.get("dispatches") or []
    makespan = _makespan(d, dispatches)
    util = d.get("utilization", {}) or {}
    granularity = d.get("granularity", {}) or {}
    buckets = granularity.get("buckets", {}) or {}
    n_ops = int(d.get("n_operations", len(dispatches)) or len(dispatches))

    # deadline precedence: explicit arg > report.deadline_us > None
    dl = deadline_us if deadline_us is not None else d.get("deadline_us")
    dl = float(dl) if dl is not None else None
    meets = (makespan <= dl) if dl is not None else None
    margin = (dl - makespan) if dl is not None else None
    margin_pct = (margin / dl * 100.0) if (dl not in (None, 0)) else None

    # bottleneck = busiest backend; idle = backends with headroom
    busy = {m: float(v.get("busy_cycles", 0.0)) for m, v in util.items()}
    frac = {m: float(v.get("frac_busy", 0.0)) for m, v in util.items()}
    bottleneck = max(busy, key=busy.get) if busy else None
    idle = sorted([m for m, f in frac.items() if f < IDLE_FRAC_THRESHOLD], key=lambda m: frac[m])

    recs: List[Recommendation] = []
    rebalance = None
    if bottleneck and frac.get(bottleneck, 0.0) >= BOTTLENECK_FRAC_THRESHOLD and idle:
        rebalance = _rebalance_rec(d, dispatches, util, makespan, bottleneck, idle)
        if rebalance:
            recs.append(rebalance)

    recs.extend(_coarsen_recs(d, makespan, top_k))

    # finer only when no rebalance parallelism was found (and a workload is given)
    if rebalance is None:
        recs.extend(_finer_recs(workload, makespan))

    # projected makespan: credit only the rebalance (a real, makespan-level
    # reduction), floored at the critical path so we never over-promise. Coarsen
    # savings are overhead-level and unmodeled in us, so they aren't credited.
    crit = float(d.get("critical_path", 0.0) or 0.0)
    credited = rebalance.expected_savings_us if rebalance is not None else 0.0
    projected = max(crit, makespan - credited) if (crit or credited) else makespan
    projected_meets = (projected <= dl) if dl is not None else None

    if meets and not recs:
        recs.append(Recommendation("none", "", 0.0, "high",
                                   "Deadline met with slack and no backend is a bottleneck."))

    recs.sort(key=lambda r: r.expected_savings_us, reverse=True)
    return Diagnosis(
        network=d.get("network") or d.get("solver_name"),
        solver_name=d.get("solver_name"),
        deadline_us=dl, makespan_us=makespan, meets_deadline=meets,
        margin_us=margin, margin_pct=(round(margin_pct, 1) if margin_pct is not None else None),
        deadline_miss_count=int(d.get("deadline_miss_count", 0) or 0),
        bottleneck_backend=bottleneck, idle_backends=idle,
        granularity_verdict=_granularity_verdict(buckets, n_ops),
        recommendations=recs,
        projected_makespan_us=round(projected, 3),
        projected_meets_deadline=projected_meets,
    )


def render_text(diag: Diagnosis) -> str:
    lines = []
    if diag.meets_deadline is None:
        verdict = "no deadline set"
    else:
        verdict = "MET" if diag.meets_deadline else f"MISSED by {-diag.margin_us:,.1f} us ({-diag.margin_pct:.1f}%)"
    lines.append(f"Scheduler advisor — solver={diag.solver_name}")
    lines.append(f"Deadline: {verdict}   (makespan {diag.makespan_us:,.1f} us"
                 + (f", deadline {diag.deadline_us:,.1f} us" if diag.deadline_us else "") + ")")
    lines.append("")
    lines.append("Diagnosis:")
    if diag.bottleneck_backend:
        lines.append(f"  • Bottleneck backend: {diag.bottleneck_backend}; idle: {diag.idle_backends or 'none'}.")
    lines.append(f"  • Granularity: {diag.granularity_verdict}.")
    lines.append(f"  • deadline_miss_count={diag.deadline_miss_count}.")
    if (diag.deadline_us is not None and diag.meets_deadline is False
            and diag.projected_makespan_us < diag.makespan_us - 1e-6):
        verb = "MEET" if diag.projected_meets_deadline else "still miss"
        lines.append(f"  • Applying the rebalance, projected makespan ~{diag.projected_makespan_us:,.1f} us "
                     f"would {verb} the deadline.")
    lines.append("")
    lines.append("Recommendations (ordered by expected impact):")
    if not diag.recommendations:
        lines.append("  (none)")
    for i, r in enumerate(diag.recommendations, 1):
        head = f"  {i}. [{r.kind}] {r.target}".rstrip()
        if r.expected_savings_us:
            head += f"  (~{r.expected_savings_us:,.1f} us, {r.confidence} confidence)"
        lines.append(head)
        if r.rationale:
            lines.append(f"      why: {r.rationale}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True, help="path to a SchedulerReport JSON")
    ap.add_argument("--deadline-us", type=float, default=None,
                    help="frame deadline in us (overrides report.deadline_us)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--emit", default=None, help="optional output JSON path")
    ap.add_argument("--json", action="store_true", help="print structured JSON")
    ap.add_argument("--gantt", action="store_true", help="also print the terminal Gantt")
    args = ap.parse_args()

    with open(args.report) as f:
        data = json.load(f)
    diag = advise_schedule(data, deadline_us=args.deadline_us, top_k=args.top_k)

    if args.json:
        print(json.dumps(diag.to_dict(), indent=2))
    else:
        print(render_text(diag))

    if args.gantt:
        try:
            import plot_gantt
            print()
            print(plot_gantt.render_terminal_gantt(data, deadline_us=args.deadline_us))
        except Exception as exc:
            print(f"[warn] terminal gantt unavailable: {exc}", file=sys.stderr)

    if args.emit:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit)) or ".", exist_ok=True)
        with open(args.emit, "w") as f:
            json.dump(diag.to_dict(), f, indent=2)
        print(f"wrote {args.emit}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
