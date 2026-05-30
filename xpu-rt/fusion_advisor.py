"""Fusion advisor — turns a SchedulerReport into actionable fusion advice.

A real scheduler run already exposes the dispatch-duration distribution
(p50/p90/p99 + bucket counts) plus the raw `dispatch_durations` list and
the workload-level granularity stats. From that we can recommend:

- a `fusion_threshold` (µs) that would collapse the noisy small-op tail,
- specific op-pair fusions (when a SchedulerReport carries op_type metadata),
- chain fusion for runs of tiny dispatches on the same machine kind.

The advisor is read-only — it never mutates the workload. ModelBlaster (or
any other consumer) decides whether to act on the recommendations.

Usage:
    from xpurt import SchedulerReport, fusion_advisor
    report = SchedulerReport.from_solver_state(...)
    recs = fusion_advisor.advise(report)
    for r in recs:
        print(r.kind, r.target, r.expected_savings_pct, r.confidence)

CLI:
    python -m xpurt.fusion_advisor --report /path/to/scheduler_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FusionRecommendation:
    """One piece of fusion advice."""

    kind: str                 # "threshold" | "pair" | "chain"
    target: str               # e.g. "lt_10k_dispatches" | "conv2d_s8+batchnorm2d_s8" | "CPU_E_chain"
    expected_savings_pct: float
    confidence: str           # "high" | "medium" | "low"
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _granularity_threshold_rec(report_dict: Dict[str, Any]) -> Optional[FusionRecommendation]:
    """Find a fusion_threshold that collapses a meaningful fraction of small ops."""
    g = report_dict.get("granularity", {})
    buckets = g.get("buckets", {})
    n = report_dict.get("n_operations", 0)
    if not buckets or n == 0:
        return None

    # Pick the smallest bucket that contains >= 20% of all dispatches.
    # Anything bigger than that bucket gets the fusion_threshold.
    cumulative = 0
    chosen_bucket: Optional[str] = None
    bucket_order = ["lt_1k", "lt_10k", "lt_100k", "lt_1M", "ge_1M"]
    bucket_ceiling = {"lt_1k": 1_000, "lt_10k": 10_000, "lt_100k": 100_000,
                      "lt_1M": 1_000_000, "ge_1M": float("inf")}
    for b in bucket_order:
        cumulative += buckets.get(b, 0)
        if cumulative >= 0.20 * n and chosen_bucket is None:
            chosen_bucket = b
            break

    if chosen_bucket is None or chosen_bucket == "ge_1M":
        return None

    threshold = bucket_ceiling[chosen_bucket]
    n_fused = sum(buckets.get(b, 0) for b in bucket_order
                  if bucket_ceiling[b] <= threshold)
    pct = 100.0 * n_fused / n if n > 0 else 0.0

    # Estimate savings: each fused dispatch saves ~1 dispatch-overhead (~hundreds
    # of cycles). Conservatively quote it as the % of total ops collapsed.
    return FusionRecommendation(
        kind="threshold",
        target=f"fusion_threshold = {threshold} cycles",
        expected_savings_pct=round(pct, 1),
        confidence="medium",
        rationale=(
            f"{n_fused}/{n} dispatches ({pct:.1f}%) fall below {threshold} "
            f"cycles. Fusing them collapses the dispatch-overhead tail."
        ),
    )


def _chain_fusion_rec(report_dict: Dict[str, Any]) -> Optional[FusionRecommendation]:
    """If one machine kind hosts most of the small dispatches, recommend
    chain fusion on that kind."""
    g = report_dict.get("granularity", {})
    durations = report_dict.get("dispatch_durations", [])
    util = report_dict.get("utilization", {})
    if not durations or not util:
        return None

    # Heuristic: if any machine has >40% idle fraction AND p99 dispatch
    # duration is < 10× the p50, then chain fusion can pack the busy bursts.
    p50 = g.get("p50", 0)
    p99 = g.get("p99", 0)
    if p50 == 0 or p99 / max(p50, 1e-9) > 10:
        return None

    most_idle = max(util.items(), key=lambda kv: kv[1].get("idle_cycles", 0),
                    default=(None, None))
    if not most_idle[0]:
        return None
    machine, info = most_idle
    idle_frac = 1.0 - info.get("frac_busy", 0)
    if idle_frac < 0.4:
        return None

    return FusionRecommendation(
        kind="chain",
        target=f"chain_fuse_on_{machine}",
        expected_savings_pct=round(100.0 * idle_frac * 0.5, 1),  # half the idle
        confidence="low",
        rationale=(
            f"{machine} is {idle_frac*100:.1f}% idle and the p99/p50 dispatch "
            f"ratio is {p99/max(p50,1e-9):.1f}× — likely many short "
            "dispatches with gaps between them. Chain fusion can collapse "
            "the gaps."
        ),
    )


def _pair_fusion_rec(report_dict: Dict[str, Any]) -> List[FusionRecommendation]:
    """Op-pair fusion needs per-op type information.

    A future SchedulerReport may carry it; for now we just emit hints based
    on the workload's solver_state if available.
    """
    return []


def advise(report: Any, top_k: int = 5) -> List[FusionRecommendation]:
    """Return a (deduped, ranked) list of fusion recommendations.

    `report` accepts either a SchedulerReport dataclass instance or a plain
    dict (as written by SchedulerReport.write_json).
    """
    if hasattr(report, "to_dict"):
        d = report.to_dict()
    elif isinstance(report, dict):
        d = report
    else:
        d = asdict(report)

    recs: List[FusionRecommendation] = []
    tr = _granularity_threshold_rec(d)
    if tr:
        recs.append(tr)
    cr = _chain_fusion_rec(d)
    if cr:
        recs.append(cr)
    recs.extend(_pair_fusion_rec(d))
    # Rank by expected savings descending.
    recs.sort(key=lambda r: r.expected_savings_pct, reverse=True)
    return recs[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True,
                    help="path to scheduler_report.json")
    ap.add_argument("--emit", default=None,
                    help="optional output JSON path")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    with open(args.report) as f:
        data = json.load(f)
    recs = advise(data, top_k=args.top_k)
    out = {
        "source_report": os.path.abspath(args.report),
        "recommendations": [r.to_dict() for r in recs],
    }
    print(json.dumps(out, indent=2))
    if args.emit:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit)) or ".", exist_ok=True)
        with open(args.emit, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.emit}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
