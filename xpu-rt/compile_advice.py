"""Turn a schedule plus measured K1 profiles into a machine-readable compiler contract.

This is the narrow interface between "the scheduler noticed something" and "the
compiler does something about it". The contract is the JSON; the prose is a
courtesy field, never the payload.

Design rules that fall out of what the measurements actually showed:

* **Evidence or nothing.** Every recommendation carries the numbers that
  produced it. A recommendation nobody can audit is worse than no
  recommendation, because it still gets acted on.

* **Recommend against a change when the data says so.** Profiling DroNet with
  IME made it 7.9% *slower* overall -- only one `vmadot` was emitted, so the
  build paid different data-tiling costs without the benefit. An advisor that
  can only ever say "use the accelerator" is a press release. Per dispatch,
  IME wins twice out of nineteen, and the advice says exactly that.

* **Separate blocking from compute.** A dispatch that is long is not
  necessarily a problem; a long dispatch that blocks a periodic deadline is.
  The K1 run spent 87.6% of its elapsed time queueing, so `blocking_time_us` is
  not an afterthought here -- it is usually the dominant term.

* **`split` and `shard` are different words.** Split cuts one dispatch into
  smaller sequential pieces so the scheduler can interleave them. Shard runs one
  dispatch's work on several cores at once. They need different legality
  evidence and they fix different problems.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

RECOMMENDATIONS = (
    "split", "fuse_with_predecessor", "fuse_with_successor",
    "choose_implementation", "pin_core_class", "shard", "coarsen", "unchanged",
)


@dataclass
class Evidence:
    service_time_us: float = 0.0
    blocking_time_us: float = 0.0
    periodic_free_slot_us: float = 0.0
    deadline_misses_attributed: int = 0
    on_critical_path: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Advice:
    model: str
    dispatch_id: Any
    recommendation: str
    priority: int
    confidence: str
    evidence: Evidence
    constraints: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        ev = d.pop("evidence")
        extra = ev.pop("extra", {}) or {}
        ev.update(extra)
        d["evidence"] = ev
        return d


def _load_jsonl(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["dispatch_id"]] = r
    return out


def implementation_advice(
    model: str,
    profiles_by_impl: Dict[str, Dict[int, dict]],
    baseline_impl: str,
    min_gain: float = 0.05,
) -> List[Advice]:
    """Per dispatch, is some other measured implementation actually faster?

    `min_gain` exists so a recommendation has to clear measurement noise before
    it is made. The CV on these profiles is 0.2-1.4%, so 5% is a comfortable
    margin; without it the advisor would churn kernels for nothing.
    """
    out: List[Advice] = []
    base = profiles_by_impl.get(baseline_impl, {})
    for did, brec in sorted(base.items()):
        b = brec["median_ms"]
        if b <= 0:
            continue
        best_impl, best = baseline_impl, b
        for impl, prof in profiles_by_impl.items():
            r = prof.get(did)
            if r and r["median_ms"] < best:
                best_impl, best = impl, r["median_ms"]
        gain = (b - best) / b
        op = brec.get("module_name", "")
        if best_impl != baseline_impl and gain >= min_gain:
            out.append(Advice(
                model=model, dispatch_id=did,
                recommendation="choose_implementation",
                priority=1 if b >= 1.0 else 2,
                confidence="high" if brec.get("cv_pct", 100) < 5 else "medium",
                evidence=Evidence(
                    service_time_us=round(b * 1000, 2),
                    extra={
                        "baseline_impl": baseline_impl,
                        "baseline_median_us": round(b * 1000, 2),
                        "proposed_impl": best_impl,
                        "proposed_median_us": round(best * 1000, 2),
                        "gain_fraction": round(gain, 4),
                        "baseline_cv_pct": brec.get("cv_pct"),
                        "op": op,
                    }),
                constraints={
                    # IME executes only on cluster 0; measured by SIGILL probe.
                    "legal_resources": (["k1_cluster0"] if best_impl.lower().startswith("ime")
                                        else ["k1_cluster0", "k1_cluster1"]),
                },
                rationale=(f"{best_impl} is {gain*100:.1f}% faster than "
                           f"{baseline_impl} for this dispatch "
                           f"({best*1000:.0f}us vs {b*1000:.0f}us)."),
            ))
        else:
            # Recording the negative result matters: it is what stops a later
            # round from re-proposing a change already measured as not worth it.
            out.append(Advice(
                model=model, dispatch_id=did, recommendation="unchanged",
                priority=5, confidence="high",
                evidence=Evidence(
                    service_time_us=round(b * 1000, 2),
                    extra={"baseline_impl": baseline_impl,
                           "best_alternative": best_impl,
                           "gain_fraction": round(gain, 4),
                           "min_gain_required": min_gain,
                           "op": op}),
                rationale=(f"no measured alternative beats {baseline_impl} by "
                           f"{min_gain*100:.0f}% (best was {best_impl} at "
                           f"{gain*100:+.1f}%)."),
            ))
    return out


def overhead_advice(model: str, profile: Dict[int, dict],
                    chain: bool, floor_ratio: float = 0.75) -> List[Advice]:
    """Flag a model whose dispatches are dominated by per-dispatch overhead.

    If the cheapest dispatch in a linear chain costs nearly as much as the
    dearest, the model is paying launch cost rather than doing work, and the
    fix is upstream: emit fewer, larger dispatches. Requires a linear chain --
    fusing a branchy region changes semantics, and `fusion.py` will not do it.
    """
    if not profile or not chain:
        return []
    meds = sorted(r["median_ms"] for r in profile.values())
    if len(meds) < 2 or meds[-1] <= 0:
        return []
    ratio = meds[0] / meds[-1]
    if ratio < floor_ratio:
        return []
    total = sum(meds)
    floor = meds[0] * len(meds)
    return [Advice(
        model=model, dispatch_id="*",
        recommendation="fuse_with_successor",
        priority=1, confidence="high",
        evidence=Evidence(
            service_time_us=round(total * 1000, 2),
            extra={"n_dispatches": len(meds),
                   "min_median_us": round(meds[0] * 1000, 2),
                   "max_median_us": round(meds[-1] * 1000, 2),
                   "min_over_max": round(ratio, 3),
                   "estimated_overhead_us": round(floor * 1000, 2),
                   "estimated_overhead_fraction": round(floor / total, 3)}),
        constraints={"requires_linear_chain": True},
        rationale=(f"all {len(meds)} dispatches cost within "
                   f"{(1-ratio)*100:.0f}% of each other "
                   f"({meds[0]*1000:.0f}-{meds[-1]*1000:.0f}us) regardless of "
                   f"work, so ~{100*floor/total:.0f}% of the "
                   f"{total*1000:.0f}us total is per-dispatch launch overhead. "
                   f"Emitting fewer, larger dispatches is the only lever that "
                   f"touches it."),
    )]


def blocking_advice(model: str, profile: Dict[int, dict],
                    free_slot_ms: float, misses: int) -> List[Advice]:
    """Dispatches too long to fit the tightest periodic slot they must share."""
    out = []
    for did, r in sorted(profile.items()):
        svc = r["median_ms"]
        if free_slot_ms > 0 and svc > free_slot_ms:
            out.append(Advice(
                model=model, dispatch_id=did,
                recommendation="split", priority=1,
                confidence="high" if r.get("cv_pct", 100) < 5 else "medium",
                evidence=Evidence(
                    service_time_us=round(svc * 1000, 2),
                    periodic_free_slot_us=round(free_slot_ms * 1000, 2),
                    deadline_misses_attributed=misses,
                    on_critical_path=True,
                    extra={"op": r.get("module_name", ""),
                           "overrun_factor": round(svc / free_slot_ms, 2)}),
                constraints={
                    "max_target_piece_us": round(free_slot_ms * 1000, 2),
                    "legal_resources": ["k1_cluster0", "k1_cluster1"],
                },
                rationale=(f"service time {svc*1000:.0f}us exceeds the "
                           f"{free_slot_ms*1000:.0f}us slot available between "
                           f"periodic releases, so it blocks non-preemptively "
                           f"for {svc/free_slot_ms:.1f}x the slot."),
            ))
    return out


def write_advice(path: str, advice: List[Advice], *,
                 schedule_id: str, notes: Optional[Dict[str, Any]] = None) -> None:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": schedule_id,
        "advice": [a.as_dict() for a in advice],
    }
    if notes:
        doc["notes"] = notes
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def load_profiles(gen_root: str, target: str, model: str, basename: str,
                  impls: List[str], topo_tag: str = "topo_0") -> Dict[str, Dict[int, dict]]:
    out = {}
    for impl in impls:
        p = os.path.join(gen_root, "profile", impl, target, model, basename,
                         topo_tag, "profile.jsonl")
        rec = _load_jsonl(p)
        if rec:
            out[impl] = rec
    return out
