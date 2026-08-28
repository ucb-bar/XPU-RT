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

import csv
import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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
    min_gain_ms: float = 0.05,
) -> List[Advice]:
    """Per dispatch, is some other measured implementation actually faster?

    Two floors, and the second one exists because the first is not enough.

    `min_gain` (relative) keeps a recommendation from chasing measurement noise.
    It was set at 5% against profiles whose CV was 0.2-1.4%.

    `min_gain_ms` (absolute) is what stops a large PERCENTAGE of a negligible
    cost from reading as an opportunity. Without it this advisor emitted, from a
    real profile:

        dronet.20  choose_implementation  scalar is 14.7% faster than rvv_x60
                   for this dispatch (1us vs 1us)

    -- a recommendation to regenerate a kernel to save a fraction of a
    microsecond on a 1.4 us op, printed with two identical numbers because the
    difference does not survive rounding to the unit it is reported in. Acting on
    it costs a Codex call, a rebuild, a board run and a re-solve.

    It matters more than tidiness because the acceptance objective is
    lexicographic with standalone kernel cycles LAST: a change worth 0.2 us
    cannot move any term above it, so the best case is a no-op and the realistic
    case is that it perturbs a schedule that currently misses zero deadlines.

    The relative floor cannot cover this on its own -- 14.7% clears any
    percentage threshold you would set for a real win -- and the absolute floor
    cannot cover it either, since a 5% gain on a 200 ms dispatch is worth having.
    Both, or neither works.

    Note both floors are still weaker than they look while profiles are n=1:
    with no `cycles_cv_pct` there is no measured noise floor to compare against,
    and a 10-12% gain on one sample is not distinguishable from run-to-run
    variation. That is why every such item comes out `medium` confidence.
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
        gain_ms = b - best
        if best_impl != baseline_impl and gain >= min_gain \
                and gain_ms >= min_gain_ms:
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
                        # Absolute, not just relative: a large percentage of a
                        # negligible cost is what min_gain_ms exists to refuse,
                        # and a consumer weighing this against the cost of a
                        # rebuild needs the milliseconds.
                        "gain_ms": round(gain_ms, 4),
                        "min_gain_ms_required": min_gain_ms,
                        "baseline_cv_pct": brec.get("cv_pct"),
                        "op": op,
                        # Which kernel the baseline number was actually timed
                        # on. Blank for producers that do not record it.
                        "baseline_kernel": brec.get("implementation") or None,
                        "stat_basis": brec.get("stat_basis") or "median_of_reps",
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
                priority=5,
                # Same gate as the positive branch. A "no alternative wins"
                # verdict drawn from a single sample is not high confidence
                # either, and hardcoding "high" here made the two branches
                # disagree about what the same measurement supports.
                confidence="high" if brec.get("cv_pct", 100) < 5 else "medium",
                evidence=Evidence(
                    service_time_us=round(b * 1000, 2),
                    extra={"baseline_impl": baseline_impl,
                           "best_alternative": best_impl,
                           "gain_fraction": round(gain, 4),
                           "min_gain_required": min_gain,
                           "min_gain_ms_required": min_gain_ms,
                           "gain_ms": round(gain_ms, 4),
                           "op": op,
                           "baseline_kernel": brec.get("implementation") or None,
                           "stat_basis": (brec.get("stat_basis")
                                          or "median_of_reps")}),
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


def _load_results_csv(path: str, n_cores: Optional[int] = None) -> Dict[int, dict]:
    """{dispatch_id: rec} from an IREE-shape `results.csv`.

    The second profile producer on this board. `runtime/scripts/profile_k1.py`
    writes `profile.jsonl` (all samples, percentiles, cv); ModelBlaster's
    `pipeline/profile_writer.py` writes IREE's `results.csv`, and that is the
    only format the corrected `rvv_x60` builds exist in. Without this the
    advisor could only ever be regenerated against the older IREE-path
    profiles -- which is exactly the stale-input failure this exists to fix.

    Two honesty markers travel with every record, because they change what the
    advice may claim:

    * `stat_basis`. `results.csv` carries a single `mean_time` per dispatch, not
      a distribution. The harness that produced these takes ONE sample, so
      `median_ms` here is that one sample, not a median over warm repetitions.
      `cv_pct` is therefore deliberately ABSENT rather than invented -- callers
      default it to 100, which downgrades confidence to "medium", which is the
      correct answer for n=1.
    * `implementation`. The column that records which kernel actually ran
      (`curated[rvv]/rvv_oc_blocked_bn_epilogue`, and so on). It exists because
      curated kernels were looked up by exact op name, so fused ops silently
      fell back to the scalar reference inside builds labelled `rvv_x60`. Any
      advice derived from a profile whose rows say `scalar` while the build says
      `rvv` is advice about a binary nobody shipped, so the string is carried
      into the evidence rather than left on disk.
    """
    out: Dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("dispatch_id") or "").strip()
            if not raw:
                continue
            try:
                did = int(raw)
            except ValueError:
                continue
            try:
                t = float(row.get("mean_time") or 0.0)
            except ValueError:
                continue
            unit = (row.get("mean_unit") or "ms").strip()
            ms = t / 1000.0 if unit == "us" else (t * 1000.0 if unit == "s" else t)
            rec = {
                "dispatch_id": did,
                "module_name": row.get("module_name", ""),
                # One sample; see `stat_basis`. Named `median_ms` because that
                # is the key every advisor here reads -- renaming it would fork
                # the two producers apart again.
                "median_ms": ms,
                "mean_ms": ms,
                "stat_basis": "single_sample_mean",
                "op": row.get("op", ""),
                "shape": row.get("shape", ""),
                "implementation": row.get("implementation", ""),
                "source": row.get("source", ""),
                "source_csv": path,
            }
            if row.get("cycles"):
                try:
                    rec["cycles"] = int(float(row["cycles"]))
                except ValueError:
                    pass
            if n_cores is not None:
                rec["n_cores"] = n_cores
            out[did] = rec
    return out


def n_cores_from_topo_tag(tag: str) -> int:
    """`topo_0_1_2_3` -> 4. The tag lists the harts the run actually held."""
    return max(1, len(tag.split("_")) - 1)


def _find_results_csv(gen_root: str, impl: str, target: str, model: str,
                      basename: str, topo_tag: str) -> Optional[str]:
    """Most recently modified `results.csv` for this (impl, model, topo).

    Handles both layouts, same as `profile_loader.find_profile_csv`: the
    ModelBlaster/IREE one interposes a spec directory
    (`<model>_<target>_<impl>_<basename>`) between the basename and the topo
    tag, and the older one does not.
    """
    root = os.path.join(gen_root, "profile", impl, target, model, basename)
    for pat in (os.path.join(root, "*", topo_tag, "results.csv"),
                os.path.join(root, topo_tag, "results.csv")):
        hits = glob.glob(pat)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def load_profiles_csv(gen_root: str, target: str, model: str, basename: str,
                      impls: List[str], topo_tag: str = "topo_0"
                      ) -> Dict[str, Dict[int, dict]]:
    """`load_profiles`, for the `results.csv` producer. Same return shape."""
    out: Dict[str, Dict[int, dict]] = {}
    n = n_cores_from_topo_tag(topo_tag)
    for impl in impls:
        p = _find_results_csv(gen_root, impl, target, model, basename, topo_tag)
        if not p:
            continue
        rec = _load_results_csv(p, n_cores=n)
        if rec:
            out[impl] = rec
    return out


def load_profiles_by_cores_csv(gen_root: str, target: str, model: str,
                               basename: str, impl: str,
                               topo_tags: Sequence[str] = (
                                   "topo_0", "topo_0_1", "topo_0_1_2_3",
                                   "topo_0_1_2_3_4_5_6_7")
                               ) -> Dict[int, Dict[int, dict]]:
    """`load_profiles_by_cores`, for the `results.csv` producer.

    Returns `{n_cores: {did: rec}}`. A tree that only ever profiled one core
    count comes back with a single entry, and `shard_advice` then has nothing
    to measure -- which is the honest outcome, not a bug to route around.
    """
    out: Dict[int, Dict[int, dict]] = {}
    for tag in topo_tags:
        p = _find_results_csv(gen_root, impl, target, model, basename, tag)
        if not p:
            continue
        n = n_cores_from_topo_tag(tag)
        rec = _load_results_csv(p, n_cores=n)
        if rec:
            out[n] = rec
    return out


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

def load_profiles_by_cores(gen_root: str, target: str, model: str,
                           basename: str, impl: str,
                           topo_tags: Sequence[str] = (
                               "topo_0", "topo_0_1", "topo_0_1_2_3",
                               "topo_0_1_2_3_4_5_6_7")
                           ) -> Dict[int, Dict[int, dict]]:
    """Per-dispatch cost as a function of CORE COUNT: {n_cores: {did: rec}}.

    Nothing in XPU-RT could express this before. `load_profiles` takes a single
    `topo_tag` (defaulting to the 1-core `topo_0`), and `profile_loader` drops
    everything but the time and module name, so the `n_cores` field that the
    profiler already writes into every record had no readers at all.

    That gap is why no advisor could ever justify a `shard` recommendation:
    sharding is a claim about how a dispatch's cost changes when you give it
    more cores, and the measurement existed on disk while being unreachable
    through the API.

    Keyed on the `n_cores` recorded in the profile rather than on the topo tag,
    so the tag naming convention stays an implementation detail of where the
    file lives.
    """
    out: Dict[int, Dict[int, dict]] = {}
    for tag in topo_tags:
        p = os.path.join(gen_root, "profile", impl, target, model, basename,
                         tag, "profile.jsonl")
        rec = _load_jsonl(p)
        if not rec:
            continue
        n = None
        for r in rec.values():
            if r.get("n_cores"):
                n = int(r["n_cores"])
                break
        if n is None:
            # Fall back to the tag, which encodes count as topo_0_1_..._n-1.
            n = len(tag.split("_")) - 1
        out[n] = rec
    return out


def shard_advice(model: str, profiles_by_cores: Dict[int, Dict[int, dict]],
                 free_slot_ms: float, min_speedup: float = 1.5,
                 min_efficiency: float = 0.5) -> List[Advice]:
    """Recommend spreading ONE dispatch across several cores.

    Distinct from `split`, and the distinction is not cosmetic: split cuts a
    dispatch into smaller sequential pieces so each fits a scheduling slot;
    shard runs one dispatch's work on several cores at once. They need
    different evidence and they fix different problems, so conflating them
    produces recommendations that cannot be acted on.

    Gated on measurement, not on hope. A dispatch is worth sharding when:

      1. it does not fit the slot on one core (otherwise leave it alone);
      2. its measured cost actually falls with core count -- `speedup(n) =
         cost(1)/cost(n)` reaches `min_speedup`;
      3. that speedup is not bought at absurd cost -- parallel efficiency
         `speedup(n)/n` stays above `min_efficiency`.

    Condition 2 is what makes this honest. On the measured K1 data it emits
    shard for DroNet's heavy convolutions (dispatch 1: 22.87 -> 6.10 ms on four
    cores, 3.75x) and correctly refuses for every MLP dispatch, whose cost gets
    *worse* with more cores (0.066 -> 0.094 ms) because they are dominated by
    per-dispatch overhead rather than work. An advisor that recommended
    sharding from op size alone would get the MLP exactly backwards.

    `sync_overhead_us = n*cost(n) - cost(1)` is reported as evidence: it is the
    extra total work the split costs, which is what a summed-cycles objective
    would (wrongly) reject the change for.
    """
    out: List[Advice] = []
    if 1 not in profiles_by_cores or free_slot_ms <= 0:
        return out
    base = profiles_by_cores[1]
    core_counts = sorted(n for n in profiles_by_cores if n > 1)

    for did, rec in sorted(base.items()):
        c1 = float(rec.get("median_ms") or 0.0)
        if c1 <= 0 or c1 <= free_slot_ms:
            continue  # fits already: nothing to fix

        best = None
        for n in core_counts:
            r = profiles_by_cores[n].get(did)
            if not r:
                continue
            cn = float(r.get("median_ms") or 0.0)
            if cn <= 0:
                continue
            speedup = c1 / cn
            eff = speedup / n
            if speedup < min_speedup or eff < min_efficiency:
                continue
            if best is None or cn < best[1]:
                best = (n, cn, speedup, eff)

        if best is None:
            # Measured and refused: record it, so the next round does not
            # re-propose a shard the data has already rejected.
            fastest = min(
                ((n, float(profiles_by_cores[n][did].get("median_ms") or 0.0))
                 for n in core_counts if did in profiles_by_cores[n]),
                key=lambda t: t[1], default=None)
            detail = (f"best measured {fastest[1]:.4f} ms on {fastest[0]} cores "
                      f"vs {c1:.4f} on 1" if fastest else "no multi-core profile")
            out.append(Advice(
                model=model, dispatch_id=did, recommendation="unchanged",
                priority=3, confidence="high",
                evidence=Evidence(
                    service_time_us=c1 * 1000.0,
                    periodic_free_slot_us=free_slot_ms * 1000.0,
                    extra={"reason": "does not shard profitably",
                           "detail": detail}),
                rationale=(f"dispatch {did} overruns the slot but does not "
                           f"speed up enough with more cores to be worth "
                           f"sharding ({detail})")))
            continue

        n, cn, speedup, eff = best
        out.append(Advice(
            model=model, dispatch_id=did, recommendation="shard",
            priority=1 if cn <= free_slot_ms else 2,
            confidence="high" if eff >= 0.7 else "medium",
            evidence=Evidence(
                service_time_us=c1 * 1000.0,
                periodic_free_slot_us=free_slot_ms * 1000.0,
                on_critical_path=True,
                extra={
                    "cost_1core_ms": round(c1, 4),
                    f"cost_{n}core_ms": round(cn, 4),
                    "n_cores": n,
                    "measured_speedup": round(speedup, 3),
                    "parallel_efficiency": round(eff, 3),
                    "sync_overhead_us": round((n * cn - c1) * 1000.0, 1),
                    "fits_slot_after": cn <= free_slot_ms,
                }),
            constraints={"n_cores": n,
                         "legal_resources": ["k1_cluster0", "k1_cluster1"]},
            rationale=(f"dispatch {did} takes {c1:.3f} ms on one core against a "
                       f"{free_slot_ms:.3f} ms slot, and measures {cn:.3f} ms on "
                       f"{n} cores ({speedup:.2f}x, {eff:.0%} efficient)"
                       + (" -- which fits" if cn <= free_slot_ms
                          else " -- still over, but closer"))))
    return out
