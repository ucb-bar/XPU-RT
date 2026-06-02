"""Axis C: decide whether to MERGE (fuse) or BREAK DOWN (split) dispatches.

Builds the firesim Workload in-process, then uses xpu-rt/rewrite.py to generate
fuse/split candidates and re-schedules each (rewrite.score_candidates) to measure
the predicted effect. It reports a merge-vs-split decision and emits the chosen
transform(s) as a ModelBlaster hint (Contract 2 in docs/iterative_firesim_loop.md).

Key modelling point (confirmed empirically): the predicted cost model has NO
per-dispatch launch/transition overhead, so *fusing* tiny dispatches leaves the
predicted makespan ~unchanged — its real payoff only shows on FireSim. We
therefore judge MERGE candidates by how many dispatches / cross-device
transitions they remove (concrete, with an optional overhead proxy), and judge
SPLIT candidates by their measured makespan delta (parallelism is visible
predicted). The merge decision is then confirmed on FireSim by ModelBlaster.

    python3 scripts/granularity_loop.py \
        --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
        --baseline-solver decomposed --max-per-type 12 --overhead-us 1.0
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import rewrite  # noqa: E402  (your M9 fuse/split generator + scorer)
import greedy_scheduler  # noqa: E402
import bundle as bundle_mod  # noqa: E402


def _build_workload(networks_json: str, solver: str):
    """Build the firesim Workload (+ baseline schedule) in-process, quietly."""
    import run_xpurt_schedule as R
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wl, t, alpha = R.schedule_iree_networks(networks_json, solver=solver)
    return wl, t, alpha


def _scheduler_fn(solver: str):
    fn = {"decomposed": greedy_scheduler.decomposed_schedule,
          "greedy": greedy_scheduler.greedy_schedule,
          "greedy_periodic": greedy_scheduler.greedy_periodic_schedule}.get(solver,
                                                                            greedy_scheduler.decomposed_schedule)

    def sched(wl):
        out = fn(wl)
        t, alpha = out[0], out[1]
        return t, alpha, None, None
    return sched


def decide(merges, splits, granularity_verdict: str):
    """Pure merge-vs-split decision from scored candidate rows.

    merges/splits are dicts with makespan_delta_us, dispatch_delta, obj_delta_us
    (merges) / makespan_delta_us (splits). Returns (decision, chosen, rationale).
    Split wins if it measurably lowers makespan; else a too_fine verdict + a
    dispatch-reducing merge wins; else 'none'.
    """
    best_merge = min(merges, key=lambda r: (r["obj_delta_us"], r["dispatch_delta"])) if merges else None
    cand_splits = [s for s in splits if s["makespan_delta_us"] < -1e-6]
    best_split = min(cand_splits, key=lambda r: r["makespan_delta_us"]) if cand_splits else None
    if best_split and best_split["makespan_delta_us"] < -0.5:
        return ("split", best_split,
                f"splitting {best_split['affected']} lowers predicted makespan by "
                f"{-best_split['makespan_delta_us']:.2f} us (exposes parallelism).")
    if granularity_verdict == "too_fine" and best_merge and best_merge["dispatch_delta"] < 0:
        return ("merge", best_merge,
                f"granularity is too_fine; fusing {best_merge['affected']} removes "
                f"{-best_merge['dispatch_delta']} dispatches (predicted makespan "
                f"{best_merge['makespan_delta_us']:+.2f} us — launch/transition overhead not "
                "modelled here, so confirm the win on FireSim).")
    return ("none", None,
            "granularity looks balanced; no merge/split clearly helps in the predicted model.")


def build_hint(decision: str, chosen: dict):
    """Map a chosen merge/split candidate to a ModelBlaster Contract-2 hint."""
    if not chosen:
        return None
    groups = {}
    for name in chosen.get("affected", []):
        root, local = bundle_mod._parse_name(name)
        if local is None:
            continue
        groups.setdefault(root, []).append(local)
    if decision == "merge":
        return {"contract": "modelblaster.fusion_hints/v1", "reason": chosen.get("rationale", ""),
                "networks": [{"network": k, "fuse_groups": [sorted(set(v))], "n_tiny": len(v)}
                             for k, v in groups.items()]}
    if decision == "split":
        return {"contract": "modelblaster.split_hints/v1", "reason": chosen.get("rationale", ""),
                "networks": [{"network": k, "split_ops": [{"op": o, "n_splits": 2} for o in sorted(set(v))]}
                             for k, v in groups.items()]}
    return None


def _select(cands, max_per_type: int):
    """Score a representative subset: the biggest linear-chain merges + the heavy
    split candidates (cap each type), plus a few producer/consumer fuses."""
    chains = [c for c in cands if c.type == "fuse_linear_chain"]
    pairs = [c for c in cands if c.type == "fuse_producer_consumer"]
    splits = [c for c in cands if c.type == "split_heavy_dispatch"]
    chains.sort(key=lambda c: -len(c.payload.get("chain_indices", [])))
    sel = chains[:max_per_type] + splits[:max_per_type] + pairs[:max(2, max_per_type // 3)]
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--baseline-solver", default="decomposed")
    ap.add_argument("--max-per-type", type=int, default=12)
    ap.add_argument("--overhead-us", type=float, default=1.0,
                    help="proxy per-dispatch launch overhead (us) for ranking merges; "
                         "the predicted scheduler models none. 0 = rank merges by raw count.")
    ap.add_argument("--out-dir", default="artifacts/iterate")
    ap.add_argument("--emit-hint", default=None, help="write the chosen Contract-2 hint here")
    args = ap.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"building workload ({args.baseline_solver})...", flush=True)
    wl, t0, a0 = _build_workload(args.networks_json, args.baseline_solver)
    base_dispatches = len(wl.operations)

    # advisor verdict for context
    import advisor
    import profiling
    base_report = profiling.SchedulerReport.from_solver_state(
        wl, t0, a0, solver_name=args.baseline_solver, solve_wall_s=0.0).to_dict()
    base_diag = advisor.advise_schedule(base_report, deadline_us=None)

    cands = rewrite.generate_candidates(wl)
    from collections import Counter
    by_type = dict(Counter(c.type for c in cands))
    sel = _select(cands, args.max_per_type)
    print(f"{len(cands)} candidates {by_type}; scoring {len(sel)} (re-scheduling each)...", flush=True)

    rows = rewrite.score_candidates(sel, wl, _scheduler_fn(args.baseline_solver))

    merges, splits = [], []
    for r in rows:
        c = r["candidate"]
        d_make = float(r.get("measured_delta", 0.0))               # makespan change (us)
        d_disp = int(r.get("new_dispatch_count", base_dispatches)) - base_dispatches
        # overhead-adjusted objective change (negative = better)
        obj = d_make + args.overhead_us * d_disp
        rec = {"id": c["candidate_id"], "type": c["type"], "affected": c.get("affected_ops", []),
               "makespan_delta_us": round(d_make, 3), "dispatch_delta": d_disp,
               "obj_delta_us": round(obj, 3)}
        (splits if c["type"] == "split_heavy_dispatch" else merges).append(rec)

    merges.sort(key=lambda r: (r["obj_delta_us"], r["dispatch_delta"]))   # most negative first
    splits.sort(key=lambda r: r["makespan_delta_us"])                     # most negative first
    best_merge = merges[0] if merges else None
    best_split = splits[0] if splits else None

    decision, chosen, rationale = decide(merges, splits, base_diag.granularity_verdict)
    if chosen is not None:
        chosen = dict(chosen, rationale=rationale)
    hint = build_hint(decision, chosen)

    result = {
        "baseline_solver": args.baseline_solver,
        "baseline_dispatches": base_dispatches,
        "granularity_verdict": base_diag.granularity_verdict,
        "n_candidates": len(cands), "candidate_types": by_type,
        "overhead_us_proxy": args.overhead_us,
        "decision": decision, "rationale": rationale,
        "best_merge": best_merge, "best_split": best_split,
        "top_merges": merges[:5], "top_splits": splits[:5],
        "hint": hint,
    }
    with open(os.path.join(out_dir, "granularity_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if args.emit_hint and hint:
        with open(args.emit_hint, "w") as f:
            json.dump(hint, f, indent=2)

    # ---- report ----
    print("\n" + "=" * 78)
    print(f"Granularity decision: **{decision.upper()}** — {rationale}")
    print(f"(baseline {base_dispatches} dispatches, advisor granularity={base_diag.granularity_verdict})")
    print("\nTop MERGE candidates (fuse):")
    print(f"  {'id':<46}{'Δmakespan_us':>13}{'Δdispatch':>10}{'obj_us':>9}")
    for r in merges[:5]:
        print(f"  {r['id'][:44]:<46}{r['makespan_delta_us']:>13.2f}{r['dispatch_delta']:>10}{r['obj_delta_us']:>9.2f}")
    print("\nTop SPLIT candidates:")
    print(f"  {'id':<46}{'Δmakespan_us':>13}{'Δdispatch':>10}")
    for r in splits[:5]:
        print(f"  {r['id'][:44]:<46}{r['makespan_delta_us']:>13.2f}{r['dispatch_delta']:>10}")
    print(f"\nwrote {os.path.relpath(os.path.join(out_dir, 'granularity_result.json'), REPO)}"
          + (f"; hint -> {args.emit_hint}" if (args.emit_hint and hint) else ""))
    if hint:
        print(f"chosen hint ({hint['contract']}): {json.dumps(hint['networks'])[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
