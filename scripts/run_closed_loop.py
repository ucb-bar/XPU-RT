"""
M10 — closed-loop optimizer.

Iterative loop:
  1. Build baseline schedule with --scheduler.
  2. Generate candidates over the current workload.
  3. Score them (deterministic by predicted_delta; random for control).
  4. For each candidate in order, apply it; if the measured objective
     improves, accept and rebase; else reject.
  5. Repeat until either max_candidates evaluations have been made or no
     candidate in a full pass yielded improvement.

The deterministic ranker uses ``rewrite.score_candidates`` (which actually
re-schedules each candidate to obtain measured deltas) and then sorts by
measured improvement. The random ranker shuffles with the supplied seed.

Compare deterministic vs random for the same number of evaluations to show
that the scorer adds value over random search.

Output:
  results/closed_loop/<scenario>/
    trace.json
    pareto.png
    before.png
    after.png
    _summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from scenarios import SCENARIOS  # noqa: E402
from realistic_workloads import build_model_graph, build_workload_from_graph  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from memory_planner import plan_memory  # noqa: E402
from rewrite import (  # noqa: E402
    apply_candidate, generate_candidates, score_candidates,
)
from report import SchedulerResult, render_gantt  # noqa: E402


def _build_workload(workload_spec: str):
    """Spec format:
      'scenario:<name>'   — uses SCENARIOS[name]()
      'model:<model>@<soc>' — uses realistic_workloads
    """
    if workload_spec.startswith("scenario:"):
        name = workload_spec.split(":", 1)[1]
        wl, _, _ = SCENARIOS[name]()
        return wl, name
    if workload_spec.startswith("model:"):
        rest = workload_spec.split(":", 1)[1]
        model, soc = rest.split("@", 1)
        g = build_model_graph(model, soc)
        return build_workload_from_graph(g), f"{model}_{soc}"
    raise ValueError(f"unknown workload spec: {workload_spec}")


def _evaluate(wl, scheduler_fn) -> Dict[str, Any]:
    t, alpha, _, _ = scheduler_fn(wl)
    if t is None:
        return {"feasible": False}
    m = compute_metrics(wl, t, alpha, scheduler_name="loop_eval")
    return {
        "feasible": True,
        "t": t, "alpha": alpha, "metrics": m,
        "makespan_us": m["makespan_us"],
        "deadline_miss_count": m["deadline_miss_count"],
        "dispatch_count": len(wl.operations),
    }


def closed_loop(
    workload, scheduler_fn,
    max_candidates: int = 10,
    ranker: str = "deterministic",
    seed: int = 0,
):
    """Run the loop. Returns (final_workload, trace_list, final_eval, baseline_eval)."""
    cur = workload
    baseline_eval = _evaluate(cur, scheduler_fn)
    if not baseline_eval["feasible"]:
        raise RuntimeError("baseline infeasible")
    trace: List[Dict[str, Any]] = []
    candidates_seen = 0

    rng = random.Random(seed)
    while candidates_seen < max_candidates:
        cands = generate_candidates(cur)
        if not cands:
            break
        if ranker == "deterministic":
            # Score by ACTUAL re-schedule deltas (the scorer does this).
            scored = score_candidates(cands, cur, scheduler_fn)
            ordering = [r["candidate"]["candidate_id"] for r in scored
                        if r.get("applied") and r.get("measured_delta", 0) < 0]
        elif ranker == "cost_model":
            # M11 fast oracle: score via cost_model_score (one model forward
            # pass per candidate, no re-scheduling).
            from scheduler_ml import cost_model_score
            scored = score_candidates(cands, cur, scheduler_fn,
                                      fast_scorer=cost_model_score)
            ordering = [r["candidate"]["candidate_id"] for r in scored
                        if r.get("applied") and r.get("measured_delta", 0) < 0]
        else:
            ids = [c.candidate_id for c in cands]
            rng.shuffle(ids)
            ordering = ids

        improved_this_pass = False
        for cand_id in ordering:
            if candidates_seen >= max_candidates:
                break
            cand = next((c for c in cands if c.candidate_id == cand_id), None)
            if cand is None:
                continue
            # The "predicted" benefit (heuristic, unchanged).
            pred = cand.expected_benefit.get("predicted_makespan_delta", 0.0)
            # Build candidate's resulting workload and re-evaluate.
            try:
                new_wl = apply_candidate(cur, cand)
            except Exception as exc:
                trace.append({"iter": len(trace), "candidate_id": cand_id,
                              "accepted": False, "reason": str(exc)})
                candidates_seen += 1
                continue
            new_eval = _evaluate(new_wl, scheduler_fn)
            measured = new_eval["makespan_us"] - baseline_eval["makespan_us"] \
                if new_eval["feasible"] else float("inf")
            cur_eval = _evaluate(cur, scheduler_fn)
            improvement = (new_eval["feasible"] and
                           new_eval["makespan_us"] < cur_eval["makespan_us"])
            entry = {
                "iter": len(trace),
                "candidate_id": cand_id,
                "candidate_type": cand.type,
                "predicted_delta": pred,
                "measured_objective_before": cur_eval["makespan_us"],
                "measured_objective_after": new_eval["makespan_us"] if new_eval["feasible"] else None,
                "delta": (new_eval["makespan_us"] - cur_eval["makespan_us"]) if new_eval["feasible"] else None,
                "accepted": improvement,
                "reason": "improved" if improvement else "no_improvement_or_infeasible",
            }
            trace.append(entry)
            candidates_seen += 1
            if improvement:
                cur = new_wl
                improved_this_pass = True
                # restart the pass with a fresh candidate generation on the new wl
                break
        if not improved_this_pass:
            break

    final_eval = _evaluate(cur, scheduler_fn)
    return cur, trace, final_eval, baseline_eval


def _render_gantt(wl, ev, path: str, title: str):
    if not ev["feasible"]:
        return
    res = SchedulerResult(scheduler_name="loop", workload=wl, t=ev["t"], alpha=ev["alpha"],
                          metrics=ev["metrics"], feasible=True)
    render_gantt(res, path, title=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="scenario:fusion_win_tiny_chain",
                    help="'scenario:<name>' or 'model:<model>@<soc>'")
    ap.add_argument("--scheduler", default="heft")
    ap.add_argument("--ranker", choices=["deterministic", "random", "cost_model"], default="deterministic")
    ap.add_argument("--max-candidates", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl, tag = _build_workload(args.workload)
    out_dir = Path(args.out or (REPO / "results" / "closed_loop" / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    sched_fn = get_scheduler(args.scheduler)

    def _wrap_kwargs_sched(workload, **kw):
        if args.scheduler == "cpsat":
            kw.setdefault("time_limit", 20)
        if args.scheduler == "mosek":
            kw.setdefault("solver_verbosity", 0)
            kw.setdefault("time_limit", 20)
            kw.setdefault("restrict_makespan_to_nonperiodic", False)
            kw.setdefault("prune_cross_period_constraints", False)
        return sched_fn(workload, **kw)

    final_wl, trace, final_eval, baseline_eval = closed_loop(
        wl, _wrap_kwargs_sched,
        max_candidates=args.max_candidates,
        ranker=args.ranker, seed=args.seed,
    )

    # Trace JSON.
    with open(out_dir / f"trace_{args.ranker}.json", "w") as f:
        json.dump({
            "workload": args.workload,
            "scheduler": args.scheduler,
            "ranker": args.ranker,
            "max_candidates": args.max_candidates,
            "baseline_makespan_us": baseline_eval["makespan_us"],
            "final_makespan_us": final_eval["makespan_us"],
            "improvement_us": baseline_eval["makespan_us"] - final_eval["makespan_us"],
            "improvement_pct": (
                (baseline_eval["makespan_us"] - final_eval["makespan_us"])
                / baseline_eval["makespan_us"] * 100
            ),
            "baseline_dispatches": baseline_eval["dispatch_count"],
            "final_dispatches": final_eval["dispatch_count"],
            "trace": trace,
        }, f, indent=2)

    # Render before/after Gantts.
    _render_gantt(wl, baseline_eval, str(out_dir / f"before_{args.ranker}.png"),
                  f"BEFORE  ms={baseline_eval['makespan_us']:.0f}us  "
                  f"{baseline_eval['dispatch_count']} dispatches")
    _render_gantt(final_wl, final_eval, str(out_dir / f"after_{args.ranker}.png"),
                  f"AFTER ({args.ranker})  ms={final_eval['makespan_us']:.0f}us  "
                  f"{final_eval['dispatch_count']} dispatches")

    # Pareto plot — dispatch_count vs makespan across trace iterations.
    if trace:
        xs = list(range(len(trace) + 1))
        ms = [baseline_eval["makespan_us"]]
        for tr in trace:
            ms.append(tr["measured_objective_after"] or ms[-1])
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(xs, ms, marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("makespan (us)")
        ax.set_title(f"closed-loop progress ({args.ranker})")
        ax.grid(True, alpha=0.3)
        for i, tr in enumerate(trace):
            if tr["accepted"]:
                ax.axvline(i + 1, color="green", alpha=0.2)
        fig.savefig(out_dir / f"pareto_{args.ranker}.png", dpi=120)
        plt.close(fig)

    # Summary markdown.
    with open(out_dir / f"_summary_{args.ranker}.md", "w") as f:
        f.write(f"# closed_loop — {args.workload} via {args.scheduler} ({args.ranker})\n\n")
        f.write(f"- baseline makespan: **{baseline_eval['makespan_us']:.1f} us** "
                f"({baseline_eval['dispatch_count']} dispatches)\n")
        f.write(f"- final makespan: **{final_eval['makespan_us']:.1f} us** "
                f"({final_eval['dispatch_count']} dispatches)\n")
        f.write(f"- improvement: **{baseline_eval['makespan_us'] - final_eval['makespan_us']:.1f} us** "
                f"({(baseline_eval['makespan_us'] - final_eval['makespan_us']) / baseline_eval['makespan_us'] * 100:.1f}%)\n")
        f.write(f"- candidates evaluated: {len(trace)}\n")
        f.write(f"- candidates accepted: {sum(1 for e in trace if e['accepted'])}\n\n")
        f.write("## Trace\n\n")
        f.write("| iter | candidate | type | before | after | delta | accepted |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        for tr in trace:
            f.write(f"| {tr['iter']} | {tr['candidate_id']} | {tr['candidate_type']} | "
                    f"{tr['measured_objective_before']:.1f} | "
                    f"{tr['measured_objective_after'] or 'n/a'} | "
                    f"{tr.get('delta', 'n/a')} | "
                    f"{'YES' if tr['accepted'] else 'no'} |\n")

    print(f"\nBaseline -> {baseline_eval['makespan_us']:.1f}us / {baseline_eval['dispatch_count']} disp")
    print(f"Final    -> {final_eval['makespan_us']:.1f}us / {final_eval['dispatch_count']} disp")
    print(f"Improvement: {baseline_eval['makespan_us'] - final_eval['makespan_us']:.1f}us "
          f"({(baseline_eval['makespan_us'] - final_eval['makespan_us']) / baseline_eval['makespan_us'] * 100:.1f}%)")
    print(f"Outputs in {out_dir}")
    return baseline_eval, final_eval


if __name__ == "__main__":
    main()
