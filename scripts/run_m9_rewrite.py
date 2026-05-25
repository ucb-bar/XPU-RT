"""
M9 driver — generate fuse/split candidates on dronet (chipyard) and on the
fusion-trap diagnostic scenario, score them with HEFT, and:

  1. Pick the top fuse candidate on a tiny-op chain — confirm it reduces
     both dispatch count and makespan vs the un-fused baseline.
  2. On the fusion_trap scenario, apply MAX-fusion (all linear pairs) and
     show it strictly hurts makespan vs the un-fused baseline (negative
     result is the headline).
  3. Compute Spearman correlation between predicted_delta and measured_delta
     across all generated candidates on the dronet graph.

Outputs:
  results/rewrite/dronet_chipyard_candidates.json
  results/rewrite/_fusion_win_delta.json
  results/rewrite/_fusion_trap_evidence.json
  results/rewrite/_scorer_correlation.json
  plots/m9_fusion_win_before.png
  plots/m9_fusion_win_after.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from realistic_workloads import build_model_graph, build_workload_from_graph  # noqa: E402
from scenarios import fusion_win_tiny_chain, fusion_trap_parallel_branches  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from postprocessing import validate_schedule  # noqa: E402
from report import SchedulerResult, render_gantt  # noqa: E402
from rewrite import (  # noqa: E402
    apply_candidate, generate_candidates,
    score_candidates, spearman_correlation,
)


def main():
    out = REPO / "results" / "rewrite"
    plots = REPO / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    heft = get_scheduler("heft")

    # ------------------------------------------------------------------
    # 1. dronet/chipyard: generate + score
    # ------------------------------------------------------------------
    g = build_model_graph("dronet", "chipyard")
    wl = build_workload_from_graph(g)
    cands = generate_candidates(wl)
    print(f"dronet/chipyard candidates: {len(cands)}")
    scored = score_candidates(cands, wl, heft)
    with open(out / "dronet_chipyard_candidates.json", "w") as f:
        json.dump(scored, f, indent=2)
    print(f"-> {out / 'dronet_chipyard_candidates.json'}")
    # Spearman correlation on candidates that were applicable.
    applied = [r for r in scored if r.get("applied")]
    pred = [r["predicted_delta"] for r in applied]
    meas = [r["measured_delta"] for r in applied]
    rho = spearman_correlation(pred, meas)
    with open(out / "_scorer_correlation.json", "w") as f:
        json.dump({
            "n_candidates_applied": len(applied),
            "spearman_rho_predicted_vs_measured": rho,
            "predicted_deltas": pred,
            "measured_deltas": meas,
        }, f, indent=2)
    print(f"Spearman rho (predicted vs measured) over {len(applied)} candidates: {rho:.3f}")

    # ------------------------------------------------------------------
    # 2. fusion_win_tiny_chain — top fuse should reduce makespan AND dispatches
    # ------------------------------------------------------------------
    wl_tiny, _, _ = fusion_win_tiny_chain()
    cands_tiny = generate_candidates(wl_tiny)
    scored_tiny = score_candidates(cands_tiny, wl_tiny, heft)
    print(f"\nfusion_win_tiny_chain: {len(cands_tiny)} candidates, "
          f"top scored: {scored_tiny[0]['candidate']['candidate_id'] if scored_tiny else 'none'}")
    if scored_tiny:
        top = scored_tiny[0]
        # Apply the top candidate and render a Gantt of before/after.
        new_wl = apply_candidate(wl_tiny, _candidate_from_dict(top["candidate"], cands_tiny))
        t_before, a_before, _, _ = heft(wl_tiny)
        t_after, a_after, _, _ = heft(new_wl)
        m_before = compute_metrics(wl_tiny, t_before, a_before, scheduler_name="heft_before")
        m_after = compute_metrics(new_wl, t_after, a_after, scheduler_name="heft_after")
        evidence = {
            "top_candidate": top["candidate"],
            "before": {"makespan_us": m_before["makespan_us"],
                       "dispatch_count": len(wl_tiny.operations)},
            "after": {"makespan_us": m_after["makespan_us"],
                      "dispatch_count": len(new_wl.operations)},
            "improved_makespan": m_after["makespan_us"] < m_before["makespan_us"],
            "reduced_dispatches": len(new_wl.operations) < len(wl_tiny.operations),
        }
        with open(out / "_fusion_win_delta.json", "w") as f:
            json.dump(evidence, f, indent=2)
        print(f"  before: {m_before['makespan_us']:.1f}us / {len(wl_tiny.operations)} dispatches")
        print(f"  after : {m_after['makespan_us']:.1f}us / {len(new_wl.operations)} dispatches")
        # Render Gantts.
        SchedulerResult.__init__  # ensure import
        res_b = SchedulerResult(scheduler_name="heft", workload=wl_tiny, t=t_before,
                                alpha=a_before, metrics=m_before, feasible=True)
        render_gantt(res_b, str(plots / "m9_fusion_win_before.png"),
                     title=f"fusion_win — BEFORE  ms={m_before['makespan_us']:.0f}us  "
                           f"{len(wl_tiny.operations)} dispatches")
        res_a = SchedulerResult(scheduler_name="heft", workload=new_wl, t=t_after,
                                alpha=a_after, metrics=m_after, feasible=True)
        render_gantt(res_a, str(plots / "m9_fusion_win_after.png"),
                     title=f"fusion_win — AFTER (top={top['candidate']['candidate_id']})  "
                           f"ms={m_after['makespan_us']:.0f}us  "
                           f"{len(new_wl.operations)} dispatches")

    # ------------------------------------------------------------------
    # 3. fusion_trap_parallel_branches — max-fusion should HURT
    # ------------------------------------------------------------------
    wl_trap, _, _ = fusion_trap_parallel_branches()
    t_base, a_base, _, _ = heft(wl_trap)
    m_base = compute_metrics(wl_trap, t_base, a_base, scheduler_name="heft_base")
    # Apply ALL fuse_producer_consumer candidates we can find.
    cands_trap = generate_candidates(wl_trap)
    max_fuse_wl = wl_trap
    applied_count = 0
    for c in cands_trap:
        if c.type == "fuse_producer_consumer":
            try:
                new_idx_prod = next((i for i, op in enumerate(max_fuse_wl.operations)
                                     if op.operation_name == c.affected_ops[0]), None)
                new_idx_cons = next((i for i, op in enumerate(max_fuse_wl.operations)
                                     if op.operation_name == c.affected_ops[1]), None)
                if new_idx_prod is None or new_idx_cons is None:
                    continue
                # Rebuild candidate with current indices.
                from rewrite import Candidate as Cand
                c_now = Cand(candidate_id=c.candidate_id, type=c.type,
                             affected_ops=c.affected_ops,
                             payload={"producer_idx": new_idx_prod,
                                      "consumer_idx": new_idx_cons})
                max_fuse_wl = apply_candidate(max_fuse_wl, c_now)
                applied_count += 1
            except Exception:
                continue
    t_max, a_max, _, _ = heft(max_fuse_wl)
    m_max = compute_metrics(max_fuse_wl, t_max, a_max, scheduler_name="heft_max_fuse")
    trap_evidence = {
        "baseline": {"makespan_us": m_base["makespan_us"],
                     "dispatches": len(wl_trap.operations)},
        "max_fusion": {"makespan_us": m_max["makespan_us"],
                       "dispatches": len(max_fuse_wl.operations),
                       "candidates_applied": applied_count},
        "max_fusion_hurts_makespan": m_max["makespan_us"] > m_base["makespan_us"],
    }
    with open(out / "_fusion_trap_evidence.json", "w") as f:
        json.dump(trap_evidence, f, indent=2)
    print(f"\nfusion_trap_parallel_branches:")
    print(f"  baseline:   {m_base['makespan_us']:.1f}us / {len(wl_trap.operations)} dispatches")
    print(f"  max-fusion: {m_max['makespan_us']:.1f}us / {len(max_fuse_wl.operations)} dispatches "
          f"(applied {applied_count} fuses)")
    print(f"  max-fusion HURTS makespan: {trap_evidence['max_fusion_hurts_makespan']}")


def _candidate_from_dict(d, originals):
    for c in originals:
        if c.candidate_id == d["candidate_id"]:
            return c
    raise KeyError(d["candidate_id"])


if __name__ == "__main__":
    main()
