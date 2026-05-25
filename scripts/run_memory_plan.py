"""
M7 driver — apply the post-schedule memory planner to a realistic Workload,
compare the three reuse policies (no_reuse, greedy_first_fit, size_aware_best_fit),
and emit a memory-timeline PNG.

Usage:
  python3 scripts/run_memory_plan.py --model dronet --soc chipyard --scheduler heft

Produces:
  results/memory/<model>_<soc>_<scheduler>_plan.json
  results/memory/<model>_<soc>_<scheduler>_timeline.png
  results/memory/_reuse_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from realistic_workloads import build_model_graph, build_workload_from_graph  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from memory_planner import plan_memory, render_memory_timeline  # noqa: E402


def _attach_annotations(workload, model_name: str, annotations: dict):
    """Merge buffer-annotations (keyed by `<model>:<symbol>`) into op.output_bytes."""
    if not annotations:
        return
    for op in workload.operations:
        sym = op.operation_name or ""
        key = f"{model_name}:{sym}"
        if key in annotations:
            op.output_bytes = int(annotations[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dronet", "mlp_wide", "yolov8n"], default="dronet")
    ap.add_argument("--soc", choices=["chipyard", "qrb5165"], default="chipyard")
    ap.add_argument("--scheduler", default="heft")
    ap.add_argument("--annotations",
                    default=str(REPO / "data" / "realistic" / "buffer_annotations.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "memory"))
    ap.add_argument("--scratchpad-bytes", type=int, default=128 * 1024,
                    help="Scratchpad capacity in bytes (default 128 KB)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        annotations = json.load(f)

    graph = build_model_graph(args.model, args.soc)
    wl = build_workload_from_graph(graph)
    _attach_annotations(wl, args.model, annotations)

    # Run the chosen scheduler.
    sched = get_scheduler(args.scheduler)
    kwargs = {}
    if args.scheduler == "mosek":
        kwargs = dict(solver_verbosity=0, time_limit=30,
                      restrict_makespan_to_nonperiodic=False,
                      prune_cross_period_constraints=False)
    elif args.scheduler == "cpsat":
        kwargs = dict(time_limit=30)
    t, alpha, _, _ = sched(wl, **kwargs)
    if t is None or alpha is None:
        print(f"[warn] {args.scheduler} returned no schedule")
        return

    print(f"=== {args.model}/{args.soc} via {args.scheduler} ===")
    print(f"  {len(wl.operations)} ops")

    region_caps = {"scratchpad": args.scratchpad_bytes}

    # Compare three policies.
    summary = {}
    for policy in ("no_reuse", "greedy_first_fit", "size_aware_best_fit"):
        plan = plan_memory(wl, t, alpha, annotations=None,
                           policy=policy, region_capacities=region_caps)
        slot_total = sum(plan["by_region_slot_total_bytes"].values())
        summary[policy] = {
            "peak_dram_bytes": plan["peak_dram_bytes"],
            "slot_total_bytes": slot_total,
            "num_slots": plan["num_slots"],
            "reuse_count": plan["reuse_count"],
            "by_region_slot_total_bytes": plan["by_region_slot_total_bytes"],
        }
        print(f"  {policy:<24s} peak_dram_inst={plan['peak_dram_bytes']:>10,}B  "
              f"slots_total={slot_total:>10,}B  "
              f"slots={plan['num_slots']:>3}  reuse={plan['reuse_count']:>3}")

    # Write full plan for the recommended policy.
    plan = plan_memory(wl, t, alpha, annotations=None,
                       policy="greedy_first_fit", region_capacities=region_caps)
    plan_path = out_dir / f"{args.model}_{args.soc}_{args.scheduler}_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nPlan -> {plan_path}")

    # Timeline plot.
    timeline_path = out_dir / f"{args.model}_{args.soc}_{args.scheduler}_timeline.png"
    render_memory_timeline(wl, t, alpha, str(timeline_path),
                           region_capacities=region_caps,
                           title=f"Memory timeline: {args.model}/{args.soc} via {args.scheduler}")
    print(f"Timeline -> {timeline_path}")

    # Reuse summary.
    reuse_summary_path = out_dir / "_reuse_summary.json"
    existing = {}
    if reuse_summary_path.exists():
        with open(reuse_summary_path) as f:
            existing = json.load(f)
    key = f"{args.model}_{args.soc}_{args.scheduler}"
    no_reuse_total = summary["no_reuse"]["slot_total_bytes"]
    first_fit_total = summary["greedy_first_fit"]["slot_total_bytes"]
    existing[key] = {
        "policies": summary,
        "slot_total_reduction_first_fit_vs_no_reuse_pct": (
            (no_reuse_total - first_fit_total) / no_reuse_total * 100
            if no_reuse_total > 0 else 0
        ),
    }
    with open(reuse_summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Reuse summary -> {reuse_summary_path}")


if __name__ == "__main__":
    main()
