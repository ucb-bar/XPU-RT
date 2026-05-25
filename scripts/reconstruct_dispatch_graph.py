"""
Reconstruct per-model dispatch graphs for Chipyard and QRB5165.

Reads:
  /scratch2/agustin/merlin/tmp/dispatch_profile.csv  (chipyard per-dispatch)
  /scratch2/agustin/merlin/tmp/e2e_profile.csv       (chipyard end-to-end)
  qnn_scheduler/qrb5165_costs.json                   (qrb5165 per-op)

Writes:
  data/realistic/<model>_<soc>.json
  data/realistic/_sanity_e2e_vs_critical_path.json
  plots/m3_<model>_<soc>_<scheduler>.png

The graph format is documented in xpu-rt/realistic_workloads.py.
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

from realistic_workloads import (
    CHIPYARD_BACKENDS,
    QRB5165_BACKENDS,
    VALID_MODELS,
    build_model_graph,
    build_workload_from_graph,
    critical_path_sum_us,
    e2e_envelope,
    load_cost_table,
)
from schedulers import get_scheduler
from metrics import compute_metrics
from report import SchedulerResult, render_gantt
from postprocessing import validate_schedule


def _write_graph(graph, out_dir: Path) -> Path:
    p = out_dir / f"{graph['model']}_{graph['soc']}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(graph, f, indent=2)
    return p


def _sanity_check(graph, e2e_us):
    """For chipyard graphs, compare per-backend critical-path sum vs e2e total.

    The reconstruction is a chain, so critical path = sum of all dispatch
    latencies on the chosen backend. Sanity passes when this is within ±25 %
    of the measured e2e total in dispatch_profile.csv.
    """
    out = {}
    for backend in CHIPYARD_BACKENDS:
        cp = critical_path_sum_us(graph, backend)
        e2e = e2e_us.get(backend)
        if e2e is None or cp <= 0:
            out[backend] = {"critical_path_us": cp, "e2e_us": e2e, "ratio": None}
            continue
        ratio = cp / e2e
        out[backend] = {
            "critical_path_us": cp,
            "e2e_us": e2e,
            "ratio": ratio,
            "within_25pct": 0.75 <= ratio <= 1.25,
        }
    return out


def _render_single_model_gantts(model: str, soc: str, graph, out_plots: Path):
    """Render a single-model Gantt for each of {mosek, heft} on the reconstructed
    graph. Used as the visual verification artifact."""
    wl = build_workload_from_graph(graph)
    print(f"  built Workload: {len(wl.operations)} ops, "
          f"{len(wl.machine_combinations)} backends ({wl.machines})")

    out_plots.mkdir(parents=True, exist_ok=True)

    for scheduler in ("heft", "mosek"):
        sched = get_scheduler(scheduler)
        kwargs = {}
        if scheduler == "mosek":
            # MOSEK on yolov8n (273 ops) within 15 s is infeasible; cap at 60s
            # and accept a feasible but possibly non-optimal solution.
            kwargs = dict(
                solver_verbosity=0, time_limit=60,
                restrict_makespan_to_nonperiodic=False,
                prune_cross_period_constraints=False,
            )
        try:
            t, alpha, _, _ = sched(wl, **kwargs)
        except Exception as exc:
            print(f"    {scheduler}: failed ({exc})")
            continue
        if t is None or alpha is None:
            print(f"    {scheduler}: solver returned no schedule (infeasible)")
            continue
        m = compute_metrics(wl, t, alpha, scheduler_name=scheduler)
        ok, _ = validate_schedule(wl, t, alpha, original_json_data={"dispatches": {}})
        print(f"    {scheduler}: makespan={m['makespan_us']:.0f}us  valid={ok}")
        gantt_path = out_plots / f"m3_{model}_{soc}_{scheduler}.png"
        res = SchedulerResult(
            scheduler_name=scheduler, workload=wl, t=t, alpha=alpha, metrics=m, feasible=True,
        )
        render_gantt(res, str(gantt_path),
                     title=f"{model} on {soc} via {scheduler} (makespan={m['makespan_us']:.0f}us)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(VALID_MODELS) + ["all"], default="all")
    parser.add_argument("--soc", choices=["chipyard", "qrb5165", "both"], default="both")
    parser.add_argument("--out", default=str(REPO / "data" / "realistic"))
    parser.add_argument("--plots", default=str(REPO / "plots"))
    parser.add_argument("--gantts", action="store_true",
                        help="Also render single-model Gantts (heft + mosek)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    plots_dir = Path(args.plots)

    models = list(VALID_MODELS) if args.model == "all" else [args.model]
    socs = ["chipyard", "qrb5165"] if args.soc == "both" else [args.soc]

    cost_chipyard = load_cost_table("chipyard")
    cost_qrb5165 = load_cost_table("qrb5165")

    sanity_report = {}
    for model in models:
        for soc in socs:
            print(f"\n=== {model} / {soc} ===")
            cost = {**cost_chipyard, **cost_qrb5165}
            graph = build_model_graph(model, soc, cost_table=cost)
            p = _write_graph(graph, out_dir)
            print(f"  graph -> {p.relative_to(REPO)}  "
                  f"({graph['meta']['node_count']} nodes, "
                  f"{graph['meta']['edge_count']} edges, "
                  f"{graph['meta'].get('unmapped_count', 0)} unmapped)")

            if soc == "chipyard":
                e2e_us = {b: cost_chipyard["e2e"].get((model, b))
                          for b in CHIPYARD_BACKENDS}
                sanity_report.setdefault(model, {})[soc] = _sanity_check(graph, e2e_us)

            if args.gantts:
                _render_single_model_gantts(model, soc, graph, plots_dir)

    # Write sanity report.
    sanity_path = out_dir / "_sanity_e2e_vs_critical_path.json"
    with open(sanity_path, "w") as f:
        json.dump(sanity_report, f, indent=2)
    print(f"\nSanity report -> {sanity_path.relative_to(REPO)}")
    for model, by_soc in sanity_report.items():
        for soc, by_backend in by_soc.items():
            for backend, st in by_backend.items():
                if st.get("ratio") is None:
                    continue
                tag = "OK" if st.get("within_25pct") else "OUT-OF-BAND"
                print(f"  {model:<10s} {soc:<9s} {backend:<8s} "
                      f"cp={st['critical_path_us']:>12,.1f}  "
                      f"e2e={st['e2e_us']:>12,.1f}  ratio={st['ratio']:.3f}  {tag}")


if __name__ == "__main__":
    main()
