#!/usr/bin/env python3
"""The inner/outer ablation, and the solver comparison, in one figure.

WHAT IT SHOWS. Four cells -- neither / inner only / outer only / both -- with one bar
per solver arm in each. Two things a reader wants are then legible at once:

  * which LOOP does the work (read across the cells), and
  * whether the exact solver earns its cost (read within a cell).

Panel (a) is instance deadline misses, the runbook's declared hero metric. Panel (b) is
worst lateness, which is where a lever that clears no miss still shows -- on some
workloads the accepted lever wins on critical-task p99 while the makespan and the miss
count do not move at all, and a misses-only figure would call that nothing.

WHAT THE BARS MEAN. Every cell is scored on BOARD costs, because that is what silicon
does; the cells differ in what the scheduler knew (cells A/B are solved on predicted
costs and re-cost with their assignment fixed; C/D are solved against the measured
multipliers). Only the at-stake stratum is plotted by default -- workloads whose naive
deployment already met every deadline cannot show a loop helping, and averaging them in
dilutes the effect. Their count is stated on the figure rather than hidden.

Reads `ablation_summary.json` only; it never re-solves. Writes png + pdf via figstyle,
plus a `_metrics.json` sidecar in compose_schedule_evolution's shape so
`scripts/emit_figure_numbers.py` can generate the caption's numbers.

Usage:
  scripts/plot_loop_ablation.py --summary results/loop_ablation/ablation_summary.json \\
      --out-dir results/codesign_feedback --stem loop_ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import figstyle  # noqa: E402

CELL_ORDER = ("A", "B", "C", "D")
CELL_SUB = {"A": "neither", "B": "inner\nonly", "C": "outer\nonly", "D": "both"}

# Greedy is the reference, not the subject: muted, so the exact arm reads as the thing
# being argued for. Any further arms take Okabe-Ito colours in order.
ARM_COLOR = {"greedy": figstyle.C_MUTED, "cpsat": figstyle.BLUE,
             "mosek": figstyle.ORANGE}


def arm_color(name, i):
    return ARM_COLOR.get(name, figstyle.OKABE_ITO[i % len(figstyle.OKABE_ITO)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="loop_ablation")
    ap.add_argument("--all-workloads", action="store_true",
                    help="plot every workload, not only the at-stake stratum. The "
                         "diluted view; the default is the pre-registered stratum.")
    a = ap.parse_args()

    d = json.load(open(a.summary))
    solvers = d.get("solvers") or []
    at_stake = set(d.get("population", {}).get("at_stake") or [])
    agg_key = "aggregate" if a.all_workloads else "aggregate_at_stake_only"
    agg = d.get(agg_key) or d.get("aggregate") or {}
    if not agg:
        print(f"{a.summary}: no aggregate to plot", file=sys.stderr)
        return 2
    n_shown = (agg.get(solvers[0], {}).get("A", {}) or {}).get("n", 0) if solvers else 0
    n_total = d.get("n_workloads", 0)

    figstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE_COL, 62 * figstyle.MM))
    width = 0.8 / max(len(solvers), 1)

    panels = []
    for pi, (ax, key, ylabel) in enumerate((
            (axes[0], "total_instance_misses", "instance deadline misses"),
            (axes[1], "median_worst_lateness_ms", "worst lateness (ms)"))):
        for si, solver in enumerate(solvers):
            xs, ys = [], []
            for ci, cell in enumerate(CELL_ORDER):
                g = (agg.get(solver) or {}).get(cell) or {}
                v = g.get(key)
                xs.append(ci + (si - (len(solvers) - 1) / 2) * width)
                ys.append(float(v) if isinstance(v, (int, float)) else 0.0)
            ax.bar(xs, ys, width=width * 0.92, color=arm_color(solver, si),
                   linewidth=0, label=solver)
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y, f"{y:g}" if key.startswith("total") else f"{y:.1f}",
                            ha="center", va="bottom", fontsize=4.4)
        ax.set_xticks(range(len(CELL_ORDER)))
        ax.set_xticklabels([f"{c}\n{CELL_SUB[c]}" for c in CELL_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        figstyle.despine(ax)
        figstyle.panel_label(ax, "ab"[pi])
        panels.append({"panel": "ab"[pi], "metric": key,
                       "values": {s: {c: (agg.get(s) or {}).get(c, {}).get(key)
                                      for c in CELL_ORDER} for s in solvers}})

    axes[0].legend(frameon=False, fontsize=5, loc="upper right",
                   title="solver", title_fontsize=5)
    # Say what the cells mean on the figure, so it survives being read without a caption.
    fig.text(0.5, 0.005,
             "inner = AOT co-design with ModelBlaster (isolated profiles)   ·   "
             "outer = HIL, measured board costs re-solved   ·   "
             "every cell scored on board costs",
             ha="center", fontsize=4.8, color="0.35")
    sub = (f"at-stake workloads only (n={n_shown} of {n_total}; the rest meet every "
           f"deadline already)" if not a.all_workloads
           else f"all workloads (n={n_shown})")
    fig.suptitle(f"Which feedback loop does the work — {sub}",
                 fontsize=7, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_dir = a.out_dir or figstyle.FIGURE_DIR
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    png = figstyle.save(fig, a.stem, out_dir)

    # Sidecar in compose_schedule_evolution's shape: record what was DRAWN.
    side = {"figure": os.path.basename(png),
            "figure_size_in": [round(v, 3) for v in fig.get_size_inches()],
            "source_summary": os.path.relpath(a.summary, REPO)
            if a.summary.startswith(REPO) else a.summary,
            "stratum": "at_stake_only" if not a.all_workloads else "all",
            "n_workloads_shown": n_shown, "n_workloads_total": n_total,
            "at_stake": sorted(at_stake),
            "solvers": solvers,
            "cells": d.get("cells"),
            "exact_vs_greedy_per_cell": d.get("exact_vs_greedy_per_cell"),
            "panels": panels}
    with open(os.path.join(out_dir, f"{a.stem}_metrics.json"), "w") as f:
        json.dump(side, f, indent=2)
    print("wrote", png)
    print("wrote", os.path.join(out_dir, f"{a.stem}_metrics.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
