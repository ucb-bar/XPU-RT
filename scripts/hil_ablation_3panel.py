#!/usr/bin/env python3
"""HIL flight-envelope ablation, per scheme (3 panels): ROS | XPU-RT greedy | XPU-RT shard(feedback).

The flight sim's outcome depends only on cruise speed x command rate (the scheduler enters only as a
rate CAP). So this shows ONE measured success(speed, rate) grid three times, each greying the rates a
given scheme cannot reach (ROS <=81 Hz, greedy <=125 Hz, shard <=204 Hz). Cells are the real measured
grid (no interpolation): colour = gate-course success fraction, text = successes/flights.
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

SCHEMES = [("ROS",            81.0, "#e2231a"),
           ("XPU-RT greedy", 125.0, "#2f6fb0"),
           ("XPU-RT shard",  204.0, "#1f9e5a")]   # (name, sustainable-rate cap Hz, colour)
INK = "#22242a"; GREY = "#d9d6cf"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_REPO + "/results/codesign_feedback/hil_dense.csv")
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/hil_ablation_3panel")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cell = defaultdict(lambda: [0, 0])            # (speed, rate) -> [successes, n]
    for r in rows:
        sp = round(float(r["cruise_speed"]), 3); hz = round(float(r["eff_cmd_hz"]))
        cell[(sp, hz)][1] += 1
        cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell})
    rates  = sorted({h for _, h in cell})         # ascending; drawn bottom->top
    ns, nr = len(speeds), len(rates)
    Z = np.full((nr, ns), np.nan); N = np.zeros((nr, ns), int)
    for (sp, hz), (k, n) in cell.items():
        i = rates.index(hz); j = speeds.index(sp)
        Z[i, j] = k / max(1, n); N[i, j] = n

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8})
    cmap = plt.cm.RdYlGn; norm = Normalize(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True,
                             gridspec_kw=dict(left=0.075, right=0.9, top=0.80, bottom=0.16, wspace=0.10))
    for ax, (name, cap, col) in zip(axes, SCHEMES):
        # capped row index: highest rate row this scheme can reach
        reach = [i for i, hz in enumerate(rates) if hz <= cap + 1e-6]
        top = max(reach) if reach else -1
        for i in range(nr):
            for j in range(ns):
                if i > top:                       # unreachable by this scheme -> grey
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=GREY, ec="white", lw=1.0, zorder=1))
                    continue
                v = Z[i, j]
                fc = cmap(norm(v)) if not np.isnan(v) else "#ffffff"
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=fc, ec="white", lw=1.0, zorder=1))
                if not np.isnan(v):
                    ax.text(j, i, f"{int(round(v*N[i,j]))}/{N[i,j]}", ha="center", va="center",
                            fontsize=6.6, weight="bold", color="white" if v < 0.5 else "#123", zorder=3)
        # scheme rate-cap boundary
        ax.axhline(top + 0.5, color=col, lw=2.6, zorder=5)
        ax.set_xlim(-.5, ns - .5); ax.set_ylim(-.5, nr - .5)
        ax.set_xticks(range(ns)); ax.set_xticklabels([f"{s:g}×" for s in speeds], fontsize=7.5)
        ax.set_yticks(range(nr)); ax.set_yticklabels([f"{h}" for h in rates], fontsize=7.5)
        ax.tick_params(length=0)
        for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
        ax.set_xlabel("cruise speed", fontsize=8.2)
        ax.set_title(f"{name}\n≤ {cap:.0f} Hz", fontsize=8.8, weight="bold", color=col, linespacing=1.25)
    axes[0].set_ylabel("command rate (Hz)", fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02); cb.set_label("gate-course success rate", fontsize=8)
    cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.6)
    fig.suptitle("In-sim flight envelope per scheme: each flies only up to its sustained command-rate "
                 "cap (grey = unreachable); cell = measured gate-course success, gain-calibrated per rate",
                 fontsize=9.2, weight="bold", x=0.075, ha="left", y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(a.out + "." + ext, dpi=300, bbox_inches="tight")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
