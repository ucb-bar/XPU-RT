#!/usr/bin/env python3
"""C1 -- SLOPEGRAPH. Un-scheduled baseline -> best XPU-RT schedule, per task.

Three metric columns, four task rows collapsed into one panel each. Only two
x positions exist, so the slope IS the effect; no bar, no axis clutter. The
endpoint values are the labels, which is the whole reason to use a slopegraph.

Success and energy move the SAME way (up / down), which is the co-optimisation
result; mission time is conditioned on success and is drawn hollow to say so.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
T = V.curated()
BEST = "p105w300"
METRICS = [("success", "success rate (%)", 1.0, "up"),
           ("mission", "mission time (s)\nSUCCESSES only", 1.0, "down"),
           ("energy", "actuator energy\n(N^2 m^2 s, x1000)", 1e-3, "down")]

fig = plt.figure(figsize=(7.16, 2.85))
gs = fig.add_gridspec(1, 3, left=0.052, right=0.985, top=0.790, bottom=0.115,
                      wspace=0.52)

for m, (key, lab, scale, good) in enumerate(METRICS):
    ax = fig.add_subplot(gs[m])
    for i, (task, name, robot, grid) in enumerate(V.TASKS):
        a, b = T[task][V.BASELINE][key] * scale, T[task][BEST][key] * scale
        better = (b > a) if good == "up" else (b < a)
        col = V.CMAP_PERIOD(0.68) if better else V.STATUS["critical"]
        ax.plot([0, 1], [a, b], color=col, lw=1.8, zorder=3, solid_capstyle="round")
        ax.plot([0, 1], [a, b], "o", ms=5.4, color=col, mec="white", mew=1.0, zorder=4)
        ax.text(-0.055, a, f"{a:.0f} ", ha="right", va="center", fontsize=6.8,
                color=V.INK2)
        pct = 100 * (b - a) / a if a else np.nan
        d = f"{b - a:+.0f} pt" if key == "success" else f"{pct:+.0f}%"
        ax.text(1.055, b, f" {b:.0f}  {d}", ha="left", va="center", fontsize=6.8,
                color=V.INK, fontweight="bold" if better else "normal")
        if m == 0:
            ax.text(-0.42, a, name, ha="left", va="center", fontsize=6.4, color=V.INK2)
    ax.set_xlim(-0.44, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"QNN baseline\n{T['egg'][V.BASELINE]['period']:.0f} ms period",
                        f"XPU-RT {V.CURATED_LABEL[BEST]}\n"
                        f"{T['egg'][BEST]['period']:.0f} ms period"], fontsize=6.8)
    ax.set_title(lab, loc="left", pad=4, fontsize=7.8)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(False)
    if key == "energy":
        ax.set_yscale("log")
    ax.margins(y=0.16)

fig.text(0.052, 0.975, "MEASURED  ·  10 seeds x 24 episodes per cell",
         fontsize=8.2, color=V.INK, fontweight="bold", va="top")
fig.text(0.052, 0.940, "line colour = did the schedule help  ·  task named once, on "
         "the left of the first panel  ·  rates in POINTS, ratios in %",
         fontsize=6.6, color=V.MUTED, va="top")
V.save(fig, "cand_c1_slopegraph")
