#!/usr/bin/env python3
"""C2 -- DUMBBELL. The same delta as C1, but as a magnitude on a common axis.

Rows are the four tasks; the dot pair is baseline and best schedule; the bar
between them is the effect with its bootstrap CI. Unlike the slopegraph this
keeps a real quantitative axis per metric, so effects can be compared BETWEEN
tasks -- which is where the actuator-saturation result lives: the two google
tasks (333 ms actuation grid) barely move, and that is visible as a short bar
rather than asserted in prose.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
T = V.curated()
BEST = "p105w300"
rng = np.random.default_rng(7)


def boot(vals, n=4000):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return np.nan, np.nan
    b = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


METRICS = [("success", "success rate (%)", "success_seeds", 1.0),
           ("mission", "mission time on successes (s)", None, 1.0),
           ("energy", "actuator energy (x1000)", "energy_seeds", 1e-3)]

fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(1, 3, left=0.163, right=0.987, top=0.760, bottom=0.162, wspace=0.16)

for m, (key, lab, seedkey, sc) in enumerate(METRICS):
    ax = fig.add_subplot(gs[m])
    for i, (task, name, robot, grid) in enumerate(V.TASKS):
        y = len(V.TASKS) - 1 - i
        a, b = T[task][V.BASELINE][key] * sc, T[task][BEST][key] * sc
        ax.plot([a, b], [y, y], color=V.GRID, lw=3.4, zorder=2, solid_capstyle="round")
        ax.plot([a, b], [y, y], color=V.CMAP_PERIOD(0.55), lw=1.6, zorder=3,
                solid_capstyle="round")
        if seedkey:
            for val, mk, col in ((a, V.BASELINE, V.STATUS["critical"]),
                                 (b, BEST, V.CMAP_PERIOD(0.80))):
                lo, hi = boot(np.array(T[task][mk][seedkey]) * sc)
                ax.plot([lo, hi], [y, y], color=col, lw=0.9, alpha=0.75, zorder=4)
        ax.plot([a], [y], "o", ms=6.6, color=V.STATUS["critical"], mec="white",
                mew=1.0, zorder=6)
        ax.plot([b], [y], "o", ms=6.6, color=V.CMAP_PERIOD(0.80), mec="white",
                mew=1.0, zorder=6)
        # a rate difference is reported in POINTS. cpu685 succeeds ~2% of the
        # time on eggplant, so a ratio there reads "+2975%" and says nothing.
        txt = f"{b - a:+.0f} pt" if key == "success" else f"{100 * (b - a) / a:+.0f}%"
        ax.text(max(a, b), y + 0.28, txt, ha="center", va="bottom",
                fontsize=6.6, color=V.INK, fontweight="bold")
    ax.set_ylim(-0.62, len(V.TASKS) - 0.28)
    ax.set_yticks(range(len(V.TASKS)))
    ax.set_yticklabels([f"{n}\n{r}, {g:.0f} ms grid" for _, n, r, g in V.TASKS][::-1],
                       fontsize=6.3)
    if m:
        ax.set_yticklabels([])
    ax.set_xlabel(lab)
    if key == "energy":
        ax.set_xscale("log")
    V.tidy(ax, grid="x")
    ax.margins(x=0.15)

fig.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=5.4, mec="white",
                               color=V.STATUS["critical"], label="QNN baseline  685 ms"),
                    plt.Line2D([], [], marker="o", ls="none", ms=5.4, mec="white",
                               color=V.CMAP_PERIOD(0.80),
                               label=f"XPU-RT {V.CURATED_LABEL[BEST]}  125 ms")],
           loc="upper left", bbox_to_anchor=(0.163, 1.010), ncol=2)
fig.text(0.163, 0.885, "10 seeds x 24 episodes  ·  thin bar = bootstrap 95% CI  ·  "
         "rates in POINTS, ratios in %", fontsize=6.6, color=V.MUTED, va="top")
V.save(fig, "cand_c2_dumbbell")
