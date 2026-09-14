#!/usr/bin/env python3
"""C7 -- CO-OPTIMISED, NOT TRADED. Success against energy, connected by period.

A Pareto plot of success (up is better) against actuator energy (left is better)
would normally show a trade-off frontier sloping the wrong way. It does not:
on the widowx tasks the 44 operating points fall on a line from bottom-right to
top-left, so the schedule that succeeds more also spends less.

The connector is drawn in PERIOD order, which is what makes it a curve rather
than a cloud -- and is the same ordering C8 uses.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V

V.style()
T, PL = V.plane()
arms = sorted(T["egg"])
per = np.array([PL[a][1] for a in arms])
order = np.argsort(per)

fig = plt.figure(figsize=(7.16, 2.30))
gs = fig.add_gridspec(1, 4, left=0.062, right=0.955, top=0.775, bottom=0.185,
                      wspace=0.24)

for c, (task, name, robot, grid) in enumerate(V.TASKS):
    ax = fig.add_subplot(gs[c])
    e = np.array([T[task][a]["energy"] for a in arms]) / 1000.0
    s = np.array([T[task][a]["success"] for a in arms])
    rho = spearmanr(s, e).statistic
    ax.plot(e[order], s[order], color=V.GRID, lw=1.0, zorder=2)
    ax.scatter(e, s, s=24, c=[V.c_period(p) for p in per], edgecolor="white",
               linewidth=0.5, zorder=4)
    k = int(np.argmin(per)); j = int(np.argmax(per))
    for i, txt, dy in ((k, "fastest\ncadence", 8), (j, "slowest\ncadence", -16)):
        ax.annotate(txt, xy=(e[i], s[i]), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=6.0, color=V.INK2)
    ax.text(0.03, 0.94, f"rho(success, energy) {rho:+.2f}", transform=ax.transAxes,
            fontsize=6.9, va="top",
            color=V.INK if abs(rho) > 0.5 else V.MUTED,
            fontweight="bold" if abs(rho) > 0.5 else "normal")
    ax.set_title(f"{name}\n{robot} · {grid:.0f} ms actuation", loc="left", pad=3,
                 fontsize=7.6)
    ax.set_xlabel("actuator energy (x1000)", fontsize=7.0)
    if c == 0:
        ax.set_ylabel("success rate (%)", fontsize=7.2)
    V.tidy(ax, grid="both")
    ax.margins(0.16)

sm = plt.cm.ScalarMappable(norm=V.norm_period(per.min(), per.max()), cmap=V.CMAP_PERIOD)
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.011, pad=0.006)
cb.set_label("release period (ms)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0); cb.outline.set_visible(False)
fig.text(0.062, 0.945, "MEASURED  ·  44 operating points  ·  up and LEFT is better "
         "on both axes", fontsize=8.2, color=V.INK, fontweight="bold")
V.save(fig, "cand_c7_cooptimised")
