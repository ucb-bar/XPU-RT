#!/usr/bin/env python3
"""B3 -- THE FRONTIER. The 44 operating points are a Pareto front, not a grid.

Left   achieved cadence against achieved observation age, one dot per operating
       point, with the non-dominated frontier connected. Spearman(age, period)
       is computed at draw time and printed as a two-word annotation.
Middle greedy on the same axes -- one period, repeated down every window.
Right  the consequence, stated once: a metric plotted against LATENCY alone runs
       backwards, because the two axes are anti-correlated by construction.

This is the figure that stops a reader reading the E2E panels backwards.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V

V.style()
A = V.arms()
lat = np.array([v[0] for v in A.values()])
per = np.array([v[1] for v in A.values()])
mk = np.array([v[2] for v in A.values()])
rho = spearmanr(lat, per).statistic

greedy = V.grid_greedy()
gl = np.array([c["lat_med"] for c in greedy.values()])
gp = np.array([c["cadence"] for c in greedy.values()])
gdist = sorted({(round(a, 1), round(b, 1)) for a, b in zip(gp, gl)})


def frontier(x, y):
    """non-dominated with BOTH minimised"""
    o = np.argsort(x)
    out, best = [], np.inf
    for i in o:
        if y[i] < best - 1e-9:
            out.append(i); best = y[i]
    return np.array(out)


fig = plt.figure(figsize=(7.16, 2.60))
gs = fig.add_gridspec(1, 3, left=0.062, right=0.988, top=0.845, bottom=0.155,
                      wspace=0.24)
ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

f = frontier(per, lat)
ax1.plot(per[f], lat[f], color=V.CMAP_PERIOD(0.72), lw=1.6, zorder=3,
         solid_capstyle="round")
ax1.scatter(per, lat, s=26, c=[V.c_period(p) for p in per], edgecolor="white",
            linewidth=0.7, zorder=4)
ax1.scatter(per[f], lat[f], s=52, facecolor="none", edgecolor=V.INK, lw=0.8, zorder=5)
ax1.set_title(f"44 operating points\nSpearman(age, period) = {rho:+.2f}",
              loc="left", pad=3, fontsize=7.8)
ax1.annotate("frontier", xy=(per[f][len(f) // 2], lat[f][len(f) // 2]),
             xytext=(12, -18), textcoords="offset points", fontsize=6.6, color=V.INK,
             arrowprops=dict(arrowstyle="-|>", lw=0.7, color=V.INK, mutation_scale=6))

ax2.scatter(gp, gl, s=26, c=[V.c_period(p) for p in gp], edgecolor="white",
            linewidth=0.7, zorder=4)
ax2.set_title(f"greedy: {len(gdist)} distinct\none period per window row",
              loc="left", pad=3, fontsize=7.8)

for ax in (ax1, ax2):
    ax.set_xlabel("achieved release period (ms)")
    ax.set_xlim(95, 300)
    ax.set_ylim(215, 415)
    V.tidy(ax, grid="both")
ax1.set_ylabel("achieved observation age (ms)")
ax2.set_yticklabels([])

# --------------------------------------------- panel 3: why latency misleads
order = np.argsort(lat)
ax3.plot(lat[order], per[order], color=V.MUTED, lw=0.8, zorder=2, alpha=0.7)
ax3.scatter(lat, per, s=26, c=[V.c_period(p) for p in per], edgecolor="white",
            linewidth=0.7, zorder=4)
z = np.polyfit(lat, per, 1)
xs = np.linspace(lat.min(), lat.max(), 20)
ax3.plot(xs, np.polyval(z, xs), color=V.STATUS["critical"], lw=1.3, ls=(0, (4, 2)),
         zorder=5)
ax3.set_xlabel("observation age (ms)")
ax3.set_ylabel("release period (ms)")
ax3.set_title("lower age BUYS a slower period\nso 'metric vs age' reads backwards",
              loc="left", pad=3, fontsize=7.8)
V.tidy(ax3, grid="both")

V.save(fig, "cand_b3_frontier", f"rho={rho:+.3f} frontier={len(f)} greedy_distinct={len(gdist)}")
