#!/usr/bin/env python3
"""C13 -- is there a Pareto front? All 44 operating points, three objective pairs.

The question C12 raises: at the top of the curated eggplant ladder the ideal arm
finishes faster while the scheduled arm succeeds more often, which looks like the
start of a trade. Across the FULL plane -- 44 operating points, 4 tasks, 10 seeds,
42,240 episodes -- it is not one.

Each panel rings the NON-DOMINATED points. On the widowx tasks the front collapses:
one point for success-vs-energy, and two for success-vs-time separated by 0.01 s,
which is three orders of magnitude below the seed noise. A front of one is not a
front -- a single schedule is best on both axes at once, and picking it costs
nothing on the other.

The two google tasks show 4-5 points per front, but that is not evidence of a
trade either: their success surfaces are unresolved (across-arm spread is 1.30x
the per-arm SEM on coke and 0.93x on drawer), so those fronts are drawn through
noise. Read them as "no structure", not as "a richer trade-off".

CAVEAT ON METHOD: domination is computed on POINT ESTIMATES. Within the n=10
uncertainty many more points are statistically indistinguishable from the front,
so the rings mark the best guess, not a resolved set.

Colour is release period, the axis the plane shows to be dominant on widowx.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V
import sys
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
T = L.table()
plane = L.arm_plane()
# (y key, x key, y-better sign, x-better sign, y label, x label)
ROWS = [("success", "mission", +1, -1, "success (%)", "mission time (s)"),
        ("success", "energy",  +1, -1, "success (%)", "energy (N²m²s)"),
        ("mission", "energy",  -1, -1, "mission time (s)", "energy (N²m²s)")]


def front(y, x, sy, sx):
    """Indices not dominated once both axes are turned into 'higher is better'."""
    P = np.c_[np.asarray(y) * sy, np.asarray(x) * sx]
    return [i for i in range(len(P))
            if not np.any(np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1))]


fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.85), dpi=170)
fig.subplots_adjust(left=0.070, right=0.935, top=0.895, bottom=0.088,
                    wspace=0.34, hspace=0.46)
sc = None
for c, (task, tlab, emb) in enumerate(L.TASKS):
    arms = sorted(T[task])
    per = np.array([plane[a][1] for a in arms])
    val = {k: np.array([T[task][a][k] for a in arms])
           for k in ("success", "mission", "energy")}
    for r, (ky, kx, sy, sx, ylab, xlab) in enumerate(ROWS):
        ax = axes[r, c]
        Y, X = val[ky], val[kx]
        sc = ax.scatter(X, Y, c=per, cmap=V.CMAP_PERIOD, norm=V.norm_period(),
                        s=17, edgecolor="white", linewidth=0.4, zorder=4)
        f = front(Y, X, sy, sx)
        fo = sorted(f, key=lambda i: X[i])
        ax.plot(X[fo], Y[fo], color=V.INK, lw=0.9, ls="-", alpha=0.55, zorder=5)
        ax.scatter(X[f], Y[f], s=52, facecolor="none", edgecolor=V.INK,
                   linewidth=1.0, zorder=6)
        rho = spearmanr(X, Y).statistic
        ax.set_title(f"{tlab if r == 0 else ''}\nfront {len(f)}/44   ρ={rho:+.2f}",
                     fontsize=6.8, pad=2.2)
        ax.set_xlabel(xlab, fontsize=6.2, labelpad=1.2)
        if c == 0:
            ax.set_ylabel(ylab, fontsize=6.2, labelpad=2.0)
        ax.tick_params(labelsize=5.4)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        ax.xaxis.get_offset_text().set_fontsize(5.0)
        ax.margins(0.14)
        V.tidy(ax, grid="both")
        # Direction of improvement, in whichever corner the objectives point to.
        cx, cy = 0.955, (0.955 if sy > 0 else 0.075)
        ax.annotate("", xy=(cx - 0.16, cy), xytext=(cx, cy - (0.10 if sy > 0 else -0.10)),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=V.MUTED, lw=0.8))

cb = fig.colorbar(sc, ax=axes, fraction=0.013, pad=0.012)
cb.set_label("release period (ms)", fontsize=6.4)
cb.ax.tick_params(labelsize=5.6)
fig.suptitle("44 operating points × 4 tasks × 10 seeds — rings mark the non-dominated set",
             fontsize=8.4, x=0.50, y=0.990)
fig.text(0.50, 0.006, "arrows point toward better on both axes  ·  a front of 1-2 points "
                      "means one schedule wins both objectives at once",
         ha="center", va="bottom", fontsize=6.2, color=V.MUTED)
V.save(fig, "cand_c13_plane_pareto")
