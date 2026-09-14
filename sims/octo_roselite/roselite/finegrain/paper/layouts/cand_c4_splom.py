#!/usr/bin/env python3
"""C4 -- SCATTERPLOT MATRIX over the 44 operating points.

Six variables: the two the scheduler hands the simulator (period, age), one it
implies (makespan), and the three the robot returns (success, mission time,
energy). Lower triangle = scatter, diagonal = the variable's own marginal, upper
triangle = Spearman rho as a filled cell on a DIVERGING scale, because a
correlation has a sign and a meaningful zero.

This is the diagnostic that decides which pairwise plots the paper should carry:
the strong cells are (period, success), (success, energy) and (period, energy),
and the (age, *) column is weak and wrong-signed.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
import vocab as V

V.style()
TASK = "egg"
T, PL = V.plane()
A = V.arms()
arms = sorted(T[TASK])
COLS = [("period", np.array([PL[a][1] for a in arms])),
        ("age", np.array([PL[a][0] for a in arms])),
        ("makespan", np.array([A[a][2] for a in arms])),
        ("success", np.array([T[TASK][a]["success"] for a in arms])),
        ("mission s", np.array([T[TASK][a]["mission"] for a in arms])),
        ("energy", np.array([T[TASK][a]["energy"] for a in arms]) / 1000.0)]
N = len(COLS)
per = COLS[0][1]
# diverging blue<->red with a neutral grey midpoint (documented pair)
DIV = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "rho", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#ef9a9a", "#d03b3b", "#8c1f1f"])

fig = plt.figure(figsize=(7.16, 4.55))
gs = fig.add_gridspec(N, N, left=0.072, right=0.905, top=0.865, bottom=0.075,
                      wspace=0.12, hspace=0.12)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

for i in range(N):
    for j in range(N):
        ax = fig.add_subplot(gs[i, j])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(V.GRID)
        xi, yi = COLS[j][1], COLS[i][1]
        ok = np.isfinite(xi) & np.isfinite(yi)
        if i == j:
            ax.hist(xi[np.isfinite(xi)], bins=12, color=V.CMAP_PERIOD(0.35),
                    linewidth=0)
            ax.text(0.5, 0.90, COLS[i][0], transform=ax.transAxes, ha="center",
                    va="top", fontsize=7.2, color=V.INK, fontweight="bold")
            ax.text(0.5, 0.66, f"{np.nanmin(xi):.0f}-{np.nanmax(xi):.0f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
                    color=V.INK2)
            ax.set_ylim(0, ax.get_ylim()[1] * 2.6)
            ax.set_facecolor("#f6f5f1")
        elif i > j:
            ax.scatter(xi[ok], yi[ok], s=7, c=[V.c_period(p) for p in per[ok]],
                       edgecolor="white", linewidth=0.25, zorder=3)
        else:
            r = spearmanr(xi[ok], yi[ok]).statistic
            ax.set_facecolor(DIV(norm(r)))
            ax.text(0.5, 0.5, f"{r:+.2f}", transform=ax.transAxes, ha="center",
                    va="center", fontsize=7.6,
                    color="white" if abs(r) > 0.55 else V.INK,
                    fontweight="bold" if abs(r) > 0.5 else "normal")
        if i == N - 1:
            ax.set_xlabel(COLS[j][0], fontsize=6.4, labelpad=1.5)
        if j == 0:
            ax.set_ylabel(COLS[i][0], fontsize=6.4, labelpad=1.5)

cax = fig.add_axes([0.918, 0.075, 0.016, 0.560])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
cb.set_label("Spearman rho", fontsize=6.4)
cb.ax.tick_params(labelsize=6.0); cb.outline.set_visible(False)
sm = plt.cm.ScalarMappable(norm=V.norm_period(per.min(), per.max()), cmap=V.CMAP_PERIOD)
cax2 = fig.add_axes([0.918, 0.665, 0.016, 0.200])
cb2 = fig.colorbar(sm, cax=cax2)
cb2.set_label("period (ms)", fontsize=6.4)
cb2.set_ticks([110, 150, 200, 283])
cb2.ax.set_yticklabels(["110", "150", "200", "283"])
cb2.ax.tick_params(labelsize=6.0, which="both"); cb2.outline.set_visible(False)
cb2.ax.minorticks_off()

fig.text(0.072, 0.978, f"MEASURED  ·  eggplant in basket  ·  {len(arms)} operating "
         f"points", fontsize=8.2, color=V.INK, fontweight="bold", va="top")
fig.text(0.072, 0.945, "lower = scatter, upper = Spearman rho, diagonal = marginal + "
         "range  ·  rho is over OPERATING POINTS, not episodes",
         fontsize=6.6, color=V.MUTED, va="top")
V.save(fig, "cand_c4_splom")
