#!/usr/bin/env python3
"""C6 -- BUMP CHART. Does one schedule win everywhere, or does the task decide?

Rank of each of the 44 operating points on each of the four tasks, connected.
A flat line means the schedule's advantage is task-independent; a crossing means
it is not. The two widowx tasks agree; the two google tasks scramble the order,
which is the actuator-saturation result seen as RANK rather than as a flat slope.

Only the top and bottom five are inked; the middle 34 are drawn as a grey band,
because 44 labelled lines is a hairball and the middle carries no claim.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V

V.style()
T, PL = V.plane()
arms = sorted(T["egg"])
per = {a: PL[a][1] for a in arms}
R = {}
for task, _, _, _ in V.TASKS:
    s = np.array([T[task][a]["success"] for a in arms])
    R[task] = {a: int(r) + 1 for a, r in zip(arms, np.argsort(np.argsort(-s)))}

TOPN = 5
hi = sorted(arms, key=lambda a: R["egg"][a])[:TOPN]
lo = sorted(arms, key=lambda a: -R["egg"][a])[:TOPN]
inked = set(hi) | set(lo)

fig = plt.figure(figsize=(7.16, 3.05))
ax = fig.add_axes([0.145, 0.115, 0.72, 0.735])
X = range(len(V.TASKS))
for a in arms:
    y = [R[t][a] for t, _, _, _ in V.TASKS]
    if a in inked:
        ax.plot(X, y, color=V.c_period(per[a]), lw=1.7, zorder=4,
                solid_capstyle="round")
        ax.plot(X, y, "o", ms=4.4, color=V.c_period(per[a]), mec="white", mew=0.8,
                zorder=5)
        ax.text(-0.09, y[0], f"{a}  {per[a]:.0f} ms ", ha="right", va="center",
                fontsize=6.2, color=V.INK2)
        ax.text(len(V.TASKS) - 0.91, y[-1], f" {a}", ha="left", va="center",
                fontsize=6.2, color=V.INK2)
    else:
        ax.plot(X, y, color=V.GRID, lw=0.8, zorder=2)
ax.set_xticks(list(X))
ax.set_xticklabels([f"{n}\n{r} · {g:.0f} ms" for _, n, r, g in V.TASKS], fontsize=6.9)
ax.set_xlim(-0.85, len(V.TASKS) - 0.15)
ax.set_ylim(len(arms) + 1, 0)
ax.set_yticks([1, 10, 20, 30, 44])
ax.set_ylabel("rank by success rate  (1 = best)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="y", color=V.GRID, lw=0.6)
ax.set_axisbelow(True)

rr = spearmanr([R["egg"][a] for a in arms], [R["spoon"][a] for a in arms]).statistic
rg = spearmanr([R["coke"][a] for a in arms], [R["drawer"][a] for a in arms]).statistic
rx = spearmanr([R["egg"][a] for a in arms], [R["coke"][a] for a in arms]).statistic
ax.text(0.5, -0.5, f"rank agreement  widowx pair {rr:+.2f}   ·   google pair {rg:+.2f}"
        f"   ·   across families {rx:+.2f}", fontsize=6.9, color=V.INK2, ha="center")

sm = plt.cm.ScalarMappable(norm=V.norm_period(min(per.values()), max(per.values())),
                           cmap=V.CMAP_PERIOD)
cb = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.02)
cb.set_label("release period (ms)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0); cb.outline.set_visible(False)
fig.text(0.145, 0.935, f"MEASURED  ·  {len(arms)} operating points ranked on each task  "
         f"·  top {TOPN} and bottom {TOPN} inked", fontsize=8.0, color=V.INK,
         fontweight="bold")
V.save(fig, "cand_c6_bump", f"rho ww={rr:+.2f} gg={rg:+.2f} wg={rx:+.2f}")
