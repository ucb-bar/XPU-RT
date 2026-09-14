#!/usr/bin/env python3
"""B2 -- THE SEARCH FUNNEL. 108 cells to 44 usable operating points, twice.

Left  a stage funnel: what the grid costs and what survives each stage, for
      warm-started CP-SAT and for greedy on the same 108 cells.
Right the collapse itself: every solved cell as a dot, connected to the distinct
      (period, age) point it collapses onto. 27 CP-SAT cells are structural
      duplicates, and that is a fact about the SEARCH, not about the robot.

Ordinal stages take a one-hue ramp (the period blue), not categorical hues.
"""
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import vocab as V

V.style()
cells, tally, _ = V.grid_warm()
greedy = V.grid_greedy()


def stages(src):
    n = len(src)
    solved = [c for c in src.values() if "lat_med" in c]
    dist = {(round(c["cadence"], 1), round(c["lat_med"], 1)) for c in solved}
    return [("grid cells", n), ("solve", len(solved)), ("distinct points", len(dist))]


SC, SG = stages(cells), stages(greedy)

fig = plt.figure(figsize=(7.16, 2.85))
gs = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.86, bottom=0.135,
                      wspace=0.20, width_ratios=[0.95, 1.15])
axF, axS = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ------------------------------------------------------------------- funnel
BARH, GAP = 0.34, 0.30
for row, (S, lab, shade) in enumerate([(SC, "warm CP-SAT", 0.62), (SG, "greedy", 0.24)]):
    for i, (name, v) in enumerate(S):
        y = -(row * (3 * (BARH + GAP)) + i * (BARH + GAP))
        axF.add_patch(Rectangle((0, y - BARH / 2), v, BARH,
                                facecolor=V.CMAP_PERIOD(shade), linewidth=0))
        axF.text(v + 2.0, y, f"{v}", va="center", fontsize=8.4, color=V.INK,
                 fontweight="bold")
        axF.text(-3.0, y, name, va="center", ha="right", fontsize=6.9, color=V.INK2)
        if i:
            axF.text(v + 15, y, f"({100 * v / S[i - 1][1]:.0f}%)", va="center",
                     fontsize=6.2, color=V.MUTED)
    axF.text(-3.0, -(row * (3 * (BARH + GAP))) + 0.46, lab, ha="right", va="bottom",
             fontsize=7.6, color=V.INK, fontweight="bold")
axF.set_xlim(-42, 132)
axF.set_ylim(-(2 * 3 * (BARH + GAP)) + 0.15, 0.80)
axF.axis("off")
axF.set_title("the search funnel", loc="left", pad=3, x=0.0)
axF.annotate("", xy=(SC[2][1], -2 * (BARH + GAP)), xytext=(SG[2][1], -3 * (BARH + GAP) * 1.0 - 2 * (BARH + GAP)),
             arrowprops=dict(arrowstyle="-", lw=0.0))
axF.text(SG[2][1] + 26, -(3 * (BARH + GAP) + 2 * (BARH + GAP)),
         f"{SC[2][1] / SG[2][1]:.1f}x more\noperating points", fontsize=6.6,
         color=V.INK, va="center")

# ---------------------------------------------------------------- collapse
pts = collections.defaultdict(list)
for k, c in cells.items():
    if "lat_med" in c:
        pts[(round(c["cadence"], 1), round(c["lat_med"], 1))].append(c)
for (cad, age), group in pts.items():
    for c in group:
        axS.plot([c["p"], cad], [c["w"], age], color=V.GRID, lw=0.5, zorder=1)
    axS.scatter([cad], [age], s=16 + 9 * (len(group) - 1), zorder=4,
                facecolor=V.CMAP_PERIOD(0.55), edgecolor="white", linewidth=0.6)
req = axS.scatter([c["p"] for c in cells.values() if "lat_med" in c],
                  [c["w"] for c in cells.values() if "lat_med" in c],
                  s=7, marker="s", facecolor="none", edgecolor=V.MUTED, linewidth=0.5,
                  zorder=2)
axS.set_xlabel("release period (ms)")
axS.set_ylabel("deadline window / achieved age (ms)")
axS.set_title(f"{SC[1][1]} solved cells collapse onto {SC[2][1]} points",
              loc="left", pad=3, x=0.0)
axS.legend([req, plt.Line2D([], [], marker="o", ls="none", ms=4.5,
                            markerfacecolor=V.CMAP_PERIOD(0.55), mec="white")],
           ["requested (period, window)", "achieved (cadence, age)  ·  size = cells collapsed"],
           loc="upper left", fontsize=6.4)
V.tidy(axS, grid="both")
V.save(fig, "cand_b2_funnel", f"cpsat={SC} greedy={SG}")
