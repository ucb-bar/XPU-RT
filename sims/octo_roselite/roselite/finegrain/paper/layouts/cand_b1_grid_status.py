#!/usr/bin/env python3
"""B1 -- THE SEARCH GRID. What the solver returned in each of the 108 cells.

Two panels on the same axes: warm-started CP-SAT, and greedy for reference.
Colour = achieved observation age (sequential blue). Absent data is NEVER a zero
on the ramp: INFEASIBLE is a 45-degree hatch, UNKNOWN a 135-degree hatch, both on
the neutral absent grey, both carrying their glyph.

Cells that became one of the 44 distinct operating points are ringed.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import vocab as V

V.style()
cells, tally, meta = V.grid_warm()
greedy = V.grid_greedy()
P = sorted({c["p"] for c in cells.values()})
W = sorted({c["w"] for c in cells.values()})

ages = [c["lat_med"] for c in cells.values() if "lat_med" in c]
vmin, vmax = min(ages), max(ages)
norm = plt.Normalize(vmin, vmax)

# which (cadence, age) pairs are distinct -> the 44
seen, distinct = set(), set()
for k, c in cells.items():
    if "lat_med" not in c:
        continue
    key = (round(c["cadence"], 1), round(c["lat_med"], 1))
    if key not in seen:
        seen.add(key)
        distinct.add(k)

fig = plt.figure(figsize=(7.16, 3.05))
gs = fig.add_gridspec(1, 3, left=0.062, right=0.905, top=0.828, bottom=0.145,
                      wspace=0.10, width_ratios=[1, 1, 0.02])
axC, axG = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

for ax, src, title in [(axC, cells, "warm-started CP-SAT"),
                       (axG, greedy, "greedy (HEFT/EDF)")]:
    for i, p in enumerate(P):
        for j, w in enumerate(W):
            k = f"{p}_{w}"
            c = src.get(k)
            st = (c or {}).get("status", "OPTIMAL" if c and "lat_med" in c else None)
            if c and "lat_med" in c:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor=V.CMAP_PERIOD(
                    0.12 + 0.80 * norm(c["lat_med"])), linewidth=0))
                ax.text(i + .5, j + .5, f"{c['lat_med']:.0f}", ha="center", va="center",
                        fontsize=4.9, color="white" if norm(c["lat_med"]) > 0.45 else V.INK)
                if src is cells and k in distinct:
                    ax.add_patch(Rectangle((i + .06, j + .06), .88, .88, facecolor="none",
                                           edgecolor=V.INK, lw=0.7, zorder=4))
            else:
                h = V.HATCH_INFEASIBLE if st == "INFEASIBLE" else V.HATCH_UNKNOWN
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor=V.ABSENT, linewidth=0,
                                       hatch=h, edgecolor="white"))
                ax.text(i + .5, j + .5, "x" if st == "INFEASIBLE" else "?",
                        ha="center", va="center", fontsize=5.6, color=V.INK2)
    ax.set_xlim(0, len(P)); ax.set_ylim(0, len(W))
    ax.set_xticks(np.arange(len(P)) + .5); ax.set_xticklabels(P, fontsize=6.0)
    ax.set_yticks(np.arange(len(W)) + .5); ax.set_yticklabels(W, fontsize=6.0)
    ax.set_xlabel("release period requested (ms)")
    n = sum(1 for c in src.values() if "lat_med" in c)
    nd = len({(round(c["cadence"], 1), round(c["lat_med"], 1))
              for c in src.values() if "lat_med" in c})
    ax.set_title(f"{title}\n{n} of 108 solve  ->  {nd} distinct operating points",
                 loc="left", pad=3, fontsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

axC.set_ylabel("deadline window duration (ms)")
axG.set_yticklabels([])

cax = fig.add_subplot(gs[2])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=V.CMAP_PERIOD), cax=cax)
cb.set_label("observation age (ms)\nPREDICTED", fontsize=6.4, labelpad=2)
cb.ax.tick_params(labelsize=6.0)
cb.outline.set_visible(False)

fig.legend(handles=[
    Rectangle((0, 0), 1, 1, facecolor=V.ABSENT, hatch=V.HATCH_INFEASIBLE,
              edgecolor="white", label="x  INFEASIBLE (proved)"),
    Rectangle((0, 0), 1, 1, facecolor=V.ABSENT, hatch=V.HATCH_UNKNOWN,
              edgecolor="white", label="?  UNKNOWN (budget)"),
    Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=V.INK, lw=0.7,
              label="one of the 44 distinct")],
    loc="upper right", bbox_to_anchor=(0.912, 1.010), ncol=3, handlelength=1.3)
V.save(fig, "cand_b1_grid_status", f"tally={tally}")
