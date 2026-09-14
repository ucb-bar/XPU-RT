#!/usr/bin/env python3
"""A3 -- OCCUPANCY AREA. Lane busy as a stacked area over time, small multiples.

The Gantt answers "which dispatch ran where"; this answers "how much of the SoC
was moving". Stacked area, one band per lane, 0..3 lanes busy. The pipelined
panel is the only one that ever reaches 3.

Same shared window and the same release rail as A1, so the two are readable as
two views of one trace.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
WINDOW, DT = 700.0, 1.0
t = np.arange(0, WINDOW, DT)

CFG = [("QNN CPU baseline", "mono", 684.8, 684.8),
       ("XPU-RT serial", "serial", 283.4, 283.4),
       ("XPU-RT pipelined", "p150w300", 283.3, 124.8)]

fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(1, 3, left=0.062, right=0.988, top=0.80, bottom=0.155, wspace=0.13)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

for ax, (title, group, lat, per) in zip(axes, CFG):
    _, rows, _ = V.board_median(group)
    tile = per if group != "p150w300" else None
    busy = {l: np.zeros_like(t) for l in V.LANE_ORDER}
    reps = [0.0] if tile is None else [k * tile for k in range(int(np.ceil(WINDOW / tile)) + 1)]
    for off in reps:
        for r in rows:
            if r["lane"] in busy:
                busy[r["lane"]][(t >= r["s"] + off) & (t < r["e"] + off)] = 1.0
    ax.stackplot(t, [busy[l] for l in V.LANE_ORDER],
                 colors=[V.LANE[l] for l in V.LANE_ORDER], linewidth=0)
    V.release_rail(ax, 3.30, 0, WINDOW, per, lat)
    tot = sum(busy[l].sum() for l in V.LANE_ORDER) * DT
    ax.set_title(f"{title}\nSoC busy {100 * tot / (3 * WINDOW):4.1f}% of 3 lanes",
                 loc="left", pad=3.0, fontsize=7.8)
    ax.set_ylim(0, 3.62)
    ax.set_xlim(0, WINDOW)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([0, 350, 700])
    V.tidy(ax, grid="y")
    if ax is axes[0]:
        ax.set_ylabel("lanes busy")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("ms" if ax is not axes[1] else "wall-clock time since window start (ms)")
    for l, y in zip(V.LANE_ORDER, [0.5, 1.5, 2.5]):
        if busy[l].mean() > 0.06:
            ax.text(WINDOW * 0.985, y, l, ha="right", va="center", fontsize=6.6,
                    color="white", fontweight="bold")

fig.legend(handles=V.glyph_handles(["release", "complete"]), loc="upper right",
           bbox_to_anchor=(0.99, 1.01), ncol=2)
V.save(fig, "cand_a3_occupancy_area")
