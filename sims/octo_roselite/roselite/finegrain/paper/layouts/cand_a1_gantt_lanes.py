#!/usr/bin/env python3
"""A1 -- LANE GANTT. Three MEASURED board schedules on one shared time window.

Encoding: x = wall-clock ms (shared), y = compute lane, bar = one dispatch.
Every panel carries the RELEASE RAIL at its top edge, so the release ticks and
the inference-completion triangles line up with the same events in A6 and C9.

The window is the CPU-only baseline's own measured inference span, so the
question every row answers is the same one: in the time the baseline needs for
ONE inference, how many actions did this schedule deliver?
"""
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import vocab as V

V.style()
WINDOW = 700.0

ROWS = [("QNN CPU baseline", "mono", None),
        ("XPU-RT serial", "serial", 283.4),
        ("XPU-RT pipelined  p150/w300", "p150w300", None)]

fig = plt.figure(figsize=(7.16, 4.05))
gs = fig.add_gridspec(3, 1, left=0.155, right=0.985, top=0.905, bottom=0.085,
                      hspace=0.42)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

for ax, (title, group, tile) in zip(axes, ROWS):
    name, rows, spans = V.board_median(group)
    iv = V.inferences(rows)
    starts = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    cad = float(np.median(np.diff(starts))) if len(starts) > 1 else (tile or span)

    reps = [(0.0, 1.0, False)]
    if tile:                       # a strictly serial chain: tile its own template
        reps = [(k * tile, 1.0 if k == 0 else 0.30, k > 0)
                for k in range(int(np.ceil(WINDOW / tile)))]

    for off, alpha, ghost in reps:
        for r in rows:
            lane = r["lane"]
            if lane not in V.LANE:
                continue
            y = V.LANE_ORDER.index(lane)
            s, e = r["s"] + off, r["e"] + off
            if s > WINDOW:
                continue
            ax.barh(y, min(e, WINDOW) - s, left=s, height=0.56,
                    color=V.LANE[lane], alpha=alpha, linewidth=0,
                    hatch="///" if ghost else None,
                    edgecolor="white" if ghost else "none", zorder=3)

    # the shared rail
    per = tile or cad
    lat = tile or span
    V.release_rail(ax, 2.72, 0, WINDOW, per, lat)

    done = int(np.sum(np.arange(0, WINDOW, per) + lat <= WINDOW))
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{l}  {V.LANE_MACHINE[l]}" for l in V.LANE_ORDER], fontsize=6.8)
    for t, l in zip(ax.get_yticklabels(), V.LANE_ORDER):
        t.set_color(V.LANE[l])
        t.set_fontweight("bold")
    ax.set_ylim(2.95, -0.62)
    ax.set_xlim(-6, WINDOW + 6)
    ax.set_title(f"{title}      latency {lat:.0f} ms · period {per:.0f} ms · "
                 f"{done} action{'s' if done != 1 else ''} in {WINDOW:.0f} ms",
                 loc="left", pad=3.0)
    V.tidy(ax, grid="x")
    ax.set_axisbelow(True)
    if ax is not axes[-1]:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("wall-clock time since window start (ms)  ·  MEASURED on QRB5165")

axes[1].text(WINDOW * 0.62, 2.35, "hatched = tiled template", fontsize=6.4,
             color=V.MUTED, ha="left")

fig.legend(handles=V.glyph_handles(["release", "complete"]),
           loc="upper right", bbox_to_anchor=(0.988, 1.004), ncol=2)
V.save(fig, "cand_a1_gantt_lanes")
