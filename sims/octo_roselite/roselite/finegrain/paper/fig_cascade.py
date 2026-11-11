#!/usr/bin/env python3
"""Splash component: the schedule cascade, CPU -> serial -> pipelined.

Three schedules of the SAME INT8 Octo policy on the SAME QRB5165, drawn on ONE shared
time window so the comparison is the picture rather than the caption:

  box WIDTH      = LATENCY   -- camera frame in, action out
  box PITCH      = CADENCE   -- how often a fresh action appears
  boxes per row  = how many actions the window actually gets

Pipelining is the point: its boxes OVERLAP, so cadence stops being tied to latency and
the row fills with actions the other two cannot produce.

Numbers are MEASURED on the board, ungated. Deliberately no title, caption, legend or
axis -- the row labels and the two numbers per row carry it.

LIGHT BACKGROUND, TRANSPARENT CANVAS: the page shows through, so this drops onto any
light splash layout.

Palette re-derived for a light ground and checked, not eyeballed. The dark-mode cyan/
amber were unusable here (cyan on white is ~1.2:1). #7A1250 / #C77400 / #0097B2 gives
min dE 24.9 in OKLab for normal vision and 19.4 under the worst of protan/deutan/tritan.
Bar contrast vs white is 10.35 / 3.54 / 3.46 -- above the 3.0 WCAG floor for graphical
objects -- and the ROW LABELS use darker variants (13.06 / 6.51 / 10.36) because text
carries the 4.5 floor, not the 3.0 one.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).parent / "fig_cascade.png"
WINDOW = 1400.0          # ms of wall clock shown, identical for every row

#      label              latency  cadence  bar        label ink
ROWS = [("QNN CPU",         684.8,  684.8, "#7A1250", "#5E0E3E"),
        ("XPU-RT SERIAL",   283.4,  283.4, "#C77400", "#8A5000"),
        ("XPU-RT PIPELINED", 258.7, 124.8, "#0097B2", "#00697E")]

INK, DIM = "#16161C", "#6B6B78"


def darken(hex_c, f=0.55):
    """Seam colour. On a TRANSPARENT canvas a background-coloured seam cannot work --
    there is no background to paint. A darker shade of the bar itself separates the
    boxes on any ground."""
    h = hex_c.lstrip("#")
    return "#%02X%02X%02X" % tuple(int(int(h[i:i + 2], 16) * f) for i in (0, 2, 4))
ROW_H, LANE_H, GAP = 1.0, 0.30, 0.055

fig = plt.figure(figsize=(15.0, 7.4), dpi=200)
fig.patch.set_alpha(0.0)
ax = fig.add_axes([0.225, 0.055, 0.755, 0.885])
ax.patch.set_alpha(0.0)
ax.set_xlim(-0.012 * WINDOW, WINDOW * 1.012)
ax.set_ylim(len(ROWS) * ROW_H, 0)
ax.axis("off")

halo = [pe.Stroke(linewidth=3.6, foreground="#FFFFFF", alpha=0.85), pe.Normal()]

for r, (name, lat, cad, col, ink) in enumerate(ROWS):
    y0 = r * ROW_H
    lanes = max(1, int(np.ceil(lat / cad)))          # how many are ever in flight
    band = min(LANE_H, (ROW_H - 0.34) / lanes)
    starts = np.arange(0.0, WINDOW, cad)
    done = int(np.sum(starts + lat <= WINDOW))

    for k, s in enumerate(starts):
        w = min(lat, WINDOW - s)
        yy = y0 + 0.10 + (k % lanes) * band
        ax.add_patch(Rectangle((s, yy), w, band * 0.80, facecolor=col,
                               edgecolor="none", zorder=3))
        # When latency == cadence the boxes butt together and read as ONE bar, so the
        # individual inferences cannot be counted. A background-coloured seam at each
        # start segments them WITHOUT altering any drawn width.
        # ...but only where they DO touch. Within a lane the pitch is cad*lanes, so a
        # pipelined row already has real gaps and a seam there is a meaningless mark.
        if k and lat >= cad * lanes - 1e-6:
            ax.plot([s, s], [yy, yy + band * 0.80], color=darken(col), lw=2.6, zorder=4)

    ax.text(-0.022 * WINDOW, y0 + 0.13, name, ha="right", va="top",
            fontsize=21, fontweight="bold", color=ink)
    ax.text(-0.022 * WINDOW, y0 + 0.30, f"{done} actions",
            ha="right", va="top", fontsize=15, color=DIM)

    # LATENCY -- spanned across the first box, read inside it
    ax.text(min(lat, WINDOW) / 2, y0 + 0.10 + band * 0.40, f"{lat:.0f} ms",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color="#FFFFFF", zorder=5)

    # CADENCE -- start-to-start, drawn under the row where nothing else sits
    yc = y0 + 0.10 + lanes * band + 0.085
    if len(starts) > 1:
        ax.add_patch(FancyArrowPatch((starts[0], yc), (starts[1], yc),
                                     arrowstyle="<|-|>", mutation_scale=15,
                                     linewidth=2.0, color=INK, zorder=4,
                                     shrinkA=0, shrinkB=0))
        ax.text((starts[0] + starts[1]) / 2, yc + 0.085, f"{cad:.0f} ms",
                ha="center", va="top", fontsize=17, fontweight="bold",
                color=INK, path_effects=halo)
    for s in starts[:2]:
        ax.plot([s, s], [y0 + 0.10, yc], color=INK, lw=1.0, alpha=0.45, zorder=2)

fig.savefig(OUT, dpi=200, transparent=True)
print(f"[ok] {OUT}")
for name, lat, cad, _, _ink in ROWS:
    st = np.arange(0.0, WINDOW, cad)
    print(f"  {name:18s} latency {lat:6.1f} ms   cadence {cad:6.1f} ms   "
          f"completed in {WINDOW:.0f} ms = {int(np.sum(st + lat <= WINDOW))}   "
          f"in flight = {max(1,int(np.ceil(lat/cad)))}")
