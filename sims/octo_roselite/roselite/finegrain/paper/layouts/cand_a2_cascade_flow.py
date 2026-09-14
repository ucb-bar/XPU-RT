#!/usr/bin/env python3
"""A2 -- CASCADE FLOW. One inference as a ribbon through the SoC, then tiled.

Panel A: the 71 dispatches of ONE inference collapsed to their 8 named segments,
drawn as a left-to-right ribbon whose x-extent is measured time and whose colour
is the lane that ran it. This is the "cascade" as a reader pictures it -- a
single observation walking CPU -> DSP -> HTA -> CPU.

Panel B: the same template released every 125 ms. The ribbon is repeated with a
vertical offset per in-flight inference, so overlap is a shape, not a claim.
"""
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import vocab as V

V.style()
name, rows, _ = V.board_median("p150w300")
inst0 = min(r["inst"] for r in rows)
one = sorted([r for r in rows if r["inst"] == inst0], key=lambda r: r["s"])
t0 = min(r["s"] for r in one)
for r in one:
    r["s"] -= t0
    r["e"] -= t0
SPAN = max(r["e"] for r in one)

# collapse to contiguous runs of the same (segment, lane)
runs = []
for r in one:
    if runs and runs[-1][0] == r["seg"] and runs[-1][1] == r["lane"] and r["s"] - runs[-1][3] < 1.2:
        runs[-1][3] = max(runs[-1][3], r["e"])
        runs[-1][4] += 1
    else:
        runs.append([r["seg"], r["lane"], r["s"], r["e"], 1])

fig = plt.figure(figsize=(7.16, 3.35))
gs = fig.add_gridspec(2, 1, left=0.075, right=0.988, top=0.885, bottom=0.105,
                      hspace=0.55, height_ratios=[1.0, 1.15])
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ------------------------------------------------------------------ panel A
LANE_Y = {"CPU": 0.62, "DSP": 0.30, "HTA": -0.02}
prev = None
for seg, lane, s, e, n in runs:
    y = LANE_Y[lane]
    axA.add_patch(Rectangle((s, y), max(e - s, 0.8), 0.22, facecolor=V.LANE[lane],
                            linewidth=0.8, edgecolor="white", zorder=3))
    if e - s > 11:
        axA.text((s + e) / 2, y + 0.11, seg, ha="center", va="center", fontsize=6.0,
                 color="white", fontweight="bold", zorder=4)
    if prev is not None:
        axA.plot([prev[0], s], [prev[1], y + 0.11], color=V.MUTED, lw=0.55,
                 zorder=2, solid_capstyle="butt")
    prev = (e, y + 0.11)

for lane, y in LANE_Y.items():
    axA.text(-6, y + 0.11, lane, ha="right", va="center", fontsize=7.4,
             color=V.LANE[lane], fontweight="bold")
V.ev(axA, [0], 0.96, "release")
V.ev(axA, [SPAN], 0.96, "complete")
axA.annotate("", xy=(SPAN, 0.99), xytext=(0, 0.99),
             arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=V.INK,
                             mutation_scale=8, shrinkA=0, shrinkB=0))
axA.text(SPAN / 2, 1.05, f"observation age {SPAN:.0f} ms", ha="center", va="bottom",
         fontsize=7.2, color=V.INK, fontweight="bold")
axA.set_xlim(-70, SPAN * 1.03)
axA.set_ylim(-0.10, 1.24)
axA.axis("off")
axA.set_title("one inference · 71 dispatches over 8 segments · MEASURED",
              loc="left", pad=1.0, x=-0.008)

# ------------------------------------------------------------------ panel B
PER, WIN = 124.8, 640.0
nlane = int(np.ceil(SPAN / PER))
for k in range(int(np.ceil(WIN / PER))):
    off = k * PER
    row = k % nlane
    for seg, lane, s, e, n in runs:
        if s + off > WIN:
            continue
        axB.add_patch(Rectangle((s + off, row * 0.30 + 0.03),
                                max(min(e + off, WIN) - (s + off), 0.8), 0.24,
                                facecolor=V.LANE[lane], linewidth=0, zorder=3))
V.release_rail(axB, nlane * 0.30 + 0.12, 0, WIN, PER, SPAN)
axB.text(WIN * 1.005, nlane * 0.30 + 0.12, "  release rail", ha="left", va="center",
         fontsize=6.4, color=V.INK2)
for k in range(nlane):
    axB.text(-6, k * 0.30 + 0.15, f"in flight {k + 1}", ha="right", va="center",
             fontsize=6.6, color=V.MUTED)
axB.set_xlim(-70, WIN * 1.03)
axB.set_ylim(-0.03, nlane * 0.30 + 0.30)
axB.axis("off")
axB.set_title(f"released every {PER:.0f} ms · up to {nlane} in flight · "
              f"{int(np.sum(np.arange(0, WIN, PER) + SPAN <= WIN))} actions in {WIN:.0f} ms",
              loc="left", pad=1.0, x=-0.008)

fig.legend(handles=[plt.Line2D([], [], marker="s", ls="none", ms=6, color=V.LANE[l],
                               label=f"{l}  {V.LANE_MACHINE[l]}") for l in V.LANE_ORDER]
           + V.glyph_handles(["release", "complete"]),
           loc="upper right", bbox_to_anchor=(0.99, 1.008), ncol=5)
V.save(fig, "cand_a2_cascade_flow", f"span={SPAN:.1f} runs={len(runs)}")
