#!/usr/bin/env python3
"""A5 -- PARTITION. What the scheduler actually decided, as three small panels.

A  the mix3 partition: 52 single-graph contexts as a waffle, one cell per
   context, coloured by the lane it was placed on. Counts direct-labelled.
B  where the WORK went: measured per-inference busy ms per lane, with the
   bottleneck lane marked -- placement is not the same as balance.
C  what it cost at run time: 710 dispatches, 0 context evictions, as a single
   stat row (a number, not a chart -- the house rule for a lone headline).
"""
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
_, rows, _ = V.board_median("p150w300")
iv = V.inferences(rows)
nin = len(iv)
busy = collections.defaultdict(float)
disp = collections.Counter()
for r in rows:
    if r["lane"] in V.LANE:
        busy[r["lane"]] += (r["e"] - r["s"]) / nin
        disp[r["lane"]] += 1
span = float(np.median([b - a for a, b in iv.values()]))

fig = plt.figure(figsize=(7.16, 2.15))
gs = fig.add_gridspec(1, 3, left=0.045, right=0.988, top=0.80, bottom=0.14,
                      wspace=0.30, width_ratios=[1.25, 1.0, 0.78])
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

# ------------------------------------------------------------------- waffle
COLS = 13
cells = [l for l in V.LANE_ORDER for _ in range(V.PARTITION[l])]
for i, l in enumerate(cells):
    r, c = divmod(i, COLS)
    axA.add_patch(Rectangle((c, -r), 0.86, 0.86, facecolor=V.LANE[l], linewidth=0))
axA.set_xlim(-0.3, COLS + 0.1)
axA.set_ylim(-(len(cells) - 1) // COLS - 0.35, 1.05)
axA.axis("off")
axA.set_title(f"partition  ·  {sum(V.PARTITION.values())} single-graph contexts",
              loc="left", pad=3, x=-0.01)
x = 0
for l in V.LANE_ORDER:
    axA.text(x, 1.02, f"{l} {V.PARTITION[l]}", fontsize=7.0, color=V.LANE[l],
             fontweight="bold", va="bottom")
    x += 4.6

# ------------------------------------------------------------ per-lane busy
vals = [busy[l] for l in V.LANE_ORDER]
bars = axB.bar(range(3), vals, color=[V.LANE[l] for l in V.LANE_ORDER], width=0.62,
               linewidth=0)
kmax = int(np.argmax(vals))
axB.axhline(span, color=V.INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
axB.text(2.42, span, f"  span {span:.0f}", fontsize=6.4, color=V.INK2, va="center")
for i, (b, v) in enumerate(zip(bars, vals)):
    axB.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center",
             fontsize=6.8, color=V.INK, fontweight="bold")
axB.annotate("bottleneck", xy=(kmax, vals[kmax]), xytext=(kmax, vals[kmax] + 40),
             ha="center", fontsize=6.6, color=V.INK,
             arrowprops=dict(arrowstyle="-|>", lw=0.8, color=V.INK, mutation_scale=7))
axB.set_xticks(range(3))
axB.set_xticklabels(V.LANE_ORDER)
for t, l in zip(axB.get_xticklabels(), V.LANE_ORDER):
    t.set_color(V.LANE[l]); t.set_fontweight("bold")
axB.set_ylim(0, span * 1.10)
axB.set_ylabel("busy ms per inference")
axB.set_title("lane load  ·  MEASURED", loc="left", pad=3, x=-0.01)
V.tidy(axB)

# --------------------------------------------------------------- stat block
axC.axis("off")
for i, (val, lab) in enumerate([(f"{V.N_DISPATCH}", "dispatches"),
                                (f"{nin}", "inferences in trace"),
                                (f"{V.N_EVICT}", "context evictions")]):
    axC.text(0.02, 0.86 - 0.33 * i, val, fontsize=17, color=V.INK,
             fontweight="bold", va="center", transform=axC.transAxes)
    axC.text(0.40, 0.86 - 0.33 * i, lab, fontsize=7.2, color=V.INK2, va="center",
             transform=axC.transAxes)
axC.set_title("run-time cost", loc="left", pad=3, x=-0.01)

V.save(fig, "cand_a5_partition_waffle", f"busy={dict(busy)} disp={dict(disp)}")
