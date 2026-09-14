#!/usr/bin/env python3
"""A7 -- CASCADE, refined. The mechanism figure the paper should carry.

Refinements over A1 and A2:
  * the RAIL IS ON TOP of every panel, where a reader looks first, and releases
    hang below it while completions sit on it, so the baseline row shows both;
  * the pipelined trace is drawn TWICE from the same dispatches -- coloured by
    LANE (which silicon moved) and by INFERENCE INDEX (what overlaps). One trace,
    two questions, no extra claim;
  * a right gutter carries per-lane busy and duty cycle, so "the SoC is used"
    stops being an adjective;
  * a single short annotation, no sentences on the canvas.
"""
from __future__ import annotations
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
WIN = 700.0
# four widely separated steps of the period ramp: at most 3 inferences are
# ever in flight (asserted below), so two concurrent ones never share a shade
SHADE = ["#b7d3f6", "#3987e5", "#0d366b", "#6da7ec"]

fig = plt.figure(figsize=(7.16, 3.95))
gs = fig.add_gridspec(4, 1, left=0.150, right=0.815, top=0.905, bottom=0.088,
                      hspace=0.66, height_ratios=[1, 1, 1, 1])
GUT = fig.add_axes([0.822, 0.088, 0.175, 0.817]); GUT.axis("off")
GUT.set_xlim(0, 1); GUT.set_ylim(0, 4)

PANELS = [("QNN CPU baseline", "mono", "lane", None),
          ("XPU-RT serial", "serial", "lane", 283.4),
          ("XPU-RT pipelined  p150/w300", "p150w300", "lane", None),
          ("the same trace, by inference", "p150w300", "inst", None)]

for r, (title, group, mode, tile) in enumerate(PANELS):
    ax = fig.add_subplot(gs[r])
    _, rows, _ = V.board_median(group)
    iv = V.inferences(rows)
    st = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    per = tile or (float(np.median(np.diff(st))) if len(st) > 1 else span)
    order = {k: i for i, k in enumerate(sorted(iv, key=lambda k: iv[k][0]))}

    reps = [(0.0, 1.0, False)] if not tile else [
        (k * tile, 1.0 if k == 0 else 0.34, k > 0)
        for k in range(int(np.ceil(WIN / tile)))]

    busy = collections.defaultdict(float)
    for off, alpha, ghost in reps:
        for rr in rows:
            lane = rr["lane"]
            if lane not in V.LANE:
                continue
            s0, e0 = rr["s"] + off, rr["e"] + off
            if s0 > WIN:
                continue
            col = V.LANE[lane] if mode == "lane" else SHADE[order[rr["inst"]] % 4]
            ax.barh(V.LANE_ORDER.index(lane), min(e0, WIN) - s0, left=s0, height=0.58,
                    color=col, alpha=alpha, linewidth=0, zorder=3,
                    hatch="///" if ghost else None,
                    edgecolor="white" if ghost else "none")
            if not ghost:
                busy[lane] += min(e0, WIN) - s0
            elif alpha < 1:
                busy[lane] += min(e0, WIN) - s0

    ax.set_ylim(3.30, -1.30)
    V.release_rail(ax, -0.95, 0, WIN, per, span, ms=5.4)
    ax.set_xlim(-6, WIN + 6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{l}  {V.LANE_MACHINE[l]}" for l in V.LANE_ORDER], fontsize=6.4)
    for t, l in zip(ax.get_yticklabels(), V.LANE_ORDER):
        t.set_color(V.LANE[l] if mode == "lane" else V.INK2)
        t.set_fontweight("bold")
    done = int(np.sum(np.arange(0, WIN, per) + span <= WIN))
    ax.set_title(f"{title}      age {span:.0f} ms  ·  period {per:.0f} ms  ·  "
                 f"{done} action{'s' if done != 1 else ''} in {WIN:.0f} ms",
                 loc="left", pad=2.5, fontsize=7.6)
    V.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    if r < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("wall-clock time since window start (ms)   ·   MEASURED on "
                      "QRB5165, ungated")

    y = 3.5 - r
    for i, l in enumerate(V.LANE_ORDER):
        GUT.text(0.02, y - 0.14 - i * 0.20, f"{l}", fontsize=6.0,
                 color=V.LANE[l] if mode == "lane" else V.MUTED, fontweight="bold")
        GUT.text(0.24, y - 0.14 - i * 0.20, f"{busy[l]:5.0f} ms   "
                 f"{100 * busy[l] / WIN:4.0f}%", fontsize=6.0, color=V.INK2)
    if r == 0:
        GUT.text(0.02, 3.86, "busy in the window", fontsize=6.2, color=V.INK,
                 fontweight="bold")

# The annotation states a MEASURED count, computed here rather than asserted:
# an earlier draft said "three in flight", which is the number of ACTIONS
# COMPLETED in the window, not the number of concurrent inferences.
mx = max(sum(1 for a, b in V.inferences(V.board_median("p150w300")[1]).values()
             if a <= t <= b) for t in np.arange(0, WIN, 2.0))
fig.axes[4].annotate(f"{mx} shades interleave = {mx} inferences in flight",
                     xy=(352, 1.30), xytext=(300, 2.72), fontsize=6.4, color=V.INK, ha="center",
                     arrowprops=dict(arrowstyle="-|>", lw=0.7, color=V.INK,
                                     mutation_scale=6))
print(f"  max concurrent inferences in the drawn window = {mx}")
fig.legend(handles=V.glyph_handles(["release", "complete"]), loc="upper right",
           bbox_to_anchor=(0.995, 1.008), ncol=2)
V.save(fig, "cand_a7_cascade_refined")
