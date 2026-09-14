#!/usr/bin/env python3
"""A1v2 -- LANE GANTT, refined. Three MEASURED board schedules, one time window.

Changes from A1:

1. THE SERIAL GAP IS GONE, because it was an artefact of mixing two different
   numbers. A1 drew the MEASURED dispatch template (span 250.07 ms) but tiled it
   at the MODELLED period from the cost model (283.4 ms), so 33 ms of phantom
   idle opened up between invocations. A serial chain has period == latency by
   construction -- the next inference cannot start until this one ends -- so the
   self-consistent measured rendering tiles at the template's own span and the
   invocations sit back to back. (The simulator was fed the cost model's 283.4;
   that belongs in the caption, not on a panel labelled MEASURED. A7 still shows
   250/283 and should be reconciled to whichever number the paper adopts.)

2. PIPELINE STAGES ARE VISIBLE. Each dispatch is shaded by the parity of the
   inference it belongs to, inside its own lane colour: full tone for even,
   45% tint for odd. Lane identity survives, and the interleave of two shades
   IS the pipelining -- where the two alternate within a lane, two inferences
   are in flight at once.

3. AGE AND PERIOD ARE BRACKETS ON THE RAIL, not words in the title. They are
   measurements of the rail's own intervals, so they belong on it: the age
   bracket spans release -> completion, the period bracket spans release ->
   release. Reading the difference off the two brackets is the point.

4. Half the height.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import vocab as V

V.style()
WINDOW = 700.0
# Wrapped by hand: these ride in the left margin, not in a title band above
# each panel -- three title bands cost more vertical space than the lanes do.
ROWS = [("QNN CPU\nbaseline", "mono"),
        ("XPU-RT\nserial", "serial"),
        ("XPU-RT pipelined\np150/w300", "p150w300")]


def tint(c, f=0.55):
    """Blend a lane colour f of the way to the surface -- the odd-stage shade."""
    r, g, b = to_rgb(c)
    s = to_rgb(V.SURFACE)
    return tuple(x + (y - x) * f for x, y in zip((r, g, b), s))


def bracket(ax, y, x0, x1, text, *, cap=0.19, fs=6.2, gap=0.30):
    """A measured interval, drawn as an interval.

    The rule is broken in the middle and the label sits in the break, so the
    text can never be struck through -- which it is if you rely on a bbox mask
    on an inverted y-axis.
    """
    mid, half = (x0 + x1) / 2, (x1 - x0) * gap / 2
    ax.plot([x0, x0, mid - half], [y - cap, y, y], color=V.INK2, lw=0.75,
            zorder=7, clip_on=False, solid_capstyle="butt")
    ax.plot([mid + half, x1, x1], [y, y, y - cap], color=V.INK2, lw=0.75,
            zorder=7, clip_on=False, solid_capstyle="butt")
    # The break is sized in DATA units and the label in points, so on a short
    # interval the text still outruns it -- the bbox is the backstop.
    ax.text(mid, y, text, ha="center", va="center", fontsize=fs,
            color=V.INK2, zorder=8, clip_on=False,
            bbox=dict(fc=V.SURFACE, ec="none", pad=1.4))


fig = plt.figure(figsize=(7.16, 2.30))
gs = fig.add_gridspec(3, 1, left=0.190, right=0.988, top=0.960, bottom=0.215,
                      hspace=0.22)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

for ax, (title, group) in zip(axes, ROWS):
    _, rows, _ = V.board_median(group)
    iv = V.inferences(rows)
    starts = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    # Measured, self-consistent: a pipelined trace states its own cadence; a
    # single-inference trace is a serial chain, so its cadence IS its span.
    per = float(np.median(np.diff(starts))) if len(starts) > 1 else span
    order = {k: i for i, k in enumerate(iv)}           # inference index, in start order
    reps = [0.0] if len(starts) > 1 else list(np.arange(0, WINDOW, per))

    for ri, off in enumerate(reps):
        for r in rows:
            lane = r["lane"]
            if lane not in V.LANE:
                continue
            s, e = r["s"] + off, r["e"] + off
            if s > WINDOW:
                continue
            k = order.get(r["inst"], 0) + ri          # tiled copies keep alternating
            base = V.LANE[lane]
            ax.barh(V.LANE_ORDER.index(lane), min(e, WINDOW) - s, left=s, height=0.62,
                    color=base if k % 2 == 0 else tint(base), linewidth=0, zorder=3)

    # Both brackets live BELOW the rail. Above it is the lane band, and an
    # age bracket drawn up there lands on top of the HTA bars.
    # Both brackets live BELOW the rail. Above it is the lane band, and an
    # age bracket drawn up there lands on top of the HTA bars.
    two = abs(span - per) > 1.0
    ax.set_ylim(4.70 if two else 3.92, -0.66)
    ax.set_xlim(-6, WINDOW + 6)
    rail = 2.62
    rel = V.release_rail(ax, rail, 0, WINDOW, per, span, drop=0.24, ms=5.4)
    if two:
        bracket(ax, rail + 0.90, rel[0], rel[0] + span, f"latency {span:.0f} ms")
        # The period bracket is measured COMPLETION to COMPLETION, between the
        # first two, so it never shares an x-range with the age bracket above it.
        bracket(ax, rail + 1.66, rel[0] + span, rel[1] + span, f"period {per:.0f} ms")
    else:
        # A serial chain: the next release IS the previous completion. One
        # bracket, and saying so is the finding.
        bracket(ax, rail + 0.90, rel[0], rel[0] + per,
                f"latency = period = {per:.0f} ms")

    n = int(np.sum(np.arange(0, WINDOW, per) + span <= WINDOW))
    ax.set_yticks(range(3))
    ax.set_yticklabels(V.LANE_ORDER, fontsize=6.6)
    for t, l in zip(ax.get_yticklabels(), V.LANE_ORDER):
        t.set_color(V.LANE[l]); t.set_fontweight("bold")
    bb = ax.get_position()
    fig.text(0.132, bb.y0 + bb.height * 0.52,
             f"{title}\n{n} action{'s' if n != 1 else ''}",
             ha="right", va="center", fontsize=6.9, color=V.INK,
             fontweight="bold", linespacing=1.34)
    V.tidy(ax, grid="x")
    if ax is not axes[-1]:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("schedule time (ms)", labelpad=1.5)

# The key belongs beside the axis it explains, not floating over the first title.
fig.legend(handles=V.glyph_handles(["release", "complete"]),
           loc="lower right", bbox_to_anchor=(0.992, 0.002), ncol=2, fontsize=6.6)
V.save(fig, "cand_a1v2_gantt_lanes")
