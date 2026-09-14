#!/usr/bin/env python3
"""A6 -- THE BRIDGE. Schedule scale and episode scale on one event vocabulary.

Top    the in-flight ladder over a 2 s window: one row per inference, drawn from
       the MEASURED board trace. Release ticks and completion triangles on the
       rail.
Bottom the MEASURED per-tick observation age for a whole 8 s rollout of the same
       two schedules, with the top panel's window marked as a bracket.

This is the figure that makes the rest of the set trackable: the SAME triangle
that ends an inference in the top panel is the SAME triangle that drops the age
in the bottom panel, and the bottom panel's x-axis is the one C9 and C10 use.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
ZOOM = 2000.0
ARMS = [("XPU-RT pipelined", "pipe110", "p150w300"),
        ("QNN CPU baseline", "cpu685", "mono")]

fig = plt.figure(figsize=(7.16, 3.55))
gs = fig.add_gridspec(2, 2, left=0.085, right=0.988, top=0.90, bottom=0.10,
                      hspace=0.50, wspace=0.10, height_ratios=[1.0, 1.25])

# ------------------------------------------------ top: the in-flight ladder
for j, (title, arm, group) in enumerate(ARMS):
    ax = fig.add_subplot(gs[0, j])
    _, rows, _ = V.board_median(group)
    iv = V.inferences(rows)
    starts = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    per = float(np.median(np.diff(starts))) if len(starts) > 1 else span
    nrel = int(np.ceil(ZOOM / per))
    nfl = int(np.ceil(span / per))
    for k in range(nrel):
        s = k * per
        ax.add_patch(Rectangle((s, -(k % nfl) - 0.78), min(span, ZOOM - s), 0.62,
                               facecolor=V.CMAP_PERIOD(0.30 + 0.13 * (k % 3)),
                               linewidth=0, zorder=3))
    V.release_rail(ax, 0.16, 0, ZOOM, per, span)
    ax.set_xlim(-20, ZOOM + 20)
    ax.set_ylim(-nfl - 0.95, 0.60)
    ax.set_yticks([])
    ax.set_xticks([0, 1000, 2000])
    ax.set_xlabel("ms since window start")
    ax.set_title(f"{title}\nperiod {per:.0f} ms · age {span:.0f} ms · "
                 f"{nfl} in flight", loc="left", pad=3, fontsize=7.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if j == 0:
        ax.text(-0.055, 0.5, "in-flight\nladder", transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=6.8, color=V.MUTED)

# ------------------------------------- bottom: the measured age, whole episode
for j, (title, arm, group) in enumerate(ARMS):
    ax = fig.add_subplot(gs[1, j])
    age, act, ok = V.age_trace(arm, 0)
    t = np.arange(len(age)) * 40.0 / 1000.0
    ax.step(t, age, where="post", color=V.INK2, lw=1.0, zorder=4)
    ax.fill_between(t, 0, np.nan_to_num(age), step="post",
                    color=V.CMAP_PERIOD(0.22), lw=0, zorder=2)
    comp = V.completions(age)
    V.ev(ax, t[comp], age[comp], "complete", ms=4.6)
    on, off, closed = V.gripper_events(act)
    if len(on):
        V.ev(ax, t[on], np.full(len(on), -70.0), "grip", ms=6)
    V.ev(ax, [t[-1]], [-70.0], "success" if ok else "failure", ms=10)
    ax.add_patch(Rectangle((0, -110), ZOOM / 1000.0, 1600, facecolor="none",
                           edgecolor=V.MUTED, lw=0.9, ls=(0, (3, 2)), zorder=5))
    ax.text(ZOOM / 1000.0, 1420, " window above", fontsize=6.2, color=V.MUTED,
            va="top")
    ax.set_xlim(-0.1, max(8.2, t[-1] + 0.1))
    ax.set_ylim(-110, 1480)
    ax.set_xlabel("episode time (s)  ·  40 ms control tick")
    ax.set_title(f"MEASURED observation age · mean {np.nanmean(age):.0f} ms",
                 loc="left", pad=3, fontsize=7.6)
    V.tidy(ax, grid="y")
    if j == 0:
        ax.set_ylabel("age of the observation\nbeing acted on (ms)")
    else:
        ax.set_yticklabels([])

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.53, 1.008), ncol=5)
V.save(fig, "cand_a6_ladder_age")
