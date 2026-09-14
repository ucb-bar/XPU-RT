#!/usr/bin/env python3
"""C9 -- EVENT-MARKED EPISODE TIMELINE. The trackability figure.

Four schedules of the SAME task on ONE shared episode-time axis. Every row is a
MEASURED rollout and every row carries the same rail and the same glyphs:

  release ticks and inference triangles  -- from the per-tick observation age
  gripper-closed span and its onset      -- from the applied action stream,
                                            threshold verified against the run's
                                            own gripper_frac_closed
  outcome                                -- star or cross at the last tick

Read down a column and you are comparing the same instant in four schedules.
Read a row and you are following one episode. The right gutter carries the three
numbers that summarise the row: commands delivered, mean observation age,
episode length.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
ARMS = [("ideal", "ideal 0 ms", 0.0, 200.0),
        ("pipe110", "XPU-RT pipe110", 117.7, 117.6),
        ("serial283", "XPU-RT serial", 283.4, 283.4),
        ("cpu685", "QNN baseline", 684.8, 684.8)]
TICK = 40.0

fig = plt.figure(figsize=(7.16, 3.55))
ax = fig.add_axes([0.135, 0.105, 0.705, 0.775])
ROW = 1.0

tmax = 0.0
for k, (arm, lab, lat, per) in enumerate(ARMS):
    age, act, ok = V.age_trace(arm, 0)
    t = np.arange(len(age)) * TICK / 1000.0
    tmax = max(tmax, t[-1])
    y0 = -k * ROW

    # age band, normalised inside the row so four very different scales coexist
    a = np.nan_to_num(age, nan=0.0)
    h = 0.46 * a / 1400.0
    ax.fill_between(t, y0, y0 + h, step="post", color=V.c_period(per), lw=0, zorder=3)

    # gripper-closed span
    on, off, closed = V.gripper_events(act)
    seg, s0 = [], None
    for i, c in enumerate(closed):
        if c and s0 is None:
            s0 = i
        elif not c and s0 is not None:
            seg.append((s0, i)); s0 = None
    if s0 is not None:
        seg.append((s0, len(closed) - 1))
    for a0, a1 in seg:
        ax.add_patch(Rectangle((t[a0], y0 - 0.20), t[a1] - t[a0], 0.115,
                               facecolor=V.MUTED, alpha=0.32, linewidth=0, zorder=2))
    if len(on):
        V.ev(ax, t[on], y0 - 0.143, "grip", ms=5.6)

    # the rail: releases and completions
    ax.plot([0, t[-1]], [y0 + 0.60, y0 + 0.60], color=V.AXIS, lw=0.9, zorder=2)
    rel = np.arange(0, t[-1] * 1000, per) / 1000.0
    V.ev(ax, rel, y0 + 0.60, "release", ms=5.2)
    comp = V.completions(age)
    V.ev(ax, t[comp], y0 + 0.60, "complete", ms=4.6)

    V.ev(ax, [t[-1] + 0.13], [y0 + 0.16], "success" if ok else "failure", ms=10)

    ax.text(-0.16, y0 + 0.30, lab, ha="right", va="center", fontsize=7.4,
            color=V.INK, fontweight="bold")
    ax.text(-0.16, y0 + 0.04, f"period {per:.0f} ms · age {lat:.0f} ms",
            ha="right", va="center", fontsize=6.2, color=V.MUTED)

    s = json.load(open(V.AGE_RUNS[arm][0] / "summary.json"))["episodes"][0]
    for j, txt in enumerate([f"{s['n_inferences']} commands",
                             f"{np.nanmean(age):.0f} ms mean age",
                             f"{t[-1]:.1f} s episode"]):
        ax.text(1.007, (y0 + 0.44 - 0.20 * j) / (len(ARMS) * ROW) + 1.0 - 1.0 / (len(ARMS)) * 0,
                txt, transform=None, fontsize=0)  # placeholder, replaced below

ax.set_xlim(-0.05, tmax + 0.55)
ax.set_ylim(-len(ARMS) * ROW + 0.32, 0.82)
ax.set_yticks([])
ax.set_xlabel("episode time (s)  ·  40 ms control tick  ·  MEASURED rollout, "
              "eggplant in basket")
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.grid(True, axis="x", color=V.GRID, lw=0.6)
ax.set_axisbelow(True)

# right gutter, in axes coordinates so it cannot drift
gut = fig.add_axes([0.845, 0.105, 0.150, 0.775])
gut.axis("off")
gut.set_xlim(0, 1); gut.set_ylim(-len(ARMS) * ROW + 0.32, 0.82)
for k, (arm, lab, lat, per) in enumerate(ARMS):
    age, act, ok = V.age_trace(arm, 0)
    s = json.load(open(V.AGE_RUNS[arm][0] / "summary.json"))["episodes"][0]
    y0 = -k * ROW
    gut.text(0.0, y0 + 0.52, f"{s['n_inferences']}", fontsize=11, color=V.INK,
             fontweight="bold", va="center")
    gut.text(0.30, y0 + 0.52, "commands", fontsize=6.2, color=V.INK2, va="center")
    gut.text(0.0, y0 + 0.24, f"{np.nanmean(age):.0f}", fontsize=8.6, color=V.INK2,
             va="center")
    gut.text(0.30, y0 + 0.24, "ms mean age", fontsize=6.2, color=V.MUTED, va="center")
    gut.text(0.0, y0 - 0.02, f"{len(age) * TICK / 1000:.1f}", fontsize=8.6,
             color=V.INK2, va="center")
    gut.text(0.30, y0 - 0.02, "s episode", fontsize=6.2, color=V.MUTED, va="center")

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"])
           + [Rectangle((0, 0), 1, 1, facecolor=V.MUTED, alpha=0.32,
                        label="gripper closed"),
              Rectangle((0, 0), 1, 1, facecolor=V.c_period(200), label="observation age")],
           loc="upper left", bbox_to_anchor=(0.135, 1.005), ncol=7, handlelength=1.2,
           columnspacing=1.0)
V.save(fig, "cand_c9_episode_timeline")
