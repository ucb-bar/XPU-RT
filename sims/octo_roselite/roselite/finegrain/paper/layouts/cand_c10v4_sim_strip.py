#!/usr/bin/env python3
"""C10v4 -- SIM STRIP at TRUE SCALE. The strip IS the timeline.

The strictest reading of "keyframes should match the timeline": each row's
frames are laid end to end across exactly the span of wall-clock time its
episode occupied, on an axis both rows share. No leader lines are needed
because a frame's horizontal position IS its instant.

The cost is that frame SIZE then encodes duration -- the XPU-RT row finishes in
23% of the baseline's window, so its frames are 23% as wide. That is a real
property of the data, not a defect, and it is the fastest read of the three
versions: one row stops a quarter of the way across.

Use this when the headline is DURATION. Use C10v3 (leader lines, uniform frame
size) when the headline is WHAT THE ARM DID, since there the frames stay big
enough to see the gripper.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vocab as V

V.style()
TICK = 40.0
TMAX = 24.4
ARMS = [("pipe110", "XPU-RT pipelined", 117.6, 1, 4),
        ("cpu685", "QNN baseline", 684.8, 1, 9)]

fig = plt.figure(figsize=(7.16, 3.30))
X0, X1 = 0.052, 0.988
SPAN = X1 - X0

for k, (arm, lab, per, ep, nf) in enumerate(ARMS):
    age, act, ok = V.age_trace(arm, ep)
    dur = len(age) * TICK / 1000.0
    V.video_frames(arm, ep, n=2, src="runs")
    fs = sorted((V.HERE / ".frames" / f"runs_egg_{arm}_{ep:02d}").glob("f_*.jpg"))
    top = 0.858 - k * 0.408
    fw = SPAN * (dur / TMAX) / nf              # a frame is one slice of real time
    for j in range(nf):
        tt = (j + 0.5) / nf * dur
        a = fig.add_axes([X0 + SPAN * (j / nf) * (dur / TMAX), top - 0.215,
                          fw * 0.965, 0.215])
        a.axis("off")
        a.imshow(mpimg.imread(fs[min(int(round(tt / dur * (len(fs) - 1))), len(fs) - 1)]))

    ax = fig.add_axes([X0, top - 0.300, SPAN, 0.070])
    ax.set_xlim(0, TMAX); ax.set_ylim(1.0, 0.0); ax.axis("off")
    rel = np.arange(0, dur, per / 1000.0)
    ax.plot([0, TMAX], [0.5, 0.5], color=V.AXIS, lw=1.0, zorder=2)
    ax.plot([0, dur], [0.5, 0.5], color=V.INK2, lw=1.4, zorder=3)   # the episode
    V.ev(ax, rel, 0.78, "release", ms=4.0)
    don = rel + per / 1000.0
    V.ev(ax, don[don <= dur], 0.5, "complete", ms=4.6)
    for t in V.gripper_events(act):
        V.ev(ax, t * TICK / 1000.0, 0.5, "grip", ms=6.0)
    V.ev(ax, dur, 0.5, "success" if ok else "failure", ms=11 if ok else 8)
    # Label ABOVE the strip: below it is the rail, and below that the shared axis.
    ax.text(0, -3.35, f"{lab}   ·   period {per:.0f} ms   ·   {len(rel)} commands"
                      f"   ·   mean age {np.nanmean(age):.0f} ms   ·   {dur:.1f} s",
            fontsize=7.2, color=V.INK, va="center", ha="left", fontweight="bold",
            clip_on=False)

tax = fig.add_axes([X0, 0.105, SPAN, 0.001])
tax.set_xlim(0, TMAX); tax.set_yticks([])
for sp in ("left", "right", "top"):
    tax.spines[sp].set_visible(False)
tax.spines["bottom"].set_color(V.AXIS)
tax.set_xticks(range(0, 25, 4))
tax.tick_params(labelsize=6.6, colors=V.MUTED, labelcolor=V.INK2, pad=1.5)
tax.set_xlabel("episode time (s)  ·  frame width = real time, both rows to scale",
               fontsize=7.0, labelpad=1.5)

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=5, fontsize=6.8)
V.save(fig, "cand_c10v4_sim_strip")
