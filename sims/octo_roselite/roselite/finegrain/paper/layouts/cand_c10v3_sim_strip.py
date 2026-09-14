#!/usr/bin/env python3
"""C10v3 -- SIM STRIP, keyframes tied to the timeline by leader lines.

The defect in C10v2: frames were laid out in EQUAL-WIDTH columns but labelled
with UNEQUAL times (0, 1.5, 3, 4.5, 6, 10, 16, 22 s), so the frame captioned
"10.0 s" sat 62% across a strip whose rail ran to 24 s -- it belongs at 42%.
Position and label disagreed, which is why the strip was hard to read against
its own rail.

Fix: frames stay uniform and readable, and a LEADER LINE drops from each frame
to its exact instant on a rail that both rows share. Where the leaders fan
inward -- the XPU-RT row -- the whole episode happened inside the left quarter
of the baseline's window. That convergence IS the result.

NOTE ON EPISODE LENGTH: on widowx_put_eggplant_in_basket a FAILURE has no early
termination; it runs the full 600-tick / 24.0 s horizon. Every rendered cpu685
egg failure on disk is exactly 24.0 s, so the two rows cannot be brought closer
in length by choosing a different episode -- the 4.3x ratio is structural, not
a sampling choice.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vocab as V

V.style()
TICK = 40.0
TMAX = 24.4
ARMS = [("pipe110", "XPU-RT pipelined", 117.6, 1),
        ("cpu685", "QNN baseline", 684.8, 1)]
NF = 8


def strip(arm, ep):
    """Uniformly spaced instants over THIS episode, and the cached frames."""
    age, act, ok = V.age_trace(arm, ep)
    dur = len(age) * TICK / 1000.0
    V.video_frames(arm, ep, n=2, src="runs")
    fs = sorted((V.HERE / ".frames" / f"runs_egg_{arm}_{ep:02d}").glob("f_*.jpg"))
    ts = np.linspace(0, dur, NF)
    idx = np.round(ts / max(dur, 1e-9) * (len(fs) - 1)).astype(int)
    return age, act, ok, dur, ts, [fs[min(i, len(fs) - 1)] for i in idx]


fig = plt.figure(figsize=(7.16, 4.10))
X0, X1 = 0.052, 0.988
FW = (X1 - X0) / NF

for k, (arm, lab, per, ep) in enumerate(ARMS):
    age, act, ok, dur, ts, frames = strip(arm, ep)
    top = 0.930 - k * 0.408
    for j, (tt, f) in enumerate(zip(ts, frames)):
        a = fig.add_axes([X0 + j * FW, top - 0.193, FW * 0.972, 0.193])
        a.axis("off")
        a.imshow(mpimg.imread(f))

    ax = fig.add_axes([X0, top - 0.330, X1 - X0, 0.137])
    ax.set_xlim(0, TMAX); ax.set_ylim(1.0, 0.0); ax.axis("off")
    # leader: frame centre (top, in axes-fraction x) -> its true instant
    for j, tt in enumerate(ts):
        xf = (j + 0.4875) * FW / (X1 - X0) * TMAX
        ax.plot([xf, xf, tt, tt], [0.0, 0.20, 0.62, 0.80], color=V.AXIS,
                lw=0.55, zorder=2, clip_on=False, solid_capstyle="butt")

    rel = np.arange(0, dur, per / 1000.0)
    ax.plot([0, TMAX], [0.86, 0.86], color=V.AXIS, lw=1.0, zorder=3)
    V.ev(ax, rel, 0.965, "release", ms=4.0)
    don = rel + per / 1000.0
    V.ev(ax, don[don <= dur], 0.86, "complete", ms=4.6)   # never past the outcome
    for t in V.gripper_events(act):
        V.ev(ax, t * TICK / 1000.0, 0.86, "grip", ms=6.0)
    V.ev(ax, dur, 0.86, "success" if ok else "failure", ms=11 if ok else 8)

    # Label sits BELOW the rail: the band above it belongs to the leaders.
    ax.text(0, 1.42, f"{lab}   ·   period {per:.0f} ms   ·   {len(rel)} commands"
                     f"   ·   mean age {np.nanmean(age):.0f} ms   ·   {dur:.1f} s",
            fontsize=7.2, color=V.INK, va="center", ha="left", fontweight="bold",
            clip_on=False)


# One shared time axis in its own strip, so its labels cannot be pushed off the
# canvas by whatever the last row happens to draw.
tax = fig.add_axes([X0, 0.088, X1 - X0, 0.001])
tax.set_xlim(0, TMAX); tax.set_yticks([])
for sp in ("left", "right", "top"):
    tax.spines[sp].set_visible(False)
tax.spines["bottom"].set_color(V.AXIS)
tax.set_xticks(range(0, 25, 4))
tax.tick_params(labelsize=6.6, colors=V.MUTED, labelcolor=V.INK2, pad=1.5)
tax.set_xlabel("episode time (s)  ·  one shared axis, both rows", fontsize=7.0,
               labelpad=1.5)

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.5, 1.004), ncol=5, fontsize=6.8)
V.save(fig, "cand_c10v3_sim_strip")
