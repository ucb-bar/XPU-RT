#!/usr/bin/env python3
"""C10v6 -- SIM STRIP on SPOON, where the two timelines are comparable.

The eggplant pairing is lopsided for a structural reason, not a sampling one:
across 26,203 failed episodes in the finished sweep a FAILURE always runs
exactly to the task horizon -- egg 600 ticks, spoon 300, coke 720, drawer 1017,
with zero variance. There is no early failure termination in SIMPLER-env, so no
choice of episode can make a failing eggplant baseline finish sooner than 24.0 s
against a 5.6 s success.

Spoon fixes it by having a shorter horizon. Its baseline failure is 12.0 s
against a 7.0 s success -- 1.7x rather than 4.3x -- so both rows fill the
canvas and the frames stay large. Better still, both rollouts here are EPISODE
21 of the same ladder run, so this is a matched pair: same initial scene, same
object pose, only the schedule differs. That is the cleanest possible statement
of "the schedule caused this".

Layout is C10v3's: uniform frames, leader lines to a shared absolute rail.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vocab as V

V.style()
TICK = 40.0
TMAX = 12.4
EP = 21
ARMS = [("pipe110", "XPU-RT pipelined", 117.6),
        ("cpu685", "QNN baseline", 684.8)]
NF = 8

# Inches, then converted: the frame axes must match the video's 4:3 aspect
# exactly, or a 4:3 image inside a squarer axes is padded top and bottom and that
# padding shows up as a gap between the images and their leader lines.
FIG_W, ASPECT = 7.16, 0.75         # cached frames are 384x288
X0, X1 = 0.030, 0.994
FW = (X1 - X0) / NF
frame_h = FW * 0.976 * FIG_W * ASPECT
band_h, gap_rows, sax_gap, sax_lab, legend_h = 0.302, 0.060, 0.020, 0.300, 0.200
FIG_H = legend_h + 2 * (frame_h + band_h) + gap_rows + sax_gap + sax_lab
fig = plt.figure(figsize=(FIG_W, FIG_H))
FRAME_H, BAND_H = frame_h / FIG_H, band_h / FIG_H
PITCH = (frame_h + band_h + gap_rows) / FIG_H
TOP0 = 1.0 - legend_h / FIG_H

for k, (arm, lab, per) in enumerate(ARMS):
    d = V.BASE / "videos_ladder" / f"spoon_{arm}"
    age = np.load(d / f"ep{EP:02d}_action_age_ms.npy")
    act = np.load(d / f"ep{EP:02d}_applied_actions.npy")
    ok = (d / f"ep{EP:02d}_success_True.mp4").exists()
    dur = len(age) * TICK / 1000.0
    V.video_frames(arm, EP, n=2, task="spoon", src="ladder")
    fs = sorted((V.HERE / ".frames" / f"ladder_spoon_{arm}_{EP:02d}").glob("f_*.jpg"))
    ts = np.linspace(0, dur, NF)
    rel = np.arange(0, dur, per / 1000.0)

    top = TOP0 - k * PITCH
    for j, tt in enumerate(ts):
        a = fig.add_axes([X0 + j * FW, top - FRAME_H, FW * 0.976, FRAME_H])
        a.axis("off")
        a.imshow(mpimg.imread(fs[min(int(round(tt / dur * (len(fs) - 1))), len(fs) - 1)]))


    # ONE LINE across the top of the strip, in FIGURE coordinates so it is not
    # tied to the first frame's axes and does not blot out that frame.
    fig.text(X0 + 0.006, top - 0.021,
             f"{lab}   ·   {per:.0f} ms period   ·   {len(rel)} commands   ·   "
             f"mean latency {np.nanmean(age):.0f} ms   ·   {dur:.1f} s",
             ha="left", va="top", fontsize=6.4, color=V.INK, fontweight="bold",
             zorder=9, bbox=dict(fc="white", ec="none", alpha=0.80, pad=2.0))

    ax = fig.add_axes([X0, top - FRAME_H - BAND_H, X1 - X0, BAND_H])
    ax.set_xlim(0, TMAX); ax.set_ylim(1.0, 0.0); ax.axis("off")
    for j, tt in enumerate(ts):
        xf = (j + 0.488) * FW / (X1 - X0) * TMAX
        ax.plot([xf, xf, tt, tt], [0.0, 0.26, 0.66, 0.84], color=V.AXIS,
                lw=1.15, zorder=2, clip_on=False, solid_capstyle="butt")

    ax.plot([0, TMAX], [0.90, 0.90], color=V.AXIS, lw=1.0, zorder=3)
    V.ev(ax, rel, 1.05, "release", ms=4.6)
    don = rel + per / 1000.0
    V.ev(ax, don[don <= dur], 0.90, "complete", ms=5.2)
    for t in V.gripper_events(act):
        V.ev(ax, t * TICK / 1000.0, 0.90, "grip", ms=6.4)
    V.ev(ax, dur, 0.90, "success" if ok else "failure", ms=11 if ok else 8)

tax = fig.add_axes([X0, TOP0 - PITCH - FRAME_H - BAND_H - sax_gap / FIG_H,
                    X1 - X0, 0.001])
tax.set_xlim(0, TMAX); tax.set_yticks([])
for sp in ("left", "right", "top"):
    tax.spines[sp].set_visible(False)
tax.spines["bottom"].set_color(V.AXIS)
tax.set_xticks(range(0, 13, 2))
tax.tick_params(labelsize=6.6, colors=V.MUTED, labelcolor=V.INK2, pad=1.5)
tax.set_xlabel("episode time (s)", fontsize=7.0, labelpad=1.5)

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.5, 1.008), ncol=5, fontsize=6.6)
V.save(fig, "cand_c10v6_sim_strip_spoon")
