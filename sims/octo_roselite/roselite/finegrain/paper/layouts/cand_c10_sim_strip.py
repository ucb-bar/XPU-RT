#!/usr/bin/env python3
"""C10 -- SIM STRIP. The rollout itself, on the same time axis as C9.

Two schedules, six frames each, sampled at the SAME episode times so the columns
are comparable. Under each strip runs the event rail from C9 -- release ticks,
inference triangles, the gripper-closed span, the outcome glyph -- with a leader
from each frame down to the instant it was taken.

Frames are extracted locally with ffmpeg from the rollout videos already on disk.
Nothing is fetched and no simulation is run.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import vocab as V

V.style()
ARMS = [("pipe110", "XPU-RT pipelined  ·  period 118 ms", 117.7, 117.6),
        ("cpu685", "QNN baseline  ·  period 685 ms", 684.8, 684.8)]
NF, TICK = 6, 40.0

fig = plt.figure(figsize=(7.16, 3.60))
OUT = 0.052
for k, (arm, lab, lat, per) in enumerate(ARMS):
    age, act, ok = V.age_trace(arm, 0)
    t = np.arange(len(age)) * TICK / 1000.0
    got = V.video_frames(arm, 0, n=NF, src="runs")
    if not got:
        continue
    frames, nfr = got
    top = 0.955 - k * 0.485
    W = (0.945 - OUT) / NF
    for j, (fi, path) in enumerate(frames):
        a = fig.add_axes([OUT + j * W, top - 0.245, W * 0.965, 0.245])
        im = mpimg.imread(path)
        a.imshow(im)
        a.axis("off")
        tt = fi / max(nfr - 1, 1) * t[-1]
        a.set_title(f"{tt:.1f} s", fontsize=6.4, color=V.INK2, pad=1.5)

    # the rail under the strip
    r = fig.add_axes([OUT, top - 0.345, 0.945 - OUT, 0.088])
    r.set_xlim(-0.03, t[-1] + 0.42)
    r.set_ylim(-1.0, 1.25)
    r.axis("off")
    r.plot([0, t[-1]], [0.55, 0.55], color=V.AXIS, lw=0.9)
    rel = np.arange(0, t[-1] * 1000, per) / 1000.0
    V.ev(r, rel, 0.55, "release", ms=5.0)
    V.ev(r, t[V.completions(age)], 0.55, "complete", ms=4.4)
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
        r.add_patch(Rectangle((t[a0], -0.55), t[a1] - t[a0], 0.34, facecolor=V.MUTED,
                              alpha=0.32, linewidth=0))
    if len(on):
        V.ev(r, t[on], -0.38, "grip", ms=5.4)
    V.ev(r, [t[-1] + 0.18], [0.55], "success" if ok else "failure", ms=9)
    for j, (fi, _) in enumerate(frames):
        tt = fi / max(nfr - 1, 1) * t[-1]
        r.plot([tt, tt], [0.55, 1.20], color=V.MUTED, lw=0.6, ls=(0, (2, 2)))
    r.text(-0.02, -0.90, f"{lab}   ·   {json.load(open(V.AGE_RUNS[arm][0] / 'summary.json'))['episodes'][0]['n_inferences']} commands over {t[-1]:.1f} s"
           f"   ·   mean observation age {np.nanmean(age):.0f} ms",
           fontsize=7.2, color=V.INK, fontweight="bold", va="center")

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success"])
           + [Rectangle((0, 0), 1, 1, facecolor=V.MUTED, alpha=0.32,
                        label="gripper closed")],
           loc="lower center", bbox_to_anchor=(0.5, -0.004), ncol=5, handlelength=1.2)
V.save(fig, "cand_c10_sim_strip")
