#!/usr/bin/env python3
"""D3 -- SCHEDULE -> BEHAVIOUR, one page. C10v6 over A1v2, one shared key.

Read it downward and the causal chain is the layout: what the robot did (top),
then the board schedule that produced it (bottom). Both halves speak the same
vocabulary -- the release tick, the completion triangle, the wall-clock rail --
so an instant carries from one to the other without a caption explaining that it
does. The key is stated once at the top and governs both halves; neither panel
repeats it.

The two halves run on different clocks ON PURPOSE, and the axes say so: the
strip is EPISODE time in seconds, the gantt is SCHEDULE time in milliseconds
over one baseline inference. That ratio -- 12 s of robot against 700 ms of
silicon -- is the reason the schedule matters at all.

TOP    spoon on towel, episode 21, a MATCHED pair: same initial scene, same
       object pose, only the schedule differs. 7.0 s success against a 12.0 s
       failure. Spoon rather than eggplant because a failure never terminates
       early -- it runs the full task horizon, which on eggplant is 24.0 s and
       makes the pairing 4.3x lopsided; spoon's horizon is 12.0 s.
BOTTOM three measured QRB5165 schedules on one 700 ms window. Dispatches are
       shaded by the parity of the inference they belong to, so where two
       shades interleave inside a lane, two inferences are in flight.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import to_rgb
import vocab as V

V.style()
TICK = 40.0
# ---- top: the rollouts ----
EP, NF, TMAX_S = 21, 8, 12.4
STRIP = [("pipe110", "XPU-RT pipelined", 117.6),
         ("cpu685", "QNN baseline", 684.8)]
# ---- bottom: the board ----
WINDOW = 700.0
# Wrapped by hand: these ride in the left margin now, not above the panel.
ROWS = [("QNN CPU\nbaseline", "mono"),
        ("XPU-RT\nserial", "serial"),
        ("XPU-RT pipelined\np150/w300", "p150w300")]

X0, X1 = 0.030, 0.994
SPAN = X1 - X0


def tint(c, f=0.55):
    s = to_rgb(V.SURFACE)
    return tuple(x + (y - x) * f for x, y in zip(to_rgb(c), s))


def bracket(ax, y, x0, x1, text, *, cap=0.19, fs=6.2, gap=0.30):
    mid, half = (x0 + x1) / 2, (x1 - x0) * gap / 2
    ax.plot([x0, x0, mid - half], [y - cap, y, y], color=V.INK2, lw=0.75,
            zorder=7, clip_on=False, solid_capstyle="butt")
    ax.plot([mid + half, x1, x1], [y, y, y - cap], color=V.INK2, lw=0.75,
            zorder=7, clip_on=False, solid_capstyle="butt")
    ax.text(mid, y, text, ha="center", va="center", fontsize=fs, color=V.INK2,
            zorder=8, clip_on=False, bbox=dict(fc=V.SURFACE, ec="none", pad=1.4))


# ---------------------------------------------------------------- geometry
# Laid out in INCHES and converted, because the frame axes must match the video's
# 4:3 aspect exactly. A square-ish axes holding a 4:3 image pads it top and
# bottom -- that padding was the gap between the images and their leader lines,
# and it cost 0.21 in per row for nothing.
FIG_W = 7.16
ASPECT = 0.75                      # cached frames are 384x288
FW = SPAN / NF
frame_w = FW * 0.976 * FIG_W
frame_h = frame_w * ASPECT
band_h = 0.302
gap_rows, sax_gap, sax_lab, seam = 0.060, 0.020, 0.300, 0.100
panel_h, gantt_bot, legend_h = 0.500, 0.400, 0.200
gantt_region = 3.44 * panel_h      # 3 panels + 2 gaps at hspace 0.22

FIG_H = (legend_h + 2 * (frame_h + band_h) + gap_rows + sax_gap + sax_lab
         + seam + gantt_region + gantt_bot)
fig = plt.figure(figsize=(FIG_W, FIG_H))
FRAME_H, BAND_H = frame_h / FIG_H, band_h / FIG_H
PITCH = (frame_h + band_h + gap_rows) / FIG_H
TOP0 = 1.0 - legend_h / FIG_H

# ========================================================= TOP: the rollouts
for k, (arm, lab, per) in enumerate(STRIP):
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

    fig.text(X0 + 0.006, top - 0.011,
             f"{lab}   ·   {per:.0f} ms period   ·   {len(rel)} commands   ·   "
             f"mean latency {np.nanmean(age):.0f} ms   ·   {dur:.1f} s",
             ha="left", va="top", fontsize=6.4, color=V.INK, fontweight="bold",
             zorder=9, bbox=dict(fc="white", ec="none", alpha=0.80, pad=2.0))

    ax = fig.add_axes([X0, top - FRAME_H - BAND_H, SPAN, BAND_H])
    ax.set_xlim(0, TMAX_S); ax.set_ylim(1.0, 0.0); ax.axis("off")
    for j, tt in enumerate(ts):
        xf = (j + 0.488) * FW / SPAN * TMAX_S
        ax.plot([xf, xf, tt, tt], [0.0, 0.26, 0.66, 0.84], color=V.AXIS,
                lw=1.15, zorder=2, clip_on=False, solid_capstyle="butt")
    ax.plot([0, TMAX_S], [0.90, 0.90], color=V.AXIS, lw=1.0, zorder=3)
    V.ev(ax, rel, 1.05, "release", ms=4.6)
    don = rel + per / 1000.0
    V.ev(ax, don[don <= dur], 0.90, "complete", ms=5.2)
    for t in V.gripper_events(act):
        V.ev(ax, t * TICK / 1000.0, 0.90, "grip", ms=6.4)
    V.ev(ax, dur, 0.90, "success" if ok else "failure", ms=11 if ok else 8)

sax_y = TOP0 - PITCH - FRAME_H - BAND_H - sax_gap / FIG_H
sax = fig.add_axes([X0, sax_y, SPAN, 0.001])
sax.set_xlim(0, TMAX_S); sax.set_yticks([])
for sp in ("left", "right", "top"):
    sax.spines[sp].set_visible(False)
sax.spines["bottom"].set_color(V.AXIS)
sax.set_xticks(range(0, 13, 2))
sax.tick_params(labelsize=6.6, colors=V.MUTED, labelcolor=V.INK2, pad=1.5)
sax.set_xlabel("episode time (s)", fontsize=7.0, labelpad=1.5)

# ========================================================= BOTTOM: the board
# No per-panel titles any more, so the rows can sit much closer together.
gs = fig.add_gridspec(3, 1, left=0.190, right=X1,
                      top=sax_y - (sax_lab + seam) / FIG_H,
                      bottom=gantt_bot / FIG_H, hspace=0.22)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
for ax, (title, group) in zip(axes, ROWS):
    _, rows, _ = V.board_median(group)
    iv = V.inferences(rows)
    starts = sorted(s for s, _ in iv.values())
    span = float(np.median([b - a for a, b in iv.values()]))
    per = float(np.median(np.diff(starts))) if len(starts) > 1 else span
    order = {kk: i for i, kk in enumerate(iv)}
    reps = [0.0] if len(starts) > 1 else list(np.arange(0, WINDOW, per))
    for ri, off in enumerate(reps):
        for r in rows:
            lane = r["lane"]
            if lane not in V.LANE:
                continue
            s0, e0 = r["s"] + off, r["e"] + off
            if s0 > WINDOW:
                continue
            kk = order.get(r["inst"], 0) + ri
            base = V.LANE[lane]
            ax.barh(V.LANE_ORDER.index(lane), min(e0, WINDOW) - s0, left=s0,
                    height=0.62, color=base if kk % 2 == 0 else tint(base),
                    linewidth=0, zorder=3)
    two = abs(span - per) > 1.0
    ax.set_ylim(4.70 if two else 3.92, -0.66)
    ax.set_xlim(-6, WINDOW + 6)
    rail = 2.62
    rel = V.release_rail(ax, rail, 0, WINDOW, per, span, drop=0.24, ms=5.4)
    if two:
        bracket(ax, rail + 0.90, rel[0], rel[0] + span, f"latency {span:.0f} ms")
        bracket(ax, rail + 1.66, rel[0] + span, rel[1] + span, f"period {per:.0f} ms")
    else:
        bracket(ax, rail + 0.90, rel[0], rel[0] + per, f"latency = period = {per:.0f} ms")
    n = int(np.sum(np.arange(0, WINDOW, per) + span <= WINDOW))
    ax.set_yticks(range(3)); ax.set_yticklabels(V.LANE_ORDER, fontsize=6.6)
    for t, l in zip(ax.get_yticklabels(), V.LANE_ORDER):
        t.set_color(V.LANE[l]); t.set_fontweight("bold")
    # Title into the left margin, beside the lanes it describes: a title band
    # above every panel is three lines of vertical cost for two lines of text.
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

# One key for both halves.
fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.5, 1.004), ncol=5, fontsize=6.8)
V.save(fig, "cand_d3_cascade_to_behaviour")
