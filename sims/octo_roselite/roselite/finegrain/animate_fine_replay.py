#!/usr/bin/env python3
"""Replay a fine-grain rollout with the per-tick staleness drawn as it happened.

The coarse harness stepped at 200 ms, so "the robot is running on stale data"
could only ever be drawn at 200 ms granularity -- and at D=2 the robot was
almost never stale in the first place.  Here the env steps every 40 ms and the
policy only updates when the MODELLED board latency says a result landed, so
the zero-order hold is a per-tick fact and is drawn as one.

NOTHING HERE IS RE-DERIVED FROM A MODEL.  The freshness of each tick is read
back out of the run's own ``ep*_action_age_ms.npy``: the age of the observation
behind the applied action grows by exactly one tick during a hold and DROPS on
the tick a new result lands, so

    fresh(t)  <=>  age[t] < age[t-1]        (plus the first landing)

recovers the harness's own dispatch grid exactly.  The applied action stream is
read from ``ep*_applied_actions.npy`` and is literally constant across a hold,
which is the point.

Usage:
  python animate_fine_replay.py --run runs_video/vid_serial283 --ep 0 \
      --out videos/fine_283.4ms_serial.mp4 --latency-ms 283.4 --cadence-ms 283.4 \
      --arm-sr 23.6 --arm-n 72
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import mediapy as media

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="a runs_video/<arm> directory")
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--latency-ms", type=float, required=True,
                help="MEASURED QRB5165 per-inference latency for this arm")
ap.add_argument("--cadence-ms", type=float, required=True,
                help="MEASURED cadence at which a fresh result arrives")
ap.add_argument("--arm-sr", type=float, required=True)
ap.add_argument("--arm-n", type=int, required=True)
ap.add_argument("--config-label", default="")
ap.add_argument("--headline", default="")
ap.add_argument("--note", default="")
ap.add_argument("--tick-ms", type=float, default=40.0)
ap.add_argument("--window-ticks", type=int, default=60)
ap.add_argument("--fps", type=float, default=25.0)
ap.add_argument("--title-card-s", type=float, default=3.5)
ap.add_argument("--dpi", type=int, default=110)
args = ap.parse_args()

RUN = Path(args.run)
TICK = args.tick_ms
age = np.load(RUN / f"ep{args.ep:02d}_action_age_ms.npy")
act = np.load(RUN / f"ep{args.ep:02d}_applied_actions.npy")
vids = sorted(glob.glob(str(RUN / f"ep{args.ep:02d}_success_*.mp4")))
assert vids, f"no video for ep{args.ep:02d} in {RUN}"
success = "True" in os.path.basename(vids[0])
frames = media.read_video(vids[0])

n = min(len(age), len(frames) - 1, len(act))
age, act, frames = age[:n], act[:n], frames[:n + 1]

# ---- recover the harness's own dispatch grid from the recorded ages --------
fresh = np.zeros(n, dtype=bool)
prev = np.inf
for t in range(n):
    a = age[t]
    if np.isnan(a):
        prev = np.inf
        continue
    if a < prev:
        fresh[t] = True
    prev = a
hold = (~fresh) & ~np.isnan(age)
initial_hold = int(np.isnan(age).sum())

print(f"[{RUN.name} ep{args.ep:02d}] {n} ticks = {n*TICK/1000:.1f} s, success={success}")
print(f"  fresh ticks {fresh.sum()} ({100*fresh.mean():.1f}%), held ticks {hold.sum()} "
      f"({100*hold.mean():.1f}%), initial dead ticks {initial_hold}")
_ok = age[~np.isnan(age)]
print(f"  observation age at actuation: mean {_ok.mean():.0f} ms, max {_ok.max():.0f} ms")

# sanity: during a hold the applied action must not change
_chg = 0
for t in range(1, n):
    if hold[t] and not np.allclose(act[t], act[t - 1]):
        _chg += 1
print(f"  applied-action changes on a HELD tick: {_chg} (must be 0 for a true ZOH)")

FRESH_C, HOLD_C, DEAD_C = "#1a9850", "#d73027", "#8c8c8c"
BG = "#101216"

fig = plt.figure(figsize=(12.6, 7.4), dpi=args.dpi, facecolor=BG)
gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.55, 0.75],
                      hspace=0.42, wspace=0.16,
                      left=0.045, right=0.985, top=0.87, bottom=0.075)
ax_img = fig.add_subplot(gs[:, 0])
ax_strip = fig.add_subplot(gs[0, 1])
ax_age = fig.add_subplot(gs[1, 1])
ax_act = fig.add_subplot(gs[2, 1])
for a in (ax_strip, ax_age, ax_act):
    a.set_facecolor("#181b21")
    for s in a.spines.values():
        s.set_color("#3a3f4a")
    a.tick_params(colors="#c8ccd4", labelsize=7.5)
    a.xaxis.label.set_color("#c8ccd4")
    a.yaxis.label.set_color("#c8ccd4")
ax_img.set_facecolor(BG); ax_img.axis("off")

im = ax_img.imshow(frames[0])
cad_note = (f"cadence {args.cadence_ms:.0f} ms" if abs(args.cadence_ms - args.latency_ms) > 1
            else "serial, one in flight")
fig.suptitle(
    f"{args.config_label or RUN.name}   |   MEASURED latency {args.latency_ms:.1f} ms, "
    f"{cad_note}   |   MODELLED onto a {TICK:.0f} ms control tick\n"
    f"arm success {args.arm_sr:.1f}% (n={args.arm_n}) - this episode: "
    f"{'SUCCESS' if success else 'FAILURE'}",
    color="#f0f2f5", fontsize=11.5, y=0.975)

W = args.window_ticks
ax_strip.set_title("every 40 ms tick: fresh result vs zero-order hold on stale data",
                   color="#e6e9ee", fontsize=9, pad=6)
ax_strip.set_ylim(0, 1); ax_strip.set_yticks([])
ax_strip.set_xlabel("tick")
ax_age.set_ylabel("obs age (ms)")
ax_age.set_xlabel("tick")
ax_act.set_ylabel("applied action")
ax_act.set_xlabel("tick")

strip_patches = []
age_line, = ax_age.plot([], [], color="#4da6ff", lw=1.8)
age_pts = ax_age.scatter([], [], s=14, color=FRESH_C, zorder=5)
act_lines = [ax_act.plot([], [], lw=1.5, color=c, label=l)[0]
             for c, l in (("#ff9f40", "dx"), ("#4dd2ff", "dy"), ("#c792ea", "dz"))]
grip_line, = ax_act.plot([], [], lw=1.5, color="#ffd166", label="grip")
ax_act.legend(fontsize=6.5, ncol=4, loc="upper right", facecolor="#181b21",
              edgecolor="#3a3f4a", labelcolor="#c8ccd4")

txt = ax_img.text(0.015, 0.985, "", transform=ax_img.transAxes, va="top", ha="left",
                  color="#f0f2f5", fontsize=9.5, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.45", fc="#000000cc", ec="#3a3f4a"))
foot = fig.text(0.045, 0.017, args.note, color="#9aa2ad", fontsize=7.8, ha="left")

age_max = float(np.nanmax(age)) if np.isfinite(np.nanmax(age)) else 1.0
ax_age.set_ylim(-0.05 * age_max, 1.18 * age_max)
amax = float(np.abs(act[:, :3]).max()) or 1.0
ax_act.set_ylim(-1.25, 1.25)

n_title = int(round(args.title_card_s * args.fps))
writer = FFMpegWriter(fps=args.fps, bitrate=3600,
                      metadata=dict(artist="RoSE-lite finegrain"))
os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)


def draw_title():
    for p in strip_patches:
        p.remove()
    strip_patches.clear()
    im.set_data(frames[0])
    txt.set_text(args.headline or "")
    for L in act_lines + [grip_line]:
        L.set_data([], [])
    age_line.set_data([], [])


with writer.saving(fig, args.out, args.dpi):
    if args.headline:
        draw_title()
        for _ in range(n_title):
            writer.grab_frame(facecolor=BG)

    for t in range(n):
        im.set_data(frames[t + 1])
        lo = max(0, t - W + 1)
        hi = t + 1
        for p in strip_patches:
            p.remove()
        strip_patches.clear()
        for k in range(lo, hi):
            if np.isnan(age[k]):
                c, h = DEAD_C, 0.55
            elif fresh[k]:
                c, h = FRESH_C, 1.0
            else:
                c, h = HOLD_C, 0.55
            r = Rectangle((k - 0.45, 0), 0.9, h, color=c,
                          alpha=1.0 if k == t else 0.82)
            ax_strip.add_patch(r)
            strip_patches.append(r)
        ax_strip.set_xlim(lo - 0.6, lo + W - 0.4)

        xs = np.arange(lo, hi)
        ax_age.set_xlim(lo - 0.6, lo + W - 0.4)
        ax_act.set_xlim(lo - 0.6, lo + W - 0.4)
        aw = age[lo:hi]
        age_line.set_data(xs, np.where(np.isnan(aw), 0.0, aw))
        fx = xs[fresh[lo:hi]]
        age_pts.set_offsets(np.c_[fx, age[fx]] if len(fx) else np.empty((0, 2)))
        for i, L in enumerate(act_lines):
            L.set_data(xs, act[lo:hi, i] / amax)
        grip_line.set_data(xs, act[lo:hi, 6])

        held_run = 0
        k = t
        while k >= 0 and hold[k]:
            held_run += 1
            k -= 1
        a = age[t]
        state = ("DEAD - nothing has landed yet" if np.isnan(a)
                 else "FRESH result applied" if fresh[t]
                 else f"HOLDING stale action ({held_run} ticks)")
        txt.set_text(
            f"t = {t*TICK:7.0f} ms   tick {t:3d}/{n}\n"
            f"{state}\n"
            f"obs age  = {'--' if np.isnan(a) else f'{a:.0f} ms'}\n"
            f"held so far {100*hold[:t+1].mean():.0f}% of ticks")
        writer.grab_frame(facecolor=BG)

print(f"-> {args.out}")
