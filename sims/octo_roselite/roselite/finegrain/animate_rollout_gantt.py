#!/usr/bin/env python3
"""Rollout on top, MEASURED QRB5165 lane schedule below -- forest-trail layout.

Layout matches sims/smolvla_demo/animate_schedule_gantt.py (itself the forest-trail
one): a 2-row gridspec, height_ratios [2.05, 1.0], rollout above, hardware lane
gantt below, with a red playhead sweeping it in sim time.

WHAT IS MEASURED AND WHAT IS CONSTRUCTED
  * the rollout            MEASURED in sim under the MODELLED board latency
  * the per-tick hold      MEASURED -- read back from the run's own
                           ep*_action_age_ms.npy (age drops exactly on the tick a
                           new result lands, so freshness is recovered, not modelled)
  * the lane gantt         MEASURED per-dispatch trace from the QRB5165
                           (AGENTS_QNN_TRACE block of the runtime log). One
                           inference's dispatches are taken as the template and
                           tiled at the arm's MEASURED issue cadence, so the
                           overlap you see for the pipelined arms is the real
                           measured concurrency, not a drawing.

Usage:
  python animate_rollout_gantt.py --video <ep.mp4> --run <run dir> --ep 0 \
      --trace <runtime.log> --latency-ms 283.4 --cadence-ms 283.4 \
      --arm-label "3-way CPU+DSP+HTA, serial" --env-label "eggplant" \
      --arm-sr 17.1 --out out.mp4
"""
from __future__ import annotations
import argparse, csv, io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Patch
import mediapy as media

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--trace", required=True, help="QRB5165 runtime log with a trace block")
ap.add_argument("--latency-ms", type=float, required=True)
ap.add_argument("--cadence-ms", type=float, required=True)
ap.add_argument("--arm-label", required=True)
ap.add_argument("--env-label", required=True)
ap.add_argument("--arm-sr", type=float, required=True)
ap.add_argument("--success", default="")
ap.add_argument("--tick-ms", type=float, default=40.0)
ap.add_argument("--window-ms", type=float, default=None,
                help="gantt window. Default: max(700, 2.2*cadence) so a few "
                     "inferences are visible without becoming a barcode.")
ap.add_argument("--fps", type=float, default=25.0)
ap.add_argument("--dpi", type=int, default=110)
ap.add_argument("--out", required=True)
args = ap.parse_args()

LANES = ["CPU", "DSP", "HTA"]
LANE_MACHINE = {"CPU": "CPU_X#0", "DSP": "CPU_E#0", "HTA": "CPU_P#0"}
INST_COLOUR = ["#8e44ad", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
               "#16a085", "#d35400", "#7f5fa0", "#b7950b", "#2c3e50"]
BG = "#ffffff"


def parse_trace(p: Path):
    body = p.read_text(errors="replace")
    body = body.split("AGENTS_QNN_TRACE_BEGIN", 1)[1].split("AGENTS_QNN_TRACE_END", 1)[0]
    raw = body.splitlines()
    i = next(j for j, l in enumerate(raw) if l.startswith("seg_id,"))
    return list(csv.DictReader(io.StringIO("\n".join(
        l for l in raw[i:] if l.strip() and not l.lstrip().startswith("===")))))


# ---- build the one-inference template from the measured trace ---------------
rows = parse_trace(Path(args.trace))
insts = sorted({r.get("instance", "0") for r in rows})
tpl = [r for r in rows if r.get("instance", "0") == insts[0]
       and (r.get("actual_backend") or "").strip() in LANES]
t0 = min(float(r["actual_start_ms"]) for r in tpl)
TEMPLATE = [((r.get("actual_backend") or "").strip(),
             float(r["actual_start_ms"]) - t0,
             float(r["actual_end_ms"]) - t0) for r in tpl]
TPL_SPAN = max(e for _, _, e in TEMPLATE)

WINDOW_MS = args.window_ms if args.window_ms else max(700.0, 2.2 * args.cadence_ms)

frames = media.read_video(args.video)
n_frames = len(frames)
EP_MS = n_frames * args.tick_ms

age = np.load(Path(args.run) / f"ep{args.ep:02d}_action_age_ms.npy")
fresh = np.zeros(len(age), dtype=bool)
prev = np.inf
for i, a in enumerate(age):                      # age DROPS on the tick a result lands
    if not np.isnan(a) and a < prev:
        fresh[i] = True
    prev = a if not np.isnan(a) else prev

# ---- tile the template at the measured cadence across the episode -----------
BARS = []                                        # (lane, start_ms, end_ms, instance)
j = 0
while j * args.cadence_ms < EP_MS:
    off = j * args.cadence_ms
    for lane, s, e in TEMPLATE:
        BARS.append((lane, off + s, off + e, j))
    j += 1
N_INST = j
max_conc = 1 + int(TPL_SPAN // args.cadence_ms) if args.cadence_ms > 0 else 1

fig = plt.figure(figsize=(12.8, 9.6), dpi=args.dpi, facecolor=BG)
gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.0], hspace=0.22,
                      left=0.06, right=0.965, top=0.925, bottom=0.085)
ax_img = fig.add_subplot(gs[0]); ax_img.axis("off")
im = ax_img.imshow(frames[0])
status = ax_img.text(0.5, -0.045, "", transform=ax_img.transAxes, ha="center",
                     va="top", fontsize=12, family="monospace")

ax = fig.add_subplot(gs[1])
for row, lane in enumerate(LANES):
    ax.axhspan(row - 0.42, row + 0.42, color="#f4f6f6", zorder=0)
ax.set_yticks(range(len(LANES)))
ax.set_yticklabels([f"{l}\n({LANE_MACHINE[l]})" for l in LANES], fontsize=9)
ax.set_ylim(-0.65, len(LANES) - 0.35)
ax.set_xlabel("sim time (ms)  —  MEASURED QRB5165 dispatches, tiled at the measured cadence",
              fontsize=9.5)
ax.grid(True, axis="x", alpha=0.3, zorder=0)
playhead = ax.axvline(0, color="red", lw=2.0, alpha=0.9, zorder=6)

fig.suptitle(
    f"Octo-small · {args.env_label} · {args.arm_label}\n"
    f"MEASURED board latency {args.latency_ms:g} ms, cadence {args.cadence_ms:g} ms  "
    f"→  up to {max_conc} inference(s) in flight   |   arm success rate {args.arm_sr:.1f}%",
    fontsize=12.5, y=0.985)

ax.legend(handles=[Patch(facecolor=INST_COLOUR[i % len(INST_COLOUR)], edgecolor="black",
                         label="successive inferences" if i == 0 else "")
                   for i in range(min(max_conc, 5))],
          loc="upper right", fontsize=8, ncol=min(max_conc, 5), framealpha=0.92,
          handletextpad=0.4, columnspacing=0.5,
          title=f"colour cycles per inference · {max_conc} concurrent",
          title_fontsize=8)

# Create every bar ONCE. Recreating them per frame is what made this take ~16 s
# per frame (3408 patches for the pipe110 trace); per-frame work is now just an
# alpha flip on the bars, which is cheap.
from matplotlib.patches import Rectangle
bar_patches = []
for lane, s, e, inst in BARS:
    r = Rectangle((s, LANES.index(lane) - 0.34), e - s, 0.68,
                  facecolor=INST_COLOUR[inst % len(INST_COLOUR)],
                  edgecolor="black", linewidth=0.25, zorder=3, alpha=0.28)
    ax.add_patch(r)
    bar_patches.append((r, s))

writer = FFMpegWriter(fps=args.fps, bitrate=3800)
with writer.saving(fig, args.out, dpi=args.dpi):
    for k in range(n_frames):
        t = k * args.tick_ms
        im.set_data(frames[k])
        lo, hi = t - 0.35 * WINDOW_MS, t + 0.65 * WINDOW_MS
        ax.set_xlim(lo, hi)
        for r, s in bar_patches:
            r.set_alpha(1.0 if s <= t else 0.28)
        playhead.set_xdata([t, t])
        a = age[k] if k < len(age) else float("nan")
        f = fresh[k] if k < len(fresh) else False
        status.set_text(
            f"t = {t/1000:6.2f} s      observation age at actuation = "
            f"{0 if np.isnan(a) else a:6.0f} ms      "
            f"{'NEW RESULT' if f else 'HOLDING stale action'}")
        status.set_color("#1a9e8f" if f else "#c0392b")
        writer.grab_frame()
plt.close(fig)
print(f"[ok] {args.out}  ({n_frames} frames, {EP_MS/1000:.1f} s, "
      f"template {len(TEMPLATE)} dispatches spanning {TPL_SPAN:.1f} ms, "
      f"{N_INST} inferences tiled, up to {max_conc} in flight)")
