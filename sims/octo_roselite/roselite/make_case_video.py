#!/usr/bin/env python3
"""Turn one recorded RoSE-lite episode into a presentation clip with a title
card and a burned-in caption.

The caption carries everything needed to read the clip on its own: the MEASURED
board latency, the schedule (serial / pipelined), D in control steps, the arm's
MEASURED success rate with n, and whether THIS episode succeeded.

Honesty rules baked in:
  * the harness is not reproducible (identical config + seed gave 13/24 then
    15/24), so a clip is *a* representative episode, never *the* episode. The
    caption says which run and episode index it came from.
  * a success shown from a low-success arm is flagged on the frame via
    ``--note``; nothing here presents an outlier as typical.
  * the int8 pipeline is a PERFORMANCE VEHICLE (3-way chain cos 0.008 vs the
    JAX golden). The policy ran at full fp32 on the host; only the *arrival
    time* of each output is board-derived. The footer says so on every frame.

Usage::

    python make_case_video.py --video runs_video/lat283_serial/ep03_success_False.mp4 \
        --out videos/case_lat283.4ms_serial.mp4 \
        --latency-ms 283.4 --schedule serial --arm-sr 15.8 --arm-n 120 \
        --episode-success False --source "runs_video/lat283_serial ep03"
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BG    = "#11151c"
PANEL = "#1b212b"
FG    = "#e8ecf2"
DIM   = "#8a97a8"
OK    = "#2ecc71"
BAD   = "#e74c3c"
WARN  = "#f39c12"

FOOTER = (
    "MEASURED  board latency, arm success rate and n, and the rollout itself.        "
    "MODELLED  latency → D = ceil(latency / 200 ms) control steps.\n"
    "The int8 QRB5165 pipeline is a PERFORMANCE VEHICLE — the 3-way chain measures cos 0.008 against the "
    "JAX golden. The policy in this rollout ran at full fp32 on the host;\nonly the ARRIVAL TIME of each action "
    "chunk is board-derived. The harness is not reproducible, so this is a representative episode, not a "
    "reproducible one."
)


def read_frames(video: Path) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--latency-ms", type=float, required=True)
    p.add_argument("--latency-note", default="", help="where the latency came from")
    p.add_argument("--schedule", choices=["serial", "pipelined"], required=True)
    p.add_argument("--arm-sr", type=float, required=True)
    p.add_argument("--arm-n", type=int, required=True)
    p.add_argument("--episode-success", default="False")
    p.add_argument("--source", default="", help="run dir + episode this came from")
    p.add_argument("--headline", default="", help="one-line claim for the title card")
    p.add_argument("--note", default="", help="extra honesty line, shown on every frame")
    p.add_argument("--n-avg", type=int, default=-1,
                   help="predictions averaged per step (pipelined arms only)")
    p.add_argument("--control-period-ms", type=float, default=200.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--title-card-s", type=float, default=4.0)
    p.add_argument("--tail-s", type=float, default=1.8)
    p.add_argument("--dpi", type=int, default=110)
    args = p.parse_args()

    ok = str(args.episode_success).lower() in ("1", "true", "yes")
    period = args.control_period_ms
    D = max(int(np.ceil(args.latency_ms / period - 1e-9)), 0)
    frames = read_frames(args.video)
    n_steps = len(frames) - 1
    sched = ("SERIAL — one inference in flight (the deployable schedule: "
             "MEASURED max_concurrent_lanes = 1)" if args.schedule == "serial" else
             "PIPELINED — a fresh inference every 200 ms. MODELLED control "
             "strategy, NOT a hardware schedule;\n            no pipelined Octo "
             "schedule has ever run on the board.")

    # ------------------------------------------------------------ title card
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, "RoSE-lite  —  Octo-small on SIMPLER under MEASURED QRB5165 latency",
            ha="center", va="top", fontsize=20, color=DIM, family="DejaVu Sans")
    ax.text(0.5, 0.845,
            f"{args.latency_ms:g} ms  •  {args.schedule.upper()}  •  D = {D} control steps",
            ha="center", va="top", fontsize=42, color=FG, fontweight="bold",
            family="DejaVu Sans")
    ax.text(0.5, 0.735,
            f"arm success rate  {args.arm_sr:.1f}%   (n = {args.arm_n})   MEASURED",
            ha="center", va="top", fontsize=26, color=WARN, family="DejaVu Sans Mono")
    ax.text(0.5, 0.655,
            f"this episode:  {'SUCCEEDED' if ok else 'FAILED'}",
            ha="center", va="top", fontsize=26, color=OK if ok else BAD,
            family="DejaVu Sans Mono", fontweight="bold")
    body = (
        f"task        widowx_put_eggplant_in_basket   (SIMPLER, control_freq = 5 → one env step = 200 ms)\n"
        f"policy      octo-small-1.0, full fp32 on the host GPU, corrected masked unnormalization\n"
        f"latency     {args.latency_ms:g} ms MEASURED on the QRB5165"
        f"{('  —  ' + args.latency_note) if args.latency_note else ''}\n"
        f"            MODELLED as D = ceil({args.latency_ms:g} / 200) = {D} control steps of delay\n"
        f"schedule    {sched}"
        + (f"\n#averaged   {args.n_avg} prediction(s) target each step"
           if args.n_avg >= 0 else "")
        + (f"\nsource      {args.source}" if args.source else ""))
    ax.text(0.085, 0.545, body, ha="left", va="top", fontsize=14.5, color=FG,
            family="DejaVu Sans Mono", linespacing=1.85)
    if args.headline:
        ax.text(0.5, 0.255, "\n".join(textwrap.wrap(args.headline, 96)),
                ha="center", va="top", fontsize=17, color=WARN,
                family="DejaVu Sans", linespacing=1.6, style="italic")
    ax.text(0.5, 0.075, FOOTER.replace("\n", " "), ha="center", va="top",
            fontsize=9.4, color=DIM, family="DejaVu Sans", linespacing=1.7,
            wrap=True)
    fig.canvas.draw()
    card = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    # ------------------------------------------------------------ rollout
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.92], wspace=0.04,
                          left=0.035, right=0.975, top=0.885, bottom=0.175)
    ax_v = fig.add_subplot(gs[0]); ax_v.axis("off"); ax_v.set_facecolor(BG)
    im = ax_v.imshow(frames[0])
    ax_t = fig.add_subplot(gs[1]); ax_t.axis("off")
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    ax_t.add_patch(Rectangle((0, 0), 1, 1, facecolor=PANEL, edgecolor="#2b3441",
                             lw=1.0, zorder=0))

    fig.suptitle(
        f"MEASURED {args.latency_ms:g} ms latency  —  {args.schedule.upper()} schedule  "
        f"—  D = {D} control steps",
        fontsize=19.5, color=FG, y=0.962, fontweight="bold")

    ax_t.text(0.5, 0.955, f"arm SR  {args.arm_sr:.1f}%   (n = {args.arm_n})",
              ha="center", va="top", fontsize=25, color=WARN,
              family="DejaVu Sans Mono", zorder=3)
    ax_t.text(0.5, 0.845, f"this episode: {'SUCCEEDED' if ok else 'FAILED'}",
              ha="center", va="top", fontsize=25, color=OK if ok else BAD,
              family="DejaVu Sans Mono", fontweight="bold", zorder=3)

    panel = (
        f"── MEASURED ─────────────────────────\n"
        f" latency    {args.latency_ms:g} ms on the QRB5165\n"
        f" arm SR     {args.arm_sr:.1f}%  over n = {args.arm_n} episodes\n"
        f"── MODELLED ─────────────────────────\n"
        f" D          {D} control steps  ({D * period:.0f} ms)\n"
        f" schedule   {args.schedule}\n"
        + (f" averaged   {args.n_avg} prediction(s) per step\n" if args.n_avg >= 0 else "")
        + f"── THIS EPISODE ─────────────────────\n"
        + f" task       put eggplant in basket\n"
        + f" policy     octo-small-1.0, fp32 on host\n"
        # the panel is ~60 monospace columns wide; wrap rather than overflow it
        + "".join(f" {'source' if i == 0 else '      '}    {ln}\n"
                  for i, ln in enumerate(textwrap.wrap(args.source, 46))))
    ax_t.text(0.055, 0.70, panel, ha="left", va="top", fontsize=13.2, color=FG,
              family="DejaVu Sans Mono", linespacing=1.72, zorder=3)

    clock = ax_t.text(0.055, 0.135, "", ha="left", va="top", fontsize=14.5,
                      color=DIM, family="DejaVu Sans Mono", zorder=3)

    if args.note:
        fig.text(0.5, 0.147, args.note, ha="center", va="bottom", fontsize=12.5,
                 color=WARN, family="DejaVu Sans", fontweight="bold")
    fig.text(0.5, 0.018, FOOTER, ha="center", va="bottom", fontsize=9.0,
             color=DIM, family="DejaVu Sans", linespacing=1.8)

    # Progress bar drawn in the image's own coordinates, so it tracks the
    # rollout exactly instead of the aspect-fitted axes box.
    h, w = frames[0].shape[:2]
    ax_v.add_patch(Rectangle((0, h - 10), w, 10, facecolor="#000000",
                             alpha=0.55, zorder=4))
    bar = ax_v.add_patch(Rectangle((0, h - 10), 0, 10,
                                   facecolor=OK if ok else "#e8ecf2",
                                   alpha=0.9, edgecolor="none", zorder=5))

    import imageio.v2 as imageio
    writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264",
                                quality=9, macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for _ in range(int(args.title_card_s * args.fps)):
            writer.append_data(card)
        # rollout plays at real time: one env step is 200 ms of wall clock.
        hold = max(int(round(args.fps * period / 1000.0)), 1)
        for k, frame in enumerate(frames):
            im.set_data(frame)
            bar.set_width(w * k / max(n_steps, 1))
            clock.set_text(f"env step {k:3d} / {n_steps}\n"
                           f"t = {k * period / 1000.0:6.2f} s")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            for _ in range(hold):
                writer.append_data(buf)
        for _ in range(int(args.tail_s * args.fps)):
            writer.append_data(buf)
    finally:
        writer.close()
        plt.close(fig)

    print(f"[ok] {args.out}  ({n_steps} steps, "
          f"{args.title_card_s + n_steps * period / 1000.0 + args.tail_s:.1f} s)")


if __name__ == "__main__":
    main()
