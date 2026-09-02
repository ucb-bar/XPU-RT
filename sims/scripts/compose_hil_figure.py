#!/usr/bin/env python3
"""Comprehensive HIL warehouse figure: isometric 3D drone views + varied sensor bank (with YOLO
classification) + a large readable K1 schedule Gantt.

Reads record_sensor_demo --dump_figure_data (figure_data.npz + iso_data.npz + snap_*.npz), projects
the drone's world poses into the isometric camera to place it precisely, draws the YOLO boxes on the
FPV snaps, and embeds a big Gantt image. Pure matplotlib — no Isaac.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

YOLO_CLS = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}


def project(K, pos_w, quat_wxyz, pts_world):
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    d = np.asarray(pts_world, np.float64) - np.asarray(pos_w, np.float64)[None, :]
    Xc = d @ R
    zc = np.clip(Xc[:, 2], 1e-6, None)
    u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    return u, v, (Xc[:, 2] > 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gantt", default=None, help="PNG of the K1 schedule Gantt to embed (big)")
    ap.add_argument("--out", default="out/paper_figure_hil")
    ap.add_argument("--n-drones", type=int, default=7)
    ap.add_argument("--label", default="XPU-RT co-design schedule",
                    help="case label for the schedule panel (e.g. 'XPU-RT · greedy', 'ROS + greedy baseline')")
    args = ap.parse_args()

    iso = np.load(os.path.join(args.data_dir, "iso_data.npz"))
    bg = iso["iso_bg"]
    H, W = bg.shape[:2]
    K, cpos, cquat = iso["isoK"], iso["isopos"], iso["isoquat"]
    poses = iso["poses"]                       # (T,7) x,y,z,qw..
    xyz = poses[:, :3]
    pu, pv, pvalid = project(K, cpos, cquat, xyz)

    snaps = sorted(glob.glob(os.path.join(args.data_dir, "snap_*.npz")))
    snaps = [np.load(s) for s in snaps]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    have_g = bool(args.gantt and os.path.exists(args.gantt))
    nrows = 3 if have_g else 2
    hr = [3.2, 1.15, 1.25] if have_g else [3.2, 1.15]
    fig = plt.figure(figsize=(11.0, 4.4 * (nrows / 2.4)))
    gs = fig.add_gridspec(nrows, 6, height_ratios=hr, hspace=0.28, wspace=0.22,
                          left=0.02, right=0.98, top=0.985, bottom=0.02)

    # ---- MAIN: a ROW of isometric FOLLOW snapshots — the drone centered at N moments ----
    cmap = matplotlib.colormaps["viridis"]
    iso_frames, iso_steps = iso["iso"], iso["snap_steps"]
    n = min(args.n_drones, len(iso_frames))
    lo = max(1, int(len(iso_frames) * 0.20))          # skip spawn frames
    sel = np.unique(np.linspace(lo, len(iso_frames) - 1, n).astype(int))
    top = gs[0, :].subgridspec(1, len(sel), wspace=0.05)
    Hh, Ww = iso_frames.shape[1:3]
    cwx, cwy = int(Ww * 0.34), int(Hh * 0.34)          # gentle center crop (drone is centered)
    for col, fi in enumerate(sel):
        ii = int(iso_steps[fi])
        a = fig.add_subplot(top[0, col])
        a.imshow(iso_frames[fi])
        a.set_xlim(Ww / 2 - cwx, Ww / 2 + cwx)
        a.set_ylim(Hh / 2 + cwy, Hh / 2 - cwy)         # inverted y
        for sp in a.spines.values():
            sp.set_edgecolor(cmap(col / max(1, len(sel) - 1))); sp.set_linewidth(2.2)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"t = {ii * 0.02:.1f} s", fontsize=8.5, weight="bold",
                    color=cmap(col / max(1, len(sel) - 1)))
    fig.text(0.015, 0.992, "Onboard-scheduled autonomous flight — isometric snapshots of the drone through the aisle",
             fontsize=10.5, weight="bold", va="top")

    # ---- SENSOR BANK strip: 3× FPV+YOLO at varied times, + cross-ToF, optical flow, IMU/alt ----
    pick = np.unique(np.linspace(max(1, int(len(snaps) * 0.2)), len(snaps) - 1, 3).astype(int))
    steps = iso["snap_steps"]
    for col, si in enumerate(pick):
        sn = snaps[si]
        a = fig.add_subplot(gs[1, col])
        a.imshow(sn["fpv"], cmap="gray", vmin=0, vmax=1)
        for det in sn["det"]:
            c, x0, y0, x1, y1, cf = det
            name, color = YOLO_CLS.get(int(c), ("obj", "#39f"))
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.6))
            a.text(x0, y0 - 1.5, f"{name} {cf:.2f}", color="black", fontsize=6.2, weight="bold",
                   bbox=dict(fc=color, ec="none", pad=0.6), va="bottom")
        a.set_title(f"FPV + YOLOv8n · t={float(steps[si])*0.02:.1f}s", fontsize=7.5)
        a.set_xticks([]); a.set_yticks([])

    mid = snaps[len(snaps) // 2]
    # cross-ToF
    a = fig.add_subplot(gs[1, 3]); tof = mid["tof"]
    grid = np.full((3, 3), np.nan)
    grid[0, 1] = np.nanmean(tof[0]); grid[2, 1] = np.nanmean(tof[2])
    grid[1, 0] = np.nanmean(tof[3]); grid[1, 2] = np.nanmean(tof[1])
    a.imshow(grid, cmap="turbo_r"); a.set_title("cross-ToF (4×VL53L5CX)", fontsize=7.5)
    a.set_xticks([]); a.set_yticks([])
    # optical flow arrow
    a = fig.add_subplot(gs[1, 4]); dx, dy = mid["flow"]
    a.arrow(0, 0, float(dx), float(dy), head_width=0.18, color="#d62728", lw=2)
    m = max(1.0, abs(float(dx)), abs(float(dy))) * 1.3
    a.set_xlim(-m, m); a.set_ylim(-m, m); a.axhline(0, color="0.8", lw=0.6); a.axvline(0, color="0.8", lw=0.6)
    a.set_title("optical flow (PMW3901)", fontsize=7.5); a.set_xticks([]); a.set_yticks([])
    # IMU ω + altitude readout
    a = fig.add_subplot(gs[1, 5]); a.axis("off")
    w = mid["w"]; r, p, y = mid["rpy"]
    a.text(0.5, 0.62, f"IMU ω (rad/s)\nx {w[0]:+.2f}  y {w[1]:+.2f}  z {w[2]:+.2f}",
           ha="center", va="center", fontsize=7.4, family="monospace")
    a.text(0.5, 0.22, f"alt(down-ToF) {float(mid['dtof']):.2f} m\nbaro {float(mid['baro']):.2f} m",
           ha="center", va="center", fontsize=7.4, family="monospace")
    a.set_title("IMU · altitude", fontsize=7.5)

    # ---- GANTT (big, full width) ----
    if have_g:
        ag = fig.add_subplot(gs[2, :]); ag.imshow(plt.imread(args.gantt)); ag.axis("off")
        ag.set_title(f"Onboard K1 schedule — {args.label}   (IME / matrix-engine dispatches darker + hatched)",
                     fontsize=8, weight="bold")

    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png and .pdf")


if __name__ == "__main__":
    main()
