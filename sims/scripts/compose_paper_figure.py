#!/usr/bin/env python3
"""Compose the single publication figure (Workstream G) from record_sensor_demo --dump_figure_data.

A stroboscopic overhead view: the drone rendered at N sampled poses along its flight path (time →
color), gates marked, plus a strip of the relevant onboard-sensor snapshots. Vector PDF + PNG at
paper column width. No Isaac needed — pure matplotlib over the dumped .npz.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import font_manager  # noqa: F401


def project(K, pos_w, quat_wxyz, pts_world):
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    d = np.asarray(pts_world, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)[None, :]
    Xc = d @ R
    zc = np.clip(Xc[:, 2], 1e-6, None)
    u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    return u, v, (Xc[:, 2] > 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir from record_sensor_demo --dump_figure_data")
    ap.add_argument("--out", default="out/paper_figure")
    ap.add_argument("--n-drones", type=int, default=8, help="number of drone instances to draw")
    ap.add_argument("--n-snaps", type=int, default=4, help="number of sensor snapshots in the strip")
    ap.add_argument("--schedule-gantt", default=None,
                    help="PNG of the K1 schedule Gantt to embed as the bottom panel, so the "
                         "figure SHOWS the onboard schedule (IME dispatches darker+hatched) that "
                         "the title claims — the co-design half of the flight.")
    args = ap.parse_args()

    d = np.load(os.path.join(args.data_dir, "figure_data.npz"))
    poses = d["poses"]                       # (T, 7) x,y,z,qw,qx,qy,qz
    ov_bg = d["ov_bg"]
    K, pos, quat = d["ovK"], d["ovpos"], d["ovquat"]
    gates = d["gates_world"]
    H, W = ov_bg.shape[:2]

    xyz = poses[:, :3]
    pu, pv, pvalid = project(K, pos, quat, xyz)
    gu, gv, gvalid = project(K, pos, quat, gates)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    has_gantt = bool(args.schedule_gantt and os.path.exists(args.schedule_gantt))
    if has_gantt:
        fig = plt.figure(figsize=(7.0, 5.7))
        gs = fig.add_gridspec(3, args.n_snaps, height_ratios=[3.1, 1.0, 1.15],
                              hspace=0.22, wspace=0.12,
                              left=0.02, right=0.98, top=0.99, bottom=0.02)
    else:
        fig = plt.figure(figsize=(7.0, 4.6))
        gs = fig.add_gridspec(2, args.n_snaps, height_ratios=[3.1, 1.0], hspace=0.18, wspace=0.12,
                              left=0.02, right=0.98, top=0.99, bottom=0.02)

    # ---- main: stroboscopic overhead ----
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(ov_bg)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    # full path
    ok = pvalid
    ax.plot(pu[ok], pv[ok], color="white", lw=3.2, alpha=0.85, zorder=3)
    ax.plot(pu[ok], pv[ok], color="#d62728", lw=1.6, alpha=0.95, zorder=4)
    # gates
    for i in range(len(gates)):
        if gvalid[i]:
            ax.add_patch(Circle((gu[i], gv[i]), 16, fill=False, ec="#ffd400", lw=2.2, zorder=5))
            ax.text(gu[i], gv[i] - 22, f"G{i+1}", color="#ffd400", fontsize=9, weight="bold",
                    ha="center", va="bottom", zorder=6)
    # N drone instances, time -> color
    idx = np.linspace(0, len(poses) - 1, args.n_drones).astype(int)
    cmap = matplotlib.colormaps["viridis"]
    for k, ii in enumerate(idx):
        if not pvalid[ii]:
            continue
        # heading in screen space from local path tangent
        j = min(ii + 3, len(poses) - 1)
        hx, hy = pu[j] - pu[ii], pv[j] - pv[ii]
        ang = np.arctan2(hy, hx) if (abs(hx) + abs(hy)) > 1e-3 else 0.0
        s = 12.0
        tri = np.array([[s, 0], [-s * 0.6, s * 0.6], [-s * 0.6, -s * 0.6]])
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        tri = tri @ rot.T + np.array([pu[ii], pv[ii]])
        col = cmap(k / max(1, args.n_drones - 1))
        ax.add_patch(Polygon(tri, closed=True, fc=col, ec="white", lw=1.0, zorder=7))
    # time colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.020, pad=0.006, ticks=[0, 1])
    cb.ax.set_yticklabels(["start", "end"], fontsize=7)
    cb.set_label("flight time", fontsize=7)
    ax.text(0.008, 0.97, "Onboard-scheduled autonomous flight through the warehouse aisle",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, weight="bold", color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))

    # ---- sensor strip ----
    snap_files = sorted(glob.glob(os.path.join(args.data_dir, "snap_*.npz")))
    pick = np.linspace(0, len(snap_files) - 1, min(args.n_snaps, len(snap_files))).astype(int)
    steps = d["snap_steps"]
    for col, pi in enumerate(pick):
        sn = np.load(snap_files[pi])
        axs = fig.add_subplot(gs[1, col])
        if col == 0:
            # ToF cross (mean over 8x8 per direction -> a + glyph) for variety
            tof = sn["tof"]
            grid = np.full((3, 3), np.nan)
            grid[0, 1] = np.nanmean(tof[0]); grid[2, 1] = np.nanmean(tof[2])
            grid[1, 0] = np.nanmean(tof[3]); grid[1, 2] = np.nanmean(tof[1])
            axs.imshow(grid, cmap="turbo_r")
            axs.set_title("cross-ToF", fontsize=7)
        else:
            axs.imshow(sn["fpv"], cmap="gray", vmin=0, vmax=1)
            axs.set_title(f"FPV t={steps[pi]*0.02:.1f}s", fontsize=7)
        axs.set_xticks([]); axs.set_yticks([])

    # ---- onboard K1 schedule (the co-design half the title claims) ----
    if has_gantt:
        axg = fig.add_subplot(gs[2, :])
        gimg = plt.imread(args.schedule_gantt)
        axg.imshow(gimg)
        axg.axis("off")
        axg.set_title("Onboard K1 XPU-RT schedule — IME (matrix-engine) dispatches darker + hatched",
                      fontsize=7.5, weight="bold")

    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png", "and .pdf")


if __name__ == "__main__":
    main()
