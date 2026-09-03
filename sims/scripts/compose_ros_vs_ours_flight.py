#!/usr/bin/env python3
"""The literal run: same course, same speed — under ROS onboard-scheduling timing the drone's
control command goes stale (refresh capped at 1/12.40 ms) and it CRASHES; under our co-design
schedule (4.89 ms) the command stays fresh and it COMPLETES 4/4.

Overlays the two REAL flight trajectories (from record_sensor_demo --sched_latency_ms runs) on the
top-down aisle: ours (green, reaches the last gate) vs ROS (red, truncated at the crash + ✗ marker).
Pure matplotlib over two figure_data.npz dumps + a clean background.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


def project(K, pos, quat, pts):
    w, x, y, z = [float(v) for v in quat]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    dd = np.asarray(pts, np.float64) - np.asarray(pos, np.float64)[None, :]
    Xc = dd @ R
    zc = np.clip(Xc[:, 2], 1e-6, None)
    return K[0, 0]*Xc[:, 0]/zc + K[0, 2], K[1, 1]*Xc[:, 1]/zc + K[1, 2], Xc[:, 2] > 0.05


def rot_uv(u, v, W, H, k):
    u = np.asarray(u, float); v = np.asarray(v, float); k %= 4
    if k == 0: return u, v
    if k == 1: return v, (W-1-u)
    if k == 2: return (W-1-u), (H-1-v)
    return (H-1-v), u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--ros-dir", required=True)
    ap.add_argument("--bg-dir", default="XPU-RT/sims/out/figdata_mega")
    ap.add_argument("--td-rot", type=int, default=1)
    ap.add_argument("--out", default="out/paper_figure_ros_vs_ours")
    a = ap.parse_args()
    cb = np.load(os.path.join(a.bg_dir, "clean_bg.npz"))
    bg, K, cpos, cquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    gates = np.load(os.path.join(a.bg_dir, "figure_data.npz"), allow_pickle=True)["gates_world"]
    ours = np.load(os.path.join(a.ours_dir, "figure_data.npz"), allow_pickle=True)["poses"][:, :3]
    ros = np.load(os.path.join(a.ros_dir, "figure_data.npz"), allow_pickle=True)["poses"][:, :3]

    img = np.rot90(bg, a.td_rot); nH, nW = img.shape[:2]
    def T(u, v): return rot_uv(u, v, bg.shape[1], bg.shape[0], a.td_rot)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 12.0))
    ax.imshow(img); ax.set_xlim(0, nW); ax.set_ylim(nH, 0); ax.axis("off")

    def draw(xyz, color, label, crash):
        u, v, ok = project(K, cpos, cquat, xyz); u, v = T(u, v)
        m = ok & (u > -30) & (u < nW+30) & (v > -30) & (v < nH+30)
        ax.plot(u[m], v[m], color="white", lw=5.0, alpha=0.6, zorder=3)
        ax.plot(u[m], v[m], color=color, lw=3.0, alpha=0.95, zorder=4, label=label)
        if crash:
            ax.scatter(u[m][-1], v[m][-1], s=420, marker="X", color=color, edgecolors="white",
                       linewidths=2.5, zorder=8)
            ax.annotate("✗ CRASH\nstale control", (u[m][-1], v[m][-1]), xytext=(u[m][-1]-22, v[m][-1]),
                        textcoords="data", ha="right", va="center", fontsize=10, weight="bold",
                        color="white", zorder=9, linespacing=1.2,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#b3121b", alpha=0.95, ec="white", lw=1.2))
        else:
            ax.scatter(u[m][-1], v[m][-1], s=180, marker="*", color=color, edgecolors="white",
                       linewidths=1.6, zorder=8)

    draw(ours, "#2f8f4e", "XPU-RT schedule — 4.89 ms → SUCCESS 4/4", crash=False)
    draw(ros, "#e2231a", "ROS per-net pinning — 12.40 ms → CRASH", crash=True)

    gu, gv, gok = project(K, cpos, cquat, gates); gu, gv = T(gu, gv)
    for i in range(len(gates)):
        if gok[i]:
            ax.add_patch(Circle((gu[i], gv[i]), 13, fill=False, ec="#ffd400", lw=2.2, zorder=6))

    ax.legend(loc="lower left", fontsize=8.6, framealpha=0.92)
    ax.set_title("Same course, same speed:\nROS's stale control crashes the drone — ours completes",
                 fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png and .pdf")


if __name__ == "__main__":
    main()
