#!/usr/bin/env python3
"""Export EVERY sensor / sub-image of the warehouse showdown figure as its own high-resolution PNG.

Reuses the exact rendering logic of showdown_gatecourse.py (same moment selection, same crops, same
colormaps / ToF layout / YOLO overlay), so each exported tile is pixel-faithful to the composite figure
— just rendered on its own canvas at high DPI. Produces, under --out-dir:

  topdown_aisle.png                     full top-down flight-path panel (both flights, gates, strobe)
  moment{n}_{tag}_chase.png             3rd-person chase crop            (n=1..4)
  moment{n}_{tag}_fpv_yolo.png          onboard HM01B0 FPV + YOLO boxes
  moment{n}_{tag}_tof.png               4x VL53L5CX cross-ToF panel
  moment{n}_{tag}_fpv_raw.png           FPV grayscale, no overlay (pure sensor)
  telem_bodyrate.png / telem_heading.png / telem_speed.png / telem_velocity.png
  gantt_k1_schedule.png                 combined onboard K1 Gantt

Run with the SAME --xpu-dir/--ros-dir the composite used so the tiles match the paper figure exactly.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# import the composite's own primitives so every tile is identical to the figure
import showdown_gatecourse as S


def _slug(lab):
    return (lab.replace("·", "").replace("→", "to").replace("(", "").replace(")", "")
            .replace("✓", "").replace(".", "p").replace(" ", "_").strip("_").lower())


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", required=True)
    ap.add_argument("--ros-dir", required=True)
    ap.add_argument("--sched-xpu", default=os.path.join(_repo, "schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json"))
    ap.add_argument("--sched-ros", default=os.path.join(_repo, "schedules/scheduled_ros_partition_deployed.json"))
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    X = S.load(a.xpu_dir); R = S.load(a.ros_dir)
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]; xt = X["t_s"]; rt = R["t_s"]
    tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    cbp = os.path.join(a.xpu_dir, "clean_bg.npz")
    cb = np.load(cbp) if os.path.exists(cbp) else X
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    nr, nx = len(rxyz), len(xxyz)

    # ---- identical moment selection to the composite ----
    _st = X["obst_pos"][0][~pm]; _st = _st[_st[:, 2] > -10.0]
    _tall = _st[_st[:, 2] > 2.0]
    ps = a.path_start
    if len(_tall) and nx > ps + 2:
        _dd = np.linalg.norm(xxyz[ps:, None, :2] - _tall[None, :, :2], axis=2)
        _rel = int(_dd.min(axis=1).argmin()); _step_near = ps + _rel
        _clear = float(_dd[_rel].min()); _near_bin = _tall[int(_dd[_rel].argmin()), :3]
    else:
        _step_near, _clear, _near_bin = int(0.45 * nx), 0.66, None
    moments = [("ROS", int(0.25 * nr), "ROS · clears gate G1"),
               ("ROS", nr - 1, "ROS · loses stability → crash"),
               ("XPU", _step_near, f"XPU-RT · barely clears crate ({_clear:.2f} m)"),
               ("XPU", int(0.96 * nx), "XPU-RT · gate G4 + person")]
    near_miss = (xxyz[_step_near, :3], _near_bin, _clear)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "pdf.fonttype": 42})
    saved = []

    def save(fig, name):
        p = os.path.join(a.out_dir, name)
        fig.savefig(p, dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig); saved.append(name)

    # ---- A. top-down aisle (its own canvas) ----
    fig = plt.figure(figsize=(24, 7.0))
    ax = fig.add_subplot(111)
    S.draw_topdown(ax, ov_bg, ovK, ovpos, ovquat, xxyz, rxyz, gates, people, tnorm, a.rot, a.flipx,
                   a.path_start, rxyz[-1], moments, ov_obj=X["ov_bg"], near_miss=near_miss)
    ax.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label="XPU-RT ✓ completes 3/6 (colour = time)"),
                       Line2D([0], [0], color=S.C_ROS, lw=5, label="ROS ✗ crashes 0/6 (50 Hz control)"),
                       Line2D([0], [0], marker="o", color=S.C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"),
                       Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")],
              loc="upper left", fontsize=14.5, framealpha=0.93, ncol=4, handlelength=1.9)
    save(fig, "topdown_aisle.png")

    # ---- B. per-moment sensor tiles: chase | FPV+YOLO | cross-ToF (+ raw FPV) ----
    for c, (src, step, lab) in enumerate(moments):
        dd = a.ros_dir if src == "ROS" else a.xpu_dir
        fs = (R if src == "ROS" else X)["frame_steps"]; f = S.frame_at(dd, fs, step)
        tag = _slug(lab)

        # chase (identical crop to the composite)
        fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111)
        ax.imshow(f["chase"][225:465, 415:655]); ax.axis("off")
        save(fig, f"moment{c+1}_{tag}_chase.png")

        # FPV + YOLO overlay
        fig = plt.figure(figsize=(6, 4)); ax = fig.add_subplot(111)
        ax.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = S.YOLO.get(int(cls), ("obj", "#39f"))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cc, lw=2.6))
        ax.set_xticks([]); ax.set_yticks([])
        save(fig, f"moment{c+1}_{tag}_fpv_yolo.png")

        # raw FPV (pure HM01B0 sensor, no overlay)
        fig = plt.figure(figsize=(6, 4)); ax = fig.add_subplot(111)
        ax.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal"); ax.axis("off")
        save(fig, f"moment{c+1}_{tag}_fpv_raw.png")

        # cross-ToF
        fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111)
        S.cross_tof(ax, f["tof"])
        save(fig, f"moment{c+1}_{tag}_tof.png")

    # ---- C. telemetry panels (each on its own canvas) ----
    def smooth(v, n=21): return S.smooth(v, n)
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    fig = plt.figure(figsize=(6, 4.5)); ax = fig.add_subplot(111)
    ax.plot(xt, smooth(xw), color=S.C_XPU, lw=2.2, label="XPU-RT"); ax.plot(rt, smooth(rw), color=S.C_ROS, lw=2.2, label="ROS")
    ax.set_ylabel("IMU |ω| (rad/s), smoothed"); ax.set_title("body-rate magnitude", weight="bold"); ax.legend(loc="upper right")
    ax.set_xlabel("time (s)"); ax.grid(True, color="0.9", lw=0.5)
    ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); ax.axvline(rt[-1], color=S.C_ROS, lw=1.8, ls=(0, (4, 2)), alpha=0.85)
    save(fig, "telem_bodyrate.png")

    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0])); rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    fig = plt.figure(figsize=(6, 4.5)); ax = fig.add_subplot(111)
    ax.plot(xt, xg, color=S.C_XPU, lw=2.2); ax.plot(rt, rg, color=S.C_ROS, lw=2.2)
    ax.set_ylabel("goal heading (°)"); ax.set_title("nav goal heading", weight="bold")
    ax.set_xlabel("time (s)"); ax.grid(True, color="0.9", lw=0.5)
    ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); ax.axvline(rt[-1], color=S.C_ROS, lw=1.8, ls=(0, (4, 2)), alpha=0.85)
    save(fig, "telem_heading.png")

    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1); rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    fig = plt.figure(figsize=(6, 4.5)); ax = fig.add_subplot(111)
    ax.plot(xt, smooth(xs, 11), color=S.C_XPU, lw=2.2); ax.plot(rt, smooth(rs, 11), color=S.C_ROS, lw=2.2)
    ax.set_ylabel("forward speed (m/s)"); ax.set_title("speed → ROS drops at crash", weight="bold")
    ax.set_xlabel("time (s)"); ax.grid(True, color="0.9", lw=0.5)
    ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); ax.axvline(rt[-1], color=S.C_ROS, lw=1.8, ls=(0, (4, 2)), alpha=0.85)
    save(fig, "telem_speed.png")

    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111)
    vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(a.path_start, len(xxyz), 12)
    ax.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=1.0, zorder=0)
    ax.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis", angles="xy", scale_units="xy", scale=7.0, width=0.007, headwidth=4, headlength=5)
    ax.set_xlabel("along-aisle y (m)"); ax.set_ylabel("lateral x (m)")
    ax.set_title("XPU-RT velocity (arrow = heading·speed)", weight="bold"); ax.grid(True, color="0.92", lw=0.5); ax.set_aspect("equal", adjustable="datalim")
    save(fig, "telem_velocity.png")

    # ---- D. combined K1 Gantt ----
    fig = plt.figure(figsize=(20, 6.5)); ax = fig.add_subplot(111)
    S.draw_combined_gantt(ax, json.load(open(a.sched_xpu))["dispatches"], json.load(open(a.sched_ros))["dispatches"])
    save(fig, "gantt_k1_schedule.png")

    print(f"wrote {len(saved)} tiles (png+pdf each) @dpi {a.dpi} -> {a.out_dir}")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    main()
