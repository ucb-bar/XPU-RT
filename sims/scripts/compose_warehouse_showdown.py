#!/usr/bin/env python3
"""Warehouse SHOWDOWN — one horizontal figure comparing XPU-RT (completes) vs ROS (crashes).

Layout (all rows full-width, horizontal; top-down aisle + combined Gantt are the big ones):
  A. Top-down aisle — BOTH flight paths overlaid: XPU-RT (time-coloured, clears all 4 gates) and ROS
     (red, crashes just past gate 1). Gates, patrolling people, crash marker.
  B. 4 key snapshots — 2 ROS (incl. the crash) + 2 XPU-RT (the gates ROS never reached). No 'aborted' tiles.
  C. Telemetry — IMU |w|, goal heading, forward speed; XPU vs ROS overlaid (ROS ends at the crash).
  D. Combined onboard K1 schedule — XPU-RT (balanced) over ROS (serial backlog, overruns), shared time axis.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

CMAP = matplotlib.colormaps["viridis"]
YOLO = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}
C_MOVER = "#9d4edd"
C_XPU = "#1f9e5a"      # XPU-RT path accent (green)
C_ROS = "#e2231a"      # ROS path (red)
GANTT_CORES = ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3", "CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]
NET_COL = {"mlp_control": "#2f8f4e", "fused_full": "#7d4edd", "yolov8_nano_64x": "#e07a3f"}


def _netcol(job):
    base = job.rstrip("0123456789")
    return NET_COL.get(base, "#8aa")


def draw_gantt_block(ax, disp, y0, budget_ms, label, sublabel):
    """Draw one schedule's dispatches as horizontal bars, cores stacked from y0 upward."""
    yof = {c: y0 + i for i, c in enumerate(GANTT_CORES)}
    used = set()
    for v in disp.values():
        col = _netcol(v["job_name"]); s = float(v["start_time"]); w = float(v["duration"])
        for h in v["hardware_target"].split("+"):
            if h in yof:
                used.add(h)
                ax.barh(yof[h], max(w, 0.12), left=s, height=0.82, color=col,
                        edgecolor="white", linewidth=0.15, zorder=3)
    for c in GANTT_CORES:                       # mark cores this scheduler left completely idle
        if c not in used:
            ax.text(budget_ms * 0.5, yof[c], "idle — core unused", ha="center", va="center",
                    fontsize=7, style="italic", color="0.5", zorder=4)
    ax.text(-2, y0 + 3.5, label, ha="right", va="center", fontsize=9.5, weight="bold", rotation=90)
    ax.text(-6.5, y0 + 3.5, sublabel, ha="right", va="center", fontsize=7.4, color="0.35", rotation=90)
    return yof


def project(K, pos, quat, pts):
    w, x, y, z = [float(v) for v in quat]
    R = np.array([[1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)]])
    dd = np.asarray(pts, np.float64) - np.asarray(pos, np.float64)[None, :]
    Xc = dd @ R
    zc = np.clip(Xc[:, 2], 1e-6, None)
    return K[0, 0]*Xc[:, 0]/zc + K[0, 2], K[1, 1]*Xc[:, 1]/zc + K[1, 2], Xc[:, 2] > 0.05


def rot_uv(u, v, W, H, k):
    u = np.asarray(u, float); v = np.asarray(v, float); k %= 4
    if k == 0: return u, v
    if k == 1: return v, (W - 1 - u)
    if k == 2: return (W - 1 - u), (H - 1 - v)
    return (H - 1 - v), u


def smooth(a, n=21):
    a = np.asarray(a, float)
    if len(a) < 3: return a
    k = np.ones(n) / n
    return np.convolve(np.pad(a, n//2, mode="edge"), k, mode="same")[n//2:-(n//2) or None][:len(a)]


def cross_tof(ax, tof, vmax=4.0):
    g = np.full((24, 24), np.nan)
    g[0:8, 8:16] = tof[0]; g[8:16, 16:24] = tof[1]; g[16:24, 8:16] = tof[2]; g[8:16, 0:8] = tof[3]
    ax.imshow(g, cmap="turbo_r", vmin=0, vmax=vmax, interpolation="nearest")
    for k in (8, 16):
        ax.axhline(k-0.5, color="white", lw=0.8); ax.axvline(k-0.5, color="white", lw=0.8)
    for lbl, (yy, xx) in {"N": (0.5, 11.5), "E": (11.5, 22.5), "S": (22.5, 11.5), "W": (11.5, 1.0)}.items():
        ax.text(xx, yy, lbl, color="white", fontsize=6.0, weight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(23.5, -0.5)


def draw_topdown(ax, bg, K, cpos, cquat, xpu, ros, gates, people, tnorm, t_s, rot, flipx,
                 path_start, ros_crash_xy, moments):
    H, W = bg.shape[:2]
    img = np.rot90(bg, rot)
    if flipx: img = img[:, ::-1]
    nH, nW = img.shape[:2]

    def T(u, v):
        u2, v2 = rot_uv(u, v, W, H, rot)
        if flipx: u2 = (nW - 1) - u2
        return u2, v2

    def proj(pts):
        u, v, ok = project(K, cpos, cquat, pts); u, v = T(u, v); return u, v, ok

    ax.imshow(img); ax.set_xlim(0, nW); ax.set_ylim(nH, 0); ax.axis("off")

    # --- patrolling people (fixed at their REAL height, not z=2) ---
    for j in range(people.shape[1]):
        pts = people[:, j, :].copy()
        uu, vv, o2 = proj(pts)
        m = o2 & (uu > -30) & (uu < nW+30) & (vv > -30) & (vv < nH+30)
        if m.sum() < 5: continue
        U, V = uu[m], vv[m]
        ax.plot(U, V, color=C_MOVER, lw=1.7, ls=(0, (1, 1.4)), alpha=0.9, zorder=3)
        step = max(6, len(U)//5)
        for i in range(step, len(U)-2, step):
            di = min(4, len(U)-1-i)
            if np.hypot(U[i+di]-U[i], V[i+di]-V[i]) < 1.5: continue
            ax.annotate("", xy=(U[i+di], V[i+di]), xytext=(U[i], V[i]),
                        arrowprops=dict(arrowstyle="-|>", color=C_MOVER, lw=1.2, alpha=0.9), zorder=4)
        ax.scatter(U[-1], V[-1], s=30, facecolors=C_MOVER, edgecolors="white", linewidths=1.0, zorder=5)

    # --- gates ---
    gu, gv, gok = proj(gates)
    for i in range(len(gates)):
        if gok[i]:
            ax.add_patch(Circle((gu[i], gv[i]), 12, fill=False, ec="#ffd400", lw=2.4, zorder=6))
            ax.text(gu[i], gv[i]-15, f"G{i+1}", color="#ffd400", fontsize=8, weight="bold",
                    ha="center", va="center", zorder=6)

    # --- ROS path (red), truncated at crash ---
    ru, rv, rok = proj(ros)
    rvis = rok & (np.arange(len(ros)) >= path_start)
    ax.plot(ru[rvis], rv[rvis], color="white", lw=4.4, alpha=0.7, zorder=3)
    ax.plot(ru[rvis], rv[rvis], color=C_ROS, lw=2.6, alpha=0.95, zorder=4)
    cu, cv, cok = proj(np.asarray(ros_crash_xy)[None, :])
    if cok[0]:
        ax.scatter(cu[0], cv[0], s=460, marker="X", color=C_ROS, edgecolors="white", linewidths=2.4, zorder=11)

    # --- XPU-RT path (time-coloured), full ---
    xu, xv, xok = proj(xpu)
    xvis = xok & (np.arange(len(xpu)) >= path_start)
    ax.plot(xu[xvis], xv[xvis], color="white", lw=4.4, alpha=0.7, zorder=5)
    P = np.column_stack([xu, xv])[xvis]; tn = tnorm[xvis]
    for i in range(len(P)-1):
        ax.plot(P[i:i+2, 0], P[i:i+2, 1], color=CMAP(tn[i]), lw=2.6, alpha=0.97, zorder=6)

    # --- numbered moment markers (2 ROS + 2 XPU) ---
    mk = []
    for mi, (src, step, lab) in enumerate(moments):
        path = ros if src == "ROS" else xpu
        u, v, o = proj(path[step:step+1])
        ec = C_ROS if src == "ROS" else "#ffd400"
        is_crash = src == "ROS" and step >= len(ros) - 2
        if o[0]:
            mu, mv = (u[0], v[0] - 34) if is_crash else (u[0], v[0])   # lift crash label off the X marker
            ax.add_patch(Circle((mu, mv), 12, fill=True, fc="black", ec=ec, lw=2.2, zorder=8))
            ax.text(mu, mv, str(mi+1), color="white", fontsize=9, weight="bold",
                    ha="center", va="center", zorder=9)
            mk.append((mu, mv))
        else:
            mk.append(None)
    return mk


def load(dd):
    return np.load(os.path.join(dd, "figure_data.npz"), allow_pickle=True)


def frame_at(dd, fs, step):
    i = int(np.argmin(np.abs(fs - step)))
    return np.load(os.path.join(dd, f"frames/frame_{i:03d}.npz"), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--xpu-dir", required=True, help="figure_data.npz dir for the XPU-RT (successful) flight")
    ap.add_argument("--ros-dir", required=True, help="figure_data.npz dir for the ROS (crash) flight")
    ap.add_argument("--sched-xpu", default=os.path.join(_repo, "schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json"))
    ap.add_argument("--sched-ros", default=os.path.join(_repo, "schedules/scheduled_ros_partition_deployed.json"))
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/warehouse_showdown"))
    a = ap.parse_args()

    X = load(a.xpu_dir); R = load(a.ros_dir)
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]
    xt = X["t_s"]; rt = R["t_s"]
    tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]
    gates = X["gates_world"]
    cb = np.load(os.path.join(a.xpu_dir, "clean_bg.npz")) if os.path.exists(os.path.join(a.xpu_dir, "clean_bg.npz")) else X
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    ros_crash_xy = rxyz[-1]

    # moments: 2 ROS (gate1 clear, the crash) + 2 XPU (gate G3, gate G4 — where ROS is already down)
    moments = [("ROS", 250, "ROS · clears gate G1"),
               ("ROS", len(rxyz)-1, "ROS · crashes into crate"),
               ("XPU", 984, "XPU-RT · gate G3"),
               ("XPU", 1180, "XPU-RT · gate G4 + person")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19, 14.5))
    outer = fig.add_gridspec(4, 1, height_ratios=[4.6, 3.0, 2.1, 3.6], hspace=0.30,
                             left=0.045, right=0.99, top=0.965, bottom=0.03)

    # ---- A top-down (full width, horizontal) ----
    axt = fig.add_subplot(outer[0])
    draw_topdown(axt, ov_bg, ovK, ovpos, ovquat, xxyz, rxyz, gates, people, tnorm, xt,
                 a.rot, a.flipx, a.path_start, ros_crash_xy, moments)
    axt.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=3, label="XPU-RT ✓ completes (colour = time)"),
                        Line2D([0], [0], color=C_ROS, lw=3, label="ROS ✗ crashes past gate 1"),
                        Line2D([0], [0], color=C_MOVER, lw=1.7, ls=(0, (1, 1.4)), label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=2, ls="none", label="gate")],
               loc="upper left", fontsize=9, framealpha=0.92, ncol=4, handlelength=1.8)
    axt.set_title("Warehouse gate-course showdown — same aisle, same obstacles: XPU-RT clears all 4 gates, "
                  "ROS crashes just past gate 1", fontsize=12.5, weight="bold", loc="left")

    # ---- B snapshots (4 across: 2 ROS + 2 XPU) ----
    bgrid = outer[1].subgridspec(1, 4, wspace=0.13)
    for c, (src, step, lab) in enumerate(moments):
        dd = a.ros_dir if src == "ROS" else a.xpu_dir
        fs = (R if src == "ROS" else X)["frame_steps"]
        f = frame_at(dd, fs, step)
        cell = bgrid[c].subgridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1.5, 1.0],
                                    hspace=0.12, wspace=0.07)
        tt = (rt if src == "ROS" else xt)[min(step, len(rt if src == "ROS" else xt)-1)]
        ac = fig.add_subplot(cell[0, :]); ac.imshow(f["chase"][140:530, 200:760]); ac.axis("off")
        ttl_c = C_ROS if src == "ROS" else C_XPU
        ac.set_title(f"{c+1}. {lab} · t={tt:.1f}s", fontsize=8.6, weight="bold", color=ttl_c)
        af = fig.add_subplot(cell[1, 0]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="auto")
        for dd2 in [x for x in f["det"] if x[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = dd2
            _, col = YOLO.get(int(cls), ("obj", "#39f"))
            af.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=col, lw=1.3))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=7)
        at = fig.add_subplot(cell[1, 1]); cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=7)

    # ---- C telemetry (IMU |w|, goal heading, speed) XPU vs ROS + velocity arrows (XPU) ----
    tg = outer[2].subgridspec(1, 4, wspace=0.26)
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    axi = fig.add_subplot(tg[0])
    axi.plot(xt, smooth(xw), color=C_XPU, lw=1.3, label="XPU-RT")
    axi.plot(rt, smooth(rw), color=C_ROS, lw=1.3, label="ROS")
    axi.axvline(rt[-1], color=C_ROS, lw=1.2, ls=(0, (2, 2)));
    axi.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=8); axi.set_title("body-rate magnitude", fontsize=9, weight="bold")
    axi.legend(fontsize=7.5, loc="upper right")
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0]))
    rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    axg = fig.add_subplot(tg[1])
    axg.plot(xt, xg, color=C_XPU, lw=1.3, label="XPU-RT"); axg.plot(rt, rg, color=C_ROS, lw=1.3, label="ROS")
    axg.set_ylabel("goal heading (°)", fontsize=8); axg.set_title("nav goal heading", fontsize=9, weight="bold")
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1)
    rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    axs = fig.add_subplot(tg[2])
    axs.plot(xt, smooth(xs, 11), color=C_XPU, lw=1.3, label="XPU-RT")
    axs.plot(rt, smooth(rs, 11), color=C_ROS, lw=1.3, label="ROS")
    axs.set_ylabel("forward speed (m/s)", fontsize=8); axs.set_title("speed → ROS drops at crash", fontsize=9, weight="bold")
    for ax in (axi, axg, axs):
        ax.set_xlabel("time (s)", fontsize=8); ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=7)
        ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.4, zorder=0)
    # velocity arrows (XPU-RT only — two overlaid quivers would be unreadable)
    axq = fig.add_subplot(tg[3])
    vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(a.path_start, len(xxyz), 12)
    axq.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=0.8, zorder=0)
    axq.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis",
               angles="xy", scale_units="xy", scale=7.0, width=0.006, headwidth=4, headlength=5)
    axq.set_xlabel("along-aisle y (m)", fontsize=8); axq.set_ylabel("lateral x (m)", fontsize=8)
    axq.set_title("XPU-RT velocity (arrow = heading·speed)", fontsize=9, weight="bold")
    axq.grid(True, color="0.92", lw=0.5); axq.tick_params(labelsize=7); axq.set_aspect("equal", adjustable="datalim")

    # ---- D combined onboard K1 Gantt (XPU-RT over ROS, shared time axis) ----
    axd = fig.add_subplot(outer[3])
    xsp = rsp = 0.0
    if a.sched_xpu and os.path.exists(a.sched_xpu):
        xd = json.load(open(a.sched_xpu))["dispatches"]
        xsp = max(float(v["start_time"]) + float(v["duration"]) for v in xd.values())
        draw_gantt_block(axd, xd, 9.5, xsp, "XPU-RT", "CP-SAT · 8 cores")
    if a.sched_ros and os.path.exists(a.sched_ros):
        rd = json.load(open(a.sched_ros))["dispatches"]
        rsp = max(float(v["start_time"]) + float(v["duration"]) for v in rd.values())
        draw_gantt_block(axd, rd, 0.0, rsp, "ROS", "static partition · 6 cores")
    tmax = max(xsp, rsp) * 1.02
    axd.axhline(8.6, color="0.6", lw=0.8)                       # divider between the two schedulers
    if xsp:                                                     # XPU finishes here (within budget)
        axd.axvline(xsp, color=C_XPU, lw=1.6, ls=(0, (4, 2)), zorder=6)
        axd.text(xsp, 17.6, f" XPU-RT done {xsp:.0f} ms ✓", color=C_XPU, fontsize=8.5, weight="bold", va="bottom")
    if rsp:                                                     # ROS still backlogged well past it → crash
        axd.axvspan(xsp, rsp, color="#f6e3e3", alpha=0.5, zorder=0)
        axd.text(rsp, 3.4, f"ROS still backlogged {rsp:.0f} ms → misses frame → ✗ CRASH ", color=C_ROS,
                 fontsize=8.5, weight="bold", va="center", ha="right")
    axd.set_xlim(-8, tmax); axd.set_ylim(-0.7, 18.2)
    axd.set_yticks([9.5 + i for i in range(8)] + [i for i in range(8)])
    axd.set_yticklabels([c.replace("CPU_", "") for c in GANTT_CORES] * 2, fontsize=6.5)
    axd.set_xlabel("onboard schedule time (ms) — real K1 measured profile", fontsize=9)
    axd.set_title("Combined onboard K1 schedule — XPU-RT balances all 8 cores and fits the frame; "
                  "ROS pins to 6 cores, serial YOLO overruns → backlog → crash", fontsize=11, weight="bold", loc="left")
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    axd.legend(handles=[Line2D([0], [0], color=NET_COL["mlp_control"], lw=6, label="CTRL (mlp)"),
                        Line2D([0], [0], color=NET_COL["fused_full"], lw=6, label="NAV (fused)"),
                        Line2D([0], [0], color=NET_COL["yolov8_nano_64x"], lw=6, label="YOLO")],
               loc="lower right", fontsize=8, ncol=3, framealpha=0.9)

    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
