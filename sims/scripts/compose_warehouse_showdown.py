#!/usr/bin/env python3
"""Warehouse SHOWDOWN — one long horizontal figure (spans both paper columns): XPU-RT completes vs ROS crashes.

  A. Top-down aisle (edge-to-edge) — BOTH flight paths: XPU-RT (time-coloured, all 4 gates) + ROS (red, crashes
     just past gate 1). Gates, patrolling people (real height), crash marker.
  B. 4 key moments (2 ROS + 2 XPU), each a horizontal strip: chase (zoomed) | FPV+YOLO | large cross-ToF.
  C. Telemetry — IMU |w|, goal heading, forward speed (XPU vs ROS) + XPU velocity arrows.
  D. Combined onboard K1 schedule — the annotated Gantt (NAV/CTRL windows, sensor-in + output arrows) with
     XPU-RT over ROS; ROS is time-cropped ("…") so the short XPU-RT schedule stays legible.
"""
import argparse, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

CMAP = matplotlib.colormaps["viridis"]
YOLO = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}
C_MOVER = "#9d4edd"; C_XPU = "#1f9e5a"; C_ROS = "#e2231a"
GANTT_CORES = ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3", "CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]
C_CTRL, C_NAV, C_YOLO = "#2f8f4e", "#7b52c0", "#e8823a"
LP, LG = "#efe7fb", "#e6f3ea"       # light NAV / CTRL window fills
NET = {"mlp_control": ("ctrl", C_CTRL), "fused_full": ("nav", C_NAV), "yolov8_nano_64x": ("yolo", C_YOLO)}


def netinfo(job):
    base = re.sub(r"\d+$", "", job)
    return NET.get(base, ("other", "#8aa"))


def inst_of(job):
    m = re.search(r"(\d+)$", job); return int(m.group(1)) if m else 0


def project(K, pos, quat, pts):
    w, x, y, z = [float(v) for v in quat]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    dd = np.asarray(pts, np.float64) - np.asarray(pos, np.float64)[None, :]
    Xc = dd @ R; zc = np.clip(Xc[:, 2], 1e-6, None)
    return K[0, 0]*Xc[:, 0]/zc + K[0, 2], K[1, 1]*Xc[:, 1]/zc + K[1, 2], Xc[:, 2] > 0.05


def rot_uv(u, v, W, H, k):
    u = np.asarray(u, float); v = np.asarray(v, float); k %= 4
    if k == 0: return u, v
    if k == 1: return v, (W-1-u)
    if k == 2: return (W-1-u), (H-1-v)
    return (H-1-v), u


def smooth(a, n=21):
    a = np.asarray(a, float)
    if len(a) < 3: return a
    k = np.ones(n)/n
    return np.convolve(np.pad(a, n//2, "edge"), k, "same")[n//2:-(n//2) or None][:len(a)]


def cross_tof(ax, tof, vmax=4.0):
    g = np.full((24, 24), np.nan)
    g[0:8, 8:16] = tof[0]; g[8:16, 16:24] = tof[1]; g[16:24, 8:16] = tof[2]; g[8:16, 0:8] = tof[3]
    ax.imshow(g, cmap="turbo_r", vmin=0, vmax=vmax, interpolation="nearest")
    for k in (8, 16):
        ax.axhline(k-0.5, color="white", lw=1.0); ax.axvline(k-0.5, color="white", lw=1.0)
    for lbl, (yy, xx) in {"N": (0.3, 11.5), "E": (11.5, 22.7), "S": (22.7, 11.5), "W": (11.5, 0.9)}.items():
        ax.text(xx, yy, lbl, color="white", fontsize=9, weight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(23.5, -0.5)


def draw_topdown(ax, bg, K, cpos, cquat, xpu, ros, gates, people, tnorm, rot, flipx, path_start, ros_crash_xy, moments):
    H, W = bg.shape[:2]
    img = np.rot90(bg, rot)
    if flipx: img = img[:, ::-1]
    nH, nW = img.shape[:2]

    def T(u, v):
        u2, v2 = rot_uv(u, v, W, H, rot)
        if flipx: u2 = (nW-1) - u2
        return u2, v2

    def proj(pts):
        u, v, ok = project(K, cpos, cquat, pts); u, v = T(u, v); return u, v, ok

    ax.imshow(img); ax.set_xlim(0, nW); ax.set_ylim(nH, 0); ax.axis("off")
    for j in range(people.shape[1]):                                 # patrolling people at REAL height
        uu, vv, o2 = proj(people[:, j, :])
        m = o2 & (uu > -30) & (uu < nW+30) & (vv > -30) & (vv < nH+30)
        if m.sum() < 5: continue
        U, V = uu[m], vv[m]
        ax.plot(U, V, color=C_MOVER, lw=2.0, ls=(0, (1, 1.3)), alpha=0.9, zorder=3)
        for i in range(max(6, len(U)//5), len(U)-2, max(6, len(U)//5)):
            di = min(4, len(U)-1-i)
            if np.hypot(U[i+di]-U[i], V[i+di]-V[i]) > 1.5:
                ax.annotate("", xy=(U[i+di], V[i+di]), xytext=(U[i], V[i]),
                            arrowprops=dict(arrowstyle="-|>", color=C_MOVER, lw=1.4, alpha=0.9), zorder=4)
        ax.scatter(U[-1], V[-1], s=42, facecolors=C_MOVER, edgecolors="white", linewidths=1.2, zorder=5)
    gu, gv, gok = proj(gates)
    for i in range(len(gates)):
        if gok[i]:
            ax.add_patch(Circle((gu[i], gv[i]), 13, fill=False, ec="#ffd400", lw=3.0, zorder=6))
            ax.text(gu[i], gv[i]-18, f"G{i+1}", color="#ffd400", fontsize=13, weight="bold", ha="center", va="center", zorder=6)
    ru, rv, rok = proj(ros); rvis = rok & (np.arange(len(ros)) >= path_start)
    ax.plot(ru[rvis], rv[rvis], color="white", lw=6, alpha=0.75, zorder=3)
    ax.plot(ru[rvis], rv[rvis], color=C_ROS, lw=3.4, alpha=0.97, zorder=4)
    cu, cv, cok = proj(np.asarray(ros_crash_xy)[None, :])
    if cok[0]:
        ax.scatter(cu[0], cv[0], s=620, marker="X", color=C_ROS, edgecolors="white", linewidths=3, zorder=11)
    xu, xv, xok = proj(xpu); xvis = xok & (np.arange(len(xpu)) >= path_start)
    ax.plot(xu[xvis], xv[xvis], color="white", lw=6, alpha=0.75, zorder=5)
    P = np.column_stack([xu, xv])[xvis]; tn = tnorm[xvis]
    for i in range(len(P)-1):
        ax.plot(P[i:i+2, 0], P[i:i+2, 1], color=CMAP(tn[i]), lw=3.4, alpha=0.97, zorder=6)
    for mi, (src, step, lab) in enumerate(moments):
        path = ros if src == "ROS" else xpu
        u, v, o = proj(path[min(step, len(path)-1):min(step, len(path)-1)+1])
        ec = C_ROS if src == "ROS" else "#ffd400"
        is_crash = src == "ROS" and step >= len(ros)-2
        if o[0]:
            mu, mv = (u[0], v[0]-40) if is_crash else (u[0], v[0])
            ax.add_patch(Circle((mu, mv), 15, fill=True, fc="black", ec=ec, lw=2.6, zorder=8))
            ax.text(mu, mv, str(mi+1), color="white", fontsize=13, weight="bold", ha="center", va="center", zorder=9)


def draw_combined_gantt(ax, xd, rd):
    xsp = max(float(v["start_time"])+float(v["duration"]) for v in xd.values())
    rsp = max(float(v["start_time"])+float(v["duration"]) for v in rd.values())
    T1 = xsp + 5.0; tail = 18.0; T2 = rsp - tail; GAP = 7.0                 # crop ROS's long middle with a "…"
    def xr(t):
        if t <= T1: return t
        if t >= T2: return T1 + GAP + (t - T2)
        return T1 + GAP * (t - T1) / max(1e-6, T2 - T1)
    xmax = xr(rsp)

    def bars(disp, y0):
        yof = {c: y0+i for i, c in enumerate(GANTT_CORES)}; used = set()
        for v in disp.values():
            _, col = netinfo(v["job_name"]); s = float(v["start_time"]); e = s + float(v["duration"])
            if s >= T1 and e <= T2:                                          # fully inside the cropped gap
                continue
            xs = xr(s); xe = xr(min(e, T1) if s < T1 else e)
            for h in v["hardware_target"].split("+"):
                if h in yof:
                    used.add(h); ax.barh(yof[h], max(xe-xs, 0.14), left=xs, height=0.82, color=col,
                                         edgecolor="white", linewidth=0.12, zorder=3)
        for c in GANTT_CORES:
            if c not in used:
                ax.text(xsp*0.5, yof[c], "idle — core unused", ha="center", va="center", fontsize=9,
                        style="italic", color="0.5", zorder=4)
        return yof

    def windows(y0, y1):                                                     # NAV 20 ms / CTRL 10 ms bands
        for seg0, seg1 in ((0, T1), (T2, rsp)):
            k = int(seg0 // 20)
            while k*20 < seg1:
                if seg0 <= k*20 < seg1:
                    ax.add_patch(Rectangle((xr(k*20), y0), xr(min((k+1)*20, seg1))-xr(k*20), y1-y0, color=LP, zorder=0.1))
                k += 1
            k = int(seg0 // 10)
            while k*10 < seg1:
                if k % 2 == 0 and seg0 <= k*10 < seg1:
                    ax.add_patch(Rectangle((xr(k*10), y0), xr(min((k+1)*10, seg1))-xr(k*10), y1-y0, color=LG, zorder=0.15))
                k += 1

    def arrows(disp, ytop, ybot):                                           # sensor-in (red, top) + output (colored, bottom)
        seen = {}
        for v in disp.values():
            key = (netinfo(v["job_name"])[0], inst_of(v["job_name"])); s = float(v["start_time"]); e = s+float(v["duration"])
            r = seen.get(key, [s, e]); seen[key] = [min(r[0], s), max(r[1], e)]
        for (net, _), (s, e) in seen.items():
            if T1 < s < T2: continue
            col = {"ctrl": C_CTRL, "nav": C_NAV, "yolo": C_YOLO}.get(net, "#8aa")
            xs = xr(s); xe = xr(min(e, T1) if s < T1 else e)
            ax.annotate("", xy=(xs, ytop-0.15), xytext=(xs, ytop+0.85),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color="#d62728", lw=1.6), zorder=6)
            ax.annotate("", xy=(xe, ybot+0.15), xytext=(xe, ybot-0.85),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=col, lw=1.8), zorder=6)

    xy = bars(xd, 9.5); windows(9.0, 17.6); arrows(xd, 17.6, 9.0)
    ry = bars(rd, 0.0); windows(-0.6, 8.0); arrows(rd, 8.0, -0.6)
    ax.axhline(8.7, color="0.55", lw=1.0)
    ax.axvline(xr(xsp), color=C_XPU, lw=2.4, ls=(0, (5, 3)), zorder=7)
    ax.text(xr(xsp)-0.4, 18.6, f"XPU-RT done {xsp:.0f} ms ✓", color=C_XPU, fontsize=12.5, weight="bold", va="bottom", ha="right")
    # "…" crop marker
    xc = T1 + GAP/2
    ax.axvspan(xr(T1), xr(T2), color="white", zorder=5)
    ax.text(xc, 8.5, "⋯", fontsize=22, ha="center", va="center", color="0.4", zorder=6)
    ax.text(xc, -1.7, "(time cropped)", fontsize=9, ha="center", va="top", color="0.5", zorder=6)
    ax.text(xr(rsp), 3.5, f"ROS still backlogged\n{rsp:.0f} ms → ✗ CRASH", color=C_ROS, fontsize=12.5, weight="bold",
            va="center", ha="right", zorder=8, linespacing=1.2)
    ax.text(-3.2, 13.5, "XPU-RT", fontsize=14, weight="bold", rotation=90, va="center", ha="center")
    ax.text(-3.2, 3.5, "ROS", fontsize=14, weight="bold", rotation=90, va="center", ha="center", color=C_ROS)
    ax.text(-6.0, 13.5, "CP-SAT · 8 cores", fontsize=10, color="0.35", rotation=90, va="center", ha="center")
    ax.text(-6.0, 3.5, "static · 6 cores", fontsize=10, color="0.35", rotation=90, va="center", ha="center")
    ax.text(0.2, 19.0, "sensors in ↓ (red)   ·   model outputs ↑ (coloured)", fontsize=10, color="0.3", va="bottom")
    ax.set_xlim(-8.5, xmax+1); ax.set_ylim(-2.4, 19.6)
    ax.set_yticks([9.5+i for i in range(8)] + list(range(8)))
    ax.set_yticklabels([c.replace("CPU_", "") for c in GANTT_CORES]*2, fontsize=8.5)
    xt = [t for t in (0, 10, 20, 30, 40) if t <= T1] + [T2 + tail]
    ax.set_xticks([xr(t) for t in xt]); ax.set_xticklabels([f"{t:.0f}" for t in xt], fontsize=10)
    ax.set_xlabel("onboard schedule time (ms) — real K1 measured profile", fontsize=12.5)
    ax.set_title("Combined onboard K1 schedule — XPU-RT balances 8 cores and fits the frame; "
                 "ROS pins 6 cores, serial YOLO overruns → backlog → crash", fontsize=13.5, weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=C_CTRL, lw=7, label="CTRL (mlp) 100 Hz"),
                       Line2D([0], [0], color=C_NAV, lw=7, label="NAV (fused) 50 Hz"),
                       Line2D([0], [0], color=C_YOLO, lw=7, label="YOLO"),
                       Rectangle((0, 0), 1, 1, fc=LP, label="NAV 20 ms window"),
                       Rectangle((0, 0), 1, 1, fc=LG, label="CTRL 10 ms window")],
              loc="lower right", fontsize=10, ncol=5, framealpha=0.92)


def load(dd): return np.load(os.path.join(dd, "figure_data.npz"), allow_pickle=True)
def frame_at(dd, fs, step): return np.load(os.path.join(dd, f"frames/frame_{int(np.argmin(np.abs(fs-step))):03d}.npz"), allow_pickle=True)


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", required=True); ap.add_argument("--ros-dir", required=True)
    ap.add_argument("--sched-xpu", default=os.path.join(_repo, "schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json"))
    ap.add_argument("--sched-ros", default=os.path.join(_repo, "schedules/scheduled_ros_partition_deployed.json"))
    ap.add_argument("--rot", type=int, default=0); ap.add_argument("--flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/warehouse_showdown"))
    a = ap.parse_args()
    X = load(a.xpu_dir); R = load(a.ros_dir)
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]; xt = X["t_s"]; rt = R["t_s"]
    tnorm = (xt-xt.min())/max(1e-6, xt.max()-xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    cbp = os.path.join(a.xpu_dir, "clean_bg.npz")
    cb = np.load(cbp) if os.path.exists(cbp) else X
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    moments = [("ROS", 250, "ROS · clears gate G1"), ("ROS", len(rxyz)-1, "ROS · crashes into crate"),
               ("XPU", 984, "XPU-RT · gate G3"), ("XPU", 1180, "XPU-RT · gate G4 + person")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(22, 15))
    outer = fig.add_gridspec(4, 1, height_ratios=[4.0, 3.3, 2.2, 3.9], hspace=0.26,
                             left=0.028, right=0.995, top=0.965, bottom=0.028)

    # A top-down (edge to edge)
    axt = fig.add_subplot(outer[0])
    draw_topdown(axt, ov_bg, ovK, ovpos, ovquat, xxyz, rxyz, gates, people, tnorm, a.rot, a.flipx,
                 a.path_start, rxyz[-1], moments)
    axt.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=4, label="XPU-RT ✓ completes (colour = time)"),
                        Line2D([0], [0], color=C_ROS, lw=4, label="ROS ✗ crashes past gate 1"),
                        Line2D([0], [0], color=C_MOVER, lw=2, ls=(0, (1, 1.3)), label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=2.5, ls="none", label="gate")],
               loc="upper left", fontsize=12, framealpha=0.93, ncol=4, handlelength=1.9)
    axt.set_title("Warehouse gate-course showdown — same aisle, same obstacles: XPU-RT clears all 4 gates, "
                  "ROS crashes just past gate 1", fontsize=16, weight="bold", loc="left")

    # B snapshots: 4 moments in a row, each a horizontal [chase | FPV+YOLO | ToF]
    bgrid = outer[1].subgridspec(1, 4, wspace=0.10)
    for c, (src, step, lab) in enumerate(moments):
        dd = a.ros_dir if src == "ROS" else a.xpu_dir
        fs = (R if src == "ROS" else X)["frame_steps"]; f = frame_at(dd, fs, step)
        tt = (rt if src == "ROS" else xt)[min(step, len(rt if src == "ROS" else xt)-1)]
        col = bgrid[c].subgridspec(2, 3, height_ratios=[1, 12], width_ratios=[1.55, 1.15, 1.15], hspace=0.02, wspace=0.06)
        tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off")
        axh.text(0.0, 0.5, f"{c+1}. {lab} · t={tt:.1f}s", fontsize=12.5, weight="bold", color=tc, va="center")
        ac = fig.add_subplot(col[1, 0]); ac.imshow(f["chase"][175:500, 340:700]); ac.axis("off")   # tighter zoom
        ac.set_title("chase cam", fontsize=10)
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1)            # natural aspect
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = YOLO.get(int(cls), ("obj", "#39f"))
            af.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=cc, lw=1.8))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=10)
        at = fig.add_subplot(col[1, 2]); cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=10)

    # C telemetry
    tg = outer[2].subgridspec(1, 4, wspace=0.24)
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    axi = fig.add_subplot(tg[0]); axi.plot(xt, smooth(xw), color=C_XPU, lw=1.6, label="XPU-RT"); axi.plot(rt, smooth(rw), color=C_ROS, lw=1.6, label="ROS")
    axi.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=11); axi.set_title("body-rate magnitude", fontsize=12.5, weight="bold"); axi.legend(fontsize=10, loc="upper right")
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0])); rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    axg = fig.add_subplot(tg[1]); axg.plot(xt, xg, color=C_XPU, lw=1.6); axg.plot(rt, rg, color=C_ROS, lw=1.6)
    axg.set_ylabel("goal heading (°)", fontsize=11); axg.set_title("nav goal heading", fontsize=12.5, weight="bold")
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1); rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    axs = fig.add_subplot(tg[2]); axs.plot(xt, smooth(xs, 11), color=C_XPU, lw=1.6); axs.plot(rt, smooth(rs, 11), color=C_ROS, lw=1.6)
    axs.set_ylabel("forward speed (m/s)", fontsize=11); axs.set_title("speed → ROS drops at crash", fontsize=12.5, weight="bold")
    for ax in (axi, axg, axs):
        ax.set_xlabel("time (s)", fontsize=11); ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=10)
        ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.4, zorder=0)
    axq = fig.add_subplot(tg[3]); vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(a.path_start, len(xxyz), 12)
    axq.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=1.0, zorder=0)
    axq.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis", angles="xy", scale_units="xy", scale=7.0, width=0.007, headwidth=4, headlength=5)
    axq.set_xlabel("along-aisle y (m)", fontsize=11); axq.set_ylabel("lateral x (m)", fontsize=11)
    axq.set_title("XPU-RT velocity (arrow = heading·speed)", fontsize=12.5, weight="bold"); axq.grid(True, color="0.92", lw=0.5); axq.tick_params(labelsize=10); axq.set_aspect("equal", adjustable="datalim")

    # D combined annotated Gantt
    axd = fig.add_subplot(outer[3])
    draw_combined_gantt(axd, json.load(open(a.sched_xpu))["dispatches"], json.load(open(a.sched_ros))["dispatches"])

    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
