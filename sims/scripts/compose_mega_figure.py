#!/usr/bin/env python3
"""The big warehouse HIL paper figure.

Tiers (top -> bottom):
  A. Top-down aisle overview (rotated to portrait so the whole long flight reads) — time-coloured
     drone PATH, the gates, and the DYNAMIC obstacles (moving people) drawn as clear directional
     arrows (start -> end) so they don't look like extra flight paths. 4 key moments marked;
     drop-lines connect them to row B.
  B. 4 key moments (in-flight, spread across the run) — closer chase-cam crop + FPV(YOLO
     gate/person) + the detailed 4x(8x8) cross-ToF (near=red, far=blue), matching the video.
  C. Full-run telemetry — velocity vectors (arrow dir = heading, length = forward speed),
     IMU |w|, and goal-cmd heading vs time.
  D. The annotated onboard K1 schedule from the REAL solver (plot_solver_gantt_annotated).

Pure matplotlib over figure_data.npz (+ frames/) — no Isaac.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

CMAP = matplotlib.colormaps["viridis"]
YOLO = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}
C_MOVER = "#9d4edd"          # purple — patrolling people (dynamic obstacles)


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
    """Rotate projected pixel coords the same way np.rot90(img, k) rotates the image."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    k %= 4
    if k == 0: return u, v
    if k == 1: return v, (W - 1 - u)          # CCW 90 ; new dims (H, W)
    if k == 2: return (W - 1 - u), (H - 1 - v)
    return (H - 1 - v), u                       # k == 3, CW 90 ; new dims (H, W)


def draw_topdown(ax, bg, K, cpos, cquat, xyz, gates, people_xy, t_norm, moments, t_s,
                 rot=0, flipx=False, gate_r=13, path_start=0, crash_step=0,
                 obst_pos=None, obst_kind=None, person_mask=None):
    H, W = bg.shape[:2]
    img = np.rot90(bg, rot)
    if flipx: img = img[:, ::-1]
    nH, nW = img.shape[:2]

    def T(u, v):                                 # project-space -> displayed-image pixel space
        u2, v2 = rot_uv(u, v, W, H, rot)
        if flipx: u2 = (nW - 1) - u2
        return u2, v2

    ax.imshow(img); ax.set_xlim(0, nW); ax.set_ylim(nH, 0); ax.axis("off")

    # highlight crate/box TOWERS with a marker overlaid on the realistic view (the user's box overlay)
    if obst_pos is not None and obst_kind is not None:
        TOWER = {"box", "crate", "klt", "pallet"}
        kinds = np.array([str(k) for k in obst_kind]); pm = person_mask if person_mask is not None else np.zeros(len(kinds), bool)
        base = obst_pos[0]
        tow = np.array([i for i in range(len(base)) if (kinds[i] in TOWER) and (not pm[i])
                        and (-11 <= base[i, 0] <= -5) and (4 <= base[i, 1] <= 23)])
        if len(tow):
            ou, ov, ook = project(K, cpos, cquat, base[tow]); ou, ov = T(ou, ov)
            for i in range(len(tow)):
                if ook[i] and -6 <= ou[i] <= nW+6 and -6 <= ov[i] <= nH+6:
                    ax.scatter(ou[i], ov[i], s=95, marker="s", facecolors="none",
                               edgecolors="#ffb14e", linewidths=1.6, alpha=0.9, zorder=5)
    pu, pv, ok = project(K, cpos, cquat, xyz)
    pu, pv = T(pu, pv)
    idx = np.arange(len(xyz))
    vis = ok & (pu > -40) & (pu < nW+40) & (pv > -40) & (pv < nH+40)
    vis &= (idx >= path_start)                    # trim the long pre-gate-1 approach
    crashed = crash_step and crash_step < len(xyz)
    if crashed:
        rem = ok & (idx > crash_step) & (pu > -40) & (pu < nW+40) & (pv > -40) & (pv < nH+40)
        ax.plot(pu[rem], pv[rem], color="0.72", lw=1.3, ls=(0, (2, 2)), alpha=0.65, zorder=2)  # never-flown plan
        vis &= (idx <= crash_step)
    # time-coloured drone path (white halo + viridis)
    ax.plot(pu[vis], pv[vis], color="white", lw=4.2, alpha=0.7, zorder=3)
    pts = np.column_stack([pu, pv])[vis]; tn = t_norm[vis]
    for i in range(len(pts)-1):
        ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], color=CMAP(tn[i]), lw=2.4, alpha=0.95, zorder=4)
    if crashed:                                   # big crash marker where the drone goes down
        cu, cv, co = project(K, cpos, cquat, xyz[crash_step:crash_step+1]); cu, cv = T(cu, cv)
        if co[0]:
            ax.scatter(cu[0], cv[0], s=430, marker="X", color="#e2231a", edgecolors="white",
                       linewidths=2.2, zorder=11)
            ax.annotate("✗ CRASH\nschedule infeasible", (cu[0], cv[0]), xytext=(cu[0]-19, cv[0]),
                        textcoords="data", ha="right", va="center", fontsize=8.5, weight="bold",
                        color="white", zorder=12, linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#b3121b", alpha=0.95, ec="white", lw=1.2))

    # dynamic obstacles (people): each PATROLS back-and-forth -> a dotted purple track with
    # direction arrows along it (arrows flip where the person reverses), drawn for every person.
    tcut = (crash_step + 1) if (crash_step and crash_step < len(xyz)) else len(xyz)
    for j in range(people_xy.shape[1]):
        px, py = people_xy[:tcut, j, 0], people_xy[:tcut, j, 1]
        uu, vv, o2 = project(K, cpos, cquat, np.column_stack([px, py, np.full(len(px), 2.0)]))
        uu, vv = T(uu, vv)
        m = o2 & (uu > -30) & (uu < nW+30) & (vv > -30) & (vv < nH+30)
        if m.sum() < 5:
            continue
        U, V = uu[m], vv[m]
        ax.plot(U, V, color=C_MOVER, lw=1.7, ls=(0, (1, 1.4)), alpha=0.95, zorder=3,
                label=("patrolling people (dotted = back-and-forth path)" if j == 0 else None))
        step = max(6, len(U) // 6)               # a few direction arrows along the patrol
        for i in range(step, len(U) - 2, step):
            di = min(4, len(U) - 1 - i)
            if np.hypot(U[i+di]-U[i], V[i+di]-V[i]) < 1.5:
                continue
            ax.annotate("", xy=(U[i+di], V[i+di]), xytext=(U[i], V[i]),
                        arrowprops=dict(arrowstyle="-|>", color=C_MOVER, lw=1.3, alpha=0.95),
                        zorder=4)
        ax.scatter(U[0], V[0], s=18, facecolors="white", edgecolors=C_MOVER, linewidths=1.3, zorder=5)
        ax.scatter(U[-1], V[-1], s=32, facecolors=C_MOVER, edgecolors="white", linewidths=1.1, zorder=5)

    def inb(u, v): return (-6 <= u <= nW+6) and (-6 <= v <= nH+6)
    named = {i for i in range(len(gates)) for (_, lab, _) in moments if f"G{i+1}" in lab}
    gu, gv, gok = project(K, cpos, cquat, gates); gu, gv = T(gu, gv)
    for i in range(len(gates)):
        if gok[i] and inb(gu[i], gv[i]):
            ax.add_patch(Circle((gu[i], gv[i]), gate_r, fill=False, ec="#ffd400", lw=2.4, zorder=6))
            if i not in named:                    # skip label if a moment descriptor already names it
                ax.text(gu[i]+gate_r+2, gv[i], f"G{i+1}", color="#ffd400", fontsize=8.5, weight="bold",
                        ha="left", va="center", zorder=6)
    mk_px = []
    for mi, (step, lab, _) in enumerate(moments):
        u, v, o = project(K, cpos, cquat, xyz[step:step+1]); u, v = T(u, v)
        if o[0] and inb(u[0], v[0]):
            ax.add_patch(Circle((u[0], v[0]), 15, fill=True, fc="black", ec="#ffd400", lw=2.2, zorder=8))
            ax.text(u[0], v[0], str(mi+1), color="white", fontsize=9.5, weight="bold",
                    ha="center", va="center", zorder=9)
            # descriptor next to the number (left of the marker, inside the aisle)
            ax.annotate(f"{mi+1} · {lab}", (u[0], v[0]), xytext=(u[0]-19, v[0]), textcoords="data",
                        ha="right", va="center", fontsize=7.6, weight="bold", color="white", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.28", fc="#111318", alpha=0.82, ec="#ffd400", lw=1.0))
            mk_px.append((u[0], v[0]))
        else:
            mk_px.append(None)
    # legend + colourbar-ish caption
    ax.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=2.6, label="drone path (colour = time)"),
                       Line2D([0], [0], color=C_MOVER, lw=1.7, ls=(0, (1, 1.4)),
                              label="patrolling people (back-and-forth)"),
                       Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=2, ls="none",
                              label="gate")],
              loc="lower left", fontsize=7.2, framealpha=0.9, handlelength=1.8)
    ax.text(0.0, 1.008, "Top-down aisle\nflight path + moving obstacles",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold", linespacing=1.25)
    return mk_px


def draw_topdown_schematic(ax, d, xyz, gates, t_s, tnorm, moments, crash_step=0, path_start=0):
    """Clean world-coordinate top-down (like the crash-trajectory figure): box-marked crate/box towers,
    the time-coloured flight path overlaid, gates, patrolling people. No camera projection / baked-in drone."""
    from matplotlib.patches import Rectangle, Circle
    TOWER = {"box", "crate", "klt", "pallet"}
    op = d["obst_pos"]; kind = np.array([str(k) for k in d["obst_kind"]])
    person = np.asarray(d["person_mask"]).astype(bool)
    XLO, XHI, YLO, YHI = -11.0, -5.0, 4.0, 23.0
    # static towers/crates (frame 0)
    for i in range(op.shape[1]):
        if person[i]:
            continue
        x, y = op[0, i, 0], op[0, i, 1]
        if not (XLO <= x <= XHI and YLO <= y <= YHI):
            continue
        if kind[i] in TOWER:
            ax.add_patch(Rectangle((x-0.24, y-0.24), 0.48, 0.48, facecolor="#a9743e",
                                   edgecolor="#6b4a26", lw=0.6, alpha=0.9, zorder=2))
        else:
            ax.plot(x, y, marker="^", color="#8a8a8a", ms=4, zorder=2)
    # patrolling people: dotted track over time
    tcut = (crash_step + 1) if (crash_step and crash_step < len(xyz)) else len(xyz)
    for i in np.where(person)[0]:
        px, py = op[:tcut, i, 0], op[:tcut, i, 1]
        m = (px >= XLO) & (px <= XHI) & (py >= YLO) & (py <= YHI)
        if m.sum() > 5:
            ax.plot(px[m], py[m], color=C_MOVER, lw=1.6, ls=(0, (1, 1.4)), alpha=0.9, zorder=3)
            ax.scatter(px[m][-1], py[m][-1], s=26, facecolors=C_MOVER, edgecolors="white", lw=1.0, zorder=4)
    # gates
    for j, g in enumerate(gates):
        ax.add_patch(Circle((g[0], g[1]), 0.55, fill=False, ec="#ffd400", lw=2.2, zorder=5))
        ax.text(g[0]+0.7, g[1], f"G{j+1}", color="#d4a800", fontsize=8, weight="bold", va="center", zorder=5)
    # flight path (time-coloured), trimmed + crash-truncated
    idx = np.arange(len(xyz)); vis = idx >= path_start
    if crash_step and crash_step < len(xyz):
        rem = idx > crash_step
        ax.plot(xyz[rem, 0], xyz[rem, 1], color="0.72", lw=1.2, ls=(0, (2, 2)), alpha=0.6, zorder=3)
        vis &= idx <= crash_step
    p = xyz[vis]; tn = tnorm[vis]
    ax.plot(p[:, 0], p[:, 1], color="white", lw=4.0, alpha=0.7, zorder=4)
    for i in range(len(p)-1):
        ax.plot(p[i:i+2, 0], p[i:i+2, 1], color=CMAP(tn[i]), lw=2.5, zorder=5)
    if len(p):
        ax.scatter(p[0, 0], p[0, 1], s=42, color=CMAP(0.0), ec="white", lw=1.2, zorder=6)
    # crash marker or finish star
    if crash_step and crash_step < len(xyz):
        cx, cy = xyz[crash_step, 0], xyz[crash_step, 1]
        ax.scatter(cx, cy, s=360, marker="X", color="#e2231a", ec="white", lw=2.0, zorder=8)
        ax.annotate("✗ CRASH\nschedule infeasible", (cx, cy), xytext=(cx-0.3, cy-1.1),
                    ha="right", va="center", fontsize=8, weight="bold", color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#b3121b", alpha=0.95, ec="white", lw=1.1))
    elif len(p):
        ax.scatter(p[-1, 0], p[-1, 1], s=150, marker="*", color=CMAP(1.0), ec="white", lw=1.2, zorder=6)
    # numbered moment markers
    mk = []
    for mi, (step, lab, _) in enumerate(moments):
        if crash_step and step > crash_step:
            mk.append(None); continue
        mx, my = xyz[step, 0], xyz[step, 1]
        ax.add_patch(Circle((mx, my), 0.34, fill=True, fc="black", ec="#ffd400", lw=2.0, zorder=8))
        ax.text(mx, my, str(mi+1), color="white", fontsize=9, weight="bold", ha="center", va="center", zorder=9)
        ax.annotate(f"{mi+1} · {lab}", (mx, my), xytext=(mx-0.45, my), ha="right", va="center",
                    fontsize=7.4, weight="bold", color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.26", fc="#111318", alpha=0.82, ec="#ffd400", lw=1.0))
        mk.append((mx, my))
    ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral x (m)", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_ylabel("along-aisle y (m)  —  flight direction ↑", fontsize=7)
    ax.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=2.6, label="drone path (colour = time)"),
                       Patch(fc="#a9743e", ec="#6b4a26", label="crate / box tower"),
                       Line2D([0], [0], color=C_MOVER, lw=1.6, ls=(0, (1, 1.4)), label="patrolling people"),
                       Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=2, ls="none", label="gate")],
              loc="lower left", fontsize=6.6, framealpha=0.9, handlelength=1.6)
    ax.text(0.0, 1.006, "Top-down aisle\nflight path + crate towers + people",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, weight="bold", linespacing=1.2)
    return mk


def crop_chase(img, box):
    r0, r1, c0, c1 = box
    return img[r0:r1, c0:c1]


def cross_tof(ax, tof, vmax=4.0):
    """Detailed cross-ToF like the video: 4 VL53L5CX (N/E/S/W), each an 8x8 grid, near=red far=blue."""
    g = np.full((24, 24), np.nan)
    g[0:8, 8:16] = tof[0]       # N (up)
    g[8:16, 16:24] = tof[1]     # E (right)
    g[16:24, 8:16] = tof[2]     # S (down)
    g[8:16, 0:8] = tof[3]       # W (left)
    ax.imshow(g, cmap="turbo_r", vmin=0, vmax=vmax, interpolation="nearest")
    for k in (8, 16):           # cell-block dividers
        ax.axhline(k-0.5, color="white", lw=0.8); ax.axvline(k-0.5, color="white", lw=0.8)
    for lbl, (yy, xx) in {"N": (0.5, 11.5), "E": (11.5, 22.5), "S": (22.5, 11.5), "W": (11.5, 1.0)}.items():
        ax.text(xx, yy, lbl, color="white", fontsize=6.4, weight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(23.5, -0.5)
    ax.set_title("cross-ToF (near=red)", fontsize=6.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gantt", default=None)
    ap.add_argument("--td-rot", type=int, default=1)
    ap.add_argument("--td-flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--crash-step", type=int, default=0,
                    help="if >0: ROS variant — truncate the flight here (schedule infeasible) + crash marker")
    ap.add_argument("--out", default="out/paper_figure_mega")
    args = ap.parse_args()
    d = np.load(os.path.join(args.data_dir, "figure_data.npz"), allow_pickle=True)
    poses = d["poses"]; xyz = poses[:, :3]; T = len(poses)
    t_s = d["t_s"]; tnorm = (t_s - t_s.min()) / max(1e-6, (t_s.max() - t_s.min()))
    op = d["obst_pos"]; pm = np.asarray(d["person_mask"]).astype(bool)
    people = op[:, pm, :]
    gates = d["gates_world"]; fs = d["frame_steps"]
    cbp = os.path.join(args.data_dir, "clean_bg.npz")
    cb = np.load(cbp) if os.path.exists(cbp) else d
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]

    # 4 in-flight moments (chronological, spread) — no dull t=0 start
    def frame_idx(step): return int(np.argmin(np.abs(fs - step)))
    moments = [(260, "gate G1", "1.5 m"), (620, "gate G2 — person nearby", "1.2 m"),
               (970, "gate G3 — tight crate", "0.55 m"), (1180, "final gate stretch", "1.1 m")]
    fr = [np.load(os.path.join(args.data_dir, f"frames/frame_{frame_idx(s):03d}.npz"), allow_pickle=True)
          for s, _, _ in moments]
    CHASE_BOX = (140, 530, 200, 760)             # closer crop, centred on the drone

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    have_g = bool(args.gantt and os.path.exists(args.gantt))
    # outer: hero (tall top-down | 2x2 moments) / telemetry / gantt
    hr = ([8.6, 1.5, 2.7] if have_g else [8.6, 1.5])
    fig = plt.figure(figsize=(12.6, 15.9))
    outer = fig.add_gridspec(len(hr), 1, height_ratios=hr, hspace=0.11,
                             left=0.055, right=0.985, top=0.985, bottom=0.028)

    hero = outer[0].subgridspec(1, 2, width_ratios=[2.35, 7.65], wspace=0.05)

    # ---- A top-down sidebar (rotated to portrait; tall & prominent) ----
    axt = fig.add_subplot(hero[0])
    top_px = draw_topdown(axt, ov_bg, ovK, ovpos, ovquat, xyz, gates, people, tnorm, moments, t_s,
                          rot=args.td_rot, flipx=args.td_flipx, path_start=args.path_start,
                          crash_step=args.crash_step, obst_pos=d["obst_pos"], obst_kind=d["obst_kind"],
                          person_mask=np.asarray(d["person_mask"]).astype(bool))
    crashed = args.crash_step and args.crash_step < T
    t_crash = t_s[args.crash_step] if crashed else None
    smm = matplotlib.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.Normalize(0, t_s.max()))
    cbar = fig.colorbar(smm, ax=axt, fraction=0.05, pad=0.02); cbar.set_label("flight time (s)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ---- B 4-moment detail in a 2x2 (closer chase + larger FPV/YOLO + detailed ToF) ----
    mgrid = hero[1].subgridspec(2, 2, hspace=0.16, wspace=0.12)
    b_top = []
    for c in range(4):
        cell = mgrid[c // 2, c % 2].subgridspec(2, 2, height_ratios=[1.32, 1.0],
                                                width_ratios=[1.55, 1.0], hspace=0.13, wspace=0.07)
        f = fr[c]; step, lab, dd = moments[c]
        if crashed and step > args.crash_step:    # this moment is past the crash -> never happens
            axx = fig.add_subplot(mgrid[c // 2, c % 2]); axx.set_xticks([]); axx.set_yticks([])
            axx.set_facecolor("#2a0e0e")
            for sp in axx.spines.values():
                sp.set_color("#b3121b"); sp.set_linewidth(1.6)
            axx.text(0.5, 0.60, "✗ FLIGHT ABORTED", ha="center", va="center", transform=axx.transAxes,
                     color="#ff6b6b", fontsize=12, weight="bold")
            axx.text(0.5, 0.40, f"drone crashed at t={t_crash:.1f}s\nthis moment never happens under ROS",
                     ha="center", va="center", transform=axx.transAxes, color="#ffb3b3",
                     fontsize=8, linespacing=1.35)
            axx.set_title(f"{c+1}. {lab} · t={t_s[step]:.1f}s", fontsize=8.2, weight="bold", color="0.55")
            b_top.append(axx)
            continue
        ac = fig.add_subplot(cell[0, :]); ac.imshow(crop_chase(f["chase"], CHASE_BOX)); ac.axis("off")
        ac.set_title(f"{c+1}. {lab} · t={t_s[step]:.1f}s · {dd}", fontsize=8.2, weight="bold")
        b_top.append(ac)
        af = fig.add_subplot(cell[1, 0]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="auto")
        dets = [dd for dd in f["det"] if dd[5] >= 0.4]
        best = {}                                 # label only the top-conf box per class (no overlap)
        for dd in dets:
            c0 = int(dd[0])
            if c0 not in best or dd[5] > best[c0][5]:
                best[c0] = dd
        for dd in dets:                           # draw every box
            cls, x0, y0, x1, y1, cf = dd
            _, col = YOLO.get(int(cls), ("obj", "#39f"))
            af.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=col, lw=1.4))
        for cls, dd in best.items():              # one label per class: gate on top, person on bottom
            _, x0, y0, x1, y1, cf = dd
            nm, col = YOLO.get(int(cls), ("obj", "#39f"))
            if int(cls) == 1:                     # person -> label under its box
                af.text(x0, y1+1, f"{nm} {cf:.2f}", color="black", fontsize=5.8, weight="bold",
                        bbox=dict(fc=col, ec="none", pad=0.4), va="top")
            else:                                 # gate -> label above its box
                af.text(x0, y0-1, f"{nm} {cf:.2f}", color="black", fontsize=5.8, weight="bold",
                        bbox=dict(fc=col, ec="none", pad=0.4), va="bottom")
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=7.2)
        at = fig.add_subplot(cell[1, 1]); cross_tof(at, f["tof"])

    # (no drop-lines: the shared 1-4 numbering + descriptors tie each marker to its panel)

    # ---- C telemetry: velocity vectors + IMU + goal heading ----
    tele = outer[1].subgridspec(1, 4, wspace=0.32)
    vxy = np.gradient(xyz[:, :2], t_s, axis=0)
    speed = np.linalg.norm(vxy, axis=1)
    axv = fig.add_subplot(tele[0, 0:2])
    Tv = (args.crash_step + 1) if crashed else T
    sel = np.arange(0, Tv, 10)
    # unrolled top-view: x = along-aisle progress (world y), y = lateral (world x); arrow = velocity
    q = axv.quiver(xyz[sel, 1], xyz[sel, 0], vxy[sel, 1], vxy[sel, 0], t_s[sel],
                   cmap="viridis", angles="xy", scale_units="xy", scale=6.0, width=0.004,
                   headwidth=4, headlength=5)
    axv.plot(xyz[:Tv, 1], xyz[:Tv, 0], color="0.75", lw=0.7, zorder=0)
    if crashed:
        axv.plot(xyz[Tv-1:, 1], xyz[Tv-1:, 0], color="0.8", lw=0.8, ls=(0, (2, 2)), zorder=0)
        axv.scatter(xyz[args.crash_step, 1], xyz[args.crash_step, 0], s=110, marker="X",
                    color="#e2231a", edgecolors="white", linewidths=1.4, zorder=5)
    axv.set_xlabel("along-aisle position y (m)", fontsize=7); axv.set_ylabel("lateral x (m)", fontsize=7)
    axv.set_title("velocity — arrow direction = heading, length = forward speed  "
                  f"(peak {speed.max():.2f} m/s)", fontsize=7.4, weight="bold")
    axv.grid(True, color="0.92", lw=0.5); axv.tick_params(labelsize=6)
    axv.set_aspect("equal", adjustable="datalim")

    axi2 = fig.add_subplot(tele[0, 2])
    w = d["imu_w"]
    def _smooth(y, win=15):                          # moving-average envelope (raw is bang-bang noise)
        if len(y) < win:
            return y
        k = np.ones(win) / win
        return np.convolve(y, k, mode="same")
    for k, (lab, cc) in enumerate([("ω_roll", "#c44e52"), ("ω_pitch", "#e08a2f"), ("ω_yaw", "#3f6fb0")]):
        axi2.plot(t_s[:Tv], w[:Tv, k], color=cc, lw=0.5, alpha=0.18, zorder=1)          # raw, faint
        axi2.plot(t_s[:Tv], _smooth(w[:Tv, k]), color=cc, lw=1.2, label=lab, zorder=3)  # smoothed
    axi2.plot(t_s[:Tv], _smooth(np.linalg.norm(w[:Tv], axis=1)), color="0.25", lw=1.0,
              ls=(0, (3, 2)), label="|ω|", zorder=3)
    axi2.set_ylabel("IMU gyro (rad/s, 0.15 s smoothed)", fontsize=7); axi2.set_xlim(t_s.min(), t_s.max())
    # zoom past the takeoff transient (t<0.6 s spikes to ~10 rad/s and squashes everything else)
    steady = w[t_s > 0.6]
    if len(steady):
        lim = 1.15 * float(np.abs(_smooth(np.linalg.norm(w, axis=1))[t_s > 0.6]).max())
        axi2.set_ylim(-lim, lim)
        axi2.text(0.02, 0.97, "(raw faint; takeoff spike clipped)", transform=axi2.transAxes, fontsize=5.2,
                  color="0.5", va="top", ha="left")
    axi2.legend(fontsize=5.2, ncol=2, loc="upper right", framealpha=0.85, handlelength=1.1,
                columnspacing=0.9, borderpad=0.3)
    axi2.grid(True, color="0.9", lw=0.5)
    axg = fig.add_subplot(tele[0, 3])
    goal = d["goal_cmd"]; ghead = np.degrees(np.arctan2(goal[:, 1], goal[:, 0]))
    axg.plot(t_s[:Tv], ghead[:Tv], color="#2f8f4e", lw=1.0)
    axg.set_ylabel("goal heading (°)", fontsize=7); axg.set_xlim(t_s.min(), t_s.max())
    axg.grid(True, color="0.9", lw=0.5)
    for a in (axi2, axg):
        a.set_xlabel("time (s)", fontsize=7); a.tick_params(labelsize=6)
        for step, _, _ in moments:
            if not (crashed and step > args.crash_step):
                a.axvline(t_s[step], color="0.4", ls=(0, (2, 2)), lw=0.7)
        if crashed:
            a.axvspan(t_crash, t_s.max(), color="#f4d7d7", alpha=0.5, zorder=0)
            a.axvline(t_crash, color="#e2231a", lw=1.5, zorder=3)
            a.text(t_crash, a.get_ylim()[1], " CRASH", color="#e2231a", fontsize=6.5,
                   weight="bold", va="top", ha="left")

    # ---- D real-solver annotated Gantt ----
    if have_g:
        ax_g = fig.add_subplot(outer[2]); ax_g.imshow(plt.imread(args.gantt)); ax_g.axis("off")

    fig.savefig(args.out + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print("wrote", args.out + ".png and .pdf")


if __name__ == "__main__":
    main()
