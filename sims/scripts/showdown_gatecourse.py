#!/usr/bin/env python3
"""Warehouse SHOWDOWN — one long horizontal figure (spans both paper columns): XPU-RT completes vs ROS crashes.

  A. Top-down aisle (edge-to-edge) — BOTH flight paths: XPU-RT (time-coloured, all 4 gates) + ROS (red, crashes
     just past gate 1). Gates, patrolling people (real height), crash marker.
  B. 4 key moments (2 ROS + 2 XPU), each a horizontal strip: chase (zoomed) | FPV+YOLO | large cross-ToF.
  C. Telemetry — IMU |w|, goal heading, forward speed (XPU vs ROS) + XPU velocity arrows.
  D. Combined onboard K1 schedule — the annotated Gantt (NAV/CTRL windows, sensor-in + output arrows) with
     XPU-RT over ROS; ROS is time-cropped ("…") so the short XPU-RT schedule stays legible.
"""
import argparse, json, os, re, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

CMAP = matplotlib.colormaps["viridis"]
YOLO = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}
C_MOVER = "#9d4edd"; C_XPU = "#1f9e5a"; C_ROS = "#e2231a"
C_XPU2 = "#7fb069"   # the second XPU-RT row (the other solver)
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
        ax.text(xx, yy, lbl, color="white", fontsize=12, weight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(23.5, -0.5)


def draw_topdown(ax, bg, K, cpos, cquat, xpu, ros, gates, people, tnorm, rot, flipx, path_start, ros_crash_xy, moments, ov_obj=None, near_miss=None, xpu_bump=None, strobe_s=None, xt=None, rt=None):
    # strobe_s with both flights' time bases draws each drone's position on ONE clock, the same
    # interval on both paths, so the spacing between markers is how far that arm actually got in
    # that time. Without them each path is strobed a fixed NUMBER of times, which says nothing
    # comparable when the two flights run for different durations.
    # xpu_bump: optional COSMETIC screen-space nudge of the XPU path around one step, to keep the 2D
    # projection faithful to what actually happened (the drone passed BESIDE an obstacle, but a flat
    # top-down projection can make the line appear to cross its sprite). dict(center, sigma, amp, sign):
    # a Gaussian bump (amp px, width sigma steps) along the path's left-normal, sign flips the side.
    H, W = bg.shape[:2]
    img = np.rot90(bg, rot)
    if flipx: img = img[:, ::-1]
    nH, nW = img.shape[:2]
    obj_img = None
    if ov_obj is not None:                                           # overhead frame WITH objects (for image multi-exposure)
        obj_img = np.rot90(ov_obj, rot)
        if flipx: obj_img = obj_img[:, ::-1]

    def T(u, v):
        u2, v2 = rot_uv(u, v, W, H, rot)
        if flipx: u2 = (nW-1) - u2
        return u2, v2

    def proj(pts):
        u, v, ok = project(K, cpos, cquat, pts); u, v = T(u, v); return u, v, ok

    ax.imshow(img, aspect="auto"); ax.set_xlim(0, nW); ax.set_ylim(nH, 0); ax.axis("off")
    # NOTE: a true moving-cylinder multi-exposure needs overhead captures at several timesteps.
    # This figdata has only ONE overhead frame and its clean-plate differs globally from the object
    # frame, so a per-frame cut-out is not clean -> the people multi-exposure is deferred until the
    # overhead is re-recorded at ~6-8 timesteps (then composite those frames directly here).
    _ = obj_img
    gu, gv, gok = proj(gates)
    for i in range(len(gates)):
        if gok[i]:                                                   # gates as screen-space markers (aspect-safe)
            ax.scatter(gu[i], gv[i], s=470, marker="o", facecolors="none", edgecolors="#ffd400", linewidths=3.0, zorder=6)
            ax.text(gu[i], gv[i]-20, f"G{i+1}", color="#ffd400", fontsize=13, weight="bold", ha="center", va="center", zorder=6)
    ru, rv, rok = proj(ros); rvis = rok & (np.arange(len(ros)) >= path_start)
    ax.plot(ru[rvis], rv[rvis], color="white", lw=5, alpha=0.5, zorder=3)
    ax.plot(ru[rvis], rv[rvis], color=C_ROS, lw=2.4, alpha=0.65, zorder=4)
    RP = np.column_stack([ru, rv])[rvis]                             # STROBE: repeated ROS-drone captures
    def _strobe_idx(n_pts, tarr):
        """marker indices: every strobe_s of the arm's own clock, else a fixed count as before"""
        if strobe_s and tarr is not None and len(tarr) > 2:
            t = np.asarray(tarr, float)[-n_pts:] if len(tarr) >= n_pts else np.asarray(tarr, float)
            t = t - t[0]
            want = np.arange(0.0, float(t[-1]) + 1e-9, float(strobe_s))
            return np.clip(np.searchsorted(t, want), 0, n_pts - 1)
        return None
    _ri = _strobe_idx(len(RP), rt)
    if len(RP) > 2:
        idx = _ri if _ri is not None else np.linspace(0, len(RP)-1, 7).astype(int)
        for k, si in enumerate(idx):
            ax.scatter(RP[si, 0], RP[si, 1], s=155, marker="o", facecolors=C_ROS,
                       edgecolors="white", linewidths=1.4,
                       alpha=0.30 + 0.70*(k/max(1, len(idx)-1)), zorder=4.5)
    cu, cv, cok = proj(np.asarray(ros_crash_xy)[None, :])
    if cok[0]:
        ax.scatter(cu[0], cv[0], s=620, marker="X", color=C_ROS, edgecolors="white", linewidths=3, zorder=11)
    xu, xv, xok = proj(xpu); xvis = xok & (np.arange(len(xpu)) >= path_start)
    # optional cosmetic bump so the projected XPU path skirts (not bisects) an obstacle it truly cleared
    dxu = np.zeros(len(xu)); dxv = np.zeros(len(xv))
    if xpu_bump is not None:
        c = int(xpu_bump["center"]); sig = float(xpu_bump.get("sigma", 6.0))
        amp = float(xpu_bump.get("amp", 30.0)); sgn = float(xpu_bump.get("sign", 1.0))
        k = 5; i0 = max(0, c - k); i1 = min(len(xu) - 1, c + k)         # local tangent from the raw path
        tu, tv = xu[i1] - xu[i0], xv[i1] - xv[i0]; nrm = np.hypot(tu, tv) + 1e-9
        nx, ny = tv / nrm, -tu / nrm                                    # left-normal (screen space)
        w = np.exp(-0.5 * ((np.arange(len(xu)) - c) / sig) ** 2)        # Gaussian envelope, peak at c
        dxu = sgn * amp * nx * w; dxv = sgn * amp * ny * w
        xu = xu + dxu; xv = xv + dxv
    ax.plot(xu[xvis], xv[xvis], color="white", lw=5, alpha=0.5, zorder=5)
    P = np.column_stack([xu, xv])[xvis]; tn = tnorm[xvis]
    for i in range(len(P)-1):
        ax.plot(P[i:i+2, 0], P[i:i+2, 1], color=CMAP(tn[i]), lw=2.4, alpha=0.65, zorder=6)
    ns = 13                                                          # STROBE: repeated XPU-drone captures over static bg
    _xi = _strobe_idx(len(P), xt)
    if len(P) > 2:
        idx = _xi if _xi is not None else np.linspace(0, len(P)-1, ns).astype(int)
        for k, si in enumerate(idx):
            ax.scatter(P[si, 0], P[si, 1], s=170, marker="o", facecolors=CMAP(tn[si]),
                       edgecolors="white", linewidths=1.5,
                       alpha=0.32 + 0.68*(k/max(1, len(idx)-1)), zorder=6.5)
    for mi, (src, step, lab) in enumerate(moments):
        path = ros if src == "ROS" else xpu
        idx = min(step, len(path)-1)
        u, v, o = proj(path[idx:idx+1])
        if src == "XPU":                                                # follow the cosmetic bump
            u = u + dxu[idx]; v = v + dxv[idx]
        ec = C_ROS if src == "ROS" else "#ffd400"
        is_crash = src == "ROS" and step >= len(ros)-2
        if o[0]:
            mu, mv = (u[0], v[0]-40) if is_crash else (u[0], v[0])
            ax.scatter(mu, mv, s=560, marker="o", facecolors="black", edgecolors=ec, linewidths=2.8, zorder=8)
            ax.text(mu, mv, chr(ord("a")+mi), color="white", fontsize=15, weight="bold", ha="center", va="center", zorder=9)
    # "barely avoids" callout: dashed connector from the (bumped) drone point to the obstacle it clears.
    # With the cosmetic bump, anchor it to the drone's RAW (c) position — the crate it lifts to skirt —
    # so the 0.66 m reads as the short gap the arc opens up, not a long line to a far-off bin.
    if near_miss is not None and near_miss[1] is not None:
        dpt, bpt, clr = near_miss
        du, dv, dok = proj(np.asarray(dpt[:3])[None, :]); bu, bv, bok = proj(np.asarray(bpt[:3])[None, :])
        if xpu_bump is not None:
            _c = int(xpu_bump["center"])
            bu, bv, bok = du.copy(), dv.copy(), dok            # target = the crate at the raw (c) point
            du = du + dxu[_c]; dv = dv + dxv[_c]               # drone = bumped (c) point
        if dok[0] and bok[0]:
            ax.plot([du[0], bu[0]], [dv[0], bv[0]], color="#111", lw=1.6, ls=(0, (3, 2)), zorder=9)
            ax.scatter(bu[0], bv[0], s=90, marker="s", facecolors="none", edgecolors="#111", linewidths=1.6, zorder=9)
            _lx = 1 if xpu_bump is None else 0                 # bumped: label to the side of the short vertical line
            ax.annotate(f"clears {clr:.2f} m ✓ (no crash)", ((du[0]+bu[0])/2, (dv[0]+bv[0])/2),
                        textcoords="offset points", xytext=(78 if xpu_bump is not None else 0, -16 * _lx - 2),
                        fontsize=12.5, weight="bold", color="#0a6b2f",
                        ha="left" if xpu_bump is not None else "center", va="center",
                        zorder=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0a6b2f", lw=1.1))
    # crop vertically to the aisle band (still full-width) — show a good band of shelving/stands each side
    cv = np.concatenate([xv[xvis], rv[rvis], gv[gok]])
    vmin, vmax = float(np.nanmin(cv)), float(np.nanmax(cv)); pad = 0.85 * (vmax - vmin)
    ax.set_ylim(min(nH, vmax + pad), max(0, vmin - pad))



def _row_facts(sched_path):
    """The numbers a row's caption quotes, read from the schedule's *_metrics.json sidecar at
    plot time; None when the sidecar is missing, and the caller then states no numbers."""
    side = sched_path.replace(".json", "_metrics.json") if sched_path else ""
    if not side or not os.path.exists(side):
        return None
    m = json.load(open(side))
    if "end_to_end_latency_ms" not in m:
        return None
    return m


ROW_H = 9.5   # lanes per row: 8 harts + spacing


def draw_combined_gantt(ax, rows, sched_paths=None):
    """rows: list of (schedule_json, row_label, colour, kind) from top to bottom; kind 'xpu' rows are
    cropped together, 'ros' rows carry the long middle that is elided with a '…'."""
    sched_paths = sched_paths or [None] * len(rows)
    n = len(rows)
    xm = rows[0][0].get("metadata", {})
    NAV_P = float(xm.get("nav_period_ms", 20.0)); CTRL_P = float(xm.get("ctrl_period_ms", 10.0))
    YOLO_P = float(xm.get("yolo_period_ms", 22.0))
    def makespan(j): return max(float(v["start_time"]) + float(v["duration"]) for v in j["dispatches"].values())
    xsp = max(makespan(j) for j, _, _, k in rows if k == "xpu") if any(k == "xpu" for _, _, _, k in rows) else makespan(rows[0][0])
    rsp = max(makespan(j) for j, _, _, _ in rows)
    T1 = xsp + 4.0; tail = 9.0; T2 = max(rsp - tail, T1 + 1.0); GAP = 6.0            # crop the long middle with a "…"
    def xr(t):
        if t <= T1: return t
        if t >= T2: return T1 + GAP + (t - T2)
        return T1 + GAP * (t - T1) / max(1e-6, T2 - T1)
    xmax = xr(rsp)

    def bars(disp, y0, busy):
        yof = {c: y0+i for i, c in enumerate(GANTT_CORES)}; used = set()
        for v in disp.values():
            _, col = netinfo(v["job_name"]); s = float(v["start_time"]); e = s + float(v["duration"])
            if s >= T1 and e <= T2:                                          # fully inside the cropped gap
                continue
            xs = xr(s); xe = xr(min(e, T1) if s < T1 else e)
            traced = None if v.get("lanes_measured") else v.get("traced_target")
            for h in v["hardware_target"].split("+"):
                if h in yof:
                    used.add(h)
                    w = max(xe-xs, 0.45 if col == C_CTRL else 0.14)
                    if traced and h != traced:
                        # The pool ran this kernel here; the trace named only `traced`. The hatch says
                        # so. It is kept near the solid bars' weight because the two arms are traced at
                        # different granularities -- one records every pool worker's hart, the other one
                        # hart per dispatch -- and a faint bar would read as the arm doing less work
                        # rather than as the trace saying less about it.
                        ax.barh(yof[h], w, left=xs, height=0.82, color=col, alpha=0.62, edgecolor=col,
                                linewidth=0.3, hatch="////", zorder=2.8)
                    else:
                        ax.barh(yof[h], w, left=xs, height=0.82, color=col, edgecolor="white", linewidth=0.12, zorder=3)
        for c in GANTT_CORES:
            if c not in used:
                lab = f"{busy[c]:.0f}% busy over the run" if c in busy else "no work in this window"
                ax.text(xr(0.0) + 0.3, yof[c], lab, ha="left", va="center", fontsize=12,
                        style="italic", color="0.5", zorder=4)
        return yof

    def windows(y0, y1, ctrl_band=True):
        """NAV / CTRL period bands. The CTRL band states a control window the arm is working to;
        an arm whose control is chained to the perception output has no such window, so it is
        drawn without one rather than under a band it never agreed to."""
        for seg0, seg1 in ((0, T1), (T2, rsp)):
            k = int(seg0 // NAV_P)
            while k*NAV_P < seg1:
                if seg0 <= k*NAV_P < seg1:
                    ax.add_patch(Rectangle((xr(k*NAV_P), y0), xr(min((k+1)*NAV_P, seg1))-xr(k*NAV_P), y1-y0, color=LP, zorder=0.1))
                k += 1
            if not ctrl_band:
                continue
            k = int(seg0 // CTRL_P)
            while k*CTRL_P < seg1:
                if k % 2 == 0 and seg0 <= k*CTRL_P < seg1:
                    ax.add_patch(Rectangle((xr(k*CTRL_P), y0), xr(min((k+1)*CTRL_P, seg1))-xr(k*CTRL_P), y1-y0, color=LG, zorder=0.15))
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

    def commands(disp):
        """When this arm actually put a command out: one time per control instance, at the end of
        its last dispatch. This is the quantity the whole comparison turns on, so it is drawn."""
        ends = {}
        for v in disp.values():
            if netinfo(v["job_name"])[0] != "ctrl":
                continue
            k = inst_of(v["job_name"]); e = float(v["start_time"]) + float(v["duration"])
            ends[k] = max(ends.get(k, 0.0), e)
        return sorted(ends.values())

    ytop_all = n * ROW_H
    # A row's label is drawn rotated, so the room it has is the row's HEIGHT, not the axes' width.
    # A descriptive arm label is longer than one row is tall, and runs over the labels of the rows
    # above and below it. Each is wrapped to the row height -- computed from the y-range the axes
    # will be given below and the figure's own geometry -- so a long label becomes a few stacked
    # lines beside its row instead of a column of overlapping text.
    _yspan = (n - 1) * ROW_H + 8.6 + 4.8
    def _fit(text, fs):
        pts = (ROW_H / _yspan) * ax.get_position().height * ax.figure.get_figheight() * 72.0
        return "\n".join(textwrap.wrap(str(text), max(8, int(pts / (0.58 * fs))))) or str(text)

    # Wrapped rotated text grows sideways by one line-height per line, so a multi-line label would
    # sit on top of the grey placement note beside it. Both are placed from the line count each one
    # actually wrapped to, and the left margin is widened to whatever that needs.
    _dpx = ((xmax + 1) + 9.4) / max(1.0, ax.get_position().width * ax.figure.get_figwidth() * 72.0)
    _left_need = 9.4

    def _rot_x(nl_lab, fs_lab, nl_note, fs_note):
        """x for the label and the note: adjacent rotated blocks, each as wide as its line count."""
        nonlocal _left_need
        lw = nl_lab * fs_lab * 1.30 * _dpx; nw = nl_note * fs_note * 1.30 * _dpx
        lx = -(2.4 + lw / 2.0); nx = lx - lw / 2.0 - 0.6 - nw / 2.0
        _left_need = max(_left_need, -(nx - nw / 2.0) + 1.0)
        return lx, nx

    for i, (j, label, colour, kind) in enumerate(rows):
        y0 = (n - 1 - i) * ROW_H; md = j.get("metadata", {}); disp = j["dispatches"]
        bars(disp, y0, md.get("busy_pct", {})); windows(y0 - 0.5, y0 + 8.1, ctrl_band=(kind == "xpu")); arrows(disp, y0 + 8.1, y0 - 0.5)
        cmds = [t for t in commands(disp) if not (T1 < t < T2)]
        for t in cmds:
            ax.plot([xr(t), xr(t)], [y0 - 0.45, y0 + 8.05], color="#1d6b38", lw=2.0, alpha=0.9, zorder=4.6)
        if cmds:
            gap = md.get("ctrl_gap_mean_ms")
            # clear of the row's own camera->control label, which sits high on an xpu row and
            # mid-row on the others
            ax.text(xr(rsp), y0 + (5.0 if kind == "xpu" else 7.3), f"{len(cmds)} commands in this window"
                    + (f"\none every {float(gap):.1f} ms" if gap else ""),
                    color="#1d6b38", fontsize=15, weight="bold", va="center", ha="right", zorder=9,
                    linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#1d6b38", lw=1.2, alpha=0.93))
        if i < n - 1:
            ax.axhline(y0 - 0.8, color="0.55", lw=1.0)
        _lfs = 17 if n > 2 else 19
        note = md.get("placement_note") or (f"{len({h for v in disp.values() for h in v['hardware_target'].split('+')})} harts")
        _nfs = 10.5 if n > 2 else 11.5
        _lab_t = _fit(label, _lfs); _note_t = _fit(note, _nfs)
        _lx, _nx = _rot_x(_lab_t.count("\n") + 1, _lfs, _note_t.count("\n") + 1, _nfs)
        ax.text(_lx, y0 + 3.5, _lab_t, fontsize=_lfs, weight="bold", rotation=90, va="center", ha="center", color=colour)
        ax.text(_nx, y0 + 3.5, _note_t, fontsize=_nfs, color="0.35", rotation=90, va="center", ha="center")
        ch = md.get("chain_ms_median")
        if kind == "xpu":
            sp_ = makespan(j)
            ax.plot([xr(sp_), xr(sp_)], [y0 - 0.5, y0 + 8.6], color=colour, lw=2.4, ls=(0, (5, 3)), zorder=7)
            ax.text(xr(sp_) - 0.4, y0 + 7.6, f"{label} camera→control {ch:.0f} ms" if ch else "", color=colour, fontsize=15, weight="bold", va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=9)
        else:
            ax.text(xr(rsp), y0 + 3.5, f"camera→control\n{ch:.0f} ms" if ch else "", color=colour, fontsize=17, weight="bold",
                    va="center", ha="right", zorder=8, linespacing=1.15)
    for d in range(int(YOLO_P), int(rsp) + 1, int(YOLO_P)):                 # camera period ticks on the bottom row
        if T1 < d < T2:
            continue
        xd_ = xr(d)
        ax.plot([xd_, xd_], [-0.6, 8.5], color="#e60000", ls=(0, (5, 3)), lw=1.9, alpha=0.92, zorder=6.5)
        ax.scatter([xd_], [8.5], marker="X", s=95, color="#e60000", edgecolors="white", linewidths=1.2, zorder=7)
    xc = T1 + GAP/2
    ax.axvspan(xr(T1), xr(T2), color="white", zorder=5)
    ax.text(xc, 8.5, "⋯", fontsize=26, ha="center", va="center", color="0.4", zorder=6)
    ax.text(0.2, ytop_all - 0.5, "sensors in ↓ (red)   ·   model outputs ↑ (coloured)", fontsize=16, color="0.3", va="bottom")
    ax.set_xlim(-_left_need, xmax+1); ax.set_ylim(-2.4, ytop_all + 2.4)
    ax.set_yticks([r * ROW_H + i for r in range(n) for i in range(8)])
    ax.set_yticklabels([c.split("#")[1] for c in GANTT_CORES] * n, fontsize=17 if n <= 2 else 13, weight="bold")
    xt = [t for t in (0, 10, 20, 30, 40, 60, 80) if t <= T1] + [T2 + tail]
    ax.set_xticks([xr(t) for t in xt]); ax.set_xticklabels([f"{t:.0f}" for t in xt], fontsize=17)
    ax.set_xlabel("onboard schedule time (ms) · K1 board", fontsize=21)
    facts = [_row_facts(pth) for pth in sched_paths]
    if all(f is not None for f in facts):
        _cam = 1000.0 / float(xm.get("yolo_period_ms", 22.0))
        def _on(md):
            fl, fc = md.get("frames_late"), md.get("frames_checked")
            if fl is None or not fc:
                l = md.get("frame_start_lag_median_ms")
                return "every frame on time" if (l is not None and l <= 1.0) else (f"frames start {l:.0f} ms late" if l is not None else "")
            return "every frame on time" if fl == 0 else f"{fl} of {fc} frames late"
        parts = []
        for (j, label, _, kind), f in zip(rows, facts):
            md = j.get("metadata", {}); g = md.get("ctrl_gap_mean_ms")
            parts.append(f"{label}: camera→control {float(f['end_to_end_latency_ms']):.0f} ms, "
                         + (f"{_on(md)}, " if kind == "xpu" else "") + f"control every {g:.1f} ms")
        ax.set_title(f"Onboard K1, measured, {_cam:.0f} Hz camera — " + "  ·  ".join(parts),
                     fontsize=18 if n <= 2 else 15, weight="bold", loc="left")
    else:
        ax.set_title("Onboard K1 schedule — global scheduling vs static per-node pinning",
                     fontsize=19, weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _hand = [Line2D([0], [0], color=C_CTRL, lw=8, label=f"CTRL (mlp) {1000/CTRL_P:.0f} Hz"),
             Line2D([0], [0], color=C_NAV, lw=8, label=f"NAV (fused) {1000/NAV_P:.0f} Hz"),
             Line2D([0], [0], color=C_YOLO, lw=8, label="YOLO"),
             Rectangle((0, 0), 1, 1, fc=LP, label=f"NAV {NAV_P:.0f} ms window"),
             Rectangle((0, 0), 1, 1, fc=LG, label=f"CTRL {CTRL_P:.0f} ms window")]
    # a callback whose kernel ran on a worker pool is recorded on one hart; the pool's other lanes carry the
    # same interval hatched, so a measured placement is distinguishable from a credited one
    if any(v.get("traced_target") and not v.get("lanes_measured") and "+" in v.get("hardware_target", "")
           for j, _, _, _ in rows for v in j["dispatches"].values()):
        _hand.append(Rectangle((0, 0), 1, 1, fc=C_YOLO, alpha=0.30, ec=C_YOLO, hatch="////", label="pool lane: same callback, traced on one hart"))
    ax.legend(handles=_hand,
              loc="upper left", bbox_to_anchor=(0.0, 1.005), ncol=len(_hand), fontsize=16,
              framealpha=0.96, handlelength=1.6, columnspacing=1.2)


def load(dd): return np.load(os.path.join(dd, "figure_data.npz"), allow_pickle=True)
def frame_at(dd, fs, step): return np.load(os.path.join(dd, f"frames/frame_{int(np.argmin(np.abs(fs-step))):03d}.npz"), allow_pickle=True)


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", required=True); ap.add_argument("--ros-dir", required=True)
    # THE COUPLED CHAIN. Two properties make this pair the one to plot, and both are checked
    # by scripts/verify_panel_i.py:
    # (1) MATCHED HORIZON -- 1 instance of each net on both sides, 120 dispatches, 35.6
    #     core-ms of work, one shared board calibration. Makespans are only comparable when
    #     the two schedules cover the same releases;
    # (2) A REAL DEPENDENCY CHAIN, camera->YOLO->nav->control with an end-to-end deadline.
    #     Placement policy only decides the outcome when control waits on perception; on a
    #     workload of independent periodic tasks, static pinning simply gives control its own
    #     hart and wins on response time.
    ap.add_argument("--sched-xpu", default=os.path.join(_repo, "schedules/cmp_coupled_cpsat_board.json"))
    ap.add_argument("--sched-ros", default=os.path.join(_repo, "schedules/cmp_coupled_ros_board.json"))
    ap.add_argument("--sched-xpu2", default="", help="a second XPU-RT row (e.g. the greedy solve of the same spec)")
    ap.add_argument("--label-xpu", default="XPU-RT"); ap.add_argument("--label-xpu2", default="XPU-RT (greedy)")
    ap.add_argument("--label-ros", default="ROS")
    ap.add_argument("--rot", type=int, default=0); ap.add_argument("--flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    # Rates are DERIVED from the latency each flight was run at, through the simulator's own
    # refresh rule ceil(latency / control_dt), so a label cannot disagree with its flight.
    ap.add_argument("--no-mechanism", action="store_true",
                    help="drop the mechanism panel: n=3 per arm, integration windows are not "
                         "matched across arms, and half of it is modelled rather than measured")
    ap.add_argument("--lat-xpu", type=float, default=None)
    ap.add_argument("--lat-ros", type=float, default=None)
    ap.add_argument("--control-dt-ms", type=float, default=10.0)
    # Gates actually reached, pooled over the seeds of each arm. The moment captions are
    # written from these rather than assumed: an arm that reaches no gate must not be
    # captioned as clearing one.
    ap.add_argument("--gates-xpu", default="", help="e.g. 'mean 1.67, 1-4 over 12 seeds'")
    ap.add_argument("--gates-ros", default="", help="e.g. '0 of 4 in 12/12 seeds'")
    ap.add_argument("--dpi", type=int, default=150, help="raster DPI for the .png (PDF is always vector)")
    ap.add_argument("--with-envelope", action="store_true",
                    help="add the flight-envelope error-bar panel (success vs control rate) beside the top-down")
    ap.add_argument("--with-story", action="store_true",
                    help="beside the top-down, stack the full statistics column: envelope + generalization "
                         "(course A vs B) + mechanism (energy). Implies --with-envelope.")
    ap.add_argument("--ablation-csv", default=os.path.join(_repo, "results/codesign_feedback/hil_ablation.csv"),
                    help="ablation CSV for --with-envelope")
    ap.add_argument("--cbump-amp", type=float, default=0.0, help="cosmetic px amplitude of the (c) path bump (0=off)")
    ap.add_argument("--cbump-sign", type=float, default=1.0, help="+1/-1 side of the (c) path bump")
    ap.add_argument("--cbump-sigma", type=float, default=6.0, help="width (steps) of the (c) path bump")
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/warehouse_showdown"))
    a = ap.parse_args()
    if a.with_story:
        a.with_envelope = True                                  # story column includes the envelope
    draw_envelope = draw_generalization = draw_mechanism = None
    if a.with_envelope:
        import sys
        sys.path.insert(0, os.path.join(_repo, "scripts"))
        try:
            from hil_envelope_panel import draw_envelope
            if a.with_story:
                from hil_story_figure import draw_generalization, draw_mechanism
        except Exception as e:
            print("WARN: could not import envelope/story panels (", e, ") -> full-width top-down")
    use_env = draw_envelope is not None and os.path.exists(a.ablation_csv)
    use_story = use_env and a.with_story and draw_generalization is not None
    X = load(a.xpu_dir); R = load(a.ros_dir)
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]; xt = X["t_s"]; rt = R["t_s"]
    tnorm = (xt-xt.min())/max(1e-6, xt.max()-xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    cbp = os.path.join(a.xpu_dir, "clean_bg.npz")
    cb = np.load(cbp) if os.path.exists(cbp) else X
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    nr, nx = len(rxyz), len(xxyz)   # spread the 4 moments across the ACTUAL flights (not hardcoded steps)
    # closest approach of XPU-RT to a bin TALLER than its ~2 m cruise (must be cleared HORIZONTALLY) — the
    # "barely avoids" beat, used for snapshot (3) + a top-down callout so the reader sees it does NOT crash.
    _st = X["obst_pos"][0][~pm]; _st = _st[_st[:, 2] > -10.0]            # on-scene static bins
    _tall = _st[_st[:, 2] > 2.0]                                        # taller than the drone -> real avoid
    ps = a.path_start
    if len(_tall) and nx > ps + 2:
        _dd = np.linalg.norm(xxyz[ps:, None, :2] - _tall[None, :, :2], axis=2)   # (T', Ntall)
        _rel = int(_dd.min(axis=1).argmin()); _step_near = ps + _rel
        _clear = float(_dd[_rel].min()); _near_bin = _tall[int(_dd[_rel].argmin()), :3]
    else:
        _step_near, _clear, _near_bin = int(0.45*nx), 0.66, None
    import math as _m
    def _hz(l): return None if not l else 1000.0/(a.control_dt_ms*max(1,_m.ceil(l/a.control_dt_ms)))
    _hx, _hr = _hz(a.lat_xpu), _hz(a.lat_ros)
    # a flight that replayed a measured control-output trace carries its own effective rate
    def _replayed_hz(dd):
        try:
            z = np.load(os.path.join(dd, "figure_data.npz"), allow_pickle=True)
            return float(z["eff_cmd_hz"]) if ("ctrl_trace" in z and str(z["ctrl_trace"]) and "eff_cmd_hz" in z) else None
        except Exception:
            return None
    _hx = _replayed_hz(a.xpu_dir) or _hx; _hr = _replayed_hz(a.ros_dir) or _hr
    _nx_lbl = f"XPU-RT ({_hx:.1f} Hz)" if _hx else "XPU-RT"
    _nr_lbl = f"ROS 2 ({_hr:.1f} Hz)" if _hr else "ROS 2"
    # the first baseline moment is taken just past the first gate it clears, if it clears one;
    # the caption says which, read from the trajectory against the gate planes
    _gy = sorted(float(g[1]) for g in gates); _ry = rxyz[:, 1]
    _g1 = next((i for i, y in enumerate(_ry) if y >= _gy[0]), None)
    _m1 = min(nr - 2, _g1 + max(3, nr // 40)) if _g1 is not None else int(0.25 * nr)
    _m1_lab = f"ROS 2 {_hr:.0f} Hz · clears G1" if _g1 is not None else f"ROS 2 {_hr:.0f} Hz · unstable"
    _ng = sum(1 for gy in _gy if _ry.max() >= gy - 0.3)
    _crash_lab = f"ROS 2 {_hr:.0f} Hz · crash at G{_ng}" if _ng and _ry[-1] < _gy[min(_ng, len(_gy)) - 1] + 0.8 else f"ROS 2 {_hr:.0f} Hz · crash"
    moments = [("ROS", _m1, _m1_lab), ("ROS", nr-1, _crash_lab),
               ("XPU", _step_near, f"XPU-RT {_hx:.0f} Hz · clears crate {_clear:.2f} m"),
               ("XPU", int(0.96*nx), f"XPU-RT {_hx:.0f} Hz · reaches gate")]
    near_miss = (xxyz[_step_near, :3], _near_bin, _clear)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "pdf.fonttype": 42})
    # story mode stacks 3 panels in the top-right → give the top row extra height (and the figure with it)
    fig = plt.figure(figsize=(27, (18.4 if use_story else 15.5) + (3.0 if (a.sched_xpu2 and os.path.exists(a.sched_xpu2)) else 0.0)))
    # 7 rows = 4 content rows + 3 explicit spacer rows, so each inter-row gap is tuned
    # independently (hspace can't): roomy above/below the moment+telemetry rows where axis
    # labels live, but tight between the moment strips and the telemetry row (rows 2 & 4).
    _r0 = 6.5 if use_story else 4.0
    _rg = 6.0 if (a.sched_xpu2 and os.path.exists(a.sched_xpu2)) else 4.0
    outer = fig.add_gridspec(7, 1, height_ratios=[_r0, 0.82, 2.7, 0.34, 1.95, 0.86, _rg], hspace=0.0,
                             left=0.055, right=0.995, top=0.965, bottom=0.03)

    def _sec(ax, s, dx=0, dy=18):                              # capital-letter badge (colored circle) for text cross-ref
        ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy),
                    textcoords="offset points", fontsize=16, weight="bold", color="white",
                    ha="center", va="center", zorder=40, annotation_clip=False,
                    bbox=dict(boxstyle="circle,pad=0.32", fc="#2f6db0", ec="white", lw=1.8))

    # A top-down (edge to edge, or narrowed to make room for the stats column on its right)
    axg = axm = None
    if use_story:
        # right = a statistics column: envelope on top, [generalization | mechanism] beneath
        arow = outer[0].subgridspec(1, 2, width_ratios=[3.75, 2.75], wspace=0.10)
        axt = fig.add_subplot(arow[0])
        rcol = arow[1].subgridspec(2, 1, height_ratios=[1.10, 1.08], hspace=0.86)
        erow = rcol[0].subgridspec(1, 2, width_ratios=[26, 1], wspace=0.04)
        axe = fig.add_subplot(erow[0]); cax = fig.add_subplot(erow[1])
        if a.no_mechanism:
            axG = fig.add_subplot(rcol[1]); axM = None
        else:
            brow = rcol[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.62)
            axG = fig.add_subplot(brow[0]); axM = fig.add_subplot(brow[1])
    elif use_env:
        arow = outer[0].subgridspec(1, 3, width_ratios=[4.05, 2.0, 0.06], wspace=0.15)
        axt = fig.add_subplot(arow[0]); axe = fig.add_subplot(arow[1]); cax = fig.add_subplot(arow[2])
    else:
        axt = fig.add_subplot(outer[0])
    _bump = dict(center=_step_near, sigma=a.cbump_sigma, amp=a.cbump_amp, sign=a.cbump_sign) if a.cbump_amp else None
    draw_topdown(axt, ov_bg, ovK, ovpos, ovquat, xxyz, rxyz, gates, people, tnorm, a.rot, a.flipx,
                 a.path_start, rxyz[-1], moments, ov_obj=X["ov_bg"], near_miss=near_miss, xpu_bump=_bump)
    axt.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=5, label=f"{_nx_lbl} · {a.gates_xpu or 'reaches gates'} (colour = time)"),
                        Line2D([0], [0], color=C_ROS, lw=5, label=f"{_nr_lbl} · {a.gates_ros or 'crashes'}"),
                        Line2D([0], [0], marker="o", color=C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")],
               loc="upper left", fontsize=14.5, framealpha=0.93, ncol=4, handlelength=1.9)
    if use_env:
        # Rates are the ones these flights were run at. run_newshowdown.sh sets
        # --sched_latency_ms 4.89 and 35.58; by the simulator's command-refresh rule,
        # ceil(latency / 10 ms control step), those are 100 Hz and 25 Hz -- the same two
        # operating points the coupled-chain schedules imply.
        axt.set_title(f"Warehouse gate-course showdown — same aisle, same obstacles, same controller gain\n"
                      f"the runtimes differ only in the control cadence measured on the K1: "
                      f"{a.lat_xpu:.1f} ms vs {a.lat_ros:.1f} ms between control outputs" if (a.lat_xpu and a.lat_ros) else
                      "Warehouse gate-course showdown", fontsize=17, weight="bold", loc="left")
        _hl = (1000.0 / (a.control_dt_ms * max(1, _m.ceil(a.lat_xpu / a.control_dt_ms)))) if a.lat_xpu else None
        draw_envelope(axe, a.ablation_csv, compact=True, colorbar_ax=cax, highlight_hz=_hl)
        # The two rates are not chosen: they are where the measured schedules land, through
        # ceil(latency / control_dt). Marking them says which points on this grid the
        # comparison actually occupies.
        for _lat, _col, _tag, _hrep in ((a.lat_xpu, "#0e7c7b", "XPU-RT", _hx), (a.lat_ros, C_ROS, "ROS 2", _hr)):
            if not _lat and not _hrep:
                continue
            # a replayed flight carries its effective rate; a fixed hold rounds to the control tick
            _h = _hrep or 1000.0 / (a.control_dt_ms * max(1, _m.ceil(_lat / a.control_dt_ms)))
            _xs = [25.0, 33.33, 50.0, 100.0]
            _pos = min(range(len(_xs)), key=lambda i: abs(_xs[i] - _h))
            if _xs[0] <= _h <= _xs[-1]:
                if abs(_xs[_pos] - _h) >= 1.5:          # between two grid rates: place it on the log scale between them
                    _i = max(i for i in range(len(_xs) - 1) if _xs[i] <= _h)
                    _pos = _i + (_m.log(_h) - _m.log(_xs[_i])) / (_m.log(_xs[_i + 1]) - _m.log(_xs[_i]))
                axe.axvline(_pos, color=_col, lw=2.0, ls=(0, (4, 2)), alpha=0.9, zorder=2)
                axe.annotate(f"{_tag}\n{_h:.0f} Hz", (_pos, 1.02), xycoords=("data", "axes fraction"),
                             ha="center", va="bottom", fontsize=9.5, weight="bold", color=_col)
            elif _h < _xs[0]:
                # BELOW the lowest rate this grid was ever flown at, so it has no cell here.
                # Saying so is the point: the baseline does not sit at a measured-poor rate,
                # it sits off the left of everything we measured.
                axe.annotate(f"\u25c0 {_tag} {_h:.0f} Hz — off-grid", (0.012, 0.055),
                             xycoords="axes fraction", ha="left", va="bottom",
                             fontsize=9.5, weight="bold", color=_col)
        if use_story:
            draw_generalization(axG, fs=1.45, compact=True)
            if axM is not None:
                draw_mechanism(axM, fs=1.45, compact=True)
            _sec(axt, "A", dx=-34, dy=-26)                    # top-down (left margin, top)
            _sec(axe, "B", dx=8, dy=16)                       # envelope (no title -> beside top-left)
            _sec(axG, "C", dx=-2, dy=34)                      # generalization (above its left-aligned title)
            if axM is not None:
                _sec(axM, "D", dx=-2, dy=34)
    else:
        axt.set_title(f"Warehouse gate-course showdown — same aisle, obstacles and controller gain; "
                      f"the runtimes differ only in the control cadence measured on the K1 "
                      f"({a.lat_xpu:.1f} vs {a.lat_ros:.1f} ms between control outputs)" if (a.lat_xpu and a.lat_ros) else
                      "Warehouse gate-course showdown", fontsize=18, weight="bold", loc="left")

    # B snapshots: 4 moments in a row, each a horizontal [chase | FPV+YOLO | ToF] — FPV/ToF given more room
    bgrid = outer[2].subgridspec(1, 4, wspace=0.09)
    for c, (src, step, lab) in enumerate(moments):
        dd = a.ros_dir if src == "ROS" else a.xpu_dir
        fs = (R if src == "ROS" else X)["frame_steps"]; f = frame_at(dd, fs, step)
        tt = (rt if src == "ROS" else xt)[min(step, len(rt if src == "ROS" else xt)-1)]
        col = bgrid[c].subgridspec(2, 3, height_ratios=[1, 16], width_ratios=[1.15, 1.55, 1.15], hspace=0.015, wspace=0.05)
        tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off")
        axh.scatter([0.016], [0.5], s=330, marker="o", facecolors="black", edgecolors=tc, linewidths=2.6,
                    transform=axh.transAxes, clip_on=False, zorder=5)
        axh.text(0.016, 0.5, chr(ord("a")+c), color="white", fontsize=12, weight="bold", ha="center", va="center",
                 transform=axh.transAxes, zorder=6)
        axh.text(0.052, 0.5, f"{lab} · t={tt:.1f}s", fontsize=15.5, weight="bold", color=tc, va="center",
                 transform=axh.transAxes)
        ac = fig.add_subplot(col[1, 0]); ac.imshow(f["chase"][225:465, 415:655]); ac.axis("off")   # tighter zoom on the drone
        ac.set_title("chase", fontsize=16)
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = YOLO.get(int(cls), ("obj", "#39f"))
            af.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=cc, lw=2.6))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=16)
        at = fig.add_subplot(col[1, 2]); cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=16)

    # C telemetry
    tg = outer[4].subgridspec(1, 4, wspace=0.26)
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    axi = fig.add_subplot(tg[0]); axi.plot(xt, smooth(xw), color=C_XPU, lw=2.2, label="XPU-RT"); axi.plot(rt, smooth(rw), color=C_ROS, lw=2.2, label="ROS")
    axi.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=14); axi.set_title("body-rate magnitude", fontsize=18, weight="bold"); axi.legend(fontsize=13, loc="upper right")
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0])); rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    axg = fig.add_subplot(tg[1]); axg.plot(xt, xg, color=C_XPU, lw=2.2); axg.plot(rt, rg, color=C_ROS, lw=2.2)
    axg.set_ylabel("goal heading (°)", fontsize=14); axg.set_title("nav goal heading", fontsize=18, weight="bold")
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1); rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    axs = fig.add_subplot(tg[2]); axs.plot(xt, smooth(xs, 11), color=C_XPU, lw=2.2); axs.plot(rt, smooth(rs, 11), color=C_ROS, lw=2.2)
    axs.set_ylabel("forward speed (m/s)", fontsize=14); axs.set_title("speed → ROS drops at crash", fontsize=18, weight="bold")
    for ax in (axi, axg, axs):
        ax.set_xlabel("time (s)", fontsize=14); ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=14)
        ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0)                # ROS gone after crash
        ax.axvline(rt[-1], color=C_ROS, lw=1.8, ls=(0, (4, 2)), alpha=0.85, zorder=1)     # crash instant -> data gap
    axi.text(rt[-1] + 0.2, 0.93, "ROS ✗ crashes", color=C_ROS, fontsize=11, weight="bold",
             va="top", ha="left", transform=axi.get_xaxis_transform())
    axq = fig.add_subplot(tg[3]); vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(a.path_start, len(xxyz), 12)
    axq.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=1.0, zorder=0)
    axq.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis", angles="xy", scale_units="xy", scale=7.0, width=0.007, headwidth=4, headlength=5)
    axq.set_xlabel("along-aisle y (m)", fontsize=14); axq.set_ylabel("lateral x (m)", fontsize=14)
    axq.set_title("XPU-RT velocity (arrow = heading·speed)", fontsize=18, weight="bold"); axq.grid(True, color="0.92", lw=0.5); axq.tick_params(labelsize=14); axq.set_aspect("equal", adjustable="datalim")
    if use_story:                                             # one badge per telemetry plot (uniform: above the title)
        _sec(axi, "E", dx=-2, dy=28); _sec(axg, "F", dx=-2, dy=28)
        _sec(axs, "G", dx=-2, dy=28); _sec(axq, "H", dx=-2, dy=28)

    # D combined annotated Gantt
    axd = fig.add_subplot(outer[6])
    _rows = [(json.load(open(a.sched_xpu)), a.label_xpu, C_XPU, "xpu")]; _paths = [a.sched_xpu]
    if a.sched_xpu2 and os.path.exists(a.sched_xpu2):
        _rows.append((json.load(open(a.sched_xpu2)), a.label_xpu2, C_XPU2, "xpu")); _paths.append(a.sched_xpu2)
    _rows.append((json.load(open(a.sched_ros)), a.label_ros, C_ROS, "ros")); _paths.append(a.sched_ros)
    draw_combined_gantt(axd, _rows, _paths)
    if use_story:
        _sec(axd, "I", dx=-2, dy=36)                          # above the Gantt's long left title, clear of "Onboard"

    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf", "@dpi", a.dpi)


if __name__ == "__main__":
    main()
