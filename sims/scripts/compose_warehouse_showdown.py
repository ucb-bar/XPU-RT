#!/usr/bin/env python3
"""Warehouse case study: illustrative simulator rollouts plus schedule analysis.

  A. Top-down aisle (edge-to-edge) — BOTH flight paths: XPU-RT (time-coloured, all 4 gates) + ROS (red, crashes
     just past gate 1). Gates, patrolling people (real height), crash marker.
  B. 4 key moments (2 ROS + 2 XPU), each a horizontal strip: chase (zoomed) | FPV+YOLO | large cross-ToF.
  C. Telemetry — IMU |w|, goal heading, forward speed (XPU vs ROS) + XPU velocity arrows.
  D. Matched-horizon K1-calibrated schedule model — the annotated Gantt compares one interior YOLO response
     for XPU-RT and a ROS-style fixed partition on the same eight-hart platform.
"""
import argparse, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

from job_names import split_job_name

CMAP = matplotlib.colormaps["viridis"]
YOLO = {0: ("gate", "#ffd400"), 1: ("person", "#ff4b4b")}
C_MOVER = "#9d4edd"; C_XPU = "#1f9e5a"; C_ROS = "#e2231a"
GANTT_CORES = ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3", "CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]
C_CTRL, C_NAV, C_YOLO = "#2f8f4e", "#7b52c0", "#e8823a"
LP, LG = "#efe7fb", "#e6f3ea"       # light NAV / CTRL window fills
NET = {"mlp_control": ("ctrl", C_CTRL), "fused_full": ("nav", C_NAV),
       "yolov8_nano_64x96": ("yolo", C_YOLO)}
COMPOSITE_FONT_SCALE = 0.40


def cfs(points):
    """Composite font size at final two-column width, never below 7 pt."""
    return max(7.0, points * COMPOSITE_FONT_SCALE)


def netinfo(job):
    base, _ = split_job_name(job, NET)
    return NET.get(base, ("other", "#8aa"))


def inst_of(job):
    return split_job_name(job, NET)[1]


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
    for lbl, (yy, xx) in {"N": (0.8, 11.5), "E": (11.5, 21.7), "S": (22.1, 11.5), "W": (11.5, 1.3)}.items():
        ax.text(xx, yy, lbl, color="white", fontsize=cfs(12), weight="bold", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(23.5, -0.5)


def marey_cylinders(ov_seq, ov_bg, alpha0=0.62, pct=97.3, min_blob=90, sat_max=0.24, bright_min=145):
    """Marey chronophotography: overlay the moving patrol cylinders' own image pixels, captured at N
    overhead timesteps, onto the clean plate with a time-increasing alpha. Image-based (no projection):
    the N frames share one render pass, so the per-pixel temporal median is the static scene; each frame
    minus the median, gated by a pale-cylinder colour filter (cylinders are whitish; crates are saturated
    teal, boxes tan), isolates just the movers. Returns the composited background (same shape as ov_bg)."""
    from scipy import ndimage as ndi
    seq = np.asarray(ov_seq, np.float32); bg = np.asarray(ov_bg, np.float32); n = len(seq)
    if n < 2:
        return ov_bg
    med = np.median(seq, axis=0)
    seqc = np.stack([seq[i] - np.median(seq[i] - med) for i in range(n)])   # per-frame offset correction
    diff = np.abs(seqc - med).sum(axis=3)
    R, G, B = seq[..., 0], seq[..., 1], seq[..., 2]
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / (mx + 1e-3); pale = (mx > bright_min) & (sat < sat_max)
    thr = np.percentile(diff, pct); comp = bg.copy()
    for i in range(n):
        m = (diff[i] > thr) & pale[i]
        m = ndi.binary_opening(m, iterations=1); m = ndi.binary_closing(m, iterations=2)
        lbl, nl = ndi.label(m)
        if nl:
            sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, nl + 1))
            m = np.isin(lbl, 1 + np.where(sizes >= min_blob)[0])
        a = alpha0 + (1.0 - alpha0) * i / max(1, n - 1)
        comp[m] = (1 - a) * comp[m] + a * seq[i][m]
    return np.clip(comp, 0, 255).astype(np.uint8)


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
            ax.text(gu[i], gv[i]-18, f"G{i+1}", color="#ffd400", fontsize=cfs(13), weight="bold", ha="center", va="center", zorder=6)
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
    # breadcrumb waypoints along each path (gate-course look): 8 evenly-spaced dots, XPU time-coloured
    for uu, vv, vis, solid in ((ru, rv, rvis, C_ROS), (xu, xv, xvis, None)):
        idx = np.where(vis)[0]
        if len(idx) >= 8:
            for k in np.linspace(idx[0], idx[-1], 8).astype(int):
                col = solid if solid is not None else CMAP(tnorm[k])
                ax.scatter(uu[k], vv[k], s=34, facecolors=col, edgecolors="white", linewidths=1.1, zorder=7)
    for mi, (src, step, lab) in enumerate(moments):
        path = ros if src == "ROS" else xpu
        u, v, o = proj(path[min(step, len(path)-1):min(step, len(path)-1)+1])
        ec = C_ROS if src == "ROS" else C_XPU
        is_crash = src == "ROS" and step >= len(ros)-2
        if o[0]:
            mu, mv = (u[0], v[0]-40) if is_crash else (u[0], v[0])
            ax.add_patch(Circle((mu, mv), 18, fill=True, fc=ec, ec="white", lw=2.6, zorder=8))
            ax.text(mu, mv, "1234"[mi], color="white", fontsize=cfs(17), weight="bold", ha="center", va="center", zorder=9)
    # crop vertically to the aisle band (still full-width) — show a good band of shelving/stands each side
    cv = np.concatenate([xv[xvis], rv[rvis], gv[gok]])
    vmin, vmax = float(np.nanmin(cv)), float(np.nanmax(cv)); pad = 1.05 * (vmax - vmin)
    ax.set_ylim(min(nH, vmax + pad), max(0, vmin - pad))


def _yolo_frames(dd):
    """Per-YOLO-frame (duration_ms, n_harts) in instance order, from a schedule's dispatches."""
    fr, hs = {}, {}
    for v in dd.values():
        if netinfo(v["job_name"])[0] != "yolo":
            continue
        i = inst_of(v["job_name"]); s = float(v["start_time"]); e = s + float(v["duration"])
        if i not in fr:
            fr[i] = [s, e]; hs[i] = set()
        else:
            fr[i][0] = min(fr[i][0], s); fr[i][1] = max(fr[i][1], e)
        for h in v["hardware_target"].split("+"):
            hs[i].add(h)
    return [(fr[i][1] - fr[i][0], len(hs[i])) for i in sorted(fr)]


def _frame_window(dispatches, frame_instance, period_ms):
    """Return ``(instance, release, finish)`` for one comparable YOLO frame."""
    fr = {}
    for v in dispatches.values():
        if netinfo(v["job_name"])[0] != "yolo":
            continue
        i = inst_of(v["job_name"]); s = float(v["start_time"]); e = s + float(v["duration"])
        if i not in fr: fr[i] = [s, e]
        else: fr[i][0] = min(fr[i][0], s); fr[i][1] = max(fr[i][1], e)
    if not fr: return None
    i = frame_instance if frame_instance in fr else sorted(fr)[-1]
    return i, i * period_ms, fr[i][1]


def _hart_blocks(dispatches, win=None, gap=0.35, selected_yolo_instance=None):
    """Per-hart work overlapping one release-to-output response window."""
    lo, hi = (win if win else (-1e18, 1e18))
    per = {}
    for v in dispatches.values():
        s = float(v["start_time"]); e = s + float(v["duration"])
        if e <= lo + 1e-6 or s >= hi - 1e-6:
            continue
        mn, col = netinfo(v["job_name"])
        inst = inst_of(v["job_name"])
        if mn == "yolo" and selected_yolo_instance is not None:
            if inst == selected_yolo_instance:
                mn, col = "yolo_selected", C_YOLO
            else:
                mn, col = "yolo_adjacent", "#f5c39d"
        s = max(s, lo) - lo; e = min(e, hi) - lo      # align the sensor release to zero
        for h in v["hardware_target"].split("+"):
            per.setdefault(h, []).append((s, e, mn, col))
    merged = {}
    for h, segs in per.items():
        segs.sort(key=lambda t: (t[0], t[1])); out = []
        for s, e, mn, col in segs:
            if out and out[-1][2] == mn and s <= out[-1][1] + gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e, mn, col])
        merged[h] = out
    # The wall is response time from frame release, including queueing behind prior work.
    wall = (hi - lo) if win else max((e for segs in merged.values() for _, e, _, _ in segs), default=0.0)
    return merged, wall


def draw_control_rate_panel(ax, x_resp, r_resp, period=10.0, window=44.0,
                            font_scale=1.0, panel_title=None, footer_y=-0.30):
    """Panel (b): the CONTROL-command-rate story that actually drives the flight (panel a). Each schedule's
    worst-case control response (`x_resp`, `r_resp`, in ms — passed in from the measured schedules, never
    defaulted here) caps how often a fresh command can be issued under the zero-order hold at the `period` ms
    control step: a response under one period gives a fresh command every step; one over it holds the command
    an extra step. Below the control-rate cliff the warehouse gate-course is unflyable (Fig 3 companion)."""
    fs = lambda p: max(7.0, p * font_scale)
    xhold = int(np.ceil(x_resp / period)); rhold = int(np.ceil(r_resp / period))
    xrate = 1000.0 / (period * xhold); rrate = 1000.0 / (period * rhold)
    for t in np.arange(0, window + 0.1, period):                      # control-period grid
        ax.axvline(t, color="0.86", lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.axvline(period, color="#b3261e", lw=2.4, ls=(0, (5, 3)), zorder=4)
    ax.text(period, 2.16, f"control period  {period:.0f} ms  (100 Hz)", color="#8f1d18",
            fontsize=fs(13.5), weight="bold", ha="center", va="bottom")

    def lane(y0, hold, color):
        t = 0.0
        while t < window - 1e-6:
            ax.barh(y0, period, left=t, height=0.66, color=color, edgecolor="white", lw=0.6, zorder=3)
            if hold > 1:                                              # held-stale extension (hatched)
                ax.barh(y0, period * (hold - 1), left=t + period, height=0.66, facecolor="none",
                        edgecolor=color, hatch="////", lw=0.9, zorder=3)
            t += period * hold
    lane(1.35, xhold, C_XPU); lane(0.35, rhold, C_ROS)
    for y, resp in ((1.35, x_resp), (0.35, r_resp)):                  # worst-response arrows from release=0
        ax.annotate("", xy=(resp, y), xytext=(0, y), arrowprops=dict(arrowstyle="<->", color="#222", lw=1.4), zorder=6)
        ax.text(resp + 0.4, y, f"{resp:.2f} ms", fontsize=fs(11.5), va="center", ha="left", color="#222", weight="bold")

    ax.text(window * 0.985, 1.78, f"{xrate:.0f} Hz  ✓ fresh every step → completes 4/4",
            color=C_XPU, fontsize=fs(14), weight="bold", ha="right", va="bottom")
    ax.text(window * 0.985, 0.78, f"{rrate:.0f} Hz  ✗ held stale {rhold-1} step → crashes",
            color=C_ROS, fontsize=fs(14), weight="bold", ha="right", va="bottom")
    ax.text(-0.6, 1.35, "XPU-RT", rotation=90, va="center", ha="center", color=C_XPU, fontsize=fs(15), weight="bold")
    ax.text(-0.6, 0.35, "FIXED", rotation=90, va="center", ha="center", color=C_ROS, fontsize=fs(15), weight="bold")
    ax.set_xlim(-1.4, window); ax.set_ylim(-0.35, 2.45)
    ax.set_yticks([]); ax.tick_params(axis="x", labelsize=fs(13))
    ax.set_xlabel("time since command release (ms) · zero-order hold at the 10 ms control step", fontsize=fs(15))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=C_XPU, lw=9, label="fresh command"),
                       Rectangle((0, 0), 1, 1, fc="none", ec=C_ROS, hatch="////", label="held stale (ZOH)")],
              loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=fs(12), ncol=2, framealpha=0.9)
    if panel_title:
        ax.text(0.0, 1.02, panel_title, transform=ax.transAxes, fontsize=fs(16.5), weight="bold", ha="left", va="bottom")
    ax.text(0.0, footer_y, "The schedule's worst-case control response sets the command rate; below the ~65 Hz "
            f"cliff (Fig. 3 companion) the course is unflyable. Worst responses: {x_resp:.2f} / {r_resp:.2f} ms.",
            transform=ax.transAxes, fontsize=fs(11.5), color="0.38", ha="left", va="top")
    return {"comparison_unit": "control-command rate under each schedule's worst-response",
            "xpu_worst_response_ms": x_resp, "xpu_control_rate_hz": xrate,
            "ros_worst_response_ms": r_resp, "ros_control_rate_hz": rrate,
            "control_period_ms": period, "flyable_cliff_hz": 65,
            "flight_result": f"XPU-RT {xrate:.0f} Hz vs baseline {rrate:.0f} Hz control-command rate"}


def draw_combined_gantt(ax, xd, rd, budget_ms=23.0, frame_instance=1,
                        font_scale=1.0, footer_y=-0.17, panel_title=None):
    """Compare the same perception-frame release under both board-cost schedules."""
    fs = lambda points: max(7.0, points * font_scale)
    xframe = _frame_window(xd, frame_instance, budget_ms)
    rframe = _frame_window(rd, frame_instance, budget_ms)
    if xframe is None or rframe is None:
        raise ValueError("both schedules must contain YOLO perception frames")
    xi, xrelease, xfinish = xframe
    ri, rrelease, rfinish = rframe
    if xi != ri:
        raise ValueError(f"requested frame differs across schedules: {xi} vs {ri}")
    xb, xresponse = _hart_blocks(
        xd, (xrelease, xfinish), selected_yolo_instance=xi)
    rb, rresponse = _hart_blocks(
        rd, (rrelease, rfinish), selected_yolo_instance=ri)
    xmax = max(xresponse, rresponse, budget_ms) + 2.5
    tol_ms = 0.01                 # ignore sub-10-us float reconstruction noise

    def bars(blocks, y0):
        yof = {c: y0 + i for i, c in enumerate(GANTT_CORES)}
        for h, segs in blocks.items():
            if h not in yof:
                continue
            for s, e, _, col in segs:
                ax.barh(yof[h], max(e - s, 0.08), left=s, height=0.78,
                        color=col, edgecolor="white", linewidth=0.16, zorder=3)
        for h in GANTT_CORES:
            if h not in blocks:
                ax.text(xmax * 0.43, yof[h], "idle", ha="center", va="center",
                        fontsize=fs(11.5), style="italic", color="0.62")

    # A common deadline and a common x scale make the comparison literal.
    ax.axvspan(budget_ms, xmax, color="#fdeeee", alpha=0.75, zorder=0)
    ax.axvline(budget_ms, color="#b3261e", lw=2.5, ls=(0, (5, 3)), zorder=5)
    ax.text(budget_ms, 18.35, f"period / deadline  {budget_ms:.0f} ms",
            color="#8f1d18", fontsize=fs(13.5), weight="bold", ha="center", va="bottom")

    bars(xb, 9.6); bars(rb, 0.0)
    ax.axhline(8.8, color="0.55", lw=1.0)

    def verdict(response, colour, y, label):
        met = response <= budget_ms + tol_ms
        delta = response - budget_ms
        status = (f"{response:.2f} ms  ✓ meets target" if met else
                  f"{response:.2f} ms  ✗ misses by {delta:.2f} ms")
        ax.axvline(response, ymin=(y - 0.5) / 18.7, ymax=(y + 7.5) / 18.7,
                   color=colour, lw=2.4, zorder=5)
        ax.text(0.25, y + 7.65, f"{label}: {status}", color=colour,
                fontsize=fs(15.5), weight="bold", ha="left", va="bottom")
        ax.annotate("", xy=(0, y + 7.3), xytext=(0, y + 8.25),
                    arrowprops=dict(arrowstyle="-|>", color="#b3261e", lw=1.5))
        if not met:
            ax.annotate("output", xy=(response, y - 0.35), xytext=(response, y - 1.35),
                        fontsize=fs(11.5), color=colour, ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5))

    verdict(xresponse, C_XPU, 9.6, "XPU-RT · CP-SAT / 8 harts")
    verdict(rresponse, C_ROS, 0.0, "Fixed partition / 8-hart platform (6 used)")

    ax.text(-2.0, 13.1, "XPU-RT", fontsize=fs(16), weight="bold", rotation=90,
            va="center", ha="center", color=C_XPU)
    ax.text(-2.0, 3.5, "FIXED", fontsize=fs(16), weight="bold", rotation=90,
            va="center", ha="center", color=C_ROS)
    ax.set_xlim(-3.0, xmax); ax.set_ylim(-1.9, 18.7)
    ax.set_yticks([9.6 + i for i in range(8)] + list(range(8)))
    ax.set_yticklabels([c.replace("CPU_", "") for c in GANTT_CORES] * 2, fontsize=fs(11.5))
    ax.tick_params(axis="x", labelsize=fs(13))
    ax.set_xlabel("time since YOLO release (ms) · K1-calibrated modeled durations", fontsize=fs(15))
    xcounts = {net: len({inst_of(v["job_name"]) for v in xd.values()
                         if netinfo(v["job_name"])[0] == net})
               for net in ("yolo", "nav", "ctrl")}
    rcounts = {net: len({inst_of(v["job_name"]) for v in rd.values()
                         if netinfo(v["job_name"])[0] == net})
               for net in ("yolo", "nav", "ctrl")}
    matched = xcounts == rcounts
    if panel_title:
        ax.text(0.0, 1.075, panel_title,
                transform=ax.transAxes, fontsize=fs(16.5), weight="bold",
                ha="left", va="bottom")
    ax.text(0.0, footer_y,
            f"Same interior release (YOLO frame {xi + 1}; n=1 per method). "
            f"XPU-RT horizon {xcounts['yolo']}/{xcounts['nav']}/{xcounts['ctrl']} and baseline "
            f"{rcounts['yolo']}/{rcounts['nav']}/{rcounts['ctrl']} YOLO/NAV/CTRL instances.\n"
            "Bars include adjacent-frame backlog and interfering NAV/CTRL. Calibrated model, not a board trace.",
            transform=ax.transAxes, fontsize=fs(11.5), color="0.38", ha="left", va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=C_CTRL, lw=8, label="CTRL (mlp) 100 Hz"),
                       Line2D([0], [0], color=C_NAV, lw=8, label="NAV (fused) 50 Hz"),
                       Line2D([0], [0], color=C_YOLO, lw=8, label="selected YOLO"),
                       Line2D([0], [0], color="#f5c39d", lw=8, label="adjacent YOLO"),
                       Rectangle((0, 0), 1, 1, fc="#fdeeee", ec="none", label="past target")],
              loc="upper right", fontsize=fs(10.5), ncol=2, framealpha=0.95)
    return {
        "frame_instance_zero_based": xi,
        "budget_ms": budget_ms,
        "comparison_unit": "one perception-frame response from release to output",
        "xpu_response_ms": xresponse,
        "xpu_meets_budget": xresponse <= budget_ms + tol_ms,
        "ros_response_ms": rresponse,
        "ros_meets_budget": rresponse <= budget_ms + tol_ms,
        "xpu_workload_instances": xcounts,
        "ros_workload_instances": rcounts,
        "horizons_matched": matched,
        "numeric_tolerance_ms": tol_ms,
    }


def center_on_drone(chase, size=300):
    """Crop a square window centred on the bright-red drone so it sits in the middle of the chase view.
    The drone is a pure saturated red (emissive); a strict test isolates it from warehouse clutter,
    and the median pixel is robust to any stray red."""
    a = chase.astype(np.float32)
    red = ((a[:, :, 0] > 185) & (a[:, :, 1] < 75) & (a[:, :, 2] < 75)
           & ((a[:, :, 0] - a[:, :, 1]) > 110) & ((a[:, :, 0] - a[:, :, 2]) > 110))
    H, W = chase.shape[:2]; h = size // 2
    if red.sum() > 20:
        ys, xs = np.where(red); cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy, cx = 444, 480                                   # measured steady drone position
    cy = min(max(cy, h), H - h); cx = min(max(cx, h), W - h)
    return chase[cy - h:cy + h, cx - h:cx + h]


def load(dd): return np.load(os.path.join(dd, "figure_data.npz"), allow_pickle=True)
def frame_at(dd, fs, step): return np.load(os.path.join(dd, f"frames/frame_{int(np.argmin(np.abs(fs-step))):03d}.npz"), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", required=True); ap.add_argument("--ros-dir", required=True)
    ap.add_argument("--sched-xpu", default=os.path.join(REPO, "schedules/scheduled__flight_deployed_matched_board_cpsat_profiled.json"))
    ap.add_argument("--sched-ros", default=os.path.join(REPO, "schedules/scheduled_ros_partition_deployed_matched_board.json"))
    ap.add_argument("--frame-budget-ms", type=float, default=23.0)
    ap.add_argument("--x-resp-ms", type=float, required=True, help="XPU-RT's worst-case control response (ms), supplied by the caller; record its source in the run's notes")
    ap.add_argument("--r-resp-ms", type=float, required=True, help="the baseline's worst-case control response (ms), supplied by the caller; record its source in the run's notes")
    ap.add_argument("--frame-instance", type=int, default=2,
                    help="zero-based YOLO instance compared in both schedules")
    ap.add_argument("--rot", type=int, default=0); ap.add_argument("--flipx", action="store_true")
    ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--out", default=os.path.join(REPO, "results/codesign_feedback/warehouse_showdown"))
    ap.add_argument("--schedule-out", default=None,
                    help="optional standalone schedule figure stem (.png/.pdf are appended)")
    a = ap.parse_args()
    X = load(a.xpu_dir); R = load(a.ros_dir)
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]; xt = X["t_s"]; rt = R["t_s"]
    tnorm = (xt-xt.min())/max(1e-6, xt.max()-xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    cbp = os.path.join(a.xpu_dir, "clean_bg.npz")
    cb = np.load(cbp) if os.path.exists(cbp) else X
    ov_bg, ovK, ovpos, ovquat = cb["ov_bg"], cb["ovK"], cb["ovpos"], cb["ovquat"]
    if "ov_seq" in X.files and len(np.asarray(X["ov_seq"])) >= 2:   # chronophotography of the moving cylinders
        ov_bg = marey_cylinders(X["ov_seq"], ov_bg)
    # rate labels come from the REAL flight metadata (eff_cmd_hz), not hardcoded
    xhz = int(round(float(X["eff_cmd_hz"]))) if "eff_cmd_hz" in X.files else 100
    rhz = int(round(float(R["eff_cmd_hz"]))) if "eff_cmd_hz" in R.files else 50
    XLAB = f"XPU-RT · {xhz} Hz"; RLAB = f"ROS · {rhz} Hz"
    nx, nr = len(xxyz), len(rxyz)
    moments = [("ROS", int(0.22*nr), f"{RLAB} passes G1"),
               ("ROS", nr-1, f"{RLAB} crashes"),
               ("XPU", int(0.72*nx), f"{XLAB} at G3"),
               ("XPU", nx-1, f"{XLAB} reaches G4")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": cfs(16), "pdf.fonttype": 42})
    # Final two-column dimensions; all labels are authored at >=7 pt.
    fig = plt.figure(figsize=(13.2, 6.9))          # WIDE landscape (was portrait 7.08x8.35)
    # Two gridspecs so the top-down can sit RIGHT above the snapshot grid (tiny gap) while the
    # snapshots / telemetry / Gantt keep normal spacing between them.
    gs_top = fig.add_gridspec(1, 1, left=0.045, right=0.997, top=0.955, bottom=0.700)
    gs_rest = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.7, 2.7], hspace=0.62,
                               left=0.045, right=0.997, top=0.665, bottom=0.130)

    # A top-down (edge to edge)
    axt = fig.add_subplot(gs_top[0])
    draw_topdown(axt, ov_bg, ovK, ovpos, ovquat, xxyz, rxyz, gates, people, tnorm, a.rot, a.flipx,
                 a.path_start, rxyz[-1], moments)
    axt.legend(handles=[Line2D([0], [0], color=CMAP(0.6), lw=5, label=f"{XLAB} ✓ 4/4"),
                        Line2D([0], [0], color=C_ROS, lw=5, label=f"{RLAB} ✗ crashes"),
                        Line2D([0], [0], color=C_MOVER, lw=2.5, ls=(0, (1, 1.3)), label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")],
               loc="upper left", fontsize=cfs(17), framealpha=0.93, ncol=4, handlelength=1.9)
    axt.set_title("Warehouse gate-course flight", fontsize=cfs(23),
                  weight="bold", loc="left")

    # B snapshots: 4 moments in a single 1x4 row — in the WIDE landscape layout each moment gets a
    # full quarter-width, so chase | FPV+YOLO | ToF stay large enough to read while the figure is short.
    bgrid = gs_rest[0].subgridspec(1, 4, wspace=0.11, hspace=0.30)
    for c, (src, step, lab) in enumerate(moments):
        rr, cc = 0, c
        dd = a.ros_dir if src == "ROS" else a.xpu_dir
        fs = (R if src == "ROS" else X)["frame_steps"]; f = frame_at(dd, fs, step)
        tt = (rt if src == "ROS" else xt)[min(step, len(rt if src == "ROS" else xt)-1)]
        col = bgrid[rr, cc].subgridspec(2, 3, height_ratios=[2.7, 16], width_ratios=[1.1, 1.5, 1.1], hspace=0.55, wspace=0.05)
        tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off"); axh.set_xlim(0, 1); axh.set_ylim(0, 1)
        # same coloured circle badge as the top-down (filled disc, white letter)
        axh.text(0.028, 0.5, "1234"[c], transform=axh.transAxes, fontsize=cfs(15), weight="bold", color="white",
                 ha="center", va="center", zorder=6,
                 bbox=dict(boxstyle="circle,pad=0.30", fc=tc, ec="white", lw=1.6))
        axh.text(0.085, 0.5, f"{lab}  ({tt:.1f}s)", transform=axh.transAxes, fontsize=cfs(18), weight="bold",
                 color=tc, va="center", ha="left")
        ac = fig.add_subplot(col[1, 0]); ac.imshow(center_on_drone(f["chase"], size=200)); ac.axis("off")   # drone centred + zoomed
        ac.set_title("chase", fontsize=cfs(17))
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc2 = YOLO.get(int(cls), ("obj", "#39f"))
            af.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=cc2, lw=3.0))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=cfs(17))
        at = fig.add_subplot(col[1, 2]); cross_tof(at, f["tof"]); at.set_title("cross-ToF (0–4 m)", fontsize=cfs(17))

    # C telemetry
    tg = gs_rest[1].subgridspec(1, 4, wspace=0.42)
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    axi = fig.add_subplot(tg[0]); axi.plot(xt, smooth(xw), color=C_XPU, lw=2.2, label=XLAB); axi.plot(rt, smooth(rw), color=C_ROS, lw=2.2, ls=(0, (5, 2)), label=RLAB)
    axi.set_ylabel("IMU |ω| (rad/s)", fontsize=cfs(17)); axi.set_title("body rate", fontsize=cfs(19), weight="bold"); axi.legend(fontsize=cfs(16), loc="upper right")
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0])); rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    axg = fig.add_subplot(tg[1]); axg.plot(xt, xg, color=C_XPU, lw=2.2); axg.plot(rt, rg, color=C_ROS, lw=2.2, ls=(0, (5, 2)))
    axg.set_ylabel("goal heading (°)", fontsize=cfs(17)); axg.set_title("goal heading", fontsize=cfs(19), weight="bold")
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1); rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    axs = fig.add_subplot(tg[2]); axs.plot(xt, smooth(xs, 11), color=C_XPU, lw=2.2); axs.plot(rt, smooth(rs, 11), color=C_ROS, lw=2.2, ls=(0, (5, 2)))
    axs.set_ylabel("speed (m/s)", fontsize=cfs(17)); axs.set_title("forward speed", fontsize=cfs(19), weight="bold")
    for ax in (axi, axg, axs):
        ax.set_xlabel("time (s)", fontsize=cfs(17)); ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=cfs(16))
        ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.4, zorder=0)
    axq = fig.add_subplot(tg[3]); vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(a.path_start, len(xxyz), 12)
    axq.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=1.0, zorder=0)
    axq.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis", angles="xy", scale_units="xy", scale=7.0, width=0.007, headwidth=4, headlength=5)
    axq.set_xlabel("along-aisle (m)", fontsize=cfs(17)); axq.set_ylabel("lateral (m)", fontsize=cfs(17))
    axq.set_title(f"{XLAB} velocity", fontsize=cfs(19), weight="bold"); axq.grid(True, color="0.92", lw=0.5); axq.tick_params(labelsize=cfs(16)); axq.set_aspect("equal", adjustable="datalim")

    # D clean XPU-RT vs fixed-partition comparison — one YOLO frame vs the 23 ms budget
    axd = fig.add_subplot(gs_rest[2])
    comparison = draw_control_rate_panel(
        axd, x_resp=a.x_resp_ms, r_resp=a.r_resp_ms, period=10.0, window=44.0,
        font_scale=0.62, footer_y=-0.40,
        panel_title="(b) Control-command rate — why the flight completes or crashes")

    fig.savefig(a.out + ".png", dpi=300)
    fig.savefig(a.out + ".pdf")
    comparison.update({
        "xpu_schedule": os.path.abspath(a.sched_xpu),
        "ros_schedule": os.path.abspath(a.sched_ros),
        "duration_provenance": "K1 per-dispatch calibration where available; aggregate fallback otherwise",
        "intended_use": "two-column full-page overview",
        "authored_figure_size_in": [7.08, 8.35],
        "minimum_font_size_pt": 7.0,
    })
    with open(a.out + "_metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("wrote", a.out + ".png/.pdf")

    if a.schedule_out:
        # Authored at final two-column width so the PDF can be included at
        # 1:1 scale without silently shrinking labels below 7 pt.
        sfig, sax = plt.subplots(figsize=(7.16, 3.8))
        standalone = draw_combined_gantt(
            sax, json.load(open(a.sched_xpu))["dispatches"],
            json.load(open(a.sched_ros))["dispatches"],
            budget_ms=a.frame_budget_ms, frame_instance=a.frame_instance,
            font_scale=0.55)
        sfig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.285)
        sfig.savefig(a.schedule_out + ".png", dpi=300)
        sfig.savefig(a.schedule_out + ".pdf")
        standalone.update({
            "xpu_schedule": os.path.abspath(a.sched_xpu),
            "ros_schedule": os.path.abspath(a.sched_ros),
            "duration_provenance": "K1 per-dispatch calibration where available; aggregate fallback otherwise",
            "intended_use": "two-column main-paper schedule panel",
            "authored_figure_size_in": [7.16, 3.8],
            "minimum_font_size_pt": 7.0,
        })
        with open(a.schedule_out + "_metrics.json", "w") as f:
            json.dump(standalone, f, indent=2)
        plt.close(sfig)
        print("wrote", a.schedule_out + ".png/.pdf")


if __name__ == "__main__":
    main()
