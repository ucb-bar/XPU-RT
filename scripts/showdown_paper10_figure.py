#!/usr/bin/env python3
"""Figure 10 for the paper, sized to its slot: the full text width (7.0 in) by the height the current figure occupies
(3.6 in). Type is sized to the paper (caption 8 pt, body 9 pt): ticks 6 pt, titles 6.6 pt bold, nothing under 5 pt. The two
panels a reader has to see — the top-down pair and the measured schedules — take the width; everything else that
does not fit lives in the atlas / final figure.

  A  the specific case (same scene, both runtimes, aisle at its true aspect) · a–d its moments (chase + FPV)
  B  the breaking point (1.7 m scene, cadence + latency replayed; CP-SAT / greedy / ROS 2 vanilla / hand-pinned)
  C  camera rate on the K1: latency and frames delivered, 5–120 Hz (colours as the arm labels of E)
  S  a second scene flown 12 times per arm (CP-SAT / ROS 2 vanilla / greedy), the pair bold
  D  the schedules themselves: CP-SAT · greedy · ROS 2 on all eight cores · ROS 2 vanilla (100 ms window), arms labelled per row
  E  where the latency goes (component medians, chain median)
Every number is read at render time; the sidecar <out>_metrics.json lists them; `--audit` prints every text that leaves the
page or collides with another panel's text (the render loop's acceptance test).
    scripts/showdown_paper10_figure.py [--width-in 7.0 --height-in 3.6 --dpi 600 --audit]
"""
from __future__ import annotations
import argparse, json, os, re, sys, datetime
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_v2.csv"))
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.ticker, matplotlib.text, matplotlib.transforms   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
import showdown_v3_figure as V   # noqa: E402
import showdown_atlas as AT   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
import showdown_final_figure as F   # noqa: E402
import figure_constants as FC   # noqa: E402

C_XPU, C_XPU2, C_ROS, C_ROS8 = V.C_XPU, V.C_XPU2, V.C_ROS, V.C_ROS8


def badge(ax, s, fz, dx=-7, dy=9, corner=(0, 1)):
    ax.annotate(s, xy=corner, xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontsize=fz["badge"], weight="bold", color="white",
                ha="center", va="center", zorder=40, annotation_clip=False, bbox=dict(boxstyle="circle,pad=0.25", fc="#2f6db0", ec="white", lw=0.6))


def shrink_marks(ax, k_line=0.35, k_scatter=0.12, lw_max=1.2, text_pt=None, skip_rotated=True):
    """bring a panel authored for a wall-sized figure down to print: marker sizes, line widths, in-axes text."""
    for l in ax.get_lines():
        l.set_markersize(l.get_markersize() * k_line); l.set_linewidth(min(l.get_linewidth(), lw_max)); l.set_markeredgewidth(min(l.get_markeredgewidth(), 0.6))
    for c in ax.collections:
        try:
            c.set_sizes(c.get_sizes() * k_scatter); c.set_linewidths([min(w, 0.6) for w in c.get_linewidths()])
        except (AttributeError, TypeError, ValueError):
            pass
    for t in ax.texts:
        if skip_rotated and t.get_rotation() == 90:
            continue
        if text_pt is not None:
            t.set_fontsize(min(t.get_fontsize(), text_pt))
        bb = t.get_bbox_patch()
        if bb is not None:
            bb.set_boxstyle(bb.get_boxstyle().__class__.__name__.lower().replace("boxstyle", "") or "round", pad=0.15); bb.set_linewidth(min(bb.get_linewidth(), 0.5))


def thin(ax, lw=0.5):
    for sp in ax.spines.values():
        sp.set_linewidth(lw)
    ax.tick_params(width=lw, length=2, pad=1.5)




def audit(fig, min_pt=5.0):
    """every visible text: inside the page, at least min_pt, and not crossing another panel's text or axes box."""
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; probs = []
    items = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ab = ax.get_window_extent(rd).expanded(1.02, 1.02)
        cand = list(ax.texts) + [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            cand += list(ax.get_legend().get_texts())
        ticks = [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible()]
        for t in cand + ticks:
            if not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0:
                continue
            try:
                bb = t.get_window_extent(rd)
            except Exception:
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            if t in ticks and not (bb.overlaps(ab)):
                continue   # a tick outside the view is not drawn
            if t.get_bbox_patch() is not None:
                try:
                    bb = matplotlib.transforms.Bbox.union([bb, t.get_bbox_patch().get_window_extent(rd)])
                except Exception:
                    pass
            items.append((t, bb))
    def owner(t):
        a = t.axes
        return a
    def in_legend(t):
        for ax in fig.axes:
            lg = ax.get_legend()
            if lg is not None and t in lg.get_texts():
                return lg
        return None
    for t, bb in items:
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            probs.append(("off-page", t.get_text()[:60], f"x {bb.x0/W:.3f}-{bb.x1/W:.3f} y {bb.y0/H:.3f}-{bb.y1/H:.3f}"))
        if t.get_fontsize() < min_pt - 1e-6:
            probs.append(("small", t.get_text()[:60], f"{t.get_fontsize():.1f} pt"))
    for i in range(len(items)):
        ti, bi = items[i]
        for j in range(i + 1, len(items)):
            tj, bj = items[j]
            if owner(ti) is owner(tj) and (in_legend(ti) is in_legend(tj)):
                continue   # same panel, same legend (or both free): the panel's own business
            if bi.overlaps(bj):
                ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0); iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ix > 0.75 and iy > 0.75:
                    probs.append(("collide", ti.get_text()[:40], tj.get_text()[:40]))
    for t, bb in items:   # titles / labels / legends spilling into a neighbouring axes box
        for ax in fig.axes:
            if ax is owner(t) or not ax.get_visible():
                continue
            ab = ax.get_window_extent(rd)
            if bb.overlaps(ab):
                ix = min(bb.x1, ab.x1) - max(bb.x0, ab.x0); iy = min(bb.y1, ab.y1) - max(bb.y0, ab.y0)
                if ix > 2 and iy > 2 and not (t in ax.texts):
                    probs.append(("into-axes", t.get_text()[:50], f"{ax.get_label() or id(ax)}"))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-in", type=float, default=7.0); ap.add_argument("--height-in", type=float, default=5.1); ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--base-pt", type=float, default=6.0, help="tick-label size in points (paper caption 8 pt, body 9 pt); titles 1.1x")
    ap.add_argument("--xpu-dir", default=os.path.join(RES, "campaign_v2/display_same/xpu_s1005_figdata")); ap.add_argument("--ros-dir", default=os.path.join(RES, "campaign_v2/display_same/ros_s1005_figdata"))
    ap.add_argument("--display-cruise", type=float, default=1.0); ap.add_argument("--out", default=os.path.join(RES, "refined", "warehouse_showdown_paper10"))
    ap.add_argument("--scene2-records", default=os.path.join(RES, "campaign_scene", "tall1000s")); ap.add_argument("--scene2-xpu-dir", default=os.path.join(RES, "campaign_v2/display_v3s_c1.4/xpu_s1000_figdata"))
    ap.add_argument("--scene2-ros-dir", default=os.path.join(RES, "campaign_v2/display_v3s_c1.4/ros_s1000_figdata")); ap.add_argument("--scene2-cruise", type=float, default=1.4)
    ap.add_argument("--audit", action="store_true", help="print texts that leave the page or collide, then exit non-zero if any")
    a = ap.parse_args()
    b = a.base_pt; fz = dict(tiny=b * 0.88, tick=b, lab=b * 1.03, leg=b * 0.9, title=b * 1.1, badge=b * 1.05, head=b * 1.3)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": b, "pdf.fonttype": 42, "axes.linewidth": 0.5, "lines.linewidth": 0.9, "patch.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "legend.handlelength": 1.4})
    X, R = S.load(a.xpu_dir), S.load(a.ros_dir)
    xxyz, rxyz, xt, rt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"], R["t_s"]; tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx, hr = V.eff_hz(X) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"), V.eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    moments, near_miss, hit = V.moments_for(X, R, 85, hx, hr); bg, bg_note, bg_std = V.backdrop(X)
    rows = [r for r in AT.all_flights() if r["camp"].startswith("campaign_")]

    fig = plt.figure(figsize=(a.width_in, a.height_in))
    Hh = a.height_in; top_in, r0_in, g1_in, r1_in, g2_in, r3_in, g3_in, r2_in, bot_in = 0.30, 0.92, 0.38, 0.52, 0.36, 0.92, 0.56, 0.77, 0.40
    outer = fig.add_gridspec(7, 1, height_ratios=[r0_in, g1_in, r1_in, g2_in, r3_in, g3_in, r2_in], hspace=0.0, left=0.012, right=0.976, top=1 - top_in / Hh, bottom=bot_in / Hh)
    r0 = outer[0].subgridspec(1, 3, width_ratios=[3.0, 1.62, 1.62], wspace=0.26)
    # A — the specific case, aisle at its true aspect (3.3:1 plate)
    axA = fig.add_subplot(r0[0]); axA.set_label("A")
    S.draw_topdown(axA, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, 85, rxyz[-1], moments, ov_obj=None, near_miss=near_miss)
    shrink_marks(axA, k_line=0.45, k_scatter=0.2, lw_max=1.6, text_pt=fz["tiny"])
    xg, rg = int(X["gates_passed"]), int(R["gates_passed"])
    mpos = {t.get_text(): np.array(t.get_position(), float) for t in axA.texts if t.get_text() in ("a", "b", "c", "d")}
    for k1, k2 in (("c", "a"), ("d", "b")):   # a moment marker that lands on another is lifted (screen px of the plate)
        if k1 in mpos and k2 in mpos and np.linalg.norm(mpos[k1] - mpos[k2]) < 80:
            for t in axA.texts:
                if t.get_text() == k1:
                    t.set_position((mpos[k1][0] + 10, mpos[k1][1] - 42))
            for c_ in axA.collections:
                off = c_.get_offsets()
                if len(off) == 1 and np.linalg.norm(np.asarray(off[0]) - mpos[k1]) < 2:
                    c_.set_offsets([[mpos[k1][0] + 10, mpos[k1][1] - 42]])
    for t in axA.texts:   # moment letters and their discs above the gate rings
        if t.get_text() in ("a", "b", "c", "d"):
            t.set_zorder(21)
            for c_ in axA.collections:
                off = c_.get_offsets()
                if len(off) == 1 and np.linalg.norm(np.asarray(off[0]) - np.asarray(t.get_position())) < 2:
                    c_.set_zorder(20)
    axA.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=2, label=f"XPU-RT · CP-SAT, {hx:.0f} Hz control · {xg}/4 gates (colour = time)"), Line2D([0], [0], color=C_ROS, lw=2, label=f"ROS 2 vanilla, {hr:.0f} Hz control · {rg}/4, then hits a {hit}"),
                        Line2D([0], [0], marker="o", color=V.C_MOVER, mec="white", ls="none", ms=3.5, label="patrolling people"), Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=1.2, ls="none", ms=3.5, label="gate")],
               loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=fz["leg"], frameon=False, ncol=2, handlelength=1.4, borderpad=0.0, borderaxespad=0.0, labelspacing=0.25, columnspacing=1.2, handletextpad=0.5)
    axA.set_title(f"Same scene and drone, {a.display_cruise:.1f} m/s: XPU-RT ({hx:.0f} Hz control)\nclears the course, ROS 2 ({hr:.0f} Hz) clears G{rg} and hits a {hit}", fontsize=fz["title"], weight="bold", loc="left", pad=3)
    badge(axA, "A", fz, dx=5, dy=-8)
    # B — the breaking point
    axB = fig.add_subplot(r0[1]); axB.set_label("B"); MB = F.draw_break(axB, rows, fz, show_dense=False, ms=3.0, lw=1.1)
    axB.set_title("Breaking point (1.7 m scene,\ncadence + latency replayed)", fontsize=fz["title"], weight="bold", loc="left", pad=3); thin(axB)
    for t in axB.texts:
        t.set_fontsize(fz["tiny"])
        if t.get_text() == "XPU-RT · CP-SAT" and hasattr(t, "xyann"):
            t.set_text("XPU-RT\nCP-SAT"); t.xyann = (4, 0); t.set_va("center"); t.set_linespacing(1.0)
        if t.get_text() == "ROS 2 vanilla" and hasattr(t, "xyann"):
            t.set_ha("left"); t.xyann = (-6, 9)
        if re.fullmatch(r"\d+/\d+", t.get_text()) and hasattr(t, "xyann") and t.xyann[1] < 0:
            t.set_visible(t.xy[1] > 0.12)   # the baseline's counts: shown where it completes anything (paired seeds: n is the XPU-RT label's n)
            t.xyann = (0, -9)
        if re.fullmatch(r"\d+/\d+", t.get_text()) and hasattr(t, "xyann") and t.xyann[1] > 0 and abs(t.xy[0] - 1.8) < 1e-6:
            t.xyann = (-8, 6)   # clear of the hand-pinned point above it
    if axB.get_legend():
        lg = axB.get_legend(); lg.set_bbox_to_anchor((1.02, 0.80))
        for t in lg.get_texts():
            t.set_fontsize(fz["tiny"]); t.set_text(t.get_text().replace("XPU-RT · greedy", "XPU-RT greedy").replace("ROS 2 hand-tuned (pinned)", "ROS 2 hand-pinned"))
    axB.set_xlim(0.7, 2.5); axB.set_ylim(-0.04, 1.14); axB.set_ylabel("course completed", fontsize=fz["lab"]); badge(axB, "B", fz)
    # C — camera rate on the board (colours = the arm labels of E)
    cg = r0[2].subgridspec(2, 1, height_ratios=[1.05, 1], hspace=0.14); axC1 = fig.add_subplot(cg[0]); axC2 = fig.add_subplot(cg[1], sharex=axC1); axC1.set_label("C1"); axC2.set_label("C2")
    MC = V.draw_F(axC1, axC2, V.BOARD_ARMS, fz); axC2.get_legend().remove()
    axC1.set_title("Camera rate on the K1:\nlatency, frames delivered", fontsize=fz["title"], weight="bold", loc="left", pad=3); thin(axC1); thin(axC2)
    axC2.set_xticks([5, 15, 30, 60, 120]); axC2.set_xticklabels(["5", "15", "30", "60", "120"]); axC2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axC2.set_xlabel("camera rate (Hz); colours as E", fontsize=fz["lab"], labelpad=1.5)
    for l in axC1.get_lines() + axC2.get_lines():
        l.set_markersize(2.4); l.set_linewidth(0.9)
    for t in axC2.texts:
        t.set_fontsize(fz["tiny"])
    axC1.set_ylim(20, 8000); axC1.tick_params(labelbottom=False); axC1.set_ylabel("ms", fontsize=fz["lab"]); axC2.set_ylabel("frames/s", fontsize=fz["lab"])
    axC1.text(0.03, 0.93, "camera→control latency", transform=axC1.transAxes, fontsize=fz["tiny"], color="0.3", va="top"); axC1.set_yticks([100, 1000]); axC1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    badge(axC1, "C", fz)
    # a–d — the moments (chase + FPV) and S — the second scene flown 12 times per arm
    r1 = outer[2].subgridspec(1, 4, wspace=0.10)
    for c, (src, step, lab) in enumerate(moments):
        dd = a.ros_dir if src == "ROS" else a.xpu_dir; Z = R if src == "ROS" else X; tt = (rt if src == "ROS" else xt)[min(step, len(Z["t_s"]) - 1)]
        f = S.frame_at(dd, Z["frame_steps"], step); col = r1[c].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.03); tc = C_ROS if src == "ROS" else C_XPU
        ac = fig.add_subplot(col[0]); ac.imshow(f["chase"][225:465, 415:655]); ac.axis("off"); ac.set_label(f"m{c}c")
        af = fig.add_subplot(col[1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="auto"); af.set_label(f"m{c}f")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = S.YOLO.get(int(cls), ("obj", "#39f")); af.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cc, lw=0.8))
        af.set_xticks([]); af.set_yticks([]); thin(af, 0.3)
        arm, act = lab.split(" · ", 1); arm = re.sub(r" [0-9]+ Hz$", "", arm)
        ac.set_title(f"{chr(ord('a') + c)}  {arm}, t = {tt:.1f} s: {act}", fontsize=fz["tiny"], weight="bold", color=tc, loc="left", pad=2)
    # S — a second scene flown 12 times per arm (the pair bold) · F — paired seeds · G — inside the pair of A
    r3 = outer[4].subgridspec(1, 3, width_ratios=[3.0, 1.55, 1.55], wspace=0.30)
    axS = fig.add_subplot(r3[0]); axS.set_label("S")
    X2, R2 = S.load(a.scene2_xpu_dir), S.load(a.scene2_ros_dir); MS = F.scene_panel(axS, X2, R2, a.scene2_records, a.scene2_cruise, "Second scene", fz)
    shrink_marks(axS, k_line=0.6, k_scatter=0.4, lw_max=1.6, text_pt=fz["tiny"])
    for l in axS.get_lines():
        l.set_linewidth(min(l.get_linewidth(), 1.6))
    lgS = axS.get_legend(); hS, lS = [], []
    for h, t in zip(lgS.legend_handles, lgS.get_texts()):
        txt = t.get_text()
        if re.search(r"\(\d+ Hz\)$", txt):
            continue   # the pair's own entries (labelled by their control rate) — their colours are those of A
        hS.append(h); lS.append(txt.replace("XPU-RT · CP-SAT", "CP-SAT").replace("XPU-RT · greedy", "greedy").replace(" completed", ""))
    lgS.remove(); axS.legend(handles=hS, labels=lS, fontsize=fz["tiny"], frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=4, handlelength=1.4, columnspacing=1.0, labelspacing=0.2, borderaxespad=0.0, handletextpad=0.5)
    axS.set_title(f"Second scene, {a.scene2_cruise:.1f} m/s: every flight, 12 per arm\n"
                  f"bold = the pair: ROS 2 clears G{int(R2['gates_passed'])}, hits a gate frame", fontsize=fz["title"], weight="bold", loc="left", pad=3)
    for sp in axS.spines.values():
        sp.set_linewidth(0.4)
    badge(axS, "S", fz, dx=8, dy=-8)
    axF = fig.add_subplot(r3[1]); axF.set_label("F"); MF = F.draw_pairs2(axF, rows, fz)
    axF.set_title(f"Same seed, 1.7 m scene:\nXPU-RT further on {MF['scene_1.7']['xpu']} pairs,\nROS 2 on {MF['scene_1.7']['ros']}, tied {MF['scene_1.7']['tie']}", fontsize=fz["title"], weight="bold", loc="left", pad=3)
    for t in axF.texts:
        t.set_fontsize(fz["tiny"])
    lgF = axF.get_legend(); lgF.set_bbox_to_anchor((0.0, 1.0)); [t.set_fontsize(fz["tiny"]) for t in lgF.get_texts()]; axF.set_ylim(0, axF.get_ylim()[1] * 1.1)
    axF.set_ylabel("seed pairs", fontsize=fz["lab"]); thin(axF); badge(axF, "F", fz)
    gg = r3[2].subgridspec(2, 1, hspace=0.12); axG1 = fig.add_subplot(gg[0]); axG2 = fig.add_subplot(gg[1], sharex=axG1); axG1.set_label("G1"); axG2.set_label("G2")
    tmp = [fig.add_subplot(outer[6]) for _ in range(2)]; MG = F.draw_telemetry([axG1, axG2, tmp[0], tmp[1]], X, R, fz); [t.remove() for t in tmp]
    axG1.set_title("Inside the pair of A:\nthe baseline thrashes", fontsize=fz["title"], weight="bold", loc="left", pad=3)
    axG2.set_title("", loc="left"); axG1.set_xlabel(""); axG1.tick_params(labelbottom=False); axG2.set_xlabel("time (s)", fontsize=fz["lab"], labelpad=1.5)
    axG1.set_ylabel("rad/s", fontsize=fz["lab"]); axG2.set_ylabel("deg", fontsize=fz["lab"])
    axG1.text(0.02, 0.95, "IMU body rate |ω|", transform=axG1.transAxes, fontsize=fz["tiny"], color="0.3", va="top"); axG2.text(0.02, 0.95, "nav goal heading", transform=axG2.transAxes, fontsize=fz["tiny"], color="0.3", va="top")
    for t in axG1.texts:
        if "crashes" in t.get_text():
            t.set_position((t.get_position()[0], 0.45))
    for ax_ in (axG1, axG2):
        shrink_marks(ax_, k_line=0.5, lw_max=1.0, text_pt=fz["tiny"]); thin(ax_); ax_.grid(True, color="0.92", lw=0.3)
    hG, lG = axG1.get_legend_handles_labels(); axG1.get_legend().remove() if axG1.get_legend() else None
    axG1.legend(handles=hG, labels=[re.sub(r" \(.*\)$", "", l) for l in lG], fontsize=fz["tiny"], frameon=False, loc="upper right", handlelength=1.4, labelspacing=0.15, borderaxespad=0.1); badge(axG1, "G", fz)
    # D — the schedules, wide
    r2 = outer[6].subgridspec(1, 2, width_ratios=[4.35, 1.5], wspace=0.52)
    axD = fig.add_subplot(r2[0]); axD.set_label("D"); cols = {"xpu": C_XPU, "xpu2": C_XPU2, "ros": C_ROS, "ros8": C_ROS8}; grows, gpaths = [], []
    for r in ["xpu:CP-SAT:xpu", "xpu2:greedy:xpu2", "ros8:ROS2 8c:ros8", "ros:ROS2:ros"]:
        name, label, ck = r.split(":"); p = os.path.join(REPO, "schedules", f"measured_gantt_v3_{name}.json")
        if os.path.exists(p):
            grows.append((json.load(open(p)), label, cols[ck], "ros" if ck.startswith("ros") else "xpu")); gpaths.append(p)
    S.draw_combined_gantt(axD, grows, gpaths); gtitle = axD.get_title(loc="left")
    V.rescale_fonts(axD, b / 15.0, row_label_pt=fz["tiny"], tick_pt=fz["tick"])
    rowpos = {}
    for t in axD.texts:
        if t.get_rotation() == 90 and t.get_position()[0] < -2:   # the rotated row labels: read their y, then hide them
            if t.get_position()[0] > -5:
                rowpos[t.get_text()] = t.get_position()[1]
            t.set_visible(False)
        if "camera→control" in t.get_text() or t.get_text().startswith("sensors in"):
            t.set_visible(False)   # the per-row label below carries chain and control period
        if getattr(t, "arrowprops", None):
            t.arrowprops["mutation_scale"] = t.arrowprops.get("mutation_scale", 13) * 0.3; t.arrowprops["lw"] = 0.4
    shrink_marks(axD, k_line=0.35, k_scatter=0.15, lw_max=1.0, text_pt=fz["tiny"])
    parts = [p.strip() for p in re.split(r"  ·  |\s—\s", gtitle)]
    nums = [re.findall(r"(\d+(?:\.\d+)?) ms", p) for p in parts[1:]]   # [chain, control gap] per row, sidecar-built
    late = [re.search(r"(\d+) of (\d+) frames late|every frame on time", p) for p in parts[1:]]
    names = {"CP-SAT": "XPU-RT · CP-SAT", "greedy": "XPU-RT · greedy", "ROS2 8c": "ROS 2, all 8 cores\n(two YOLO nodes)", "ROS2": "ROS 2 vanilla"}
    ylabs = []
    for (j, label, colour, kind), nm, lt in zip(grows, nums, late):
        ex = f"{nm[0]} ms · ctrl {nm[1]} ms" if len(nm) >= 2 else ""
        ylabs.append((rowpos.get(label, None), f"{names.get(label, label)}\n{ex}", colour))
    axD.set_yticks([])
    for y, l, c in ylabs:   # arm, chain and control period at the top-left of each row
        if y is not None:
            axD.text(0.6, y + 4.4, l.replace("\n(two YOLO nodes)\n", " (two YOLO nodes) — ").replace("\n", " — "), fontsize=fz["tiny"], weight="bold", color=c, va="top", ha="left", zorder=30,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88))
    axD.set_ylabel(""); axD.set_xlim(-0.8, axD.get_xlim()[1])
    axD.get_legend().remove()
    axD.set_xlabel("onboard schedule time (ms), K1 board — ▼ frame in, ▲ model output", fontsize=fz["lab"], labelpad=1.5)
    axD.set_title("Schedules measured on the K1: 45 Hz camera, 100 ms window, 8 harts per row\n"
                  "YOLO orange · nav purple, 22 ms windows shaded · control green, 10 ms windows", fontsize=fz["title"], weight="bold", loc="left", pad=3, x=0.05)
    for t in axD.texts:
        if t.get_visible() and t.get_fontsize() < 5.0:
            t.set_fontsize(5.0)
    axD.tick_params(labelsize=fz["tick"]); thin(axD); badge(axD, "D", fz, dx=8, dy=9)
    # E — where the latency goes
    axE = fig.add_subplot(r2[1]); axE.set_label("E"); ME = AT.draw_waterfall(axE, dict(fz, tiny=fz["tiny"]))
    axE.set_title("Where the latency goes\n(45 Hz camera, medians)", fontsize=fz["title"], weight="bold", loc="left", pad=3)
    short = {"XPU-RT · CP-SAT": "XPU-RT · CP-SAT", "XPU-RT · greedy": "XPU-RT · greedy", "ROS 2 hand-pinned": "ROS 2 hand-pinned", "ROS 2 vanilla, 4-hart YOLO": "ROS 2 vanilla",
             "ROS 2 two YOLO nodes, 8 cores": "ROS 2, all 8 cores", "ROS 2 vanilla, serial YOLO": "ROS 2, serial YOLO"}
    cols_e = [l.get_color() for l in axE.get_yticklabels()]
    axE.set_yticklabels([short.get(t.get_text(), t.get_text()) for t in axE.get_yticklabels()])
    for t, c in zip(axE.get_yticklabels(), cols_e):
        t.set_fontsize(fz["tiny"]); t.set_color(c); t.set_weight("bold")
    axE.set_xlabel("camera→control (ms): queue grey,\nYOLO orange, nav + control", fontsize=fz["lab"] * 0.9, labelpad=1.5); axE.get_legend().remove()
    for t in axE.texts:   # component labels narrower than the type are noise at print; the chain medians stay
        if re.fullmatch(r"\d+", t.get_text()) and int(t.get_text()) < 60:
            t.set_visible(False)
    axE.set_xlim(0, 900); axE.set_xticks([0, 200, 400, 600, 800]); thin(axE); badge(axE, "E", fz)
    for ax in (axB, axC1, axC2, axE, axF, axG1, axG2):
        ax.xaxis.label.set_fontsize(fz["lab"]); ax.yaxis.label.set_fontsize(fz["lab"]); ax.tick_params(labelsize=fz["tick"])
    probs = audit(fig)
    if a.audit:
        for p_ in probs:
            print("AUDIT", *p_)
        print(f"AUDIT {len(probs)} problem(s)")
    fig.savefig(a.out + ".png", dpi=a.dpi); fig.savefig(a.out + ".pdf"); print("wrote", a.out + ".pdf")
    side = {"figure": a.out + ".pdf", "size_in": [a.width_in, a.height_in], "base_pt": b, "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "A": {"xpu_gates": xg, "ros_gates": rg, "hit": hit, "xpu_eff_hz": round(hx, 1), "ros_eff_hz": round(hr, 1), "backdrop": bg_note}, "B": MB,
            "C": {re.sub(r"[^A-Za-z]", "", lab): {str(h): {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for h, p in pts.items()} for lab, pts in MC.items()},
            "S": {"cruise": a.scene2_cruise, "records": a.scene2_records, "counts": MS, "ros_gates": int(R2["gates_passed"])}, "F": MF, "G": MG,
            "D": gpaths, "E": ME, "audit": [list(p_) for p_ in probs],
            "sources": {"xpu_dir": a.xpu_dir, "ros_dir": a.ros_dir, "scene2_xpu_dir": a.scene2_xpu_dir, "scene2_ros_dir": a.scene2_ros_dir, "scene2_records": a.scene2_records,
                        "gantt_sidecars": [p_.replace(".json", "_metrics.json") for p_ in gpaths], "campaigns": sorted({r["camp"] for r in rows})},
            **FC.sidecar_common("showdown_paper10_figure", V.LOADED_CSVS + AT.LOADED_EXTRA + AT.WATERFALL_INPUTS + [os.path.join(RES, "ros_traced", "summary.csv")] + [p_.replace(".json", "_metrics.json") for p_ in gpaths])}
    json.dump(side, open(a.out + "_metrics.json", "w"), indent=1)
    return 0 if not (a.audit and probs) else 3


if __name__ == "__main__":
    sys.exit(main())
