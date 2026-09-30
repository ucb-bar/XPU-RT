#!/usr/bin/env python3
"""The warehouse showdown as one argument: measured ROS 2 arms against XPU-RT, every census paired.

One figure in several layouts (`--variant`), all drawn from data already on disk:

  A  top-down flight pair on the display scene: solver-placed XPU-RT against the ROS 2 arrangement
     that uses every hart (two YOLO pools and a nav pool, control chained to the goal), 36 Hz camera,
     with every recorded run of that scene drawn faintly beneath it (the scene census, 12 seeds/arm)
  B  paired censuses: every ROS 2 arm flown against an XPU-RT arm on the SAME (cruise, seed) cells
     with the same number of replicates, grouped by how the ROS 2 graph paces control -- chained to
     perception (the default) or on its own 100 Hz timer. Two columns share the rows: mean gates
     cleared (with a paired bootstrap interval on the difference) and course completion (Wilson 95 %)
  C  mechanism, measured on the K1: the control rate each deployment delivers against the camera rate
  D  the rate-injected envelope (hil_ablation.csv), both display arms' measured rates marked
  a-d  chase / FPV + YOLO / cross-ToF at four moments of the display pair
  E  body rate of the display pair
  I  the board-measured Gantt of the display pair's two schedules (ROS 2 misses in red, dashed)

Variants: `hero` (all of the above), `compact` (A+B over C+I, sized for a full-width paper figure),
`slide` (16:9), `envelope` (completion against cruise speed per arm, Wilson 95 %, standard scene at
45 Hz and the 1.7 m-people scene).

Pairing rule (panel B and the envelope): flights come through flight_quarantine.flight_rows via
showdown_v3_figure.load_campaigns; a cell is kept only when both arms flew it; within a cell each arm
keeps the first k flights, k the fewer either arm flew (showdown_v3_figure.equalise). A timeout is a
flight that did not complete the course and counts as not completed; gates are 4 for a completion
and gates_passed otherwise (figure_candidates.gates). The gate difference carries a percentile
bootstrap over the paired per-flight differences (figure_candidates.boot_gate_diff, 4000 resamples,
seed 11); completion carries a Wilson 95 % interval per arm (hil_envelope_panel.wilson).

    scripts/showdown_improved_figure.py --variant hero
        [--out results/codesign_feedback/refined/showdown_improved_hero] [--dpi 300]

Every drawn number is written to <out>_metrics.json with the sha256 of every input read, and
scripts/verify_showdown_improved.py re-derives them.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import math
import os
import sys
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_navpool36.csv"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

import showdown_v3_figure as V   # noqa: E402
import showdown_final_figure as F   # noqa: E402
import showdown_paper_figure as P   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
import figure_constants as FC   # noqa: E402
import measured_timing as MT   # noqa: E402
from hil_envelope_panel import wilson   # noqa: E402
from figure_candidates import gates as flight_gates, boot_gate_diff   # noqa: E402
from showdown_atlas import newcombe   # noqa: E402

C_XPU, C_ROS = V.C_XPU, V.C_ROS          # #1f9e5a, #e2231a
INK, GREY = "#22242a", "#8b8b8b"
BOOT_N, BOOT_SEED = 4000, 11

# ------------------------------------------------------------------------------------------ inputs
DISPLAY = dict(
    xpu_dir=os.path.join(RES, "campaign_free36/display/pairs_ac36/xpu_s1006_figdata"),
    ros_dir=os.path.join(RES, "campaign_free36/display/pairs_ns4/ros_s1006_figdata"),
    scene=os.path.join(RES, "campaign_scene/tall1000_ac36"),
    xpu_trace="xpu_p36free.csv", ros_trace="ros_vanilla4x236ns4.csv",
    xpu_scene_dir="xpu_p36free", ros_scene_dir="ros_vanilla4x236ns4",
    gantt_prefix=os.path.join(RES, "refined/navshard36/measured_gantt_navshard36"),
    camera_hz=36, cruise=1.4)

# Every ROS 2 arm that was flown against an XPU-RT arm on the same cells of the standard scene
# (course a, prop density 0.30, 2.4 m people, latency replayed). (group, camera Hz, ROS trace, what the
# ROS arm is, XPU-RT trace, what the XPU-RT arm is, campaigns the pair is read from, gain).
CHAINED, TIMER = "chained", "timer"
CENSUS = [
    (CHAINED, 45, "ros_vanilla_c5045.csv", "serial YOLO, ctrl in goal callback (as submitted)",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_submitted",), 0.0055),
    (CHAINED, 45, "ros_vanilla445.csv", "4-hart YOLO pool, unpinned",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_percep",), 0.0055),
    (CHAINED, 45, "ros_vanilla4x245.csv", "two YOLO instances over all 8 harts",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_percep",), 0.0055),
    (CHAINED, 45, "ros_cp345.csv", "hand-pinned to 6 cores",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_static6_45",), 0.0055),
    (CHAINED, 45, "ros_cp345.csv", "hand-pinned to 6 cores",
     "xpu_p45free.csv", "CP-SAT, solver-placed", ("campaign_free45",), 0.0055),
    (CHAINED, 40, "ros_vanilla4x240.csv", "two YOLO instances over all 8 harts",
     "xpu_w2pg40.csv", "greedy on measured costs", ("campaign_rate3640",), 0.0055),
    (CHAINED, 36, "ros_vanilla4x236.csv", "two YOLO instances over all 8 harts",
     "xpu_w2pg36.csv", "greedy on measured costs", ("campaign_rate3640",), 0.0055),
    (CHAINED, 30, "ros_vanilla4x230.csv", "two YOLO instances over all 8 harts",
     "xpu_a30_cpsat.csv", "CP-SAT", ("campaign_rate30",), 0.0055),
    (CHAINED, 30, "ros_cp3n430.csv", "partitioned, nav on the E cores",
     "xpu_p30free.csv", "CP-SAT, solver-placed", ("campaign_free30_eq",), 0.005),
    (TIMER, 45, "ros_vanilla4tm45.csv", "4-hart YOLO pool + control timer",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_percep",), 0.0055),
    (TIMER, 45, "ros_p345.csv", "hand-pinned + control timer",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_percep",), 0.0055),
    (TIMER, 45, "ros_p3_q145.csv", "hand-pinned + control timer + keep-last-1",
     "xpu_a_cpsat_hard.csv", "CP-SAT", ("campaign_percep",), 0.0055),
    (TIMER, 30, "ros_p330.csv", "hand-pinned + control timer",
     "xpu_p30free.csv", "CP-SAT, solver-placed", ("campaign_p3_30", "campaign_free30_eq"), 0.005),
]
# the one-line row names the short layouts print, in CENSUS order
SHORT = ["serial YOLO (as submitted)", "4-hart YOLO pool", "2 YOLO instances, 8 harts", "pinned, 6 cores",
         "pinned, 6 cores (vs solver)", "2 YOLO instances, 8 harts", "2 YOLO instances, 8 harts",
         "2 YOLO instances, 8 harts", "partitioned, nav on E", "4-hart pool + timer", "pinned + timer",
         "pinned + timer + keep-last-1", "pinned + timer"]
GROUP_TITLE = {CHAINED: "control chained to perception — ROS 2 as normally written",
               TIMER: "control on its own 100 Hz timer — the expert change"}

# Panel C: board runs of the same chain at several camera rates. ROS 2 arms by measured_timing key;
# XPU-RT from the registered CP-SAT replay arms (camera rate and control gap from figure_constants).
RATE_ARMS = [("vanilla4x2", "ROS 2 · two instances, chained", C_ROS, "-"),
             ("cp3", "ROS 2 · hand-pinned, chained", "#b0170f", "--"),
             ("vanilla", "ROS 2 · serial YOLO, chained", "#f08a84", ":"),
             ("p3", "ROS 2 · hand-pinned, control timer", "#5c6bc0", "-"),
             ("vanilla4x2tm", "ROS 2 · two instances, control timer", "#8e99d6", "--")]

# The envelope variant's two cells: the standard scene at 45 Hz, and the 1.7 m-people scene
ENV45 = [("ros_vanilla445.csv", "ROS 2 · 4-hart pool, chained", C_ROS, "-"),
         ("ros_cp345.csv", "ROS 2 · hand-pinned, chained", "#b0170f", "--"),
         ("ros_p345.csv", "ROS 2 · hand-pinned + timer", "#5c6bc0", "-"),
         ("xpu_a_cpsat_hard.csv", "XPU-RT · CP-SAT", C_XPU, "-"),
         ("xpu_p45free.csv", "XPU-RT · CP-SAT, solver-placed", "#0e6b3b", "--")]
ENV17 = [("ros_vanilla445.csv", "ROS 2 · 4-hart pool, chained", C_ROS, "-"),
         ("ros_p345.csv", "ROS 2 · hand-pinned + timer", "#5c6bc0", "-"),
         ("xpu_a_cpsat_hard.csv", "XPU-RT · CP-SAT", C_XPU, "-")]


def ctrl_hz(trace):
    return round(1000.0 / FC.ctrl_gap_ms(trace), 1)


# ------------------------------------------------------------------------------------------ census
def pair_cells(rows, arms, camps, gain, person_h=2.4, dens=0.30):
    """{trace: [flights]} on the (cruise, seed) cells every arm in `arms` flew, equal replicates."""
    cells = V.flight_cells(rows, "a", dens, gain, latency=True, person_h=person_h, camps=set(camps))
    per = {t: collections.defaultdict(list) for t in arms}
    for (t, c), v in cells.items():
        if t in per:
            for r in v:
                per[t][(c, r["seed"])].append(r)
    common = set.intersection(*(set(per[t]) for t in arms)) if all(per[t] for t in arms) else set()
    sub = {}
    for (t, c), v in cells.items():
        if t in arms:
            keep = [r for r in v if (c, r["seed"]) in common]
            if keep:
                sub[(t, c)] = keep
    eq, dropped = V.equalise(sub, list(arms))
    out = {t: [] for t in arms}
    for (t, c), v in sorted(eq.items(), key=lambda kv: kv[0][1]):
        if t in out:
            out[t] += sorted(v, key=lambda r: r["seed"])
    return out, sorted(common), dropped


def _g(r):
    return flight_gates({"outcome": r["outcome"], "gates_passed": r["gates"]})


def census_rows(rows):
    """Panel B: one record per paired census."""
    out = []
    for grp, cam, rtr, rlab, xtr, xlab, camps, gain in CENSUS:
        fl, common, dropped = pair_cells(rows, (xtr, rtr), camps, gain)
        X, R = fl[xtr], fl[rtr]
        # pair flight i of one arm with flight i of the other within each cell
        bx, br = collections.defaultdict(list), collections.defaultdict(list)
        for r in X:
            bx[(r["cruise"], r["seed"])].append(r)
        for r in R:
            br[(r["cruise"], r["seed"])].append(r)
        pairs = [(_g(a), _g(b)) for k in sorted(bx) for a, b in zip(bx[k], br[k])]
        d, lo, hi = boot_gate_diff(pairs, n=BOOT_N, seed=BOOT_SEED)
        kx = sum(r["outcome"] == "success" for r in X); kr = sum(r["outcome"] == "success" for r in R)
        rec = dict(group=grp, short=SHORT[len(out)], camera_hz=cam, ros_trace=rtr, ros_arm=rlab, xpu_trace=xtr, xpu_arm=xlab,
                   campaigns=list(camps), gain=gain, cells=len(common), flights_per_arm=len(X),
                   replicates_set_aside=dropped,
                   ros_ctrl_hz=ctrl_hz(rtr), xpu_ctrl_hz=ctrl_hz(xtr),
                   xpu_completed=[kx, len(X)], ros_completed=[kr, len(R)],
                   xpu_wilson95=[round(v, 4) for v in wilson(kx, len(X))[1:]],
                   ros_wilson95=[round(v, 4) for v in wilson(kr, len(R))[1:]],
                   xpu_timeouts=sum(r["outcome"] == "timeout" for r in X),
                   ros_timeouts=sum(r["outcome"] == "timeout" for r in R),
                   xpu_mean_gates=round(float(np.mean([a for a, _ in pairs])), 3),
                   ros_mean_gates=round(float(np.mean([b for _, b in pairs])), 3),
                   gate_diff=[round(d, 3), round(lo, 3), round(hi, 3)],
                   per_flight={"xpu_more": sum(a > b for a, b in pairs), "tie": sum(a == b for a, b in pairs),
                               "xpu_fewer": sum(a < b for a, b in pairs)})
        cd = newcombe(kx, len(X), kr, len(R))
        rec["completion_diff_newcombe95"] = [round(v_, 4) for v_ in cd]
        # a paired difference whose interval straddles zero is a tie; one that sits below zero favours ROS 2
        rec["verdict"] = "XPU-RT ahead" if lo > 0 else ("ROS 2 ahead" if hi < 0 else "tie")
        out.append(rec)
    return out


def envelope_series(rows, arms, camps, person_h, cruises=None, censor_timeouts=False):
    """completion per cruise speed on the cells every arm flew, equal replicates (envelope variant)."""
    traces = [a[0] for a in arms]
    cells = V.flight_cells(rows, "a", 0.30, 0.0055, latency=True, person_h=person_h, camps=set(camps) if camps else None)
    if censor_timeouts:     # the breaking-point panel's rule: a flight airborne at the horizon is left out
        cells = {k: [r for r in v if r["outcome"] != "timeout"] for k, v in cells.items()}
    per = {t: collections.defaultdict(list) for t in traces}
    for (t, c), v in cells.items():
        if t in per and (cruises is None or c in cruises):
            for r in v:
                per[t][(c, r["seed"])].append(r)
    common = set.intersection(*(set(per[t]) for t in traces))
    sub = {(t, c): [r for r in v if (c, r["seed"]) in common] for (t, c), v in cells.items() if t in per}
    sub = {k: v for k, v in sub.items() if v}
    eq, dropped = V.equalise(sub, traces)
    out = {}
    for t in traces:
        pts = []
        for c in sorted({c for (tt, c) in eq if tt == t}):
            v = eq[(t, c)]; k = sum(r["outcome"] == "success" for r in v)
            p, lo, hi = wilson(k, len(v))
            pts.append({"cruise": c, "completed": k, "flown": len(v), "wilson95": [round(lo, 4), round(hi, 4)],
                        "mean_gates": round(float(np.mean([_g(r) for r in v])), 3)})
        tot_k = sum(p["completed"] for p in pts); tot_n = sum(p["flown"] for p in pts)
        out[t] = {"points": pts, "completed": [tot_k, tot_n]}
    return out, len(common), dropped


# ------------------------------------------------------------------------------------------ rates
def rate_curves():
    """Panel C: delivered control rate against camera rate, from the board runs."""
    ra = MT.derive()["ros_arms"]; out = {}
    for arm, lab, col, ls in RATE_ARMS:
        pts = sorted((int(k.split("@")[1]), round(1000.0 / v["gap_mean_pooled"], 2)) for k, v in ra.items()
                     if k.split("@")[0] == arm and v.get("gap_mean_pooled"))
        out[arm] = {"label": lab, "points": pts}
    xp = {}
    for a in FC._ARMS:
        if a.family == "xpu" and a.derive and a.derive[0] == "SOLVER_ARMS" and "CP-SAT" in a.label and a.ctrl_gap:
            g = FC._resolve(a.ctrl_gap)
            if g:
                xp.setdefault(a.cam_hz, []).append((a.derive[1], round(1000.0 / float(g), 2)))
    out["xpu"] = {"label": "XPU-RT · CP-SAT schedules", "points": sorted((h, min(v for _, v in l)) for h, l in xp.items()),
                  "runs": {str(h): l for h, l in sorted(xp.items())}}
    return out


# ------------------------------------------------------------------------------------------ scene
def scene_runs(scene_dir, sub):
    fl = []
    for f in sorted(glob.glob(os.path.join(scene_dir, sub, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        fl.append(dict(path=f, poses=z["poses"][:, :3], outcome=str(z["outcome"]), gates=int(z["gates_passed"]),
                       seed=int(z["seed"]), layout_seed=int(z["layout_seed"]), obst0=z["obst_pos0"], pmask=z["person_mask"].astype(bool)))
    return fl


def draw_runs_overlay(ax, X, runs_by_arm, path_start=85, lw=1.3):
    """the scene census beneath the display pair: each recorded run projected into the overhead camera,
    thin and translucent, with its end marked (dot = completed, x = collision)."""
    H, W = X["ov_bg"].shape[:2] if "ov_bg" in X.files else (None, None)
    for col, runs in runs_by_arm:
        for r in runs:
            u, v, ok = S.project(X["ovK"], X["ovpos"], X["ovquat"], r["poses"])
            vis = ok & (np.arange(len(u)) >= path_start)
            ax.plot(u[vis], v[vis], color=col, lw=lw, alpha=0.30, zorder=2.6, solid_capstyle="round")
            if vis.any():
                j = np.where(vis)[0][-1]
                if r["outcome"] == "success":
                    ax.scatter(u[j], v[j], s=36, marker="o", color=col, edgecolors="white", linewidths=0.8, alpha=0.9, zorder=2.7)
                else:
                    ax.scatter(u[j], v[j], s=46, marker="x", color=col, linewidths=1.8, alpha=0.9, zorder=2.7)


# ------------------------------------------------------------------------------------------ drawing
def badge(ax, s, fz, dx=-8, dy=26):
    P.badge(ax, s, fz, dx=dx, dy=dy)


def provenance(ax, text, fz, xy=(0.995, 1.0), ha="right", va="bottom"):
    """the small stamp each panel carries: where its numbers come from and how many."""
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va, fontsize=fz["tiny"] * 0.92, color="#4a4a4a",
            style="italic", zorder=30, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))


def draw_topdown(ax, X, R, fz, runs, strobe_s=1.0, legend=True, legend_ncol=2):
    """panel A: the display pair as the showdown figures draw it, the scene census beneath."""
    xxyz, rxyz, xt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"]
    tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx, hr = V.eff_hz(X), V.eff_hz(R)
    r_out = str(R["outcome"]) if "outcome" in R.files else ""
    moments, near_miss, hit = V.moments_for(X, R, 85, hx, hr, outcome=r_out)
    bg, bg_note, bg_std = V.backdrop(X)
    S.draw_topdown(ax, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, 85, rxyz[-1], moments,
                   ov_obj=None, near_miss=near_miss, strobe_s=strobe_s, xt=X["t_s"], rt=R["t_s"])
    draw_runs_overlay(ax, X, [(C_XPU, runs["xpu"]), (C_ROS, runs["ros"])])
    kx = sum(r["outcome"] == "success" for r in runs["xpu"]); kr = sum(r["outcome"] == "success" for r in runs["ros"])
    if legend:
        hs = [Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=f"XPU-RT display flight: {int(X['gates_passed'])} of 4 gates (colour = time)"),
              Line2D([0], [0], color=C_ROS, lw=5, label=f"ROS 2 display flight: {int(R['gates_passed'])} of 4, hits a {hit}"),
              Line2D([0], [0], color=C_XPU, lw=1.6, alpha=0.6, marker="o", ms=5, label=f"XPU-RT, all {len(runs['xpu'])} scene runs: {kx} complete"),
              Line2D([0], [0], color=C_ROS, lw=1.6, alpha=0.6, marker="x", ms=6, mew=1.6, label=f"ROS 2, all {len(runs['ros'])} scene runs: {kr} complete"),
              Line2D([0], [0], marker="o", color="0.45", mec="white", ls="none", ms=9, label=f"● = display drone every {strobe_s:.0f} s"),
              Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")]
        ax.legend(handles=hs, loc="lower left", fontsize=fz["leg"], framealpha=0.93, ncol=legend_ncol, handlelength=1.9)
    return dict(moments=moments, near_miss=near_miss, hit=hit, bg_note=bg_note, bg_std=bg_std, hx=hx, hr=hr, r_out=r_out,
                scene_completed={"xpu": [kx, len(runs["xpu"])], "ros": [kr, len(runs["ros"])]},
                scene_mean_gates={"xpu": round(float(np.mean([4 if r["outcome"] == "success" else r["gates"] for r in runs["xpu"]])), 3),
                                  "ros": round(float(np.mean([4 if r["outcome"] == "success" else r["gates"] for r in runs["ros"]])), 3)})


def draw_census(fig, gs, recs, fz, show_xpu_arm=True, title=True, widths=(1.55, 0.95, 0.85, 1.05), wrap=54, one_line=False, legend_dy=-0.30):
    """panel B: one row per paired census; columns = the arms (labels) | mean gates cleared, both arms |
    the paired difference XPU-RT − ROS 2 with its bootstrap interval | course completion, Wilson 95 %."""
    sub = gs.subgridspec(1, 4, width_ratios=list(widths), wspace=0.07)
    axl, axg, axd, axc = (fig.add_subplot(sub[i]) for i in range(4))
    ys, heads, y = [], [], 0.0
    for grp in (CHAINED, TIMER):
        heads.append((grp, y)); y += 0.9
        for r in [r for r in recs if r["group"] == grp]:
            ys.append((r, y)); y += 1.0
        y += 0.3
    ymax = y
    for ax in (axl, axg, axd, axc):
        ax.set_ylim(ymax - 0.25, -0.7); ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=fz["tick"])
    axl.axis("off"); axl.set_xlim(0, 1)
    for ax in (axg, axd, axc):
        ax.patch.set_alpha(0.0)
        ax.grid(axis="x", ls=":", lw=0.6, color="#d8d5cf", zorder=0)
    band = {"tie": "#eceef7", "ROS 2 ahead": "#fbe9e8"}
    for r, yy in ys:                                         # rows that tie (or favour ROS 2) are banded across every column
        if r["verdict"] in band:
            for ax in (axl, axg, axd, axc):
                ax.axhspan(yy - 0.48, yy + 0.48, color=band[r["verdict"]], zorder=0, lw=0)
    for grp, yy in heads:
        axl.text(0.0, yy, GROUP_TITLE[grp], ha="left", va="center", fontsize=fz["lab"], weight="bold", color=INK, zorder=20)
    for r, yy in ys:
        line1 = f"{r['camera_hz']} Hz · {r['ros_arm']}"
        line2 = (f"ROS 2 {r['ros_ctrl_hz']:.0f} Hz ctrl · vs XPU-RT {r['xpu_arm']}" if show_xpu_arm
                 else f"ROS 2 control {r['ros_ctrl_hz']:.0f} Hz")
        if r["gain"] != 0.0055:          # a different controller gain is part of the row's name
            line1 += f" · gain {r['gain']:g}"
        if one_line:
            axl.text(0.03, yy, f"{r['camera_hz']} Hz · {r['short']} · {r['ros_ctrl_hz']:.0f} Hz ctrl" + (f" · g {r['gain']:g}" if r["gain"] != 0.0055 else ""), ha="left", va="center", fontsize=fz["tick"], color=INK)
            continue
        axl.text(0.03, yy - 0.17, textwrap.shorten(line1, wrap, placeholder="…"), ha="left", va="center", fontsize=fz["tick"], color=INK)
        axl.text(0.03, yy + 0.25, textwrap.shorten(line2, int(wrap * 1.12), placeholder="…"), ha="left", va="center", fontsize=fz["tiny"] * 0.95, color="#555")
    ms = fz["tick"] * 7.5
    for r, yy in ys:                                         # mean gates, both arms
        gx, gr = r["xpu_mean_gates"], r["ros_mean_gates"]
        axg.plot([gr, gx], [yy, yy], color="#b9b7b2", lw=2.4, zorder=2, solid_capstyle="round")
        axg.scatter([gr], [yy], s=ms, color=C_ROS, edgecolors="white", linewidths=0.9, zorder=4)
        axg.scatter([gx], [yy], s=ms, color=C_XPU, edgecolors="white", linewidths=0.9, zorder=5)
    axg.set_xlim(0, 2.6); axg.set_xticks([0, 1, 2])
    axg.set_xlabel("mean gates\n(of 4)", fontsize=fz["lab"])
    for r, yy in ys:                                         # the paired difference
        d, lo, hi = r["gate_diff"]
        col = C_XPU if r["verdict"] == "XPU-RT ahead" else (C_ROS if r["verdict"] == "ROS 2 ahead" else "#4a57a6")
        axd.errorbar([d], [yy], xerr=[[d - lo], [hi - d]], fmt="D", ms=fz["tick"] * 0.5, color=col, ecolor=col, elinewidth=1.8, capsize=2.5, mec="white", mew=0.7, zorder=4)
        axd.annotate(f"{d:+.2f}", (hi, yy), textcoords="offset points", xytext=(4, 0), ha="left", va="center", fontsize=fz["tiny"] * 0.92, color=col, weight="bold")
    axd.axvline(0, color=INK, lw=1.1, zorder=1)
    axd.set_xlim(-0.6, 2.6); axd.set_xticks([0, 1, 2])
    axd.set_xlabel("gates, XPU-RT − ROS 2\n(paired 95 % CI)", fontsize=fz["lab"])
    for r, yy in ys:                                         # completion, Wilson 95 %
        for key, ci, col, dy in (("ros_completed", "ros_wilson95", C_ROS, 0.18), ("xpu_completed", "xpu_wilson95", C_XPU, -0.18)):
            k, n = r[key]; lo, hi = r[ci]; p = k / n
            axc.errorbar([100 * p], [yy + dy], xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o", ms=fz["tick"] * 0.5, color=col,
                         ecolor=col, elinewidth=1.4, capsize=2.0, mec="white", mew=0.7, zorder=4)
            axc.annotate(f"{k}/{n}", (100 * hi, yy + dy), textcoords="offset points", xytext=(3, 0), ha="left", va="center",
                         fontsize=fz["tiny"] * 0.88, color=col, weight="bold")
    axc.set_xlim(0, 45); axc.set_xticks([0, 20, 40]); axc.xaxis.set_major_formatter(lambda v, _p: f"{v:.0f} %")
    axc.set_xlabel("completed\n(Wilson 95 %)", fontsize=fz["lab"])
    hs = [Line2D([0], [0], marker="o", color=C_XPU, ls="none", ms=7, label="XPU-RT"), Line2D([0], [0], marker="o", color=C_ROS, ls="none", ms=7, label="ROS 2"),
          plt.Rectangle((0, 0), 1, 1, fc=band["tie"], ec="none", label="level (gate CI spans 0)")]
    if any(r["verdict"] == "ROS 2 ahead" for r in recs):
        hs.append(plt.Rectangle((0, 0), 1, 1, fc=band["ROS 2 ahead"], ec="none", label="ROS 2 ahead"))
    if legend_dy is None:
        axc.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=fz["tiny"], frameon=False, ncol=len(hs), handlelength=1.0, columnspacing=0.8)
    else:
        axd.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, legend_dy), fontsize=fz["tiny"], frameon=False, ncol=len(hs), handlelength=1.1, columnspacing=1.2)
    if title:
        axl.set_title(title if isinstance(title, str) else "Every paired census: the same cells and replicates per arm",
                      fontsize=fz["title"], weight="bold", loc="left", pad=fz["title"] * 0.6)
    n_fl = sum(2 * r["flights_per_arm"] for r in recs)
    return axl, axc, n_fl


def draw_rates(ax, curves, fz, cam=None, legend=True):
    """panel C: delivered control rate against camera rate, board runs of the same chain."""
    for arm, lab, col, ls in RATE_ARMS:
        pts = curves[arm]["points"]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker="o", ms=fz["tick"] * 0.42, color=col, lw=2.0, label=lab, zorder=3)
    xp = curves["xpu"]["points"]
    ax.plot([p[0] for p in xp], [p[1] for p in xp], "-s", color=C_XPU, lw=3.0, ms=fz["tick"] * 0.55, mec="white", label=curves["xpu"]["label"], zorder=5)
    ax.plot([0, 130], [0, 130], color="0.6", lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.text(70, 70, "control = camera", rotation=0, fontsize=fz["tiny"], color="0.45", ha="left", va="top")
    if cam:
        ax.axvline(cam, color="0.35", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(cam + 1.5, 4, f"display {cam} Hz", ha="left", va="bottom", fontsize=fz["tiny"], color="0.3")
    ax.set_xlim(0, 125); ax.set_ylim(0, 112)
    ax.set_xlabel("camera rate (Hz)", fontsize=fz["lab"]); ax.set_ylabel("control rate delivered (Hz)", fontsize=fz["lab"])
    ax.grid(ls=":", lw=0.6, color="#d8d5cf"); ax.tick_params(labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(fontsize=fz["tiny"] * 0.92, frameon=True, framealpha=0.95, loc="center right", bbox_to_anchor=(1.0, 0.63), handlelength=2.2)


def draw_bodyrate(ax, X, R, fz):
    """panel E: the display pair's body rate (the showdown figures' panel E, one axis)."""
    xw = S.smooth(np.linalg.norm(X["imu_w"], axis=1)); rw = S.smooth(np.linalg.norm(R["imu_w"], axis=1))
    xt, rt = X["t_s"], R["t_s"]
    ax.plot(xt, xw, color=C_XPU, lw=2.0, label=f"XPU-RT ({V.eff_hz(X):.0f} Hz control)")
    ax.plot(rt, rw, color=C_ROS, lw=2.0, label=f"ROS 2 ({V.eff_hz(R):.0f} Hz control)")
    ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); ax.axvline(rt[-1], color=C_ROS, lw=1.6, ls=(0, (4, 2)))
    ax.set_xlim(0, xt.max()); ax.set_xlabel("time (s)", fontsize=fz["lab"]); ax.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=fz["lab"])
    ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=fz["tick"]); ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return {"xpu_mean_w": round(float(np.mean(np.linalg.norm(X["imu_w"], axis=1))), 3),
            "ros_mean_w": round(float(np.mean(np.linalg.norm(R["imu_w"], axis=1))), 3)}


def draw_envelope_cell(ax, series, arms, fz, title, n_note):
    for tr, lab, col, ls in arms:
        pts = series[tr]["points"]
        xs = [p["cruise"] for p in pts]; ps = [p["completed"] / p["flown"] for p in pts]
        lo = [p["wilson95"][0] for p in pts]; hi = [p["wilson95"][1] for p in pts]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.07, lw=0)
        k, n = series[tr]["completed"]
        ax.plot(xs, ps, ls=ls, marker="o", color=col, lw=2.2, ms=fz["tick"] * 0.5, mec="white", label=f"{lab}: {k}/{n}", zorder=4)
    ax.set_ylim(-0.02, 1.0); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("course completed (fraction)", fontsize=fz["lab"])
    ax.grid(ls=":", lw=0.6, color="#d8d5cf"); ax.tick_params(labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper left")
    ax.set_title(title, fontsize=fz["title"], weight="bold", loc="left")
    provenance(ax, n_note, fz, xy=(1.0, -0.13), va="top")


# ------------------------------------------------------------------------------------------ layouts
def fonts(width_in, min_pt, ref_w=7.1):
    base = min_pt * width_in / ref_w
    return base, dict(tiny=base * 0.9, tick=base, lab=base * 1.05, leg=base * 0.95, title=base * 1.2, badge=base * 1.1, head=base * 1.45)


def verdicts(recs):
    return {v: sum(r["verdict"] == v for r in recs) for v in ("XPU-RT ahead", "tie", "ROS 2 ahead")}


def headline(fig, fz, recs, y=0.995, x=0.01, width=150, dy=0.022):
    """the figure's claim, counted from the censuses rather than typed in."""
    v = verdicts(recs)
    ch = [r for r in recs if r["group"] == CHAINED]; tm = [r for r in recs if r["group"] == TIMER]
    lv_ch = [r for r in ch if r["verdict"] != "XPU-RT ahead"]; lv_tm = [r for r in tm if r["verdict"] != "XPU-RT ahead"]
    lv = [r for r in recs if r["verdict"] != "XPU-RT ahead"]
    pinned = [r for r in lv if "hand-pinned" in r["ros_arm"]]; other = [r for r in lv if r not in pinned]
    more = [r for r in recs if r["ros_completed"][0] > r["xpu_completed"][0]]
    more_txt = [f"{r['camera_hz']} Hz {r['ros_arm']}: {r['ros_completed'][0]} vs {r['xpu_completed'][0]}" for r in more]
    t1 = ("XPU-RT derives automatically what ROS 2 needs an expert to hand-build, and beats ROS 2 as normally deployed — "
          f"on gates cleared, ahead in {v['XPU-RT ahead']} of {len(recs)} paired censuses, level in {v['tie']}, behind in {v['ROS 2 ahead']}")
    t2 = (f"With control chained to perception, as ROS 2 is normally written, XPU-RT is ahead in {len(ch) - len(lv_ch)} of {len(ch)}. "
          f"ROS 2 draws level when hand-pinned ({len(pinned)} censuses)" + (f" or at a {', '.join(sorted({str(r['camera_hz']) for r in other}))} Hz camera" if other else "")
          + f"; with a hand-added 100 Hz control timer it is level in {len(lv_tm)} of {len(tm)} (the one behind keeps a "
          f"{FC.lat_ms('ros_vanilla4tm45.csv'):.0f} ms camera→goal chain). "
          + (f"ROS 2 completes more courses in {len(more)} (" + "; ".join(more_txt) + ")" + (", each inside its Newcombe interval. " if all(r["completion_diff_newcombe95"][1] < 0 < r["completion_diff_newcombe95"][2] for r in more) else ". ") if more else "")
          + "XPU-RT schedules control in its own 100 Hz slot at every camera rate; its solver finds the placement without hand-pinning.")
    fig.text(x, y, t1, fontsize=fz["head"], weight="bold", ha="left", va="top", color=INK)
    fig.text(x, y - dy, "\n".join(textwrap.wrap(t2, width)), fontsize=fz["lab"], ha="left", va="top", color="#333")
    return {"verdicts": v, "ros_completes_more": [[r["camera_hz"], r["ros_trace"], r["xpu_trace"]] for r in more],
            "chained": {"n": len(ch), "level": [r["ros_trace"] for r in lv_ch]},
            "timer": {"n": len(tm), "level": [r["ros_trace"] for r in lv_tm]}, "text": [t1, t2]}


def load_display():
    X, R = S.load(DISPLAY["xpu_dir"]), S.load(DISPLAY["ros_dir"])
    runs = {"xpu": scene_runs(DISPLAY["scene"], DISPLAY["xpu_scene_dir"]), "ros": scene_runs(DISPLAY["scene"], DISPLAY["ros_scene_dir"])}
    return X, R, runs


def title_A(A, width):
    t = (f"A · The display scene, {DISPLAY['camera_hz']} Hz camera, {DISPLAY['cruise']:.1f} m/s: solver-placed XPU-RT ({A['hx']:.0f} Hz control) "
         f"against ROS 2 over all 8 harts — two YOLO pools and a nav pool, control chained ({A['hr']:.0f} Hz)")
    return "\n".join(textwrap.wrap(t, width))


def gantt_legend_below(ax, fz, dy=-0.30, ncol=6):
    """in the short layouts the Gantt's legend sits under the time axis rather than over the XPU-RT lanes."""
    lg = ax.get_legend()
    if lg:
        hs, ls = lg.legend_handles, [t.get_text() for t in lg.get_texts()]
        lg.remove()
        ax.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, fontsize=fz["tiny"], frameon=False,
                  handlelength=1.6, columnspacing=1.2)


def gantt(ax, fz, base):
    return P.gantt_two_rows(ax, fz, base, names=("xpu", "ros"), prefix=DISPLAY["gantt_prefix"], camera_hz=DISPLAY["camera_hz"],
                            xpu_label="XPU-RT · solver-placed CP-SAT")


def layout_hero(a, recs, curves, X, R, runs):
    W = a.width_in or 22.0
    base, fz = fonts(W, a.min_print_pt)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, W * 0.90))
    outer = fig.add_gridspec(6, 1, height_ratios=[6.0, 0.95, 2.3, 0.85, 2.9, 4.4], hspace=0.0, left=0.035, right=0.985, top=0.925, bottom=0.03)
    r0 = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.05)
    axA = fig.add_subplot(r0[0])
    A = draw_topdown(axA, X, R, fz, runs)
    axA.set_title(title_A(A, 92).split(" · ", 1)[1], fontsize=fz["title"], weight="bold", loc="left")
    provenance(axA, f"Isaac Sim, replaying each arm's K1-measured cadence + latency · scene census n = {len(runs['xpu'])} runs per arm",
               fz, xy=(1.0, -0.01), va="top")
    badge(axA, "A", fz, dx=-24, dy=14)
    axl, axc, nB = draw_census(fig, r0[1], recs, fz, legend_dy=None, wrap=58)
    badge(axl, "B", fz, dx=-24, dy=16)
    provenance(axl, f"Isaac Sim, K1 timing replayed · {nB} flights", fz, xy=(0.0, -0.02), ha="left", va="top")
    F.strips(fig, outer[2].subgridspec(1, 4, wspace=0.09), A["moments"], X, R, DISPLAY["xpu_dir"], DISPLAY["ros_dir"], fz)
    r2 = outer[4].subgridspec(1, 3, width_ratios=[1.0, 1.25, 0.95], wspace=0.28)
    axC = fig.add_subplot(r2[0]); draw_rates(axC, curves, fz, cam=DISPLAY["camera_hz"])
    axC.set_title("Mechanism, measured on the K1: chained control\ntracks the camera; a timer or XPU-RT's slot holds 100 Hz", fontsize=fz["title"], weight="bold", loc="left")
    nruns = sum(len(v["points"]) for k, v in curves.items() if k != "xpu") + sum(len(l) for l in curves["xpu"]["runs"].values())
    provenance(axC, f"K1 board runs · {nruns} arm × camera points", fz, xy=(0.01, 0.80), ha="left", va="bottom")
    dg = r2[1].subgridspec(1, 2, width_ratios=[40, 1], wspace=0.04); axD = fig.add_subplot(dg[0]); cax = fig.add_subplot(dg[1])
    MB = F.draw_envelope_paper(axD, cax, A["hx"], A["hr"], fz, base); MB["k_n_per_hz"] = P.envelope_counts(os.path.join(RES, "hil_ablation.csv"))
    axD.set_title(f"Why the rate matters: completion against control rate,\nrate injected in Isaac Sim ({MB['flights']} flights), the display arms marked",
                  fontsize=fz["title"], weight="bold", loc="left")
    axE = fig.add_subplot(r2[2]); ME = draw_bodyrate(axE, X, R, fz)
    for ax_, l_ in ((axC, "C"), (axD, "D"), (axE, "E")):
        badge(ax_, l_, fz, dx=-14, dy=40)
    axE.set_title("Inside the display pair: body rate\n(the baseline thrashes at its control rate)", fontsize=fz["title"], weight="bold", loc="left")
    gs5 = outer[5].subgridspec(2, 1, height_ratios=[0.33, 1.0])
    axI = fig.add_subplot(gs5[1]); sides, MI = gantt(axI, fz, base)
    badge(axI, "I", fz, dx=-22, dy=30)
    provenance(axI, "Measured on K1 · board traces of the two display arms", fz, xy=(1.0, -0.13), va="top")
    H = headline(fig, fz, recs, width=int(W * 9.8), dy=0.019)
    return fig, A, dict(B_flights=nB, D=MB, E=ME, I=MI, headline=H), sides


def layout_compact(a, recs, curves, X, R, runs):
    W = a.width_in or 12.0
    base, fz = fonts(W, a.min_print_pt)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, W * 0.84))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.40, left=0.02, right=0.985, top=0.95, bottom=0.07)
    r0 = outer[0].subgridspec(1, 2, width_ratios=[0.78, 1.0], wspace=0.04)
    axA = fig.add_subplot(r0[0]); A = draw_topdown(axA, X, R, fz, runs, legend=False)
    kx, nx = A["scene_completed"]["xpu"]; kr, nr = A["scene_completed"]["ros"]
    axA.set_title(f"A · {DISPLAY['camera_hz']} Hz camera: XPU-RT solver-placed ({A['hx']:.0f} Hz ctrl) vs\nROS 2 on all 8 harts, chained ({A['hr']:.0f} Hz); "
                  f"scene runs {kx}/{nx} vs {kr}/{nr}", fontsize=fz["title"], weight="bold", loc="left")
    axA.legend(handles=[Line2D([0], [0], color=C_XPU, lw=2.5, label="XPU-RT"), Line2D([0], [0], color=C_ROS, lw=2.5, label="ROS 2"),
                        Line2D([0], [0], color="0.4", lw=1.0, alpha=0.6, label="faint: every scene run")],
               loc="lower left", fontsize=fz["tiny"], framealpha=0.9, ncol=3, handlelength=1.4, columnspacing=0.8)
    provenance(axA, "Isaac Sim · K1 timing replayed", fz, xy=(1.0, -0.01), va="top")
    axl, axc, nB = draw_census(fig, r0[1], recs, fz, show_xpu_arm=False, title=f"B · {len(recs)} paired censuses: on gates {verdicts(recs)['XPU-RT ahead']} ahead, {verdicts(recs)['tie']} level, {verdicts(recs)['ROS 2 ahead']} behind",
                               widths=(2.1, 0.66, 0.7, 0.8), wrap=46, one_line=True, legend_dy=-0.24)
    provenance(axl, f"Isaac Sim, K1 timing replayed · {nB} flights", fz, xy=(0.0, -0.03), ha="left", va="top")
    r1 = outer[1].subgridspec(1, 2, width_ratios=[0.62, 1.7], wspace=0.25)
    axC = fig.add_subplot(r1[0]); draw_rates(axC, curves, fz, cam=DISPLAY["camera_hz"], legend=False)
    axC.set_title("C · control rate (K1)", fontsize=fz["title"], weight="bold", loc="left")
    axC.text(122, 96, "XPU-RT, ROS 2 timer", color=C_XPU, fontsize=fz["tiny"], ha="right", va="top", weight="bold")
    axC.text(122, 56, "ROS 2 chained", color=C_ROS, fontsize=fz["tiny"], ha="right", va="top", weight="bold")
    axI = fig.add_subplot(r1[1]); sides, MI = gantt(axI, fz, base); gantt_legend_below(axI, fz, -0.22)
    axI.set_title(f"I · the two schedules, measured on the K1: XPU-RT control every {MI['xpu']['ctrl_gap_mean_ms']:.1f} ms, "
                  f"ROS 2 every {MI['ros']['ctrl_gap_mean_ms']:.1f} ms", fontsize=fz["title"], weight="bold", loc="left")
    return fig, A, dict(B_flights=nB, I=MI), sides


def layout_slide(a, recs, curves, X, R, runs):
    W = a.width_in or 13.333
    base, fz = fonts(W, a.min_print_pt, ref_w=13.333)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, W * 9 / 16))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.10, left=0.015, right=0.99, top=0.855, bottom=0.075)
    lc = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.42)
    rc = outer[1].subgridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.62)
    axA = fig.add_subplot(lc[0]); A = draw_topdown(axA, X, R, fz, runs, legend=False, strobe_s=1.0)
    kx, nx = A["scene_completed"]["xpu"]; kr, nr = A["scene_completed"]["ros"]
    axA.set_title(f"One scene, {DISPLAY['camera_hz']} Hz camera, every recorded run faint, one pair bold:\n"
                  f"XPU-RT {kx}/{nx} complete, ROS 2 on all 8 harts {kr}/{nr}", fontsize=fz["title"] * 0.92, weight="bold", loc="left")
    axA.legend(handles=[Line2D([0], [0], color=C_XPU, lw=3, label=f"XPU-RT · solver-placed · {A['hx']:.0f} Hz control"),
                        Line2D([0], [0], color=C_ROS, lw=3, label=f"ROS 2 · control chained · {A['hr']:.0f} Hz")],
               loc="lower left", fontsize=fz["tiny"], framealpha=0.9, ncol=2, handlelength=1.6)
    v = verdicts(recs)
    axl, axc, nB = draw_census(fig, rc[0], recs, fz, show_xpu_arm=False, widths=(1.7, 0.66, 0.7, 0.8), wrap=40, one_line=True, legend_dy=None,
                               title=f"{len(recs)} paired censuses · {sum(2 * r['flights_per_arm'] for r in recs)} flights")
    axC = fig.add_subplot(rc[1]); draw_rates(axC, curves, fz, cam=DISPLAY["camera_hz"], legend=False)
    axC.text(122, 96, "XPU-RT, ROS 2 timer", color=C_XPU, fontsize=fz["tiny"], ha="right", va="top", weight="bold")
    axC.text(122, 56, "ROS 2 chained", color=C_ROS, fontsize=fz["tiny"], ha="right", va="top", weight="bold")
    axC.set_ylabel("control (Hz)", fontsize=fz["lab"])
    axC.set_title("Why: control rate delivered, measured on the K1", fontsize=fz["title"] * 0.92, weight="bold", loc="left")
    axI = fig.add_subplot(lc[1]); sides, MI = gantt(axI, fz, base); gantt_legend_below(axI, fz, -0.2, ncol=3)
    axI.set_title(f"The two schedules on the K1 at {DISPLAY['camera_hz']} Hz: XPU-RT control every "
                  f"{MI['xpu']['ctrl_gap_mean_ms']:.1f} ms, ROS 2 every {MI['ros']['ctrl_gap_mean_ms']:.1f} ms", fontsize=fz["title"] * 0.92, weight="bold", loc="left")
    fig.text(0.015, 0.99, f"XPU-RT derives what ROS 2 needs an expert to hand-build — on gates cleared ahead in {v['XPU-RT ahead']}, "
             f"level in {v['tie']}, behind in {v['ROS 2 ahead']} of {len(recs)} paired censuses", fontsize=fz["head"] * 0.9, weight="bold", ha="left", va="top")
    fig.text(0.015, 0.935, "ROS 2 draws level when hand-pinned or given a control timer; chained to perception its control rate is its camera's. "
             "XPU-RT holds 100 Hz without hand-pinning. Timing measured on the SpacemiT K1, flights in Isaac Sim.",
             fontsize=fz["lab"] * 0.95, ha="left", va="top", color="#333")
    return fig, A, dict(B_flights=nB, I=MI, headline={"verdicts": v}), sides


def layout_envelope(a, recs, curves, X, R, runs, env):
    W = a.width_in or 20.0
    base, fz = fonts(W, a.min_print_pt, ref_w=7.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, W * 0.30), sharey=True)
    s45, n45, _ = env["std45"]; s17, n17, _ = env["tall17"]; s2, n2, _ = env["tall17_two_arm"]
    draw_envelope_cell(a1, s45, ENV45, fz, "Standard scene (2.4 m people), five arms",
                       f"{n45} cells × first replicate, every arm")
    draw_envelope_cell(a2, s17, ENV17, fz, "1.7 m-people scene, three arms",
                       f"{n17} cells every arm flew, airborne-at-horizon left out")
    draw_envelope_cell(a3, s2, [ENV17[0], ENV17[2]], fz, "1.7 m-people scene, the two arms flown most",
                       f"{n2} cells, airborne-at-horizon left out")
    for ax in (a2, a3):
        ax.set_ylabel("")
    fig.suptitle("Completion against cruise speed at a 45 Hz camera, every arm on the same cells (Isaac Sim, K1-measured cadence + latency, Wilson 95 %)",
                 fontsize=fz["title"] * 1.1, weight="bold", x=0.01, ha="left", y=1.04)
    fig.tight_layout(w_pad=2.0)
    return fig, None, {}, []


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("hero", "compact", "slide", "envelope"), required=True)
    ap.add_argument("--out", default=None, help="output stem (default refined/showdown_improved_<variant>)")
    ap.add_argument("--width-in", type=float, default=None); ap.add_argument("--min-print-pt", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    a.out = a.out or os.path.join(RES, "refined", f"showdown_improved_{a.variant}")
    if a.min_print_pt is None:
        a.min_print_pt = {"hero": 3.4, "compact": 5.0, "slide": 7.5, "envelope": 3.2}[a.variant]

    rows = V.load_campaigns()
    recs = census_rows(rows)
    curves = rate_curves()
    X, R, runs = load_display()
    env = None
    if a.variant == "envelope":
        env = {"std45": envelope_series(rows, ENV45, None, 2.4),
               "tall17": envelope_series(rows, ENV17, ("campaign_break",), 1.7, censor_timeouts=True),
               # the two-arm pairing the final figure's breaking-point panel draws (sidecar I_break), for comparison
               "tall17_two_arm": envelope_series(rows, [ENV17[0], ENV17[2]], ("campaign_break",), 1.7, censor_timeouts=True)}
        fig, A, M, sides = layout_envelope(a, recs, curves, X, R, runs, env)
    else:
        fig, A, M, sides = {"hero": layout_hero, "compact": layout_compact, "slide": layout_slide}[a.variant](a, recs, curves, X, R, runs)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png")

    inputs = list(V.LOADED_CSVS) + [os.path.join(RES, "flight_quarantine.csv"), os.path.join(RES, "ros_traced", "summary.csv")]
    side = {"figure": os.path.relpath(a.out + ".png", REPO), "variant": a.variant,
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "pairing_rule": {"cells": "both arms flew the (cruise, seed) cell", "replicates": "first k per cell, k = the fewer either arm flew",
                             "timeout": "counted as not completed", "gates": "4 for a completion, else gates_passed",
                             "gate_diff_bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED}, "completion_interval": "Wilson 95 %"},
            "census": recs, "control_rate": curves}
    if a.variant != "envelope":
        inputs += [os.path.join(DISPLAY["xpu_dir"], "figure_data.npz"), os.path.join(DISPLAY["ros_dir"], "figure_data.npz")]
        inputs += [r["path"] for v in runs.values() for r in v] + list(sides)
        inputs += [f"{DISPLAY['gantt_prefix']}_{n}.json" for n in ("xpu", "ros")]
        side["A"] = {"xpu_dir": os.path.relpath(DISPLAY["xpu_dir"], REPO), "ros_dir": os.path.relpath(DISPLAY["ros_dir"], REPO),
                     "xpu_trace": DISPLAY["xpu_trace"], "ros_trace": DISPLAY["ros_trace"], "camera_hz": DISPLAY["camera_hz"],
                     "display_cruise": DISPLAY["cruise"],
                     "episode_seed": {"xpu": int(X["seed"]), "ros": int(R["seed"])}, "layout_seed": {"xpu": int(X["layout_seed"]), "ros": int(R["layout_seed"])},
                     "xpu_gates": int(X["gates_passed"]), "ros_gates": int(R["gates_passed"]), "ros_outcome": A["r_out"], "hit": A["hit"],
                     "xpu_eff_hz": round(A["hx"], 1), "ros_eff_hz": round(A["hr"], 1),
                     "clearance_m": round(float(A["near_miss"][2]), 3), "backdrop": A["bg_note"],
                     "moments": [{"arm": s_, "step": int(st_), "label": lb_} for s_, st_, lb_ in A["moments"]],
                     "scene_dir": os.path.relpath(DISPLAY["scene"], REPO), "scene_completed": A["scene_completed"], "scene_mean_gates": A["scene_mean_gates"],
                     "scene_runs": {k: [{"seed": r["seed"], "outcome": r["outcome"], "gates": r["gates"]} for r in v] for k, v in runs.items()}}
        side.update(M)
    if env:
        inputs += []
        side["envelope"] = {k: {"series": v[0], "cells": v[1], "replicates_set_aside": v[2]} for k, v in env.items()}
        side["envelope"]["std45"]["arms"] = [t for t, *_ in ENV45]; side["envelope"]["tall17"]["arms"] = [t for t, *_ in ENV17]
        side["envelope"]["tall17_two_arm"]["arms"] = [ENV17[0][0], ENV17[2][0]]
        side["envelope"]["tall17"]["timeouts"] = "left out (airborne at the horizon), as the breaking-point panel does"
    side.update(FC.sidecar_common("showdown_improved_figure", [p for p in inputs if p and os.path.exists(p)]))
    with open(a.out + "_metrics.json", "w") as f:
        json.dump(side, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", a.out + "_metrics.json")
    for r in recs:
        print(f"  {r['group']:<8}{r['camera_hz']:>3} Hz  {r['ros_trace']:<24} vs {r['xpu_trace']:<22} n={r['flights_per_arm']:<3} "
              f"gates {r['xpu_mean_gates']:.2f}/{r['ros_mean_gates']:.2f} diff {r['gate_diff']}  done {r['xpu_completed']}/{r['ros_completed']}  {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
