#!/usr/bin/env python3
"""The showdown atlas: what was run, on the board and in flight, and what every axis of it says.

  A the census — every flight campaign and board run family, by runtime and outcome
  B the specific comparison — the same-scene pair, top-down
  C frequency on the K1 — camera-rate envelope: latency and frames delivered
  D the hand-tuning ladder on the K1 at 45 and 90 Hz
  E where the latency goes — the chain split from the traces
  F the ablation forest — Δ course completion (XPU-RT − ROS 2) with 95 % intervals, one row per condition
  G paired seeds — who gets further on the same seed, per speed
  H course fraction against cruise, graded
  I CP-SAT against greedy in every regime — frames late per camera rate, flights
  J added load; K the feedback loop closing
Every number is read from results/codesign_feedback/ at render time; the sidecar lists them.
    scripts/showdown_atlas.py [--out results/codesign_feedback/refined/warehouse_showdown_atlas] [--dpi 300]
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, math, os, re, statistics, sys, datetime
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_v2.csv"))
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import showdown_v3_figure as V   # noqa: E402
import story_figures as SF   # noqa: E402
import figure_constants as FC   # noqa: E402
import measured_timing as MT   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
from hil_envelope_panel import wilson   # noqa: E402
from make_measured_gantt_pair import read_trace   # noqa: E402
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

INK = "#22242a"; C_XPU, C_XPU2, C_ROS, C_ROS8, C_P3 = V.C_XPU, V.C_XPU2, V.C_ROS, V.C_ROS8, V.C_P3
XPU_L, ROS_L = "#b7e0c8", "#f5c6c2"


def sec(ax, s, fz, dx=-8, dy=26):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontsize=fz["badge"], weight="bold", color="white",
                ha="center", va="center", zorder=40, annotation_clip=False, bbox=dict(boxstyle="circle,pad=0.32", fc="#2f6db0", ec="white", lw=1.8))


# ---------------------------------------------------------------------------------------------- data
def load_csv(path, camp):
    rows = []
    for r in flight_rows(path):
        try:
            rows.append(dict(camp=camp, trace=os.path.basename(r.get("ctrl_trace", "") or ""), course=r.get("course") or "a", dens=round(float(r.get("prop_density") or 0.3), 2),
                             walk=float(r.get("walk_speed") or 0), cross=int(float(r.get("walk_cross") or 0)), gain=round(float(r["moment_scale"]), 5),
                             lat=round(float(r.get("percep_latency_ms") or 0), 1), hold=round(float(r.get("percep_hold_ms") or 0), 1), cruise=round(float(r["cruise_speed"]), 2),
                             gates=int(float(r["gates_passed"])), outcome=r["outcome"], seed=int(float(r["seed"])), sched=float(r.get("sched_latency_ms") or 0), eff_hz=float(r.get("eff_cmd_hz") or 0), ph=round(float(r.get("person_h") or 2.4), 2)))
        except (KeyError, ValueError):
            continue
    return rows


LOADED_EXTRA: list[str] = []   # the CSVs all_flights() adds beyond campaign_*/campaign.csv, for the sidecar's input hashes


def all_flights():
    rows = V.load_campaigns()
    for f, camp in (("campaign_env/env_sweep.csv", "campaign_env"), ("campaign_v2/campaign_v2.csv", "campaign_v2"), ("campaign_v2_courseB/campaign_v2.csv", "campaign_v2_courseB"),
                    ("campaign/campaign.csv", "campaign_v1"), ("hil_ablation.csv", "hil_ablation"), ("hil_ablation_courseB.csv", "hil_ablation_courseB"), ("gain_controlled/gain_controlled.csv", "gain_controlled")):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            rows += load_csv(p, camp); LOADED_EXTRA.append(p)
    return rows


def one_per_seed(v):
    seen = set(); out = []
    for r in v:
        if (r["cruise"], r["seed"]) not in seen:
            seen.add((r["cruise"], r["seed"])); out.append(r)
    return out


def newcombe(k1, n1, k2, n2):
    """difference of proportions p1 - p2 with a Newcombe (Wilson) 95 % interval."""
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


# ---------------------------------------------------------------------------------------- panels
CAMP_LABEL = {"campaign_percep": "cadence + latency replay (course A, 3 densities, 13 arms)", "campaign_env": "environment sweep, cadence replay (courses A/B × 3 densities)",
              "campaign_v2": "cadence replay, people 1.7 m (10 speeds, 2 gain policies)", "campaign_rich": "heavier stack (5 networks, 45/90 Hz)", "hil_ablation": "control-rate grid, rate injected (course A)",
              "gain_controlled": "control-rate grid, gain 0.5/rate", "campaign_v1": "first cadence campaign", "campaign_cross": "people crossing the aisle", "campaign_courseC": "course C",
              "hil_ablation_courseB": "control-rate grid, course B", "campaign_tallcal": "gain calibrated per rate, people 2.4 m", "campaign_tallnet": "retrained guidance net",
              "campaign_qos1": "baseline with QoS depth 1", "campaign_v2_courseB": "course B, cadence replay", "campaign_seeds24": "extra seeds on the display cell", "campaign_walk": "people walking 1.5 m/s",
              "campaign_break": "breaking point: people 1.7 m, cadence + latency replay (7 speeds)"}


def draw_census(ax, rows, fz):
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        fam = "xpu" if r["trace"].startswith("xpu") else ("ros" if r["trace"].startswith("ros") else "sim")
        per[r["camp"]][(fam, r["outcome"] == "success")] += 1
    camps = sorted(per, key=lambda c: -sum(per[c].values()))
    y = np.arange(len(camps)); left = np.zeros(len(camps))
    for key, col, lab in ((("xpu", True), C_XPU, "XPU-RT, course completed"), (("xpu", False), XPU_L, "XPU-RT, crashed"), (("ros", True), C_ROS, "ROS 2, course completed"), (("ros", False), ROS_L, "ROS 2, crashed"),
                          (("sim", True), "#777", "rate injected, completed"), (("sim", False), "#cfcfcf", "rate injected, crashed")):
        vals = np.array([per[c][key] for c in camps]); ax.barh(y, vals, left=left, color=col, edgecolor="white", lw=0.4, height=0.7, label=lab); left += vals
    for i, c in enumerate(camps):
        ax.text(left[i] + 8, i, f"{int(left[i])}", va="center", fontsize=fz["tiny"], color=INK)
    ax.set_yticks(y); ax.set_yticklabels([CAMP_LABEL.get(c, c) for c in camps], fontsize=fz["tiny"]); ax.invert_yaxis()
    ax.set_xlabel("flights", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="x", ls=":", lw=0.5); ax.legend(fontsize=fz["tiny"], frameon=False, loc="lower right", ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n_fl = len(rows); n_ros = len(glob.glob(os.path.join(RES, "ros_traced", "*_r*", "summary.json"))); n_xpu = len(glob.glob(os.path.join(RES, "xpurt_long", "trace_*_other_run*.csv")))
    n_sol = len(glob.glob(os.path.join(REPO, "schedules", "fig_*_clamped.json"))); n_fb = len(glob.glob(os.path.join(RES, "refined", "hil_feedback_*_metrics.json")))
    ax.set_title(f"What was run: {n_fl:,} recorded flights in {len(camps)} campaigns (12 seeds a cell),\n{n_ros} ROS 2 and {n_xpu} XPU-RT runs on the K1, {n_sol} solved tables executed, {n_fb} feedback studies", fontsize=fz["title"], weight="bold", loc="left")
    return {"flights": n_fl, "campaigns": len(camps), "ros_board_runs": n_ros, "xpu_board_runs": n_xpu, "solved_tables": n_sol, "feedback_studies": n_fb, "per_campaign": {c: dict((f"{k[0]}_{'done' if k[1] else 'crash'}", v) for k, v in per[c].items()) for c in camps}}


def draw_topdown(ax, xpu_dir, ros_dir, fz):
    X, R = S.load(xpu_dir), S.load(ros_dir)
    xxyz, rxyz, xt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"]; tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx, hr = V.eff_hz(X) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"), V.eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    moments, near_miss, hit = V.moments_for(X, R, 85, hx, hr); bg, note, std = V.backdrop(X)
    S.draw_topdown(ax, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, 85, rxyz[-1], [], ov_obj=None, near_miss=near_miss)
    xg, rg = int(X["gates_passed"]), int(R["gates_passed"])
    xl, rl = FC.lat_label("xpu_a_cpsat_hard.csv"), FC.lat_label("ros_vanilla445.csv")
    ax.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=f"XPU-RT · CP-SAT ({hx:.0f} Hz control, {xl}) · {xg} of 4 gates"), Line2D([0], [0], color=C_ROS, lw=5, label=f"ROS 2 vanilla ({hr:.0f} Hz, {rl}) · {rg} of 4, then hits {hit}")],
              loc="upper left", fontsize=fz["leg"], framealpha=0.93, ncol=1, handlelength=1.9)
    cru = float(X["cruise_speed"]) if "cruise_speed" in X.files else 1.0; ph = float(X["person_h"]) if "person_h" in X.files else 2.4
    ax.set_title(f"The same scene, controller and gain — only what the K1 delivers differs ({cru:.1f} m/s, people {ph:.1f} m)", fontsize=fz["title"], weight="bold", loc="left")
    return {"xpu_gates": xg, "ros_gates": rg, "hit": hit, "backdrop": note}


def draw_ladder(ax_lat, ax_goal, fz):
    ladder = [("vanilla\nserial", "vanilla"), ("vanilla\n4-hart", "vanilla4"), ("+ control\ntimer", "vanilla4tm"), ("+ QoS 1", "vanilla4_q1"), ("2 YOLO\nnodes", "vanilla4x2"), ("pinned\n3 proc.", "p3"), ("pinned\n+ QoS 1", "p3_q1"), ("XPU-RT\nCP-SAT", None)]
    xp = V.xpu_rate_points("cpsat"); pts = {lay: V.ros_rate_points(lay) for _, lay in ladder if lay}
    x = np.arange(len(ladder)); cols = [C_XPU if lay is None else C_ROS for _, lay in ladder]; out = {}
    for ax, key, ylab in ((ax_lat, "lat", "camera → control (ms)"), (ax_goal, "deliv", "goals to control (/s)")):
        for hz, dx, hatch, al in ((45, -0.19, None, 0.95), (90, 0.19, "///", 0.6)):
            vals = [(xp.get(hz, {}) if lay is None else pts[lay].get(hz, {})).get(key) for _, lay in ladder]
            ax.bar(x + dx, [v or 0 for v in vals], color=cols, width=0.36, edgecolor="k", lw=0.4, hatch=hatch, alpha=al, label=f"{hz} Hz camera")
            for i, v in enumerate(vals):
                if v:
                    ax.annotate(f"{v:.0f}", (i + dx, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=fz["tiny"] * 0.85)
                    out[f"{ladder[i][0]}@{hz}".replace("\n", " ")] = out.get(f"{ladder[i][0]}@{hz}".replace("\n", " "), {}) | {key: round(v, 1)}
        if key == "lat":
            ax.set_yscale("log"); ax.set_ylim(20, 1500); ax.tick_params(labelbottom=False); ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper left")
        else:
            for hz in (45, 90):
                ax.axhline(hz, color="k", ls="--", lw=0.7, alpha=0.5); ax.text(-0.45, hz + 1, f"offered {hz}/s", ha="left", va="bottom", fontsize=fz["tiny"] * 0.85)
            ax.set_xticks(x); ax.set_xticklabels([l for l, _ in ladder], fontsize=fz["tiny"])
        ax.set_ylabel(ylab, fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="y", ls=":", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ros90 = [pts[lay][90]["deliv"] for _, lay in ladder if lay and 90 in pts[lay] and pts[lay][90].get("deliv")]; xpu90 = xp.get(90, {}).get("deliv")
    ax_lat.set_title("Hand-tuning ROS 2 on the K1: pinning reaches the schedule's\nlatency at 45 Hz, but every layout stops at "
                     + (f"{min(ros90):.0f}–{max(ros90):.0f} goals/s\nat 90 Hz, where the solved table delivers {xpu90:.0f}" if ros90 and xpu90 else "its goal rate at 90 Hz"), fontsize=fz["title"], weight="bold", loc="left")
    return out


WATERFALL_INPUTS: list[str] = []


def draw_waterfall(ax, fz):
    arms = SF.WATERFALL_ARMS; out = {}
    for i, (lab, col, paths) in enumerate(arms):
        have = [p for p in paths if os.path.exists(p)]; w = [x for p in have for x in SF.waterfall(read_trace(p))]
        if not w:
            continue
        WATERFALL_INPUTS.extend(have); sm = SF.waterfall_summary(w); left = 0
        for j in range(4):
            m = sm["parts_ms"][j]
            ax.barh(i, m, left=left, color=SF.WATERFALL_COLOURS[j], edgecolor="white", lw=0.5, height=0.66, label=SF.WATERFALL_PARTS[j] if i == 0 else None)
            if m > 12:
                ax.text(left + m / 2, i, f"{m:.0f}", ha="center", va="center", fontsize=fz["tiny"], color=INK if j == 0 else "white", weight="bold")
            left += m
        ax.text(left + 6, i, f"{sm['chain_median_ms']:.0f} ms", va="center", fontsize=fz["tiny"], color=col, weight="bold"); out[lab] = sm
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([l for l, _, _ in arms], fontsize=fz["tiny"]); ax.invert_yaxis()
    for t, (_, col, _) in zip(ax.get_yticklabels(), arms):
        t.set_color(col)
    ax.set_xlabel("camera release → control output over warm frames (ms), 45 Hz camera — bars: component medians; label: the chain's median", fontsize=fz["lab"]); ax.grid(axis="x", ls=":", lw=0.5); ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); ax.tick_params(labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Where the latency goes: the schedule's chain is compute;\nthe callback graph's is queueing", fontsize=fz["title"], weight="bold", loc="left")
    return out


def pool(rows, trace, flt, speeds=None):
    v = [r for r in rows if r["trace"] == trace and all(abs(r[k] - val) < 1e-6 if isinstance(val, float) else r[k] == val for k, val in flt.items()) and (speeds is None or r["cruise"] in speeds)]
    return one_per_seed(v)


SHORT = {"course A · density 0.30 · people 2.4 m": "course A, d 0.30, 2.4 m people", "course A · density 0.20": "course A, d 0.20", "course A · density 0.40": "course A, d 0.40",
         "people crossing the aisle": "people crossing", "people walking 1.5 m/s (cadence only)": "people walking 1.5 m/s †", "course C (cadence only)": "course C †",
         "heavier stack (5 networks), 90 Hz camera": "heavier stack, 90 Hz", "heavier stack (5 networks), 45 Hz camera": "heavier stack, 45 Hz", "90 Hz camera": "90 Hz camera",
         "fastest camera each sustains: 120 Hz vs 90 Hz": "120 Hz vs 90 Hz camera", "goal refreshed at the deployment's rate": "goal at deployment rate", "baseline with QoS depth 1": "baseline + QoS 1",
         "gain calibrated per rate, 0.5 / f (cadence only)": "gain per rate †", "retrained guidance net (cadence only)": "retrained net †", "cadence only, no latency, course A": "cadence only, course A †",
         "course B (cadence only)": "course B †", "people 1.7 m, fixed gain (cadence only)": "people 1.7 m †", "people 1.7 m, gain per rate (cadence only)": "people 1.7 m, gain per rate †", "people 1.7 m, cadence + latency": "people 1.7 m, with latency",
         "ROS 2 on all 8 cores (two YOLO nodes)": "ROS 2 two-YOLO, 8 cores", "ROS 2 on all 8 cores, 90 Hz camera": "ROS 2 two-YOLO, 90 Hz", "ROS 2 multi-threaded executor, 8 threads": "ROS 2 MT executor",
         "ROS 2 hand-pinned instead of vanilla": "vs ROS 2 hand-pinned", "all conditions pooled": "all conditions pooled", "all conditions (summary)": "summary"}


A_TRACE, R4_TRACE, R4TM_TRACE = "xpu_a_cpsat_hard.csv", "ros_vanilla445.csv", "ros_vanilla4tm45.csv"


def _F(trace=None, lat=None, hold=None, gain=0.0055, course="a", dens=0.30, walk=0.0, cross=0, ph=2.4):
    """a row filter for one arm: latency and hold default to the registry's launch values for `trace`."""
    lat = FC.lat_ms(trace) if lat is None and trace else (lat if lat is not None else 0.0)
    hold = FC.hold_ms(trace) if hold is None and trace else (hold if hold is not None else 0.0)
    return dict(lat=lat, hold=hold, gain=gain, course=course, dens=dens, walk=walk, cross=cross, ph=ph)


_A, _R4, _R4TM = A_TRACE, R4_TRACE, R4TM_TRACE
_DEP = FC.GOAL_HOLD_DEPLOYMENT_MS
CONDS = [  # label, xpu trace, xpu filter, ros trace, ros filter, campaigns, baseline label
    ("course A · density 0.30 · people 2.4 m", _A, _F(_A), _R4, _F(_R4), ("campaign_percep", "campaign_seeds24"), "vanilla"),
    ("course A · density 0.20", _A, _F(_A, dens=0.20), _R4TM, _F(_R4TM, dens=0.20), ("campaign_percep",), "vanilla, control timer"),
    ("course A · density 0.40", _A, _F(_A, dens=0.40), _R4TM, _F(_R4TM, dens=0.40), ("campaign_percep",), "vanilla, control timer"),
    ("people crossing the aisle", _A, _F(_A, cross=1), _R4, _F(_R4, cross=1), ("campaign_cross",), "vanilla"),
    ("people walking 1.5 m/s (cadence only)", _A, _F(lat=0.0, walk=1.5), _R4, _F(lat=0.0, walk=1.5), ("campaign_walk",), "vanilla"),
    ("course C (cadence only)", _A, _F(lat=0.0, course="c"), _R4, _F(lat=0.0, course="c"), ("campaign_courseC",), "vanilla"),
    ("heavier stack (5 networks), 90 Hz camera", "xpu_b5_cpsat.csv", _F("xpu_b5_cpsat.csv"), "ros_rvanilla490.csv", _F("ros_rvanilla490.csv"), ("campaign_rich",), "vanilla + stack"),
    ("heavier stack (5 networks), 45 Hz camera", "xpu_b5_cpsat.csv", _F("xpu_b5_cpsat.csv"), "ros_rvanilla445.csv", _F("ros_rvanilla445.csv"), ("campaign_rich",), "vanilla + stack"),
    ("90 Hz camera", "xpu_a90_cpsat.csv", _F("xpu_a90_cpsat.csv"), "ros_vanilla4tm90.csv", _F("ros_vanilla4tm90.csv"), ("campaign_percep",), "vanilla, control timer"),
    ("fastest camera each sustains: 120 Hz vs 90 Hz", _A, _F(lat=FC.arm_for(_A, 59.9).csv_lat, hold=FC.arm_for(_A, 59.9).csv_hold), "ros_vanilla4tm90.csv", _F("ros_vanilla4tm90.csv"), ("campaign_percep",), "vanilla, control timer, 90 Hz"),
    ("goal refreshed at the deployment's rate", _A, _F(_A, hold=_DEP[_A]), "ros_p345.csv", _F("ros_p345.csv", hold=_DEP["ros_p345.csv"]), ("campaign_percep",), "hand-pinned"),
    ("baseline with QoS depth 1", _A, _F(_A), "ros_vanilla4_q145.csv", _F("ros_vanilla4_q145.csv"), ("campaign_percep", "campaign_seeds24", "campaign_qos1"), "vanilla, QoS 1"),
    ("gain calibrated per rate, 0.5 / f (cadence only)", _A, _F(lat=0.0, gain=0.00521), _R4, _F(lat=0.0, gain=0.01277), ("campaign_tallcal",), "vanilla, its own gain"),
    ("retrained guidance net (cadence only)", _A, _F(lat=0.0), _R4, _F(lat=0.0), ("campaign_tallnet",), "vanilla"),
    ("cadence only, no latency, course A", _A, _F(lat=0.0), _R4, _F(lat=0.0), ("campaign_env",), "vanilla"),
    ("course B (cadence only)", _A, _F(lat=0.0, course="b"), _R4, _F(lat=0.0, course="b"), ("campaign_env",), "vanilla"),
    ("people 1.7 m, fixed gain (cadence only)", _A, _F(lat=0.0), _R4, _F(lat=0.0), ("campaign_v2",), "vanilla"),
    ("people 1.7 m, gain per rate (cadence only)", _A, _F(lat=0.0, gain=0.0052), _R4, _F(lat=0.0, gain=0.01277), ("campaign_v2",), "vanilla, its own gain"),
    ("people 1.7 m, cadence + latency", _A, _F(_A, ph=1.7), _R4, _F(_R4, ph=1.7), ("campaign_break",), "vanilla"),
    ("ROS 2 on all 8 cores (two YOLO nodes)", _A, _F(_A), "ros_vanilla4x245.csv", _F("ros_vanilla4x245.csv"), ("campaign_percep",), f"two YOLO nodes, 8 cores ({FC.lat_label('ros_vanilla4x245.csv')})"),
    ("ROS 2 on all 8 cores, 90 Hz camera", "xpu_a90_cpsat.csv", _F("xpu_a90_cpsat.csv"), "ros_vanilla4x290.csv", _F("ros_vanilla4x290.csv"), ("campaign_percep",), f"two YOLO nodes, 90 Hz ({FC.lat_label('ros_vanilla4x290.csv')})"),
    ("ROS 2 multi-threaded executor, 8 threads", _A, _F(_A), "ros_multi45.csv", _F("ros_multi45.csv"), ("campaign_percep",), f"multi-threaded executor ({FC.lat_label('ros_multi45.csv')})"),
    ("ROS 2 hand-pinned instead of vanilla", _A, _F(_A), "ros_p345.csv", _F("ros_p345.csv"), ("campaign_percep", "campaign_seeds24"), f"hand-pinned ({FC.lat_label('ros_p345.csv')})")]
FLIGHT_KEY = ("camp", "trace", "seed", "cruise", "lat", "hold", "gain", "course", "dens", "walk", "cross", "ph")


def flight_id(r):
    return tuple(r.get(k) for k in FLIGHT_KEY)


def forest_rows(rows, conds=None):
    """one row per condition both arms flew: (label, baseline label, k/n for each arm, Newcombe delta and interval, the
    speeds pooled) plus, in parallel, the identities of the flights each row counted."""
    res, flights = [], []
    for lab, tx, fx, tr, fr, camps, rlab in (conds or CONDS):
        sub = [r for r in rows if r["camp"] in camps]
        vx = pool(sub, tx, fx); vr = pool(sub, tr, fr)
        speeds = sorted({r["cruise"] for r in vx} & {r["cruise"] for r in vr} & {1.0, 1.2, 1.4}) or sorted({r["cruise"] for r in vx} & {r["cruise"] for r in vr})
        vx = [r for r in vx if r["cruise"] in speeds]; vr = [r for r in vr if r["cruise"] in speeds]
        if len(vx) < 12 or len(vr) < 12:
            continue
        kx, kr = sum(r["outcome"] == "success" for r in vx), sum(r["outcome"] == "success" for r in vr)
        d, lo, hi = newcombe(kx, len(vx), kr, len(vr)); res.append((lab, rlab, kx, len(vx), kr, len(vr), d, lo, hi, speeds))
        flights.append({"xpu": {flight_id(r): r["outcome"] == "success" for r in vx}, "ros": {flight_id(r): r["outcome"] == "success" for r in vr}})
    order = sorted(range(len(res)), key=lambda i: -res[i][6])
    return [res[i] for i in order], [flights[i] for i in order]


def forest_summary(res, flights_by_cond, seed=0, n_boot=10000):
    """the across-conditions summary: the mean of the per-condition deltas with a percentile bootstrap over conditions
    (conditions, not flights, are the units — many conditions reuse the same flights), and each flight counted once."""
    deltas = np.array([r[6] for r in res], float); rng = np.random.RandomState(seed)
    boots = np.array([deltas[rng.randint(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)]) if len(deltas) > 1 else deltas
    distinct = {}
    for arm in ("xpu", "ros"):
        seen = {}
        for f in flights_by_cond:
            for fid, ok in f[arm].items():
                seen.setdefault(fid, ok)
        distinct[arm] = [int(sum(seen.values())), len(seen)]
    KX = sum(r[2] for r in res); NX = sum(r[3] for r in res); KR = sum(r[4] for r in res); NR = sum(r[5] for r in res)
    return {"method": "mean of the per-condition deltas; 95 % percentile bootstrap over conditions; flights counted once, at the first condition that includes them",
            "mean_delta": round(float(deltas.mean()), 4) if len(deltas) else None, "ci": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if len(deltas) else None,
            "n_conditions": len(res), "distinct_flights": distinct, "summed_counts": [KX, NX, KR, NR], "bootstrap": {"seed": seed, "n": n_boot}}


def draw_forest(ax, rows, fz, compact=False):
    res, flights = forest_rows(rows); summ = forest_summary(res, flights)
    y = np.arange(len(res) + 1); out = {}
    for i, (lab, rlab, kx, nx, kr, nr, d, lo, hi, speeds) in enumerate(res):
        col = C_XPU if lo > 0 else (C_ROS if hi < 0 else "0.45")
        ax.plot([lo, hi], [i, i], color=col, lw=2.0); ax.plot(d, i, "o", color=col, ms=7, mec="white", mew=0.8)
        ax.text(1.02, i, f"{kx}/{nx} vs {kr}/{nr}", transform=ax.get_yaxis_transform(), va="center", fontsize=fz["tiny"] * (0.9 if compact else 1.0), color=INK)
        out[lab] = {"ros_arm": rlab, "xpu": [kx, nx], "ros": [kr, nr], "delta": round(d, 3), "ci": [round(lo, 3), round(hi, 3)], "speeds": speeds}
    i = len(res); m, (lo, hi) = summ["mean_delta"], summ["ci"]; dx, dr = summ["distinct_flights"]["xpu"], summ["distinct_flights"]["ros"]
    col = C_XPU if lo > 0 else (C_ROS if hi < 0 else "0.45")
    ax.plot([lo, hi], [i, i], color=col, lw=3.0); ax.plot(m, i, "D", color=col, ms=9, mec="white", mew=0.8)
    ax.text(1.02, i, f"{dx[0]}/{dx[1]} vs {dr[0]}/{dr[1]} distinct flights", transform=ax.get_yaxis_transform(), va="center", fontsize=fz["tiny"] * (0.9 if compact else 1.0), color=INK, weight="bold")
    out["all conditions (summary)"] = summ
    ax.axhline(len(res) - 0.5, color="0.6", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    labels = [SHORT.get(lab, lab) if compact else f"{lab}\n(vs ROS 2 {rlab})" for lab, rlab, *_ in res] + ["summary: mean Δ over the conditions"]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fz["tiny"] * (0.95 if compact else 1.0)); ax.invert_yaxis()
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.set_xlabel("Δ courses completed, XPU-RT − ROS 2 (95 % interval); right: completed/flown"
                  + ("\n† cadence replayed without the latency; 1.0–1.4 m/s pooled, one flight per seed" if compact else "\n1.0–1.4 m/s pooled where both arms flew them, one flight per seed, the baseline named per row")
                  + "\nsummary row: mean of the per-condition Δ, 95 % bootstrap over conditions; its counts are distinct flights", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="x", ls=":", lw=0.5)
    ax.set_xlim(-0.55, 0.65)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n_ahead = sum(1 for r in res if r[6] > 0); n_sig = sum(1 for r in res if r[7] > 0); n_neg = sum(1 for r in res if r[8] < 0)
    ax.set_title(f"Every condition flown by both arms: XPU-RT ahead in {n_ahead} of {len(res)}\n(clear of zero in {n_sig}, behind in {n_neg}); mean Δ = {m:+.2f} [{lo:+.2f}, {hi:+.2f}], bootstrap over conditions", fontsize=fz["title"], weight="bold", loc="left")
    return out


def draw_pairs(ax, rows, fz):
    speeds = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    A = SF.cells_by(rows, A_TRACE, FC.lat_ms(A_TRACE), 0.0); B = SF.cells_by(rows, R4_TRACE, FC.lat_ms(R4_TRACE), 0.0)
    tot = collections.Counter(); per = {}
    for c in speeds:
        a = {r["seed"]: r for r in A.get(c, [])}; b = {r["seed"]: r for r in B.get(c, [])}; cnt = collections.Counter()
        for s in set(a) & set(b):
            ga = 4 if a[s]["outcome"] == "success" else a[s]["gates"]; gb = 4 if b[s]["outcome"] == "success" else b[s]["gates"]
            cnt["xpu" if ga > gb else "ros" if gb > ga else "tie"] += 1
        per[c] = cnt; tot.update(cnt)
    x = np.arange(len(speeds)); bottom = np.zeros(len(speeds))
    for key, col, lab in (("xpu", C_XPU, "XPU-RT gets further"), ("tie", "#cfcfcf", "same gate"), ("ros", C_ROS, "ROS 2 gets further")):
        vals = np.array([per[c][key] for c in speeds]); ax.bar(x, vals, bottom=bottom, color=col, width=0.66, edgecolor="white", label=lab)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=fz["tiny"], color="white" if key != "tie" else INK, weight="bold")
        bottom += vals
    ax.set_ylim(0, bottom.max() * 1.3); ax.set_xticks(x); ax.set_xticklabels([f"{c:.1f}" for c in speeds], fontsize=fz["tick"]); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("seed pairs", fontsize=fz["lab"])
    ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper right"); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="y", ls=":", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Same seed, both runtimes: XPU-RT further on {tot['xpu']},\nROS 2 on {tot['ros']}, tie on {tot['tie']} seed–speed pairs", fontsize=fz["title"], weight="bold", loc="left")
    return dict(tot)


def draw_progress(ax, fz):
    recs = SF.load_records(); arms = [a for a in ("xpu_cpsat", "xpu_greedy", "ros_vanilla4", "ros_p3", "ros_vanilla4x2", "ros_multi") if any(k[0] == a for k in recs)]
    y0 = min(r["poses"][0, 1] for v in recs.values() for r in v); rng = np.random.RandomState(1); out = {}
    for a in arms:
        lab, col, _ = SF.ARMS[a]; xs, ms, lo, hi = [], [], [], []
        for c in sorted({cc for (aa, cc) in recs if aa == a}):
            v = recs[(a, c)]
            if len(v) < 12:
                continue
            p = np.array([SF.along(r, y0)[0] for r in v]); boots = [rng.choice(p, len(p)).mean() for _ in range(1000)]
            xs.append(c); ms.append(p.mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5)); out[f"{a}_c{c}"] = round(float(p.mean()), 3)
        if xs:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0); ax.plot(xs, ms, "-o", color=col, lw=2.2, ms=5, label=lab)
    ax.set_ylim(0, 1.0); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("fraction of the course covered", fontsize=fz["lab"]); ax.grid(ls=":", lw=0.5); ax.legend(fontsize=fz["tiny"], frameon=False, loc="lower left"); ax.tick_params(labelsize=fz["tick"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"How far each arm gets, graded (mean, 95 % bootstrap):\nfurthest apart where a {FC.lat_label('ros_vanilla445.csv')} decision delay matters;\nthe arms converge once perception limits everyone", fontsize=fz["title"], weight="bold", loc="left")
    return out


def draw_solvers(ax, fz):
    cp = V.xpu_rate_points("cpsat"); gr = V.xpu_rate_points("greedy"); hz = sorted(set(cp) & set(gr)); x = np.arange(len(hz)); out = {}
    for pts, dx, col, lab in ((cp, -0.19, C_XPU, "CP-SAT"), (gr, 0.19, C_XPU2, "greedy")):
        fr = [100.0 * pts[h]["late"] / max(1, pts[h]["checked"]) for h in hz]
        ax.bar(x + dx, fr, width=0.36, color=col, edgecolor="k", lw=0.4, label=lab)
        for i, h in enumerate(hz):
            ax.annotate(f"{pts[h]['lat']:.0f} ms", (i + dx, fr[i]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=fz["tiny"] * 0.85, color=col, rotation=90 if fr[i] < 20 else 0, va="bottom")
            out[f"{lab}@{h}"] = {"late_pct": round(fr[i], 1), "lat_ms": round(pts[h]["lat"], 1)}
    ax.set_xticks(x); ax.set_xticklabels([f"{h} Hz" for h in hz], fontsize=fz["tick"]); ax.set_ylabel("YOLO frames late (%)", fontsize=fz["lab"]); ax.set_ylim(0, 135); ax.set_yticks([0, 50, 100])
    ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper left"); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="y", ls=":", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    gl = [h for h in hz if gr[h]["late"] > 0]; g_from = f"above {max(h for h in hz if h < min(gl)):.0f} Hz" if gl and any(h < min(gl) for h in hz) else "at every rate"
    ax.set_title(f"Same spec, two solvers: CP-SAT holds every window at every\ncamera rate; greedy loses frames {g_from} (labels:\ncamera→control) — in flight greedy clears no gate", fontsize=fz["title"], weight="bold", loc="left")
    return out


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", default=os.path.join(RES, "campaign_v2/display_same/xpu_s1005_figdata")); ap.add_argument("--ros-dir", default=os.path.join(RES, "campaign_v2/display_same/ros_s1005_figdata"))
    ap.add_argument("--width-in", type=float, default=26.0); ap.add_argument("--min-print-pt", type=float, default=3.6); ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(RES, "refined", "warehouse_showdown_atlas"))
    a = ap.parse_args()
    base = a.min_print_pt * a.width_in / 7.1
    fz = dict(tiny=base * 0.9, tick=base, lab=base * 1.05, leg=base * 0.95, title=base * 1.2, badge=base * 1.1, head=base * 1.35)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    rows = all_flights(); rows_c = [r for r in rows if r["camp"].startswith("campaign_")]
    fig = plt.figure(figsize=(a.width_in, a.width_in * 1.08))
    fig.text(0.04, 0.994, "One workload, two runtimes, one board — the same kernels scheduled by XPU-RT's solvers against ROS 2 as one writes it: what was run, what the K1 delivered, where the flights ended",
             fontsize=fz["head"], weight="bold", va="top", ha="left")
    outer = fig.add_gridspec(7, 1, height_ratios=[3.6, 1.1, 3.4, 1.2, 3.6, 1.0, 3.4], hspace=0.0, left=0.06, right=0.985, top=0.955, bottom=0.03)
    # row 0: A census | B top-down
    r0 = outer[0].subgridspec(1, 2, width_ratios=[1.35, 1.9], wspace=0.30); axA = fig.add_subplot(r0[0]); axB = fig.add_subplot(r0[1])
    MA = draw_census(axA, rows, fz); MB = draw_topdown(axB, a.xpu_dir, a.ros_dir, fz)
    # row 2: C frequency | D ladder | E waterfall
    r2 = outer[2].subgridspec(1, 3, width_ratios=[1.2, 1.25, 1.25], wspace=0.30)
    cg = r2[0].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=0.16); axC1 = fig.add_subplot(cg[0]); axC2 = fig.add_subplot(cg[1], sharex=axC1)
    dg = r2[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.16); axD1 = fig.add_subplot(dg[0]); axD2 = fig.add_subplot(dg[1], sharex=axD1)
    axE = fig.add_subplot(r2[2])
    MC = V.draw_F(axC1, axC2, V.BOARD_ARMS, fz)
    cp_pts = MC.get("XPU-RT · CP-SAT", {}); on_time_to = max((h for h, v in cp_pts.items() if v.get("late") == 0), default=None)
    axC1.set_title(f"Frequency on the K1: the solved table keeps every frame\non time to {on_time_to:.0f} Hz; every ROS 2 layout plateaus," if on_time_to else "Frequency on the K1: latency and frames delivered;", fontsize=fz["title"], weight="bold", loc="left")
    MD = draw_ladder(axD1, axD2, fz); ME = draw_waterfall(axE, fz)
    # rows 4+6: F forest (tall, left) | G pairs, H progress | I solvers, J load, K feedback
    r46 = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.15, 1.15], height_ratios=[3.6, 3.4], hspace=0.30, wspace=0.30, left=0.06, right=0.985, top=outer[4].get_position(fig).y1, bottom=outer[6].get_position(fig).y0)
    axF = fig.add_subplot(r46[:, 0]); axG = fig.add_subplot(r46[0, 1]); axH = fig.add_subplot(r46[0, 2]); axI = fig.add_subplot(r46[1, 1])
    jg = r46[1, 2].subgridspec(4, 1, hspace=0.32); axJ = [fig.add_subplot(jg[i]) for i in range(4)]
    MF = draw_forest(axF, rows, fz); MG = draw_pairs(axG, rows_c, fz); MH = draw_progress(axH, fz); MI = draw_solvers(axI, fz)
    cells = V.flight_cells(rows_c, "a", 0.30, 0.0055); cells_rich = {k: v for k, v in cells.items() if k[0].startswith(("xpu_b5", "ros_rvanilla"))}
    cells_rich, _ = V.equalise(cells_rich, [t for t in ("xpu_b5_cpsat.csv", "xpu_b5_greedy.csv", "ros_rvanilla445.csv", "ros_rvanilla490.csv") if any(k[0] == t for k in cells_rich)])
    MJ = V.draw_H(axJ, cells_rich, fz); axJ[0].set_title("Added load: two more networks on the same eight harts —\nCP-SAT > greedy > ROS 2 on the board and in flight", fontsize=fz["title"], weight="bold", loc="left")
    for ax_, s_ in ((axA, "A"), (axB, "B"), (axC1, "C"), (axD1, "D"), (axE, "E"), (axF, "F"), (axG, "G"), (axH, "H"), (axI, "I"), (axJ[0], "J")):
        sec(ax_, s_, fz, dx=-8, dy=30)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight"); print("wrote", a.out + ".png")
    json.dump({"figure": a.out + ".png", "written": datetime.datetime.now().isoformat(timespec="seconds"), "A": MA, "B": MB, "C": {k: {str(h): {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in p.items()} for h, p in pts.items()} for k, pts in MC.items()},
               "D": MD, "E": ME, "F": MF, "G": MG, "H": MH, "I": MI, "J": MJ,
               **FC.sidecar_common("showdown_atlas", V.LOADED_CSVS + LOADED_EXTRA + WATERFALL_INPUTS + [os.path.join(RES, "ros_traced", "summary.csv")])}, open(a.out + "_metrics.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
