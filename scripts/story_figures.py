#!/usr/bin/env python3
"""Stories in the flight campaigns and the board traces, one figure each, every number read at render time.

  crash_position   where along the aisle each arm's flights end, per cruise speed (records)
  course_progress  how much of the course each arm covers, mean and 95 % bootstrap band against cruise (records)
  rate_speed_map   camera rate x cruise speed: success per runtime arm, one flight per seed (campaign CSVs)
  ros_ladder       what hand-tuning buys ROS 2 and where it stops: board latency / goals per layout at 45 Hz and
                   the same layouts' flights (summary.csv + campaign CSVs)
  seed_pairs       the same seed under both runtimes: paired outcomes per (seed, cruise) (campaign CSVs)
  latency_waterfall camera->control on the board split into queue wait / YOLO / nav / control per arm (traces)

    scripts/story_figures.py [crash_position course_progress ...] [--dpi 220]
Outputs results/codesign_feedback/refined/<name>.{png,pdf} and <name>_metrics.json.
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, statistics, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback"); OUT = os.path.join(RES, "refined")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import showdown_v3_figure as V   # noqa: E402
import figure_constants as FC   # noqa: E402
from make_measured_gantt_pair import read_trace   # noqa: E402

INK = "#22242a"


def _lab(base, trace, hz=True):
    """'<base> (<latency>[, <control rate>])' from the registry — no board number is typed here."""
    parts = [FC.lat_label(trace)] + ([FC.ctrl_hz_label(trace)] if hz else [])
    return f"{base} ({', '.join(parts)})"


ARMS = {  # record-dir prefix -> (label, colour, trace file)
    "xpu_cpsat": (_lab("XPU-RT · CP-SAT", "xpu_a_cpsat_hard.csv"), V.C_XPU, "xpu_a_cpsat_hard.csv"),
    "xpu_greedy": (_lab("XPU-RT · greedy", "xpu_a_greedy.csv", hz=False), V.C_XPU2, "xpu_a_greedy.csv"),
    "ros_vanilla4": (_lab("ROS 2 vanilla", "ros_vanilla445.csv"), V.C_ROS, "ros_vanilla445.csv"),
    "ros_vanilla4tm": (_lab("ROS 2 vanilla, control timer", "ros_vanilla4tm45.csv"), "#8e24aa", "ros_vanilla4tm45.csv"),
    "ros_p3": (_lab("ROS 2 hand-tuned, pinned", "ros_p345.csv"), V.C_P3, "ros_p345.csv"),
    "ros_p3_q1": (_lab("ROS 2 hand-tuned + QoS 1", "ros_p3_q145.csv", hz=False), "#26a69a", "ros_p3_q145.csv"),
    "ros_vanilla4x2": (_lab("ROS 2 two YOLO nodes, 8 cores", "ros_vanilla4x245.csv", hz=False), V.C_ROS8, "ros_vanilla4x245.csv"),
    "ros_multi": (_lab("ROS 2 multi-threaded executor", "ros_multi45.csv", hz=False), V.C_MULTI, "ros_multi45.csv"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
INPUTS: list[str] = []   # files a figure read, for the sidecar's hashes (records dirs are summarised by their campaign CSV)


def save(fig, name, metrics, dpi, inputs=None):
    p = os.path.join(OUT, name)
    fig.savefig(p + ".png", dpi=dpi, bbox_inches="tight"); fig.savefig(p + ".pdf", bbox_inches="tight")
    side = dict(metrics); side.update(FC.sidecar_common(f"story_figures:{name}", (inputs if inputs is not None else V.LOADED_CSVS + INPUTS)))
    json.dump(side, open(p + "_metrics.json", "w"), indent=1); print("wrote", p + ".png")


# ------------------------------------------------------------------------------------------ records
def load_records(course="a", dens=0.30, gain=0.0055, latency=True, camps=("campaign_percep", "campaign_seeds24")):
    """{(arm, cruise): [record dicts]} from the named campaigns (one record per seed and cell; a cell flown on a second
    seed set carries an `_s<seed0>` suffix and pools with the first)."""
    out = collections.defaultdict(dict)
    for camp in camps:
        if os.path.exists(os.path.join(RES, camp, "campaign.csv")):
            INPUTS.append(os.path.join(RES, camp, "campaign.csv"))
        for d in sorted(glob.glob(os.path.join(RES, camp, "records", "*"))):
            m = re.match(r"(.+?)_lat([\d.]+)_h([\d.]+)_([a-z])_d([\d.]+)_w([\d.]+)_g([\d.]+)_c([\d.]+)(?:_s\d+)?$", os.path.basename(d))
            if not m:
                continue
            arm, lat, hold, c, dn, walk, g, cru = m.groups()
            if c != course or abs(float(dn) - dens) > 1e-9 or abs(float(g) - gain) > 2e-5 or float(walk) != 0 or float(hold) != 0:
                continue
            if latency and float(lat) <= 0:
                continue
            for f in glob.glob(os.path.join(d, "ep*.npz")):
                z = np.load(f, allow_pickle=True)
                if int(z["walk_cross"]) if "walk_cross" in z.files else 0:
                    continue
                key = (arm, round(float(cru), 2)); seed = int(z["seed"])
                out[key].setdefault(seed, dict(poses=z["poses"], outcome=str(z["outcome"]), gates=int(z["gates_passed"]), gates_world=z["gates_world"], seed=seed))
    return {k: list(v.values()) for k, v in out.items()}


def along(rec, y0):
    """course fraction covered: furthest along-aisle position between the start line and the last gate."""
    g = rec["gates_world"]; return float(np.clip((rec["poses"][:, 1].max() - y0) / (g[-1, 1] - y0), 0, 1)), float(rec["poses"][-1, 1] - y0)


def fig_crash_position(dpi):
    recs = load_records(); arms = [a for a in ("xpu_cpsat", "xpu_greedy", "ros_vanilla4", "ros_p3", "ros_vanilla4x2") if any(k[0] == a for k in recs)]
    speeds = sorted({c for (a, c) in recs if a == "xpu_cpsat"})
    y0 = min(r["poses"][0, 1] for v in recs.values() for r in v); gw = next(iter(recs.values()))[0]["gates_world"]; gy = gw[:, 1] - y0
    fig, axes = plt.subplots(1, len(speeds), figsize=(2.6 * len(speeds) + 1.5, 4.6), sharey=True)
    metrics = {}
    for ax, c in zip(np.atleast_1d(axes), speeds):
        for i, a in enumerate(arms):
            v = recs.get((a, c), [])
            if not v:
                continue
            lab, col, _ = ARMS[a]
            ends = [r["poses"][-1, 1] - y0 for r in v]; succ = [r["outcome"] == "success" for r in v]
            x = i + (np.random.RandomState(0).rand(len(ends)) - 0.5) * 0.5
            ax.scatter(x, ends, s=[46 if s_ else 26 for s_ in succ], marker="o", c=[col] * len(ends), edgecolors="white", lw=0.6, alpha=0.9, zorder=3)
            ax.scatter([x[j] for j in range(len(x)) if not succ[j]], [ends[j] for j in range(len(x)) if not succ[j]], marker="x", s=30, c="k", lw=0.8, zorder=4)
            med = statistics.median(ends); ax.plot([i - 0.3, i + 0.3], [med, med], color=col, lw=2.4, zorder=5)
            metrics[f"{a}_c{c}"] = {"n": len(v), "median_along_m": round(med, 2), "completed": int(sum(succ))}
        for gi, y in enumerate(gy):
            ax.axhline(y, color="#f2a900", lw=1.2, ls=(0, (4, 3)), zorder=1); ax.text(len(arms) - 0.55, y, f"G{gi + 1}", color="#b07a00", fontsize=8, va="bottom", ha="right")
        short = {"xpu_cpsat": "XPU-RT\nCP-SAT", "xpu_greedy": "XPU-RT\ngreedy", "ros_vanilla4": "ROS 2\nvanilla", "ros_p3": "ROS 2\npinned", "ros_vanilla4x2": "ROS 2\n8-core", "ros_multi": "ROS 2\nMT exec."}
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([short.get(a, a) for a in arms], fontsize=8)
        ax.set_title(f"{c:.1f} m/s", fontsize=10, weight="bold"); ax.grid(axis="y", ls=":", lw=0.5); ax.set_xlim(-0.6, len(arms) - 0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("where the flight ends: distance along the aisle (m)")
    handles = [Line2D([], [], marker="o", ls="", color="0.4", label="flight end (larger = course completed)"), Line2D([], [], marker="x", ls="", color="k", label="collision"), Line2D([], [], color="0.4", lw=2.4, label="median"), Line2D([], [], color="#f2a900", ls=(0, (4, 3)), label="gate line")]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Where each arm's flights end — course A, obstacle density 0.30, people 2.4 m, each arm at its K1 cadence and camera→control latency, one flight per seed",
                 fontsize=11, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "crash_position", metrics, dpi)


def fig_course_progress(dpi):
    recs = load_records(); arms = [a for a in ("xpu_cpsat", "xpu_greedy", "ros_vanilla4", "ros_p3", "ros_vanilla4x2", "ros_multi") if any(k[0] == a for k in recs)]
    y0 = min(r["poses"][0, 1] for v in recs.values() for r in v); rng = np.random.RandomState(1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); metrics = {}
    for a in arms:
        lab, col, _ = ARMS[a]; xs, ms, lo, hi = [], [], [], []
        for c in sorted({cc for (aa, cc) in recs if aa == a}):
            v = recs[(a, c)]
            if len(v) < 12:
                continue
            p = np.array([along(r, y0)[0] for r in v]); boots = [rng.choice(p, len(p)).mean() for _ in range(2000)]
            xs.append(c); ms.append(p.mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
            metrics[f"{a}_c{c}"] = {"n": len(v), "course_fraction": round(float(p.mean()), 3), "ci": [round(float(lo[-1]), 3), round(float(hi[-1]), 3)]}
        if xs:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0); ax.plot(xs, ms, "-o", color=col, lw=2.2, ms=6, label=lab)
    ax.set_ylim(0, 1.02); ax.set_xlabel("cruise speed (m/s)"); ax.set_ylabel("fraction of the course covered (mean, 95 % bootstrap)")
    ax.grid(ls=":", lw=0.5); ax.legend(fontsize=8, frameon=False, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("How far each arm gets, graded: course fraction against cruise speed\n(course A, density 0.30, people 2.4 m, cadence and latency replayed from the K1, one flight per seed)", fontsize=10.5, weight="bold", loc="left")
    fig.tight_layout(); save(fig, "course_progress", metrics, dpi)


# --------------------------------------------------------------------------------------- campaign CSVs
def _rate_arm(label, trace, lat=None):
    a = FC.arm_for(trace, lat); return (label, a.cam_hz, a.trace, a.csv_lat, a.csv_hold)


RATE_ARMS = [_rate_arm("XPU-RT · CP-SAT", "xpu_a_cpsat_hard.csv"), _rate_arm("XPU-RT · CP-SAT", "xpu_a90_cpsat.csv"), _rate_arm("XPU-RT · CP-SAT", "xpu_a_cpsat_hard.csv", 59.9),
             _rate_arm("XPU-RT · greedy", "xpu_a_greedy.csv"), _rate_arm("XPU-RT · greedy", "xpu_a90_greedy.csv"),
             _rate_arm("ROS 2 vanilla (4-hart YOLO)", "ros_vanilla445.csv"), _rate_arm("ROS 2 vanilla, control timer", "ros_vanilla4tm45.csv"), _rate_arm("ROS 2 vanilla, control timer", "ros_vanilla4tm90.csv"),
             _rate_arm("ROS 2 hand-tuned (pinned)", "ros_p345.csv"), _rate_arm("ROS 2 hand-tuned (pinned)", "ros_p390.csv"),
             _rate_arm("ROS 2 two YOLO nodes, 8 cores", "ros_vanilla4x245.csv"), _rate_arm("ROS 2 two YOLO nodes, 8 cores", "ros_vanilla4x290.csv"),
             _rate_arm("ROS 2 multi-threaded executor", "ros_multi45.csv")]


def cells_by(rows, trace, lat, hold, course="a", dens=0.30, gain=0.0055, person_h=2.4):
    out = collections.defaultdict(dict)
    for r in rows:
        if r["trace"] == trace and abs(r["lat"] - lat) < 0.15 and abs(r["hold"] - hold) < 0.15 and r["course"] == course and r["dens"] == dens and abs(r["gain"] - gain) < 2e-5 and r["walk"] == 0 and r["cross"] == 0 and r.get("ph", 2.4) == person_h:
            out[r["cruise"]].setdefault(r["seed"], r)               # one flight per seed
    return {c: list(v.values()) for c, v in out.items()}


def fig_rate_speed_map(dpi):
    rows = V.load_campaigns(); speeds = [1.0, 1.2, 1.4, 1.6, 1.8]
    present = [(lab, hz, tr, lat, hold) for lab, hz, tr, lat, hold in RATE_ARMS if cells_by(rows, tr, lat, hold)]
    M = np.full((len(present), len(speeds)), np.nan); T = [[""] * len(speeds) for _ in present]; metrics = {}
    for i, (lab, hz, tr, lat, hold) in enumerate(present):
        cb = cells_by(rows, tr, lat, hold)
        for j, c in enumerate(speeds):
            v = cb.get(c, [])
            if len(v) >= 12:
                k = sum(x["outcome"] == "success" for x in v); M[i, j] = k / len(v); T[i][j] = f"{k}/{len(v)}"; metrics[f"{lab}@{hz}Hz_c{c}"] = [k, len(v)]
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(present) + 1.8))
    cmap = plt.cm.YlGn.copy(); cmap.set_bad("#eeeeee"); ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    for i in range(len(present)):
        for j in range(len(speeds)):
            ax.text(j, i, T[i][j] or "—", ha="center", va="center", fontsize=8.5, color=INK if (np.isnan(M[i, j]) or M[i, j] < 0.3) else "white", weight="bold")
    ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f"{c:.1f} m/s" for c in speeds]); ax.set_yticks(range(len(present)))
    ax.set_yticklabels([f"{lab} · {hz} Hz camera" for lab, hz, *_ in present], fontsize=8.5)
    for t, (lab, *_) in zip(ax.get_yticklabels(), present):
        t.set_color(V.C_XPU if "CP-SAT" in lab else V.C_XPU2 if "greedy" in lab else V.C_ROS)
    ax.tick_params(length=0); ax.set_title("Camera rate × cruise speed: courses completed, each arm at its measured cadence, latency and goal rate\n(course A, density 0.30, people 2.4 m, one flight per seed; grey = not flown)", fontsize=10, weight="bold", loc="left")
    fig.tight_layout(); save(fig, "rate_speed_map", metrics, dpi)


def fig_ros_ladder(dpi):
    rows = V.load_campaigns()
    ladder = [(lab, lay, tr, FC.lat_ms(tr)) for lab, lay, tr in (("vanilla\nserial YOLO", "vanilla", "ros_vanilla45.csv"), ("vanilla\n4-hart YOLO", "vanilla4", "ros_vanilla445.csv"), ("+ control\non a timer", "vanilla4tm", "ros_vanilla4tm45.csv"),
              ("+ QoS\ndepth 1", "vanilla4_q1", "ros_vanilla4_q145.csv"), ("two YOLO\nnodes, 8 cores", "vanilla4x2", "ros_vanilla4x245.csv"), ("hand-pinned\n3 processes", "p3", "ros_p345.csv"), ("hand-pinned\n+ QoS 1", "p3_q1", "ros_p3_q145.csv"),
              ("XPU-RT\nCP-SAT", None, "xpu_a_cpsat_hard.csv"))]
    INPUTS.append(os.path.join(RES, "ros_traced", "summary.csv"))
    roll = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RES, "ros_traced", "summary.csv"))):
        m = re.match(r"(45|90)_(.+)_r(\d+)$", r["tag"])
        if m and r.get("gap_mean_ms") and r.get("e2e_goal_med_ms"):
            secs = 20.0; manp = os.path.join(RES, "ros_traced", r["tag"], "manifest.json")
            if os.path.exists(manp):
                secs = float(json.load(open(manp)).get("seconds", 20.0))
            roll[(m.group(2), int(m.group(1)))].append((float(r["e2e_goal_med_ms"]), float(r["gap_mean_ms"]), float(r["n_consumed"]) / (secs - float(r["warmup_ms"]) / 1000)))
    xp = V.xpu_rate_points("cpsat")
    def board(lay, hz):
        if lay is None:
            q = xp.get(hz, {}); return (q.get("lat"), q.get("gap"), q.get("deliv"))
        v = roll.get((lay, hz), []); return (statistics.median([x[0] for x in v]) if v else None, statistics.mean([x[1] for x in v]) if v else None, statistics.mean([x[2] for x in v]) if v else None)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.8), gridspec_kw={"height_ratios": [1, 1.15]}); x = np.arange(len(ladder)); metrics = {}
    cols = [V.C_XPU if lay is None else V.C_ROS for _, lay, _, _ in ladder]
    for ax, ylab, idx, log in ((axes[0][0], "camera → control (ms)", 0, True), (axes[0][1], "goals delivered to control (/s)", 2, False)):
        for hz, dx, hatch in ((45, -0.19, None), (90, 0.19, "///")):
            vals = [board(lay, hz)[idx] for _, lay, _, _ in ladder]
            ax.bar(x + dx, [v or 0 for v in vals], color=cols, width=0.36, edgecolor="k", lw=0.4, hatch=hatch, alpha=0.95 if hz == 45 else 0.6, label=f"{hz} Hz camera")
            for i, v in enumerate(vals):
                ax.annotate(f"{v:.0f}" if v else "", (i + dx, v or 0), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7, weight="bold" if i == len(ladder) - 1 else "normal")
                metrics[f"{ladder[i][0]}@{hz}".replace("\n", " ")] = {"lat": vals[i] if idx == 0 else metrics.get(f"{ladder[i][0]}@{hz}".replace("\n", " "), {}).get("lat"), "goals": vals[i] if idx == 2 else metrics.get(f"{ladder[i][0]}@{hz}".replace("\n", " "), {}).get("goals")}
        if log:
            ax.set_yscale("log"); ax.set_ylim(20, 1200)
        else:
            for hz, y_ in ((45, 45), (90, 90)):
                ax.axhline(y_, color="k", ls="--", lw=0.7, alpha=0.5); ax.text(-0.45, y_ + 1, f"offered {hz}/s", ha="left", va="bottom", fontsize=7)
        ax.set_ylabel(ylab, fontsize=9); ax.set_xticks(x); ax.set_xticklabels([]); ax.grid(axis="y", ls=":", lw=0.5); ax.legend(fontsize=7.5, frameon=False, loc="upper left" if log else "upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_title("On the K1: the hand-tuning ladder, camera → control", fontsize=10.5, weight="bold", loc="left")
    ros_goals = [board(lay, hz)[2] for _, lay, _, _ in ladder for hz in (45, 90) if lay is not None and board(lay, hz)[2]]
    axes[0][1].set_title(f"… and what reaches the control loop: every ROS 2 layout stops at {min(ros_goals):.0f}–{max(ros_goals):.0f}/s", fontsize=10.5, weight="bold", loc="left")
    axf = axes[1][0]; axs = axes[1][1]
    for i, (lab, lay, tr, lat) in enumerate(ladder):
        cb = cells_by(rows, tr, lat, 0.0); tot = [r for c in (1.0, 1.2, 1.4) for r in cb.get(c, [])]
        if tot:
            k = sum(r["outcome"] == "success" for r in tot); axf.bar(i, k / len(tot), color=cols[i], width=0.62, edgecolor="k", lw=0.4)
            axf.annotate(f"{k}/{len(tot)}", (i, k / len(tot)), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8); metrics[lab.replace("\n", " ")] = {"flights_k": k, "flights_n": len(tot)}
            xs = sorted(cb); axs.plot(xs, [sum(r["outcome"] == "success" for r in cb[c]) / len(cb[c]) for c in xs], "-o", color=cols[i], alpha=1.0 if lay in (None, "vanilla4", "p3") else 0.45, lw=2.2 if lay in (None, "vanilla4", "p3") else 1.2, ms=5, label=lab.replace("\n", " "))
        else:
            axf.text(i, 0.02, "not flown", ha="center", va="bottom", fontsize=8, color="#888", rotation=90)
    axf.set_ylim(0, 0.6); axf.set_ylabel("courses completed, 1.0–1.4 m/s pooled", fontsize=9); axf.set_xticks(x); axf.set_xticklabels([l for l, *_ in ladder], fontsize=7); axf.grid(axis="y", ls=":", lw=0.5)
    axf.set_title("In flight (45 Hz), each layout at its cadence and latency", fontsize=10.5, weight="bold", loc="left")
    axs.set_ylim(0, 0.6); axs.set_xlabel("cruise speed (m/s)"); axs.set_ylabel("courses completed"); axs.grid(ls=":", lw=0.5); axs.legend(fontsize=7, frameon=False, ncol=2)
    axs.set_title("… against cruise speed", fontsize=10.5, weight="bold", loc="left")
    for ax in (axf, axs):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("What hand-tuning buys ROS 2, and where it stops — same kernels, same K1; the solved table needs none of it and keeps scaling with the camera", fontsize=11.5, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "ros_ladder", metrics, dpi)


def fig_seed_pairs(dpi):
    rows = V.load_campaigns(); speeds = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    A = cells_by(rows, "xpu_a_cpsat_hard.csv", FC.lat_ms("xpu_a_cpsat_hard.csv"), 0.0); B = cells_by(rows, "ros_vanilla445.csv", FC.lat_ms("ros_vanilla445.csv"), 0.0)
    seeds = sorted({r["seed"] for c in A for r in A[c]} | {r["seed"] for c in B for r in B[c]})
    fig, ax = plt.subplots(figsize=(0.95 * len(speeds) + 2.4, 0.34 * len(seeds) + 1.6)); metrics = collections.Counter()
    shade = {4: V.C_XPU, 3: "#b7d7c4", 2: "#f5c6c2", 1: "#f5c6c2", 0: "#eeeeee"}
    for j, c in enumerate(speeds):
        a = {r["seed"]: r for r in A.get(c, [])}; b = {r["seed"]: r for r in B.get(c, [])}
        for i, s in enumerate(seeds):
            for half, src, dx in ((0, a, -0.24), (1, b, 0.24)):
                r = src.get(s)
                if r is None:
                    continue
                g = 4 if r["outcome"] == "success" else min(r["gates"], 3)
                ax.add_patch(plt.Rectangle((j + dx - 0.22, i - 0.42), 0.44, 0.84, fc=shade[g], ec="white", lw=0.6))
                ax.text(j + dx, i, "✓" if g == 4 else str(g), ha="center", va="center", fontsize=7.5, color="white" if g == 4 else INK, weight="bold")
            if s in a and s in b:
                ga = 4 if a[s]["outcome"] == "success" else a[s]["gates"]; gb = 4 if b[s]["outcome"] == "success" else b[s]["gates"]
                metrics["xpu_further" if ga > gb else "ros_further" if gb > ga else "tie"] += 1
    ax.set_xlim(-0.6, len(speeds) - 0.4); ax.set_ylim(len(seeds) - 0.5, -0.6); ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f"{c:.1f} m/s" for c in speeds]); ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(fc=V.C_XPU, label="course completed"), Patch(fc="#b7d7c4", label="crash after G3"), Patch(fc="#f5c6c2", label="crash after G1 / G2"), Patch(fc="#eeeeee", label="crash before G1")], fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4)
    ax.set_title(f"The same seed under both runtimes — left half XPU-RT CP-SAT, right half ROS 2 vanilla (gates cleared)\n"
                 f"XPU-RT gets further on {metrics['xpu_further']} seed-speed pairs, ROS 2 on {metrics['ros_further']}, tie on {metrics['tie']}", fontsize=9.5, weight="bold", loc="left")
    fig.tight_layout(); save(fig, "seed_pairs", dict(metrics), dpi)


# ------------------------------------------------------------------------------------------- traces
def waterfall(rows):
    """per frame: release -> YOLO start (queue), YOLO span, YOLO end -> nav end, nav end -> control out."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["net"], r["inst"])].append(r)
    ctrl_ends = sorted(max(x["e"] for x in v) for (n, _), v in by.items() if n == "mlp_control")
    out = []
    for (n, k), v in by.items():
        if n != "yolov8_nano_64x96" or k < 1:
            continue
        nav = by.get(("fused_full", k))
        rel = min((x["rel"] for x in v if x.get("rel") is not None), default=None)
        if not nav or rel is None:
            continue
        s = min(x["s"] for x in v); e = max(x["e"] for x in v); en = max(x["e"] for x in nav)
        ec = next((t for t in ctrl_ends if t >= en), None)
        if ec is not None:
            out.append((s - rel, e - s, en - e, ec - en))
    return out


# the arms of the latency waterfall, shared with the atlas: (label, colour, the three replicate traces)
WATERFALL_ARMS = [("XPU-RT · CP-SAT", V.C_XPU, [os.path.join(RES, "xpurt_long", f"trace_acpsat_hardr{k}_other_run1.csv") for k in (1, 2, 3)]),
                  ("XPU-RT · greedy", V.C_XPU2, [os.path.join(RES, "xpurt_long", f"trace_agreedyr{k}_other_run1.csv") for k in (1, 2, 3)]),
                  ("ROS 2 hand-pinned", V.C_P3, [os.path.join(RES, "ros_traced", f"45_p3_r{k}", "trace.csv") for k in (1, 2, 3)]),
                  ("ROS 2 vanilla, 4-hart YOLO", V.C_ROS, [os.path.join(RES, "ros_traced", f"45_vanilla4_r{k}", "trace.csv") for k in (1, 2, 3)]),
                  ("ROS 2 two YOLO nodes, 8 cores", V.C_ROS8, [os.path.join(RES, "ros_traced", f"45_vanilla4x2_r{k}", "trace.csv") for k in (1, 2, 3)]),
                  ("ROS 2 vanilla, serial YOLO", "#b03018", [os.path.join(RES, "ros_traced", f"45_vanilla_r{k}", "trace.csv") for k in (1, 2, 3)])]
WATERFALL_PARTS = ["queue wait (release → YOLO start)", "YOLO (first start → last end)", "YOLO end → nav goal", "nav goal → control out"]
WATERFALL_COLOURS = ["#bdbdbd", V.S.C_YOLO, V.S.C_NAV, V.S.C_CTRL]


def waterfall_summary(w):
    """the bars and the label of one waterfall row: component medians (the bars) and the chain's own median (the label);
    the two totals differ because the median of a sum is not the sum of medians."""
    med = [statistics.median([x[j] for x in w]) for j in range(4)]
    return {"parts_ms": [round(m, 1) for m in med], "sum_of_part_medians_ms": round(sum(med), 1),
            "chain_median_ms": round(statistics.median([sum(x) for x in w]), 1), "frames": len(w)}


def fig_latency_waterfall(dpi):
    fig, ax = plt.subplots(figsize=(9.5, 4.2)); metrics = {}; used = []
    for i, (lab, col, paths) in enumerate(WATERFALL_ARMS):
        have = [p for p in paths if os.path.exists(p)]; w = [x for p in have for x in waterfall(read_trace(p))]
        if not w:
            continue
        used += have; sm = waterfall_summary(w); left = 0
        for j in range(4):
            m = sm["parts_ms"][j]
            ax.barh(i, m, left=left, color=WATERFALL_COLOURS[j], edgecolor="white", lw=0.6, height=0.66, label=WATERFALL_PARTS[j] if i == 0 else None)
            if m > 8:
                ax.text(left + m / 2, i, f"{m:.0f}", ha="center", va="center", fontsize=8, color=INK if j == 0 else "white", weight="bold")
            left += m
        ax.text(left + 4, i, f"{sm['chain_median_ms']:.0f} ms", va="center", fontsize=9, color=col, weight="bold"); metrics[lab] = sm
    ax.set_yticks(range(len(WATERFALL_ARMS))); ax.set_yticklabels([l for l, _, _ in WATERFALL_ARMS], fontsize=9); ax.invert_yaxis()
    for t, (_, col, _) in zip(ax.get_yticklabels(), WATERFALL_ARMS):
        t.set_color(col)
    ax.set_xlabel("camera release → control output over warm frames (ms), 45 Hz camera — bars: component medians; label: the chain's median"); ax.grid(axis="x", ls=":", lw=0.5); ax.legend(fontsize=8, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Where the latency goes: the same chain on the K1 — a scheduled table is compute; a callback graph is queueing", fontsize=10.5, weight="bold", loc="left")
    fig.tight_layout(); save(fig, "latency_waterfall", metrics, dpi, inputs=used)


FIGS = {"crash_position": fig_crash_position, "course_progress": fig_course_progress, "rate_speed_map": fig_rate_speed_map, "ros_ladder": fig_ros_ladder, "seed_pairs": fig_seed_pairs, "latency_waterfall": fig_latency_waterfall}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("figs", nargs="*", default=list(FIGS)); ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()
    for f in a.figs:
        FIGS[f](a.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
