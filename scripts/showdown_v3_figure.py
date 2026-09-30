#!/usr/bin/env python3
"""Warehouse showdown, third form: the flight pair on top, every evidence panel a sweep or a board trace.

  A  top-down, same scene both arms (backdrop = median of the overhead sequence, so props stay and the
     drone and walkers vanish)                          a–d  chase / FPV+YOLO / cross-ToF at four moments
  B  success vs cruise speed per real arm, replayed K1 cadence + camera→control latency, Wilson 95 %
  C  gates cleared before the crash at the display speed, per arm
  D' environment map: arm × (course × density × crossing people), success pooled over the display speeds
  D  mechanism (moment measured, power modelled)      E  body-rate of the two display flights
  F  camera-rate envelope on the K1: camera→control latency and frames delivered, 15…120 Hz
  G  where the work lands: per-hart share of executed inference time, from the traces
  H  added load (90 Hz camera + ffn_block + dronet): control gap / latency / frames, plus the flights
  J  feedback closes the loop: executed − planned drift, isolated profile vs board-calibrated
  K  many runs, one scene: every recorded flight of the display scene over its obstacles and gates
  I  measured Gantt rows (CP-SAT, greedy, ROS 2 on all eight cores, ROS 2 vanilla), 0–100 ms

Every number is read from results/codesign_feedback/ at render time; the sidecar <out>_metrics.json lists them.

    scripts/showdown_v3_figure.py --cell tall1005 --tuned board [--out results/codesign_feedback/refined/warehouse_showdown_v3]
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, math, os, re, statistics, sys, datetime
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_v2.csv"))
sys.path.insert(0, os.path.join(REPO, "sims", "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Rectangle, Patch   # noqa: E402
import showdown_gatecourse as S
import figure_constants as FC   # noqa: E402
from hil_envelope_panel import wilson   # noqa: E402
from hil_story_figure import draw_mechanism   # noqa: E402
from plot_deployment_layers import LOADS, ros_arm   # noqa: E402
from make_measured_gantt_pair import read_trace, per_frame_chain, HZ   # noqa: E402
from env_crash_map import progress   # noqa: E402
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

C_XPU, C_XPU2, C_ROS, C_MOVER = S.C_XPU, S.C_XPU2, S.C_ROS, S.C_MOVER
C_ROS8, C_RICH, C_P3, C_MULTI = "#f28c28", "#8e0f0a", "#5c6bc0", "#c2185b"
INK = "#22242a"
NETC = {"yolov8_nano_64x96": S.C_YOLO, "fused_full": S.C_NAV, "mlp_control": S.C_CTRL, "ffn_block": "#9467bd", "dronet": "#17becf"}
NETL = {"yolov8_nano_64x96": "YOLO", "fused_full": "nav", "mlp_control": "control", "ffn_block": "ffn_block", "dronet": "dronet"}

# flight arms: trace file -> (label, colour); the replay files carry the board run they came from
FLIGHT_ARMS = {"xpu_a_cpsat_hard.csv": ("XPU-RT · CP-SAT", C_XPU), "xpu_a_greedy.csv": ("XPU-RT · greedy", C_XPU2),
               "ros_vanilla445.csv": ("ROS 2 vanilla (4-hart YOLO)", C_ROS), "ros_rvanilla445.csv": ("ROS 2 vanilla + heavier stack", C_RICH),
               "ros_vanilla4x245.csv": ("ROS 2 on all 8 cores (two YOLO nodes)", C_ROS8), "ros_p345.csv": ("ROS 2 hand-tuned (pinned)", C_P3)}
ROS_FAMILY = {"ros_vanilla445.csv": "ros", "ros_vanilla4tm45.csv": "ros", "xpu_a_cpsat_hard.csv": "xpu", "ros_p345.csv": "p3", "xpu_a_greedy.csv": "greedy"}
# board arms for the camera-rate envelope: label -> (colour, XPU tag prefixes by rate | ROS layout)
XPU_TAGS = {"cpsat": {30: "a30cpsat_hard", 45: "acpsat_hard", 60: "a60cpsat_hard", 90: "a90cpsat_hard", 120: "a120hcpsat_hard"},
            "greedy": {30: "a30greedy", 45: "agreedy", 60: "a60greedy", 90: "a90greedy", 120: "a120hgreedy"}}
SPEC_OF_RATE = {30: "wh_chain30_solve_500", 45: "wh_chain45_solve", 60: "wh_chain60_solve_500", 90: "wh_chain90_solve_500", 120: "wh_chain120_solve_h200"}
BOARD_ARMS = [("XPU-RT · CP-SAT", C_XPU, ("xpu", "cpsat")), ("XPU-RT · greedy", C_XPU2, ("xpu", "greedy")),
              ("ROS 2 vanilla (4-hart YOLO)", C_ROS, ("ros", "vanilla4")), ("ROS 2 on all 8 cores", C_ROS8, ("ros", "vanilla4x2")),
              ("ROS 2 vanilla, serial YOLO", "#b03018", ("ros", "vanilla")), ("ROS 2 hand-tuned (pinned)", C_P3, ("ros", "p3"))]


def word(x):
    """letters-only key for TeX macro names: 1.2 -> OnePointTwo, 120 -> OneTwenty-ish (digit words)."""
    D = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    s = f"{x:g}" if isinstance(x, (int, float)) else str(x)
    return "".join(D[int(c)] if c.isdigit() else ("Point" if c == "." else re.sub(r"[^A-Za-z]", "", c)) for c in s)


# ----------------------------------------------------------------------------------------------- flights
LOADED_CSVS: list[str] = []   # every campaign CSV a render read, for the sidecar's input hashes


def load_campaigns():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "campaign_*", "campaign.csv"))):
        camp = os.path.basename(os.path.dirname(f)); LOADED_CSVS.append(f)
        for r in flight_rows(f):
            try:
                rows.append(dict(camp=camp, trace=os.path.basename(r.get("ctrl_trace", "")), course=r.get("course") or "a",
                                 dens=round(float(r.get("prop_density") or 0.3), 2), walk=float(r.get("walk_speed") or 0),
                                 cross=int(float(r.get("walk_cross") or 0)), gain=round(float(r["moment_scale"]), 5),
                                 lat=round(float(r.get("percep_latency_ms") or 0), 1), hold=round(float(r.get("percep_hold_ms") or 0), 1),
                                 cruise=round(float(r["cruise_speed"]), 2), gates=int(float(r["gates_passed"])), outcome=r["outcome"],
                                 crash_type=r.get("crash_type") or "", seed=int(float(r["seed"])), ph=round(float(r.get("person_h") or 2.4), 2)))
            except (KeyError, ValueError):
                continue
    return rows


def load_extra_csv(path, camp):
    """rows of one campaign CSV outside campaign_*/ (same schema), e.g. campaign_v2 (people 1.7 m, cadence only)."""
    rows = []; LOADED_CSVS.append(path)
    for r in flight_rows(path):
        try:
            rows.append(dict(camp=camp, trace=os.path.basename(r.get("ctrl_trace", "") or ""), course=r.get("course") or "a", dens=round(float(r.get("prop_density") or 0.3), 2),
                             walk=float(r.get("walk_speed") or 0), cross=int(float(r.get("walk_cross") or 0)), gain=round(float(r["moment_scale"]), 5),
                             lat=round(float(r.get("percep_latency_ms") or 0), 1), hold=round(float(r.get("percep_hold_ms") or 0), 1), cruise=round(float(r["cruise_speed"]), 2),
                             gates=int(float(r["gates_passed"])), outcome=r["outcome"], crash_type=r.get("crash_type") or "", seed=int(float(r["seed"])), ph=round(float(r.get("person_h") or 2.4), 2)))
        except (KeyError, ValueError):
            continue
    return rows


def flight_cells(rows, course="a", dens=0.30, gain=0.0055, latency=True, walk=0.0, cross=0, gain_by_trace=None, camps=None, person_h=2.4):
    """(trace, cruise) -> list of rows, the latency-replayed cells of one scene at fixed gain (or a gain per arm);
    the people's height is part of the scene, so a cell never pools the 2.4 m and 1.7 m campaigns."""
    out = collections.defaultdict(list)
    for r in rows:
        g = (gain_by_trace or {}).get(r["trace"], gain)
        if camps is not None and r["camp"] not in camps:
            continue
        if r["course"] != course or r["dens"] != dens or abs(r["gain"] - g) > 2e-5 or r["walk"] != walk or r["cross"] != cross or r.get("ph", 2.4) != person_h:
            continue
        if latency and r["lat"] <= 0:
            continue
        out[(r["trace"], r["cruise"])].append(r)
    return out


def equalise(cells, arms):
    """Fair counts: for every (cruise, seed) keep the first k flights of each arm, k = the fewest any of `arms`
    flew that seed at that cruise (the simulator is non-deterministic, so arms carry replicates; a panel compares
    the same seeds, the same number of times). Returns the trimmed cells and the number of flights set aside."""
    out = {}; dropped = 0
    cruises = sorted({c for (t, c) in cells if t in arms})
    for c in cruises:
        per = {t: collections.defaultdict(list) for t in arms}
        for t in arms:
            for r in cells.get((t, c), []):
                per[t][r["seed"]].append(r)
        seeds = set.union(*[set(per[t]) for t in arms]) if arms else set()
        for t in arms:
            keep = []
            for sd in sorted(seeds):
                k = min(len(per[u][sd]) for u in arms if per[u][sd]) if any(per[u][sd] for u in arms) else 0
                keep += per[t][sd][:k]; dropped += max(0, len(per[t][sd]) - k)
            if keep:
                out[(t, c)] = keep
    for k, v in cells.items():                                   # arms not drawn keep their rows
        if k[0] not in arms:
            out[k] = v
    return out, dropped


def draw_B(ax, cells, arms, fz, display_cruise=None, replay="cadence and latency"):
    for tr in arms:
        lab, col = FLIGHT_ARMS[tr]
        pts = sorted((c, sum(x["outcome"] == "success" for x in v), len(v)) for (t, c), v in cells.items() if t == tr and len(v) >= 12)
        if not pts:
            continue
        xs = [c for c, _, _ in pts]; ps, lo, hi = zip(*[wilson(k, n) for _, k, n in pts])
        ax.fill_between(xs, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(xs, ps, "-o", color=col, lw=2.2, ms=6, label=lab, zorder=4)
        if tr in ("xpu_a_cpsat_hard.csv", "ros_vanilla445.csv"):          # k/n on the display pair's arms only
            for (c, k, n), p in zip(pts, ps):
                up = tr.startswith("xpu")
                ax.annotate(f"{k}/{n}", (c, p), textcoords="offset points", xytext=(0 if up else 5, 7 if up else -11), ha="center" if up else "left", va="bottom" if up else "top", fontsize=fz["tiny"], color=col)
    if display_cruise:
        ax.axvline(display_cruise, color="0.6", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(display_cruise, 1.0, "display\ncell", ha="center", va="top", fontsize=fz["tiny"], color="0.4")
    ax.set_ylim(-0.12, 1.0); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("course completed (fraction)", fontsize=fz["lab"])
    ax.grid(ls=":", lw=0.6, color="#d4d1cb"); ax.tick_params(labelsize=fz["tick"]); ax.legend(fontsize=fz["leg"], frameon=False, loc="upper right", ncol=1)
    ax.set_title(f"Success against cruise speed — each arm at its own\nK1 {replay} (same seeds; Wilson 95 %)", fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_C(ax, cells, arms, cruise, fz):
    names, hist = [], []
    for tr in arms:
        v = cells.get((tr, cruise), [])
        if not v:
            continue
        names.append((FLIGHT_ARMS[tr][0], FLIGHT_ARMS[tr][1], len(v)))
        hist.append(np.array([sum(x["gates"] == g and x["outcome"] != "success" for x in v) for g in range(4)] + [sum(x["outcome"] == "success" for x in v)], float) / len(v))
    shades = ["#f1f1f1", "#f5c6c2", "#ee9c95", "#b7d7c4", C_XPU]
    labs = ["crash before G1", "crash after G1", "crash after G2", "crash after G3", "course completed"]
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for g in range(5):
        vals = np.array([h[g] for h in hist])
        ax.barh(y, vals, left=left, color=shades[g], edgecolor="white", lw=0.8, label=labs[g], height=0.66)
        for i, v in enumerate(vals):
            if v >= 0.08:
                ax.text(left[i] + v / 2, y[i], f"{v * names[i][2]:.0f}", ha="center", va="center", fontsize=fz["tiny"], color=INK if g in (0, 1, 3) else "white", weight="bold")
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([f"{n}\n(n={k})" for n, _, k in names], fontsize=fz["tick"])
    for t, (_, col, _) in zip(ax.get_yticklabels(), names):
        t.set_color(col)
    ax.set_ylim(len(names) + 0.6, -0.5); ax.set_xlim(0, 1); ax.set_xlabel("fraction of flights", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"])
    ax.legend(fontsize=fz["tiny"], frameon=False, ncol=3, loc="lower left", handlelength=1.2, columnspacing=0.9)
    ax.set_title(f"How far each arm gets at {cruise:.1f} m/s\n(gates cleared before the collision)", fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return dict(zip([n for n, _, _ in names], [h.tolist() for h in hist]))


def _one_per_seed(v):
    seen = set(); out = []
    for r in v:
        if (r["cruise"], r["seed"]) not in seen:
            seen.add((r["cruise"], r["seed"])); out.append(r)
    return out


def dprime_counts(rows, families=("xpu", "ros", "p3"), speeds=(1.0, 1.2, 1.4), gain=0.0055, person_h=2.4):
    """(envs, K, N): courses completed per (course, density, crossing) and arm family, pooled over `speeds`, latency-replayed
    cells only, one flight per seed; the people's height is part of the scene, so only rows of `person_h` are pooled and
    an environment is kept only when every family has at least 12 flights in it."""
    def ok(r):
        return r["lat"] > 0 and r["walk"] == 0 and abs(r["gain"] - gain) < 2e-5 and r["cruise"] in speeds and r.get("ph", 2.4) == person_h
    envs = sorted({(r["course"], r["dens"], r["cross"]) for r in rows if ok(r)})
    N = np.zeros((len(families), len(envs)), dtype=int); K = np.zeros_like(N)
    for j, env in enumerate(envs):
        for i, fam in enumerate(families):
            v = _one_per_seed([r for r in rows if (r["course"], r["dens"], r["cross"]) == env and ROS_FAMILY.get(r["trace"]) == fam and ok(r)])
            if len(v) >= 12:
                K[i, j] = sum(x["outcome"] == "success" for x in v); N[i, j] = len(v)
    keep = [j for j in range(len(envs)) if all(N[i, j] >= 12 for i in range(len(families)))]
    return [envs[j] for j in keep], K[:, keep], N[:, keep]


def draw_Dprime(ax, rows, fz, families=("xpu", "ros", "p3"), speeds=(1.0, 1.2, 1.4), gain=0.0055, person_h=2.4):
    """success pooled over `speeds` per (course, density, crossing), latency-replayed cells only."""
    envs, K, N = dprime_counts(rows, families, speeds, gain, person_h)
    fam_lab = {f: FC.forest_family_label(f) for f in families}
    M = np.where(N > 0, K / np.maximum(N, 1), np.nan)
    cmap = plt.cm.YlGn; cmap.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    for i in range(len(families)):
        for j in range(len(envs)):
            if N[i, j]:
                ax.text(j, i, f"{K[i, j]}/{N[i, j]}", ha="center", va="center", fontsize=fz["tiny"], color=INK if M[i, j] < 0.3 else "white", weight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=fz["tiny"], color="#999")
    ax.set_xticks(range(len(envs))); ax.set_xticklabels([f"course {c.upper()}\nd {d:.2f}" + ("\ncrossing" if x else "") for c, d, x in envs], fontsize=fz["tiny"])
    ax.set_yticks(range(len(families))); ax.set_yticklabels([fam_lab[f] for f in families], fontsize=fz["tick"])
    ax.set_title(f"Environments both arms flew: course completed,\n{min(speeds):.1f}–{max(speeds):.1f} m/s pooled, one flight per seed, people {person_h:.1f} m", fontsize=fz["title"], weight="bold", loc="left")
    return {"envs": [f"{c}_d{d:.2f}{'_cross' if x else ''}" for c, d, x in envs], "families": list(families), "k": K.tolist(), "n": N.tolist(), "person_h": person_h}


# ------------------------------------------------------------------------------------------------- board
def cam_hz_of(meta):
    per = meta.get("camera_period_ms")
    pn = meta.get("periodic_networks")
    if not per and isinstance(pn, dict):
        e = pn.get("yolov8_nano_64x96"); per = e if isinstance(e, (int, float)) else (e or {}).get("period") or (e or {}).get("period_ms")
    elif not per and isinstance(pn, list):
        e = next((x for x in pn if str(x.get("name", x.get("network", ""))).startswith("yolov8")), None); per = (e or {}).get("period") or (e or {}).get("period_ms")
    per = float(per) if per else FC.fallback("camera_period_ms", 1000.0 / 45.0, "schedule metadata names no camera period")
    return 1000.0 / per * int(meta.get("cameras", 1) or 1)


def xpu_run(tag, yolo_win):
    """one XPU-RT board run: camera Hz, chain median, delivered/s, frames late, control gap."""
    t = os.path.join(RES, "xpurt_long", f"trace_{tag}_other_run1.csv")
    if not os.path.exists(t):
        return None
    rows = read_trace(t)
    if len(rows) < 200:
        return None
    man = json.load(open(t.replace("trace_", "manifest_").replace(".csv", ".json")))
    sched = json.load(open(man["schedule"])).get("metadata", {}) if os.path.exists(man.get("schedule", "")) else {}
    ch = per_frame_chain(rows); warm = [c for k, c, _ in ch if k >= 1]
    run_s = (max(r["e"] for r in rows) - min(r["s"] for r in rows)) / 1000.0
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["net"], r["inst"])].append(r)
    ends = sorted(max(x["e"] for x in v) for (n, _), v in by.items() if n == "mlp_control")
    gaps = [ends[i + 1] - ends[i] for i in range(1, len(ends) - 1)]
    late = []
    for (n, k), fr in by.items():
        if n != "yolov8_nano_64x96" or k < 1:
            continue
        rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
        if rel is not None:
            late.append(max(x["e"] for x in fr) - rel > yolo_win)
    return dict(tag=tag, cam=cam_hz_of(sched), lat=statistics.median(warm) if warm else None, deliv=len(warm) / run_s if run_s else 0.0,
                late=int(sum(late)), checked=len(late), gap=statistics.mean(gaps) if gaps else None, n_frames=len(warm))


def spec_window(name):
    p = os.path.join(REPO, "data", "toplevel", name + ".json")
    d = json.load(open(p))["networks"]["yolov8_nano_64x96"] if os.path.exists(p) else {}
    w = d.get("window_duration") or d.get("period")
    return float(w) if w else FC.fallback("spec_window_ms", 66.667, f"{name} names no YOLO window; the frames-late verdict needs one")


def xpu_rate_points(kind):
    out = {}
    for hz, pref in XPU_TAGS[kind].items():
        runs = [xpu_run(f"{pref}r{k}", spec_window(SPEC_OF_RATE[hz])) for k in (1, 2, 3)]
        runs = [r for r in runs if r and r["lat"] is not None]
        if runs:
            out[hz] = dict(lat=statistics.median([r["lat"] for r in runs]), deliv=statistics.mean([r["deliv"] for r in runs]),
                           late=sum(r["late"] for r in runs), checked=sum(r["checked"] for r in runs), gap=statistics.mean([r["gap"] for r in runs if r["gap"]]),
                           n=len(runs), tags=[r["tag"] for r in runs], cam=runs[0]["cam"])
    return out


def ros_rate_points(layout):
    out = collections.defaultdict(lambda: dict(lat=[], deliv=[], deliv_full=[], gap=[], n=0, tags=[]))
    for r in csv.DictReader(open(os.path.join(RES, "ros_traced", "summary.csv"))):
        m = re.match(r"(\d+)_(.+)_r(\d+)$", r["tag"])
        if not m or m.group(2) != layout or not r.get("gap_mean_ms"):
            continue
        hz = int(m.group(1)); manp = os.path.join(RES, "ros_traced", r["tag"], "manifest.json")
        secs = float(json.load(open(manp)).get("seconds", 20.0)) if os.path.exists(manp) else 20.0
        warm = float(r.get("warmup_ms") or 3000.0) / 1000.0
        o = out[hz]; o["lat"].append(float(r["e2e_goal_med_ms"])); o["gap"].append(float(r["gap_mean_ms"]))
        o["deliv"].append(float(r["n_consumed"]) / max(1e-6, secs - warm)); o["deliv_full"].append(float(r["n_consumed"]) / secs); o["n"] += 1; o["tags"].append(r["tag"])
    return {hz: dict(lat=statistics.median(o["lat"]), deliv=statistics.mean(o["deliv"]), deliv_full=statistics.mean(o["deliv_full"]), gap=statistics.mean(o["gap"]), n=o["n"], tags=o["tags"], cam=hz)
            for hz, o in out.items()}


def draw_F(ax_lat, ax_del, arms, fz):
    out = {}
    for lab, col, (kind, key) in arms:
        pts = xpu_rate_points(key) if kind == "xpu" else ros_rate_points(key)
        if not pts:
            continue
        out[lab] = pts
        hz = sorted(pts); lat = [pts[h]["lat"] for h in hz]; de = [pts[h]["deliv"] for h in hz]
        st = dict(color=col, lw=2.2, ms=7, mec="white", mew=0.8)
        if kind == "xpu":                          # the frame verdict, from the traces: rates at which every frame met its window
            ok = [h for h in hz if pts[h]["late"] == 0]; bad = [h for h in hz if pts[h]["late"] > 0]
            lab = lab + (f" — every frame on time to {max(ok)} Hz" if ok and not bad else (f" — frames late above {max(ok)} Hz" if ok else " — frames late at every rate"))
        ax_lat.plot(hz, lat, "-o" if kind == "xpu" else "-s", label=lab, zorder=4, **st); ax_del.plot(hz, de, "-o" if kind == "xpu" else "-s", zorder=4, **st)
    hz_all = sorted({h for p in out.values() for h in p})
    ax_del.plot(hz_all, hz_all, ls=(0, (4, 3)), color="0.45", lw=1.2, zorder=2); ax_del.text(hz_all[0] * 1.15, hz_all[0] * 1.6 + 8, "every frame delivered", fontsize=fz["tiny"], color="0.35", ha="left", va="bottom", rotation=0)
    ax_lat.set_yscale("log"); ax_lat.set_ylabel("cam→ctrl (ms)", fontsize=fz["lab"]); ax_del.set_ylabel("frames to ctrl (/s)", fontsize=fz["lab"])
    ax_del.set_xlabel("camera rate (Hz)", fontsize=fz["lab"])
    for ax in (ax_lat, ax_del):
        ax.set_xscale("log"); ax.set_xticks(hz_all); ax.set_xticklabels([f"{h}" for h in hz_all], fontsize=fz["tick"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(ls=":", lw=0.6, color="#d4d1cb"); ax.tick_params(labelsize=fz["tick"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_lat.tick_params(labelbottom=False)
    ax_del.set_ylim(0, max(hz_all) * 1.9)                        # headroom for the legend above the delivered-frames lines
    ax_del.legend(handles=ax_lat.get_lines()[:len(out)], labels=[l.get_label() for l in ax_lat.get_lines()[:len(out)]], fontsize=fz["tiny"] * 0.92, frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8)
    ax_lat.set_title("Camera rate on the K1: camera→control latency\nand frames delivered, same kernels", fontsize=fz["title"], weight="bold", loc="left")
    return out


def hart_share(tag_or_dir, kind):
    """{hart: {net: ms}} from a trace; ROS YOLO callbacks credited to their node's pool harts ∪ callback hart."""
    if kind == "xpu":
        path = os.path.join(RES, "xpurt_long", f"trace_{tag_or_dir}_other_run1.csv"); pools = {}
    else:
        path = os.path.join(RES, "ros_traced", tag_or_dir, "trace.csv"); man = json.load(open(os.path.join(RES, "ros_traced", tag_or_dir, "manifest.json")))
        pools = {}
        for p in (man.get("processes") or {}).values():
            if p.get("yolo_pool") and p.get("pool_harts"):
                for node in str(p.get("nodes", "")).split(","):
                    pools[node.strip()] = [int(h) for h in str(p["pool_harts"]).split(",") if h.strip()]
    share = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in read_trace(path):
        ms = r["e"] - r["s"]; harts = {r["hart"]}
        if kind == "ros" and r["net"] == "yolov8_nano_64x96":
            harts |= set(pools.get(r["name"], []))
        for h in harts:
            share[h][r["net"]] += ms
    return share


def draw_G(ax, arms, fz):
    """arm × hart map of the share of executed inference time (all networks), pooled over the runs."""
    out = {}; M = np.zeros((len(arms), 8)); yolo = np.zeros((len(arms), 8))
    for i, (lab, col, kind, tags) in enumerate(arms):
        tot = collections.defaultdict(lambda: collections.defaultdict(float))
        for t in tags:
            for h, d in hart_share(t, kind).items():
                for n, v in d.items():
                    tot[h][n] += v
        allsum = sum(v for d in tot.values() for v in d.values()) or 1.0
        for h in range(8):
            M[i, h] = 100.0 * sum(tot[h].values()) / allsum; yolo[i, h] = 100.0 * tot[h].get("yolov8_nano_64x96", 0.0) / allsum
        out[lab] = {f"Hart{word(h)}": round(float(M[i, h]), 2) for h in range(8)}
    ax.imshow(M, cmap="Greens", vmin=0, vmax=max(30.0, M.max()), aspect="auto")
    for i in range(len(arms)):
        for h in range(8):
            ax.text(h, i, f"{M[i, h]:.0f}", ha="center", va="center", fontsize=fz["tiny"], weight="bold", color="white" if M[i, h] > 0.55 * max(30.0, M.max()) else INK)
    ax.set_xticks(range(8)); ax.set_xticklabels([f"{'P' if h < 4 else 'E'}{h % 4}" for h in range(8)], fontsize=fz["tick"])
    short = {"XPU-RT · CP-SAT": "XPU-RT CP-SAT", "XPU-RT · greedy": "XPU-RT greedy", "ROS 2 on all 8 cores\n(two YOLO nodes)": "ROS 2 two-YOLO", "ROS 2 vanilla\n(4-hart YOLO pool)": "ROS 2 vanilla"}
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([f"{short.get(lab, lab)}\n{int((M[i] >= 3.0).sum())}/8 harts busy" for i, (lab, _, _, _) in enumerate(arms)], fontsize=fz["tiny"])
    for t, (_, col, _, _) in zip(ax.get_yticklabels(), arms):
        t.set_color(col)
    ax.set_xlabel("hart (P = performance, E = efficiency cluster) — share of executed inference time, %", fontsize=fz["lab"]); ax.tick_params(length=0)
    # the camera rate these runs were asked for, from the executed table of the first XPU-RT arm
    t0 = next((t for _, _, k, ts in arms if k == "xpu" for t in ts), None)
    manp = os.path.join(RES, "xpurt_long", f"manifest_{t0}_other_run1.json") if t0 else ""
    cam = cam_hz_of(json.load(open(json.load(open(manp))["schedule"])).get("metadata", {})) if manp and os.path.exists(manp) else None
    ax.set_title(f"Where the work lands, {cam:.0f} Hz camera\n(board traces, three runs pooled)" if cam else "Where the work lands\n(board traces, three runs pooled)", fontsize=fz["title"], weight="bold", loc="left")
    return out


def xpu_load_arm(prefix):
    runs = [xpu_run(f"{prefix}r{k}", spec_window("wh_chain90_rich_solve_500")) for k in (1, 2, 3)]
    runs = [r for r in runs if r and r["lat"] is not None]
    if not runs:
        return None
    return {"gap": [r["gap"] for r in runs], "lat": [r["lat"] for r in runs], "deliv": [r["deliv"] for r in runs], "drop": [0.0], "n": len(runs), "offered": runs[0]["cam"]}


def draw_H(axes, cells_rich, fz):
    title, offered, ros_arms, xpu_arms = LOADS[1]
    short = {"rvanilla": "ROS 2\nvanilla", "rvanilla4": "ROS 2\n4-hart", "rp3": "ROS 2\npinned", "b5greedy": "XPU-RT\ngreedy", "b5cpsat_hard": "XPU-RT\nCP-SAT"}
    arms = [(short.get(t, n), ros_arm(t, 90), "ROS", t) for n, t in ros_arms.items()] + [(short.get(p, n), xpu_load_arm(p), "XPU", p) for n, p in xpu_arms]
    arms = [a for a in arms if a[1]]
    names = [n for n, _, _, _ in arms]; x = np.arange(len(arms)); out = {}
    for ax, (key, ylab, ref, log) in zip(axes[:3], [("gap", "ctrl gap\n(ms)", 10.0, False), ("lat", "cam→ctrl\n(ms)", None, True), ("deliv", "frames\n(/s)", offered, False)]):
        for i, (n, d, kind, tag) in enumerate(arms):
            v = statistics.mean(d[key]); col = C_XPU if kind == "XPU" and "CP-SAT" in n else (C_XPU2 if kind == "XPU" else C_RICH)
            ax.bar(i, v, color=col, width=0.64, edgecolor="k", lw=0.4)
            ax.annotate(f"{v:.0f}" if key != "gap" else f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=fz["tiny"], weight="bold" if kind == "XPU" else "normal")
            out.setdefault(re.sub(r"[^A-Za-z]", "", n), {})[key] = round(v, 2)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.8, alpha=0.6)
        if log:
            ax.set_yscale("log"); ax.set_ylim(20, 700); ax.set_yticks([50, 200, 500]); ax.set_yticklabels(["50", "200", "500"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.set_ylim(0, max(statistics.mean(d[key]) for _, d, _, _ in arms) * 1.4)
        ax.set_ylabel(ylab, fontsize=fz["tiny"], labelpad=2); ax.set_xticks(x); ax.set_xticklabels([]); ax.tick_params(labelsize=fz["tiny"]); ax.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # the same arms flown: completed courses over 1.0–1.8 m/s, heavier-stack campaign
    axf = axes[3]; fl = {"rvanilla": "ros_rvanilla490.csv", "rvanilla4": "ros_rvanilla490.csv", "rp3": None, "b5greedy": "xpu_b5_greedy.csv", "b5cpsat_hard": "xpu_b5_cpsat.csv"}
    for i, (n, d, kind, tag) in enumerate(arms):
        tr = fl.get(tag)
        v = [r for (t, c), rs in cells_rich.items() if t == tr for r in rs] if tr else []
        if v:
            k = sum(r["outcome"] == "success" for r in v); col = C_XPU if kind == "XPU" and "CP-SAT" in n else (C_XPU2 if kind == "XPU" else C_RICH)
            axf.bar(i, k / len(v), color=col, width=0.64, edgecolor="k", lw=0.4); axf.annotate(f"{k}/{len(v)}", (i, k / len(v)), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=fz["tiny"])
            out.setdefault(re.sub(r"[^A-Za-z]", "", n), {})["flightsK"] = k; out[re.sub(r"[^A-Za-z]", "", n)]["flightsN"] = len(v)
        else:
            axf.text(i, 0.02, "not flown", ha="center", va="bottom", fontsize=fz["tiny"], color="#888", rotation=90)
    axf.set_ylim(0, 0.5); axf.set_yticks([0, 0.25, 0.5]); axf.set_ylabel("completed\n(fraction)", fontsize=fz["tiny"], labelpad=2); axf.set_xticks(x); axf.set_xticklabels(names, fontsize=fz["tiny"])
    axf.tick_params(labelsize=fz["tiny"]); axf.grid(axis="y", ls=":", lw=0.6, color="#d4d1cb")
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)
    axes[0].set_title("Added load: 90 Hz camera + ffn_block\n+ dronet on the same 8 harts", fontsize=fz["title"], weight="bold", loc="left")
    return out


def draw_J(ax, fz, spec="wh_chain120_solve_h200", isolated="fba120hr0", calibrated="a120h"):
    try:
        import hil_feedback_closeup as HC
        from hil_feedback_figure import IRS
        from k1_trace import ir_slot_map
        sp = json.load(open(os.path.join(REPO, "data/toplevel", spec + ".json"))); nets = set(sp["networks"])
        slotmap = {n: ir_slot_map(json.load(open(os.path.join(REPO, IRS[n])))) for n in nets if n in IRS and os.path.exists(os.path.join(REPO, IRS[n]))}
    except Exception as e:   # the feedback tooling is optional for the composite
        ax.text(0.5, 0.5, f"feedback data unavailable\n({e})", ha="center", va="center", fontsize=fz["tiny"], transform=ax.transAxes); ax.axis("off"); return {}
    out = {}
    for lab, tag, col in [("isolated profile", isolated, "#d62728"), ("board-calibrated (per-dispatch cost)", calibrated, C_XPU)]:
        E, D, span, busy = HC.load(tag, nets, slotmap)
        pts = collections.defaultdict(list)
        for h in D:
            for x, y in D[h]:
                pts[round(x / 10) * 10].append(y)
        xs = sorted(pts); ys = [statistics.median(pts[x]) for x in xs]
        ax.plot(xs, ys, "-o", ms=3, color=col, lw=2.0, label=f"{lab}: harts {busy:.0%} busy")
        out[re.sub(r"[^A-Za-z]", "", lab)] = {"final_drift_ms": round(float(statistics.median([d for h in D for x, d in D[h] if x > 180])), 1), "span_ms": round(float(span), 1), "busy": round(float(busy), 3), "tag": tag}
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("planned start (ms), 200 ms table at 120 Hz", fontsize=fz["lab"]); ax.set_ylabel("executed − planned (ms)", fontsize=fz["lab"])
    ax.legend(fontsize=fz["leg"], frameon=False, loc="upper left"); ax.grid(ls=":", lw=0.6, color="#d4d1cb"); ax.tick_params(labelsize=fz["tick"])
    ax.set_title("Feedback closes the loop:\nsame spec, re-costed from the board", fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return out


# ----------------------------------------------------------------------------------------------- scene K
def draw_K(ax, X, R, scene_dirs, fz, ros_label, xpu_label):
    """every recorded flight of the display scene over its obstacles, aisle drawn horizontally as in panel A:
    crates/racks as footprints, people as dots on their patrol, gate frames as posts; the display pair bold."""
    gates = X["gates_world"]; pm = X["person_mask"].astype(bool); ob0 = X["obst_pos"][0]; kinds = X["obst_kind"] if "obst_kind" in X.files else np.array(["obj"] * len(ob0))
    st = ob0[~pm]; keep = st[:, 2] > -10.0; sk = kinds[~pm][keep]; st = st[keep]
    def P(xy):                                                   # world (x lateral, y along the aisle) -> (along, lateral)
        xy = np.asarray(xy); return xy[..., 1], xy[..., 0]
    for (x, y, z), k in zip(st, sk):
        big = str(k).lower().startswith(("rack", "shelf", "pallet")); w = 1.2 if big else 0.62
        ax.add_patch(Rectangle((y - w / 2, x - w / 2), w, w, fc="#c9c3b8" if not big else "#a89f92", ec="#8c857a", lw=0.5, zorder=1))
    ppl = X["obst_pos"][:, pm, :]
    for j in range(ppl.shape[1]):
        ax.plot(ppl[:, j, 1], ppl[:, j, 0], color=C_MOVER, lw=0.8, alpha=0.35, zorder=1.5)
        ax.plot(ppl[0, j, 1], ppl[0, j, 0], "o", color=C_MOVER, ms=5, mec="white", zorder=2)
    for i, g in enumerate(gates):
        for sgn in (-1, 1):
            ax.plot([g[1], g[1]], [g[0] + sgn * 0.75, g[0] + sgn * 0.95], color="#f2a900", lw=5, solid_capstyle="butt", zorder=2.5)
        ax.text(g[1], g[0] + 1.15, f"G{i + 1}", ha="center", va="bottom", fontsize=fz["tiny"], color="#b07a00", weight="bold")
    counts = {}
    for lab, col, d in scene_dirs:
        fl = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))] if d and os.path.isdir(d) else []
        k = 0
        for z in fl:
            p = z["poses"]; ax.plot(p[:, 1], p[:, 0], color=col, lw=0.8, alpha=0.6, zorder=3)
            if str(z["outcome"]) == "crash":
                ax.plot(p[-1, 1], p[-1, 0], "x", color=col, ms=6, mew=1.5, zorder=4)
            k += str(z["outcome"]) == "success"
        counts[lab] = (k, len(fl))
    y0 = min(float(X["poses"][0, 1]), float(R["poses"][0, 1]))
    for p, col in [(R["poses"], C_ROS), (X["poses"], C_XPU)]:
        ax.plot(p[:, 1], p[:, 0], color="white", lw=4, zorder=5); ax.plot(p[:, 1], p[:, 0], color=col, lw=2.2, zorder=6)
    ax.plot(R["poses"][-1, 1], R["poses"][-1, 0], "X", color=C_ROS, ms=11, mec="white", mew=1.2, zorder=7)
    xs = np.concatenate([X["poses"][:, 0], R["poses"][:, 0], gates[:, 0]])
    ax.set_xlim(y0 - 1.0, gates[-1, 1] + 1.6); ax.set_ylim(xs.min() - 1.7, xs.max() + 1.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    hs = [Line2D([], [], color=C_XPU, lw=2.2, label=xpu_label), Line2D([], [], color=C_ROS, lw=2.2, label=ros_label)]
    hs += [Line2D([], [], color=col, lw=1.2, alpha=0.7, label=f"{lab}: {counts[lab][0]}/{counts[lab][1]} completed") for lab, col, _ in scene_dirs if counts.get(lab, (0, 0))[1]]
    hs += [Patch(fc="#c9c3b8", ec="#8c857a", label="crate / rack"), Line2D([], [], marker="o", color=C_MOVER, ls="", label="person (patrol)"), Line2D([], [], color="#f2a900", lw=4, label="gate frame"), Line2D([], [], marker="x", color="k", ls="", label="collision")]
    ax.legend(handles=hs, fontsize=fz["tiny"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, handlelength=1.4, columnspacing=1.0)
    n_runs = sum(v[1] for v in counts.values())
    ax.set_title(f"The display scene, every recorded run ({n_runs} flights):\nobstacles, people, gate frames, collisions" if n_runs else "The display scene: obstacles, people, gate frames,\nthe two flights (every recorded run once the scene runs land)", fontsize=fz["title"], weight="bold", loc="left")
    return {k: {"completed": v[0], "n": v[1]} for k, v in counts.items()}


# -------------------------------------------------------------------------------------------------- A/a-d/E
def backdrop(X):
    if "ov_seq" in X.files and "ov_seq_t" in X.files and len(X["ov_seq"]) >= 3:
        seq = X["ov_seq"]; t = np.asarray(X["ov_seq_t"], float); keep = seq[t >= 2.0]
        if len(keep) >= 3:
            img = np.median(keep, axis=0).astype(seq.dtype); return img, f"median of {len(keep)} overhead frames (t ≥ 2 s)", float(np.std(img))
        return seq[1], "second overhead frame", float(np.std(seq[1]))
    return X["ov_bg"], "first overhead frame", float(np.std(X["ov_bg"]))


def moments_for(X, R, path_start, hx, hr, outcome=None):
    """The four moments the a-d strips draw. `outcome` is the baseline's own recorded outcome; a
    baseline that ran out of time hit nothing, so passing it keeps the labels off a collision it
    never had. Left unset the labels read as they always have, for a baseline that crashed."""
    xxyz = X["poses"][:, :3]; rxyz = R["poses"][:, :3]; rt = R["t_s"]; pm = X["person_mask"].astype(bool); gates = X["gates_world"]
    nr, nx = len(rxyz), len(xxyz)
    st = X["obst_pos"][0][~pm]; st = st[st[:, 2] > -10.0]; tall = st[st[:, 2] > 2.0]
    if len(tall) and nx > path_start + 2:
        dd = np.linalg.norm(xxyz[path_start:, None, :2] - tall[None, :, :2], axis=2); rel = int(dd.min(axis=1).argmin())
        step_near = path_start + rel; clear = float(dd[rel].min()); near_bin = tall[int(dd[rel].argmin()), :3]
    else:
        step_near, clear, near_bin = int(0.45 * nx), 0.66, None
    ry = rxyz[:, 1]
    ng = int(R["gates_passed"]) if "gates_passed" in R.files else sum(1 for g in gates if np.linalg.norm(rxyz[:, :2] - g[:2], axis=1).min() < 0.8)
    # the baseline's first moment: just after the last credited gate that it passed a clear 1.5 s before the end
    # (closest approach to that gate's centre), so the moment is not the crash itself
    dt = float(rt[1] - rt[0]) if len(rt) > 1 else 0.01; g_last = None; ng_a = 0
    for k in range(ng, 0, -1):
        gk = int(np.linalg.norm(rxyz[:, :2] - gates[k - 1][:2], axis=1).argmin())
        if (nr - 1 - gk) * dt >= 1.5:
            g_last, ng_a = gk, k; break
    m1 = min(nr - 2, g_last + max(3, nr // 40)) if g_last is not None else int(0.25 * nr)
    m1_lab = f"ROS 2 {hr:.0f} Hz · clears G{ng_a}" if ng_a else f"ROS 2 {hr:.0f} Hz · has not reached G1"
    # what the baseline hit: the nearest static object to the last pose
    hit = "obstacle"
    if len(st):
        d = np.linalg.norm(st[:, :2] - rxyz[-1, :2], axis=1); k = int(d.argmin())
        kinds = X["obst_kind"][~pm][X["obst_pos"][0][~pm][:, 2] > -10.0] if "obst_kind" in X.files else None
        hit = (str(kinds[k]) if kinds is not None else "obstacle") if d[k] < 1.6 else "obstacle"
        gd = np.abs(gates[:, 1] - rxyz[-1, 1]).min()
        if gd < 0.6 and d[k] >= 0.9:
            hit = "gate frame"
    if str(outcome) == "timeout":         # it ran out of time in the course; nothing was struck
        hit = "time limit"
        crash_lab = f"ROS 2 {hr:.0f} Hz · short of G{ng + 1} at the time limit"
    else:
        crash_lab = f"ROS 2 {hr:.0f} Hz · hits {hit} after G{ng}" if ng else f"ROS 2 {hr:.0f} Hz · hits {hit} before G1"
    moments = [("ROS", m1, m1_lab), ("ROS", nr - 1, crash_lab), ("XPU", step_near, f"XPU-RT {hx:.0f} Hz · clears crate {clear:.2f} m"), ("XPU", int(0.96 * nx), f"XPU-RT {hx:.0f} Hz · reaches gate")]
    return moments, (xxyz[step_near, :3], near_bin, clear), hit


def eff_hz(Z):
    return float(Z["eff_cmd_hz"]) if "eff_cmd_hz" in Z.files else None


def sec(ax, s, fz, dx=0, dy=18):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontsize=fz["badge"], weight="bold", color="white",
                ha="center", va="center", zorder=40, annotation_clip=False, bbox=dict(boxstyle="circle,pad=0.32", fc="#2f6db0", ec="white", lw=1.8))


def rescale_fonts(ax, k, row_label_pt=None, tick_pt=None):
    """scale the Gantt's hard-coded sizes; rotated row labels get an absolute size so four rows fit."""
    for t in ax.texts + [ax.xaxis.label, ax.yaxis.label]:
        t.set_fontsize(row_label_pt if (row_label_pt and t.get_rotation() == 90) else t.get_fontsize() * k)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(tick_pt or t.get_fontsize() * k)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(t.get_fontsize() * k)


# ---------------------------------------------------------------------------------------------------- main
def paper_form(a, fz, base, X, R, xxyz, rxyz, xt, rt, tnorm, people, gates, hx, hr, moments, near_miss, hit, bg, bg_note, bg_std, rows, cells, cells_rich, b_arms, board_arms, fam, n_set_aside, n_set_aside_rich, xpu_dir, ros_dir, display_cruise, out, gain_note):
    """The submitted figure's skeleton with the measured content: A top-down | [B envelope; C generalisation | D mechanism],
    a–d strips, E body rate | F camera rate on the K1 | G where the work lands | H added load, I the measured Gantt rows."""
    H_in = a.width_in * 0.80
    fig = plt.figure(figsize=(a.width_in, H_in))
    outer = fig.add_gridspec(7, 1, height_ratios=[6.3, 0.8, 2.5, 0.9, 3.2, 1.2, 5.2], hspace=0.0, left=0.045, right=0.99, top=0.965, bottom=0.04)
    ph = float(X["person_h"]) if "person_h" in X.files else None
    # row 0: A | [B ; C | D]
    r0 = outer[0].subgridspec(1, 2, width_ratios=[3.6, 2.9], wspace=0.10); axA = fig.add_subplot(r0[0])
    rcol = r0[1].subgridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.62); axB = fig.add_subplot(rcol[0])
    brow = rcol[1].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.48); axC = fig.add_subplot(brow[0]); axD = fig.add_subplot(brow[1])
    S.draw_topdown(axA, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, a.path_start, rxyz[-1], moments, ov_obj=None, near_miss=near_miss)
    xg, rg = int(X["gates_passed"]) if "gates_passed" in X.files else FC.fallback("xpu_gates", 4, "XPU-RT dump lacks gates_passed"), int(R["gates_passed"]) if "gates_passed" in R.files else FC.fallback("ros_gates", 1, "ROS 2 dump lacks gates_passed")
    axA.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=f"XPU-RT · CP-SAT ({hx:.1f} Hz) · {xg} of 4 gates (colour = time)"),
                        Line2D([0], [0], color=C_ROS, lw=5, label=f"ROS 2 vanilla ({hr:.1f} Hz) · {rg} of 4, then hits {hit}"),
                        Line2D([0], [0], marker="o", color=C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")], loc="upper left", fontsize=fz["leg"], framealpha=0.93, ncol=2, handlelength=1.9)
    axA.set_title(f"Warehouse gate-course showdown — same scene and controller ({display_cruise:.1f} m/s" + (f", people {ph:.1f} m" if ph else "") + ")\n"
                  f"only what the K1 delivers differs: " + (gain_note if len(gain_note) < 90 else f"control every {1000 / hx:.1f} vs {1000 / hr:.1f} ms, latency and goal cadence replayed"),
                  fontsize=fz["title"] * 1.05, weight="bold", loc="left")
    draw_B(axB, cells, b_arms, fz, display_cruise); axB.set_title("Flight envelope: success against cruise speed, each arm at its own K1 cadence\n" + ("and gain 0.5 / control rate (same seeds; Wilson 95 %)" if a.cell == "cal17" else "and camera→control latency (same seeds and replicates; Wilson 95 %)"), fontsize=fz["title"], weight="bold", loc="left")
    Dp = draw_Dprime(axC, rows, fz, families=fam, gain=a.gain); axC.set_title("Generalisation: environments\n(1.0–1.4 m/s pooled, one flight per seed)", fontsize=fz["title"], weight="bold", loc="left")
    draw_mechanism(axD, fs=base / 10.5, compact=True); axD.set_title("Mechanism: the baseline\nthrashes (moment, power)", fontsize=fz["title"], weight="bold", loc="left")
    sec(axA, "A", fz, dx=-30, dy=-24); sec(axB, "B", fz, dx=-8, dy=30); sec(axC, "C", fz, dx=-8, dy=30); sec(axD, "D", fz, dx=-8, dy=30)
    # row 2: strips a–d
    bgrid = outer[2].subgridspec(1, 4, wspace=0.09)
    for c, (src, step, lab) in enumerate(moments):
        dd = ros_dir if src == "ROS" else xpu_dir; Z = R if src == "ROS" else X; tt = (rt if src == "ROS" else xt)[min(step, len(Z["t_s"]) - 1)]
        f = S.frame_at(dd, Z["frame_steps"], step)
        col = bgrid[c].subgridspec(2, 3, height_ratios=[2.4, 14], width_ratios=[1.15, 1.55, 1.15], hspace=0.02, wspace=0.05); tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off")
        axh.scatter([0.02], [0.72], s=fz["badge"] * 14, marker="o", facecolors="black", edgecolors=tc, linewidths=2.2, transform=axh.transAxes, clip_on=False, zorder=5)
        axh.text(0.02, 0.72, chr(ord("a") + c), color="white", fontsize=fz["tiny"], weight="bold", ha="center", va="center", transform=axh.transAxes, zorder=6)
        axh.text(0.065, 0.72, f"{lab} · t={tt:.1f}s", fontsize=fz["tick"], weight="bold", color=tc, va="center", transform=axh.transAxes)
        ac = fig.add_subplot(col[1, 0]); ac.imshow(f["chase"][225:465, 415:655]); ac.axis("off"); ac.set_title("chase", fontsize=fz["tick"])
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = S.YOLO.get(int(cls), ("obj", "#39f")); af.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cc, lw=2.2))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=fz["tick"])
        at = fig.add_subplot(col[1, 2]); S.cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=fz["tick"])
    # row 4: E body rate | F camera rate | G where the work lands | H added load
    r4 = outer[4].subgridspec(1, 4, width_ratios=[0.9, 1.3, 1.4, 1.35], wspace=0.42)
    axE = fig.add_subplot(r4[0]); fgs = r4[1].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=0.16); axF1 = fig.add_subplot(fgs[0]); axF2 = fig.add_subplot(fgs[1], sharex=axF1)
    axG = fig.add_subplot(r4[2]); hgs = r4[3].subgridspec(4, 1, hspace=0.32); axH = [fig.add_subplot(hgs[i]) for i in range(4)]
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    axE.plot(xt, S.smooth(xw), color=C_XPU, lw=2.0, label="XPU-RT"); axE.plot(rt, S.smooth(rw), color=C_ROS, lw=2.0, label="ROS 2")
    axE.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); axE.axvline(rt[-1], color=C_ROS, lw=1.6, ls=(0, (4, 2)), alpha=0.85)
    axE.text(rt[-1] + 0.2, 0.93, "ROS 2 ✗ crashes", color=C_ROS, fontsize=fz["tiny"], weight="bold", va="top", ha="left", transform=axE.get_xaxis_transform())
    axE.set_xlabel("time (s)", fontsize=fz["lab"]); axE.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=fz["lab"]); axE.legend(fontsize=fz["leg"], frameon=False); axE.tick_params(labelsize=fz["tick"]); axE.grid(ls=":", lw=0.6, color="#d4d1cb")
    axE.set_title("Body-rate magnitude\nof the display flights", fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        axE.spines[sp].set_visible(False)
    F = draw_F(axF1, axF2, board_arms, fz)
    G = draw_G(axG, [("XPU-RT · CP-SAT", C_XPU, "xpu", ["acpsat_hardr1", "acpsat_hardr2", "acpsat_hardr3"]), ("XPU-RT · greedy", C_XPU2, "xpu", ["agreedyr1", "agreedyr2", "agreedyr3"]),
                     ("ROS 2 on all 8 cores\n(two YOLO nodes)", C_ROS8, "ros", ["45_vanilla4x2_r1", "45_vanilla4x2_r2", "45_vanilla4x2_r3"]), ("ROS 2 vanilla\n(4-hart YOLO pool)", C_ROS, "ros", ["45_vanilla4_r1", "45_vanilla4_r2", "45_vanilla4_r3"])], fz)
    Hm = draw_H(axH, cells_rich, fz)
    sec(axE, "E", fz, dx=-8, dy=28); sec(axF1, "F", fz, dx=-8, dy=28); sec(axG, "G", fz, dx=-8, dy=28); sec(axH[0], "H", fz, dx=-8, dy=28)
    # row 6: I Gantt
    axI = fig.add_subplot(outer[6]); cols = {"xpu": C_XPU, "xpu2": C_XPU2, "ros": C_ROS, "ros8": C_ROS8}
    grows, gpaths, gside = [], [], []
    for r in a.gantt_rows:
        name, label, ck = r.split(":"); p = f"{a.gantt_prefix}_{name}.json"
        if os.path.exists(p):
            grows.append((json.load(open(p)), label, cols[ck], "ros" if ck.startswith("ros") else "xpu")); gpaths.append(p); gside.append(p.replace(".json", "_metrics.json"))
    S.draw_combined_gantt(axI, grows, gpaths); gtitle = axI.get_title(loc="left")
    rescale_fonts(axI, base / 16.0, row_label_pt=fz["tick"] * 0.74, tick_pt=fz["tick"])
    for t in axI.texts:
        if t.get_rotation() == 90 and t.get_position()[0] < -5:
            t.set_visible(False)
    axI.get_legend().set_bbox_to_anchor((0.5, -0.14)); axI.get_legend().set_loc("upper center")
    parts = [p.strip() for p in re.split(r"  ·  |\s—\s", gtitle)]
    parts = [re.sub(r"(\d+) of (\d+) frames late", r"\1/\2 late", p).replace("camera→control ", "").replace("control every ", "control ") for p in parts]
    axI.set_title("\n".join([parts[0] + " — " + "  ·  ".join(parts[1:3]), "  ·  ".join(parts[3:])]) if len(parts) > 3 else gtitle, fontsize=fz["title"], weight="bold", loc="left", pad=fz["title"] * 0.9)
    sec(axI, "I", fz, dx=-8, dy=30)
    fig.savefig(out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); print("wrote", out + ".png/.pdf")
    def cellvals(tr):
        return {f"Cruise{word(c)}": [sum(x["outcome"] == "success" for x in v), len(v)] for (t, c), v in sorted(cells.items()) if t == tr}
    side = {"figure": out + ".png", "written": datetime.datetime.now().isoformat(timespec="seconds"), "form": "paper",
            "variant": {"cell": a.cell, "tuned": a.tuned, "display_cruise": display_cruise, "gain": a.gain, "n_rule": "equal replicates per seed across the arms drawn", "flights_set_aside": n_set_aside, "rich_flights_set_aside": n_set_aside_rich},
            "canvas": {"width_in": a.width_in, "height_in": H_in, "dpi": a.dpi, "min_print_pt": a.min_print_pt, "base_pt": round(base, 2)},
            "sources": {"xpu_dir": xpu_dir, "ros_dir": ros_dir, "campaigns": sorted({r["camp"] for r in rows}), "gantt_sidecars": gside, "energy_csv": os.environ["ENERGY_CSV"], "scene_records": a.scene_records},
            "A": {"backdrop": bg_note, "plate_std": round(bg_std, 1), "xpu_eff_hz": hx, "ros_eff_hz": hr, "xpu_gates": xg, "ros_gates": rg, "ros_hit": hit, "ros_outcome": str(R["outcome"]) if "outcome" in R.files else None},
            "B": {re.sub(r"[^A-Za-z]", "", FLIGHT_ARMS[t][0]): cellvals(t) for t in b_arms}, "C": {}, "Dprime": Dp,
            "F": {re.sub(r"[^A-Za-z]", "", lab): {f"Hz{word(h)}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for h, p in pts.items()} for lab, pts in F.items()},
            "G": {re.sub(r"[^A-Za-z]", "", k.replace("\n", " ")): v for k, v in G.items()}, "H": Hm, "J": {}, "K": {},
            **FC.sidecar_common("showdown_v3_figure", LOADED_CSVS + [os.path.join(RES, "ros_traced", "summary.csv")] + list(gside))}
    json.dump(side, open(out + "_metrics.json", "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="tall1005", help="tall1005 | tall1008 | cal17 (which display pair)")
    ap.add_argument("--tuned", default="board", choices=["board", "inb", "none"])
    ap.add_argument("--xpu-dir"); ap.add_argument("--ros-dir"); ap.add_argument("--scene-records", default="")
    ap.add_argument("--display-cruise", type=float, default=None); ap.add_argument("--scene-course", default="a"); ap.add_argument("--scene-density", type=float, default=0.30)
    ap.add_argument("--gain", type=float, default=0.0055, help="the fixed gain of the flight cells drawn in B/C/D'")
    ap.add_argument("--gantt-prefix", default=os.path.join(REPO, "schedules", "measured_gantt_v3"))
    ap.add_argument("--gantt-rows", nargs="+", default=["xpu:CP-SAT:xpu", "xpu2:greedy:xpu2", "ros8:ROS 2 8-core:ros8", "ros:ROS 2 vanilla:ros"])
    # canvas convention of the submitted figure: 27 in wide, 14 pt type (≈ 3.7 pt at the 7.1 in print width, read zoomed)
    ap.add_argument("--width-in", type=float, default=26.0); ap.add_argument("--min-print-pt", type=float, default=3.6); ap.add_argument("--print-width-in", type=float, default=7.1)
    ap.add_argument("--dpi", type=int, default=300); ap.add_argument("--path-start", type=int, default=85)
    ap.add_argument("--out", default=None); ap.add_argument("--companion", action="store_true", help="also render the column-width companion (F + H + J)")
    ap.add_argument("--paper-form", action="store_true", help="the submitted figure's skeleton: top-down + envelope/generalisation/mechanism column, strips, a four-panel row, the Gantt")
    a = ap.parse_args()
    cells_dir = {"tall1005": ("campaign_v2/display_same/xpu_s1005_figdata", "campaign_v2/display_same/ros_s1005_figdata", 1.0),
                 "tall1008": ("campaign_v2/display_v3/xpu_s1008_figdata", "campaign_v2/display_v3/ros_s1008_figdata", 1.2),
                 "cal17": ("campaign_v2/display_same_cal17/xpu_figdata", "campaign_v2/display_same_cal17/ros_figdata", 1.2)}
    xd, rd, dc = cells_dir.get(a.cell, (None, None, 1.0))
    xpu_dir = a.xpu_dir or os.path.join(RES, xd); ros_dir = a.ros_dir or os.path.join(RES, rd); display_cruise = a.display_cruise or dc
    for d in (xpu_dir, ros_dir):
        if not os.path.exists(os.path.join(d, "figure_data.npz")):
            raise SystemExit(f"display dump missing: {d}")
    out = a.out or os.path.join(RES, "refined", f"warehouse_showdown_v3_{a.cell}_{a.tuned}")
    base = a.min_print_pt * a.width_in / a.print_width_in           # canvas pt that prints at min-print-pt
    fz = dict(tiny=base * 0.9, tick=base, lab=base * 1.05, leg=base * 0.95, title=base * 1.2, badge=base * 1.1, head=base * 1.35)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})

    X, R = S.load(xpu_dir), S.load(ros_dir)
    xxyz, rxyz, xt, rt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"], R["t_s"]
    tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx, hr = eff_hz(X) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"), eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    moments, near_miss, hit = moments_for(X, R, a.path_start, hx, hr)
    bg, bg_note, bg_std = backdrop(X)
    rows = load_campaigns()
    if a.cell == "cal17":                                   # the 1.7 m scene, cadence only, gain 0.5 / control rate per arm
        rows += load_extra_csv(os.path.join(RES, "campaign_v2", "campaign_v2.csv"), "campaign_v2")
        cells = flight_cells(rows, "a", 0.30, latency=False, gain_by_trace={"xpu_a_cpsat_hard.csv": 0.0052, "ros_vanilla445.csv": 0.01277}, camps={"campaign_v2"})
        gain_note = "gain 0.5 / control rate per arm (XPU-RT 0.0052, ROS 2 0.0128), cadence replayed"
    else:
        cells = flight_cells(rows, a.scene_course, a.scene_density, a.gain)
        gain_note = f"control every {1000 / hx:.1f} vs {1000 / hr:.1f} ms, camera→control latency and goal cadence, replayed"
    cells_rich = {k: v for k, v in flight_cells(rows, "a", 0.30, 0.0055).items() if k[0].startswith(("xpu_b5", "ros_rvanilla"))}
    b_arms = ["xpu_a_cpsat_hard.csv", "xpu_a_greedy.csv", "ros_vanilla445.csv", "ros_rvanilla445.csv", "ros_vanilla4x245.csv"] + (["ros_p345.csv"] if a.tuned == "inb" else [])
    b_arms = [t for t in b_arms if any(k[0] == t for k in cells)]
    cells, n_set_aside = equalise(cells, b_arms)                        # same seeds, same replicates, every arm
    cells_rich, n_set_aside_rich = equalise(cells_rich, [t for t in ("xpu_b5_cpsat.csv", "xpu_b5_greedy.csv", "ros_rvanilla445.csv", "ros_rvanilla490.csv") if any(k[0] == t for k in cells_rich)])
    board_arms = [b for b in BOARD_ARMS if a.tuned != "none" or b[2][1] != "p3"]
    fam = ("xpu", "ros") + (("p3",) if a.tuned == "inb" else ())

    if a.paper_form:
        return paper_form(a, fz, base, X, R, xxyz, rxyz, xt, rt, tnorm, people, gates, hx, hr, moments, near_miss, hit, bg, bg_note, bg_std, rows, cells, cells_rich, b_arms, board_arms, fam, n_set_aside, n_set_aside_rich, xpu_dir, ros_dir, display_cruise, out, gain_note)
    H_in = a.width_in * 0.98
    fig = plt.figure(figsize=(a.width_in, H_in))
    outer = fig.add_gridspec(9, 1, height_ratios=[5.2, 1.0, 2.3, 1.15, 3.6, 1.5, 3.8, 2.0, 4.8], hspace=0.0, left=0.045, right=0.99, top=0.955, bottom=0.045)
    ph = float(X["person_h"]) if "person_h" in X.files else None
    fig.text(0.045, 0.985, f"Warehouse gate course on the K1 — same scene and controller for both runtimes at {display_cruise:.1f} m/s" + (f", people {ph:.1f} m" if ph else "")
             + f"; only what the board delivers differs — {gain_note}",
             fontsize=fz["head"], weight="bold", va="top", ha="left")
    # row 0: A | B | C
    r0 = outer[0].subgridspec(1, 3, width_ratios=[3.4, 1.7, 1.35], wspace=0.18)
    axA = fig.add_subplot(r0[0]); axB = fig.add_subplot(r0[1]); axC = fig.add_subplot(r0[2])
    S.draw_topdown(axA, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, a.path_start, rxyz[-1], moments, ov_obj=None, near_miss=near_miss)
    xg, rg = int(X["gates_passed"]) if "gates_passed" in X.files else FC.fallback("xpu_gates", 4, "XPU-RT dump lacks gates_passed"), int(R["gates_passed"]) if "gates_passed" in R.files else FC.fallback("ros_gates", 1, "ROS 2 dump lacks gates_passed")
    axA.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=f"XPU-RT · CP-SAT ({hx:.1f} Hz) · {xg} of 4 gates (colour = time)"),
                        Line2D([0], [0], color=C_ROS, lw=5, label=f"ROS 2 vanilla ({hr:.1f} Hz) · {rg} of 4, then hits {hit}"),
                        Line2D([0], [0], marker="o", color=C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")], loc="upper left", fontsize=fz["leg"], framealpha=0.93, ncol=2, handlelength=1.9)
    axA.set_title(f"Top-down: XPU-RT CP-SAT clears the course; ROS 2 vanilla clears G{rg} and hits a {hit}", fontsize=fz["title"], weight="bold", loc="left")
    draw_B(axB, cells, b_arms, fz, display_cruise, replay="cadence, gain 0.5 / rate per arm" if a.cell == "cal17" else "cadence and latency")
    C_hist = draw_C(axC, cells, b_arms, display_cruise, fz)
    sec(axA, "A", fz, dx=-30, dy=-24); sec(axB, "B", fz, dx=-8, dy=30); sec(axC, "C", fz, dx=-8, dy=46)
    # row 2: strips a–d
    bgrid = outer[2].subgridspec(1, 4, wspace=0.09)
    for c, (src, step, lab) in enumerate(moments):
        dd = ros_dir if src == "ROS" else xpu_dir; Z = R if src == "ROS" else X; tt = (rt if src == "ROS" else xt)[min(step, len(Z["t_s"]) - 1)]
        f = S.frame_at(dd, Z["frame_steps"], step)
        col = bgrid[c].subgridspec(2, 3, height_ratios=[2.4, 14], width_ratios=[1.15, 1.55, 1.15], hspace=0.02, wspace=0.05); tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off")
        axh.scatter([0.02], [0.72], s=fz["badge"] * 14, marker="o", facecolors="black", edgecolors=tc, linewidths=2.2, transform=axh.transAxes, clip_on=False, zorder=5)
        axh.text(0.02, 0.72, chr(ord("a") + c), color="white", fontsize=fz["tiny"], weight="bold", ha="center", va="center", transform=axh.transAxes, zorder=6)
        axh.text(0.065, 0.72, f"{lab} · t={tt:.1f}s", fontsize=fz["tick"], weight="bold", color=tc, va="center", transform=axh.transAxes)
        ac = fig.add_subplot(col[1, 0]); ac.imshow(f["chase"][225:465, 415:655]); ac.axis("off"); ac.set_title("chase", fontsize=fz["tick"])
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = S.YOLO.get(int(cls), ("obj", "#39f")); af.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cc, lw=2.2))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=fz["tick"])
        at = fig.add_subplot(col[1, 2]); S.cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=fz["tick"])
    # row 4 — why, on the board: D camera-rate envelope | E where the work lands | F added load
    r4 = outer[4].subgridspec(1, 3, width_ratios=[1.45, 1.25, 1.45], wspace=0.30)
    fgs = r4[0].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=0.16); axF1 = fig.add_subplot(fgs[0]); axF2 = fig.add_subplot(fgs[1], sharex=axF1)
    axG = fig.add_subplot(r4[1])
    hgs = r4[2].subgridspec(4, 1, hspace=0.32); axH = [fig.add_subplot(hgs[i]) for i in range(4)]
    F = draw_F(axF1, axF2, board_arms, fz)
    G = draw_G(axG, [("XPU-RT · CP-SAT", C_XPU, "xpu", ["acpsat_hardr1", "acpsat_hardr2", "acpsat_hardr3"]), ("XPU-RT · greedy", C_XPU2, "xpu", ["agreedyr1", "agreedyr2", "agreedyr3"]),
                     ("ROS 2 on all 8 cores\n(two YOLO nodes)", C_ROS8, "ros", ["45_vanilla4x2_r1", "45_vanilla4x2_r2", "45_vanilla4x2_r3"]), ("ROS 2 vanilla\n(4-hart YOLO pool)", C_ROS, "ros", ["45_vanilla4_r1", "45_vanilla4_r2", "45_vanilla4_r3"])], fz)
    Hm = draw_H(axH, cells_rich, fz)
    sec(axF1, "D", fz, dx=-8, dy=26); sec(axG, "E", fz, dx=-8, dy=26); sec(axH[0], "F", fz, dx=-8, dy=26)
    # row 6 — how it generalises: G the display scene, every run | H environments | I mechanism
    r6 = outer[6].subgridspec(1, 3, width_ratios=[2.3, 1.35, 0.95], wspace=0.26)
    axK = fig.add_subplot(r6[0]); axDp = fig.add_subplot(r6[1]); axD = fig.add_subplot(r6[2])
    scene_dirs = []
    if a.scene_records and os.path.isdir(a.scene_records):
        for lab, col, sub in [("XPU-RT · CP-SAT", C_XPU, "xpu_cpsat"), ("ROS 2 vanilla", C_ROS, "ros_vanilla"), ("XPU-RT · greedy", C_XPU2, "xpu_greedy")]:
            scene_dirs.append((lab, col, os.path.join(a.scene_records, sub)))
    K = draw_K(axK, X, R, scene_dirs, fz, f"ROS 2 vanilla ({hr:.0f} Hz)", f"XPU-RT CP-SAT ({hx:.0f} Hz)")
    Dp = draw_Dprime(axDp, rows, fz, families=fam, gain=a.gain)
    draw_mechanism(axD, fs=base / 10.5, compact=True); axD.set_title("Mechanism: the baseline thrashes\n(commanded moment, modelled power)", fontsize=fz["title"], weight="bold", loc="left")
    J = {}                                                       # the feedback drift lives in the companion
    sec(axK, "G", fz, dx=-8, dy=28); sec(axDp, "H", fz, dx=-8, dy=28); sec(axD, "I", fz, dx=-8, dy=28)
    # row 8: I
    axI = fig.add_subplot(outer[8])
    cols = {"xpu": C_XPU, "xpu2": C_XPU2, "ros": C_ROS, "ros8": C_ROS8}
    grows, gpaths, gside = [], [], []
    for r in a.gantt_rows:
        name, label, ck = r.split(":"); p = f"{a.gantt_prefix}_{name}.json"
        if not os.path.exists(p):
            continue
        grows.append((json.load(open(p)), label, cols[ck], "ros" if ck.startswith("ros") else "xpu")); gpaths.append(p); gside.append(p.replace(".json", "_metrics.json"))
    S.draw_combined_gantt(axI, grows, gpaths); gtitle = axI.get_title(loc="left")
    rescale_fonts(axI, base / 16.0, row_label_pt=fz["tick"] * 0.74, tick_pt=fz["tick"])
    for t in axI.texts:                                         # the per-row placement notes are in the title; the lane labels stay
        if t.get_rotation() == 90 and t.get_position()[0] < -5:
            t.set_visible(False)
    axI.get_legend().set_bbox_to_anchor((0.5, -0.14)); axI.get_legend().set_loc("upper center")   # legend under the time axis
    parts = [p.strip() for p in re.split(r"  ·  |\s—\s", gtitle)]          # the board line, then one clause per row
    parts = [re.sub(r"(\d+) of (\d+) frames late", r"\1/\2 late", p).replace("camera→control ", "").replace("control every ", "control ") for p in parts]
    axI.set_title("\n".join([parts[0] + " — " + "  ·  ".join(parts[1:3]), "  ·  ".join(parts[3:])]) if len(parts) > 3 else gtitle,
                  fontsize=fz["title"], weight="bold", loc="left", pad=fz["title"] * 0.9)
    sec(axI, "J", fz, dx=-8, dy=30)
    fig.savefig(out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".png/.pdf")

    # sidecar: every number on the page, grid-shaped for emit_figure_numbers.py
    def cellvals(tr):
        return {f"Cruise{word(c)}": [sum(x["outcome"] == "success" for x in v), len(v)] for (t, c), v in sorted(cells.items()) if t == tr}
    side = {"figure": out + ".png", "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "variant": {"cell": a.cell, "tuned": a.tuned, "display_cruise": display_cruise, "gain": a.gain, "n_rule": "equal replicates per seed across the arms drawn", "flights_set_aside": n_set_aside, "rich_flights_set_aside": n_set_aside_rich},
            "canvas": {"width_in": a.width_in, "height_in": H_in, "dpi": a.dpi, "min_print_pt": a.min_print_pt, "base_pt": round(base, 2)},
            "sources": {"xpu_dir": xpu_dir, "ros_dir": ros_dir, "campaigns": sorted({r["camp"] for r in rows}), "gantt_sidecars": gside, "energy_csv": os.environ["ENERGY_CSV"], "scene_records": a.scene_records},
            "A": {"backdrop": bg_note, "plate_std": round(bg_std, 1), "xpu_eff_hz": hx, "ros_eff_hz": hr, "xpu_gates": xg, "ros_gates": rg, "ros_hit": hit, "ros_outcome": str(R["outcome"]) if "outcome" in R.files else None},
            "B": {re.sub(r"[^A-Za-z]", "", FLIGHT_ARMS[t][0]): cellvals(t) for t in b_arms}, "C": C_hist, "Dprime": Dp,
            "F": {re.sub(r"[^A-Za-z]", "", lab): {f"Hz{word(h)}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for h, p in pts.items()} for lab, pts in F.items()},
            "G": {re.sub(r"[^A-Za-z]", "", k.replace("\n", " ")): v for k, v in G.items()}, "H": Hm, "J": J, "K": K,
            **FC.sidecar_common("showdown_v3_figure", LOADED_CSVS + [os.path.join(RES, "ros_traced", "summary.csv")] + list(gside))}
    json.dump(side, open(out + "_metrics.json", "w"), indent=1)
    if a.companion:
        cfig = plt.figure(figsize=(a.width_in * 0.72, a.width_in * 0.26))
        cg = cfig.add_gridspec(1, 3, width_ratios=[1.45, 1.5, 1.0], wspace=0.42, left=0.05, right=0.99, top=0.84, bottom=0.14)
        f1 = cg[0].subgridspec(2, 1, hspace=0.12); c1 = cfig.add_subplot(f1[0]); c2 = cfig.add_subplot(f1[1], sharex=c1)
        h1 = cg[1].subgridspec(4, 1, hspace=0.32); ch = [cfig.add_subplot(h1[i]) for i in range(4)]; cj = cfig.add_subplot(cg[2])
        draw_F(c1, c2, board_arms, fz); draw_H(ch, cells_rich, fz); draw_J(cj, fz)
        for ax_, s_ in ((c1, "(a)"), (ch[0], "(b)"), (cj, "(c)")):
            ax_.annotate(s_, xy=(0, 1), xycoords="axes fraction", xytext=(-46, 30), textcoords="offset points", fontsize=fz["head"], weight="bold", ha="left", va="bottom", annotation_clip=False)
        cout = out.replace("warehouse_showdown_v3", "hil_envelope_story_v3")
        cfig.savefig(cout + ".png", dpi=a.dpi, bbox_inches="tight"); cfig.savefig(cout + ".pdf", bbox_inches="tight"); print("wrote", cout + ".png/.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
