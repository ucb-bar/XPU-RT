#!/usr/bin/env python3
# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Fine-grain latency response across all FOUR environments, two embodiments.

Keeps the established encoding:
  colour     category   ideal / HW accelerated (CPU+DSP+HTA) / CPU only
  marker     schedule   star = none, circle = pipelined, square = serial
  linestyle  ENVIRONMENT, labelled directly at the right of each curve

Panel 1  marginal success, DESIGN-CORRECTED intervals (a seed is a replicate over
         the SAME 24 configs, so a binomial overstates precision by
         DEFF = 1+(m-1)*ICC; google_robot ICCs run to 0.58, DEFF 6.3).
Panel 2  the PAIRED per-seed penalty against each env's own zero-latency arm.
         This is the comparable quantity: it cancels config difficulty and each
         env's own baseline. It carries the headline -- the three GRASPING tasks
         collapse, the one PUSHING task barely moves.

CAVEAT drawn on the figure: the widowx arms actuate on a fine 40 ms grid, the
google_robot arms on their native 333 ms grid (their planner-interpolated
controller cannot be driven at a fine tick -- see GOOGLE_ROBOT_PORT.md). A
latency change therefore only alters the google_robot command sequence when an
arrival crosses a control boundary, which damps their measured sensitivity. The
grasp-vs-push reading is better supported than a raw embodiment comparison.

All numbers recomputed from g5fine/runs/ at plot time; nothing hardcoded.
"""
from __future__ import annotations
import json, glob, math, collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
RUNS = HERE / "g5fine" / "runs"
OUT = HERE / "finegrain_four_tasks.png"

ARMS = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
LAT = {"lat0": 0.0, "pipe110": 117.7, "pipe200": 231.8,
       "serial283": 283.4, "fp32_555": 555.0, "cpu685": 684.8}
CAT = {"lat0": "ideal", "pipe110": "HW accelerated", "pipe200": "HW accelerated",
       "serial283": "HW accelerated", "fp32_555": "CPU only", "cpu685": "CPU only"}
SCHED = {"lat0": "none", "pipe110": "pipelined", "pipe200": "pipelined",
         "serial283": "serial", "fp32_555": "serial", "cpu685": "serial"}
CAT_COLOUR = {"ideal": "#7f8c8d", "HW accelerated": "#1a9e8f", "CPU only": "#c0392b"}
SCHED_MARKER = {"none": "*", "pipelined": "o", "serial": "s"}

#      key,     label,                      linestyle, filled, skill
#      key,     label,                      ls,   filled, skill,  label dy (panel1, panel2)
ENVS = [("egg",    "eggplant (widowx)",      "-",   True,  "grasp",  (7, -4)),
        ("spoon",  "spoon (widowx)",         "--",  False, "grasp",  (-9, 4)),
        ("coke",   "coke can (google)",      "-.",  True,  "grasp",  (0, 0)),
        ("drawer", "close drawer (google)",  ":",   False, "push",   (0, 0))]

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def load():
    per_ep, per_seed = collections.defaultdict(list), collections.defaultdict(dict)
    for f in glob.glob(str(RUNS / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name
        if "_rng" not in name: continue
        head, seed = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        eps = d.get("episodes", [])
        if not eps: continue
        for e in eps:
            per_ep[(task, arm)].append((e["episode_id"], bool(e["success"])))
        per_seed[(task, arm)][int(seed)] = 100.0*sum(e["success"] for e in eps)/len(eps)
    return per_ep, per_seed


def design_ci(rows):
    by = collections.defaultdict(list)
    for ep, ok in rows: by[ep].append(ok)
    n, K = len(rows), len(by)
    p = sum(ok for _, ok in rows)/n
    if K < 2: return 100*p, 100*p, 100*p
    ns = [len(v) for v in by.values()]; m = sum(ns)/K
    msb = sum(len(v)*(sum(v)/len(v) - p)**2 for v in by.values())/(K-1)
    msw = sum(sum((x - sum(v)/len(v))**2 for x in v) for v in by.values())/max(n-K, 1)
    den = msb + (m-1)*msw
    icc = max((msb-msw)/den, 0.0) if den > 0 else 0.0
    se = math.sqrt(max(p*(1-p), 1e-12)/(n/(1+(m-1)*icc)))
    return 100*p, 100*max(p-1.96*se, 0), 100*min(p+1.96*se, 1)


def paired(per_seed, task, arm):
    a, b = per_seed.get((task, arm), {}), per_seed.get((task, "lat0"), {})
    common = sorted(set(a) & set(b))
    if not common: return None
    d = [a[s]-b[s] for s in common]
    mu = sum(d)/len(d)
    sd = math.sqrt(sum((x-mu)**2 for x in d)/max(len(d)-1, 1))
    se = sd/math.sqrt(len(d)) if len(d) > 1 else 0.0
    tc = TCRIT.get(len(d)-1, 1.96)
    return mu, mu-tc*se, mu+tc*se


per_ep, per_seed = load()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.0), dpi=130,
                              gridspec_kw={"wspace": 0.20})

for key, label, ls, filled, skill, dy in ENVS:
    xs, ys = [], []
    for arm in ARMS:
        rows = per_ep.get((key, arm))
        if not rows: continue
        p, lo, hi = design_ci(rows)
        c = CAT_COLOUR[CAT[arm]]
        ax.errorbar(LAT[arm], p, yerr=[[p-lo], [hi-p]], fmt="none", ecolor=c,
                    elinewidth=1.3, capsize=3.5, alpha=0.6, zorder=3)
        ax.scatter(LAT[arm], p, s=240 if SCHED[arm] == "none" else 125,
                   marker=SCHED_MARKER[SCHED[arm]],
                   facecolor=c if filled else "white",
                   edgecolor=c if not filled else "black",
                   linewidth=1.7 if not filled else 0.7, zorder=4)
        xs.append(LAT[arm]); ys.append(p)
    ax.plot(xs, ys, ls=ls, lw=1.3, color="#555", alpha=0.55, zorder=2)
    ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -2 + dy[0]),
                fontsize=8.5, color="#333", fontweight="bold", va="center")

ax.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax.set_ylabel("success rate (%)   ·   n=240 per point (10 seeds x 24 configs)")
ax.set_title("Marginal success, design-corrected intervals", fontsize=11.5)
ax.grid(True, alpha=0.3, zorder=0); ax.set_xlim(-45, 980); ax.set_ylim(-5, 85)
handles = [Line2D([], [], ls="", marker="o", ms=10, color=c, mec="black", label=k)
           for k, c in CAT_COLOUR.items()]
handles += [Line2D([], [], ls="", marker=m, ms=10, color="#444", mec="black",
                   label=f"schedule: {k}") for k, m in SCHED_MARKER.items()]
ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.95, ncol=2)

# ---- panel 2: the paired penalty -- the comparable quantity ----------------
ax2.axhline(0, ls="--", lw=1.2, color="#555", zorder=1)
for key, label, ls, filled, skill, dy in ENVS:
    xs, ys = [], []
    for arm in ARMS:
        if arm == "lat0": continue
        r = paired(per_seed, key, arm)
        if not r: continue
        mu, lo, hi = r
        c = CAT_COLOUR[CAT[arm]]
        ax2.errorbar(LAT[arm], mu, yerr=[[mu-lo], [hi-mu]], fmt="none", ecolor=c,
                     elinewidth=1.3, capsize=3.5, alpha=0.6, zorder=3)
        ax2.scatter(LAT[arm], mu, s=125, marker=SCHED_MARKER[SCHED[arm]],
                    facecolor=c if filled else "white",
                    edgecolor=c if not filled else "black",
                    linewidth=1.7 if not filled else 0.7, zorder=4)
        xs.append(LAT[arm]); ys.append(mu)
    lw = 2.4 if skill == "push" else 1.3
    ax2.plot(xs, ys, ls=ls, lw=lw, color="#2c3e50" if skill == "push" else "#555",
             alpha=0.9 if skill == "push" else 0.55, zorder=2)
    ax2.annotate(f"{label}\n({skill})", (xs[-1], ys[-1]), textcoords="offset points",
                 xytext=(8, dy[1]), fontsize=8.5, va="center", fontweight="bold",
                 color="#2c3e50" if skill == "push" else "#333")

ax2.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax2.set_ylabel("paired penalty vs that env's own zero-latency arm (points)")
ax2.set_title("Latency PENALTY, paired per seed (t intervals, df=9)\n"
              "the three GRASPING tasks collapse; the one PUSHING task barely moves",
              fontsize=11.5)
ax2.grid(True, alpha=0.3, zorder=0); ax2.set_xlim(-45, 980)

fig.text(0.5, -0.035,
         "CAVEAT: widowx arms actuate on a fine 40 ms grid; google_robot arms on their native 333 ms grid "
         "(their planner-interpolated controller cannot be driven at a fine tick).\n"
         "A latency change alters the google_robot command sequence only when an arrival crosses a control "
         "boundary, which damps their measured sensitivity — so grasp-vs-push is better supported than a raw "
         "embodiment comparison.",
         ha="center", fontsize=8.5, color="#555", style="italic")
fig.suptitle("Octo-small under MEASURED QRB5165 latency — four SIMPLER environments, "
             "two embodiments (fine tick, zero-order hold)", fontsize=13.5, y=1.0)
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}\n")
for key, label, _, _, skill, _dy in ENVS:
    print(f"{label} [{skill}]:")
    for arm in ARMS:
        rows = per_ep.get((key, arm))
        if not rows: continue
        p, lo, hi = design_ci(rows)
        r = paired(per_seed, key, arm)
        pen = f"   penalty {r[0]:+6.1f} [{r[1]:+6.1f},{r[2]:+6.1f}]" if r and arm != "lat0" else ""
        print(f"  {arm:11s} lat={LAT[arm]:6.1f}  SR={p:5.1f}% [{lo:5.1f},{hi:5.1f}]{pen}")
