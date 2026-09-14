#!/usr/bin/env python3
# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Fine-grain success and latency PENALTY, as separate series per environment.

Extends finegrain_scatter.png (which was eggplant-only, 3 seeds) to both widowx
environments at 10 seeds each, keeping the established encoding:

  colour   category   ideal / HW accelerated (CPU+DSP+HTA) / CPU only
  marker   schedule   star = none, circle = pipelined, square = serial
  fill     ENV        filled = eggplant, hollow = spoon

Panel 1 is the marginal success rate, with DESIGN-CORRECTED intervals -- a seed is
a policy-noise replicate over the SAME 24 episode configs, so a binomial on n
episodes overstates precision by DEFF = 1+(m-1)*ICC. Panel 2 is the PAIRED
per-seed penalty against each env's own zero-latency arm, which is the honest
cross-env comparison: it cancels the config difficulty and each env's own baseline.

All numbers are recomputed from g5fine/runs/ at plot time; nothing is hardcoded.
Latencies are MEASURED on the QRB5165; success is MEASURED in sim under that
MODELLED latency.
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
OUT = HERE / "finegrain_tasks.png"

ARMS = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
LAT = {"lat0": 0.0, "pipe110": 117.7, "pipe200": 231.8,
       "serial283": 283.4, "fp32_555": 555.0, "cpu685": 684.8}
CAT = {"lat0": "ideal", "pipe110": "HW accelerated", "pipe200": "HW accelerated",
       "serial283": "HW accelerated", "fp32_555": "CPU only", "cpu685": "CPU only"}
SCHED = {"lat0": "none", "pipe110": "pipelined", "pipe200": "pipelined",
         "serial283": "serial", "fp32_555": "serial", "cpu685": "serial"}
CAT_COLOUR = {"ideal": "#7f8c8d", "HW accelerated": "#1a9e8f", "CPU only": "#c0392b"}
SCHED_MARKER = {"none": "*", "pipelined": "o", "serial": "s"}
ENVS = [("egg", "eggplant_in_basket", True), ("spoon", "spoon_on_towel", False)]

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
        per_seed[(task, arm)][int(seed)] = 100.0 * sum(e["success"] for e in eps) / len(eps)
    return per_ep, per_seed


def design_ci(rows):
    """Marginal rate with the interval the CLUSTERED design actually supports."""
    by = collections.defaultdict(list)
    for ep, ok in rows: by[ep].append(ok)
    n, K = len(rows), len(by)
    p = sum(ok for _, ok in rows) / n
    if K < 2: return 100*p, 100*p, 100*p
    ns = [len(v) for v in by.values()]; m = sum(ns) / K
    msb = sum(len(v)*(sum(v)/len(v) - p)**2 for v in by.values()) / (K-1)
    msw = sum(sum((x - sum(v)/len(v))**2 for x in v) for v in by.values()) / max(n-K, 1)
    denom = msb + (m-1)*msw
    icc = max((msb - msw)/denom, 0.0) if denom > 0 else 0.0
    deff = 1 + (m-1)*icc
    se = math.sqrt(max(p*(1-p), 1e-12) / (n/deff))
    return 100*p, 100*max(p-1.96*se, 0), 100*min(p+1.96*se, 1)


def paired(per_seed, task, arm):
    a, b = per_seed.get((task, arm), {}), per_seed.get((task, "lat0"), {})
    common = sorted(set(a) & set(b))
    if not common: return None
    d = [a[s] - b[s] for s in common]
    mu = sum(d)/len(d)
    sd = math.sqrt(sum((x-mu)**2 for x in d)/max(len(d)-1, 1))
    se = sd/math.sqrt(len(d)) if len(d) > 1 else 0.0
    tc = TCRIT.get(len(d)-1, 1.96)
    return mu, mu-tc*se, mu+tc*se, len(d)


per_ep, per_seed = load()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.4), dpi=130,
                              gridspec_kw={"wspace": 0.22})

# ---- panel 1: marginal success -------------------------------------------
for task, label, filled in ENVS:
    xs, ys = [], []
    for arm in ARMS:
        rows = per_ep.get((task, arm))
        if not rows: continue
        p, lo, hi = design_ci(rows)
        c = CAT_COLOUR[CAT[arm]]
        ax.errorbar(LAT[arm], p, yerr=[[p-lo], [hi-p]], fmt="none", ecolor=c,
                    elinewidth=1.5, capsize=4, alpha=0.75, zorder=3)
        ax.scatter(LAT[arm], p, s=260 if SCHED[arm] == "none" else 140,
                   marker=SCHED_MARKER[SCHED[arm]],
                   facecolor=c if filled else "white",
                   edgecolor=c if not filled else "black",
                   linewidth=1.6 if not filled else 0.7, zorder=4)
        xs.append(LAT[arm]); ys.append(p)
    if xs:
        ax.plot(xs, ys, ls="-" if filled else "--", lw=1.0, color="#666",
                alpha=0.45, zorder=2)
ax.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax.set_ylabel("success rate (%)   ·   n=240 per point (10 seeds x 24 configs)")
ax.set_title("Marginal success, design-corrected intervals\n"
             "solid/filled = eggplant, dashed/hollow = spoon", fontsize=11)
ax.grid(True, alpha=0.3, zorder=0); ax.set_xlim(-45, 760); ax.set_ylim(-5, 85)

# ---- panel 2: paired penalty (the honest cross-env comparison) ------------
ax2.axhline(0, ls="--", lw=1.2, color="#555", zorder=1)
for task, label, filled in ENVS:
    for arm in ARMS:
        if arm == "lat0": continue
        r = paired(per_seed, task, arm)
        if not r: continue
        mu, lo, hi, n = r
        c = CAT_COLOUR[CAT[arm]]
        off = -6 if filled else 6
        ax2.errorbar(LAT[arm]+off, mu, yerr=[[mu-lo], [hi-mu]], fmt="none", ecolor=c,
                     elinewidth=1.5, capsize=4, alpha=0.75, zorder=3)
        ax2.scatter(LAT[arm]+off, mu, s=140, marker=SCHED_MARKER[SCHED[arm]],
                    facecolor=c if filled else "white",
                    edgecolor=c if not filled else "black",
                    linewidth=1.6 if not filled else 0.7, zorder=4)
ax2.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax2.set_ylabel("paired penalty vs that env's own zero-latency arm (points)")
ax2.set_title("Latency PENALTY, paired per seed (t intervals)\n"
              "the two environments are statistically indistinguishable", fontsize=11)
ax2.grid(True, alpha=0.3, zorder=0); ax2.set_xlim(-45, 760)

handles = [Line2D([], [], ls="", marker="o", ms=10, color=c, mec="black", label=k)
           for k, c in CAT_COLOUR.items()]
handles += [Line2D([], [], ls="", marker=m, ms=10, color="#444", mec="black",
                   label=f"schedule: {k}") for k, m in SCHED_MARKER.items()]
handles += [Line2D([], [], ls="-", marker="o", ms=9, color="#444", mec="black",
                   label="env: eggplant (filled)"),
            Line2D([], [], ls="--", marker="o", ms=9, mfc="white", color="#444",
                   mec="#444", mew=1.6, label="env: spoon (hollow)")]
ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.95, ncol=2)

fig.suptitle("Octo-small under MEASURED QRB5165 latency — fine 40 ms tick with "
             "zero-order hold, both widowx environments", fontsize=13, y=0.99)
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
for task, label, _ in ENVS:
    print(f"\n{label}:")
    for arm in ARMS:
        rows = per_ep.get((task, arm))
        if not rows: continue
        p, lo, hi = design_ci(rows)
        r = paired(per_seed, task, arm)
        pen = f"  penalty {r[0]:+6.1f} [{r[1]:+6.1f},{r[2]:+6.1f}] n={r[3]}" if r else ""
        print(f"  {arm:11s} lat={LAT[arm]:6.1f}  SR={p:5.1f}% [{lo:5.1f},{hi:5.1f}]{pen}")
