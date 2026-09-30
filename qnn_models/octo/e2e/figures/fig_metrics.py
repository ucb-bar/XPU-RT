#!/usr/bin/env python3
"""Success, mission time and the paired penalty, across every schedule and task.

All nine arms at n=10 seeds x 24 episode configs per task (960 episodes per arm).

Row 1  SUCCESS RATE, with the DESIGN-CORRECTED interval. A seed is a policy-noise
       replicate over the SAME 24 configs, so episodes cluster by config and a naive
       binomial overstates precision by DEFF = 1+(m-1)*ICC (MEASURED ICC 0.03-0.58).
Row 2  MISSION TIME on SUCCESSFUL episodes only -- an episode ends the tick success is
       detected, so sim_ms IS the completion time. CONDITIONED, unavoidably: a failure
       has no completion time. That biases high-latency arms DOWNWARD (they only
       succeed on the configs they can still do), and n is printed for that reason.
Row 3  PAIRED penalty vs each task's own zero-latency arm, per seed, t intervals.
       This is the comparable quantity: it cancels config difficulty and each task's
       own baseline.

Arms ordered by MEASURED release cadence, which is what the closed loop responds to.
"""
from __future__ import annotations
import json, glob, math, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "fig_metrics.png"

# arm: (MEASURED latency, MEASURED cadence, family) -- ordered by cadence
ARMS = [("lat0", 0.0, 0.0, "ideal"), ("pipe110fix", 385.1, 111.4, "pipelined"),
        ("p105w300", 258.7, 124.8, "pipelined"), ("p130w275", 272.3, 130.1, "pipelined"),
        ("p150w300", 281.6, 150.4, "pipelined"), ("pipe200fix", 260.5, 219.2, "pipelined"),
        ("serial283", 283.4, 283.4, "serial"), ("fp32_555", 555.0, 555.0, "CPU only"),
        ("cpu685", 684.8, 684.8, "CPU only")]
COL = {"ideal": "#7f8c8d", "pipelined": "#16a085", "serial": "#e67e22", "CPU only": "#c0392b"}
TASKS = [("egg", "eggplant (grasp)"), ("spoon", "spoon (grasp)"),
         ("coke", "coke can (grasp)"), ("drawer", "close drawer (push)")]
TC = {9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776, 3: 3.182, 2: 4.303}


def load():
    ep = collections.defaultdict(list); sd = collections.defaultdict(dict); tm = collections.defaultdict(list)
    for f in glob.glob(str(HERE / "g5fine" / "runs" / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        n = Path(f).parent.name
        if "_rng" not in n: continue
        h, s = n.rsplit("_rng", 1); t, a = h.split("_", 1)
        e = d.get("episodes", [])
        if not e: continue
        for x in e:
            ep[(t, a)].append((x["episode_id"], bool(x["success"])))
            if x["success"]: tm[(t, a)].append(x.get("sim_ms", np.nan) / 1000.0)
        sd[(t, a)][int(s)] = 100.0 * sum(x["success"] for x in e) / len(e)
    return ep, sd, tm


def design_ci(rows):
    by = collections.defaultdict(list)
    for e, ok in rows: by[e].append(ok)
    n, K = len(rows), len(by)
    p = sum(ok for _, ok in rows) / n
    if K < 2: return 100*p, 100*p, 100*p
    ns = [len(v) for v in by.values()]; m = sum(ns)/K
    msb = sum(len(v)*(sum(v)/len(v)-p)**2 for v in by.values())/(K-1)
    msw = sum(sum((z-sum(v)/len(v))**2 for z in v) for v in by.values())/max(n-K, 1)
    den = msb + (m-1)*msw
    icc = max((msb-msw)/den, 0.0) if den > 0 else 0.0
    se = math.sqrt(max(p*(1-p), 1e-12)/(n/(1+(m-1)*icc)))
    return 100*p, 100*max(p-1.96*se, 0), 100*min(p+1.96*se, 1)


ep, sd, tm = load()
names = [a[0] for a in ARMS]
fig, axes = plt.subplots(3, len(TASKS), figsize=(19.5, 11.9), dpi=140,
                         gridspec_kw={"hspace": 0.40, "wspace": 0.20})
for c, (task, tlab) in enumerate(TASKS):
    x = np.arange(len(names))
    cols = [COL[f] for _, _, _, f in ARMS]

    ax = axes[0, c]
    vals = [design_ci(ep[(task, a)]) if ep.get((task, a)) else (np.nan,)*3 for a in names]
    ax.bar(x, [v[0] for v in vals], 0.68, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(x, [v[0] for v in vals],
                yerr=[[v[0]-v[1] for v in vals], [v[2]-v[0] for v in vals]],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=5)
    ax.axhline(vals[0][0], ls="--", lw=1.1, color="#555", zorder=2)
    ax.set_ylim(0, 85); ax.set_title(tlab, fontsize=11, fontweight="bold")
    if c == 0: ax.set_ylabel("success rate (%)\ndesign-corrected 95% CI")

    ax = axes[1, c]
    series = [[z for z in tm.get((task, a), []) if np.isfinite(z)] or [np.nan] for a in names]
    bp = ax.boxplot(series, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for p_, cc in zip(bp["boxes"], cols): p_.set_facecolor(cc); p_.set_alpha(0.8); p_.set_edgecolor("black")
    for i, s in enumerate(series):
        n = 0 if (len(s) == 1 and not np.isfinite(s[0])) else len(s)
        ax.text(i+1, ax.get_ylim()[1], f"n={n}", ha="center", va="top", fontsize=6.4,
                color="#333" if n >= 20 else "#c0392b", fontweight="normal" if n >= 20 else "bold")
    if c == 0: ax.set_ylabel("mission time (s)\nSUCCESSFUL episodes only")

    ax = axes[2, c]
    base = sd.get((task, "lat0"), {})
    mus, los, his = [], [], []
    for a in names:
        A = sd.get((task, a), {}); com = sorted(set(A) & set(base))
        if a == "lat0" or len(com) < 2: mus.append(0); los.append(0); his.append(0); continue
        d = [A[k]-base[k] for k in com]; mu = float(np.mean(d))
        se = float(np.std(d, ddof=1))/math.sqrt(len(d)); t = TC.get(len(d)-1, 1.96)
        mus.append(mu); los.append(mu-t*se); his.append(mu+t*se)
    ax.axhline(0, ls="--", lw=1.2, color="#555", zorder=2)
    ax.bar(x, mus, 0.68, color=cols, edgecolor="black", linewidth=0.5, zorder=3)
    ax.errorbar(x, mus, yerr=[np.array(mus)-np.array(los), np.array(his)-np.array(mus)],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=5)
    ax.set_ylim(-58, 30)
    if c == 0: ax.set_ylabel("paired penalty vs zero-latency\n(points, t interval)")

    for r in range(3):
        axes[r, c].set_xticks(x)
        axes[r, c].set_xticklabels([f"{a}\n{cad:.0f} ms" for a, _, cad, _ in ARMS],
                                   fontsize=6.3, rotation=38, ha="right")
        axes[r, c].grid(True, axis="y", alpha=0.3, zorder=0)

fig.suptitle("Success, mission time and paired penalty — nine schedules x four tasks, "
             "960 episodes per arm.  Arms ordered by MEASURED release cadence.",
             fontsize=13, y=0.975)
# Reserve the caption band explicitly and anchor the text by its TOP edge. The bottom
# row's tick labels are two lines, rotated 38 deg, so they hang well below the axes;
# with bbox_inches="tight" the canvas grew to fit but the caption still landed on them.
fig.subplots_adjust(top=0.930, bottom=0.128)
fig.text(0.5, 0.058,
         "Dashed line = the zero-latency arm. Mission time is CONDITIONED on success (a failure has no completion "
         "time), which biases high-latency arms DOWNWARD — they only succeed on the configs they can still do; n is "
         "printed, red where n<20.\nSuccess intervals are design-corrected for clustering by episode config "
         "(MEASURED ICC 0.03-0.58); a naive binomial would be roughly half as wide.",
         ha="center", va="top", fontsize=8.3, style="italic", color="#555")
fig.savefig(OUT)
print(f"[ok] {OUT}")
