#!/usr/bin/env python3
"""Beyond success rate: where latency breaks the task, in BOTH widowx environments.

Companion to finegrain_tasks.png. Same encoding conventions:
  colour  category  ideal / HW accelerated (CPU+DSP+HTA) / CPU only
  hatch   ENV       solid = eggplant, hatched = spoon

Three panels:
  1. the funnel -- what fraction of episodes reach each stage. Success collapses
     long before REACHING does, so this locates the failure.
  2. real sensor->actuation age -- compute latency PLUS the zero-order hold.
  3. duty cycle -- how much of the time the robot is acting on stale data, and
     how many inferences an episode actually consumes.

Recomputed from g5fine/runs/ at plot time; nothing hardcoded.
"""
from __future__ import annotations
import json, glob, collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).parent
RUNS = HERE / "g5fine" / "runs"
OUT = HERE / "finegrain_tasks_comparison.png"

ARMS = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
ALABEL = {"lat0": "ideal\n(0 ms)", "pipe110": "PIPE\n110 ms", "pipe200": "PIPE\n200 ms",
          "serial283": "SERIAL\n283 ms", "fp32_555": "CPU fp32\n555 ms",
          "cpu685": "CPU int8\n685 ms"}
CAT = {"lat0": "ideal", "pipe110": "HW accelerated", "pipe200": "HW accelerated",
       "serial283": "HW accelerated", "fp32_555": "CPU only", "cpu685": "CPU only"}
CAT_COLOUR = {"ideal": "#7f8c8d", "HW accelerated": "#1a9e8f", "CPU only": "#c0392b"}
ENVS = [("egg", "eggplant", ""), ("spoon", "spoon", "///")]

STAGES = [("moved_correct_obj", "moved object"),
          ("is_src_obj_grasped", "grasped"),
          ("consecutive_grasp", "held"),
          ("src_on_target", "on target (= success)")]


def load():
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(str(RUNS / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name
        if "_rng" not in name: continue
        head, _ = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        for e in d.get("episodes", []):
            st = e.get("episode_stats", {})
            for k, _ in STAGES:
                agg[(task, arm)][k].append(bool(st.get(k, False)))
            agg[(task, arm)]["age"].append(e.get("age_mean_ms", 0.0))
            agg[(task, arm)]["agemax"].append(e.get("age_max_ms", 0.0))
            agg[(task, arm)]["hold"].append(100.0*(1.0 - e.get("frac_fresh", 0.0)))
            agg[(task, arm)]["inf"].append(e.get("n_inferences", 0))
    return agg


agg = load()
pct = lambda v: 100.0*sum(v)/len(v) if v else float("nan")
mean = lambda v: sum(v)/len(v) if v else float("nan")

fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0), dpi=130,
                         gridspec_kw={"width_ratios": [1.55, 1, 1], "wspace": 0.26})
x = np.arange(len(ARMS)); w = 0.19

# ---- panel 1: funnel ------------------------------------------------------
ax = axes[0]
SHADE = ["#d5dbdb", "#aeb6bf", "#7f8c8d", None]
for ei, (task, elabel, hatch) in enumerate(ENVS):
    for j, (key, sl) in enumerate(STAGES):
        vals = [pct(agg[(task, a)][key]) for a in ARMS]
        cols = [CAT_COLOUR[CAT[a]] for a in ARMS] if SHADE[j] is None else SHADE[j]
        ax.bar(x + (ei - 0.5)*4*w/2 + (j - 1.5)*w/2, vals, w/2, color=cols, hatch=hatch,
               edgecolor="black", linewidth=0.4, zorder=3,
               label=f"{sl}" if ei == 0 else None)
ax.set_ylabel("% of episodes reaching stage")
ax.set_title("Where latency breaks the task\nleft bars = eggplant, right (hatched) = spoon",
             fontsize=10.5)
ax.set_ylim(0, 108)
ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.95)

# ---- panel 2: sensor -> actuation age ------------------------------------
ax = axes[1]
for ei, (task, elabel, hatch) in enumerate(ENVS):
    a_mean = [mean(agg[(task, a)]["age"]) for a in ARMS]
    a_max = [mean(agg[(task, a)]["agemax"]) for a in ARMS]
    ax.bar(x + (ei-0.5)*0.42, a_mean, 0.40, color=[CAT_COLOUR[CAT[a]] for a in ARMS],
           hatch=hatch, edgecolor="black", linewidth=0.4, zorder=3)
    ax.plot(x + (ei-0.5)*0.42, a_max, "k^" + ("--" if ei else "-"), ms=5, lw=1.0, zorder=4)
ax.set_ylabel("observation age when the action is applied (ms)")
ax.set_title("Real sensor -> actuation age\nbars = mean, triangles = max. The two envs "
             "MATCH exactly:\nage is a property of the SCHEDULE, not the task", fontsize=10.5)

# ---- panel 3: duty cycle --------------------------------------------------
ax = axes[2]
for ei, (task, elabel, hatch) in enumerate(ENVS):
    hold = [mean(agg[(task, a)]["hold"]) for a in ARMS]
    ax.bar(x + (ei-0.5)*0.42, hold, 0.40, color=[CAT_COLOUR[CAT[a]] for a in ARMS],
           hatch=hatch, edgecolor="black", linewidth=0.4, zorder=3)
ax.set_ylabel("% of ticks running on a HELD (stale) action")
ax.set_ylim(0, 108)
ax2 = ax.twinx()
for ei, (task, elabel, hatch) in enumerate(ENVS):
    inf = [mean(agg[(task, a)]["inf"]) for a in ARMS]
    ax2.plot(x, inf, "s" + ("--" if ei else "-"), color="#8e44ad", ms=6, lw=1.3,
             zorder=5, label=f"{elabel}: inferences/episode", alpha=0.9 if ei == 0 else 0.6)
ax2.set_ylabel("inferences per episode", color="#8e44ad")
ax2.tick_params(axis="y", labelcolor="#8e44ad")
ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
ax.set_title("Duty cycle: stale-action fraction also matches;\nonly inferences/episode "
             "differ (spoon episodes are half\nas long: 12000 ms vs 24000 ms)", fontsize=10.5)

for a in axes:
    a.set_xticks(x); a.set_xticklabels([ALABEL[k] for k in ARMS], fontsize=8)
    a.grid(True, axis="y", alpha=0.3, zorder=0)

cat_h = [Patch(facecolor=c, edgecolor="black", label=k) for k, c in CAT_COLOUR.items()]
cat_h += [Patch(facecolor="white", edgecolor="black", label="eggplant (solid)"),
          Patch(facecolor="white", edgecolor="black", hatch="///", label="spoon (hatched)")]
fig.legend(handles=cat_h, loc="lower center", ncol=5, fontsize=9, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Octo-small under MEASURED QRB5165 latency — both widowx environments, "
             "10 seeds x 24 configs per arm (fine 40 ms tick, zero-order hold)",
             fontsize=12.5, y=1.01)
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}\n")
for task, elabel, _ in ENVS:
    print(f"{elabel}:")
    for a in ARMS:
        g = agg[(task, a)]
        print(f"  {a:11s} moved={pct(g['moved_correct_obj']):5.1f} "
              f"grasp={pct(g['is_src_obj_grasped']):5.1f} held={pct(g['consecutive_grasp']):5.1f} "
              f"succ={pct(g['src_on_target']):5.1f} | age={mean(g['age']):6.1f} "
              f"hold={mean(g['hold']):5.1f}% inf/ep={mean(g['inf']):5.1f}")
