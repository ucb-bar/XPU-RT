#!/usr/bin/env python3
"""C12 -- mission time against success rate, the six curated widowx schedules.

The third pairing in the set. C11 shows success and energy move together; this
shows the SAME is true of completion time, so the schedule ladder is not trading
speed for reliability anywhere except at its very top.

Points are joined in RELEASE-PERIOD order, so the line is the ladder itself and
its direction is the finding: every step down the ladder moves DOWN and RIGHT --
less likely to finish, and slower when it does. Better is UP and LEFT.

MISSION TIME IS CONDITIONED ON SUCCESS and cannot not be: a failure has no
completion time. That biases the slow arms DOWNWARD, because they are timed only
on the episodes they could still win -- so the true gap is WIDER than drawn. The
episode count behind each point is printed, and `cpu int8` is drawn hollow
because on both scenes its time rests on 4 successful episodes out of 240.

n = 30 seeds x 24 episodes per arm; bars are bootstrap 95% CIs over seeds.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
SRC = V.BASE / "traces_torque3"
ARMS = ["lat0", "p105w300", "p150w300", "pipe200fix", "serial283", "cpu685"]
COL = {"lat0": "#7f8c8d", "p105w300": "#148f77", "p150w300": "#45b39d",
       "pipe200fix": "#d68910", "serial283": "#e67e22", "cpu685": "#7b241c"}
NAME = {"lat0": "ideal", "p105w300": "pipe 125", "p150w300": "pipe 150",
        "pipe200fix": "pipe 219", "serial283": "serial", "cpu685": "cpu int8"}
SCENES = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]
LOW_N = 20                      # below this many successful episodes, draw hollow
# The top of the ladder is a tight cluster, so labels are placed per arm rather
# than all "above": stacked vertically they overlap each other, not the data.
OFF = {"lat0": (-6, -14, "right"), "p105w300": (0, 12, "center"),
       "p150w300": (13, 2, "left"), "pipe200fix": (12, 2, "left"),
       "serial283": (12, 2, "left"), "cpu685": (9, 9, "left")}   # sits on the floor: label must go UP

D = collections.defaultdict(dict)
for f in glob.glob(str(SRC / "*" / "energy2.json")):
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", Path(f).parent.name)
    D[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))


def ci(v, B=4000, seed=0):
    v = np.asarray(v, float)
    if v.size < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    return tuple(np.percentile([r.choice(v, v.size, replace=True).mean()
                                for _ in range(B)], [2.5, 97.5]))


fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.86), dpi=170)
fig.subplots_adjust(left=0.070, right=0.995, top=0.878, bottom=0.135, wspace=0.185)

for ax, (task, tlab) in zip(axes, SCENES):
    per = {a: next(iter(D[(task, a)].values()))["issue_period_ms"] for a in ARMS}
    order = sorted(ARMS, key=lambda a: per[a])          # the ladder, fastest first
    P = {}
    for a in order:
        d = D[(task, a)]
        sr = [100 * x["n_success"] / x["n_episodes"] for x in d.values()]
        mt_ep = [e["duration_s"] for x in d.values() for e in x["episodes"] if e["success"]]
        mt_seed = [float(np.mean([e["duration_s"] for e in x["episodes"] if e["success"]]))
                   for x in d.values() if x["n_success"] > 0]
        P[a] = dict(sr=float(np.mean(sr)), mt=float(np.mean(mt_ep)), n=len(mt_ep),
                    sr_ci=ci(sr), mt_ci=ci(mt_seed), seeds=len(mt_seed))

    ax.plot([P[a]["mt"] for a in order], [P[a]["sr"] for a in order],
            color=V.AXIS, lw=1.0, zorder=2, solid_capstyle="round")
    for a in order:
        p = P[a]
        thin = p["n"] < LOW_N
        ax.errorbar(p["mt"], p["sr"],
                    yerr=[[p["sr"] - p["sr_ci"][0]], [p["sr_ci"][1] - p["sr"]]],
                    xerr=[[max(p["mt"] - p["mt_ci"][0], 0)], [max(p["mt_ci"][1] - p["mt"], 0)]],
                    fmt="none", ecolor=V.INK2, lw=0.8, capsize=1.6, zorder=3)
        ax.scatter(p["mt"], p["sr"], s=64, zorder=5,
                   c="none" if thin else COL[a], edgecolor=COL[a],
                   linewidth=1.6 if thin else 0.8)
        dx, dy, ha = OFF[a]
        ax.annotate(f"{NAME[a]}\nn={p['n']}", (p["mt"], p["sr"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="bottom" if dy >= 0 else "top",
                    fontsize=5.4, color=V.STATUS["critical"] if thin else V.INK2,
                    linespacing=1.05, zorder=7,
                    bbox=dict(fc=V.SURFACE, ec="none", alpha=0.72, pad=0.7))

    ax.set_xlabel("mission time on successful episodes (s)", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("success rate (%)", fontsize=6.8, labelpad=2.0)
    ax.tick_params(labelsize=6.2)
    ax.set_title(tlab, fontsize=8.0, pad=3.0)
    ax.margins(x=0.18, y=0.20)
    ax.set_ylim(0, ax.get_ylim()[1])          # a success rate has a floor
    V.tidy(ax, grid="both")
    # Direction of improvement, parked in the empty top-right corner: the data
    # runs top-left to bottom-right, so that corner is the one nothing occupies.
    ax.annotate("", xy=(0.815, 0.960), xytext=(0.955, 0.855),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=V.MUTED, lw=1.0))
    ax.text(0.968, 0.845, "better", transform=ax.transAxes, fontsize=6.2,
            color=V.MUTED, ha="right", va="top", style="italic")

fig.text(0.53, 0.005, "hollow = fewer than 20 successful episodes behind the time  ·  "
                      "line joins the arms in release-period order",
         ha="center", va="bottom", fontsize=6.2, color=V.MUTED)
V.save(fig, "cand_c12_time_vs_success")
