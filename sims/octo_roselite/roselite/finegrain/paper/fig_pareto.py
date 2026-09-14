#!/usr/bin/env python3
"""Pareto frontier of schedules in (observation age, release cadence).

Both axes are COSTS the scheduler trades: an inference's observation age when its
action is applied, and how often a fresh result arrives. A schedule is dominated if
another achieves lower age AND lower cadence. Success is the outcome, shown as colour
-- deliberately NOT an axis, so the frontier is computed from the schedule's own
properties and success is free to disagree with it.

Physical bounds drawn as the unreachable region:
  cadence >= 83 ms   the bottleneck lane's MEASURED busy time per inference
                     (DSP 83.8 ms in the 10-instance pipelined trace)
  age     >= 232.7   the critical path: the 3-way chain is serial, so an inference
                     cannot finish faster than the sum of its per-lane work
                     (PREDICTED 232.69; the fastest MEASURED serial span is 239.4)

All schedule coordinates are MEASURED on the QRB5165, ungated.
"""
from __future__ import annotations
import json, glob, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "fig_pareto.png"

ARMS = {  # arm: (MEASURED age ms, MEASURED cadence ms, family)
    "pipe110fix": (385.1, 111.4, "pipelined"),
    "p105w300":   (258.7, 124.8, "pipelined"),
    "p130w275":   (272.3, 130.1, "pipelined"),
    "p150w300":   (281.6, 150.4, "pipelined"),
    "pipe200fix": (260.5, 219.2, "pipelined"),
    "serial283":  (283.4, 283.4, "serial"),
    "fp32_555":   (555.0, 555.0, "CPU only"),
    "cpu685":     (684.8, 684.8, "CPU only"),
}
FAM_C = {"pipelined": "#16a085", "serial": "#e67e22", "CPU only": "#c0392b"}
CADENCE_FLOOR, AGE_FLOOR = 83.0, 232.7
TASKS = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]
OFF = {"pipe110fix": (0, -30), "p105w300": (-46, 4), "p130w275": (14, -28),
       "p150w300": (46, 6), "pipe200fix": (44, 4), "serial283": (0, 20),
       "fp32_555": (0, 20), "cpu685": (0, -26)}


def load():
    per = collections.defaultdict(dict)
    for f in glob.glob(str(HERE / "g5fine" / "runs" / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        n = Path(f).parent.name
        if "_rng" not in n: continue
        head, seed = n.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        eps = d.get("episodes", [])
        if eps: per[(task, arm)][int(seed)] = 100.0*sum(e["success"] for e in eps)/len(eps)
    return per


def pareto(pts):
    """Non-dominated set: no other point has BOTH lower age and lower cadence."""
    out = []
    for a, (x, y) in pts.items():
        if not any((xx <= x and yy <= y and (xx < x or yy < y)) for b, (xx, yy) in pts.items() if b != a):
            out.append(a)
    return out


per = load()
coords = {a: (v[0], v[1]) for a, v in ARMS.items()}
front = pareto(coords)
fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0), dpi=150, gridspec_kw={"wspace": 0.13})
for ax, (task, lab) in zip(axes, TASKS):
    ideal = float(np.mean(list(per.get((task, "lat0"), {0: np.nan}).values())))
    ax.axvspan(150, AGE_FLOOR, color="#d5d8dc", alpha=0.55, zorder=0)
    ax.axhspan(40, CADENCE_FLOOR, color="#d5d8dc", alpha=0.55, zorder=0)
    ax.text(AGE_FLOOR - 4, 700, "critical path\n232.7 ms", fontsize=7.2, color="#666",
            ha="right", va="top", style="italic")
    ax.text(715, CADENCE_FLOOR + 6, "bottleneck lane 83 ms", fontsize=7.2, color="#666",
            ha="right", va="bottom", style="italic")
    fx = sorted([coords[a] for a in front])
    ax.step([p[0] for p in fx] + [740], [p[1] for p in fx] + [fx[-1][1]], where="post",
            color="#2c3e50", lw=1.6, ls="--", alpha=0.75, zorder=2)
    for arm, (x, y) in coords.items():
        v = per.get((task, arm), {})
        if len(v) < 5: continue
        sr = float(np.mean(list(v.values())))
        on = arm in front
        sc = ax.scatter(x, y, s=430 if on else 250, c=[sr], cmap="viridis", vmin=0, vmax=60,
                        edgecolor=FAM_C[ARMS[arm][2]], linewidth=3.0 if on else 1.4,
                        zorder=5 if on else 4)
        dx, dy = OFF.get(arm, (0, 18))
        ax.annotate(f"{arm}\n{sr:.1f}%" + ("  ★" if on else ""), (x, y),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=7.8, fontweight="bold" if on else "normal",
                    color="#111" if on else "#666", zorder=6)
    ax.set_xlabel("MEASURED observation age at actuation (ms)")
    ax.set_title(f"{lab}   ·   zero-latency ceiling {ideal:.1f}%", fontsize=11)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xlim(150, 740); ax.set_ylim(40, 740)
axes[0].set_ylabel("MEASURED release cadence (ms)")
cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.022, pad=0.015)
cb.set_label("task success rate (%)", fontsize=9)
h = [Line2D([], [], ls="", marker="o", ms=10, mfc="white", mec=c, mew=2.0, label=k)
     for k, c in FAM_C.items()]
h += [Line2D([], [], ls="--", color="#2c3e50", lw=1.6, label="Pareto frontier"),
      Line2D([], [], ls="", marker="s", ms=9, color="#d5d8dc", label="physically unreachable")]
axes[0].legend(handles=h, fontsize=8.2, loc="upper left", framealpha=0.95)
fig.suptitle("Six of eight schedules are Pareto-DOMINATED — and the frontier does not "
             "pick the winner by itself", fontsize=12.5, y=0.99)
fig.text(0.5, -0.055,
         f"Frontier (★): {', '.join(front)}. Everything else is beaten on BOTH axes at once, including every "
         "shipped configuration. But the frontier is not sufficient: pipe110fix sits on it and scores worst of the\n"
         "pipelined arms, because trading 126 ms of extra observation age for 13 ms of cadence is a bad trade. "
         "Success has to be measured; it cannot be read off the schedule's own cost axes.",
         ha="center", fontsize=8, style="italic", color="#555")
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
print(f"  Pareto frontier: {front}")
print(f"  dominated: {[a for a in coords if a not in front]}")
