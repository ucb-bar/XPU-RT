#!/usr/bin/env python3
"""Operating points in (latency, cadence) space, sized by task success.

The headline figure: every schedule is a point, and success is NOT a function of
either axis alone. Among schedules of similar latency success rises with cadence;
but the fastest-cadence schedule (pipe110fix, 111 ms) is beaten by slower ones
because its 385 ms observation age costs more than its rate buys.

Latencies and cadences are MEASURED on the QRB5165 (ungated, per-instance span and
release period from the runtime's own trace). Success is MEASURED in sim under that
MODELLED latency, 10 seeds x 24 episode configs per point.
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
OUT = Path(__file__).parent / "fig_operating_points.png"

# arm -> (measured latency ms, measured cadence ms, family)
ARMS = {
    "lat0":       (0.0,   0.0,   "ideal"),
    "pipe110fix": (385.1, 111.4, "pipelined"),
    "p105w300":   (258.7, 124.8, "pipelined"),
    "p130w275":   (272.3, 130.1, "pipelined"),
    "p150w300":   (281.6, 150.4, "pipelined"),
    "pipe200fix": (260.5, 219.2, "pipelined"),
    "serial283":  (283.4, 283.4, "serial"),
    "fp32_555":   (555.0, 555.0, "CPU only"),
    "cpu685":     (684.8, 684.8, "CPU only"),
}
FAM_C = {"ideal": "#7f8c8d", "pipelined": "#16a085", "serial": "#e67e22", "CPU only": "#c0392b"}
TASKS = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]


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


per = load()

# Manual label offsets: the four interesting schedules cluster in a ~25 x 95 ms box,
# so automatic placement collides. (dx, dy) in points, per arm.
OFF = {"pipe110fix": (34, -4), "p105w300": (-40, -6), "p130w275": (2, -26),
       "p150w300": (40, 8), "pipe200fix": (38, 6), "serial283": (0, 20),
       "fp32_555": (0, 18), "cpu685": (0, -24)}

fig, axes = plt.subplots(1, len(TASKS), figsize=(14.0, 5.8), dpi=150,
                         gridspec_kw={"wspace": 0.13})
axes = np.atleast_1d(axes)
vmax = 60
for ax, (task, lab) in zip(axes, TASKS):
    ideal = float(np.mean(list(per.get((task, "lat0"), {0: np.nan}).values())))
    for arm, (lat, cad, fam) in ARMS.items():
        v = per.get((task, arm), {})
        if arm == "lat0" or len(v) < 5: continue
        sr = float(np.mean(list(v.values())))
        sc = ax.scatter(lat, cad, s=340, c=[sr], cmap="viridis", vmin=0, vmax=vmax,
                        edgecolor=FAM_C[fam], linewidth=2.4, zorder=4)
        dx, dy = OFF.get(arm, (0, 16))
        ax.annotate(f"{arm}\n{sr:.1f}%", (lat, cad), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=7.8, fontweight="bold",
                    color="#222", zorder=6)
    ax.plot([170, 730], [170, 730], ls=":", lw=1.0, color="#bbb", zorder=1)
    ax.text(725, 705, "cadence = latency\n(serial, 1 in flight)", fontsize=7,
            color="#999", ha="right", va="top", style="italic")
    ax.set_xlabel("MEASURED observation age at actuation (ms)")
    ax.set_title(f"{lab}   ·   zero-latency ceiling {ideal:.1f}%", fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(190, 740); ax.set_ylim(60, 740)
axes[0].set_ylabel("MEASURED release cadence (ms)")
cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.022, pad=0.015)
cb.set_label("task success rate (%)", fontsize=9)
h = [Line2D([], [], ls="", marker="o", ms=9, mfc="white", mec=c, mew=2.4, label=k)
     for k, c in FAM_C.items() if k != "ideal"]
axes[0].legend(handles=h, fontsize=8.5, loc="upper left", framealpha=0.95,
               title="schedule family", title_fontsize=8.5)
fig.suptitle("Success depends on BOTH observation age and release cadence — "
             "the best operating point is interior to the frontier", fontsize=12.5, y=0.99)
fig.text(0.5, -0.05,
         "Below the dotted line several inferences are in flight. pipe110fix has the FASTEST cadence (111 ms) yet "
         "loses to p105w300 (125 ms): its 385 ms observation age costs more than the extra rate buys. Equally, "
         "pipe200fix has LOWER latency (260 ms)\nthan p105w300 and loses on cadence. Latency and cadence MEASURED "
         "on the QRB5165 (ungated); success MEASURED in sim, 10 seeds x 24 episode configs per point.",
         ha="center", fontsize=8, style="italic", color="#555")
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
for task, lab in TASKS:
    print(f"  {lab}: " + ", ".join(
        f"{a}={np.mean(list(per[(task,a)].values())):.1f}%" for a in ARMS if per.get((task, a))))
