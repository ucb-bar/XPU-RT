#!/usr/bin/env python3
# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Fine-grain-only success-vs-latency, as a scatter coloured by schedule type.

Unlike success_vs_latency_finegrain.png this drops the coarse 200 ms model
entirely -- the 40 ms control tick with zero-order hold is the faithful physics
timing, so the coarse overlay is only a distraction here. Points are not joined:
each is a distinct hardware configuration, not a sample of one continuous curve.

Categories
  ideal          no compute latency; the upper reference
  HW accelerated CPU+DSP+HTA on the QRB5165 (pipelined or serial)
  CPU only       everything on the Kryo (int8 monolith, or the fp32 path,
                 which is CPU-only because fp32-on-DSP is refused)

Latencies are MEASURED on the board; success is MEASURED-in-sim under that
MODELLED latency. n=72 per arm (seeds 0/2/4). See METRICS_FINEGRAIN.txt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).with_name("finegrain_scatter.png")

CAT_COLOUR = {"ideal": "#7f8c8d", "HW accelerated": "#1a9e8f", "CPU only": "#c0392b"}
SCHED_MARKER = {"none": "*", "pipelined": "o", "serial": "s"}

#   label, latency, SR%, ci_lo, ci_hi, category, schedule, age_mean, age_max, hold%
ARMS = [
    ("ideal (no latency)",  0.0, 52.8, 41.4, 63.9, "ideal",          "none",       80.0,  160.0, 80.0),
    ("pipelined 110 ms",  117.7, 59.7, 48.2, 70.3, "HW accelerated", "pipelined", 157.6,  200.0, 66.3),
    ("pipelined 200 ms",  231.8, 43.1, 32.3, 54.6, "HW accelerated", "pipelined", 312.9,  400.0, 80.5),
    ("serial 3-way",      283.4, 23.6, 15.3, 34.6, "HW accelerated", "serial",    405.2,  560.0, 86.0),
    ("fp32 (CPU)",        555.0,  5.6,  2.2, 13.4, "CPU only",       "serial",    811.8, 1080.0, 92.8),
    ("int8 monolith",     684.8,  2.8,  0.8,  9.6, "CPU only",       "serial",   1008.2, 1360.0, 94.3),
]

BASE_SR, BASE_LO, BASE_HI = 52.8, 41.4, 63.9  # 5 Hz stock baseline, 38/72

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=130,
                              gridspec_kw={"wspace": 0.24})

# ---- panel 1: success vs latency -----------------------------------------
ax.axhspan(BASE_LO, BASE_HI, color="#95a5a6", alpha=0.16, zorder=0)
ax.axhline(BASE_SR, ls="--", lw=1.1, color="#555", zorder=1,
           label=f"stock 5 Hz baseline {BASE_SR:.1f}% (38/72)")

for lbl, lat, sr, lo, hi, cat, sched, *_ in ARMS:
    ax.errorbar(lat, sr, yerr=[[sr - lo], [hi - sr]], fmt="none",
                ecolor=CAT_COLOUR[cat], elinewidth=1.6, capsize=5, alpha=0.85, zorder=3)
    ax.scatter(lat, sr, s=210 if sched == "none" else 130,
               marker=SCHED_MARKER[sched], color=CAT_COLOUR[cat],
               edgecolor="black", linewidth=0.7, zorder=4)
    ax.annotate(f"{lbl}\n{sr:.1f}%", (lat, sr), textcoords="offset points",
                xytext=(11, 10), fontsize=8.5, color=CAT_COLOUR[cat],
                fontweight="bold")

ax.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax.set_ylabel("success rate (%)  ·  widowx_put_eggplant_in_basket, n=72")
ax.set_title("Fine 40 ms control tick with zero-order hold\n"
             "each point is a distinct hardware configuration, not a curve", fontsize=11)
ax.set_xlim(-45, 780)
ax.set_ylim(-4, 82)
ax.grid(True, alpha=0.3, zorder=0)

handles = [Line2D([], [], ls="--", color="#555", label=f"stock baseline {BASE_SR:.1f}%")]
handles += [Line2D([], [], ls="", marker="o", ms=9, color=c, mec="black",
                   label=k) for k, c in CAT_COLOUR.items()]
handles += [Line2D([], [], ls="", marker=m, ms=9, color="#444", mec="black",
                   label=f"schedule: {k}") for k, m in SCHED_MARKER.items() if k != "none"]
ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.95)

# ---- panel 2: sensor -> actuation age ------------------------------------
ax2.plot([0, 760], [0, 760], ls=":", lw=1.2, color="#888", zorder=1,
         label="age == raw board latency")
for lbl, lat, sr, _lo, _hi, cat, sched, am, ax_, hold in ARMS:
    ax2.vlines(lat, am, ax_, color=CAT_COLOUR[cat], lw=1.5, alpha=0.55, zorder=3)
    ax2.scatter(lat, am, s=210 if sched == "none" else 130,
                marker=SCHED_MARKER[sched], color=CAT_COLOUR[cat],
                edgecolor="black", linewidth=0.7, zorder=4)
    ax2.scatter(lat, ax_, s=48, marker="_", color=CAT_COLOUR[cat], zorder=4)
    ax2.annotate(f"{am:.0f}\n({hold:.0f}% held)", (lat, am),
                 textcoords="offset points", xytext=(11, -16), fontsize=8,
                 color=CAT_COLOUR[cat])

ax2.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax2.set_ylabel("observation age when the action is applied (ms)")
ax2.set_title("Real sensor -> actuation age\n"
              "filled = mean, tick = max; the gap above the dotted line is the hold",
              fontsize=11)
ax2.grid(True, alpha=0.3, zorder=0)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

fig.suptitle("Octo-small under MEASURED QRB5165 latency — fine-grain physics timing only",
             fontsize=13, y=0.99)
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
for lbl, lat, sr, lo, hi, cat, sched, am, _ax, hold in ARMS:
    print(f"  {cat:15s} {sched:10s} {lbl:20s} lat={lat:6.1f}  SR={sr:5.1f}% "
          f"[{lo:.1f},{hi:.1f}]  age={am:6.1f} ms  held={hold:.1f}%")
