#!/usr/bin/env python3
# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Compare the three deployable configurations on metrics beyond success rate.

  accelerated pipelined  vs  accelerated serial  vs  baseline CPU-only serial

All success/funnel/timing numbers are MEASURED-in-sim under MODELLED QRB5165
latency (fine 40 ms control tick + zero-order hold); the latencies themselves
are MEASURED on the board. See RESULTS_FINEGRAIN.txt / METRICS_FINEGRAIN.txt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).with_name("config_comparison.png")

# arm, label, latency ms, colour
ARMS = [
    ("ctrl_lat0",  "zero-latency\ncontrol",          0.0,   "#7f8c8d"),
    ("pipe110",    "accelerated\nPIPELINED 110 ms",  117.7, "#16a085"),
    ("pipe200",    "accelerated\nPIPELINED 200 ms",  231.8, "#2980b9"),
    ("serial283",  "accelerated\nSERIAL",            283.4, "#e67e22"),
    ("cpu685",     "baseline\nCPU-ONLY serial",      684.8, "#c0392b"),
]

# MEASURED, from METRICS_FINEGRAIN.txt (n=72 each, seeds 0/2/4)
FUNNEL = {  # moved object, grasped, held 1 s, on target (== success)
    "ctrl_lat0": (97.2, 86.1, 80.6, 52.8),
    "pipe110":   (97.2, 94.4, 90.3, 59.7),
    "pipe200":   (95.8, 76.4, 69.4, 43.1),
    "serial283": (87.5, 59.7, 51.4, 23.6),
    "cpu685":    (65.3, 23.6, 20.8,  2.8),
}
AGE = {  # observation age at actuation: mean, max (ms)
    "ctrl_lat0": (80.0, 160.0),   "pipe110": (157.6, 200.0),
    "pipe200":   (312.9, 400.0),  "serial283": (405.2, 560.0),
    "cpu685":    (1008.2, 1360.0),
}
EFFORT = {  # hold-tick %, inferences/episode, time-to-success s (+/- sd)
    "ctrl_lat0": (80.0,  79.0,  8.4, 3.9),
    "pipe110":   (66.3, 124.2,  8.3, 2.5),
    "pipe200":   (80.5,  90.9, 10.8, 3.2),
    "serial283": (86.0,  74.5, 11.4, 3.0),
    "cpu685":    (94.3,  34.5, 12.2, 2.4),
}

STAGES = ["moved object", "grasped", "held 1 s", "on target (= success)"]
SHADES = ["#d5dbdb", "#aeb6bf", "#7f8c8d", None]  # last uses the arm colour

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), dpi=130,
                         gridspec_kw={"width_ratios": [1.45, 1, 1], "wspace": 0.28})

# ---- panel 1: the funnel -------------------------------------------------
ax = axes[0]
x = np.arange(len(ARMS))
w = 0.20
for j, stage in enumerate(STAGES):
    vals = [FUNNEL[a][j] for a, *_ in ARMS]
    cols = [c for *_, c in ARMS] if SHADES[j] is None else SHADES[j]
    ax.bar(x + (j - 1.5) * w, vals, w, label=stage, color=cols,
           edgecolor="black", linewidth=0.4, zorder=3)
for i, (a, *_rest) in enumerate(ARMS):
    ax.text(i + 1.5 * w, FUNNEL[a][3] + 1.8, f"{FUNNEL[a][3]:.1f}",
            ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("% of episodes reaching stage")
ax.set_title("Where latency breaks the task\nreaching survives; GRASPING collapses", fontsize=10)
ax.set_ylim(0, 108)
ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.95)

# ---- panel 2: sensor->actuation age --------------------------------------
ax = axes[1]
mean = [AGE[a][0] for a, *_ in ARMS]
mx = [AGE[a][1] for a, *_ in ARMS]
lat = [l for _, _, l, _ in ARMS]
ax.bar(x, mean, 0.55, color=[c for *_, c in ARMS], edgecolor="black",
       linewidth=0.4, zorder=3, label="mean age at actuation")
ax.plot(x, mx, "k^--", ms=6, lw=1.1, zorder=4, label="max age")
ax.plot(x, lat, "o:", color="#555", ms=5, lw=1.1, zorder=4,
        label="raw board latency")
for i, v in enumerate(mean):
    ax.text(i, v + 28, f"{v:.0f}", ha="center", fontsize=8)
ax.set_ylabel("observation age when the action is applied (ms)")
ax.set_title("Real sensor -> actuation age\ncompute latency + zero-order hold", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95)

# ---- panel 3: effort ------------------------------------------------------
ax = axes[2]
hold = [EFFORT[a][0] for a, *_ in ARMS]
ax.bar(x, hold, 0.55, color=[c for *_, c in ARMS], edgecolor="black",
       linewidth=0.4, zorder=3)
for i, v in enumerate(hold):
    ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_ylabel("% of 40 ms ticks running on a HELD (stale) action")
ax.set_ylim(0, 108)
ax2 = ax.twinx()
ax2.plot(x, [EFFORT[a][1] for a, *_ in ARMS], "s--", color="#8e44ad",
         ms=6, lw=1.3, zorder=5, label="inferences / episode")
ax2.set_ylabel("inferences per episode", color="#8e44ad")
ax2.tick_params(axis="y", labelcolor="#8e44ad")
ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
ax.set_title("Duty cycle: how much of the time\nthe robot acts on stale data", fontsize=10)

for a in axes:
    a.set_xticks(x)
    a.set_xticklabels([lbl for _, lbl, *_ in ARMS], fontsize=8)
    a.grid(True, axis="y", alpha=0.3, zorder=0)

fig.suptitle(
    "Octo-small on widowx_put_eggplant_in_basket under MEASURED QRB5165 latency — "
    "n=72 per arm, fine 40 ms control tick with zero-order hold",
    fontsize=12, y=1.005)
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
for a, lbl, l, _ in ARMS:
    print(f"  {lbl.replace(chr(10),' '):32s} lat={l:6.1f} ms  "
          f"success={FUNNEL[a][3]:5.1f}%  grasp={FUNNEL[a][1]:5.1f}%  "
          f"age={AGE[a][0]:6.1f} ms  hold={EFFORT[a][0]:4.1f}%  inf/ep={EFFORT[a][1]:5.1f}")
