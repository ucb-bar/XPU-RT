"""The measured YOLOv8n frequency frontier, built at final print size."""
import csv
import os

import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle  # noqa: E402

# The print rcParams and the palette live in `figstyle` because they were
# copy-pasted into five renderers and drifted: DroNet was blue in one figure
# and orange in another, and yolov8_nano was blue in that one. Colour is an
# identity claim, so it is made once.
figstyle.use()
MM = figstyle.MM
SINGLE_COL = figstyle.SINGLE_COL
DOUBLE_COL = figstyle.DOUBLE_COL
DOUBLE = DOUBLE_COL   # this file's older spelling
SINGLE = SINGLE_COL
OK, BAD, MUTE = figstyle.BLUE, figstyle.C_DEADLINE, "#666666"

rows = list(csv.DictReader(open("results/frontier/yolo_frequency_frontier.csv")))
hz    = [float(r["yolo_hz"]) for r in rows]
miss  = [int(r["predicted_deadline_misses"]) for r in rows]
cores = [float(r["cores_needed_by_service_time"]) for r in rows]
solv  = [float(r["solver_s"]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 52*MM))

ax = axes[0]
feas = [h for h, m in zip(hz, miss) if m == 0]
# The boundary lies between the last feasible point and the first infeasible one;
# drawing it AT a sampled point would claim precision the sweep does not have.
edge = (max(feas) + min(h for h, m in zip(hz, miss) if m)) / 2
ax.axvspan(min(hz)-0.3, edge, color=OK, alpha=0.07, lw=0)
ax.plot([h for h, m in zip(hz, miss) if m == 0],
        [m for m in miss if m == 0], "o", color=OK, label="feasible (0 misses)")
ax.plot([h for h, m in zip(hz, miss) if m],
        [m for m in miss if m], "o-", color=BAD, label="deadline misses")
ax.axvline(edge, color=MUTE, ls=":", lw=0.6)
ax.annotate(f"frontier\n{max(feas)}–{min(h for h,m in zip(hz,miss) if m)} Hz",
            xy=(edge, max(miss)*0.62), xytext=(edge+0.55, max(miss)*0.62),
            fontsize=5, color=MUTE, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.5))
ax.set_xlabel("yolov8n frequency (Hz)")
ax.set_ylabel("Predicted deadline misses")
ax.set_ylim(bottom=-30)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(hz, cores, "o-", color=MUTE)
ax.axhline(1.0, color=MUTE, ls=":", lw=0.5)
ax.axvspan(min(hz)-0.3, edge, color=OK, alpha=0.07, lw=0)
ax.axvline(edge, color=MUTE, ls=":", lw=0.6)
# Anchored at the RIGHT edge: the curve rises left-to-right, so it has moved
# well clear of the y=1 line by here. On the left the two overlap and the label
# sat on top of the data.
ax.text(max(hz), 1.03, "1 core of pure service time  ",
        fontsize=4.5, color=MUTE, va="bottom", ha="right")
ax.set_xlabel("yolov8n frequency (Hz)")
ax.set_ylabel("Cores needed by measured service time")
ax.spines[["top", "right"]].set_visible(False)
ax2 = ax.twinx()
ax2.plot(hz, solv, "s--", color=figstyle.GREEN, ms=2.5, lw=0.7)
ax2.set_ylabel("edf solver time (s)", color=figstyle.GREEN)
ax2.tick_params(axis="y", colors=figstyle.GREEN)
ax2.spines[["top"]].set_visible(False)
ax2.spines["right"].set_color(figstyle.GREEN)

for a, lab in zip(axes, "ab"):
    a.text(-0.19, 1.06, lab, transform=a.transAxes, fontsize=8,
           fontweight="bold", va="bottom")

fig.suptitle("yolov8n frequency alongside dronet 30 Hz + mlp_control 100 Hz, "
             "measured K1 costs", fontsize=7, y=1.03)
fig.tight_layout(rect=(0.02, 0, 1, 0.98))
for ext in ("png", "pdf"):
    fig.savefig(f"results/frontier/yolo_frequency_frontier.{ext}",
                bbox_inches="tight", pad_inches=0.03)
    print(f"wrote results/frontier/yolo_frequency_frontier.{ext}")
