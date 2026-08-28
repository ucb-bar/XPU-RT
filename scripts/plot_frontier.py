"""The measured YOLOv8n frequency frontier, built at final print size."""
import csv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

MM = 1/25.4
SINGLE, DOUBLE = 89*MM, 183*MM
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6,
    "axes.labelsize": 6, "axes.titlesize": 7,
    "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
OK, BAD, MUTE = "#0072B2", "#D55E00", "#666666"

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
ax2.plot(hz, solv, "s--", color="#009E73", ms=2.5, lw=0.7)
ax2.set_ylabel("edf solver time (s)", color="#009E73")
ax2.tick_params(axis="y", colors="#009E73")
ax2.spines[["top"]].set_visible(False)
ax2.spines["right"].set_color("#009E73")

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
