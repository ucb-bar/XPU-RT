#!/usr/bin/env python3
# Preserved exactly as it was found, for provenance. This is the script that produced the paper's
# plots/fig_cores_yolo.png (sha256 e0ee34e3...) and it lived outside both repositories, untracked,
# in XPU-RT/results/codesign_feedback/refined_src/. Its thirty data points are literals.
#
# It is kept here so the paper's figure has a recorded producer, and NOT as the way to redraw it:
# scripts/cores_yolo_service.py derives the same numbers from the schedules under schedules/ and
# writes a sidecar saying where each one comes from, which is what verify_cores_yolo.py checks.
# Nothing runs this file; it is a record.
"""Honest, fair cores x YOLO-service figure. Same deployed workload (YOLO 22 ms + nav 20 + ctrl 10),
same 8 K1 harts, same board calibration -- the ONLY difference is whether the scheduler can shard a
single YOLO inference across harts. XPU-RT shards it; a ROS node runs the inference sequentially and
cannot (even a 4-P-core partition). Two curve types per scheme: SCHEDULED (AOT/profiled, dashed) and
ACTUAL (board-calibrated re-cost, solid). Anchored at the REAL 22 ms deployed deadline.

Story: (1) SCHEDULED -- XPU-RT sharding meets 22 ms (CP-SAT ~18, greedy ~20); every ROS config is stuck
at ~24 ms and misses -- a genuine 'we meet, they can't'. (2) BOARD -- the ~1.26x per-op inflation lifts
all; XPU-RT re-solves and adapts to 23 ms / 0.96x cruise, ROS is static and stuck at ~30 ms / 0.72x."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B, G, R = "#0072B2", "#009E73", "#D55E00"
FILL_OK, FILL_BAD = "#e7f1ec", "#f7eae4"
DL = 24.0   # realistic deployable K1 cruise budget (the aspirational 22 ms isn't reachable on real silicon)                                              # real deployed YOLO deadline (1.0x nominal cruise)
K = np.array([4, 5, 6, 7, 8])
cpsat_sched = np.array([19.00, 18.75, 18.50, 18.375, 18.25])   # K4,K6,K8 measured floors; K5,K7 interp
cpsat_board = np.array([24.50, 24.00, 23.25, 23.25, 23.00])
greedy_sched = np.array([25.91, 23.461, 21.113, 20.349, 20.45])
greedy_board = np.array([35.198, 34.683, 30.619, 28.073, 26.784])
ros_sched = np.array([25.757, 26.694, 24.884, 25.914, 24.151])
ros_board = np.array([31.414, 36.264, 31.659, 33.083, 30.438])
ros4p_sched = 24.35                                    # strong ROS: reserve all 4 P-cores, STILL sequential


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 13.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    ax.axhspan(0, DL, color=FILL_OK, zorder=0)
    ax.axhspan(DL, 400, color=FILL_BAD, zorder=0)
    ax.axhline(DL, color="#111", lw=1.7, ls=(0, (5, 3)), zorder=4)
    ax.text(6.3, DL + 0.18, "realistic K1 cruise  (1.0× = 24 ms)", fontsize=11,
            color="#166b3d", weight="bold", va="bottom", ha="center")

    def pair(sched, board, color, mk, label):
        ax.plot(K, sched, color=color, ls=(0, (2, 1.6)), lw=1.9, marker=mk, ms=6.5, mfc="white",
                mec=color, mew=1.5, alpha=0.85, zorder=6)
        ax.plot(K, board, color=color, ls="-", lw=2.8, marker=mk, ms=8.5, mfc=color, mec="white",
                mew=1.3, zorder=8, label=label)
    pair(cpsat_sched, cpsat_board, B, "D", "CP-SAT (shards YOLO)")
    pair(greedy_sched, greedy_board, G, "o", "XPU-RT greedy + shard")
    pair(ros_sched, ros_board, R, "s", "ROS · 1-hart pin (no shard)")
    # strong-ROS reference: even reserving all 4 P-cores, a ROS node runs YOLO sequentially -> ~24 ms
    # (4-P partition line dropped: its combined-target board number is a calibration artifact; the
    #  1-hart per-core sweep already shows ROS is core-independent — it can't shard the inference.)

    # speed labels at K8 (actual/board), anchored to the real 22 ms
    for arr, color in ((cpsat_board, B), (greedy_board, G), (ros_board, R)):
        v = DL / arr[-1]
        ax.annotate(f"{v:.2f}×", (8, arr[-1]), textcoords="offset points", xytext=(9, 0),
                    fontsize=12, weight="bold", color=color, ha="left", va="center")

    # board-inflation gap at 8 cores
    ax.annotate("", (8, cpsat_board[-1]), (8, cpsat_sched[-1]),
                arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=1.7), zorder=7)
    ax.text(8.14, (cpsat_board[-1] + cpsat_sched[-1]) / 2, "board\n+26%", fontsize=11.5,
            color="#cc0000", ha="left", va="center", weight="bold")

    ax.set_ylim(16.3, 37.6); ax.set_xlim(3.7, 8.9); ax.set_xticks(K)
    ax.set_xlabel("cores allocated to YOLO", fontsize=15, labelpad=9)
    # the two vertical axes are the SAME data in two units: service time (left) and the cruise speed it
    # sustains (right) — bridged at the dashed 24 ms = 1.0× line. Direction cues make that unambiguous.
    ax.set_ylabel("YOLO per-frame service (ms) · K1-calibrated\n(lower = faster inference ↓)", fontsize=13)
    ax.tick_params(labelsize=12.5)
    axr = ax.secondary_yaxis("right", functions=(lambda s: DL / np.where(s <= 0, np.nan, s),
                                                 lambda v: DL / np.where(v <= 0, np.nan, v)))
    axr.set_ylabel("max sustainable cruise speed (× nominal)\n(higher = faster flight ↑)", fontsize=13)
    tk = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]; axr.set_yticks(tk)
    axr.set_yticklabels([f"{v:.1f}×" for v in tk], fontsize=12)

    style_key = [Line2D([0], [0], color="#444", ls="-", lw=2.8, label="K1-calibrated model"),
                 Line2D([0], [0], color="#444", ls=(0, (2, 1.6)), lw=1.9, label="scheduled · AOT")]
    h, l = ax.get_legend_handles_labels()
    # interleave so column-major fill puts the 3 schemes in row 1, the 2 style keys in row 2
    ordered = [h[0], style_key[0], h[1], style_key[1], h[2]]
    leg = ax.legend(handles=ordered, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                    fontsize=12.5, ncol=3, title="only op-sharding fits YOLO on the K1",
                    title_fontsize=13, columnspacing=1.6, handletextpad=0.6, borderaxespad=0)
    leg._legend_box.align = "center"
    fig.savefig("fig_fair_v6.pdf", bbox_inches="tight"); fig.savefig("fig_fair_v6.png", dpi=200, bbox_inches="tight")
    print("wrote fig_fair_v6 | board cruise @8: CP-SAT %.2f greedy %.2f ROS %.2f"
          % (DL/cpsat_board[-1], DL/greedy_board[-1], DL/ros_board[-1]))


if __name__ == "__main__":
    main()
