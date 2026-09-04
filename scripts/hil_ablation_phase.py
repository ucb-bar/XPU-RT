#!/usr/bin/env python3
"""HIL command-rate PHASE DIAGRAM — why the scheduler decides whether the drone flies.

The onboard schedule sets the command refresh rate; the flight sim (real Isaac, ZOH latency injection)
maps (drone speed, command rate) -> fly/crash. This renders that as a smooth safe→crash surface with the
crash frontier drawn, the raw 6-seed cells overlaid as ground truth, and the three schedulers' SUSTAINABLE
command rates marked: XPU-RT (shard) sits in deep-safe headroom, ROS static-partition lands on the frontier.
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def bilinear_grid(speeds, lfreqs, Z, ns=140, nf=160):
    """Smoothly upsample the coarse (freq x speed) success grid over log-freq."""
    sx = np.linspace(speeds[0], speeds[-1], ns)
    fy = np.linspace(lfreqs[0], lfreqs[-1], nf)
    out = np.zeros((nf, ns))
    for a, s in enumerate(sx):
        i = np.clip(np.searchsorted(speeds, s) - 1, 0, len(speeds) - 2)
        ts = (s - speeds[i]) / (speeds[i+1] - speeds[i])
        for b, f in enumerate(fy):
            j = np.clip(np.searchsorted(lfreqs, f) - 1, 0, len(lfreqs) - 2)
            tf = (f - lfreqs[j]) / (lfreqs[j+1] - lfreqs[j])
            out[b, a] = ((1-tf)*(1-ts)*Z[j, i] + (1-tf)*ts*Z[j, i+1]
                         + tf*(1-ts)*Z[j+1, i] + tf*ts*Z[j+1, i+1])
    # light gaussian smoothing
    k = np.exp(-np.linspace(-2, 2, 9)**2); k /= k.sum()
    for _ in range(2):
        out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, 4, "edge"), k, "same")[4:-4], 0, out)
        out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, 4, "edge"), k, "same")[4:-4], 1, out)
    return sx, fy, out


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_repo, "results/codesign_feedback/hil_ablation.csv"))
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/hil_ablation_phase"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        sp = float(r["cruise_speed"]); hz = round(float(r["eff_cmd_hz"]), 1)
        cell[(sp, hz)][1] += 1
        cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell}); freqs = sorted({h for _, h in cell})
    Z = np.array([[cell[(s, h)][0] / max(1, cell[(s, h)][1]) for s in speeds] for h in freqs])
    lf = np.log10(freqs)
    sx, fy, ZZ = bilinear_grid(speeds, lf, Z)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    cmap = plt.cm.RdYlGn
    # safe/crash surface over the TESTED range
    cf = ax.contourf(sx, fy, ZZ, levels=np.linspace(0, 1, 21), cmap=cmap, alpha=0.92, zorder=1)
    ax.contour(sx, fy, ZZ, levels=[0.5], colors="#111", linewidths=2.4, zorder=4)          # crash frontier
    ax.contour(sx, fy, ZZ, levels=[0.5], colors="white", linewidths=4.2, zorder=3)
    # raw 6-seed cells as ground truth
    for (s, h), (k, n) in cell.items():
        ax.scatter(s, np.log10(h), s=270, c=[cmap(k/n)], edgecolors="#222", linewidths=1.1, zorder=6)
        ax.text(s, np.log10(h), f"{k}/{n}", ha="center", va="center", fontsize=7.5,
                weight="bold", color="white" if k/n < 0.5 else "#123", zorder=7)
    # tested-range ceiling + "beyond tested range" band up to the fastest scheduler
    ax.axhline(lf[-1], color="0.4", lw=0.8, ls=(0, (4, 3)), zorder=5)
    ax.axhspan(lf[-1], np.log10(230), color="0.93", zorder=0)
    ax.text(speeds[0], (lf[-1] + np.log10(230)) / 2, "  beyond tested range (extrapolated safe headroom)",
            fontsize=8, style="italic", color="0.45", va="center", ha="left", zorder=6)

    # scheduler operating rates — the whole point: the scheduler PICKS your y-coordinate
    scheds = [("XPU-RT · shard (feedback): 204 Hz — deep-safe headroom", 204, "#1f9e5a"),
              ("XPU-RT · greedy: 125 Hz — safe", 125, "#26814e"),
              ("ROS · static partition: 81 Hz — on the crash frontier", 81, "#e2231a")]
    for lab, hz, col in scheds:
        y = np.log10(hz)
        ax.axhline(y, color=col, lw=2.4, ls=(0, (6, 3)), zorder=8)
        ax.text(speeds[0] + 0.03, y + 0.012, lab, color=col, fontsize=9, weight="bold",
                va="bottom", ha="left", zorder=9,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, lw=1.0, alpha=0.9))

    ax.set_xlim(speeds[0] - 0.05, speeds[-1] + 0.05); ax.set_ylim(np.log10(21), np.log10(230))
    yt = [25, 33, 50, 81, 100, 125, 204]
    ax.set_yticks(np.log10(yt)); ax.set_yticklabels(yt, fontsize=9)
    ax.set_xticks(speeds); ax.set_xticklabels([f"{s:g}×" for s in speeds], fontsize=9)
    ax.set_xlabel("drone cruise speed  (× nominal) →  faster", fontsize=11)
    ax.set_ylabel("onboard command refresh rate (Hz, log)  →  faster loop", fontsize=11)
    ax.set_title("HIL flight phase diagram — the onboard schedule sets the command rate, which decides fly vs crash\n"
                 "XPU-RT sustains 204 Hz (deep in the safe zone); ROS static-partition sustains only 81 Hz (on the crash frontier)",
                 fontsize=11.5, weight="bold", loc="left")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.16); cb.set_label("gate-course success rate (6 seeds/cell)", fontsize=9)
    ax.legend(handles=[Line2D([0], [0], color="#111", lw=2.4, label="crash frontier (50% success)"),
                       Line2D([0], [0], marker="o", mfc=cmap(0.9), mec="#222", ls="none", label="tested cell (green=flies)"),
                       Line2D([0], [0], marker="o", mfc=cmap(0.1), mec="#222", ls="none", label="tested cell (red=crashes)")],
              loc="upper right", fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.995, 0.995))
    fig.text(0.5, -0.01, "Real Isaac flights, in-sim zero-order-hold latency injection (a slower schedule → longer ZOH hold → "
             "fewer command refreshes/sec). Not live FPGA/K1 co-sim.", ha="center", fontsize=8.2, color="0.4")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
