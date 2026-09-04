#!/usr/bin/env python3
"""HIL — two views of one mechanism, sharing the command-rate axis:

  LEFT  : the schedule's worst-case loop response sets the fastest command rate it can sustain
          (rate_max = 1000 / worst_response).  The AOT feedback loop cuts response 8.00 -> 4.89 ms, lifting
          the sustainable rate 125 -> 204 Hz; ROS per-net pinning (12.40 ms) tops out at 81 Hz.
  RIGHT : a flight PHASE DIAGRAM (real Isaac flights, ZOH latency) — at each command rate, how fast the drone
          can fly before it crashes. The three schedulers' rates carry straight across from the left panel:
          ROS (81 Hz) sits on the crash frontier; XPU-RT (125 / 204 Hz) sits clear above it.
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# measured worst-case per-loop response (ms) -> sustainable rate = 1000/response
SCHEDS = [("ROS · per-net pinning", 12.40, "#e2231a"),
          ("XPU-RT · greedy", 8.00, "#2f6fb0"),
          ("XPU-RT · shard (feedback)", 4.89, "#1f9e5a")]


def upsample(speeds, lf, Z, ns=140, nf=160):
    sx = np.linspace(speeds[0], speeds[-1], ns); fy = np.linspace(lf[0], lf[-1], nf)
    out = np.zeros((nf, ns))
    for a, s in enumerate(sx):
        i = np.clip(np.searchsorted(speeds, s)-1, 0, len(speeds)-2); ts = (s-speeds[i])/(speeds[i+1]-speeds[i])
        for b, f in enumerate(fy):
            j = np.clip(np.searchsorted(lf, f)-1, 0, len(lf)-2); tf = (f-lf[j])/(lf[j+1]-lf[j])
            out[b, a] = ((1-tf)*(1-ts)*Z[j, i]+(1-tf)*ts*Z[j, i+1]+tf*(1-ts)*Z[j+1, i]+tf*ts*Z[j+1, i+1])
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
        cell[(sp, hz)][1] += 1; cell[(sp, hz)][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell}); freqs = sorted({h for _, h in cell})
    Z = np.array([[cell[(s, h)][0]/max(1, cell[(s, h)][1]) for s in speeds] for h in freqs])
    lf = np.log10(freqs); sx, fy, ZZ = upsample(speeds, lf, Z)
    ymin, ymax = np.log10(22), np.log10(240)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 13})
    fig = plt.figure(figsize=(16.5, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.9], wspace=0.05,
                          left=0.075, right=0.9, top=0.86, bottom=0.11)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1], sharey=axL)

    # ---------- LEFT: worst-response -> sustainable rate ----------
    yy = np.linspace(ymin, ymax, 200); budget = 1000.0 / (10**yy)          # ms available per loop at each rate
    axL.plot(budget, yy, color="#111", lw=2.6, zorder=4)
    axL.fill_betweenx(yy, budget, 16, color="#f0dcdc", zorder=0)           # right of curve = over budget
    axL.text(15.4, np.log10(205), "over budget →\nmisses the loop →\ncrash", color="#a83232", fontsize=11,
             ha="left", va="center", weight="bold", linespacing=1.25)
    for name, w, col in SCHEDS:                                            # names+rates are on the right panel
        r = 1000.0 / w
        axL.plot([w, w], [ymin, np.log10(r)], color=col, lw=3, zorder=5)
        axL.scatter([w], [np.log10(r)], s=170, color=col, edgecolors="white", linewidths=1.8, zorder=6, clip_on=False)
        axL.annotate(f"{w:.2f} ms", (w, ymin), textcoords="offset points", xytext=(0, 6), ha="center", va="bottom",
                     fontsize=11.5, weight="bold", color=col, zorder=7)
    axL.annotate("", xy=(4.89, np.log10(1000/4.89)), xytext=(8.0, np.log10(1000/8.0)),
                 arrowprops=dict(arrowstyle="-|>", color="#1f9e5a", lw=3), zorder=7)
    axL.text(6.4, np.log10(178), "feedback loop\n−39% response", color="#1f9e5a", fontsize=11, weight="bold",
             ha="center", va="center", linespacing=1.2,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f9e5a", alpha=0.9))
    axL.set_xlim(16, 2.5); axL.set_xlabel("worst-case response per control loop (ms)", fontsize=13.5)
    axL.set_ylabel("command rate (Hz, log)  →  faster loop", fontsize=13.5)
    axL.set_title("① schedule speed sets the rate", fontsize=14, weight="bold", loc="left")
    axL.grid(True, color="0.92", lw=0.5)

    # ---------- RIGHT: flight phase diagram ----------
    cmap = plt.cm.RdYlGn
    cf = axR.contourf(sx, fy, ZZ, levels=np.linspace(0, 1, 21), cmap=cmap, zorder=1)
    axR.contour(sx, fy, ZZ, [0.5], colors="white", linewidths=5, zorder=3)
    axR.contour(sx, fy, ZZ, [0.5], colors="#111", linewidths=2.6, zorder=4)
    x0, x1 = speeds[0]-0.06, speeds[-1]+0.06
    axR.add_patch(plt.Rectangle((x0, lf[-1]), x1-x0, ymax-lf[-1], facecolor="0.9", edgecolor="0.75", hatch="///", lw=0, zorder=2))
    axR.axhline(lf[-1], color="0.35", lw=1.2, zorder=5)
    axR.text((x0+x1)/2, (lf[-1]+ymax)/2+0.02, "not flight-tested above 100 Hz", fontsize=12,
             style="italic", color="0.4", ha="center", va="center", zorder=6)
    for (s, h), (k, n) in cell.items():
        axR.scatter(s, np.log10(h), s=520, c=[cmap(k/n)], edgecolors="#222", linewidths=1.4, zorder=6)
        axR.text(s, np.log10(h), f"{k}/{n}", ha="center", va="center", fontsize=10.5, weight="bold",
                 color="white" if k/n < 0.5 else "#123", zorder=7)
    for name, w, col in SCHEDS:                                            # rates carry across from the left
        y = np.log10(1000/w)
        axR.axhline(y, color=col, lw=3, ls=(0, (6, 3)), zorder=8)
        axR.text(x1-0.03, y, f"{name.split(' · ')[0]}  {1000/w:.0f} Hz", color="white", fontsize=12, weight="bold",
                 va="center", ha="right", zorder=9, bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="white", lw=1.2))
    axR.set_xlim(x0, x1); axR.set_ylim(ymin, ymax)
    yt = [25, 33, 50, 81, 100, 125, 204]
    axR.set_yticks(np.log10(yt)); axR.set_yticklabels(yt, fontsize=12)
    axR.tick_params(labelleft=False)
    axR.set_xticks(speeds); axR.set_xticklabels([f"{s:g}×" for s in speeds], fontsize=12)
    axR.set_xlabel("drone cruise speed  (× nominal)  →  faster", fontsize=13.5)
    axR.set_title("② the rate decides fly vs crash  (real Isaac flights)", fontsize=14, weight="bold", loc="left")
    cb = fig.colorbar(cf, ax=axR, fraction=0.045, pad=0.02); cb.set_label("gate-course success rate", fontsize=12); cb.ax.tick_params(labelsize=11)
    axR.legend(handles=[Line2D([0], [0], color="#111", lw=2.6, label="crash frontier (50%)"),
                        Line2D([0], [0], marker="o", mfc=cmap(0.9), mec="#222", ms=11, ls="none", label="flies (6 seeds)"),
                        Line2D([0], [0], marker="o", mfc=cmap(0.1), mec="#222", ms=11, ls="none", label="crashes")],
               loc="upper left", fontsize=11, framealpha=0.97, bbox_to_anchor=(0.008, 0.995))

    fig.suptitle("Fly faster — the schedule's worst-response sets the command rate, and the rate decides fly vs crash: "
                 "feedback (4.89 ms, 204 Hz) flies clear while ROS (12.40 ms, 81 Hz) sits on the crash edge",
                 fontsize=15, weight="bold", x=0.075, ha="left")
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
