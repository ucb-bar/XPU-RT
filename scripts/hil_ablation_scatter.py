#!/usr/bin/env python3
"""HIL-in-loop ablation scatter: drone SPEED x command FREQUENCY x crash/success.
Reads the per-episode CSV from sweep_rate_demo.py --sweep-csv (real Isaac flights, ZOH latency injection),
aggregates per (cruise_speed, eff_cmd_hz) cell to a success rate, and scatters it. Overlays the analytic
crash-frontier (worst-response vs 1/rate) for context. Honest label: in-sim ZOH HIL, not live FPGA/K1.
"""
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--band", default=os.path.join(_repo, "results/microros_baseline_k1/flyfaster_crash_band.json"))
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/hil_ablation_scatter"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cell = defaultdict(lambda: [0, 0])   # (speed, hz) -> [succ, total]
    for r in rows:
        sp = float(r["cruise_speed"]); hz = float(r["eff_cmd_hz"])
        cell[(sp, hz)][1] += 1
        if r["outcome"] == "success":
            cell[(sp, hz)][0] += 1

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    cmap = plt.cm.RdYlGn
    for (sp, hz), (s, n) in sorted(cell.items()):
        rate = s / n if n else 0
        ax.scatter(sp, hz, s=430, c=[cmap(rate)], edgecolors="#333", linewidths=1.1, zorder=4)
        ax.text(sp, hz, f"{s}/{n}", ha="center", va="center", fontsize=8,
                weight="bold", color="white" if rate < 0.55 else "#173", zorder=5)
    # analytic crash frontier: for each per-arm worst response, max safe command rate = 1000/resp
    try:
        band = json.load(open(a.band))
        for name, resp in band.get("worst_critical_response_ms", {}).items():
            thr = 1000.0 / resp
            ax.axhline(thr, color="#555", ls=(0, (5, 3)), lw=1.0, alpha=0.6, zorder=2)
            ax.text(0.012, thr, f"sched. sustains {name} · {thr:.0f} Hz", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", fontsize=7.2, color="#444", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.85))
    except Exception:
        pass
    speeds = sorted({sp for sp, _ in cell})
    hzs = sorted({hz for _, hz in cell})
    ax.set_xticks(speeds); ax.set_yticks(hzs)
    ax.set_xlabel("drone cruise speed (× nominal)  —  faster →", fontsize=11)
    ax.set_ylabel("effective command frequency (Hz)  —  set by schedule latency via ZOH hold", fontsize=11)
    ax.set_title("HIL-in-loop ablation: where the drone flies vs crashes (real Isaac flights, ZOH latency injection)\n"
                 "each point = 6 seeds; colour = success rate (green=flies, red=crashes)", fontsize=11, weight="bold")
    ax.grid(True, color="0.92", lw=0.5, zorder=0)
    ax.margins(0.12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label("gate-course success rate", fontsize=9)
    fig.text(0.5, -0.02, "In-sim zero-order-hold latency injection (RoSE-style software lockstep), NOT live FPGA/K1 co-sim. "
             "A slower onboard schedule → higher latency → fewer command refreshes/sec → the aggressive weave destabilizes.",
             ha="center", fontsize=8, color="#555", wrap=True)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=165, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")
    print(f"cells: {len(cell)}  flights: {sum(n for _, n in cell.values())}")


if __name__ == "__main__":
    main()
