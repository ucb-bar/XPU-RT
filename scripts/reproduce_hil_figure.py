#!/usr/bin/env python3
"""Reproduce every number in the HIL flight-envelope figure straight from the CSVs.

This is the review artifact: run it and it prints exactly the statistics the figure and its
caption assert, computed from the raw per-flight CSVs — so a reviewer can confirm the plot is
derived from the data (nothing hand-entered). It also (optionally) regenerates the envelope
panel PNG via the same draw code the paper uses.

Two data sources, deliberately kept separate (they are DIFFERENT experiments — see README_hil_data.md):
  * ENVELOPE  = results/codesign_feedback/hil_ablation.csv
      the error-bar panel. Fixed controller gain (moment_scale=0.0055), rate set by clean ZOH
      decimation. 5 cruise speeds x 4 control rates x N seeds/cell.
  * SHOWDOWN  = results/codesign_feedback/crash_verify/{new_xpu,new_ros50,new_ros}.csv
      the single flight's 3/6 & 0/6 aggregate. Calibrated gain law (0.5/eff_hz) + real
      sched_latency ZOH. cruise 1.4. new_xpu=100Hz, new_ros50=50Hz, new_ros=25Hz.

Usage:  python scripts/reproduce_hil_figure.py [--render]
Numbers update automatically if more seeds are appended to the CSVs.
"""
import argparse, csv, math, os, sys
from collections import defaultdict

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cf = os.path.join(_repo, "results/codesign_feedback")
sys.path.insert(0, os.path.join(_repo, "scripts"))
from hil_envelope_panel import wilson            # reuse the exact interval the figure uses


def two_prop_p(k1, n1, k2, n2):
    if not (n1 and n2):
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=os.path.join(_cf, "hil_ablation.csv"))
    ap.add_argument("--crash-dir", default=os.path.join(_cf, "crash_verify"))
    ap.add_argument("--render", action="store_true", help="also regenerate the envelope panel PNG")
    a = ap.parse_args()

    # ---- ENVELOPE ----------------------------------------------------------------
    rows = load(a.ablation)
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (round(float(r["cruise_speed"]), 2), round(float(r["eff_cmd_hz"])))
        cell[key][1] += 1
        cell[key][0] += int(r["outcome"] == "success")
    speeds = sorted({s for s, _ in cell}); rates = sorted({h for _, h in cell})
    print(f"== ENVELOPE ({os.path.relpath(a.ablation, _repo)}) — {len(rows)} flights, "
          f"{len(speeds)} speeds x {len(rates)} rates ==")
    print("  per-cell success k/n (rows=rate Hz, cols=speed m/s):")
    print("     rate\\spd " + " ".join(f"{s:>6g}" for s in speeds))
    for h in rates:
        print(f"     {h:>5g}Hz  " + " ".join(f"{cell[(s,h)][0]:>2}/{cell[(s,h)][1]:<3}" for s in speeds))

    def pool(h):
        k = sum(cell[(s, h)][0] for s in speeds); n = sum(cell[(s, h)][1] for s in speeds); return k, n
    print("\n  pooled over speed (the significant trend, n per rate):")
    P = {}
    for h in rates:
        k, n = pool(h); p, lo, hi = wilson(k, n); P[h] = (k, n, p, lo, hi)
        print(f"     {h:>5g}Hz : {k:>3}/{n:<3}  p={p:.3f}  Wilson95%[{lo:.3f},{hi:.3f}]")

    if 25 in P and 50 in P:
        k25, n25, p25, *_ = P[25]; k50, n50, p50, *_ = P[50]
        print(f"\n  CAPTION CHECKS:")
        print(f"     '+37 pts' (50Hz-25Hz pooled) = {(p50-p25)*100:+.0f} pts")
        print(f"     'p<0.001' (25->50 two-prop)  = {two_prop_p(k25,n25,k50,n50):.4f}")
    best = max((cell[k][0] / cell[k][1], k, cell[k]) for k in cell)
    print(f"     'up to two-thirds' best cell = {best[2][0]}/{best[2][1]} @ speed={best[1][0]} rate={best[1][1]}Hz ({best[0]:.2f})")

    # ---- SHOWDOWN ----------------------------------------------------------------
    print(f"\n== SHOWDOWN (crash_verify) — control-rate-monotone, Wilson 95% CI + pairwise tests ==")
    sd = {}
    for f, lbl in [("new_xpu.csv", "XPU-RT 100Hz"), ("new_ros50.csv", "ROS 50Hz"), ("new_ros.csv", "ROS 25Hz")]:
        rr = load(os.path.join(a.crash_dir, f))
        if rr:
            k = sum(x["outcome"] == "success" for x in rr); n = len(rr); p, lo, hi = wilson(k, n)
            sd[lbl] = (k, n)
            print(f"     {lbl:14} {k:>2}/{n:<2} = {p:.2f}  Wilson95%[{lo:.2f},{hi:.2f}]   ({os.path.join('crash_verify', f)})")
    keys = list(sd)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (k1, n1), (k2, n2) = sd[keys[i]], sd[keys[j]]
            pv = two_prop_p(k1, n1, k2, n2)
            print(f"     {keys[i]} vs {keys[j]}: p={pv:.3f} {'SIGNIFICANT' if pv < 0.05 else 'trend (n.s.)'}")
    print("\n  NOTE: anchor the claim on the rate-MONOTONE ordering + the significant 100-vs-starved-ROS")
    print("  endpoint, not the 100-vs-nominal-50 point estimate (a trend at this n). The envelope 50Hz")
    print("  cell != showdown ROS: different experiments (clean decimation+fixed gain vs ZOH latency+")
    print("  calibrated gain) — see README_hil_data.md.")

    if a.render:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from hil_envelope_panel import draw_envelope, INK
        plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
        fig = plt.figure(figsize=(8.6, 6.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[26, 1], left=0.10, right=0.90, top=0.88, bottom=0.135, wspace=0.04)
        draw_envelope(fig.add_subplot(gs[0]), a.ablation, colorbar_ax=fig.add_subplot(gs[1]))
        out = os.path.join(_cf, "hil_envelope_combined")
        fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
        print(f"\n  rendered {os.path.relpath(out, _repo)}.png/.pdf")


if __name__ == "__main__":
    main()
