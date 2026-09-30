#!/usr/bin/env python3
"""Analyze the course-B envelope sweep and compare it to course A (cross-course generalization).

Prints, for course B: per-cell success k/n, the pooled-over-speed trend with Wilson 95% CI, and the
25->50 Hz cliff test — then a side-by-side vs course A so we can state whether the control-rate floor
+ speed-limited band GENERALIZES to a different gate layout. Course A here = the CURRENT figure source
(hil_ablation.csv, i.e. v2 after finalize); the two are NEVER pooled, only compared.
"""
import csv, math, os, sys
from collections import defaultdict

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cf = os.path.join(_repo, "results/codesign_feedback")
sys.path.insert(0, os.path.join(_repo, "scripts"))
from hil_envelope_panel import wilson


def two_prop_p(k1, n1, k2, n2):
    if not (n1 and n2): return float("nan")
    p1, p2 = k1 / n1, k2 / n2; p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0: return float("nan")
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def grid(path):
    if not os.path.exists(path): return None
    cell = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(path)):
        k = (round(float(r["cruise_speed"]), 2), round(float(r["eff_cmd_hz"])))
        cell[k][1] += 1; cell[k][0] += int(r["outcome"] == "success")
    return cell


def pooled(cell):
    rates = sorted({h for _, h in cell}); speeds = sorted({s for s, _ in cell})
    out = {}
    for h in rates:
        kk = sum(cell[(s, h)][0] for s in speeds); nn = sum(cell[(s, h)][1] for s in speeds)
        out[h] = (kk, nn)
    return out, rates, speeds


def show(name, cell):
    P, rates, speeds = pooled(cell)
    print(f"\n== {name} — pooled over speed ==")
    for h in rates:
        k, n = P[h]; p, lo, hi = wilson(k, n)
        print(f"   {h:>5g}Hz : {k:>3}/{n:<3}  p={p:.3f}  Wilson95%[{lo:.3f},{hi:.3f}]")
    if 25 in P and 50 in P:
        k25, n25 = P[25]; k50, n50 = P[50]
        print(f"   floor 25->50Hz: {(k50/n50-k25/n25)*100:+.0f} pts, p={two_prop_p(k25,n25,k50,n50):.4f}")
    return P


def main():
    B = grid(os.path.join(_cf, "hil_ablation_courseB.csv"))
    A = grid(os.path.join(_cf, "hil_ablation.csv"))
    if B is None:
        print("course-B CSV not present yet (hil_ablation_courseB.csv)"); return
    PA = show("COURSE A (figure source)", A) if A else {}
    PB = show("COURSE B (alternate gates)", B)
    print("\n== GENERALIZATION verdict ==")
    common = sorted(set(PA) & set(PB))
    if common:
        print("   rate |   A k/n  |   B k/n  (does the floor + rise hold on B?)")
        for h in common:
            ka, na = PA[h]; kb, nb = PB[h]
            print(f"   {h:>4g}Hz | {ka:>2}/{na:<2}   | {kb:>2}/{nb:<2}")
    print("   -> Read: if B also shows ~0 below 50Hz and a rise above it, the control-rate floor")
    print("      generalizes across gate layouts (same unchanged stack). Do NOT pool A and B.")


if __name__ == "__main__":
    main()
