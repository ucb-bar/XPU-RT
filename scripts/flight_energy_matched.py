#!/usr/bin/env python3
"""MATCHED-WINDOW propulsive-energy comparison: fair by construction.

The full-flight totals in flight_energy.csv are integrated to each flight's OWN end, so a run that
crashes at 1.7 s and one that flies 9 s are integrated over different windows. This instead pairs
runs by SEED (same scene: gates fixed, obstacles/people seeded), truncates BOTH to the window where
BOTH are still airborne (up to the earlier crash / the shorter run), and integrates propulsive power
only over that common window. So we compare energy actually spent up to the point at least one of the
pair goes down — apples to apples.

Power model = X-quad mixer -> rotor thrusts -> momentum-theory P(t) = sum_i T_i^1.5 on the LOGGED
commanded wrench (reused from flight_energy_model). Reported as ROS-over-XPU ratios, robust to the
unknown rotor constants.
"""
import argparse, csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flight_energy_model import rotor_thrusts
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

RES = _REPO + "/results/codesign_feedback"


def series(npz, arm, kappa):
    d = np.load(npz, allow_pickle=True)
    t = d["t_s"].astype(float)
    P = (rotor_thrusts(d["wrench"].astype(float), arm, kappa) ** 1.5).sum(axis=1)   # momentum-theory power (∝)
    return t - t[0], P, str(d["outcome"])


def E_upto(t, P, t_end):
    """integrate P over [0, t_end] (trapezoid), clipping the series at t_end."""
    m = t <= t_end + 1e-9
    if m.sum() < 2:
        return 0.0
    return float(np.trapezoid(P[m], t[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=float, default=0.09)
    ap.add_argument("--kappa", type=float, default=0.016)
    ap.add_argument("--seeds", default="")   # default: every seed the XPU-RT arm has on disk
    ap.add_argument("--out", default=f"{RES}/flight_energy_matched.csv")
    a = ap.parse_args()
    seeds = a.seeds.split(",") if a.seeds else sorted(
        os.path.basename(d).split("_s", 1)[1] for d in glob.glob(f"{RES}/energy_runs/xpu100_s*") if os.path.isdir(d))
    # the baseline conditions are whatever the energy runs recorded, named by their measured rate
    bases = sorted({os.path.basename(d).split("_s", 1)[0] for d in glob.glob(f"{RES}/energy_runs/ros*_s*") if os.path.isdir(d)})

    rows = []
    for base in bases:
        print(f"\n=== XPU-RT (100 Hz)  vs  {base}  — matched to the shared airborne window (per seed) ===")
        print(f"{'seed':>5} {'t_win(s)':>8} {'who ends window':>16} {'E_xpu':>9} {'E_'+base:>9} {'ratio':>7}")
        er, ex = [], []
        for s in seeds:
            fx = f"{RES}/energy_runs/xpu100_s{s}/figure_data.npz"
            fb = f"{RES}/energy_runs/{base}_s{s}/figure_data.npz"
            if not (os.path.exists(fx) and os.path.exists(fb)):
                continue
            tx, Px, ox = series(fx, a.arm, a.kappa)
            tb, Pb, ob = series(fb, a.arm, a.kappa)
            tw = min(tx[-1], tb[-1])                       # shared window = up to the earlier end (first to crash)
            who = "XPU" if tx[-1] <= tb[-1] else base
            Ex = E_upto(tx, Px, tw); Eb = E_upto(tb, Pb, tw)
            r = Eb / Ex if Ex else float("nan")
            ex.append(Ex); er.append(Eb)
            print(f"{s:>5} {tw:>8.2f} {who:>16} {Ex:>9.2f} {Eb:>9.2f} {r:>6.1f}x")
            rows.append(dict(pair=f"xpu100_vs_{base}", seed=s, t_window_s=round(tw, 3),
                             window_ended_by=who, E_xpu=round(Ex, 3), E_base=round(Eb, 3),
                             ratio_base_over_xpu=round(r, 3)))
        if ex:
            pooled = sum(er) / sum(ex)
            mean_r = float(np.mean([e / x for e, x in zip(er, ex) if x]))
            print(f"  -> over the shared window, {base} spends {pooled:.1f}x XPU-RT's propulsive energy "
                  f"(pooled); per-seed mean {mean_r:.1f}x")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "seed", "t_window_s", "window_ended_by",
                                           "E_xpu", "E_base", "ratio_base_over_xpu"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
