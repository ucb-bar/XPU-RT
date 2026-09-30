#!/usr/bin/env python3
"""Modeled propulsive energy per flight from the LOGGED commanded wrench [thrust, Mx, My, Mz].

Distributes the body wrench to 4 rotor thrusts (X-quad mixer), converts each rotor thrust to
momentum-theory power (P_i ∝ T_i^1.5), sums, and integrates over the flight. Because producing body
MOMENTS spreads thrust unevenly across rotors, sum_i T_i^1.5 > 4·(T/4)^1.5 (Jensen), so a loop that
thrashes (large moments / high body-rate) spends more propulsive energy than a smooth one — the term
the hover-only estimate misses. Reported as OURS-vs-BASELINE RATIOS, robust to the unknown rotor
constants (arm length / kappa / disk area only rescale, they don't change the ratio much).

Needs figure_data.npz produced by `sweep_rate_demo.py --dump_figure_data ...` (now carries `wrench`).
Usage: python scripts/flight_energy_model.py --glob '<dir>/**/figure_data.npz' [--arm 0.09] [--kappa 0.016]
"""
import argparse, csv, glob, os
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout


def rotor_thrusts(wrench, arm, kappa):
    """X-quad mixer: wrench (T,4)=[thrust,Mx,My,Mz] -> (T,4) rotor thrusts (clamped >=0)."""
    Tt, Mx, My, Mz = wrench[:, 0], wrench[:, 1], wrench[:, 2], wrench[:, 3]
    d = arm / np.sqrt(2.0)                                   # roll/pitch moment arm per rotor
    # rotor order (X): 0 FR, 1 BL, 2 FL, 3 BR ; signs give +Mx roll, +My pitch, +Mz yaw
    T0 = Tt / 4 - Mx / (4 * d) + My / (4 * d) - Mz / (4 * kappa)
    T1 = Tt / 4 + Mx / (4 * d) - My / (4 * d) - Mz / (4 * kappa)
    T2 = Tt / 4 + Mx / (4 * d) + My / (4 * d) + Mz / (4 * kappa)
    T3 = Tt / 4 - Mx / (4 * d) - My / (4 * d) + Mz / (4 * kappa)
    return np.clip(np.stack([T0, T1, T2, T3], axis=1), 0.0, None)


def energy(npz, arm, kappa):
    d = np.load(npz, allow_pickle=True)
    t = d["t_s"].astype(float)
    if "wrench" not in d.files or np.allclose(d["wrench"], 0):
        return None                                         # no logged wrench (older dump / classical ctrl)
    w = d["wrench"].astype(float).copy()                   # (T,4) [thrust,Mx,My,Mz] — NORMALIZED command
    # The logged thrust channel is a normalized command around hover (u=-1 -> 0 thrust, u=0 -> hover,
    # u>0 -> climb), not absolute Newtons. Map to total thrust in HOVER units (hover total = 1) so the
    # momentum-theory power is finite and physical; mg cancels in the ours-vs-baseline ratios below.
    w[:, 0] = np.clip(1.0 + w[:, 0], 0.0, None)            # normalized -> total thrust (hover=1)
    Ti = rotor_thrusts(w, arm, kappa)                      # (T,4)
    P = (Ti ** 1.5).sum(axis=1)                            # momentum-theory total power (∝)
    P_hover = 4 * (w[:, 0] / 4) ** 1.5                     # same total thrust, evenly split (no moments)
    dt = np.gradient(t)
    E = float((P * dt).sum()); E_hover = float((P_hover * dt).sum())
    return dict(dur_s=float(t[-1] - t[0]), mean_thrust_N=float(w[:, 0].mean()),
                mean_absM=float(np.abs(w[:, 1:4]).mean()), energy=E, energy_hover=E_hover,
                maneuver_frac=(E - E_hover) / E if E else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=_REPO + "/results/codesign_feedback/energy_runs/**/figure_data.npz")
    ap.add_argument("--arm", type=float, default=0.09, help="rotor arm length (m)")
    ap.add_argument("--kappa", type=float, default=0.016, help="yaw drag/thrust ratio (m)")
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/flight_energy.csv")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(a.glob, recursive=True)):
        m = energy(f, a.arm, a.kappa)
        if m: m["flight"] = os.path.basename(os.path.dirname(f)).replace("_figdata", ""); rows.append(m)
    if not rows:
        print("no figure_data.npz with a logged `wrench` found under", a.glob,
              "\n(run sweep_rate_demo.py --dump_figure_data on --controller rl first)"); return
    cols = ["flight", "dur_s", "mean_thrust_N", "mean_absM", "energy", "energy_hover", "maneuver_frac"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in cols})
    print(f"{'flight':16} {'dur_s':>6} {'meanT(N)':>9} {'mean|M|':>8} {'energy(∝)':>10} {'maneuver%':>9}")
    for r in rows:
        print(f"{r['flight']:16} {r['dur_s']:>6.1f} {r['mean_thrust_N']:>9.3f} {r['mean_absM']:>8.4f} "
              f"{r['energy']:>10.1f} {100*r['maneuver_frac']:>8.1f}%")
    xpu = next((r for r in rows if "xpu" in r["flight"].lower()), None)
    if xpu:
        for r in rows:
            if r is xpu: continue
            print(f"\n  {r['flight']} vs XPU: total energy {r['energy']/xpu['energy']:.2f}x, "
                  f"maneuver energy fraction {100*r['maneuver_frac']:.1f}% vs {100*xpu['maneuver_frac']:.1f}%")
    print(f"\nwrote {a.out}  (ratios robust to arm/kappa; absolute ∝ units)")


if __name__ == "__main__":
    main()
