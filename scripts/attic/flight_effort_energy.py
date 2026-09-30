#!/usr/bin/env python3
"""Control-effort and modeled-energy metrics per drone flight, from the recorded state traces.

The warehouse drone is a lumped body-wrench model ([thrust, Mx, My, Mz]) — no rotor RPM/PWM signals —
so this derives effort/energy from the flight kinematics that ARE recorded in figure_data.npz:

  DIRECT (no modeling assumptions):
    mean/peak |body-rate|  (rad/s)   — angular control activity (oscillation)
    mean |accel|           (m/s^2)   — translational control activity
    mean |jerk|            (m/s^3)   — smoothness

  MODELED (inverse dynamics; stated assumptions):
    per-step specific thrust  T(t) = |a(t) - g|            (thrust per unit mass; mass cancels in ratios)
    momentum-theory power   P(t) ∝ T(t)^1.5                (hover/climb power scales as thrust^1.5)
    mean specific power = mean(P)  and  energy/time proxy   — reported as OURS-vs-BASELINE RATIOS, which
    are robust to the unknown mass / rotor-disk / motor constants.

Usage: python scripts/flight_effort_energy.py [--glob '<dir>/**/figure_data.npz'] [--out effort_energy.csv]
"""
import argparse, csv, glob, os
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root, so this runs from any checkout

G = np.array([0.0, 0.0, -9.81])


def metrics(npz):
    d = np.load(npz, allow_pickle=True)
    pos = d["poses"][:, :3].astype(float); t = d["t_s"].astype(float)
    w = np.linalg.norm(d["imu_w"], axis=1)                       # body-rate magnitude
    v = np.gradient(pos, t, axis=0); a = np.gradient(v, t, axis=0); j = np.gradient(a, t, axis=0)
    amag = np.linalg.norm(a, axis=1); jmag = np.linalg.norm(j, axis=1)
    T = np.linalg.norm(a - G, axis=1)                            # specific thrust (per unit mass)
    P = T ** 1.5                                                 # momentum-theory specific power (∝)
    return dict(steps=len(t), dur_s=float(t[-1] - t[0]),
                brate_mean=float(w.mean()), brate_peak=float(w.max()),
                accel_mean=float(amag.mean()), jerk_mean=float(jmag.mean()),
                spec_thrust_mean=float(T.mean()), spec_power_mean=float(P.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=_REPO + "/results/codesign_feedback/crash_verify/**/figure_data.npz")
    ap.add_argument("--out", default=_REPO + "/results/codesign_feedback/flight_effort_energy.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob, recursive=True))
    rows = []
    for f in files:
        cond = os.path.basename(os.path.dirname(f)).replace("_figdata", "")
        m = metrics(f); m["flight"] = cond; rows.append(m)
    if not rows:
        print("no figure_data.npz traces found under", a.glob); return
    cols = ["flight", "steps", "dur_s", "brate_mean", "brate_peak", "accel_mean", "jerk_mean",
            "spec_thrust_mean", "spec_power_mean"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in cols})
    print(f"{'flight':16} {'dur_s':>6} {'brate_mean':>10} {'brate_peak':>10} {'accel_mean':>10} {'spec_power':>10}")
    for r in rows:
        print(f"{r['flight']:16} {r['dur_s']:>6.1f} {r['brate_mean']:>10.2f} {r['brate_peak']:>10.1f} "
              f"{r['accel_mean']:>10.2f} {r['spec_power_mean']:>10.2f}")
    # ours-vs-baseline ratios (robust to mass/rotor constants), if both present
    xpu = next((r for r in rows if "xpu" in r["flight"]), None)
    for r in rows:
        if r is xpu or xpu is None: continue
        print(f"\n  {r['flight']} vs XPU-RT — control activity {r['brate_mean']/xpu['brate_mean']:.1f}x body-rate, "
              f"{r['accel_mean']/xpu['accel_mean']:.1f}x accel; modeled specific power "
              f"{r['spec_power_mean']/xpu['spec_power_mean']:.2f}x  (ratios are mass/rotor-constant independent)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
