#!/usr/bin/env python3
"""Aggregate the environment sweep: per (course, density, arm, speed) the success fraction, gates
cleared, and — from every flight's commanded wrench through the rotor model — the mean propulsive
power and the mean commanded moment, with the arm's cadence trace named next to each cell.

    scripts/env_sweep_summary.py [--csv results/codesign_feedback/campaign_env/env_sweep.csv]
                                 [--records results/codesign_feedback/campaign_env/records]
                                 [--out results/codesign_feedback/campaign_env/env_sweep_summary.csv]
                                 [--plot results/codesign_feedback/refined/env_sweep.png]
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, statistics, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from flight_energy_model import rotor_thrusts   # noqa: E402

ARM_OF = {"xpu_a_cpsat_hard.csv": "XPU-RT CP-SAT", "xpu_a_greedy.csv": "XPU-RT greedy",
          "ros_vanilla445.csv": "ROS 2 vanilla (4-hart YOLO)", "ros_vanilla4t45.csv": "ROS 2 vanilla (timer)", "ros_vanilla45.csv": "ROS 2 as shipped",
          "ros_vanilla4x245.csv": "ROS 2 vanilla, two YOLO nodes", "ros_vanilla4tm45.csv": "ROS 2 vanilla, control timer", "ros_p345.csv": "ROS 2 tuned (p3)", "ros_p3_q145.csv": "ROS 2 tuned (p3), QoS 1", "ros_vanilla4_q145.csv": "ROS 2 vanilla, QoS depth 1"}


def flight_power(npz, arm=0.09, kappa=0.016):
    d = np.load(npz, allow_pickle=True)
    w = d["wrench"].astype(float).copy(); t = d["t_s"].astype(float)
    if len(t) < 2 or np.allclose(w, 0):
        return None
    w[:, 0] = np.clip(1.0 + w[:, 0], 0.0, None)
    P = (rotor_thrusts(w, arm, kappa) ** 1.5).sum(axis=1)
    dt = np.gradient(t)
    return {"power": float((P * dt).sum() / max(t[-1] - t[0], 1e-6)), "absM": float(np.abs(w[:, 1:4]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{REPO}/results/codesign_feedback/campaign_env/env_sweep.csv")
    ap.add_argument("--records", default=f"{REPO}/results/codesign_feedback/campaign_env/records")
    ap.add_argument("--out", default=f"{REPO}/results/codesign_feedback/campaign_env/env_sweep_summary.csv")
    ap.add_argument("--plot", default=f"{REPO}/results/codesign_feedback/refined/env_sweep.png")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r.get("course", "a"), float(r["prop_density"]), r["ctrl_trace"], float(r["cruise_speed"]), float(r["moment_scale"]))].append(r)
    # per-flight power from the records: records/<arm>_<course>_d<dens>_c<cru>/ep<k>_s<seed>.npz
    power = {}
    for f in glob.glob(os.path.join(a.records, "*", "*.npz")):
        d = np.load(f, allow_pickle=True)
        key = (str(d["course"]), float(d["prop_density"]), os.path.basename(str(d["ctrl_trace"])), float(d["cruise_speed"]), float(d["moment_scale"]), int(d["seed"]))
        p = flight_power(f)
        if p:
            power[key] = p
    out = []
    for (course, dens, tr, cru, gain), rs in sorted(cells.items()):
        n = len(rs); succ = sum(r["outcome"] == "success" for r in rs)
        gates = collections.Counter(int(float(r["gates_passed"])) for r in rs)
        entered = sum(1 for r in rs if r["outcome"] == "crash" and 1 <= int(float(r["gates_passed"])) <= 2)
        pw = [power[(course, dens, tr, cru, gain, int(r["seed"]))]["power"] for r in rs if (course, dens, tr, cru, gain, int(r["seed"])) in power]
        mm = [power[(course, dens, tr, cru, gain, int(r["seed"]))]["absM"] for r in rs if (course, dens, tr, cru, gain, int(r["seed"])) in power]
        out.append({"course": course, "prop_density": dens, "arm": ARM_OF.get(tr, tr), "ctrl_trace": tr, "cruise": cru, "moment_scale": gain,
                    "n": n, "success": succ, "success_frac": round(succ / n, 3), "entered_then_crashed": entered,
                    "gates_hist": " ".join(f"{k}:{v}" for k, v in sorted(gates.items())),
                    "mean_power": round(statistics.mean(pw), 4) if pw else "", "mean_absM": round(statistics.mean(mm), 5) if mm else "", "n_power": len(pw)})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"wrote {a.out}: {len(out)} cells")
    for o in out:
        print(f"{o['course']} d{o['prop_density']:.2f} {o['arm']:<28} c{o['cruise']:.1f} g{o['moment_scale']:<7} {o['success']:>2}/{o['n']:<2} gates {o['gates_hist']:<18} power {o['mean_power']} |M| {o['mean_absM']}")
    # plot: success vs speed per arm, one panel per (course, density)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    envs = sorted({(o["course"], o["prop_density"]) for o in out})
    arms = sorted({o["arm"] for o in out})
    cols = {"XPU-RT CP-SAT": "#1f9e5a", "XPU-RT greedy": "#7fb069", "ROS 2 vanilla (4-hart YOLO)": "#e2231a", "ROS 2 vanilla (timer)": "#b03018", "ROS 2 as shipped": "#5c0b08",
            "ROS 2 vanilla, two YOLO nodes": "#f28c28", "ROS 2 vanilla, QoS depth 1": "#c2185b"}
    fig, axes = plt.subplots(1, max(1, len(envs)), figsize=(4.2 * max(1, len(envs)), 3.6), sharey=True, squeeze=False)
    for ax, env in zip(axes[0], envs):
        for arm in arms:
            pts = sorted((o["cruise"], o["success_frac"], o["n"]) for o in out if (o["course"], o["prop_density"]) == env and o["arm"] == arm and abs(o["moment_scale"] - 0.0055) < 1e-9)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=cols.get(arm, "#888"), label=arm, lw=1.8, ms=5)
        ax.set_title(f"course {env[0].upper()} · density {env[1]:.2f}", fontsize=10, weight="bold", loc="left")
        ax.set_xlabel("cruise speed (m/s)"); ax.set_ylim(-0.03, 1.03); ax.grid(ls=":", lw=0.6)
    axes[0][0].set_ylabel("success fraction (12 seeds)"); axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Where each runtime crashes: success against cruise speed, per environment, replayed K1 cadences, people 2.4 m", fontsize=11, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.plot, dpi=200, bbox_inches="tight"); print("wrote", a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
