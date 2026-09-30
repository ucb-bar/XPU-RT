#!/usr/bin/env python3
"""Where each runtime crashes: every recorded flight of one environment drawn on the aisle, one panel
per cruise speed, both arms overlaid — paths as thin lines, the crash point as a cross, the gates as
bars — plus, per arm and speed, the fraction of the course completed (the furthest gate line
reached over the span from the start line to the last gate), a graded companion to the success count.

    scripts/env_crash_map.py [--records results/codesign_feedback/campaign_env/records] [--course a]
                             [--density 0.30] [--out results/codesign_feedback/refined/env_crash_map_a_d0.30.png]
"""
from __future__ import annotations
import argparse, collections, glob, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {"xpu_a_cpsat_hard.csv": ("XPU-RT · CP-SAT", "#1f9e5a"), "ros_vanilla445.csv": ("ROS 2 vanilla", "#e2231a"),
        "xpu_a_greedy.csv": ("XPU-RT · greedy", "#7fb069"), "ros_vanilla4_q145.csv": ("ROS 2 vanilla, QoS 1", "#c2185b"),
        "ros_vanilla4x245.csv": ("ROS 2 vanilla, two YOLO nodes", "#f28c28"),
        "ros_vanilla4tm45.csv": ("ROS 2 vanilla, control timer", "#8e24aa"), "ros_p345.csv": ("ROS 2 tuned (p3)", "#5c6bc0"), "ros_p3_q145.csv": ("ROS 2 tuned (p3), QoS 1", "#26a69a")}


def progress(poses, gates, y0):
    """Fraction of the course covered: furthest y reached between the start line and the last gate."""
    yend = gates[-1, 1]
    return float(np.clip((poses[:, 1].max() - y0) / (yend - y0), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=f"{REPO}/results/codesign_feedback/campaign_env/records")
    ap.add_argument("--course", default="a"); ap.add_argument("--density", type=float, default=0.30)
    ap.add_argument("--gain", type=float, default=0.0055)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{REPO}/results/codesign_feedback/refined/env_crash_map_{a.course}_d{a.density:.2f}.png"
    flights = collections.defaultdict(list)
    gates = None
    for f in glob.glob(os.path.join(a.records, "*", "*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["course"]) != a.course or abs(float(d["prop_density"]) - a.density) > 1e-9 or abs(float(d["moment_scale"]) - a.gain) > 1e-9:
            continue
        tr = os.path.basename(str(d["ctrl_trace"]))
        if tr not in ARMS:
            continue
        gates = d["gates_world"]
        flights[(round(float(d["cruise_speed"]), 2), tr)].append(dict(poses=d["poses"], outcome=str(d["outcome"]), gates=int(d["gates_passed"]), seed=int(d["seed"])))
    if not flights:
        raise SystemExit("no recorded flights for that environment")
    speeds = sorted({k[0] for k in flights}); arms = [t for t in ARMS if any(k[1] == t for k in flights)]
    y0 = min(fl["poses"][0, 1] for v in flights.values() for fl in v)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(speeds) + 1, figsize=(2.3 * len(speeds) + 3.6, 6.2), gridspec_kw={"width_ratios": [1] * len(speeds) + [1.6]})
    table = []
    for ax, cru in zip(axes[:-1], speeds):
        for g in gates:   # the frame: two posts either side of the 1.5 m opening (the opening is the gap)
            for sgn in (-1, 1):
                ax.plot([g[0] + sgn * 0.75, g[0] + sgn * 0.89], [g[1], g[1]], color="#f2a900", lw=5, solid_capstyle="butt", zorder=1)
        for tr in arms:
            name, col = ARMS[tr]; fl = flights.get((cru, tr), [])
            for f_ in fl:
                p = f_["poses"]
                ax.plot(p[:, 0], p[:, 1], color=col, lw=0.7, alpha=0.55, zorder=2)
                if f_["outcome"] == "crash":
                    ax.plot(p[-1, 0], p[-1, 1], "x", color=col, ms=6, mew=1.6, zorder=3)
                gp = f_["gates"]
                if gp < len(gates) and p[:, 1].max() >= gates[gp, 1]:   # flew past the next gate's line without credit: went round the frame
                    i = int(np.argmax(p[:, 1] >= gates[gp, 1]))
                    ax.plot(p[i, 0], p[i, 1], "o", mfc="none", mec=col, ms=7, mew=1.2, zorder=3)
            if fl:
                succ = sum(f_["outcome"] == "success" for f_ in fl)
                prog = np.mean([progress(f_["poses"], gates, y0) for f_ in fl])
                table.append((cru, name, succ, len(fl), prog))
        ax.set_xlim(-10.4, -5.6); ax.set_ylim(y0 - 1.0, gates[-1, 1] + 1.5); ax.set_aspect("equal")
        ax.set_title(f"{cru:.1f} m/s", fontsize=10, weight="bold"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#bbb")
    axb = axes[-1]
    for tr in arms:
        name, col = ARMS[tr]
        pts = sorted((c, p) for c, n, s, k, p in table if n == name)
        ptsS = sorted((c, s / k) for c, n, s, k, p in table if n == name)
        if pts:
            axb.plot([c for c, _ in pts], [p for _, p in pts], "-o", color=col, lw=2, ms=5, label=f"{name}: course completed")
            axb.plot([c for c, _ in ptsS], [p for _, p in ptsS], "--s", color=col, lw=1.2, ms=4, alpha=0.7, label=f"{name}: success fraction")
    axb.set_ylim(-0.03, 1.03); axb.set_xlabel("cruise speed (m/s)"); axb.set_ylabel("fraction (12 seeds)"); axb.grid(ls=":", lw=0.6)
    axb.legend(fontsize=7.5, frameon=False, loc="upper right"); axb.set_title("progress and success", fontsize=10, weight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARMS[t][1], lw=2, label=ARMS[t][0]) for t in arms] + [Line2D([], [], color="#f2a900", lw=3, label="gate"), Line2D([], [], color="k", marker="x", ls="", label="collision (contact sensor)"), Line2D([], [], color="k", marker="o", mfc="none", ls="", label="passed beside a gate (not credited)")]
    axes[0].legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(f"Where each runtime collides — course {a.course.upper()}, obstacle density {a.density:.2f}, people 2.4 m, replayed K1 cadences, gain {a.gain:g}",
                 fontsize=11, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=200, bbox_inches="tight")
    for cru, name, s, k, p in sorted(table):
        print(f"{cru:.1f} m/s  {name:<24} success {s:>2}/{k:<2}  course completed {p:.2f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
