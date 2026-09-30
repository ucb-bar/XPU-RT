#!/usr/bin/env python3
"""Three ROS 2 deployments against the schedule, same kernels, same board.

Two loads: the deployed chain at a 45 Hz camera, and the combined load (two 45 Hz cameras plus
ffn_block 10 Hz and dronet 30 Hz). For each, four arms:

  ROS 2 as deployed        one process, default (single-threaded) executor, YOLO on a 4-hart pool
  ROS 2 multi-threaded     the same, MultiThreadedExecutor
  ROS 2 hand-partitioned   one process per node (per camera under the combined load), cores pinned
  XPU-RT                   the schedule (alt2 at one camera, alt1 with two)

and three measured quantities per arm: control-output gap, camera→control latency, frames the
control loop actually received per second (with the fraction the queues dropped). Every bar is
read from results/codesign_feedback/{ros_traced/<tag>/summary.json, xpurt_long/trace_*.csv} at
plot time; an arm with no run on disk is drawn as "not run".

    scripts/plot_deployment_layers.py [--out results/codesign_feedback/refined/deployment_layers]
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from plot_throughput_latency import xpurt_points  # noqa: E402

# each load: (title, frames offered per second, {ROS bar label: arm tag}, [(XPU-RT bar label, trace label prefix)])
# XPU-RT prefixes pool every replicate whose trace label is <prefix>r<k> (scripts/board_stage2.sh) or <prefix> itself.
LOADS = [
    ("one camera, 45 Hz", 45,
     {"ROS 2 vanilla\n(process per node,\nserial YOLO)": "vanilla", "ROS 2 vanilla\n4-hart YOLO": "vanilla4",
      "ROS 2 vanilla\n4-hart YOLO,\nQoS keep-last-1": "vanilla4_q1", "ROS 2 pipelined\nby hand (2 YOLO\nprocesses)": "vanilla4x2",
      "ROS 2 hand-\npartitioned, pinned": "p3"},
     [("XPU-RT\ngreedy", "agreedy"), ("XPU-RT\nCP-SAT", "acpsat_hard")]),
    ("one camera, 90 Hz + heavier stack", 90,
     {"ROS 2 vanilla\n+ heavier stack": "rvanilla", "ROS 2 vanilla\n4-hart YOLO + stack": "rvanilla4", "ROS 2 hand-\npartitioned + stack": "rp3"},
     [("XPU-RT\ngreedy", "b5greedy"), ("XPU-RT\nCP-SAT", "b5cpsat_hard")]),
]


def ros_arm(arm: str, hz: int):
    """Pool every replicate of one ROS arm at one camera rate."""
    out = {"gap": [], "lat": [], "deliv": [], "drop": [], "n": 0}
    for d in sorted(glob.glob(f"{RES}/ros_traced/{hz}_{arm}_r*/summary.json")):
        s = json.load(open(d)); man = json.load(open(d.replace("summary.json", "manifest.json")))
        if not s.get("gap_mean_ms") or not re.match(rf"{re.escape(hz.__str__())}_{re.escape(arm)}_r\d+$", s["tag"]):
            continue
        secs = float(man.get("seconds", 20.0)) - float(s.get("warmup_ms", 3000)) / 1000.0
        out["gap"].append(s["gap_mean_ms"]); out["lat"].append(s["e2e_med_ms"] or s["e2e_goal_med_ms"])
        out["deliv"].append(s["n_consumed"] / secs); out["n"] += 1
        out["drop"].append(1.0 - s["n_goals"] / s["n_frames"] if s["n_frames"] else 0.0)
    return out if out["n"] else None


_XP = None
def xpurt_arm(prefix: str):
    """Pool the XPU-RT runs whose trace label is the prefix or the prefix followed by r<k>."""
    global _XP
    if _XP is None:
        _XP = xpurt_points()
    ps = [p for lab, p in _XP.items() if p["lat"] and (lab == prefix or re.fullmatch(rf"{re.escape(prefix)}r\d+", lab))]
    if not ps:
        return None
    return {"gap": [statistics.mean(p["gap"]) for p in ps], "lat": [statistics.median(p["lat"]) for p in ps],
            "deliv": [statistics.mean(p["deliv"]) for p in ps], "drop": [0.0], "n": sum(len(p["deliv"]) for p in ps), "offered": ps[0]["cam"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{RES}/refined/deployment_layers"); ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(17.5, 9.2))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.11, hspace=0.32, wspace=0.22)
    C = {"ROS": "#c0392b", "XPU": "#1e8449"}
    for col, (title, offered, ros_arms, xpu_arms) in enumerate(LOADS):
        hz = 45 if offered == 45 else 90
        arms = [(name, ros_arm(tag, hz), "ROS") for name, tag in ros_arms.items()] + [(name, xpurt_arm(pref), "XPU") for name, pref in xpu_arms]
        names = [n for n, _, _ in arms]; xs = range(len(arms))
        for row, (key, ylab, ref, log) in enumerate([("gap", "control-output gap (ms)", 10.0, False),
                                                       ("lat", "camera → control (ms)", None, True),
                                                       ("deliv", "frames delivered to control (/s)", offered, False)]):
            ax = axes[row][col]
            for i, (name, d, kind) in enumerate(arms):
                if d is None:
                    ax.text(i, 0.04, "not run", ha="center", va="bottom", fontsize=8, color="#888", transform=ax.get_xaxis_transform()); continue
                v = statistics.mean(d[key]); ax.bar(i, v, color=C[kind], alpha=0.85 if kind == "XPU" else 0.7, width=0.62, edgecolor="k", lw=0.5)
                txt = f"{v:.1f}" if key != "deliv" else f"{v:.0f}"
                if key == "deliv" and statistics.mean(d["drop"]) > 0.005:
                    txt += f"\n{100 * statistics.mean(d['drop']):.0f}% dropped"
                if key == "gap" and abs(v - 10.0) < 0.05:
                    txt = "10.00"
                ax.annotate(txt, (i, v), textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=8.2,
                            weight="bold" if kind == "XPU" else "normal")
            if ref is not None:
                ax.axhline(ref, color="k", ls="--", lw=0.8, alpha=0.6)
                ax.text(len(arms) - 0.45, ref, "100 Hz loop" if key == "gap" else f"offered {offered:.0f}/s", ha="right", va="bottom", fontsize=7.5, color="#444")
            if log:
                ax.set_yscale("log")
            ax.set_ylabel(ylab, fontsize=9); ax.grid(axis="y", ls=":", lw=0.6); ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8.2)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            vals = [statistics.mean(d[key]) for _, d, _ in arms if d]
            if key == "gap":
                ax.set_ylim(0, max(vals + [12.0]) * 1.3)
            if key == "lat":
                ax.set_ylim(10, max(vals + [100.0]) * 2.2)
            if key == "deliv":
                ax.set_ylim(0, max(vals + [offered]) * 1.35)
            if row < 2:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(title, fontsize=11, weight="bold", loc="left")
        # provenance line: replicates behind each bar
        reps = ", ".join(f"{n.splitlines()[0]}: n={d['n']}" for n, d, _ in arms if d)
        axes[2][col].text(0.0, -0.30, f"runs pooled — {reps}", transform=axes[2][col].transAxes, fontsize=7.2, color="#555")
    fig.suptitle("Same kernels, same K1: ROS 2 as one would write it against XPU-RT's two solvers — ROS 2 can use the cores; the schedule decides when",
                 fontsize=11.5, weight="bold", x=0.01, ha="left")
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png")
    for title, offered, ros_arms, xpu_arms in LOADS:
        print(f"== {title}"); hz = 45 if offered == 45 else 90
        for name, tag in ros_arms.items():
            d = ros_arm(tag, hz)
            print(f"   ROS {tag:<12} " + (f"gap {statistics.mean(d['gap']):6.2f}  lat {statistics.mean(d['lat']):7.1f}  deliv {statistics.mean(d['deliv']):5.1f}/s  dropped {100 * statistics.mean(d['drop']):3.0f}%  n={d['n']}" if d else "not run"))
        for name, pref in xpu_arms:
            d = xpurt_arm(pref)
            print(f"   XPU {pref:<12} " + (f"gap {statistics.mean(d['gap']):6.2f}  lat {statistics.mean(d['lat']):7.1f}  deliv {statistics.mean(d['deliv']):5.1f}/s  n={d['n']}" if d else "not run"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
