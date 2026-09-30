#!/usr/bin/env python3
"""Throughput against latency, both runtimes, every measured operating point.

    scripts/plot_throughput_latency.py [--out results/codesign_feedback/refined/throughput_latency]

One point per board run. x = the frame rate actually delivered to control (frames whose result
reached a control output, per second of run); y = camera-release -> control-output latency
(median over warm frames); marker colour = the control-output gap the run held (green = the
100 Hz loop intact, red = starved). XPU-RT points come from xpurt_long/ traces (one per
layout x camera rate), ROS points from ros_traced/ (one per layout x camera rate, replicates
pooled). The camera rate each point was asked for is written beside it.

What the plot is for: a ROS node graph has one operating point per layout; the runtime's
per-dispatch width and frames-in-flight are knobs that move along the curve. Nothing here is
modelled -- every point is a run.
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from make_measured_gantt_pair import read_trace, per_frame_chain   # noqa: E402
RES = "results/codesign_feedback"
HZ = 24e6


def xpurt_points():
    pts = {}
    for t in sorted(glob.glob(f"{RES}/xpurt_long/trace_best*_other_run*.csv") + glob.glob(f"{RES}/xpurt_long/trace_rich*_other_run*.csv") + glob.glob(f"{RES}/xpurt_long/trace_cam2*_other_run*.csv") + glob.glob(f"{RES}/xpurt_long/trace_[ab]*greedy*_other_run*.csv") + glob.glob(f"{RES}/xpurt_long/trace_[ab]*cpsat*_other_run*.csv")):
        lab = re.search(r"trace_(\w+?)_other_run", os.path.basename(t)).group(1)
        rows = read_trace(t)
        if len(rows) < 1000:
            continue                                  # a truncated pull
        man = json.load(open(t.replace("trace_", "manifest_").replace(".csv", ".json")))
        sched = json.load(open(man["schedule"])).get("metadata", {}) if os.path.exists(man.get("schedule", "")) else {}
        # frames offered per second: the explicit tables record the camera period; the solver tables
        # record each periodic network's period
        per = sched.get("camera_period_ms")
        if not per and isinstance(sched.get("periodic_networks"), (dict, list)):
            pn = sched["periodic_networks"]
            ent = pn.get("yolov8_nano_64x96") if isinstance(pn, dict) else next((e for e in pn if str(e.get("name", e.get("network", ""))).startswith("yolov8")), None)
            per = (ent or {}).get("period") or (ent or {}).get("period_ms")
        cam = 1000.0 / float(per or 40.0) * int(sched.get("cameras", 1) or 1)
        ch = per_frame_chain(rows); warm = [c for k, c, _ in ch if k >= 1]
        run_s = (max(r["e"] for r in rows) - min(r["s"] for r in rows)) / 1000.0
        ends = sorted(max(x["e"] for x in v) for (n, _), v in
                      collections.defaultdict(list, {(r["net"], r["inst"]): [x for x in rows if (x["net"], x["inst"]) == (r["net"], r["inst"])]
                                                     for r in rows if r["net"] == "mlp_control"}).items())
        gaps = [ends[i + 1] - ends[i] for i in range(1, len(ends) - 1)]
        lags = []
        for k in sorted({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}):
            fr = [r for r in rows if (r["net"], r["inst"]) == ("yolov8_nano_64x96", k)]
            rel = min((r["rel"] for r in fr if r.get("rel") is not None), default=None)
            if rel is not None and k >= 1:
                lags.append(min(r["s"] for r in fr) - rel)
        p = pts.setdefault(lab, {"cam": cam, "lat": [], "gap": [], "deliv": [], "lag": [], "layout": sched.get("layout", lab)})
        p["lat"] += warm; p["gap"] += gaps; p["deliv"].append(len(warm) / run_s if run_s else 0); p["lag"] += lags
    return pts


def ros_points():
    pts = {}
    for d in sorted(glob.glob(f"{RES}/ros_traced/*_r*/summary.json")):
        s = json.load(open(d)); m = re.match(r"(\d+)_([a-z0-9]+)_r(\d+)", s["tag"])
        if not m or not s.get("gap_mean_ms"):
            continue
        hz, arm = int(m.group(1)), m.group(2)
        man = json.load(open(d.replace("summary.json", "manifest.json")))
        secs = float(man.get("seconds", 20.0)) - 3.0
        p = pts.setdefault((arm, hz), {"cam": hz, "lat": [], "gap": [], "deliv": [], "arm": arm})
        if s.get("e2e_med_ms"):
            p["lat"].append(s["e2e_med_ms"])
        p["gap"].append(s["gap_mean_ms"]); p["deliv"].append(s["n_consumed"] / secs if secs else 0)
    return pts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{RES}/refined/throughput_latency"); ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    X, R = xpurt_points(), ros_points()
    fig, ax = plt.subplots(figsize=(9.2, 6.2)); norm = Normalize(10, 55); cmap = plt.cm.RdYlGn_r
    print(f"{'arm':<22} {'cam':>5} {'delivered':>9} {'latency':>8} {'ctrl gap':>8}")
    def frontier(points):                       # points nobody beats on both throughput and latency
        out = []
        for i, (x, y) in enumerate(points):
            if not any((x2 >= x and y2 <= y and (x2 > x or y2 < y)) for j, (x2, y2) in enumerate(points) if j != i):
                out.append(i)
        return set(out)
    xp = [(lab, p, statistics.mean(p["deliv"]), statistics.median(p["lat"]), statistics.mean(p["gap"])) for lab, p in X.items() if p["lat"]]
    rp = [((arm, hz), p, statistics.mean(p["deliv"]), statistics.median(p["lat"]), statistics.mean(p["gap"])) for (arm, hz), p in R.items() if p["lat"]]
    fx = frontier([(x, y) for _, _, x, y, _ in xp]); fr = frontier([(x, y) for _, _, x, y, _ in rp])
    NAMES = {"vanilla": "vanilla: process per node, serial YOLO", "vanilla4": "vanilla, 4-hart YOLO", "vanilla4t": "vanilla, 4-hart YOLO, timer control",
             "rvanilla": "vanilla + heavier stack",
             "ship": "as shipped", "spin": "4-hart YOLO, 1 thread", "p3": "control own process", "multi": "multi-threaded executor",
             "cspin": "chained control", "cp3": "chained, 3 processes", "yproc": "YOLO own process", "smte": "4-hart YOLO, multi", "nproc": "nav own process", "p8": "8 cores, 3 processes", "cship": "chained, as shipped"}
    xs = sorted(xp, key=lambda t: t[2])
    # frontier lines: what each runtime can reach; one label per XPU-RT frontier point
    for i, (lab, p, x, y, g) in enumerate(xs):
        ax.scatter(x, y, s=150, marker="D", c=[cmap(norm(g))], edgecolors="k", zorder=5)
        if i in fx:
            ax.annotate(f"XPU-RT {p['layout']} · {p['cam']:.0f} Hz cam", (x, y), textcoords="offset points", xytext=(9, 7), fontsize=8.5, color="#1a5d3a", weight="bold")
        print(f"XPU-RT {p['layout']:<15} {p['cam']:5.0f} {x:9.1f} {y:8.1f} {g:8.2f}")
    fxp = sorted([(x, y) for i, (_, _, x, y, _) in enumerate(xs) if i in fx])
    if len(fxp) > 1:
        ax.plot([q[0] for q in fxp], [q[1] for q in fxp], "-", color="#1a5d3a", lw=1.2, alpha=0.6, zorder=3)
    # ROS: one label per layout, at the point where the layout plateaus (its highest delivered rate)
    best_by_arm = {}
    for (arm, hz), p, x, y, g in rp:
        if arm not in best_by_arm or x > best_by_arm[arm][2]:
            best_by_arm[arm] = ((arm, hz), p, x, y, g)
    for (arm, hz), p, x, y, g in sorted(rp, key=lambda t: t[2]):
        ax.scatter(x, y, s=80, marker="o", c=[cmap(norm(g))], edgecolors="#7a1a12", zorder=4, alpha=0.9)
        print(f"ROS {arm:<18} {hz:5d} {x:9.1f} {y:8.1f} {g:8.2f}")
    for arm, ((_, hz), p, x, y, g) in best_by_arm.items():
        if arm in ("vanilla", "vanilla4", "vanilla4t", "rvanilla", "ship", "spin", "p3", "multi", "cspin", "cp3", "yproc"):
            ax.annotate(f"ROS {NAMES.get(arm, arm)}: plateau {x:.0f} fps", (x, y), textcoords="offset points",
                        xytext=(8, -12 if arm not in ("p3", "yproc") else 8), fontsize=7.8, color="#7a1a12")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="D", color="w", markerfacecolor="#888", markeredgecolor="k", markersize=10, label="XPU-RT (one per layout × camera rate)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="#7a1a12", markersize=8, label="ROS 2 (one per layout × camera rate)")],
              loc="upper left", fontsize=9, frameon=False)
    ax.set_xlabel("frames delivered to control (per second, measured)"); ax.set_ylabel("camera release → control output (ms, median)")
    ax.set_yscale("log"); ax.grid(ls=":", lw=0.6); ax.set_title("Throughput against latency on the K1 — every point is a run; colour = control-output gap", fontsize=11, weight="bold", loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = fig.colorbar(sm, ax=ax); cb.set_label("control output gap (ms); 10 = the 100 Hz loop intact")
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
