"""Predicted (XPU-RT) vs actual (K3 board) Gantt per walker condition: predicted
schedule on top, actual execution below, one lane per kind#hart, colored by
network, red border = ran past its predicted finish (the layout of ModelBlaster's
scripts/plot_xpurt_trace.py).

Difference from plot_xpurt_trace.py: actual bars are drawn on the hart that RAN
the dispatch (trace column ran_on_hart), so moved (dynamic) and pooled tasks
appear where they executed, not on their planned lane.

Also draws dronet deadlines (green = met, red = missed) and a summary of
actual-minus-predicted finish per condition.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xpu-rt"))
import walker_sim as W  # noqa: E402

RUNS = OUT = None  # set in main()
SPEC = ROOT / "data/toplevel/networks_k3_mlp1600_dronet480_yolo64_singletons.json"
NETS = ["dronet", "mlp_control", "yolov8_nano"]
COL = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(NETS)}
# (run directory, title) in the order of run_before_after_campaign.sh's conditions
CONDS = [
    ("before_static_walker_run1", "Before: static walker (greedy singletons)"),
    ("ratio_feedback_r1_run1", "Ratio feedback (mean measured/predicted), round 1"),
    ("dynamic_walker_run1", "Dynamic walker (safety 1.0)"),
    ("calibrated_feedback_r1_run1", "Calibrated feedback, round 1 (static walker)"),
    ("global_pool_yolo_run1", "Global pool (YOLO pooled, periodic static)"),
]
WINDOWS = PERIODS = None


def lane_name(h):
    return f"a100#hart{h}" if h >= 8 else f"x100#hart{h}"


def load(run):
    d = RUNS / run
    sched = next(d.glob("scheduled_*.json"))
    trace = next(d.glob("*_trace.csv"))
    tab = W.load_table(sched)
    return tab, W.read_trace(trace, tab)


def misses(tab, tr):
    mm = W.instance_misses(tab, [x["f"] for x in tr], WINDOWS, PERIODS)
    return {n: (v["misses"], v["n"]) for n, v in mm.items()}


def gantt(run, title, t0, t1, path):
    tab, tr = load(run)
    mm = misses(tab, tr)
    lanes = [lane_name(h) for h in list(range(8, 16)) + list(range(0, 8))]
    li = {n: i for i, n in enumerate(lanes)}
    fig, (ap, aa) = plt.subplots(2, 1, figsize=(15, 8.2), sharex=True,
                                 gridspec_kw={"hspace": 0.32})
    late = 0
    for e, t in zip(tab, tr):
        if t["zero"]:
            continue
        pe = e.ps + e.pd
        if pe >= t0 and e.ps <= t1:
            ap.barh(li[lane_name(e.hart)], e.pd, left=e.ps, height=0.72,
                    color=COL[e.net], edgecolor="black", linewidth=0.3)
        if t["f"] >= t0 and t["s"] <= t1:
            over = t["f"] > pe + 0.001
            late += over
            aa.barh(li[lane_name(t["ran_on"])], t["f"] - t["s"], left=t["s"], height=0.72,
                    color=COL[e.net], edgecolor="red" if over else "black",
                    linewidth=1.2 if over else 0.3)
    # dronet deadlines in the window: green = met, red = missed
    last = {}
    for e, t in zip(tab, tr):
        if e.net == "dronet" and not t["zero"]:
            last[e.inst] = max(last.get(e.inst, 0.0), t["f"])
    for k, f in last.items():
        dl = k * PERIODS["dronet"] + WINDOWS["dronet"]
        if t0 <= dl <= t1:
            c = "tab:red" if f > dl + 1e-9 else "tab:green"
            for ax in (ap, aa):
                ax.axvline(dl, color=c, lw=1.0, alpha=0.8)
            aa.text(dl, len(lanes) - 0.4, f"#{k}", color=c, fontsize=7, ha="center", va="bottom")
    for ax, sub in ((ap, "XPU-RT predicted schedule"),
                    (aa, "Actual execution on K3 board (red border = ran past predicted finish)")):
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes, fontsize=7)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.6, len(lanes) - 0.1)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(sub, fontsize=10)
    aa.set_xlabel("time (ms)")
    miss_txt = "   ".join(f"{n} {m}/{k}" for n, (m, k) in mm.items())
    fig.suptitle(f"{title}\nmisses (whole 62.5 ms run): {miss_txt}   |   dronet deadlines: green = met, red = missed",
                 fontsize=11, y=0.995)
    fig.legend(handles=[Patch(color=COL[n], label=n) for n in NETS], loc="upper right",
               fontsize=8, ncol=3, bbox_to_anchor=(0.99, 0.955))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summary(path):
    """Actual minus predicted finish per dispatch, and misses, per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    data, labels, colors = [], [], []
    rows = []
    conds = [(r, t) for r, t in CONDS if (RUNS / r).is_dir()]
    for run, title in conds:
        tab, tr = load(run)
        for n in NETS:
            xs = [t["f"] - (e.ps + e.pd) for e, t in zip(tab, tr) if e.net == n and not t["zero"]]
            data.append(xs)
            colors.append(COL[n])
        labels.append(title.split("(")[0].strip().replace(", round", "\nround"))
        rows.append((title, misses(tab, tr)))
    pos = []
    for i in range(len(conds)):
        pos += [i * 4 + j for j in range(3)]
    bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c)
        b.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([i * 4 + 1 for i in range(len(conds))])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("actual finish - predicted finish (ms)")
    ax.set_title("Discrepancy from the predicted schedule, per dispatch (box = IQR, whiskers = 1.5 IQR)", fontsize=10)
    ax.legend(handles=[Patch(color=COL[n], label=n) for n in NETS], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    w = 0.26
    for j, n in enumerate(NETS):
        vals = [100 * r[1][n][0] / r[1][n][1] for r in rows]
        bars = ax.bar([i + (j - 1) * w for i in range(len(rows))], vals, w, color=COL[n], label=n)
        for b, (m, k) in zip(bars, [r[1][n] for r in rows]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{m}", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([l for l in labels], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("instances missing their deadline (%)")
    ax.set_title("Deadline misses on the K3 (count above bar)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global RUNS, OUT, SPEC, WINDOWS, PERIODS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="campaign output directory")
    ap.add_argument("--out", type=Path, help="plot directory (default: <runs>/plots)")
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--window", type=float, nargs=2, default=(0.0, 12.0), metavar=("T0", "T1"),
                    help="zoomed window in ms (default 0 12)")
    a = ap.parse_args()
    RUNS, OUT, SPEC = a.runs, a.out or a.runs / "plots", a.spec
    WINDOWS, PERIODS = W.spec_windows(SPEC)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (run, title) in enumerate(CONDS):
        if not (RUNS / run).is_dir():
            print(f"skip {run}: not in {RUNS}")
            continue
        gantt(run, title, a.window[0], a.window[1], OUT / f"{i + 1}_{run}_{a.window[0]:g}-{a.window[1]:g}ms.png")
        gantt(run, title, 0.0, 62.5, OUT / f"{i + 1}_{run}_full.png")
    summary(OUT / "0_summary.png")
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
