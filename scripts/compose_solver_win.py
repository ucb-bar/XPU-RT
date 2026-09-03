#!/usr/bin/env python3
"""Before/after Gantt: on a genuinely contended workload, EXACT scheduling recovers what greedy drops.

Top  = greedy: misses hard deadlines (late dispatches ringed red, deadline lines drawn).
Bottom = CP-SAT: 0 misses, lower makespan, proven optimal.
Reads the two scheduled_*.json + the spec (for per-instance deadlines). Pure matplotlib.
"""
import argparse, json, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NETCOLOR = {"attn_block": "#4ba3d3", "fused_full": "#c77fa6", "mlp_control": "#2f8f4e",
            "ffn_block": "#e8c033", "yolov8_nano_64x96": "#2f6fb0"}
CORE_ORDER = ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3",
              "CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]


def net_of(job, nets):
    for n in sorted(nets, key=len, reverse=True):
        if job.startswith(n) and (job[len(n):] == "" or job[len(n):].isdigit()):
            return n, int(job[len(n):] or 0)
    m = re.match(r"^(.*?)(\d+)$", job)
    return (m.group(1), int(m.group(2))) if m else (job, 0)


def load(path):
    d = json.load(open(path))["dispatches"]
    return [{"job": v["job_name"], "hw": v["hardware_target"], "s": float(v["start_time"]),
             "d": float(v["duration"]), "e": float(v["start_time"]) + float(v["duration"])}
            for v in d.values()]


def deadlines(spec):
    d = json.load(open(spec))["networks"]
    return {n: (float(v.get("period", 0) or 0), float(v.get("window_duration", 0) or 0))
            for n, v in d.items()}


def draw(ax, rows, dl, nets, title, window):
    y = {c: i for i, c in enumerate(CORE_ORDER)}
    misses = 0
    for r in rows:
        if r["hw"] not in y:
            continue
        net, inst = net_of(r["job"], nets)
        col = NETCOLOR.get(net, "#888")
        late = False
        if net in dl:
            per, win = dl[net]
            if r["e"] > inst * per + win + 1e-6:
                late = True; misses += 1
        ax.barh(y[r["hw"]], r["d"], left=r["s"], height=0.72, color=col,
                edgecolor=("#d11" if late else "none"), linewidth=(1.8 if late else 0), zorder=3)
    # cluster divider
    ax.axhline(3.5, color="0.5", lw=0.8, ls="-", zorder=1)
    # deadline lines for the periodic critical jobs (ffn/attn @20, mlp/fused @4 grid)
    for t in [4, 8, 12, 16, 20]:
        ax.axvline(t, color="#c77", lw=0.7, ls=(0, (4, 3)), zorder=2, alpha=0.6)
    ax.set_yticks(list(y.values())); ax.set_yticklabels([c.replace("CPU_", "") for c in CORE_ORDER], fontsize=8)
    ax.set_ylim(-0.6, len(CORE_ORDER) - 0.4); ax.set_xlim(0, window)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10.5, weight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--cpsat", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--window", type=float, default=31.0)
    ap.add_argument("--out", default="results/codesign_feedback/solver_win")
    ap.add_argument("--title", default=None, help="figure suptitle override")
    ap.add_argument("--annotate", default=None,
                    help="optional 'text@x,y' callout arrow (e.g. an overrunning net); omit for none")
    a = ap.parse_args()
    dl = deadlines(a.spec); nets = list(dl.keys())
    g = load(a.greedy); c = load(a.cpsat)
    gm = json.load(open(a.greedy.replace(".json", "_metrics.json")))
    cm = json.load(open(a.cpsat.replace(".json", "_metrics.json")))

    def report(path):
        try:
            return json.load(open(path.replace(".json", "_report.json")))
        except Exception:
            return {}
    gr, cr = report(a.greedy), report(a.cpsat)

    def verdict(m, rep):
        miss = m["deadline_miss_count"]
        wall = m.get("solver_wall_time_s") or rep.get("solve_wall_s")
        optimal = str(rep.get("solver_status", "")).lower() == "optimal"
        if miss > 0:
            tail = f"✗ INFEASIBLE ({miss} misses, {m['total_lateness_ms']:.1f} ms late)"
        elif optimal:
            tail = f"✓ FEASIBLE, PROVEN OPTIMAL ({wall:.0f} s)" if wall else "✓ FEASIBLE, PROVEN OPTIMAL"
        else:
            tail = f"✓ FEASIBLE ({wall:.0f} s)" if wall else "✓ FEASIBLE"
        return tail

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
    draw(ax1, g, dl, nets,
         f"GREEDY heuristic  —  makespan {gm['makespan_ms']:.2f} ms  ·  {verdict(gm, gr)}",
         a.window)
    draw(ax2, c, dl, nets,
         f"CP-SAT exact  —  makespan {cm['makespan_ms']:.2f} ms  ·  {verdict(cm, cr)}",
         a.window)
    ax1.set_ylabel("K1 cores", fontsize=9); ax2.set_ylabel("K1 cores", fontsize=9)
    ax2.set_xlabel("time (ms, predicted from measured K1 dispatch profiles)", fontsize=10)
    if a.annotate:
        txt, xy = a.annotate.rsplit("@", 1)
        xx, yy = (float(v) for v in xy.split(","))
        ax1.annotate(txt.replace("\\n", "\n"),
                     xy=(xx, yy), xytext=(xx + 0.4, max(0.6, yy - 2.8)), fontsize=8.4,
                     color="#a11", weight="bold",
                     arrowprops=dict(arrowstyle="-|>", color="#a11", lw=1.2))
    handles = [Patch(fc=NETCOLOR.get(n, "#888"), label=n) for n in nets]
    handles.append(Patch(fc="none", ec="#d11", lw=1.8, label="late dispatch (deadline miss)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(a.title or "Exact scheduling recovers what greedy drops — 217-op sensor-fusion + IME workload (~0.9 util)",
                 fontsize=13, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
