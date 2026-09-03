#!/usr/bin/env python3
"""The feedback loop, iteration by iteration: the schedule Gantt after EACH round, stacked.

Original on top, then every feedback rewrite below, with the newly-changed dispatches highlighted
(IME-routed = hatched teal edge; newly sharded = diagonal hatch) and the metric delta labeled per
round. Each panel is a REAL solved schedule. Predicted from measured K1 profiles, not board traces.
"""
import argparse, json, re
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
             "d": float(v["duration"]), "e": float(v["start_time"]) + float(v["duration"]),
             "impl": v.get("impl", "rvv"), "w": v["hardware_target"].count("+") + 1}
            for v in d.values()]


def deadlines(spec):
    d = json.load(open(spec))["networks"]
    return {n: (float(v.get("period", 0) or 0), float(v.get("window_duration", 0) or 0))
            for n, v in d.items()}


def draw(ax, rows, dl, nets, title, window, hi):
    y = {c: i for i, c in enumerate(CORE_ORDER)}
    for r in rows:
        if r["hw"].split("+")[0] not in y:
            continue
        yy = y[r["hw"].split("+")[0]]
        net, inst = net_of(r["job"], nets)
        col = NETCOLOR.get(net, "#888")
        late = net in dl and r["e"] > inst * dl[net][0] + dl[net][1] + 1e-6
        hatch = None; ec = "none"; lw = 0
        if hi == "ime" and r["impl"] == "ime":
            hatch = "///"; ec = "#0a6b6b"; lw = 0.8      # newly IME-routed
        elif hi == "shard" and r["w"] > 1:
            hatch = "xx"; ec = "#333"; lw = 0.6          # newly sharded
        if late:
            ec = "#d11"; lw = 1.8
        ax.barh(yy, r["d"], left=r["s"], height=0.72, color=col, edgecolor=ec,
                linewidth=lw, hatch=hatch, zorder=3)
    ax.axhline(3.5, color="0.5", lw=0.8, zorder=1)
    for t in (dl.get("ffn_block", (0, 0))[1],):          # ffn deadline line
        if t:
            ax.axvline(t, color="#c33", lw=1.0, ls=(0, (5, 3)), zorder=2)
            ax.text(t, -0.9, "ffn deadline", color="#c33", fontsize=7, ha="center")
    ax.set_yticks(list(y.values())); ax.set_yticklabels([c.replace("CPU_", "") for c in CORE_ORDER], fontsize=7.5)
    ax.set_ylim(-1.1, len(CORE_ORDER) - 0.4); ax.set_xlim(0, window); ax.invert_yaxis()
    ax.set_title(title, fontsize=10.5, weight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="'TITLE|highlight(none|ime|shard)|schedule.json', repeatable top→bottom")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--out", default="results/codesign_feedback/feedback_evolution")
    a = ap.parse_args()
    dl = deadlines(a.spec); nets = list(dl.keys())
    panels = []
    for p in a.panel:
        title, hi, path = p.split("|", 2)
        m = json.load(open(path.replace(".json", "_metrics.json")))
        panels.append((title, hi, load(path), m))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (title, hi, rows, m) in zip(axes, panels):
        misses = m.get("deadline_miss_count", "?"); mk = m.get("makespan_ms", 0)
        tag = "✓ ALL DEADLINES MET" if misses == 0 else f"✗ {misses} misses ({m.get('total_lateness_ms',0):.0f} ms late)"
        draw(ax, rows, dl, nets, f"{title}   —   makespan {mk:.1f} ms · {tag}", a.window, hi)
        ax.set_ylabel("K1 cores", fontsize=8.5)
    axes[-1].set_xlabel("time (ms, predicted from measured K1 dispatch profiles — not board traces)", fontsize=10)
    handles = [Patch(fc=NETCOLOR[x], label=x) for x in nets]
    handles += [Patch(fc="0.8", hatch="///", ec="#0a6b6b", label="Δ routed to IME"),
                Patch(fc="0.8", hatch="xx", ec="#333", label="Δ sharded (multi-hart)"),
                Patch(fc="none", ec="#d11", lw=1.8, label="deadline miss")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Co-design capability ablation: ModelBlaster makes more kernel variants available each round "
                 "(RVV → +IME → +multi-hart shard);\nthe scheduler uses the best AVAILABLE kernel — each round "
                 "re-solved and provably better. (Global per-round; targeted per-op loop = feedback_loop_ffn.)",
                 fontsize=11.5, weight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(a.out + ".png", dpi=170, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
