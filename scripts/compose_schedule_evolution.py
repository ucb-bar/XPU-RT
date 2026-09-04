#!/usr/bin/env python3
"""The schedule-evolution MEGA plot: the onboard schedule Gantt after each co-design step, stacked, with
the makespan delta + a one-line 'what changed & why' between panels. Each panel is a REAL solved schedule.
Story: og -> +sharding -> +unfuse (graph rewrite) -> runtime feedback (board-calibrated re-solve) -> +other.
Honest: shows accept/reject per lever; nothing faked. Shards span ALL harts they occupy.
"""
import argparse, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D

NETCOLOR = {"attn_block": "#4ba3d3", "fused_full": "#c77fa6", "mlp_control": "#2f8f4e",
            "ffn_block": "#e8c033", "yolov8_nano": "#2f6fb0", "yolov8_nano_64x96": "#2f6fb0",
            "dronet": "#e07a3f"}
CORE_ORDER = ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3", "CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]
NICE = {"mlp_control": "CTRL", "fused_full": "NAV", "ffn_block": "FFN", "attn_block": "ATTN",
        "yolov8_nano_64x96": "YOLO", "yolov8_nano": "YOLO"}     # friendly net names for the miss labels


def net_of(job, nets):
    for n in sorted(nets, key=len, reverse=True):
        if job.startswith(n) and (job[len(n):] == "" or job[len(n):].isdigit()):
            return n, int(job[len(n):] or 0)
    m = re.match(r"^(.*?)(\d+)$", job)
    return (m.group(1), int(m.group(2))) if m else (job, 0)


def load(path):
    d = json.load(open(path))["dispatches"]
    return [{"job": v["job_name"], "harts": v["hardware_target"].split("+"), "s": float(v["start_time"]),
             "d": float(v["duration"]), "e": float(v["start_time"]) + float(v["duration"]),
             "impl": v.get("impl", "rvv"), "w": v["hardware_target"].count("+") + 1} for v in d.values()]


def deadlines(spec):
    d = json.load(open(spec))["networks"]
    return {n: (float(v.get("period", 0) or 0), float(v.get("window_duration", 0) or 0)) for n, v in d.items()}


def build_remap(P, gap_min=9.0, gap_vis=6.0):
    """Compress idle time (no dispatch on ANY core/panel) so the Gantt shows only where work happens.
    One shared remap across every panel so the time axis stays consistent. Returns (remap, xmax, breaks, merged)."""
    ivs = []
    for entry in P:
        rows = entry[2]
        for r in rows:
            ivs.append((r["s"], r["e"]))
    if not ivs:
        return (lambda t: t), 10.0, [], [(0.0, 10.0)]
    ivs.sort()
    merged = []
    for s, e in ivs:                               # union of busy intervals; gaps < gap_min stay uncompressed
        if merged and s <= merged[-1][1] + gap_min:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    segs, breaks, cur = [], [], 0.0
    for i, (s, e) in enumerate(merged):
        if i > 0:
            gap = s - merged[i - 1][1]
            segs.append((merged[i - 1][1], s, cur, gap_vis / gap))   # compress the idle gap to gap_vis wide
            breaks.append((cur, cur + gap_vis))
            cur += gap_vis
        segs.append((s, e, cur, 1.0))              # busy region: full (1:1) scale
        cur += (e - s)

    def remap(t):
        if t <= segs[0][0]:
            return 0.0
        for o0, o1, n0, sc in segs:
            if o0 <= t <= o1 + 1e-9:
                return n0 + (t - o0) * sc
        return cur
    return remap, cur, breaks, [(a, b) for a, b in merged]


def gen_ticks(merged, remap, step=25.0):
    """Real-time tick labels, but only where time actually flows (inside busy regions); remapped to screen x."""
    tmax = merged[-1][1]; ticks, labels = [], []; t = 0.0
    while t <= tmax + 1e-6:
        if any(o0 - 1e-6 <= t <= o1 + 1e-6 for o0, o1 in merged):
            ticks.append(remap(t)); labels.append(f"{t:.0f}")
        t += step
    return ticks, labels


def draw(ax, rows, dl, nets, hi, remap, xmax, breaks, feedback):
    y = {c: i for i, c in enumerate(CORE_ORDER)}
    misses = 0
    if feedback:
        ax.set_facecolor("#fbf3ec")   # runtime-feedback panel tinted distinct
    for bx0, bx1 in breaks:           # shade + dash the compressed-idle columns so the break is explicit
        ax.axvspan(bx0, bx1, color="0.90", zorder=0)
        ax.plot([(bx0 + bx1) / 2] * 2, [-0.7, len(CORE_ORDER) - 0.3], ls=(0, (2, 2)), lw=0.7, color="0.62", zorder=1)
    late_marks = []; missed = {}
    for r in rows:
        net, inst = net_of(r["job"], nets)
        col = NETCOLOR.get(net, "#9aa"); late = net in dl and dl[net][0] and r["e"] > inst * dl[net][0] + dl[net][1] + 1e-6
        if late:
            misses += 1; missed[NICE.get(net, net)] = missed.get(NICE.get(net, net), 0) + 1
        hatch = None; ec = "white"; lw = 0.3     # hairline separators so dense back-to-back dispatches read as texture
        if hi == "ime" and r["impl"] == "ime":
            hatch = "///"; ec = "#0a6b6b"; lw = 0.7
        elif hi == "shard" and r["w"] > 1:
            hatch = "xxx"; ec = "#2a2a2a"; lw = 0.5
        if late:
            ec = "#d11"; lw = 2.2
        x0 = remap(r["s"]); wd = max(remap(r["e"]) - x0, 0.2)
        for h in r["harts"]:                       # span EVERY hart the dispatch occupies
            if h in y:
                ax.barh(y[h], wd, left=x0, height=0.82, color=col, edgecolor=ec,
                        linewidth=lw, hatch=hatch, zorder=3)
                if late:
                    late_marks.append((x0 + wd / 2, y[h]))
    for mx, my in late_marks:                      # a bold red ✗ over each missed-deadline dispatch
        ax.scatter([mx], [my], marker="X", s=300, color="#e60000", edgecolors="white", linewidths=1.8, zorder=11)
    ax.axhline(3.5, color="0.55", lw=0.8, zorder=1)
    ax.set_yticks(list(y.values())); ax.set_yticklabels([c.replace("CPU_", "") for c in CORE_ORDER], fontsize=10)
    ax.set_ylim(-0.7, len(CORE_ORDER) - 0.3); ax.set_xlim(0, xmax); ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return misses, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", default=[], help="'TITLE|highlight(none|ime|shard)|sched.json'")
    ap.add_argument("--panels-json", default=None, help="JSON file with a list of panel strings")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--metric", default="makespan_ms", choices=["makespan_ms", "worst_critical_response_ms"])
    ap.add_argument("--title", default="Co-design schedule evolution — the onboard K1 schedule after each optimization")
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/schedule_evolution_mega"))
    a = ap.parse_args()
    panels = list(a.panel)
    if a.panels_json:
        panels += json.load(open(a.panels_json))
    dl = deadlines(a.spec); nets = list(dl.keys())
    P = []
    for p in panels:
        parts = p.split("|")
        title, hi, path = parts[0], parts[1], parts[2]
        driver = parts[3] if len(parts) > 3 else ""      # optional: the closed-loop action that PRODUCED this panel
        m = json.load(open(path.replace(".json", "_metrics.json")))
        tl = title.lower()
        P.append((title, hi, load(path), m, any(w in tl for w in ("feedback", "runtime", "board")), driver))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    n = len(P)
    # each panel gets its OWN tight x-axis so it fills the full width (no wasted whitespace from the longest
    # panel) — sized larger for single-column legibility.
    fig, axes = plt.subplots(n, 1, figsize=(12.5, 2.85 * n + 1.5), sharex=False)
    if n == 1:
        axes = [axes]
    _late = lambda mm: float(mm.get("total_lateness_ms", mm.get("total_lateness", 0)) or 0)
    prev_mk = None; prev_miss = None; prev_late = None
    for i, (ax, (title, hi, rows, m, fb, driver)) in enumerate(zip(axes, P)):
        mk = m.get("makespan_ms", 0); miss = m.get("deadline_miss_count", 0); late = _late(m)
        remap, xmax, breaks, merged = build_remap([P[i]])     # per-panel idle compression + scale
        miss, missed = draw(ax, rows, dl, nets, hi, remap, xmax, breaks, fb)
        tks, tlbls = gen_ticks(merged, remap)
        ax.set_xticks(tks); ax.set_xticklabels(tlbls, fontsize=11)
        # title (short) + the closed-loop driver on one line above it
        ax.set_title(f"{'①②③④⑤⑥⑦⑧'[min(i,7)]}  {title}", fontsize=15, weight="bold", loc="left", color="#111", pad=4)
        if driver:
            ax.text(0.012, 1.40, "▶ " + driver, transform=ax.transAxes, ha="left", va="center",
                    fontsize=12, style="italic", color="#5a3ea8")
        # HERO badge = the DEADLINE VERDICT (big); makespan is tiny secondary text
        if miss == 0:
            hero = "✓  ALL DEADLINES MET"; bc = "#2f7d4f"
        else:
            who = ", ".join(f"{k}×{v}" for k, v in sorted(missed.items(), key=lambda x: -x[1]))
            hero = f"✗  {miss} DEADLINES MISSED   ({who})"; bc = "#c0392b"
        ax.text(0.5, 1.20, hero, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=16, weight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.42", fc=bc, ec="white", lw=1.4, alpha=0.98))
        ax.set_ylabel("K1 cores", fontsize=12)
        ax.set_xlabel("onboard time (ms)  ·  " + ("measured board costs" if fb else "predicted costs")
                      + f"  ·  makespan {mk:.0f} ms", fontsize=12)
        prev_mk = mk; prev_miss = miss; prev_late = late
    handles = [Patch(fc=NETCOLOR[x], label=x) for x in nets if x in NETCOLOR]
    handles += [Patch(fc="0.8", hatch="xxx", ec="#2a2a2a", label="Δ sharded (multi-hart)"),
                Patch(fc="0.8", hatch="///", ec="#0a6b6b", label="Δ IME-routed"),
                Line2D([0], [0], marker="X", color="#e60000", mec="white", ls="none", ms=11, label="deadline MISSED"),
                Patch(fc="#fbf3ec", ec="0.7", label="runtime-feedback round (board)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(a.title, fontsize=14.5, weight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0.045, 1, 0.978), h_pad=2.6)
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
