#!/usr/bin/env python3
"""The schedule-evolution plot: four real schedule states in a page-width grid.

Story: baseline -> AOT sharding/IME -> K1-calibrated replay -> K1-calibrated
re-solve. Shards span every hart they occupy. The default 2x2 layout is sized
for the top of a two-column paper page; ``--layout vertical`` remains available.
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


def draw(ax, rows, dl, nets, hi, remap, xmax, breaks, feedback, compact=False):
    y = {c: i for i, c in enumerate(CORE_ORDER)}
    misses = 0
    if feedback:
        ax.set_facecolor("#fbf3ec")   # runtime-feedback panel tinted distinct
    for bx0, bx1 in breaks:           # shade + dash the compressed-idle columns so the break is explicit
        ax.axvspan(bx0, bx1, color="0.90", zorder=0)
        ax.plot([(bx0 + bx1) / 2] * 2, [-0.7, len(CORE_ORDER) - 0.3], ls=(0, (2, 2)), lw=0.7, color="0.62", zorder=1)
    # Pass 1 — decide misses at the INSTANCE level (a net-instance misses if its LAST dispatch ends past its
    # deadline). Counting per-dispatch would inflate a single YOLO-instance miss into ~50 (one per dispatch).
    inst_info = {}          # (net,inst) -> dict(ddl, end, y) for its LATEST dispatch
    for r in rows:
        net, inst = net_of(r["job"], nets)
        if not (net in dl and dl[net][0]):
            continue
        ddl = inst * dl[net][0] + dl[net][1]
        hh = [h for h in r["harts"] if h in y]
        info = inst_info.get((net, inst))
        if info is None or r["e"] > info["end"]:
            inst_info[(net, inst)] = {"ddl": ddl, "end": r["e"], "y": (y[hh[0]] if hh else 0), "net": net}
    missing = {k for k, v in inst_info.items() if v["end"] > v["ddl"] + 1e-6}
    missed = {}
    for (net, inst) in missing:
        missed[NICE.get(net, net)] = missed.get(NICE.get(net, net), 0) + 1
    misses = len(missing)
    for r in rows:
        net, inst = net_of(r["job"], nets)
        col = NETCOLOR.get(net, "#9aa")
        # mark the two co-design levers off the DISPATCH itself (so they show in EVERY panel that carries
        # them, not only the panel that introduced them), and OVERLAY them independently so a dispatch that
        # is BOTH sharded and on the IME shows both: shard = left-leaning "\" (black), IME = right-leaning
        # "/" (teal). Each lever is a separate transparent hatch pass over the coloured fill, keeping its
        # own colour where the two overlap (rather than collapsing to a single crosshatch colour).
        overlays = []
        if r["w"] > 1:
            overlays.append(("\\\\\\", "#2a2a2a", 0.6))   # shard
        if r["impl"] == "ime":
            overlays.append(("///", "#0a6b6b", 0.7))       # IME
        x0 = remap(r["s"]); wd = max(remap(r["e"]) - x0, 0.2)
        for h in r["harts"]:                       # span EVERY hart the dispatch occupies
            if h in y:
                # coloured fill with a hairline separator so dense back-to-back dispatches read as texture
                ax.barh(y[h], wd, left=x0, height=0.82, color=col, edgecolor="white", linewidth=0.3, zorder=3)
                for hh, ec, lw in overlays:        # lever hatches, each its own colour, overlaid
                    ax.barh(y[h], wd, left=x0, height=0.82, facecolor="none", edgecolor=ec,
                            linewidth=lw, hatch=hh, zorder=3.4)
    # show WHERE each miss happens: a red deadline LINE, a hatched OVERRUN bar (deadline→finish), and the ✗
    ylo, yhi = -0.7, len(CORE_ORDER) - 0.3
    for k in sorted(missing, key=lambda kk: inst_info[kk]["ddl"]):
        v = inst_info[k]; xd = remap(v["ddl"]); xe = remap(v["end"]); yy = v["y"]
        ax.plot([xd, xd], [ylo, yhi], color="#e60000", ls=(0, (5, 3)), lw=1.5, alpha=0.85, zorder=8)  # deadline
        ax.barh(yy, max(xe - xd, 0.25), left=xd, height=0.82, facecolor="none", edgecolor="#e60000",
                hatch="////", linewidth=1.6, zorder=10)                                               # overrun
        ax.scatter([xe], [yy], marker="X", s=(90 if compact else 280), color="#e60000",
                   edgecolors="white", linewidths=(1.1 if compact else 1.8), zorder=11)
    ax.axhline(3.5, color="0.55", lw=0.8, zorder=1)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels([c.replace("CPU_", "").replace("#", "") for c in CORE_ORDER],
                       fontsize=(5.6 if compact else 8))
    ax.tick_params(axis="y", pad=1.5, length=0)
    # tiny right headroom so an overrun ✗ marker sitting at the makespan isn't clipped by the spine
    ax.set_ylim(-0.7, len(CORE_ORDER) - 0.3); ax.set_xlim(0, xmax * 1.025); ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return misses, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", default=[],
                    help="'TITLE|highlight(none|ime|shard|shard+ime)|sched.json[|DRIVER]'")
    ap.add_argument("--panels-json", default=None, help="JSON file with a list of panel strings")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--metric", default="makespan_ms", choices=["makespan_ms", "worst_critical_response_ms"])
    ap.add_argument("--title", default="Co-design schedule evolution — the onboard K1 schedule after each optimization")
    ap.add_argument("--layout", choices=["grid", "vertical"], default="grid",
                    help="grid is a 2x2 two-column-page figure; vertical preserves the legacy stack")
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(_repo, "results/codesign_feedback/schedule_evolution_mega"))
    ap.add_argument("--height", type=float, default=3.7, help="grid figure height (in); smaller = less tall")
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
        P.append((title, hi, load(path), m,
                  any(w in tl for w in ("feedback", "runtime", "board", "calibrated")),
                  driver))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    n = len(P)
    if a.layout == "grid":
        if n > 4:
            raise ValueError("the compact grid supports at most four feedback stages")
        fig, axgrid = plt.subplots(2, 2, figsize=(7.16, a.height), sharex=False)
        axes = list(axgrid.flat)
        for ax in axes[n:]:
            ax.set_visible(False)
        axes = axes[:n]
    else:
        fig, axes = plt.subplots(n, 1, figsize=(3.95, 2.12 * n + 0.35), sharex=False)
        if n == 1:
            axes = [axes]
    _late = lambda mm: float(mm.get("total_lateness_ms", mm.get("total_lateness", 0)) or 0)
    prev_mk = None; prev_miss = None; prev_late = None
    rendered = []
    # panel 1 (baseline) keeps its own WIDER time scale; panels 2..n share one tighter scale so they
    # fill their axes to their own makespan instead of looking shrunk under the baseline's long tail.
    remap_first = build_remap([P[0]]) if a.layout == "grid" else None
    remap_rest = (build_remap(P[1:]) if len(P) > 1 else remap_first) if a.layout == "grid" else None
    for i, (ax, (title, hi, rows, m, fb, driver)) in enumerate(zip(axes, P)):
        mk = m.get("makespan_ms", 0); miss = m.get("deadline_miss_count", 0); late = _late(m)
        if a.layout == "grid":
            remap, xmax, breaks, merged = remap_first if i == 0 else remap_rest
        else:
            remap, xmax, breaks, merged = build_remap([P[i]])
        miss, missed = draw(ax, rows, dl, nets, hi, remap, xmax, breaks, fb,
                            compact=(a.layout == "grid"))
        tks, tlbls = gen_ticks(merged, remap)
        ax.set_xticks(tks); ax.set_xticklabels(tlbls, fontsize=(7 if a.layout == "grid" else 8))
        if a.layout == "grid" and i == 0:            # panel 1 has a WIDER time scale than 2-4 — flag it
            ax.spines["bottom"].set_color("#d1720b"); ax.spines["bottom"].set_linewidth(2.2)
            ax.tick_params(axis="x", colors="#d1720b")
            ax.text(0.0, -0.30, "wider time scale", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.6, style="italic", weight="bold", color="#d1720b", clip_on=False)
        verdict_c = "#2f7d4f" if miss == 0 else "#c0392b"
        # step number in a verdict-coloured circle (far left) — makes the iteration step obvious
        ax.text(0.012, 1.15, str(i + 1), transform=ax.transAxes,
                fontsize=(8.5 if a.layout == "grid" else 12), weight="bold",
                color="white", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.26", fc=verdict_c, ec="none"))
        ax.text(0.072, 1.15, title, transform=ax.transAxes,
                fontsize=(7.4 if a.layout == "grid" else 9.2), weight="bold", color="#111",
                ha="left", va="center")
        if driver:
            ax.text(0.075, 1.02, "▶ " + driver, transform=ax.transAxes, ha="left", va="center",
                    fontsize=(6.8 if a.layout == "grid" else 8.0), style="italic", color="#5a3ea8",
                    clip_on=False)
        # verdict + makespan as ONE compact badge in the header strip (above the gantt), top-right —
        # keeps the schedule interior clean instead of floating labels beside the bars
        hero = "✓ ALL MET" if miss == 0 else f"✗ {miss} MISSED"
        ax.text(0.988, 1.36, f"{hero}  ·  {mk:.0f} ms", transform=ax.transAxes, ha="right", va="center",
                fontsize=(7.0 if a.layout == "grid" else 8.4), weight="bold", color="white", zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc=verdict_c, ec="white", lw=1.1))
        rendered.append({"stage": i + 1, "title": title, "driver": driver,
                         "cost_source": "K1-calibrated model" if fb else "AOT model",
                         "makespan_ms": mk,
                         "displayed_instance_deadline_misses": miss,
                         "missed_instances_by_network": missed})
        prev_mk = mk; prev_miss = miss; prev_late = late
    _sl = {"yolov8_nano_64x96": "yolov8n"}
    handles = [Patch(fc=NETCOLOR[x], label=_sl.get(x, x)) for x in nets if x in NETCOLOR]
    handles += [Patch(fc="0.8", hatch="\\\\\\", ec="#2a2a2a", label="shard \\"),
                Patch(fc="0.8", hatch="///", ec="#0a6b6b", label="IME /"),
                Line2D([0], [0], color="#e60000", ls=(0, (5, 3)), lw=1.5, label="deadline"),
                Patch(fc="none", ec="#e60000", hatch="////", label="overrun"),
                Line2D([0], [0], marker="X", color="#e60000", mec="white", ls="none", ms=8, label="missed"),
                Patch(fc="#fbf3ec", ec="0.7", label="board round")]
    if a.layout == "grid":
        fig.supylabel("K1 cores", x=0.012, fontsize=8.5)
        fig.supxlabel("onboard time (ms)", y=0.12, fontsize=8.5)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.0, frameon=False,
                   bbox_to_anchor=(0.5, 0.02), columnspacing=0.7, handletextpad=0.3, handlelength=1.3)
        fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.205,
                            wspace=0.17, hspace=0.8)
        fig.savefig(a.out + ".png", dpi=300)
        fig.savefig(a.out + ".pdf")
    else:
        axes[-1].set_xlabel("onboard time (ms)", fontsize=9)
        for ax in axes:
            ax.set_ylabel("K1 cores", fontsize=9, labelpad=1.5)
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.004), columnspacing=1.1, handletextpad=0.4)
        fig.tight_layout(rect=(0, 0.075, 1, 0.975), w_pad=0.3, h_pad=0.5)
        fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.06)

    with open(a.out + "_metrics.json", "w") as f:
        # Record what was DRAWN, not a literal. This said [7.16, 4.35] for every grid
        # figure, including after --height's default became 3.7 -- so the artifact
        # reported a size the figure did not have, which is the same class of drift as
        # a hand-typed caption.
        json.dump({"layout": a.layout,
                   "figure_size_in": [round(v, 3) for v in fig.get_size_inches()],
                   "height_arg": a.height,
                   "panels": rendered}, f, indent=2)
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
