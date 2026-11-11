#!/usr/bin/env python3
"""Three-panel gantt for a 3net shape: XPU-RT, ROS naive, ROS oracle.

Two panels are not enough on this arm. The whole point of the isolation-best
baseline is that ROS's answer depends on which placement it gets, and on
`3net_fused2_mlp8_yolo1` the two placements land on OPPOSITE SIDES of the
scheduler: naive is 1.167x (XPU-RT wins), oracle is 0.970x (pinning wins).
A two-panel figure has to pick one and would state the opposite conclusion
depending on which. So all three run in one figure, on one time axis.

Traces are the measured ones on both sides:
    ROS      logs/<shape>__a<N>.log                between ROS_PINSWEEP_TRACE_*
    XPU-RT   logs/xpurt3net/<shape>__<solver>/rep<k>/run.log
                                                  between MODELBLASTER_XPURT_TRACE_*

The marker is the aperiodic completion, drawn per panel, and the times are the
offset-corrected ones (each run timed from its own first dispatch) so that no
runtime is charged for another's start barrier -- see ANALYSIS.md 0.11.

    python3 plot_gantt_3net.py [--shape 3net_fused2_mlp8_yolo1]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)

SURFACE = "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
MARK = "#d03b3b"

P = re.compile(r"([A-Za-z0-9_]+?)@(dsp|hta|cpu|gpu)(?=_[A-Za-z]|$)")
NODE = re.compile(r"node\s+(\S+)\s+backend=(\S+)\s+inst=\s*(\d+)\s+period=\s*(\S+)")


def parse_label(s):
    return {(n.lstrip("_"), l) for n, l in P.findall(s)}


def block(t, a, b):
    i = t.index(a) + len(a)
    return t[i:t.index(b, i)]


def ros_run(shape, want_label):
    """The log whose header reproduces `want_label`, its median rep, its
    aperiodic set, and the offset-corrected completion."""
    import glob
    want = parse_label(want_label)
    for path in sorted(glob.glob(os.path.join(SWEEP, "logs", f"{shape}__a*.log"))):
        txt = open(path).read()
        place, aper = {}, set()
        for m in NODE.finditer(txt):
            place[m.group(1)] = m.group(2)
            if float(m.group(4)) < 0:
                aper.add(m.group(1))
        if set(place.items()) != want:
            continue
        rows = list(csv.DictReader(io.StringIO(
            block(txt, "ROS_PINSWEEP_TRACE_BEGIN ===", "=== ROS_PINSWEEP_TRACE_END").strip())))
        reps = {int(m.group(1)): float(m.group(2)) for m in
                re.finditer(r"\[summary\] rep=(\d+).*?np_makespan=([\d.]+)", txt)}
        med = st.median(reps.values())
        rep = min(reps, key=lambda r: abs(reps[r] - med))
        sel = [r for r in rows if int(r["rep"]) == rep and int(r["warm"]) == 0]
        spans = {}
        for r in sel:
            spans.setdefault(r["backend"], []).append(
                (float(r["start_ms"]), float(r["end_ms"]), r["network"]))
        off = min(float(r["start_ms"]) for r in sel)
        end = max(float(r["end_ms"]) for r in sel if r["network"] in aper)
        return spans, aper, off, end - off
    return None, None, None, None


def xrt_run(shape, tag, aper):
    import glob
    best = []
    for d in sorted(glob.glob(os.path.join(SWEEP, "logs", "xpurt3net", tag, "rep*"))):
        p = os.path.join(d, "run.log")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        rows = list(csv.DictReader(io.StringIO(
            block(txt, "MODELBLASTER_XPURT_TRACE_BEGIN ===",
                  "=== MODELBLASTER_XPURT_TRACE_END").strip())))
        for r in rows:
            r["_s"] = float(r["actual_start_cycles"]) / 1000.0
            r["_e"] = float(r["actual_end_cycles"]) / 1000.0
        off = min(r["_s"] for r in rows)
        end = max((r["_e"] for r in rows if r["network"] in aper), default=0.0)
        best.append((rows, off, end - off))
    if not best:
        return None, None, None
    med = st.median([b[2] for b in best])
    rows, off, np_ms = min(best, key=lambda b: abs(b[2] - med))
    spans = {}
    for r in rows:
        spans.setdefault(r["backend"].lower(), []).append((r["_s"], r["_e"], r["network"]))
    return spans, off, np_ms


def draw(ax, spans, rows, colors, aper, off, np_ms, title, xmax):
    for yi, (lane, net) in enumerate(rows):
        for s, e, n in spans.get(lane, []):
            if n != net:
                continue
            ax.barh(yi, max(e - s, 0.02), left=s - off, height=0.5,
                    color=colors[net], edgecolor=SURFACE, linewidth=0.7,
                    zorder=3, hatch="///" if net in aper else None)
    ax.axvline(np_ms, color=MARK, lw=1.6, ls="--", zorder=4)
    ax.annotate(f"aperiodic done  {np_ms:.2f} ms", xy=(np_ms, len(rows) - 0.45),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                color=MARK, va="center", ha="left", zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{l.upper()}  ·  {n}" for l, n in rows], fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(-xmax * 0.012, xmax * 1.02)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=2)
    ax.set_title(title, fontsize=9.8, color=INK, loc="left", pad=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="3net_fused2_mlp8_yolo1")
    ap.add_argument("--out", default=os.path.join(SWEEP, "plots"))
    a = ap.parse_args()

    rec = [s for s in json.load(open(os.path.join(SWEEP, "results", "analysis_3net.json")))
           ["shapes"] if s["shape"] == a.shape]
    if not rec:
        print(f"  {a.shape}: not in analysis_3net.json")
        return 1
    rec = rec[0]

    iso_spans, aper, iso_off, iso_np = ros_run(a.shape, rec["iso_label"])
    orc_spans, _, orc_off, orc_np = ros_run(a.shape, rec["ros_np_label"])
    if iso_spans is None or orc_spans is None:
        print(f"  {a.shape}: could not resolve both ROS placements")
        return 1
    x_spans, x_off, x_np = xrt_run(a.shape, rec["xrt_warmbest_via"], aper)
    if x_spans is None:
        print(f"  {a.shape}: no XPU-RT trace under {rec['xrt_warmbest_via']}")
        return 1

    nets = sorted({n for sp in (iso_spans, orc_spans, x_spans)
                   for v in sp.values() for _, _, n in v})
    if len(nets) > len(SLOTS):
        print(f"  {a.shape}: {len(nets)} networks exceeds the 3-slot cap")
        return 1
    colors = {n: SLOTS[i] for i, n in enumerate(nets)}
    rows = sorted({(lane, n) for sp in (iso_spans, orc_spans, x_spans)
                   for lane, v in sp.items() for _, _, n in v},
                  key=lambda t: (t[0], nets.index(t[1])))
    xmax = max(e - o for sp, o in ((iso_spans, iso_off), (orc_spans, orc_off),
                                   (x_spans, x_off))
               for v in sp.values() for _, e, _ in v)

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 1.4 + 0.40 * len(rows) * 3),
                             sharex=True, gridspec_kw=dict(hspace=0.40))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    draw(axes[0], x_spans, rows, colors, aper, x_off, x_np,
         "XPU-RT — per-op scheduling, `cpsat:warmbest`", xmax)
    draw(axes[1], iso_spans, rows, colors, aper, iso_off, iso_np,
         "ROS, naive — each network on its own best lane in isolation:  "
         + ",  ".join(f"{n}→{l}" for n, l in sorted(parse_label(rec["iso_label"]))), xmax)
    draw(axes[2], orc_spans, rows, colors, aper, orc_off, orc_np,
         "ROS, oracle — best of all 8 legal placements:  "
         + ",  ".join(f"{n}→{l}" for n, l in sorted(parse_label(rec["ros_np_label"]))), xmax)
    axes[2].set_xlabel("ms from that run's own first dispatch", fontsize=9.5, color=INK2)

    fig.suptitle(
        f"{a.shape} — the naive placement loses to the scheduler "
        f"({iso_np/x_np:.2f}×); the oracle placement beats it ({orc_np/x_np:.2f}×)",
        fontsize=12.5, color=INK, x=0.006, ha="left", y=0.985)
    fig.legend(handles=[Patch(facecolor=colors[n], label=n) for n in nets]
               + [Patch(facecolor="#ffffff", edgecolor=MUTED, hatch="///",
                        label="aperiodic (the timed work)")],
               fontsize=8.5, frameon=False, labelcolor=INK2, ncol=len(nets) + 1,
               loc="upper left", bbox_to_anchor=(0.006, 0.948))
    fig.subplots_adjust(left=0.175, right=0.995, top=0.845, bottom=0.088)

    p = os.path.join(a.out, f"gantt_{a.shape}.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {p}")
    print(f"     xrt={x_np:.2f}  naive={iso_np:.2f} ({iso_np/x_np:.3f}×)  "
          f"oracle={orc_np:.2f} ({orc_np/x_np:.3f}×)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
