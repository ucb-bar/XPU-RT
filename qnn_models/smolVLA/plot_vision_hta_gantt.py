#!/usr/bin/env python3
"""Gantt of the vision encoder's actual hardware acceleration.

This is the one place on the QRB5165 where an accelerator genuinely carries a
whole component. It is worth drawing accurately, because the shape is easy to
misread: **HTA never runs a vision segment.** It runs only the extracted
Conv1x1 kernels, and the surrounding trampoline phases stay on DSP or CPU. A
"segment on HTA" bar would be a fiction; what actually executes is

    dsp_seg_NN_tramp_p0 (DSP) -> conv1 (HTA) -> tramp_p1 (DSP)
                              -> conv2 (HTA) -> tramp_p2 (DSP) -> cpu_seg_NN (CPU)

so the chart shows the real per-dispatch placement from the committed cost
model, with the CPU-only cost of the same work drawn above it for comparison.

Data comes from the emitted results.csv trio (one per backend, real cost on the
backends a dispatch has been measured on, 1e9 sentinel elsewhere), so the chart
cannot disagree with what the scheduler sees.

    python3 plot_vision_hta_gantt.py --out ../../plots/smolvla_vision_hta_gantt.png
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CSV = os.path.join(REPO, "gen/profile/{}/qrb5165_v66/smolvlm_vision_v3_bundles"
                         "/smolvlm_vision_v3_bundles.int8/topo_0/results.csv")
# The CPU-only baseline must come from the UNBUNDLED profile. In the bundle
# CSVs a sub-dispatch (conv1, tramp_p1, ...) carries the 1e9 sentinel on every
# backend but its own, so summing "the CPU column" there silently reproduces
# the mixed schedule instead of a CPU-only one.
CSV_V3 = os.path.join(REPO, "gen/profile/CPU/qrb5165_v66/smolvlm_vision_v3"
                            "/smolvlm_vision_v3.int8/topo_0/results.csv")
SENTINEL = 1e8
LANE = {"HTA": 0.4, "DSP": 1.2, "CPU": 2.0}
COL = {"HTA": "#DD8452", "DSP": "#4C72B0", "CPU": "#B0B7C3"}


def cpu_only_by_segment():
    """{segment_name: CPU-only us} from the unbundled v3 profile."""
    out = {}
    for r in csv.DictReader(open(CSV_V3)):
        v = float(r["mean_time"])
        if v < SENTINEL:
            out[r["module_name"]] = v
    return out


def group_of(module):
    """Map a bundle dispatch back to the v3 segment it decomposes."""
    if module.startswith("cpu_seg_"):
        return module
    return "_".join(module.split("_")[:3])   # dsp_seg_NN[_phase] -> dsp_seg_NN


def load():
    rows, order = {}, []
    for b in ("CPU", "DSP", "HTA"):
        for r in csv.DictReader(open(CSV.format(b))):
            k = (int(r["dispatch_id"]), r["module_name"])
            if k not in rows:
                rows[k] = {}
                order.append(k)
            rows[k][b] = float(r["mean_time"])
    out = []
    for k in sorted(order):
        real = {b: v for b, v in rows[k].items() if v < SENTINEL}
        if not real:
            continue
        b, v = min(real.items(), key=lambda x: x[1])
        out.append((k[0], k[1], b, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../plots/smolvla_vision_hta_gantt.png")
    ap.add_argument("--segments", type=int, default=4,
                    help="how many dsp_seg groups to draw before truncating")
    a = ap.parse_args()
    disp = load()

    # keep whole segment groups so a bundle is never cut in half
    keep, seen = [], 0
    for d in disp:
        if d[1].startswith("dsp_seg") and "_" not in d[1][len("dsp_seg_00"):]:
            seen += 1
        elif d[1].startswith("dsp_seg") and d[1].endswith("tramp_p0"):
            seen += 1
        if seen > a.segments:
            break
        keep.append(d)

    fig, ax = plt.subplots(figsize=(15, 4.2))
    t_mixed = 0.0
    for _, name, be, us in keep:
        ax.broken_barh([(t_mixed / 1000, us / 1000)], (LANE[be], 0.62),
                       facecolors=COL[be], edgecolor="white", linewidth=0.7)
        t_mixed += us
    # the same work if the CPU did all of it, one bar per v3 segment
    cpu_only = cpu_only_by_segment()
    t_cpu = 0.0
    drawn = set()
    for _, name, _, us in keep:
        g = group_of(name)
        if g in drawn:
            continue
        drawn.add(g)
        c = cpu_only.get(g)
        if c is None:
            continue
        ax.broken_barh([(t_cpu / 1000, c / 1000)], (2.9, 0.62),
                       facecolors=COL["CPU"], edgecolor="white", linewidth=0.7)
        t_cpu += c

    for t, col, lab in ((t_cpu, "#555555", "CPU-only"), (t_mixed, "#9C2C2C", "mixed")):
        ax.axvline(t / 1000, color=col, lw=1.0, ls=":", alpha=0.85)
    ax.annotate("", xy=(t_mixed / 1000, 3.72), xytext=(t_cpu / 1000, 3.72),
                arrowprops=dict(arrowstyle="<->", color="#9C2C2C", lw=1.2))
    ax.text((t_cpu + t_mixed) / 2000, 3.76,
            f"−{(t_cpu-t_mixed)/1000:.0f} ms over {a.segments} segments "
            f"({t_cpu/max(t_mixed,1e-9):.2f}×)",
            ha="center", va="bottom", fontsize=9, color="#9C2C2C")

    ax.set_yticks([LANE["HTA"] + .31, LANE["DSP"] + .31, LANE["CPU"] + .31, 3.21])
    ax.set_yticklabels(["mixed\nHTA", "mixed\nDSP", "mixed\nCPU", "CPU-only\nCPU"], fontsize=9)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, max(t_cpu, t_mixed) / 1000 * 1.01)
    ax.set_ylim(0.1, 4.05)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    n_hta = sum(1 for d in disp if d[2] == "HTA")
    tot_mixed = sum(d[3] for d in disp) / 1000
    tot_cpu = sum(cpu_only.values()) / 1000
    ax.set_title(
        f"SmolVLA vision encoder — first {a.segments} of 25 segments, real per-dispatch placement\n"
        f"HTA runs only the extracted Conv1x1 kernels ({n_hta} of {len(disp)} dispatches); "
        f"trampolines stay on DSP/CPU.   full encoder: {tot_cpu:.0f} → {tot_mixed:.0f} ms "
        f"({tot_cpu/tot_mixed:.2f}×)", fontsize=10.5, loc="left")
    ax.legend(handles=[Patch(facecolor=COL["HTA"], label="HTA — extracted Conv1x1 kernels"),
                       Patch(facecolor=COL["DSP"], label="DSP — trampoline phases"),
                       Patch(facecolor=COL["CPU"], label="CPU — mono segments and attention tails")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print(f"  {n_hta} of {len(disp)} dispatches on HTA")
    print(f"  full encoder  CPU-only {tot_cpu:8.1f} ms -> mixed {tot_mixed:8.1f} ms  ({tot_cpu/tot_mixed:.2f}x)")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
