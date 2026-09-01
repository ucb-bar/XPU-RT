#!/usr/bin/env python3
"""Gantt of one expert-prefill layer, CPU-only against the mixed CPU+HTA tiling.

The point of the chart is what the mixed schedule does NOT do. The expert is
strictly sequential -- layer N needs layer N-1, and inside a layer
qkv -> attention -> o_proj -> MLP is a chain -- so putting the MLP on HTA
overlaps with nothing. The CPU simply stops and waits. The gain is only that
HTA finishes the MLP in 2.452 ms where the Kryo needs 4.415.

What the idle band is worth is a scheduling question, not an expert question:
2.45 ms per layer x 16 = 39 ms of CPU that XPU-RT can fill with another
network's work, which is the actual argument for the offload.

Numbers are measured (profile_seg, 50 iters, 3 interleaved repeats, performance
governor, gap-phase median) except where marked:

  nc_qkv    cpu 1193.1   hta 2455.3        measured
  nc_oproj  cpu  925.1   hta 1773.3        measured
  nc_mlp    cpu 4414.9   hta 2452.2        measured
  remainder 12070                          DERIVED, see below
  handoff     230                          ESTIMATED, see below

`remainder` is attention + RoPE + both RMSNorms + layout, which was never cut
into its own tile. It is the whole-layer CPU cost minus the three linear tiles:
297.6 ms / 16 layers = 18.60 ms, less 6.53 ms of linears = 12.07 ms. The
whole-layer figure is corroborated by the L1conv probe at 18.51 ms.

`handoff` is 2 x (8.6 us measured dispatch latency + ~108 KB moved), the MLP
tile's [1,113,960] int8 input and output. Not measured directly -- it is the
one soft number here and it works against the mixed case, so the saving shown
is conservative.

    python3 plot_expert_gantt.py --out ../../plots/smolvla_expert_gantt.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

QKV, OPROJ = 1193.1, 925.1
MLP_CPU, MLP_HTA = 4414.9, 2452.2
REMAIN, HANDOFF = 12070.0, 230.0
LAYERS = 16

C = {"qkv": "#4C72B0", "rem": "#B0B7C3", "oproj": "#55A868",
     "mlp_c": "#C44E52", "mlp_h": "#DD8452", "ho": "#8172B3"}


def layer_cpu(t0):
    """CPU-only: one lane, four blocks back to back."""
    b = []
    b.append(("qkv", t0, QKV, C["qkv"], "cpu"))
    t0 += QKV
    b.append(("attention + RoPE + norms", t0, REMAIN, C["rem"], "cpu"))
    t0 += REMAIN
    b.append(("o_proj", t0, OPROJ, C["oproj"], "cpu"))
    t0 += OPROJ
    b.append(("MLP", t0, MLP_CPU, C["mlp_c"], "cpu"))
    return b, t0 + MLP_CPU


def layer_mixed(t0):
    """Mixed: MLP on HTA. Note the CPU lane has a hole, not overlapped work."""
    b = []
    b.append(("qkv", t0, QKV, C["qkv"], "cpu"))
    t0 += QKV
    b.append(("attention + RoPE + norms", t0, REMAIN, C["rem"], "cpu"))
    t0 += REMAIN
    b.append(("o_proj", t0, OPROJ, C["oproj"], "cpu"))
    t0 += OPROJ
    b.append(("handoff", t0, HANDOFF / 2, C["ho"], "cpu"))
    t0 += HANDOFF / 2
    b.append(("MLP", t0, MLP_HTA, C["mlp_h"], "hta"))
    t0 += MLP_HTA
    b.append(("handoff", t0, HANDOFF / 2, C["ho"], "cpu"))
    return b, t0 + HANDOFF / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../plots/smolvla_expert_gantt.png")
    ap.add_argument("--layers-shown", type=int, default=3)
    a = ap.parse_args()

    n = a.layers_shown
    cpu_b, t_cpu = [], 0.0
    for _ in range(n):
        blocks, t_cpu = layer_cpu(t_cpu)
        cpu_b += blocks
    mix_b, t_mix = [], 0.0
    for _ in range(n):
        blocks, t_mix = layer_mixed(t_mix)
        mix_b += blocks

    fig, ax = plt.subplots(figsize=(14, 4.6))
    lane_y = {("cpu-only", "cpu"): 2.6, ("mixed", "cpu"): 1.2, ("mixed", "hta"): 0.4}
    H = 0.5

    for name, t, d, col, lane in cpu_b:
        ax.broken_barh([(t / 1000, d / 1000)], (lane_y[("cpu-only", "cpu")], H),
                       facecolors=col, edgecolor="white", linewidth=0.8)
    for name, t, d, col, lane in mix_b:
        y = lane_y[("mixed", lane)]
        ax.broken_barh([(t / 1000, d / 1000)], (y, H),
                       facecolors=col, edgecolor="white", linewidth=0.8)

    # the CPU idle band in the mixed schedule -- the thing worth seeing
    for i in range(n):
        blocks = mix_b[i * 6:(i + 1) * 6]
        mlp = [b for b in blocks if b[4] == "hta"][0]
        ax.broken_barh([(mlp[1] / 1000, mlp[2] / 1000)],
                       (lane_y[("mixed", "cpu")], H),
                       facecolors="none", edgecolor="#9C2C2C",
                       linewidth=1.3, linestyle=(0, (3, 2)))
        if i == 0:
            ax.annotate(f"CPU idle — schedulable\n"
                        f"{MLP_HTA/1000:.2f} ms/layer, {MLP_HTA*LAYERS/1000:.0f} ms over {LAYERS}",
                        xy=((mlp[1] + mlp[2] / 2) / 1000, lane_y[("mixed", "cpu")]),
                        xytext=((mlp[1] + mlp[2] / 2) / 1000, 0.02),
                        ha="center", fontsize=8.5, color="#9C2C2C",
                        arrowprops=dict(arrowstyle="->", color="#9C2C2C", lw=1.1))

    ax.set_yticks([lane_y[("cpu-only", "cpu")] + H / 2,
                   lane_y[("mixed", "cpu")] + H / 2,
                   lane_y[("mixed", "hta")] + H / 2])
    ax.set_yticklabels(["CPU-only\nCPU", "mixed\nCPU", "mixed\nHTA"], fontsize=9)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, max(t_cpu, t_mix) / 1000 * 1.005)
    ax.set_ylim(-0.15, 3.35)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    per_cpu = QKV + REMAIN + OPROJ + MLP_CPU
    per_mix = QKV + REMAIN + OPROJ + MLP_HTA + HANDOFF
    ax.set_title(
        f"SmolVLA expert prefill — {n} of {LAYERS} layers, CPU-only vs MLP-on-HTA\n"
        f"per layer {per_cpu/1000:.2f} → {per_mix/1000:.2f} ms   "
        f"({LAYERS} layers: {per_cpu*LAYERS/1000:.1f} → {per_mix*LAYERS/1000:.1f} ms, "
        f"−{(per_cpu-per_mix)*LAYERS/1000:.1f} ms, {per_cpu/per_mix:.2f}×)",
        fontsize=10.5, loc="left")

    ax.legend(handles=[Patch(facecolor=C["qkv"], label="qkv (CPU 1.19 ms)"),
                       Patch(facecolor=C["rem"], label="attention + RoPE + norms (CPU 12.07 ms, untiled)"),
                       Patch(facecolor=C["oproj"], label="o_proj (CPU 0.93 ms)"),
                       Patch(facecolor=C["mlp_c"], label="MLP on CPU (4.41 ms)"),
                       Patch(facecolor=C["mlp_h"], label="MLP on HTA (2.45 ms)"),
                       Patch(facecolor=C["ho"], label="handoff (est. 0.23 ms round trip)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8.5,
              frameon=False)

    fig.tight_layout()
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print(f"  per layer   cpu-only {per_cpu/1000:7.3f} ms   mixed {per_mix/1000:7.3f} ms")
    print(f"  x{LAYERS} layers   {per_cpu*LAYERS/1000:7.1f} ms   ->    {per_mix*LAYERS/1000:7.1f} ms"
          f"   saves {(per_cpu-per_mix)*LAYERS/1000:.1f} ms ({per_cpu/per_mix:.3f}x)")
    print(f"  CPU idle created  {MLP_HTA/1000:.3f} ms/layer, {MLP_HTA*LAYERS/1000:.1f} ms over {LAYERS}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
