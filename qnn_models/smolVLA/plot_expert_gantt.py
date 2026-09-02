#!/usr/bin/env python3
"""Gantt of one expert layer, CPU-only against the best mixed tiling.

Run for either expert. They reach opposite conclusions and the charts show why.

  prefill (113 tokens)  the MLP block is worth exporting: HTA does it in
                        2.452 ms where the Kryo needs 4.415. Per layer
                        18.60 -> 16.87 ms, 297.6 -> 269.9 ms over 16.

  decode  (50 tokens)   nothing is worth exporting. The best accelerator on the
                        best block is DSP on the MLP at 1.679 vs 1.701 ms --
                        a 21.5 us edge that the 90 us handoff then more than
                        erases. The mixed bar is LONGER than the CPU bar.

In both cases the expert is strictly sequential -- layer N needs layer N-1, and
inside a layer qkv -> attention -> o_proj -> MLP is a chain -- so an offloaded
block overlaps with nothing. The CPU stops and waits. On prefill that idle band
is worth having because HTA finishes sooner anyway; on decode it is pure loss.

Measured (profile_seg, 50 iters, 3 interleaved repeats, performance governor,
gap-phase median). Two numbers per component are not:

  remainder  DERIVED: whole-layer CPU cost minus the three linear tiles.
             prefill 297.6/16 = 18.60 ms less 6.53 -> 12.07 ms
             decode  111.63/16 = 6.977 ms less 2.675 -> 4.302 ms
  handoff    ESTIMATED: 2 x (8.6 us measured dispatch + the block's activation
             moved). prefill [1,113,960] int8 = 108 KB -> ~230 us round trip;
             decode [1,50,720] int8 = 36 KB -> ~90 us.

    python3 plot_expert_gantt.py --component prefill --out ../../plots/smolvla_expert_gantt.png
    python3 plot_expert_gantt.py --component decode  --out ../../plots/smolvla_decode_gantt.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Tile costs are READ from the committed cost model, not duplicated here, so the
# chart cannot drift from the measurements it claims to plot. Only `remain` and
# `handoff` are literals, because neither is a measured tile -- see the module
# docstring for how each is derived.
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "flow_c", "measurements", "qrb5165_v66_smolvla_expert_nc.json")

_SHAPE = {
    "prefill": dict(title="expert prefill", seq=113, layers=16, steps=1,
                    net="smolvla_expert_linear", pfx="nc",
                    remain=12070.0, handoff=230.0),
    "decode":  dict(title="expert decode", seq=50, layers=16, steps=10,
                    net="smolvla_expert_decode_linear", pfx="ncd",
                    remain=4301.6, handoff=90.0),
}


def load_components(path=CELLS):
    """Build the per-component cost dict from the measured cells on disk."""
    cells = json.load(open(path))["cells"]
    out = {}
    for key, sp in _SHAPE.items():
        get = lambda tile: cells[f"{sp['net']}/{sp['pfx']}_{tile}"]
        mlp = get("mlp")
        # the offload lane is whichever accelerator actually beats the CPU here,
        # decided by the data rather than asserted
        acc = min(("hta", "dsp"), key=lambda b: mlp[b])
        out[key] = dict(title=sp["title"], seq=sp["seq"], layers=sp["layers"],
                        steps=sp["steps"], remain=sp["remain"], handoff=sp["handoff"],
                        qkv=get("qkv")["cpu"], oproj=get("oproj")["cpu"],
                        mlp_cpu=mlp["cpu"], off_lane=acc.upper(), mlp_off=mlp[acc])
    return out

C = {"qkv": "#4C72B0", "rem": "#B0B7C3", "oproj": "#55A868",
     "mlp_c": "#C44E52", "mlp_o": "#DD8452", "ho": "#8172B3"}


def lay_cpu(p, t):
    b = [("qkv", t, p["qkv"], C["qkv"], "cpu")]
    t += p["qkv"]
    b.append(("rem", t, p["remain"], C["rem"], "cpu")); t += p["remain"]
    b.append(("oproj", t, p["oproj"], C["oproj"], "cpu")); t += p["oproj"]
    b.append(("mlp", t, p["mlp_cpu"], C["mlp_c"], "cpu")); t += p["mlp_cpu"]
    return b, t


def lay_mix(p, t):
    b = [("qkv", t, p["qkv"], C["qkv"], "cpu")]
    t += p["qkv"]
    b.append(("rem", t, p["remain"], C["rem"], "cpu")); t += p["remain"]
    b.append(("oproj", t, p["oproj"], C["oproj"], "cpu")); t += p["oproj"]
    b.append(("ho", t, p["handoff"] / 2, C["ho"], "cpu")); t += p["handoff"] / 2
    b.append(("mlp", t, p["mlp_off"], C["mlp_o"], "off")); t += p["mlp_off"]
    b.append(("ho", t, p["handoff"] / 2, C["ho"], "cpu")); t += p["handoff"] / 2
    return b, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=sorted(_SHAPE), default="prefill")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers-shown", type=int, default=3)
    a = ap.parse_args()
    p = load_components()[a.component]
    out = a.out or f"../../plots/smolvla_{a.component}_gantt.png"
    n, L, ST = a.layers_shown, p["layers"], p["steps"]

    cpu_b, t_cpu, mix_b, t_mix = [], 0.0, [], 0.0
    for _ in range(n):
        bl, t_cpu = lay_cpu(p, t_cpu); cpu_b += bl
        bl, t_mix = lay_mix(p, t_mix); mix_b += bl

    per_cpu = p["qkv"] + p["remain"] + p["oproj"] + p["mlp_cpu"]
    per_mix = p["qkv"] + p["remain"] + p["oproj"] + p["mlp_off"] + p["handoff"]
    delta = per_cpu - per_mix
    win = delta > 0

    fig, ax = plt.subplots(figsize=(14, 4.6))
    Y = {"cpu0": 2.6, "cpu1": 1.2, "off": 0.4}
    H = 0.5
    for _, t, d, col, lane in cpu_b:
        ax.broken_barh([(t / 1000, d / 1000)], (Y["cpu0"], H),
                       facecolors=col, edgecolor="white", linewidth=0.8)
    for _, t, d, col, lane in mix_b:
        ax.broken_barh([(t / 1000, d / 1000)], (Y["cpu1"] if lane == "cpu" else Y["off"], H),
                       facecolors=col, edgecolor="white", linewidth=0.8)

    edge = "#9C2C2C" if win else "#8A6D1F"
    for i in range(n):
        off = [x for x in mix_b[i * 6:(i + 1) * 6] if x[4] == "off"][0]
        ax.broken_barh([(off[1] / 1000, off[2] / 1000)], (Y["cpu1"], H),
                       facecolors="none", edgecolor=edge, linewidth=1.3,
                       linestyle=(0, (3, 2)))
        if i == 0:
            ax.annotate(f"CPU idle {p['mlp_off']/1000:.2f} ms/layer"
                        + ("\n(schedulable elsewhere)" if win
                           else "\n— and the layer still ends LATER"),
                        xy=((off[1] + off[2] / 2) / 1000, Y["cpu1"]),
                        xytext=((off[1] + off[2] / 2) / 1000, 0.02),
                        ha="center", fontsize=8.5, color=edge,
                        arrowprops=dict(arrowstyle="->", color=edge, lw=1.1))

    # mark where each schedule finishes the shown layers
    for t, lab, col in ((t_cpu, "CPU-only", "#333333"), (t_mix, "mixed", edge)):
        ax.axvline(t / 1000, color=col, lw=1.0, ls=":", alpha=0.8)
    ax.annotate("", xy=(t_mix / 1000, 3.15), xytext=(t_cpu / 1000, 3.15),
                arrowprops=dict(arrowstyle="<->", color=edge, lw=1.2))
    ax.text((t_cpu + t_mix) / 2000, 3.19,
            f"{'−' if win else '+'}{abs(t_cpu-t_mix)/1000:.2f} ms over {n} layers",
            ha="center", va="bottom", fontsize=8.5, color=edge)

    ax.set_yticks([Y["cpu0"] + H / 2, Y["cpu1"] + H / 2, Y["off"] + H / 2])
    ax.set_yticklabels(["CPU-only\nCPU", "mixed\nCPU", f"mixed\n{p['off_lane']}"], fontsize=9)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, max(t_cpu, t_mix) / 1000 * 1.02)
    ax.set_ylim(-0.15, 3.45)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    verdict = (f"−{delta*L/1000:.1f} ms, {per_cpu/per_mix:.2f}×" if win
               else f"+{-delta*L/1000:.1f} ms — offload is a NET LOSS")
    step_note = f" × {ST} denoising steps: {per_cpu*L*ST/1000:.0f} → {per_mix*L*ST/1000:.0f} ms" if ST > 1 else ""
    ax.set_title(
        f"SmolVLA {p['title']} ({p['seq']} tokens) — {n} of {L} layers, "
        f"CPU-only vs MLP-on-{p['off_lane']}\n"
        f"per layer {per_cpu/1000:.3f} → {per_mix/1000:.3f} ms   "
        f"({L} layers: {per_cpu*L/1000:.1f} → {per_mix*L/1000:.1f} ms, {verdict}){step_note}",
        fontsize=10.5, loc="left")
    ax.legend(handles=[
        Patch(facecolor=C["qkv"], label=f"qkv (CPU {p['qkv']/1000:.2f} ms)"),
        Patch(facecolor=C["rem"], label=f"attention + RoPE + norms (CPU {p['remain']/1000:.2f} ms, untiled)"),
        Patch(facecolor=C["oproj"], label=f"o_proj (CPU {p['oproj']/1000:.2f} ms)"),
        Patch(facecolor=C["mlp_c"], label=f"MLP on CPU ({p['mlp_cpu']/1000:.2f} ms)"),
        Patch(facecolor=C["mlp_o"], label=f"MLP on {p['off_lane']} ({p['mlp_off']/1000:.2f} ms)"),
        Patch(facecolor=C["ho"], label=f"handoff (est. {p['handoff']/1000:.2f} ms round trip)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  {a.component}: per layer {per_cpu/1000:7.3f} -> {per_mix/1000:7.3f} ms"
          f"   ({'saves' if win else 'COSTS'} {abs(delta)/1000:.3f} ms/layer)")
    print(f"    x{L} layers x{ST} steps: {per_cpu*L*ST/1000:8.1f} -> {per_mix*L*ST/1000:8.1f} ms")
    print(f"    -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
