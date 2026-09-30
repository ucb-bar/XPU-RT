"""Pie chart of SmolVLA op-time, with the Linear family split into
GEMM (vision + prefill, batched) and GEMV (decode + projectors, M=1).

Same data sources as pie_smolvla_op_breakdown.py — diff is that linear
ops are attributed to GEMM-vs-GEMV based on the *component* they came
from (the M-dim of the MatMul is set by the component's input shape).
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
PROF_DIR = REPO / "qnn_models/boards/qrb5165_v66/profiles"

# (label, csv stem, instance count, "GEMM" or "GEMV") — the latter only
# applies when classifying linear ops from that component.
COMPONENTS = [
    ("vision (v3 sliced)",      "smolvlm_vision",          1, "GEMM"),
    ("text expert",             "smolvlm_text",            1, "GEMV"),
    ("state projector",         "state_projector",         1, "GEMV"),
    ("prefill expert",          "smolvlm_expert_prefill",  1, "GEMM"),
    ("decode expert ×10",       "smolvlm_expert_decode",  10, "GEMV"),
    ("action_in projector ×10", "action_in_projector",    10, "GEMV"),
    ("action_out projector ×10","action_out_projector",   10, "GEMV"),
    ("time_in projector ×10",   "time_in_projector",      10, "GEMV"),
    ("time_out projector ×10",  "time_out_projector",     10, "GEMV"),
]

# Per-op-name → family. The Linear category gets an extra "mode" suffix
# (GEMM vs GEMV) added at aggregation time based on the source component.
OP_FAMILIES = [
    ("Linear", r"conv1x1|node_MatMul_\d+$|sng_MatMul"),
    ("Conv (patch / head)",        r"^_?(node_)?Conv_?\d*$|conv_modules|/Conv"),
    ("Attention MatMul (batched)", r"MatMul"),
    ("LayerNorm / RMSNorm decomp", r"LayerNorm|ReduceMean|Pow|Sqrt|Reciprocal|rms_norm"),
    ("Softmax",                    r"Softmax"),
    ("GELU / Tanh / SiLU",         r"Tanh|Gelu|Silu|Sigmoid|elementwise[_]?neuron"),
    ("Rotary (Sin/Cos)",           r"(?:^|_)Sin(?:_|$)|(?:^|_)Cos(?:_|$)"),
    ("Residual Add",               r"(?:^|_)Add(?:_|$)|(?:^|_)Sub(?:_|$)"),
    ("Elementwise Mul / Div",      r"(?:^|_)Mul(?:_|$)|(?:^|_)Div(?:_|$)"),
    ("KV cache write (ScatterND)", r"ScatterND"),
    ("Attention mask (Where)",     r"Where"),
    ("Layout (Reshape/Transp/Cast)", r"Reshape|Transpose|reshape|nchw|nhwc|Cast|Squeeze|Unsqueeze|_quant"),
    ("Tensor (Concat/Split/Gather/Expand/Slice)",
                                   r"Concat|Split|Gather|Expand|Slice|ReduceMin|ReduceMax"),
    ("Other",                      r".*"),
]


def classify(op_name: str) -> str:
    for fam, pat in OP_FAMILIES:
        if re.search(pat, op_name, re.IGNORECASE):
            return fam
    return "Other"


def extract_op_times(csv_path: Path, component_mode: str) -> dict[str, float]:
    fam_us: dict[str, float] = defaultdict(float)
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            _, msg, dur_us, _unit, src, level, ident = row[:7]
            if msg != "EXECUTE" or src != "BACKEND" or level != "SUB-EVENT":
                continue
            try:
                d = float(dur_us)
            except ValueError:
                continue
            fam = classify(ident)
            # Split Linear by GEMM vs GEMV based on the component's mode
            if fam == "Linear":
                fam = f"Linear — {component_mode}"
            fam_us[fam] += d
    return dict(fam_us)


def best_csv(stem: str) -> Path | None:
    candidates = sorted(PROF_DIR.glob(f"{stem}__*.csv"))
    if not candidates:
        return None
    pref = {"CPU_int8": 0, "CPU": 1, "DSP": 2, "HTA": 3, "GPU_fp16": 4, "GPU": 5}
    candidates.sort(key=lambda p: pref.get(p.stem.split("__", 1)[1], 99))
    for p in candidates:
        with open(p) as f:
            for row in csv.reader(f):
                if len(row) >= 6 and row[1] == "EXECUTE" and row[4] == "BACKEND" and row[5] == "SUB-EVENT":
                    return p
    return candidates[0]


def main():
    totals: dict[str, float] = defaultdict(float)
    for label, stem, mult, mode in COMPONENTS:
        csv_path = best_csv(stem)
        if csv_path is None:
            continue
        fam_us = extract_op_times(csv_path, mode)
        for k, v in fam_us.items():
            totals[k] += v * mult
        print(f"  {label:<32s} {csv_path.name:<40s} → {sum(fam_us.values())*mult/1000:7.1f} ms")

    grand_ms = sum(totals.values()) / 1000
    print(f"\n  grand total: {grand_ms:.1f} ms")
    print(f"\n  per-family contribution:")
    rows = sorted(totals.items(), key=lambda x: -x[1])
    for fam, us in rows:
        pct = us / sum(totals.values()) * 100
        print(f"    {fam:<48s} {us/1000:8.1f} ms  ({pct:5.1f}%)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [fam for fam, _ in rows]
    values = [us for _, us in rows]
    pcts = [v / sum(values) * 100 for v in values]

    family_color = {
        "Linear — GEMM":                        "#d62728",
        "Linear — GEMV":                        "#ff9896",
        "Conv (patch / head)":                  "#9467bd",
        "Attention MatMul (batched)":           "#ff7f0e",
        "LayerNorm / RMSNorm decomp":           "#bcbd22",
        "Softmax":                              "#e377c2",
        "GELU / Tanh / SiLU":                   "#8c564b",
        "Rotary (Sin/Cos)":                     "#17becf",
        "Residual Add":                         "#7f7f7f",
        "Elementwise Mul / Div":                "#aec7e8",
        "KV cache write (ScatterND)":           "#ff7f7f",
        "Attention mask (Where)":               "#c5b0d5",
        "Layout (Reshape/Transp/Cast)":         "#1f77b4",
        "Tensor (Concat/Split/Gather/Expand/Slice)": "#2ca02c",
        "Other":                                "#bbbbbb",
    }
    colors = [family_color.get(f, "#bbbbbb") for f in families]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9),
                                    gridspec_kw={"width_ratios": [1.0, 1.2]})

    show_labels = [f"{p:.1f}%" if p >= 4 else "" for p in pcts]
    ax1.pie(values, labels=show_labels, colors=colors,
             startangle=90, counterclock=False,
             wedgeprops={"edgecolor": "white", "linewidth": 1.2},
             labeldistance=0.72,
             textprops={"fontsize": 11, "fontweight": "bold", "color": "white"})
    ax1.set_title(f"SmolVLA op-time breakdown (unrolled10) — Linear split by GEMM vs GEMV\n"
                   f"total = {grand_ms:.0f} ms (sum of per-op SUB-EVENT µs)",
                   loc="center", fontsize=12, pad=14)

    legend_labels = [f"{f}  —  {v/1000:7.1f} ms  ({p:5.1f}%)"
                     for f, v, p in zip(families, values, pcts)]
    ax2.axis("off")
    ax2.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
                legend_labels, loc="upper left", fontsize=11,
                frameon=False, handlelength=1.6, labelspacing=0.6,
                bbox_to_anchor=(0.0, 0.95))
    notes = (
        "Notes\n"
        "  • Workload: 1× vision + 1× prefill + 1× text + 1× state proj\n"
        "             + 10× decode + 10× of each of 4 projectors  (50 calls)\n"
        "  • Per-op times from CPU_int8 SUB-EVENT data.\n"
        "  • GEMM = MatMul with sequence dim M > 1 (vision ViT runs M=1024,\n"
        "    prefill runs M=prompt_len). GEMV = MatMul with M=1 (decode is\n"
        "    one new token per step; projectors operate on a single state).\n"
        "  • Vision GEMM dominates (~9.1 s of the Linear bucket); decode\n"
        "    GEMVs add ~3.4 s across 10 iterations.\n"
        "  • Attention MatMul (batched) is separate — both operands are\n"
        "    activations, not Conv1x1-rewritable."
    )
    ax2.text(0.0, 0.05, notes, transform=ax2.transAxes,
              fontsize=9, va="bottom", ha="left", family="monospace")

    out = REPO / "plots/smolvla_unrolled10_op_breakdown_gemm_gemv.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
