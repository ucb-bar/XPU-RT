"""Build a pie chart of per-operator-type runtime contribution for the
unrolled10 SmolVLA workload (1× vision, 1× prefill, 10× decode, plus
projectors and text).

Data sources:
  - qnn_models/boards/qrb5165_v66/profiles/<component>__<backend>.csv
    EXECUTE SUB-EVENT lines = per-op times in µs
  - data/toplevel/networks_smolvla_v3_unrolled10_qrb5165.json metadata
    tells which backend each component was assigned to

For each component, we read the SUB-EVENT block of its best CSV,
classify each op by name into an op family, scale by the component's
instance count (1 or 10), and aggregate.

Output: plots/smolvla_unrolled10_op_breakdown.png
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
PROF_DIR = REPO / "qnn_models/boards/qrb5165_v66/profiles"

# Component → (backend_csv_suffix, instance_count). When a component
# doesn't have per-op SUB-EVENT data on its assigned accelerator (HTA/DSP
# tend to omit it), fall back to CPU_int8 so the *shape* of the op-type
# distribution is preserved.
COMPONENTS = [
    # name, csv stem, instance count
    ("vision (v3 sliced)",      "smolvlm_vision",          1),
    ("text expert",             "smolvlm_text",            1),
    ("state projector",         "state_projector",         1),
    ("prefill expert",          "smolvlm_expert_prefill",  1),
    ("decode expert ×10",       "smolvlm_expert_decode",  10),
    ("action_in projector ×10", "action_in_projector",    10),
    ("action_out projector ×10","action_out_projector",   10),
    ("time_in projector ×10",   "time_in_projector",      10),
    ("time_out projector ×10",  "time_out_projector",     10),
]

# Op-type families. Order matters: first matching pattern wins. Designed
# to bucket QNN op-names emitted by snpe-onnx-to-dlc.
OP_FAMILIES = [
    # name, regex (case-insensitive)
    ("Linear / Conv1x1 (MatMul-rewritten)", r"conv1x1|node_MatMul_\d+$|sng_MatMul"),
    ("Conv (patch / head)",        r"^_?(node_)?Conv_?\d*$|conv_modules|/Conv"),
    ("Attention MatMul (batched)", r"MatMul"),                # remaining MatMul ops
    ("LayerNorm / RMSNorm decomp", r"LayerNorm|ReduceMean|Pow|Sqrt|Reciprocal|rms_norm"),
    ("Softmax",                    r"Softmax"),
    # QNN often emits GELU/Tanh/SiLU under a generic "elementwise_neuron" op
    ("GELU / Tanh / SiLU",         r"Tanh|Gelu|Silu|Sigmoid|elementwise[_]?neuron"),
    ("Rotary (Sin/Cos)",           r"(?:^|_)Sin(?:_|$)|(?:^|_)Cos(?:_|$)"),
    # _node_Add_NNN / _node_Sub_NNN — underscores are word chars so use explicit boundaries
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


def extract_op_times(csv_path: Path) -> dict[str, float]:
    """Return {op_family: total_us} from a QNN profile CSV's EXECUTE
    SUB-EVENT block. Sums all per-op SUB-EVENT durations classified by
    family."""
    fam_us: dict[str, float] = defaultdict(float)
    with open(csv_path) as f:
        for row in csv.reader(f):
            # Header rows are short
            if len(row) < 7:
                continue
            _, msg, dur_us, _unit, src, level, ident = row[:7]
            if msg != "EXECUTE": continue
            if src  != "BACKEND": continue
            if level != "SUB-EVENT": continue
            try:
                d = float(dur_us)
            except ValueError:
                continue
            fam_us[classify(ident)] += d
    return dict(fam_us)


def best_csv(stem: str) -> Path | None:
    """Pick the CSV with the most SUB-EVENT rows (proxy for richest
    per-op data). Prefer CPU_int8 → CPU → DSP → GPU_fp16 → others."""
    candidates = sorted(PROF_DIR.glob(f"{stem}__*.csv"))
    if not candidates:
        return None
    pref = {"CPU_int8": 0, "CPU": 1, "DSP": 2, "HTA": 3, "GPU_fp16": 4, "GPU": 5}
    def score(p):
        be = p.stem.split("__", 1)[1]
        return pref.get(be, 99)
    candidates.sort(key=score)
    # First with at least one SUB-EVENT.
    for p in candidates:
        with open(p) as f:
            for row in csv.reader(f):
                if len(row) >= 6 and row[1] == "EXECUTE" and row[4] == "BACKEND" and row[5] == "SUB-EVENT":
                    return p
    return candidates[0]


def main():
    totals: dict[str, float] = defaultdict(float)
    per_component: dict[str, dict[str, float]] = {}

    for label, stem, mult in COMPONENTS:
        csv_path = best_csv(stem)
        if csv_path is None:
            print(f"  {label}: no CSV for {stem}, skip")
            continue
        fam_us = extract_op_times(csv_path)
        scaled = {k: v * mult for k, v in fam_us.items()}
        per_component[label] = scaled
        for k, v in scaled.items():
            totals[k] += v
        comp_total_ms = sum(scaled.values()) / 1000
        print(f"  {label:<32s} {csv_path.name:<40s} → {comp_total_ms:7.1f} ms")

    grand_ms = sum(totals.values()) / 1000
    print(f"\n  grand total (op-time aggregate): {grand_ms:.1f} ms")
    print(f"\n  per-family contribution:")
    rows = sorted(totals.items(), key=lambda x: -x[1])
    for fam, us in rows:
        pct = us / sum(totals.values()) * 100
        print(f"    {fam:<48s} {us/1000:8.1f} ms  ({pct:5.1f}%)")

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [fam for fam, _ in rows]
    values = [us for _, us in rows]
    pcts = [v / sum(values) * 100 for v in values]

    # Color-coordinated palette: warm colors for compute-heavy (MatMul/Conv/Attention),
    # cool for layout/elementwise.
    family_color = {
        "Linear / Conv1x1 (MatMul-rewritten)": "#d62728",
        "Conv (patch / head)":                  "#9467bd",
        "Attention MatMul (batched)":           "#ff7f0e",
        "LayerNorm / RMSNorm decomp":           "#bcbd22",
        "Softmax":                              "#e377c2",
        "GELU / Tanh / SiLU":                   "#8c564b",
        "Rotary (Sin/Cos)":                     "#17becf",
        "Residual Add":                         "#7f7f7f",
        "Elementwise Mul / Div":                "#aec7e8",
        "KV cache write (ScatterND)":           "#ff9896",
        "Attention mask (Where)":               "#c5b0d5",
        "Layout (Reshape/Transp/Cast)":         "#1f77b4",
        "Tensor (Concat/Split/Gather/Expand/Slice)": "#2ca02c",
        "Other":                                "#bbbbbb",
    }
    colors = [family_color.get(f, "#bbbbbb") for f in families]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9),
                                    gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Left: pie. Only label slices ≥ 4% directly on the wedge; smaller
    # slices live only in the legend so labels don't overlap.
    show_labels = [f"{p:.1f}%" if p >= 4 else "" for p in pcts]
    wedges, _ = ax1.pie(values, labels=show_labels, colors=colors,
                         startangle=90, counterclock=False,
                         wedgeprops={"edgecolor": "white", "linewidth": 1.2},
                         labeldistance=0.72, textprops={"fontsize": 11,
                                                         "fontweight": "bold",
                                                         "color": "white"})
    ax1.set_title(f"SmolVLA op-time breakdown (unrolled10)\n"
                   f"total = {grand_ms:.0f} ms (sum of per-op SUB-EVENT µs)",
                   loc="center", fontsize=12, pad=14)

    # Right: legend with absolute numbers + notes below.
    legend_labels = [f"{f}  —  {v/1000:7.1f} ms  ({p:5.1f}%)"
                     for f, v, p in zip(families, values, pcts)]
    ax2.axis("off")
    leg = ax2.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
                     legend_labels, loc="upper left", fontsize=11,
                     frameon=False, handlelength=1.6, labelspacing=0.6,
                     bbox_to_anchor=(0.0, 0.95))
    notes = (
        "Notes\n"
        "  • Workload: 1× vision + 1× prefill + 1× text + 1× state proj\n"
        "             + 10× decode + 10× action_in + 10× action_out\n"
        "             + 10× time_in + 10× time_out  (50 invocations)\n"
        "  • Per-op times from CPU_int8 SUB-EVENT data — richest per-op\n"
        "    breakdown across all backends. Accelerator runs would shift\n"
        "    absolute numbers but the family distribution is similar.\n"
        "  • Linear/Conv1x1 includes the MatMul→Conv1x1 rewrite from\n"
        "    rewrite_matmul_to_conv1x1.py applied to ViT and the SmolVLM\n"
        "    decoder/prefill expert.\n"
        "  • LayerNorm/RMSNorm shows their *decomposed* form: ReduceMean +\n"
        "    Sub + Pow + Sqrt + Reciprocal + Mul. The DSP backend rejects\n"
        "    decomposed RMSNorm — this is the smolvlm-decoder blocker."
    )
    ax2.text(0.0, 0.05, notes, transform=ax2.transAxes,
             fontsize=9, va="bottom", ha="left", family="monospace")

    out = REPO / "plots/smolvla_unrolled10_op_breakdown.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
