"""Slice smolvlm_expert_decode_patched3.onnx into transformer-layer segments.

Decode is a 16-layer transformer expert. Per-layer op pattern:

  S1 (DSP target):  LayerNorm-1 + Q,K,V projections + Q reshapes
                    (~13 ops; 3 heavy MatMul = Conv1x1-rewritable Linear)
  T1 (CPU only):    Rotary frequency setup (Sin/Cos in layers 0,1 only),
                    apply rotation (Split/Mul/Mul/Sub/Add), ScatterND for
                    KV cache writes (4 × per layer), Concat/Expand reshapes
                    (~30 ops; ScatterND is the dealbreaker on DSP/HTA)
  S2 (DSP target):  Attention prep + Q×K^T MatMul + scale
                    (~9 ops; the matmul has activation-on-activation so
                     can stay as raw MatMul on DSP, not Conv-rewritable)
  T2 (CPU only):    Mul/Unsqueeze/Cast/Where/Softmax (attention mask + softmax)
                    (~5 ops; Where + Softmax not supported on HTA)
  S3 (DSP target):  attn×V + output_proj + residual + LayerNorm-2 + SwiGLU FFN
                    (~20 ops; output_proj, fc_gate, fc_up, fc_down all
                     Conv1x1-rewritable; attn×V is activation×activation)

We cut at:
   - Sin / Cos prologue (only present in layers 0,1)
   - The first Cast → ScatterND cluster (start of T1)
   - The last ScatterND output (end of T1)
   - The Where → Softmax block (T2)

Output: vision_slices_decode_v1/
   dsp_seg_XX.onnx  (~3 per layer × 16 = ~48 DSP segments)
   cpu_seg_XX.onnx  (~2 per layer × 16 = ~32 CPU segments)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import onnx

_HERE = Path(__file__).parent
_DECODE_ONNX = _HERE / "smolvlm_expert_decode_patched3.onnx"
_OUT_DIR = _HERE / "decode_slices_v1"


def find_cpu_ranges(graph) -> list[tuple[int, int, str]]:
    """Find CPU segment ranges (start_idx, end_idx_exclusive, label).

    Two patterns we cut at:
      (1) ROTARY+SCATTERND block per layer:
          starts at the first op after the V_proj Reshape (which precedes
          either Unsqueeze→Cast→Div→Sin→Cos rotary-freq-setup OR directly
          Split/Mul/Sub rotary-apply), ends at the last ScatterND in the
          block (4 ScatterND per layer for K/V cache writes; or sometimes
          a Concat right after that consolidates cache+new entries).
      (2) MASK+SOFTMAX block:
          Mul → Unsqueeze → Cast → Where → Softmax (5 ops).

    For (1) we identify by scanning forward for a maximal run that contains
    only "CPU-friendly" small-op types {Cast, Div, Sin, Cos, Split, Mul,
    Sub, Add, Transpose, Reshape, Unsqueeze, ScatterND, Concat, Expand,
    ReduceMin, Slice}. The run starts on a Cast/Unsqueeze/Split AFTER the
    final V-projection Reshape, ends on the last Concat or ScatterND
    before the next "heavy" op (MatMul).
    """
    nodes = graph.node
    n = len(nodes)
    ranges: list[tuple[int, int, str]] = []

    # First pass: find each Softmax → cut a 5-op block ending at it.
    softmax_blocks = []
    for i in range(n):
        if nodes[i].op_type == "Softmax":
            # Walk back to find the start: Mul or Where typically precedes.
            start = i
            for j in range(i - 1, max(-1, i - 6), -1):
                ot = nodes[j].op_type
                if ot in ("Where", "Cast", "Unsqueeze", "Mul"):
                    start = j
                else:
                    break
            softmax_blocks.append((start, i + 1, "softmax_block"))

    # Second pass: find each ScatterND cluster.
    # A "rotary+ScatterND CPU block" per layer is bounded:
    #   - on the left by the last MatMul (Q/K/V proj) preceding it,
    #     OR by the start of a Sin/Cos rotary-freq sequence.
    #   - on the right by the last ScatterND in the cluster + any
    #     immediately following Concat/Expand/Reshape/Transpose ops
    #     before the next MatMul (which would be Q×K^T).
    visited = set()
    scatter_blocks = []
    for i in range(n):
        if nodes[i].op_type != "ScatterND" or i in visited:
            continue
        # Find left boundary: walk back over "light" ops until we hit a
        # MatMul, or until we cross the last MatMul that produced V_proj.
        LIGHT = {"Cast", "Div", "Sin", "Cos", "Split", "Mul", "Sub", "Add",
                  "Transpose", "Reshape", "Unsqueeze", "ScatterND", "Concat",
                  "Expand", "ReduceMin", "Slice"}
        start = i
        for j in range(i - 1, -1, -1):
            ot = nodes[j].op_type
            if ot in LIGHT:
                start = j
            else:
                break
        # Find right boundary: walk forward until we hit a MatMul.
        end = i + 1
        for j in range(i + 1, n):
            ot = nodes[j].op_type
            if ot in LIGHT:
                end = j + 1
            else:
                break
        # Mark all ScatterND in this range visited
        for k in range(start, end):
            if nodes[k].op_type == "ScatterND":
                visited.add(k)
        scatter_blocks.append((start, end, "rotary_scatter_block"))

    ranges = scatter_blocks + softmax_blocks
    # Sort by start; merge overlaps (the softmax block sometimes falls
    # inside the broader scatter_block when both are between adjacent
    # MatMuls — softmax_blocks should take priority since they include
    # the actual Softmax/Where ops).
    ranges.sort()

    merged: list[tuple[int, int, str]] = []
    for r in ranges:
        if not merged:
            merged.append(r); continue
        last = merged[-1]
        if r[0] < last[1]:
            # Overlap: combine, keep wider span; prefer the label that
            # mentions "softmax" if present.
            combined_start = min(last[0], r[0])
            combined_end   = max(last[1], r[1])
            combined_label = "softmax_block" if "softmax" in (last[2] + r[2]) else last[2]
            merged[-1] = (combined_start, combined_end, combined_label)
        else:
            merged.append(r)

    # Split merged "rotary_scatter+softmax_block" into TWO separate ones
    # if a MatMul sits between them — that's the Q×K^T MatMul which we
    # want on DSP. We rely on the input ranges already being separate
    # in that case (left scatter ends before MatMul, softmax starts after).
    return merged


def compute_segment_io(graph, start: int, end: int, init_names: set[str]):
    """Compute external inputs and outputs for nodes [start, end)."""
    seg_produced = set()
    for i in range(start, end):
        for out in graph.node[i].output:
            seg_produced.add(out)

    external_inputs = []
    seen_inputs = set()
    for i in range(start, end):
        for inp in graph.node[i].input:
            if inp in init_names or inp in seg_produced or inp in seen_inputs or inp == "":
                continue
            external_inputs.append(inp)
            seen_inputs.add(inp)

    graph_output_names = {out.name for out in graph.output}
    seg_node_indices = set(range(start, end))

    external_outputs = []
    seen_outputs = set()
    for i in range(start, end):
        for out in graph.node[i].output:
            if out in seen_outputs:
                continue
            is_external = out in graph_output_names
            if not is_external:
                for j, node in enumerate(graph.node):
                    if j in seg_node_indices:
                        continue
                    if out in node.input:
                        is_external = True
                        break
            if is_external:
                external_outputs.append(out)
                seen_outputs.add(out)

    return external_inputs, external_outputs


def slice_model(src_path: str, out_path: str, input_names: list[str],
                output_names: list[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.utils.extract_model(str(src_path), str(out_path),
                              input_names, output_names)
    model = onnx.load(str(out_path))
    try:
        onnx.checker.check_model(model, full_check=False)
    except Exception as e:
        print(f"    warn: checker {e}")
    return model


def main():
    print(f"Loading {_DECODE_ONNX} ...")
    model = onnx.load(str(_DECODE_ONNX))
    graph = model.graph
    print(f"  {len(graph.node)} nodes")

    init_names = {init.name for init in graph.initializer}
    cpu_ranges = find_cpu_ranges(graph)

    sb = sum(1 for _, _, t in cpu_ranges if "softmax" in t)
    rs = sum(1 for _, _, t in cpu_ranges if "rotary" in t)
    print(f"  Found {len(cpu_ranges)} CPU segments (softmax_block × {sb}, "
          f"rotary_scatter × {rs})")

    if _OUT_DIR.exists():
        shutil.rmtree(_OUT_DIR)
    _OUT_DIR.mkdir(parents=True)

    # DSP segment boundaries from CPU range gaps
    dsp_ranges = []
    prev_end = 0
    for cpu_start, cpu_end, _ in cpu_ranges:
        if cpu_start > prev_end:
            dsp_ranges.append((prev_end, cpu_start))
        prev_end = cpu_end
    if prev_end < len(graph.node):
        dsp_ranges.append((prev_end, len(graph.node)))
    print(f"  Implied {len(dsp_ranges)} DSP segments\n")

    # Emit DSP and CPU segments
    n_dsp = 0
    n_cpu = 0
    for i, (s, e) in enumerate(dsp_ranges):
        ins, outs = compute_segment_io(graph, s, e, init_names)
        op_counts = {}
        for j in range(s, e):
            ot = graph.node[j].op_type
            op_counts[ot] = op_counts.get(ot, 0) + 1
        if not ins or not outs:
            print(f"  skip empty/degenerate dsp_seg_{n_dsp:02d}: in={ins} out={outs}")
            continue
        op_summary = ", ".join(f"{k}×{v}" for k,v in sorted(op_counts.items(),
                                                              key=lambda x: -x[1])[:6])
        print(f"  dsp_seg_{n_dsp:02d}  [{s:>4}..{e-1:>4}]  {e-s:>3} ops  "
              f"in={len(ins)} out={len(outs)}  ({op_summary})")
        slice_model(_DECODE_ONNX, _OUT_DIR / f"dsp_seg_{n_dsp:02d}.onnx",
                     ins, outs)
        n_dsp += 1

    for i, (s, e, lbl) in enumerate(cpu_ranges):
        ins, outs = compute_segment_io(graph, s, e, init_names)
        if not ins or not outs:
            print(f"  skip empty/degenerate cpu_seg_{n_cpu:02d}: in={ins} out={outs}")
            continue
        op_counts = {}
        for j in range(s, e):
            ot = graph.node[j].op_type
            op_counts[ot] = op_counts.get(ot, 0) + 1
        op_summary = ", ".join(f"{k}×{v}" for k,v in sorted(op_counts.items(),
                                                              key=lambda x: -x[1])[:5])
        print(f"  cpu_seg_{n_cpu:02d}  [{s:>4}..{e-1:>4}]  {e-s:>3} ops  "
              f"({lbl:<22})  ({op_summary})")
        slice_model(_DECODE_ONNX, _OUT_DIR / f"cpu_seg_{n_cpu:02d}.onnx",
                     ins, outs)
        n_cpu += 1

    print()
    print(f"  Emitted: {n_dsp} DSP + {n_cpu} CPU = {n_dsp + n_cpu} segments")
    print(f"  Output dir: {_OUT_DIR}")


if __name__ == "__main__":
    main()
