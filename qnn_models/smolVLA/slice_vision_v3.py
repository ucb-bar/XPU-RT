"""Slice smolvlm_vision.onnx v3: cut at Tanh AND (Softmax+V_MatMul) boundaries.

The v2 approach of cutting exactly at single Softmax ops causes a 307ms layout
conversion penalty on DSP because the 4D attention tensor [1,12,1024,1024] needs
expensive requantize+transpose when entering the DSP graph.

v3 fix: include {Softmax → V_MatMul → Transpose → Reshape} as the CPU trampoline
(4 ops instead of 1). The DSP segment then starts at a 3D tensor (view_7 [1,1024,768])
which is cheap to quantize.

Structure per ViT layer:
  DSP_A: [LayerNorm → QKV split → reshape heads → Q×K^T + scale]  (pre-Softmax)
  CPU:   [Softmax → V matmul → Transpose → Reshape]               (attention core)
  DSP_B: [output proj → Add residual → LayerNorm → fc1 → GELU prep]  (post-attention)
  CPU:   [Tanh]                                                    (GELU activation)

Output: vision_slices_v3/
  dsp_seg_XX.onnx  (25 DSP segments)
  cpu_seg_XX.onnx  (24 CPU segments: 12 Softmax-blocks + 12 Tanh)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import onnx

_HERE = Path(__file__).parent
_VISION_ONNX = _HERE / "smolvlm_vision.onnx"
_OUT_DIR = _HERE / "vision_slices_v3"


def find_cpu_ranges(graph) -> list[tuple[int, int, str]]:
    """Find CPU segment ranges: (start_idx, end_idx_exclusive, label).

    For Softmax: the CPU block is [Softmax, V_MatMul, Transpose, Reshape] = 4 ops.
    For Tanh: just the single Tanh op.
    """
    ranges = []
    n_nodes = len(graph.node)

    for i, node in enumerate(graph.node):
        if node.op_type == "Softmax":
            # Softmax → MatMul(V) → Transpose → Reshape
            # Verify the expected pattern
            if i + 3 < n_nodes:
                n1 = graph.node[i + 1]
                n2 = graph.node[i + 2]
                n3 = graph.node[i + 3]
                if (n1.op_type == "MatMul" and
                    n2.op_type == "Transpose" and
                    n3.op_type == "Reshape"):
                    ranges.append((i, i + 4, "softmax_block"))
                else:
                    # Fallback: just the Softmax alone
                    print(f"  WARNING: unexpected pattern after Softmax at {i}: "
                          f"{n1.op_type}, {n2.op_type}, {n3.op_type}")
                    ranges.append((i, i + 1, "softmax"))
            else:
                ranges.append((i, i + 1, "softmax"))
        elif node.op_type == "Tanh":
            ranges.append((i, i + 1, "tanh"))

    return ranges


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
    onnx.checker.check_model(model, full_check=False)
    return model


def main():
    print(f"Loading {_VISION_ONNX} ...")
    model = onnx.load(str(_VISION_ONNX))
    graph = model.graph
    print(f"  {len(graph.node)} nodes")

    init_names = {init.name for init in graph.initializer}
    cpu_ranges = find_cpu_ranges(graph)

    print(f"  Found {len(cpu_ranges)} CPU segments:")
    softmax_blocks = sum(1 for _, _, t in cpu_ranges if t == "softmax_block")
    tanhs = sum(1 for _, _, t in cpu_ranges if t == "tanh")
    print(f"    Softmax blocks (4 ops each): {softmax_blocks}")
    print(f"    Tanh (1 op each): {tanhs}")

    if _OUT_DIR.exists():
        shutil.rmtree(_OUT_DIR)
    _OUT_DIR.mkdir(parents=True)

    # Build DSP segment boundaries from CPU range gaps
    # DSP segments fill the space between CPU ranges
    dsp_ranges = []
    prev_end = 0
    for cpu_start, cpu_end, _ in cpu_ranges:
        if cpu_start > prev_end:
            dsp_ranges.append((prev_end, cpu_start))
        prev_end = cpu_end
    if prev_end < len(graph.node):
        dsp_ranges.append((prev_end, len(graph.node)))

    total_dsp_nodes = sum(end - start for start, end in dsp_ranges)
    total_cpu_nodes = sum(end - start for start, end, _ in cpu_ranges)

    print(f"\n=== Extracting {len(dsp_ranges)} DSP segments ===")
    for seg_idx, (start, end) in enumerate(dsp_ranges):
        n_nodes = end - start
        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"dsp_seg_{seg_idx:02d}.onnx"

        print(f"  dsp_seg_{seg_idx:02d}: nodes [{start}, {end}) = {n_nodes} nodes, "
              f"{len(inputs)} in, {len(outputs)} out")
        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print(f"\n=== Extracting {len(cpu_ranges)} CPU segments ===")
    for cpu_idx, (start, end, label) in enumerate(cpu_ranges):
        n_nodes = end - start
        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"cpu_seg_{cpu_idx:02d}.onnx"

        print(f"  cpu_seg_{cpu_idx:02d}: {label} @ [{start},{end}) = {n_nodes} nodes, "
              f"{len(inputs)} in, {len(outputs)} out")
        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print(f"\n=== Summary ===")
    print(f"  DSP nodes: {total_dsp_nodes}")
    print(f"  CPU nodes: {total_cpu_nodes}")
    print(f"  Total:     {total_dsp_nodes + total_cpu_nodes} (original: {len(graph.node)})")
    assert total_dsp_nodes + total_cpu_nodes == len(graph.node)
    print(f"  {len(dsp_ranges)} DSP + {len(cpu_ranges)} CPU = {len(dsp_ranges) + len(cpu_ranges)} segments")
    print(f"  Output: {_OUT_DIR}")


if __name__ == "__main__":
    main()
