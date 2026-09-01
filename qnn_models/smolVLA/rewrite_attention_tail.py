"""Replace the head-merge Transpose in the vision attention trampolines.

The 12 even-index v3 vision trampolines (`vision_slices_v3/cpu_seg_{00,02,...,22}.onnx`)
each hold the tail of one SigLIP attention block:

    Softmax(scores)[1,12,1024,1024] @ V[1,12,1024,64] -> [1,12,1024,64]
      -> Transpose(perm=[0,2,1,3])                    -> [1,1024,12,64]
      -> Reshape([1,1024,768])                        -> [1,1024,768]

The `Transpose` is a genuine stride permute of a 3 MB tensor and QNN's CPU
backend is slow at it: it and the Reshape account for 6.5 ms of the block's
36.2 ms (measured, see ATTENTION_MAPPING.md).

Merging heads does not actually need a permute. `[1,1024,12,64]` flattened over
its last two axes is head-major along the feature axis, so the same result is
obtained by splitting the attention output per head and concatenating the 12
`[1024,64]` slices along the feature axis:

      MatMul -> Split(axis=1, 12x1) -> 12x Reshape([1024,64])
             -> Concat(axis=1) -> [1024,768] -> Reshape([1,1024,768])

That is bit-exact (verified against the original in onnxruntime on real
activations, and against the shipped DLC's own output on the board), and it
runs in 28.4 ms instead of 36.2 ms -- 7.8 ms per block, 93 ms over the 12.

This does NOT make the blocks accelerator-eligible. HTA has no MatMul kernel at
any rank, so they stay on the CPU either way; see ATTENTION_MAPPING.md.

Usage:
    python rewrite_attention_tail.py --in-dir  vision_slices_v3 \
                                     --out-dir vision_slices_v3/attn_tail
    python rewrite_attention_tail.py --in-dir vision_slices_v3 --out-dir X --check
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _shape(graph, name):
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None


def match_attention_tail(model: onnx.ModelProto):
    """Return (softmax, matmul, transpose, reshape) if the graph is the
    attention-tail pattern with a head-merge permute, else None."""
    g = model.graph
    if [n.op_type for n in g.node] != ["Softmax", "MatMul", "Transpose", "Reshape"]:
        return None
    sm, mm, tr, rs = g.node
    perm = None
    for a in tr.attribute:
        if a.name == "perm":
            perm = list(helper.get_attribute_value(a))
    if perm != [0, 2, 1, 3]:
        return None
    a_shape = _shape(g, mm.output[0])          # [1, H, M, D]
    if a_shape is None or len(a_shape) != 4 or a_shape[0] != 1:
        return None
    return sm, mm, tr, rs


def rewrite(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    m = match_attention_tail(model)
    if m is None:
        return model, False
    sm, mm, tr, rs = m
    g = model.graph
    _, H, M, D = _shape(g, mm.output[0])
    out_name = rs.output[0]
    out_shape = _shape(g, out_name)

    keep = [n for n in g.node if n not in (tr, rs)]
    a = mm.output[0]

    inits = [
        numpy_helper.from_array(np.full(H, 1, np.int64), "attn_tail_split"),
        numpy_helper.from_array(np.array([M, D], np.int64), "attn_tail_head_shape"),
        numpy_helper.from_array(np.array(out_shape, np.int64), "attn_tail_out_shape"),
    ]
    heads = [f"attn_tail_h{h}" for h in range(H)]
    flat = [f"attn_tail_f{h}" for h in range(H)]
    new = [helper.make_node("Split", [a, "attn_tail_split"], heads, axis=1)]
    for h in range(H):
        new.append(helper.make_node("Reshape", [heads[h], "attn_tail_head_shape"], [flat[h]]))
    new.append(helper.make_node("Concat", flat, ["attn_tail_cat"], axis=1))
    new.append(helper.make_node("Reshape", ["attn_tail_cat", "attn_tail_out_shape"], [out_name]))

    vi = [v for v in g.value_info if v.name not in (tr.output[0],)]
    for h in range(H):
        vi.append(helper.make_tensor_value_info(heads[h], TensorProto.FLOAT, [1, 1, M, D]))
        vi.append(helper.make_tensor_value_info(flat[h], TensorProto.FLOAT, [M, D]))
    vi.append(helper.make_tensor_value_info("attn_tail_cat", TensorProto.FLOAT, [M, H * D]))

    ng = helper.make_graph(keep + new, g.name, list(g.input), list(g.output),
                           list(g.initializer) + inits, value_info=vi)
    nm = helper.make_model(ng, opset_imports=list(model.opset_import))
    nm.ir_version = model.ir_version
    onnx.checker.check_model(nm)
    return nm, True


def check(src_path: str, dst_path: str, seed: int = 0) -> float:
    """Max |difference| between original and rewritten graph on random inputs."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    rng = np.random.default_rng(seed)
    src = onnx.load(src_path)
    feeds = {}
    for inp in src.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feeds[inp.name] = rng.standard_normal(dims, dtype=np.float32)
    a = ort.InferenceSession(src_path, so, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    b = ort.InferenceSession(dst_path, so, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    return float(np.abs(a - b).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="directory of cpu_seg_*.onnx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--glob", default="cpu_seg_*.onnx")
    ap.add_argument("--check", action="store_true",
                    help="also run both graphs in onnxruntime and report max abs diff")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_hit = 0
    for src in sorted(glob.glob(os.path.join(args.in_dir, args.glob))):
        model = onnx.load(src)
        new, hit = rewrite(model)
        name = os.path.basename(src)
        if not hit:
            print(f"  skip {name} (not an attention tail)")
            continue
        dst = os.path.join(args.out_dir, name)
        onnx.save(new, dst)
        n_hit += 1
        msg = f"  rewrote {name}"
        if args.check:
            msg += f"   max|diff| = {check(src, dst):.3e}"
        print(msg)
    print(f"rewrote {n_hit} attention tails -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
