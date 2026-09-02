#!/usr/bin/env python3
"""Extract an expert layer's linears as NATIVE conv-layout blocks.

This is the only form in which the SmolVLA experts reach an accelerator.
Rewriting a `[B, M, K]` transformer's MatMuls to Conv1x1 in place (see
rewrite_matmul_to_conv1x1.py) wraps every conv in a rank-3 -> rank-4 -> rank-3
Transpose/Reshape pair, and the Hexagon v66 DSP then refuses to finalize the
graph -- `Error from rpc` / `finalizeInit()` / error 6022 -- at EVERY size
tested, 1588 ops down to a single 103-op layer, while the same layer in MatMul
form composes at 73 ops. Built natively as `[1, C, 1, S]` Conv(1x1) and
converted WITHOUT `--preserve_io layout`, the converter declares the input NHWC
`[1, 1, S, C]`, emits no layout ops, and every block composes on cpu, dsp and
hta.

Three blocks per layer cover all seven linears:

    <p>_qkv    q, k, v          from the post-RMSNorm activation
    <p>_oproj  attention out    from the attention output
    <p>_mlp    gate, up, down   from the second post-RMSNorm activation, with
                                Sigmoid and two Mul (SwiGLU) kept in the block

Calibration is the genuine activation each block sees, replayed through the
source model in onnxruntime, not synthesized.

    python3 build_nativeconv_blocks.py --model smolvlm_expert_prefill_trunk.onnx \
        --calib-dir expert_rewrite/prefill_calib --seq 113 \
        --out expert_rewrite/nativeconv --prefix nc
    python3 build_nativeconv_blocks.py --model smolvlm_expert_decode.onnx \
        --calib-dir expert_rewrite/decode_calib --seq 50 \
        --out expert_rewrite/nativeconv_dec --prefix ncd
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

# MatMul output names are stable across both experts (torch.export order)
LIN = {"q": "linear", "k": "linear_1", "v": "linear_2", "o": "linear_3",
       "gate": "linear_4", "up": "linear_5", "down": "linear_6"}
TAP = {"qkv": "mul_1", "oproj": "_unsafe_view_2", "mlp": "mul_14"}
ONNX_DT = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_}


def conv_w(x, name):
    """MatMul weight [K, N] -> Conv2d weight [N, K, 1, 1]."""
    return numpy_helper.from_array(
        np.ascontiguousarray(x.T.reshape(x.shape[1], x.shape[0], 1, 1)).astype(np.float32), name)


def dlc_layout(shape):
    """perform_axes_to_spatial_first_order: [1,A,B]->[1,B,A], [1,A,B,C]->[1,B,C,A]."""
    if len(shape) == 3:
        return [shape[0], shape[2], shape[1]], (0, 2, 1)
    if len(shape) == 4:
        return [shape[0], shape[2], shape[3], shape[1]], (0, 3, 1, 2)
    return list(shape), None


def build_feed(sess, calib_dir, k):
    """Rebuild one calibration sample in ONNX layout from the DLC-layout raws."""
    feed = {}
    for i in sess.get_inputs():
        p = os.path.join(calib_dir, f"{i.name}_{k:03d}.raw")
        if not os.path.exists(p):
            continue
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        dl, perm = dlc_layout(shp)
        a = np.fromfile(p, np.float32)
        if a.size != int(np.prod(shp)):
            raise SystemExit(f"{p}: {a.size} floats, expected {int(np.prod(shp))}")
        a = a.reshape(dl)
        if perm is not None:
            a = np.ascontiguousarray(a.transpose(perm))
        if "bool" in i.type:
            a = a.astype(bool)
        elif "int64" in i.type:
            a = a.astype(np.int64)
        elif "int32" in i.type:
            a = a.astype(np.int32)
        feed[i.name] = a
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="nc")
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = a.seq

    m = onnx.load(a.model)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    W = {}
    for n in g.node:
        if n.op_type == "MatMul":
            w = [x for x in n.input if x in init]
            if w:
                W[n.output[0]] = init[w[0]]
    miss = [k for k, v in LIN.items() if v not in W]
    if miss:
        raise SystemExit(f"{a.model}: missing linears {miss}; got {sorted(W)[:10]}")
    q, k_, v = W[LIN["q"]], W[LIN["k"]], W[LIN["v"]]
    o, gt, up, dn = W[LIN["o"]], W[LIN["gate"]], W[LIN["up"]], W[LIN["down"]]
    print(f"  {a.model}: q{q.shape} k{k_.shape} v{v.shape} o{o.shape} "
          f"gate{gt.shape} up{up.shape} down{dn.shape}")

    def vi(n, c):
        return helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, c, 1, S])

    def save(name, nodes, ins, outs, inits):
        mo = helper.make_model(helper.make_graph(nodes, name, ins, outs, inits),
                               opset_imports=[helper.make_opsetid("", 17)])
        mo.ir_version = 9
        onnx.checker.check_model(mo, full_check=False)
        dst = os.path.join(a.out, f"{name}.onnx")
        onnx.save(mo, dst)
        print(f"    {name:<14} {len(nodes)} ops  {os.path.getsize(dst)/1e6:5.1f} MB")

    P = a.prefix
    save(f"{P}_qkv",
         [helper.make_node("Conv", ["X", "Wq"], ["Q"], kernel_shape=[1, 1]),
          helper.make_node("Conv", ["X", "Wk"], ["K"], kernel_shape=[1, 1]),
          helper.make_node("Conv", ["X", "Wv"], ["V"], kernel_shape=[1, 1])],
         [vi("X", q.shape[0])],
         [vi("Q", q.shape[1]), vi("K", k_.shape[1]), vi("V", v.shape[1])],
         [conv_w(q, "Wq"), conv_w(k_, "Wk"), conv_w(v, "Wv")])
    save(f"{P}_oproj",
         [helper.make_node("Conv", ["X", "Wo"], ["Y"], kernel_shape=[1, 1])],
         [vi("X", o.shape[0])], [vi("Y", o.shape[1])], [conv_w(o, "Wo")])
    save(f"{P}_mlp",
         [helper.make_node("Conv", ["X", "Wg"], ["gg"], kernel_shape=[1, 1]),
          helper.make_node("Sigmoid", ["gg"], ["sg"]),
          helper.make_node("Mul", ["gg", "sg"], ["silu"]),
          helper.make_node("Conv", ["X", "Wu"], ["u"], kernel_shape=[1, 1]),
          helper.make_node("Mul", ["silu", "u"], ["h"]),
          helper.make_node("Conv", ["h", "Wd"], ["Y"], kernel_shape=[1, 1])],
         [vi("X", gt.shape[0])], [vi("Y", dn.shape[1])],
         [conv_w(gt, "Wg"), conv_w(up, "Wu"), conv_w(dn, "Wd")])

    # replay the source model to capture each block's real input
    have = {x.name for x in g.output}
    for t in TAP.values():
        if t not in have:
            g.output.extend([onnx.helper.make_empty_tensor_value_info(t)])
    tap = os.path.join(a.out, "tap.onnx")
    onnx.save(m, tap, save_as_external_data=True, location="tap.onnx.data",
              all_tensors_to_one_file=True)
    del m

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(tap, so, providers=["CPUExecutionProvider"])
    names = [x.name for x in s.get_outputs()]
    for kk in range(a.n):
        out = dict(zip(names, s.run(None, build_feed(s, a.calib_dir, kk))))
        for tag, tensor in TAP.items():
            arr = out[tensor]
            np.ascontiguousarray(arr.transpose(0, 2, 1).reshape(1, -1, 1, S),
                                 dtype=np.float32).tofile(
                os.path.join(a.out, f"in_{tag}_{kk:03d}.raw"))
    print(f"    tapped {len(TAP)} block inputs x {a.n} samples -> {a.out}")
    print("    convert these WITHOUT --preserve_io layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
