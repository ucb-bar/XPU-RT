#!/usr/bin/env python3
"""Parametrised single-operator ONNX models, one per (op kind, size point).

The point of a one-op model is that it isolates the thing being measured. A
whole network's time is a sum over ops plus whatever the runtime charges per
dispatch, and those two are not separable after the fact. Sweeping ONE op over
a size ladder makes them separable: fit `t = overhead + work/throughput` and the
intercept IS the dispatch cost. That is how the SmolVLA work established HTA's
543 us warm / ~2470 us cold floor, and this generalises it to every op kind in
the model family.

Op kinds are taken from the IRs actually in the repo, not invented -- see
`FAMILIES` for which model each one comes from. Where a kind has a known better
mapping, BOTH forms are emitted so the sweep measures the choice rather than
assuming it: `linear` vs `linear_conv1x1` is the important pair, since QNN's
FullyConnected kernel measured ~13x slower than its Conv2d kernel for identical
arithmetic on HTA and DSP.

Layout matters as much as op choice on this board. Conv-shaped models are built
natively as NCHW `[1,C,H,W]` and converted WITHOUT `--preserve_io layout`, so the
converter declares the input NHWC and emits no layout ops. A rank-3 `[B,M,K]`
wrapper around a conv is what makes the v66 DSP refuse to finalize a graph, so
it is avoided everywhere here.

    python3 opzoo.py --list
    python3 opzoo.py --emit conv2d --out /tmp/z
"""
from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    FP32 = TensorProto.FLOAT
except ImportError:
    # The builders only ever run inside the qnn-convert container, which has
    # onnx.  On the host we still want `--list` and the grid to import.
    onnx = helper = numpy_helper = TensorProto = None
    FP32 = 1  # TensorProto.FLOAT
_rng = np.random.default_rng(0)


def _t(name, shape):
    return helper.make_tensor_value_info(name, FP32, list(shape))


def _w(shape, name, scale=0.05):
    return numpy_helper.from_array((_rng.standard_normal(shape) * scale).astype(np.float32), name)


def _model(nodes, ins, outs, inits, name):
    g = helper.make_graph(nodes, name, ins, outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m, full_check=False)
    return m


# --------------------------------------------------------------------------
# builders. each returns (model, input_shapes: {name: shape}, macs)
# --------------------------------------------------------------------------

def conv2d(C_in, C_out, HW, K=3):
    p = K // 2
    n = [helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[K, K], pads=[p, p, p, p])]
    return (_model(n, [_t("X", [1, C_in, HW, HW])], [_t("Y", [1, C_out, HW, HW])],
                   [_w((C_out, C_in, K, K), "W")], "conv2d"),
            {"X": [1, C_in, HW, HW]}, HW * HW * C_in * C_out * K * K)


def depthwise_conv2d(C, HW, K=3):
    p = K // 2
    n = [helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[K, K],
                          pads=[p, p, p, p], group=C)]
    return (_model(n, [_t("X", [1, C, HW, HW])], [_t("Y", [1, C, HW, HW])],
                   [_w((C, 1, K, K), "W")], "depthwise_conv2d"),
            {"X": [1, C, HW, HW]}, HW * HW * C * K * K)


def conv1x1(C_in, C_out, S):
    """The mapping target for a constant-weight linear. `S` is the token count
    laid out on the W axis, which is the shape the expert blocks use."""
    n = [helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])]
    return (_model(n, [_t("X", [1, C_in, 1, S])], [_t("Y", [1, C_out, 1, S])],
                   [_w((C_out, C_in, 1, 1), "W")], "conv1x1"),
            {"X": [1, C_in, 1, S]}, S * C_in * C_out)


def linear(M, K, N):
    """Constant-weight MatMul -- the converter lowers this to FullyConnected.
    Paired with conv1x1 at the same arithmetic to measure the kernel choice."""
    n = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return (_model(n, [_t("X", [M, K])], [_t("Y", [M, N])], [_w((K, N), "W")], "linear"),
            {"X": [M, K]}, M * K * N)


def matmul_dyn(B, M, K, N):
    """Two activation operands -- attention's QK^T / A.V. HTA has no kernel for
    this at any rank, which is why attention is HTA-impossible by construction."""
    n = [helper.make_node("MatMul", ["A", "Bm"], ["Y"])]
    return (_model(n, [_t("A", [1, B, M, K]), _t("Bm", [1, B, K, N])],
                   [_t("Y", [1, B, M, N])], [], "matmul_dyn"),
            {"A": [1, B, M, K], "Bm": [1, B, K, N]}, B * M * K * N)


def softmax(rows, cols):
    n = [helper.make_node("Softmax", ["X"], ["Y"], axis=-1)]
    return (_model(n, [_t("X", [1, rows, cols])], [_t("Y", [1, rows, cols])], [], "softmax"),
            {"X": [1, rows, cols]}, rows * cols)


def layernorm(rows, C):
    n = [helper.make_node("LayerNormalization", ["X", "S", "B"], ["Y"], axis=-1)]
    return (_model(n, [_t("X", [1, rows, C])], [_t("Y", [1, rows, C])],
                   [_w((C,), "S", 1.0), _w((C,), "B", 0.01)], "layernorm"),
            {"X": [1, rows, C]}, rows * C)


def _act(kind, n_elem):
    if kind == "gelu":
        # Not a single node: `Gelu` is only registered from opset 20 and the
        # converter is on 17.  Decomposing is the faithful thing anyway -- this
        # tanh-approximation chain is exactly what SmolVLA's graph contains and
        # what the DSP and HTA are actually asked to execute.
        #   0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        c = [numpy_helper.from_array(np.array([v], np.float32), nm) for nm, v in
             (("g_half", 0.5), ("g_one", 1.0), ("g_k", 0.044715),
              ("g_s", 0.7978845608028654), ("g_three", 3.0))]
        n = [helper.make_node("Pow", ["X", "g_three"], ["x3"]),
             helper.make_node("Mul", ["x3", "g_k"], ["kx3"]),
             helper.make_node("Add", ["X", "kx3"], ["inner"]),
             helper.make_node("Mul", ["inner", "g_s"], ["scaled"]),
             helper.make_node("Tanh", ["scaled"], ["t"]),
             helper.make_node("Add", ["t", "g_one"], ["t1"]),
             helper.make_node("Mul", ["X", "g_half"], ["xh"]),
             helper.make_node("Mul", ["xh", "t1"], ["Y"])]
        return (_model(n, [_t("X", [1, n_elem])], [_t("Y", [1, n_elem])], c, kind),
                {"X": [1, n_elem]}, n_elem)
    node = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh",
            "elu": "Elu"}[kind]
    n = [helper.make_node(node, ["X"], ["Y"])]
    return (_model(n, [_t("X", [1, n_elem])], [_t("Y", [1, n_elem])], [], kind),
            {"X": [1, n_elem]}, n_elem)


def _binary(kind, n_elem):
    node = {"add": "Add", "mul": "Mul"}[kind]
    n = [helper.make_node(node, ["A", "B"], ["Y"])]
    return (_model(n, [_t("A", [1, n_elem]), _t("B", [1, n_elem])],
                   [_t("Y", [1, n_elem])], [], kind),
            {"A": [1, n_elem], "B": [1, n_elem]}, n_elem)


def maxpool2d(C, HW, K=2):
    n = [helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[K, K], strides=[K, K])]
    return (_model(n, [_t("X", [1, C, HW, HW])], [_t("Y", [1, C, HW // K, HW // K])],
                   [], "maxpool2d"),
            {"X": [1, C, HW, HW]}, C * HW * HW)


def avgpool_global(C, HW):
    n = [helper.make_node("GlobalAveragePool", ["X"], ["Y"])]
    return (_model(n, [_t("X", [1, C, HW, HW])], [_t("Y", [1, C, 1, 1])], [], "avgpool_global"),
            {"X": [1, C, HW, HW]}, C * HW * HW)


def transpose(rows, cols):
    n = [helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])]
    return (_model(n, [_t("X", [1, rows, cols])], [_t("Y", [1, cols, rows])], [], "transpose"),
            {"X": [1, rows, cols]}, rows * cols)


def concat_c(C, S, n_in=2):
    ins = [_t(f"X{i}", [1, C, 1, S]) for i in range(n_in)]
    n = [helper.make_node("Concat", [f"X{i}" for i in range(n_in)], ["Y"], axis=1)]
    return (_model(n, ins, [_t("Y", [1, C * n_in, 1, S])], [], "concat_c"),
            {f"X{i}": [1, C, 1, S] for i in range(n_in)}, C * S * n_in)


BUILDERS = {
    "conv2d": conv2d, "depthwise_conv2d": depthwise_conv2d, "conv1x1": conv1x1,
    "linear": linear, "matmul_dyn": matmul_dyn, "softmax": softmax,
    "layernorm": layernorm, "maxpool2d": maxpool2d, "avgpool_global": avgpool_global,
    "transpose": transpose, "concat_c": concat_c,
}
for _k in ("relu", "sigmoid", "tanh", "gelu", "elu"):
    BUILDERS[_k] = (lambda k: (lambda n_elem: _act(k, n_elem)))(_k)
for _k in ("add", "mul"):
    BUILDERS[_k] = (lambda k: (lambda n_elem: _binary(k, n_elem)))(_k)

# which model each op kind was taken from, so the sweep can be justified
FAMILIES = {
    "conv2d": "dronet, yolov8, vint encoders, fusedsensornet, smolvla patch-embed",
    "depthwise_conv2d": "vint (EfficientNet-b0)",
    "conv1x1": "smolvla experts + vision (the Conv1x1 mapping target), vint decoder",
    "linear": "mlp_control, dronet head, vint decoder, smolvla (FullyConnected form)",
    "matmul_dyn": "smolvla attention, vint decoder attention",
    "softmax": "smolvla vision + experts, vint decoder",
    "layernorm": "vint decoder, smolvla (RMSNorm is the same shape of work)",
    "relu": "dronet, fusedsensornet, vint", "sigmoid": "vint, dronet, smolvla SwiGLU",
    "tanh": "smolvla vision GELU trampolines", "gelu": "vint decoder, smolvla vision",
    "elu": "mlp_control", "add": "dronet residuals, vint, smolvla residuals",
    "mul": "vint SE blocks, smolvla SwiGLU", "maxpool2d": "dronet",
    "avgpool_global": "vint (EfficientNet SE)", "transpose": "smolvla, vint (layout tax)",
    "concat_c": "fusedsensornet, vint, smolvla KV concat",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit"); ap.add_argument("--out", default="/tmp/opzoo")
    ap.add_argument("--params", default="")
    a = ap.parse_args()
    if a.list:
        for k in sorted(BUILDERS):
            print(f"  {k:<20} {FAMILIES.get(k,'')}")
        return 0
    if a.emit:
        os.makedirs(a.out, exist_ok=True)
        prm = [int(x) for x in a.params.split(",")] if a.params else []
        m, shapes, macs = BUILDERS[a.emit](*prm)
        p = os.path.join(a.out, f"{a.emit}.onnx")
        onnx.save(m, p)
        print(f"  {p}  inputs={shapes}  macs={macs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
