#!/usr/bin/env python3
"""Graph rewrites that move a network onto a backend it could not otherwise
reach, applied as CANDIDATES and adopted only on measured evidence.

Phase 1 measured three distinct HTA blockers across the reference's network
set, each with the verbatim validator string the compose log printed:

    dronet_s[a-g]        QnnHtaunsupported op Batchnorm
    mlp_control_s[abdf]  unsupported elementwise neuson op 0 /
                         failed to create IHtaOp type name ElementWiseNeuron
    yolov8_nano_s[cefh]  QnnHtaunsupported op StridedSlice

`qnn_models/PARTITIONING_GUIDE.md` §5 is explicit that a rewrite is the first
move and a cut is the fallback: on dronet, full HTA after the BN rewrite
measured 4.14 ms against 7.23 ms for the best cut and 31.89 ms full DSP. So
each blocker is removed here rather than routed around, and the rewritten
variant is carried alongside the original so the cost model takes whichever
measures better.

EVERY REWRITE IS LABELLED numerics-preserving OR numerics-changing, and the
label travels with the variant into the binding manifest, the cost model and
the ledger. The two are different classes of claim and are never blurred:

  numerics-preserving   algebraically the same function, up to requantization
                        error. `fold_bn_into_conv`, `bn_to_mul_add`,
                        `flatten_gemm_to_conv`, `channel_slice_to_conv1x1`.
  numerics-changing     a different function. `elu_to_relu`. The scaled rungs
                        already carry seeded random init so no accuracy is
                        being claimed either way, but that is not a licence to
                        blur the categories.

Re-quantization after a rewrite is mandatory (PARTITIONING_GUIDE §8 rule 5:
splicing pre-quantized constants into a rewritten graph does not work), and
the driver in phase1_rewrite.py always re-runs qairt-quantizer over freshly
generated calibration data rather than reusing the base network's DLC.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

PRESERVING = {"onnxsim", "fold_bn_into_conv", "bn_to_mul_add",
              "flatten_gemm_to_conv", "channel_slice_to_conv1x1"}
CHANGING = {"elu_to_relu"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _inits(g):
    return {i.name: numpy_helper.to_array(i) for i in g.initializer}


def _put(g, name, arr):
    g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))


def _shapes(model):
    """Every intermediate tensor's shape, by running ORT once on random input.

    Shape inference alone is not enough here: the yolov8 export computes its
    channel-chunk bounds from Shape/Gather/Div, so the static shape of the
    Slice inputs only exists after those fold.
    """
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.append(onnx.ValueInfoProto(name=o))
                have.add(o)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    s = ort.InferenceSession(m.SerializeToString(), so,
                             providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    feed = {}
    for i in s.get_inputs():
        dims = [d if isinstance(d, int) else 1 for d in i.shape]
        feed[i.name] = rng.randn(*dims).astype(np.float32)
    names = [o.name for o in s.get_outputs()]
    out = s.run(names, feed)
    return {a: tuple(b.shape) for a, b in zip(names, out) if hasattr(b, "shape")}


# --------------------------------------------------------------------------
# rewrites
# --------------------------------------------------------------------------

def onnxsim(model, **_):
    """Constant-fold and shape-specialise. NUMERICS-PRESERVING.

    Run first: the yolov8 export derives its channel-chunk bounds through
    Shape/Gather/Div, and until those fold the Slice ops have no static
    channel range for `channel_slice_to_conv1x1` to read.
    """
    from onnxsim import simplify
    m2, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim reported the simplified model did not check out")
    return m2, {"nodes_before": len(model.graph.node),
                "nodes_after": len(m2.graph.node)}


def fold_bn_into_conv(model, **_):
    """BatchNormalization directly after a Conv -> folded into the conv weights.
    NUMERICS-PRESERVING (exact, up to float rounding).

    Only applies where the BN's input is a Conv output with no other consumer.
    dronet's BNs sit after the residual Add, so this folds none of them there
    and `bn_to_mul_add` is what removes the blocker; it is kept because it is
    the cheaper transform wherever it does apply.
    """
    g = model.graph
    init = _inits(g)
    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    folded = 0
    drop = set()
    for n in list(g.node):
        if n.op_type != "BatchNormalization":
            continue
        src = prod.get(n.input[0])
        if src is None or src.op_type != "Conv" or len(consumers.get(n.input[0], [])) != 1:
            continue
        if any(x not in init for x in n.input[1:5]) or src.input[1] not in init:
            continue
        eps = 1e-5
        for a in n.attribute:
            if a.name == "epsilon":
                eps = a.f
        gamma, beta, mean, var = (init[x] for x in n.input[1:5])
        s = gamma / np.sqrt(var + eps)
        W = init[src.input[1]]
        Wn = W * s.reshape([-1] + [1] * (W.ndim - 1))
        b = init[src.input[2]] if len(src.input) > 2 and src.input[2] in init \
            else np.zeros(W.shape[0], dtype=np.float32)
        bn_ = (b - mean) * s + beta
        wn, bn_name = src.input[1] + "_bnfold", src.input[1] + "_bnfold_b"
        _put(g, wn, Wn.astype(np.float32))
        _put(g, bn_name, bn_.astype(np.float32))
        src.input[1] = wn
        if len(src.input) > 2:
            src.input[2] = bn_name
        else:
            src.input.append(bn_name)
        src.output[0] = n.output[0]
        drop.add(id(n))
        folded += 1
    if folded:
        keep = [n for n in g.node if id(n) not in drop]
        del g.node[:]
        g.node.extend(keep)
    return model, {"folded": folded}


def bn_to_mul_add(model, **_):
    """BatchNormalization -> Mul + Add. NUMERICS-PRESERVING (exact rewrite of
    BN's own affine form: y = (x-mean)/sqrt(var+eps)*gamma + beta).

    This is the rewrite that removes `QnnHtaunsupported op Batchnorm`: HTA has
    mul_s8 and add_s8 but no Batchnorm, and dronet's BNs follow the residual
    Add rather than a Conv so there is nothing to fold them into.
    """
    g = model.graph
    init = _inits(g)
    out, n_rw = [], 0
    for n in g.node:
        if n.op_type != "BatchNormalization" or any(x not in init for x in n.input[1:5]):
            out.append(n)
            continue
        eps = 1e-5
        for a in n.attribute:
            if a.name == "epsilon":
                eps = a.f
        gamma, beta, mean, var = (init[x] for x in n.input[1:5])
        s = (gamma / np.sqrt(var + eps)).astype(np.float32)
        b = (beta - mean * s).astype(np.float32)
        base = (n.name or f"bn{n_rw}").replace("/", "_")
        sn, bn_, mid = base + "_s", base + "_b", base + "_mul"
        _put(g, sn, s.reshape(1, -1, 1, 1))
        _put(g, bn_, b.reshape(1, -1, 1, 1))
        out.append(helper.make_node("Mul", [n.input[0], sn], [mid], name=base + "_Mul"))
        out.append(helper.make_node("Add", [mid, bn_], [n.output[0]], name=base + "_Add"))
        n_rw += 1
    del g.node[:]
    g.node.extend(out)
    return model, {"rewritten": n_rw}


#: unary, shape-preserving activations a Flatten can be pushed past
_PASSTHROUGH = {"Relu", "Sigmoid", "Tanh", "Clip", "Elu", "LeakyRelu",
                "HardSigmoid", "HardSwish", "Softplus", "Identity"}


def flatten_gemm_to_conv(model, shapes=None, **_):
    """Flatten (+ pass-through activations) + Gemm head -> Conv2d with an HxW
    kernel. NUMERICS-PRESERVING (the same inner product, re-indexed).

    PARTITIONING_GUIDE §4.2. The Flatten is what the converter turns into the
    Transpose/Reshape that HTA rejects; expressing the head as a convolution
    over the whole spatial extent removes the layout change rather than cutting
    before it.

    The Flatten and its Gemm are NOT adjacent in dronet -- the export is
    Add -> Flatten -> Relu -> {Gemm, Gemm} -- so the rewrite walks forward
    through shape-agnostic activations to find the Gemm frontier, drops the
    Flatten, and leaves the activations to operate on the 4-D tensor, which is
    the same elementwise function. Graph outputs downstream become [1,N,1,1]
    instead of [1,N]: a shape change, not a value change. If anything on the
    path is not a pass-through activation or a constant-weight Gemm of the
    right K, the whole rewrite is abandoned rather than half-applied.
    """
    g = model.graph
    init = _inits(g)
    shapes = shapes or _shapes(model)
    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    n_rw, converted, drop, abandoned = 0, {}, set(), []
    for f in [n for n in g.node if n.op_type == "Flatten"]:
        src = shapes.get(f.input[0])
        if not src or len(src) != 4:
            abandoned.append((f.name, f"flatten input is {src}, not 4-D"))
            continue
        _, C, H, W = (int(x) for x in src)
        # walk forward to the Gemm frontier
        region, frontier, seen, ok = [f], [], set(), True
        queue = [f.output[0]]
        while queue and ok:
            t = queue.pop()
            if t in seen:
                continue
            seen.add(t)
            for c in consumers.get(t, []):
                if c.op_type == "Gemm":
                    frontier.append(c)
                elif c.op_type in _PASSTHROUGH:
                    region.append(c)
                    queue.extend(c.output)
                else:
                    ok = False
                    abandoned.append((f.name, f"{c.op_type} on the flat path "
                                              f"is not a pass-through"))
                    break
        if not ok or not frontier:
            if ok:
                abandoned.append((f.name, "no Gemm frontier"))
            continue
        # every frontier Gemm must be a constant-weight [N, C*H*W]
        plan = []
        for gm in frontier:
            if len(gm.input) < 2 or gm.input[1] not in init:
                ok = False
                abandoned.append((f.name, f"{gm.name}: non-constant weight"))
                break
            Wt = init[gm.input[1]]
            if not next((a.i for a in gm.attribute if a.name == "transB"), 0):
                Wt = Wt.T
            if Wt.ndim != 2 or Wt.shape[1] != C * H * W:
                ok = False
                abandoned.append((f.name, f"{gm.name}: K={Wt.shape} != "
                                          f"C*H*W={C*H*W}"))
                break
            plan.append((gm, Wt))
        if not ok:
            continue
        # apply: drop the Flatten, rewire, convert the Gemms
        for c in consumers.get(f.output[0], []):
            for i, t in enumerate(c.input):
                if t == f.output[0]:
                    c.input[i] = f.input[0]
        drop.add(id(f))
        for gm, Wt in plan:
            N = Wt.shape[0]
            base = (gm.name or f"gemm{n_rw}").replace("/", "_")
            wn = base + "_convw"
            _put(g, wn, Wt.reshape(N, C, H, W).astype(np.float32))
            ins = [gm.input[0], wn]
            if len(gm.input) > 2 and gm.input[2] in init:
                bn_ = base + "_convb"
                _put(g, bn_, init[gm.input[2]].astype(np.float32).reshape(-1))
                ins.append(bn_)
            converted[id(gm)] = helper.make_node(
                "Conv", ins, list(gm.output), name=base + "_Conv",
                kernel_shape=[H, W], strides=[1, 1], pads=[0, 0, 0, 0], group=1)
            n_rw += 1

    if n_rw:
        newn = []
        for n in g.node:
            if id(n) in drop:
                continue
            newn.append(converted.get(id(n), n))
        del g.node[:]
        g.node.extend(newn)
        del g.value_info[:]
        # the head outputs go 2-D -> 4-D; re-infer rather than leaving a
        # stale (and now wrong) declared shape, which the checker rejects
        for o in g.output:
            o.type.tensor_type.ClearField("shape")
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False,
                                                  data_prop=True)
    return model, {"rewritten": n_rw, "abandoned": abandoned[:6]}


def channel_slice_to_conv1x1(model, shapes=None, **_):
    """Static channel-axis Slice -> 1x1 Conv with a one-hot selection weight.
    NUMERICS-PRESERVING (a channel gather written as a convolution).

    This is what removes `QnnHtaunsupported op StridedSlice` on yolov8_nano.
    modelblaster's export writes the C2f channel chunk as a pair of Slice ops
    (ultralytics' `chunk(2, 1)`), and the converter lowers each to a
    StridedSlice, which HTA has no kernel for. A 1x1 conv whose weight is the
    corresponding row selection computes exactly the same tensor with an op
    HTA does have.

    Only static, single-axis, step-1 slices on axis 1 are rewritten; anything
    else is left alone and will still show up as a blocker.
    """
    g = model.graph
    init = _inits(g)
    shapes = shapes or _shapes(model)
    out, n_rw, skipped = [], 0, []
    for n in g.node:
        if n.op_type != "Slice" or len(n.input) < 3:
            out.append(n)
            continue
        try:
            starts = init[n.input[1]].reshape(-1)
            ends = init[n.input[2]].reshape(-1)
            axes = init[n.input[3]].reshape(-1) if len(n.input) > 3 else np.array([0])
            steps = init[n.input[4]].reshape(-1) if len(n.input) > 4 else np.array([1])
        except (KeyError, IndexError):
            skipped.append((n.name, "non-constant slice bounds"))
            out.append(n)
            continue
        src = shapes.get(n.input[0])
        if (src is None or len(src) != 4 or len(axes) != 1 or int(axes[0]) != 1
                or int(steps[0]) != 1):
            skipped.append((n.name, f"axes={list(axes)} steps={list(steps)} "
                                    f"src={src}"))
            out.append(n)
            continue
        C = int(src[1])
        a = int(starts[0]) if starts[0] >= 0 else C + int(starts[0])
        b = min(int(ends[0]) if ends[0] >= 0 else C + int(ends[0]), C)
        a, b = max(0, a), max(0, b)
        if b <= a:
            skipped.append((n.name, f"empty range {a}:{b}"))
            out.append(n)
            continue
        W = np.zeros((b - a, C, 1, 1), dtype=np.float32)
        for k in range(b - a):
            W[k, a + k, 0, 0] = 1.0
        base = (n.name or f"slice{n_rw}").replace("/", "_")
        wn = base + "_selw"
        _put(g, wn, W)
        out.append(helper.make_node("Conv", [n.input[0], wn], [n.output[0]],
                                    name=base + "_SelConv", kernel_shape=[1, 1],
                                    strides=[1, 1], pads=[0, 0, 0, 0], group=1))
        n_rw += 1
    del g.node[:]
    g.node.extend(out)
    return model, {"rewritten": n_rw, "skipped": skipped[:8],
                   "n_skipped": len(skipped)}


def elu_to_relu(model, **_):
    """Elu -> Relu. **NUMERICS-CHANGING** — a different function, not an
    algebraic rewrite.

    HTA rejects Elu (`unsupported elementwise neuson op 0`, `failed to create
    IHtaOp type name ElementWiseNeuron`) and there is no frozen-constant linear
    equivalent, so the only way to put mlp_control on HTA at all is to
    substitute the activation. Carried as a candidate so the question "would
    HTA be worth it for the control loop?" is answered by measurement rather
    than by argument; every number produced from this variant is labelled
    numerics-changing wherever it appears.
    """
    g = model.graph
    n_rw = 0
    for n in g.node:
        if n.op_type == "Elu":
            n.op_type = "Relu"
            del n.attribute[:]
            n_rw += 1
    return model, {"rewritten": n_rw}


REWRITES = {
    "onnxsim": onnxsim,
    "fold_bn_into_conv": fold_bn_into_conv,
    "bn_to_mul_add": bn_to_mul_add,
    "flatten_gemm_to_conv": flatten_gemm_to_conv,
    "channel_slice_to_conv1x1": channel_slice_to_conv1x1,
    "elu_to_relu": elu_to_relu,
}


def apply_chain(src_path, dst_path, chain):
    """Apply named rewrites in order. Returns (report, numerics_label)."""
    model = onnx.load(src_path)
    report, shapes = [], None
    for name in chain:
        fn = REWRITES.get(name)
        if fn is None:
            raise ValueError(f"unknown rewrite {name!r}; have {sorted(REWRITES)}")
        if name in ("channel_slice_to_conv1x1", "flatten_gemm_to_conv"):
            shapes = _shapes(model)
        model, info = fn(model, shapes=shapes)
        shapes = None
        report.append(dict(rewrite=name, numerics=("preserving" if name in PRESERVING
                                                   else "changing"), **info))
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst_path)
    label = "changing" if any(c in CHANGING for c in chain) else "preserving"
    from collections import Counter
    return dict(chain=list(chain), numerics=label, steps=report,
                nodes=len(model.graph.node),
                op_types=dict(Counter(n.op_type for n in model.graph.node))), label


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain", required=True, help="comma list of rewrite names")
    a = ap.parse_args()
    rep, _ = apply_chain(a.onnx, a.out, [x for x in a.chain.split(",") if x])
    print(json.dumps(rep, indent=1))
