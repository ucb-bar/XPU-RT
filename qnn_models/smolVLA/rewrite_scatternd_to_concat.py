"""Rewrite the SmolVLA expert ScatterND pairs as a single Concat.

The expert graphs carry 64 (prefill) / 48 (decode) ScatterND ops, and neither
DSP nor HTA composes them -- which is why `SMOLVLA_DSP_SLICING_PLAN.md` put the
experts out of scope and why they still run CPU-only at 583.8 / 149.6 ms.

They are not general scatters. Every one of them falls into one of two classes,
verified across all 64 in prefill:

    32x  data = an ALL-ZERO initializer, indices = constant arange(32)      -> [0..31]
    32x  data = the output of the previous ScatterND, indices = arange+32   -> [32..63]

So each consecutive pair is

    tmp = ScatterND(zeros(64,...), [0..31],  A)     # tmp[0:32]=A, tmp[32:64]=0
    out = ScatterND(tmp,           [32..63], B)     # out[0:32]=A, out[32:64]=B

which is exactly `Concat([A, B], axis=0)`. The rewrite is value-identical, not
an approximation: the base is all zeros, the two index blocks are contiguous,
disjoint, cover the full axis, and are in order.

Concat composes on DSP and HTA, so this removes the blocker rather than
carving a CPU trampoline around it -- the same move that made the vision
encoder work when MatMul was rewritten to Conv1x1.

    python3 rewrite_scatternd_to_concat.py --in smolvlm_expert_prefill.onnx \
                                           --out smolvlm_expert_prefill_concat.onnx
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper


def _resolve_const(name, init, prod, depth=4):
    """Follow Cast/Identity back to an initializer, if there is one."""
    cur = name
    for _ in range(depth):
        if cur in init:
            return numpy_helper.to_array(init[cur])
        p = prod.get(cur)
        if p is None or p.op_type not in ("Cast", "Identity"):
            return None
        cur = p.input[0]
    return None


def rewrite(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    g = model.graph
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    drop, add = set(), []
    pairs = 0
    for second in g.node:
        if second.op_type != "ScatterND":
            continue
        first = prod.get(second.input[0])
        if first is None or first.op_type != "ScatterND":
            continue                                   # not the tail of a pair
        base = first.input[0]
        if base not in init or np.count_nonzero(numpy_helper.to_array(init[base])):
            continue                                   # base must be all zeros
        i1 = _resolve_const(first.input[1], init, prod)
        i2 = _resolve_const(second.input[1], init, prod)
        if i1 is None or i2 is None:
            continue
        a, b = i1.reshape(-1), i2.reshape(-1)
        n_axis = numpy_helper.to_array(init[base]).shape[0]
        # contiguous, disjoint, in order, and together covering the whole axis
        if not (np.array_equal(a, np.arange(len(a)))
                and np.array_equal(b, np.arange(len(a), len(a) + len(b)))
                and len(a) + len(b) == n_axis):
            continue
        # the intermediate must feed only this second scatter
        if len(consumers.get(first.output[0], [])) != 1:
            continue
        add.append(helper.make_node(
            "Concat", [first.input[2], second.input[2]], [second.output[0]],
            name=f"{second.name or 'scatter'}_as_concat", axis=0))
        drop.add(id(first)); drop.add(id(second))
        pairs += 1

    if pairs:
        kept = [n for n in g.node if id(n) not in drop]
        # keep topological order: insert each Concat where its second scatter was
        order = {id(n): i for i, n in enumerate(g.node)}
        newnodes = kept + add
        newnodes.sort(key=lambda n: order.get(id(n), 10**9))
        # the appended Concats have no original index; place them by output name
        pos = {}
        for i, n in enumerate(g.node):
            for o in n.output:
                pos[o] = i
        newnodes.sort(key=lambda n: pos.get(n.output[0], order.get(id(n), 10**9)))
        del g.node[:]
        g.node.extend(newnodes)
    return model, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()

    m = onnx.load(a.src, load_external_data=True)
    before = sum(1 for n in m.graph.node if n.op_type == "ScatterND")
    m, pairs = rewrite(m)
    after = sum(1 for n in m.graph.node if n.op_type == "ScatterND")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True,
              all_tensors_to_one_file=True, location=a.dst.split("/")[-1] + ".data")
    print(f"  {a.src}")
    print(f"    ScatterND {before} -> {after}   ({pairs} pairs rewritten to Concat)")
    print(f"    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
