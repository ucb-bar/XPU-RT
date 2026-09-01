"""Rewrite the SmolVLA expert attention-mask `Where` ops as mask arithmetic.

After rewrite_scatternd_to_concat.py, `Where` is the last substantial blocker
in both experts: 16 in prefill, 16 in decode. Neither DSP nor HTA composes it.

Every one has the same shape -- verified across all 16 in each graph:

    Where(cond, scores, -3.4028235e+38)

with a single shared `cond` (the broadcast attention mask) and Y the scalar
-FLT_MAX. This is standard additive attention masking.

The exactly-equivalent arithmetic form is

    mask_f = Cast(cond, float32)                     # 1.0 keep, 0.0 mask
    neg    = (1 - mask_f) * -FLT_MAX                 # 0.0 keep, -FLT_MAX mask
    out    = scores * mask_f + neg

which is value-identical, not approximate:
    cond true  -> scores*1 + 0        = scores
    cond false -> scores*0 + -FLT_MAX = -FLT_MAX

Note this is stronger than the usual `scores + bias` trick, which relies on
-FLT_MAX absorbing the addend and is only bit-exact because the ulp at that
magnitude is huge. Multiplying the masked lane to exactly zero first avoids
depending on that.

`mask_f` and `neg` are computed once and shared by all 16 sites, so the graph
grows by 3 setup ops and 2 ops per site, all of which compose on DSP and HTA.

    python3 rewrite_where_to_mask_arith.py --in  smolvlm_expert_prefill_concat.onnx \
                                           --out smolvlm_expert_prefill_nomask.onnx
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NEG = -3.4028234663852886e38


def rewrite(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    g = model.graph
    init = {i.name: i for i in g.initializer}

    targets = []
    for n in g.node:
        if n.op_type != "Where":
            continue
        y = n.input[2]
        if y not in init:
            continue
        v = numpy_helper.to_array(init[y])
        if v.size != 1 or not np.isclose(float(v.reshape(-1)[0]), NEG, rtol=1e-6):
            continue                      # only the -FLT_MAX masking pattern
        targets.append(n)
    if not targets:
        return model, 0

    # One setup chain per distinct mask tensor. prefill shares a single mask;
    # decode has two (self- and cross-attention), so this must not assume one.
    conds = sorted({n.input[0] for n in targets})
    g.initializer.extend([
        helper.make_tensor("wmask_one", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("wmask_neg", TensorProto.FLOAT, [], [NEG]),
    ])
    names = {c: (f"wmask_f_{i}", f"wmask_bias_{i}") for i, c in enumerate(conds)}
    setup = {}
    for c in conds:
        f_, b_ = names[c]
        setup[c] = [
            helper.make_node("Cast", [c], [f_], name=f"{f_}_cast", to=TensorProto.FLOAT),
            helper.make_node("Sub", ["wmask_one", f_], [f"{f_}_inv"], name=f"{f_}_sub"),
            helper.make_node("Mul", [f"{f_}_inv", "wmask_neg"], [b_], name=f"{b_}_mul"),
        ]

    drop = {id(n) for n in targets}
    new = []
    for n in g.node:
        if id(n) not in drop:
            new.append(n); continue
        f_, b_ = names[n.input[0]]
        base = n.name or f"where_{len(new)}"
        new.append(helper.make_node("Mul", [n.input[1], f_],
                                    [f"{base}__masked"], name=f"{base}_mul"))
        new.append(helper.make_node("Add", [f"{base}__masked", b_],
                                    [n.output[0]], name=f"{base}_add"))
    # each setup chain must precede its first use
    for c in reversed(conds):
        f_, b_ = names[c]
        first = min(i for i, n in enumerate(new) if f_ in n.input or b_ in n.input)
        new[first:first] = setup[c]

    del g.node[:]
    g.node.extend(new)
    return model, len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()
    m = onnx.load(a.src, load_external_data=True)
    before = sum(1 for n in m.graph.node if n.op_type == "Where")
    m, k = rewrite(m)
    after = sum(1 for n in m.graph.node if n.op_type == "Where")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    print(f"  {a.src}\n    Where {before} -> {after}   ({k} rewritten)\n    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
