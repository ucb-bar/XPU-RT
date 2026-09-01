#!/usr/bin/env python3
"""Remove Transposes that only exist to serve the ScatterND that is no longer there.

The exported RoPE wrapped each rotary half in a Transpose purely so ScatterND
could index axis 0:

    A -T(3,1,2,0)-> ... -.
                          Concat(axis=0) -> C -T(3,2,1,0)-> out
    B -T(3,1,2,0)-> ... -'

`rewrite_scatternd_to_concat.py` replaced the ScatterND with that Concat, but a
Concat can join on ANY axis, so the wrapping transposes became dead weight.
Two standard passes remove them:

  1. push-transpose-through-concat
        Concat([T(X_i, p)], axis=a)  ==  T(Concat([X_i], axis=p[a]), p)
     applied only when every input is a Transpose with the SAME perm and each
     feeds nothing else.

  2. fuse-consecutive-transposes
        T(T(X, p), q) == T(X, r)  where r[k] = p[q[k]]
     followed by dropping any perm that came out as the identity.

ONNX Transpose semantics: output axis k reads input axis perm[k].

    python3 rewrite_collapse_transposes.py --in  smolvlm_expert_decode_norot113.onnx \
                                           --out smolvlm_expert_decode_flat.onnx
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import onnx


def _perm(n):
    for a in n.attribute:
        if a.name == "perm":
            return list(a.ints)
    return None


def _consumers(g):
    c = defaultdict(list)
    for n in g.node:
        for i in n.input:
            c[i].append(n)
    return c


def push_through_concat(g):
    """Concat([T(X_i,p)],axis=a) -> T(Concat([X_i],axis=p[a]),p)."""
    cons = _consumers(g)
    prod = {o: n for n in g.node for o in n.output}
    outs = {o.name for o in g.output}
    made = 0
    for cat in [n for n in g.node if n.op_type == "Concat"]:
        srcs = [prod.get(i) for i in cat.input]
        if not srcs or any(s is None or s.op_type != "Transpose" for s in srcs):
            continue
        perms = [_perm(s) for s in srcs]
        p = perms[0]
        if p is None or any(q != p for q in perms):
            continue
        # each feeding Transpose must be private to this Concat
        if any(len(cons[s.output[0]]) != 1 or s.output[0] in outs for s in srcs):
            continue
        a = next((at.i for at in cat.attribute if at.name == "axis"), 0)
        if a < 0:
            a += len(p)
        mid = cat.output[0] + "_pre_t"
        for i, s in enumerate(srcs):
            cat.input[i] = s.input[0]
        for at in cat.attribute:
            if at.name == "axis":
                at.i = p[a]
        old = cat.output[0]
        cat.output[0] = mid
        t = onnx.helper.make_node("Transpose", [mid], [old],
                                  name=(cat.name or "cat") + "_pushed_t", perm=p)
        idx = list(g.node).index(cat)
        g.node.insert(idx + 1, t)
        made += 1
    return made


def fuse_chains(g):
    """T(T(X,p),q) -> T(X, r), r[k]=p[q[k]]; then drop identity perms."""
    fused = dropped = 0
    changed = True
    while changed:
        changed = False
        cons = _consumers(g)
        prod = {o: n for n in g.node for o in n.output}
        outs = {o.name for o in g.output}
        for second in [n for n in g.node if n.op_type == "Transpose"]:
            first = prod.get(second.input[0])
            if first is None or first.op_type != "Transpose":
                continue
            if len(cons[first.output[0]]) != 1 or first.output[0] in outs:
                continue
            p, q = _perm(first), _perm(second)
            if p is None or q is None:
                continue
            r = [p[k] for k in q]
            second.input[0] = first.input[0]
            for a in second.attribute:
                if a.name == "perm":
                    del a.ints[:]
                    a.ints.extend(r)
            g.node.remove(first)
            fused += 1
            changed = True
            break
    # drop identity transposes
    cons = _consumers(g)
    outs = {o.name for o in g.output}
    for t in [n for n in g.node if n.op_type == "Transpose"]:
        p = _perm(t)
        if p is None or p != sorted(p) or t.output[0] in outs:
            continue
        for c in cons[t.output[0]]:
            for i, nm in enumerate(c.input):
                if nm == t.output[0]:
                    c.input[i] = t.input[0]
        g.node.remove(t)
        dropped += 1
    return fused, dropped


def drop_dead(g):
    """Remove nodes whose outputs nothing reads. Pass 1 orphans the wrapping
    Transposes rather than deleting them in place, so this collects them."""
    outs = {o.name for o in g.output}
    removed = 0
    changed = True
    while changed:
        changed = False
        used = set(outs)
        for n in g.node:
            used.update(n.input)
        for n in list(g.node):
            if all(o not in used for o in n.output):
                g.node.remove(n)
                removed += 1
                changed = True
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()

    m = onnx.load(a.src, load_external_data=True)
    g = m.graph
    before = sum(1 for n in g.node if n.op_type == "Transpose")
    pushed = push_through_concat(g)
    dead = drop_dead(g)
    fused, dropped = fuse_chains(g)
    dead += drop_dead(g)
    after = sum(1 for n in g.node if n.op_type == "Transpose")
    print(f"  {a.src}")
    print(f"    pushed through Concat  {pushed}")
    print(f"    fused chains           {fused}")
    print(f"    dropped identity       {dropped}")
    print(f"    dead nodes collected   {dead}")
    print(f"    Transpose {before} -> {after}   ({before-after} removed, "
          f"{100.0*(before-after)/max(before,1):.0f}%)")
    print(f"    total ops {len(g.node)}")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    print(f"    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
