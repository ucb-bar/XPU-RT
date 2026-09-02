#!/usr/bin/env python3
"""Break the RmsNorm fusion with a barrier that cannot be folded away.

`rewrite_block_rmsnorm_fusion.py` inserts `Mul` by a constant 1.0 between
ReduceMean and Add. That works only while the converter leaves it alone -- it
is a no-op, so `RemoveNoOps` / `SquashConstantInput` are entitled to delete it,
after which the matcher sees

    Pow -> ReduceMean -> Add -> Sqrt -> Reciprocal -> Mul -> Mul

again and emits `qti.aisw:RmsNorm`, for which Hexagon v66 ships no op package.
Skipping those two passes via `--ir_optimizer_config` does NOT prevent it
(measured: still 32 RmsNorm), and no RmsNorm pass is exposed in
`--dump_ir_optimizer_config_template` to disable directly.

This uses a barrier that is not a no-op and therefore cannot be legally
removed, while remaining bit-exact in float32. In

    r = 1 / sqrt(mean + eps)

scale the variance by 4 and undo it after the reciprocal:

    m4  = mean * 4.0
    a   = m4 + 4*eps        = 4*(mean + eps)
    s   = sqrt(a)           = 2*sqrt(mean + eps)
    rec = 1/s               = 0.5 * r
    out = rec * 2.0         = r

4.0, 2.0 and the scaled epsilon are exact powers of two times the original, so
every step is exact in binary floating point -- no rounding is introduced.

    python3 rewrite_rmsnorm_scale_barrier.py --in  X.onnx --out Y.onnx
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--scale", type=float, default=4.0,
                    help="variance scale; must be a power of two (default 4.0)")
    a = ap.parse_args()
    s = float(a.scale)
    assert s > 0 and abs(np.log2(s) - round(np.log2(s))) < 1e-12, "scale must be a power of two"
    inv = float(np.sqrt(s))

    m = onnx.load(a.src, load_external_data=True)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)

    made = 0
    for sq in [n for n in g.node if n.op_type == "Sqrt"]:
        add = prod.get(sq.input[0])
        if add is None or add.op_type != "Add":
            continue
        # Add takes (variance, eps_const) in either order. The variance side is
        # either the ReduceMean directly, or -- once
        # rewrite_block_rmsnorm_fusion.py has run -- its Mul-by-1.0 barrier.
        red_in = eps_in = None
        existing_bar = None
        for i in add.input:
            if i in init:
                eps_in = i
                continue
            src = prod.get(i)
            if src is None:
                continue
            if src.op_type == "ReduceMean":
                red_in = i
            elif src.op_type == "Mul":
                up = prod.get(src.input[0])
                if up is not None and up.op_type == "ReduceMean":
                    red_in, existing_bar = i, src
        if red_in is None or eps_in is None:
            continue
        rec = None
        for c in cons[sq.output[0]]:
            if c.op_type == "Reciprocal":
                rec = c
        if rec is None or len(cons[sq.output[0]]) != 1:
            continue

        tag = f"_rmsbar{made}"
        mul_in = None
        if existing_bar is not None:
            # Reuse the no-op barrier: turning its 1.0 into `s` makes the very
            # same node unremovable instead of adding another one.
            done = False
            for i, nm in enumerate(existing_bar.input):
                if nm in init:
                    old_c = numpy_helper.to_array(init[nm])
                    nc = numpy_helper.from_array(
                        (old_c.astype(np.float32) * s).astype(old_c.dtype),
                        f"rmsbar_s{tag}")
                    g.initializer.append(nc)
                    existing_bar.input[i] = nc.name
                    done = True
            if not done:
                existing_bar = None
        if existing_bar is None:
            # scale the variance before the epsilon add
            sc = numpy_helper.from_array(np.array(s, np.float32), f"rmsbar_s{tag}")
            g.initializer.append(sc)
            mul_in = onnx.helper.make_node("Mul", [red_in, sc.name], [f"rmsvar{tag}"],
                                           name=f"rmsvar_scale{tag}")
            for i, nm in enumerate(add.input):
                if nm == red_in:
                    add.input[i] = f"rmsvar{tag}"
        # 2. scale epsilon to match
        e = numpy_helper.to_array(init[eps_in])
        ne = numpy_helper.from_array((e * s).astype(e.dtype), f"rmseps{tag}")
        g.initializer.append(ne)
        for i, nm in enumerate(add.input):
            if nm == eps_in:
                add.input[i] = ne.name
        # 3. undo after the reciprocal
        old = rec.output[0]
        rec.output[0] = f"rmsrec{tag}"
        iv = numpy_helper.from_array(np.array(inv, np.float32), f"rmsbar_i{tag}")
        g.initializer.append(iv)
        mul_out = onnx.helper.make_node("Mul", [f"rmsrec{tag}", iv.name], [old],
                                        name=f"rmsrec_unscale{tag}")
        if mul_in is not None:
            g.node.insert(list(g.node).index(add), mul_in)
        g.node.insert(list(g.node).index(rec) + 1, mul_out)
        made += 1

    print(f"  {a.src}")
    print(f"    RMSNorm chains given a scale barrier: {made}  (scale {s}, undo {inv})")
    print(f"    total ops {len(g.node)}")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    print(f"    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
