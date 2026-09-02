"""Stop the converter fusing SmolVLA's RMSNorm chains into an unsupported op.

With ScatterND, Where, the bool input and Sin/Cos all resolved, the DSP context
build clears ~1100 of ~1200 ops and dies on the last one:

    NATIVE OpValidator::validateOpConfig rms_norm_node_:qti.aisw:RmsNorm
    QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND: Could not find specified op package

This is NOT an unsupported op in the usual sense. The ONNX contains no norm op
at all -- it has 33 decomposed chains:

    Pow -> ReduceMean -> Add -> Sqrt -> Reciprocal -> Mul -> Mul

The converter pattern-matches that and emits a single `qti.aisw:RmsNorm`, and
the v66 DSP backend does not carry that op package. The decomposed ops all
validate individually (the earlier verbose log shows Pow, ReduceMean, Add,
Sqrt and Reciprocal each passing), so the fix is to stop the fusion, not to
find a different backend.

The fusion is not exposed in `--dump_ir_optimizer_config_template` (no RmsNorm
pass is listed), so it cannot be switched off by config. Instead this inserts a
neutral barrier -- Mul by a constant 1.0 -- between ReduceMean and Add in each
chain, which breaks the matcher's pattern while leaving the arithmetic
identical. Mul is ElementWiseBinary and validates on DSP.

Pair this with an ir_optimizer_config that skips RemoveNoOps and
SquashConstantInput, or the converter will simply fold the barrier away again.

    python3 rewrite_block_rmsnorm_fusion.py --in  smolvlm_expert_prefill_norot.onnx \
                                            --out smolvlm_expert_prefill_nofuse.onnx
"""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()

    m = onnx.load(a.src, load_external_data=True)
    g = m.graph
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    one = "rmsbar_one"
    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), one))

    new, k = [], 0
    for n in g.node:
        new.append(n)
        if n.op_type != "ReduceMean":
            continue
        outs = cons.get(n.output[0], [])
        if len(outs) != 1 or outs[0].op_type != "Add":
            continue                     # only the RMSNorm shape
        bar = f"{n.output[0]}__rmsbar"
        new.append(helper.make_node("Mul", [n.output[0], one], [bar],
                                    name=f"{n.name}_rmsbar"))
        for c in outs:
            for j, i in enumerate(c.input):
                if i == n.output[0]:
                    c.input[j] = bar
        k += 1

    del g.node[:]; g.node.extend(new)
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    print(f"  {a.src}\n    inserted {k} fusion barriers (Mul by 1.0 after ReduceMean)"
          f"\n    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
