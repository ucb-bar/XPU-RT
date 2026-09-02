"""Retype the SmolVLA experts' bool `attention_mask` graph input to float32.

After the ScatterND and Where rewrites, every OP in the expert composes on DSP
and HTA (0 of 1197 unsupported per snpe-dlc-info). The context build still fails:

    QnnDsp <E> Input[0] has incorrect Datatype 0x508
    Validate OpConfig failed ... Failed to successfully compose graph

0x508 is QNN_DATATYPE_BOOL_8 and Input[0] is `attention_mask`. The DSP backend
does not accept a boolean GRAPH INPUT -- an op-support fix cannot help, because
the rejection is on the input tensor itself.

rewrite_where_to_mask_arith.py already turns the mask into float immediately
(`Cast(cond, float32)`), so nothing downstream needs the bool: retyping the
input to float32 makes that Cast a no-op and removes the unsupported dtype.
Callers then pass 1.0/0.0 instead of True/False, which is what the mask
arithmetic consumes anyway.

    python3 rewrite_bool_input_to_float.py --in  smolvlm_expert_prefill_nomask.onnx \
                                           --out smolvlm_expert_prefill_f32mask.onnx
"""
from __future__ import annotations

import argparse
import onnx
from onnx import TensorProto


def rewrite(model, names):
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    changed = []
    for inp in g.input:
        if inp.name not in names:
            continue
        if inp.type.tensor_type.elem_type != TensorProto.BOOL:
            continue
        inp.type.tensor_type.elem_type = TensorProto.FLOAT
        changed.append(inp.name)

    # any value_info carrying the old bool type downstream of a retyped input
    # would now disagree; drop inferred shapes and let the checker re-infer.
    if changed:
        del g.value_info[:]
    return model, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--inputs", default="attention_mask")
    a = ap.parse_args()
    names = set(a.inputs.split(","))
    m = onnx.load(a.src, load_external_data=True)
    m, changed = rewrite(m, names)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    print(f"  {a.src}\n    retyped bool -> float32: {changed}\n    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
