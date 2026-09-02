"""Constant-fold the SmolVLA experts' rotary Sin/Cos.

This is the LAST blocker, and unlike the previous two it was found empirically
rather than predicted. With the bool input retyped and the graph quantized, the
DSP context build fails at:

    NATIVE OpValidator::validateOpConfig node_Sin_93:qti.aisw:ElementWiseUnary
    QnnDsp <E> Param[0] has incorrect Value 14.

Param[0]=14 is the Sin opcode: the DSP's ElementWiseUnary does not implement it.
No slicing fixes that -- the op simply has no DSP kernel.

Sin and Cos here are rotary position embeddings and depend on NOTHING but
`position_ids`:

    position_ids -> Unsqueeze -> Cast -> Div -> Unsqueeze -> {Sin, Cos}

so for a fixed position sequence they are constants. This evaluates them once
and replaces the two nodes with initializers.

IMPORTANT -- this is the one rewrite in the set that is NOT unconditionally
value-identical. It is exact only while `position_ids` equals the sequence it
was folded at (default arange(seq)). That is what this export is: a
fixed-shape [1,113] prefill with static positions. If position_ids ever varies
at runtime, this fold is wrong and the sin/cos must instead be lifted to graph
inputs. The assumption is recorded in the model's doc_string.

    python3 rewrite_fold_rotary.py --in  smolvlm_expert_prefill_f32mask.onnx \
                                   --out smolvlm_expert_prefill_norot.onnx
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
    ap.add_argument("--seq", type=int, default=113)
    ap.add_argument("--start", type=int, default=0,
                    help="first position id. Prefill starts at 0; a decode whose\n"
                         "tokens follow an N-token cache starts at N (SmolVLA decode: 113).")
    a = ap.parse_args()

    import onnxruntime as ort
    m = onnx.load(a.src, load_external_data=True)
    g = m.graph
    targets = [n for n in g.node if n.op_type in ("Sin", "Cos")]
    if not targets:
        print("  no Sin/Cos found"); return 0

    # expose the sin/cos tensors so we can read their folded values
    probe = onnx.load(a.src, load_external_data=True)
    have = {o.name for o in probe.graph.output}
    for n in targets:
        if n.output[0] not in have:
            probe.graph.output.append(helper.make_empty_tensor_value_info(n.output[0]))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feed = {}
    for i in s.get_inputs():
        sh = [d if isinstance(d, int) else 1 for d in i.shape]
        if i.name == "position_ids":
            # sequence length comes from the model, not the flag -- prefill is
            # 113 and decode is 50, and folding at the wrong length is silent.
            n_pos = int(np.prod(sh))
            # Length is read from the model, but the OFFSET cannot be -- a decode
            # step's tokens sit AFTER the cached prefix, so its positions are
            # arange(start, start+n) and folding at arange(n) is silently wrong.
            feed[i.name] = np.arange(a.start, a.start + n_pos,
                                     dtype=np.int64).reshape(sh)
        elif "int" in i.type:
            feed[i.name] = np.zeros(sh, np.int64)
        else:
            feed[i.name] = rng.standard_normal(sh).astype(np.float32)
    names = [o.name for o in s.get_outputs()]
    vals = dict(zip(names, s.run(None, feed)))

    drop = set()
    for n in targets:
        v = np.asarray(vals[n.output[0]])
        g.initializer.append(numpy_helper.from_array(v.astype(np.float32), n.output[0]))
        drop.add(id(n))
        print(f"    folded {n.op_type:<4} {n.name:<18} -> constant {v.shape} "
              f"range [{v.min():+.4f}, {v.max():+.4f}]")

    kept = [n for n in g.node if id(n) not in drop]
    del g.node[:]; g.node.extend(kept)

    # drop anything upstream that is now unreachable
    for _ in range(12):
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead: break
        ids = {id(n) for n in dead}
        alive = [n for n in g.node if id(n) not in ids]
        del g.node[:]; g.node.extend(alive)

    m.doc_string = (f"rotary Sin/Cos constant-folded at "
                    f"position_ids=arange({a.start}, {a.start}+n). "
                    "Valid ONLY for that position sequence.")
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=a.dst.split("/")[-1] + ".data")
    left = sum(1 for n in m.graph.node if n.op_type in ("Sin", "Cos"))
    print(f"  {a.src}\n    Sin/Cos {len(targets)} -> {left}\n    wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
