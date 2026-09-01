#!/usr/bin/env python3
"""Cut small probes out of smolvlm_expert_decode.onnx for backend bisection.

A whole-graph `ComposeGraphs Failed with error = 1` names no op. These probes
narrow it down by structure rather than by guessing:

    probe_mlp     the SwiGLU block alone   (3 FullyConnected + Sigmoid + Mul)
                  -- 67.2% of decode's MACs live here, so this is also the
                     partial-offload candidate, not just a diagnostic
    probe_attn    the attention core       (batched MatMul + Where + Softmax)
    probe_layer   one complete decoder layer

    python3 make_decode_probes.py --out expert_rewrite/decode_probes
"""
from __future__ import annotations

import argparse
import os
import onnx
from onnx.utils import extract_model

PROBES = {
    # name        inputs                                              outputs
    "probe_mlp":   (["mul_14"],                                       ["linear_6"]),
    "probe_attn":  (["permute_3", "permute_4", "unsqueeze_6"],        ["matmul_1"]),
    "probe_layer": (["expert_embeds", "attention_mask", "position_ids",
                     "past_key_0", "past_value_0"],                   ["add_5"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm_expert_decode.onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for name, (ins, outs) in PROBES.items():
        if a.only and name != a.only:
            continue
        dst = os.path.join(a.out, f"{name}.onnx")
        try:
            extract_model(a.model, dst, ins, outs)
            m = onnx.load(dst, load_external_data=False)
            n = len(m.graph.node)
            sz = os.path.getsize(dst) / 1e6
            print(f"  {name:<12} {n:>4} ops  {sz:>7.1f} MB   {ins} -> {outs}")
        except Exception as e:
            print(f"  {name:<12} FAILED: {str(e)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
