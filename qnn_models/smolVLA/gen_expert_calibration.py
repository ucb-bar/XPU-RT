#!/usr/bin/env python3
"""Generate calibration data for quantizing the rewritten SmolVLA experts.

The experts take three inputs and no calibration set exists for them:

    vlm_embeds      float32 [1,113,960]   the fused vision+text+state embedding
    attention_mask  bool    [1,113,113]
    position_ids    int32   [1,113]   (int64 in ONNX; converter downcasts)

`vlm_embeds` is the one that matters for quantization quality, and it is the
one we can ground in reality: it is dominated by the vision encoder's output,
so this runs the REAL smolvlm_vision.onnx and uses its activation statistics
(per-channel mean/std) to shape the samples, rather than sampling white noise
at unit scale. That is weaker than capturing the true fused embedding -- doing
that properly needs the text encoder and state projector wired up too -- and
the difference is recorded here rather than glossed:

    vlm_embeds      REAL statistics from smolvlm_vision.onnx, resampled
    attention_mask  SYNTHETIC: all-visible, which is what a 113-token prefill
                    with no padding actually sees
    position_ids    EXACT: arange(113)

Consequence: this is sufficient to test whether the graph COMPOSES and how
FAST it runs, both of which are insensitive to calibration quality. It is NOT
sufficient to claim int8 accuracy -- that needs the true fused embedding.

    python3 gen_expert_calibration.py --model smolvlm_expert_prefill_nomask.onnx \
                                      --out expert_rewrite/calib --n 16
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def vision_stats(path, n_probe=4, seed=0):
    """Per-channel mean/std of the real vision encoder's output."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n_probe):
        feed = {}
        for i in s.get_inputs():
            sh = [d if isinstance(d, int) else 1 for d in i.shape]
            feed[i.name] = rng.standard_normal(sh).astype(np.float32)
        outs.append(s.run(None, feed)[0].reshape(-1, s.get_outputs()[0].shape[-1]))
    a = np.concatenate(outs, 0)
    return a.mean(0), a.std(0) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--vision", default="smolvlm_vision.onnx")
    ap.add_argument("--seq", type=int, default=113)
    ap.add_argument("--hidden", type=int, default=960)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1234)

    mu = sd = None
    if os.path.exists(a.vision):
        try:
            mu, sd = vision_stats(a.vision)
            print(f"  vision stats from {a.vision}: {mu.shape[0]} channels, "
                  f"mean |mu| {np.abs(mu).mean():.4f}, mean sd {sd.mean():.4f}")
        except Exception as e:
            print(f"  vision probe failed ({e}); falling back to unit normal")
    if mu is None:
        mu = np.zeros(a.hidden, np.float32); sd = np.ones(a.hidden, np.float32)
    if mu.shape[0] != a.hidden:                 # vision hidden != expert hidden
        idx = np.linspace(0, mu.shape[0] - 1, a.hidden).astype(int)
        mu, sd = mu[idx], sd[idx]
        print(f"  resampled vision channels -> {a.hidden}")

    lines = []
    for k in range(a.n):
        # The converter runs perform_axes_to_spatial_first_order, so the DLC
        # declares vlm_embeds as [1, hidden, seq] -- NOT the ONNX [1, seq, hidden].
        # Raws in the wrong order quantize against transposed statistics.
        emb = (rng.standard_normal((1, a.seq, a.hidden)).astype(np.float32)
               * sd.astype(np.float32) + mu.astype(np.float32))
        emb = np.ascontiguousarray(emb.transpose(0, 2, 1))        # -> [1,960,113]
        mask = np.ones((1, a.seq, a.seq), dtype=np.uint8)          # all visible
        # The converter runs keep_int64_inputs=False, so the DLC declares
        # position_ids as Int_32. An int64 raw is exactly 2x too large and the
        # quantizer reports it as a batch-size mismatch, not a dtype error.
        pos = np.arange(a.seq, dtype=np.int32).reshape(1, a.seq)
        f_e = os.path.join(a.out, f"vlm_embeds_{k:03d}.raw")
        f_m = os.path.join(a.out, f"attention_mask_{k:03d}.raw")
        f_p = os.path.join(a.out, f"position_ids_{k:03d}.raw")
        emb.tofile(f_e); mask.tofile(f_m); pos.tofile(f_p)
        lines.append(f"vlm_embeds:={f_e} attention_mask:={f_m} position_ids:={f_p}")

    lst = os.path.join(a.out, "input_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {a.n} samples -> {a.out}")
    print(f"  list -> {lst}")
    print("  NOTE: mask/position_ids are synthetic; only vlm_embeds is grounded "
          "in real vision statistics. Adequate for composition and speed, NOT "
          "for an int8 accuracy claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
