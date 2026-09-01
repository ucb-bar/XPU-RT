#!/usr/bin/env python3
"""Generate calibration data for quantizing smolvlm_expert_decode.

Decode takes 35 inputs. 32 of them are the KV cache, and those do NOT have to
be synthesized: `smolvlm_expert_prefill.onnx` emits `present_key_N` /
`present_value_N` with exactly decode's `past_key_N` / `past_value_N` shapes,
so this runs the real prefill and feeds its cache straight through. That makes
32 of 35 inputs -- and ~96% of the calibration tensor volume -- real activations
rather than noise.

    expert_embeds   [1,50,720]     SYNTHETIC (see below)
    attention_mask  [1,50,163]     SYNTHETIC: all-visible
    position_ids    [1,50]         EXACT: arange(113,163), the action tokens
                                   sit after the 113-token prefix
    past_key_N      [1,113,5,64]   REAL, from prefill  (x16)
    past_value_N    [1,113,5,64]   REAL, from prefill  (x16)

On `expert_embeds`: it is the flow-matching action embedding, which needs the
action head to produce exactly. Sampled N(0,1) here. The scale error this
introduces is bounded in an unusually clean way -- decode's very first op is an
RMSNorm, which normalizes the input scale away, so only the input tensor's own
quantization range and the Pow/ReduceMean feeding that norm see it. Everything
downstream is scale-invariant. Adequate for composition and speed; NOT an int8
accuracy claim.

Three converter traps, all handled below:
  * raws must be FLOAT32 for every input regardless of the DLC's declared dtype
    (a uint8 mask is 1/4 the expected extent and is misreported as a batch-size
    mismatch)
  * perform_axes_to_spatial_first_order transposes the declared dims:
        [1,50,720]    -> [1,720,50]
        [1,50,163]    -> [1,163,50]
        [1,113,5,64]  -> [1,5,64,113]
  * keep_int64_inputs=False: int64 position_ids are declared Int_32

    python3 gen_decode_calibration.py --out expert_rewrite/decode_calib --n 8
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
    ap.add_argument("--prefill", default="smolvlm_expert_prefill.onnx")
    ap.add_argument("--vision", default="smolvlm_vision.onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=113)   # prefill / cache length
    ap.add_argument("--act", type=int, default=50)    # action chunk length
    ap.add_argument("--hidden", type=int, default=960)
    ap.add_argument("--ehidden", type=int, default=720)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1234)

    # --- prefill's vlm_embeds, grounded in real vision statistics -----------
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
    if mu.shape[0] != a.hidden:
        idx = np.linspace(0, mu.shape[0] - 1, a.hidden).astype(int)
        mu, sd = mu[idx], sd[idx]
        print(f"  resampled vision channels -> {a.hidden}")

    # --- run the real prefill to get a real KV cache ------------------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    print(f"  loading {a.prefill} ...")
    sp = ort.InferenceSession(a.prefill, so, providers=["CPUExecutionProvider"])
    onames = [o.name for o in sp.get_outputs()]
    kv_names = [n for n in onames if n.startswith(("present_key_", "present_value_"))]
    print(f"  prefill has {len(kv_names)} KV outputs")

    lines = []
    for k in range(a.n):
        emb = (rng.standard_normal((1, a.ctx, a.hidden)).astype(np.float32)
               * sd.astype(np.float32) + mu.astype(np.float32))
        feed = {
            "vlm_embeds": emb,
            "attention_mask": np.ones((1, a.ctx, a.ctx), dtype=bool),
            "position_ids": np.arange(a.ctx, dtype=np.int64).reshape(1, a.ctx),
        }
        outs = dict(zip(onames, sp.run(None, feed)))
        print(f"  [{k+1}/{a.n}] prefill ok")

        files = {}

        def w(name, arr):
            p = os.path.join(a.out, f"{name}_{k:03d}.raw")
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
            files[name] = p

        # DLC layout [1,720,50]; float32 despite RMSNorm making scale moot
        w("expert_embeds",
          rng.standard_normal((1, a.act, a.ehidden)).astype(np.float32).transpose(0, 2, 1))
        # DLC layout [1,163,50]; Bool_8 in the DLC but the raw must be float32
        w("attention_mask",
          np.ones((1, a.act, a.ctx + a.act), dtype=np.float32).transpose(0, 2, 1))
        # DLC declares Int_32; the raw is still float32
        w("position_ids",
          np.arange(a.ctx, a.ctx + a.act, dtype=np.float32).reshape(1, a.act))
        # real cache, ONNX [1,113,5,64] -> DLC [1,5,64,113]
        for i in range(len(kv_names) // 2):
            for tag in ("key", "value"):
                src = outs[f"present_{tag}_{i}"]
                w(f"past_{tag}_{i}", src.transpose(0, 2, 3, 1))

        lines.append(" ".join(f"{n}:={p}" for n, p in files.items()))

    lst = os.path.join(a.out, "input_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_in = len(lines[0].split())
    sz = sum(os.path.getsize(os.path.join(a.out, x)) for x in os.listdir(a.out)
             if x.endswith(".raw"))
    print(f"\n  {a.n} samples x {n_in} inputs -> {a.out}  ({sz/1e6:.1f} MB)")
    print(f"  list -> {lst}")
    print("  32/35 inputs are REAL prefill KV cache; expert_embeds and "
          "attention_mask are synthetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
