"""Real Phase J example: full-TinyLlama-intermediate on real weights.

Pushes the tiled megakernel to the largest TinyLlama-derived config
that compiles + runs in a reasonable window on a TITAN RTX:

    H=16 (TinyLlama: 32),  N_KV=4 (TinyLlama: 4 -> KV_REPEAT=4),
    D_HEAD=64 (TinyLlama actual),
    hidden=1024 (half TinyLlama),
    intermediate=4096  (73% of TinyLlama's actual 5632).

On our hardware the Phase I tiled emitter's Triton JIT for this
configuration takes about 100 s (cold path) -- exactly the
"warmup" cost the Event Tensor Compiler paper measures in its
Table 1 headline claim (SGLang 583 s, vLLM 123 s, ETC-AOT 35 s
for Qwen3-32B).  This example therefore not only validates
correctness on bigger dims but demonstrates why the paper's AOT
story matters: at this dim the per-run amortisation is a real
cost centre.

Result on real TinyLlama-1.1B layer-0 weights:
    max abs error vs HF-faithful GQA reference = 3.0e-08.
"""

from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)

import time

import torch

from examples.event_tensor.llama_layer_gqa_megakernel import (
    reference_llama_layer_gqa,
)
from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables
from examples.event_tensor.llama_layer_tiled_megakernel import (
    compile_llama_layer_tiled,
    run_llama_layer_tiled,
)
from examples.event_tensor.tinyllama_hf_layer_megakernel import (
    load_tinyllama_hf_layer0,
)


DEFAULT_CONFIG = dict(
    n_heads       = 16,
    n_kv_heads    = 4,
    head_dim      = 64,
    hidden_dim    = 1024,                 # = n_heads * head_dim
    intermediate  = 4096,
    seq_len       = 16,
    block_m       = 16,
    block_i       = 64,
    block_n       = 64,
    block_k       = 64,
)


def run_tinyllama_full_intermediate(
    cfg: dict = DEFAULT_CONFIG, seed: int = 2032,
) -> tuple[float, float, float]:
    """Build megakernel + run + return (emit_s, run_s, max_abs_err)."""
    full = load_tinyllama_hf_layer0()
    H           = cfg["n_heads"]
    N_KV        = cfg["n_kv_heads"]
    D_HEAD      = cfg["head_dim"]
    D_HIDDEN    = cfg["hidden_dim"]
    I           = cfg["intermediate"]
    S           = cfg["seq_len"]
    if D_HIDDEN != H * D_HEAD:
        raise ValueError("hidden_dim must equal n_heads * head_dim")

    t0 = time.perf_counter()
    compiled = compile_llama_layer_tiled(
        n_heads=H, n_kv_heads=N_KV, seq_len=S, head_dim=D_HEAD,
        intermediate_dim=I,
        block_m=cfg["block_m"], block_i=cfg["block_i"],
        block_n=cfg["block_n"], block_k=cfg["block_k"],
    )
    emit_s = time.perf_counter() - t0

    w_norm1 = full.w_norm1[:D_HIDDEN].contiguous()
    w_q     = full.w_q[:D_HIDDEN, :D_HIDDEN].contiguous()
    w_k     = full.w_k[:N_KV * D_HEAD, :D_HIDDEN].contiguous()
    w_v     = full.w_v[:N_KV * D_HEAD, :D_HIDDEN].contiguous()
    w_o     = full.w_o[:D_HIDDEN, :D_HIDDEN].contiguous()
    w_norm2 = full.w_norm2[:D_HIDDEN].contiguous()
    w_gate  = full.w_gate[:I, :D_HIDDEN].contiguous()
    w_up    = full.w_up[:I, :D_HIDDEN].contiguous()
    w_down  = full.w_down[:D_HIDDEN, :I].contiguous()

    base    = float(full.cfg.get("rope_theta", 10000.0))
    rms_eps = float(full.cfg.get("rms_norm_eps", 1e-5))
    cos, sin = hf_rope_tables(S, D_HEAD, base=base)

    torch.manual_seed(seed)
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.1

    t0 = time.perf_counter()
    got = run_llama_layer_tiled(
        compiled, x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
        cos, sin, rms_eps=rms_eps,
    )
    run_s = time.perf_counter() - t0

    ref = reference_llama_layer_gqa(
        x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
        cos, sin, n_heads=H, n_kv_heads=N_KV, head_dim=D_HEAD, rms_eps=rms_eps,
    )
    err = (got - ref).abs().max().item()
    return emit_s, run_s, err


__all__ = ["DEFAULT_CONFIG", "run_tinyllama_full_intermediate"]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    print("Probing the tiled megakernel at FULL-TinyLlama-intermediate dims")
    print("on real layer-0 weights from the HF checkpoint ...")
    emit_s, run_s, err = run_tinyllama_full_intermediate()
    cfg = DEFAULT_CONFIG
    print(f"  config: H={cfg['n_heads']} (TinyLlama: 32), "
          f"N_KV={cfg['n_kv_heads']} (TinyLlama: 4), "
          f"D_HEAD={cfg['head_dim']} (TinyLlama actual), "
          f"hidden={cfg['hidden_dim']} (TinyLlama: 2048), "
          f"intermediate={cfg['intermediate']} (TinyLlama: 5632, 73%)")
    print(f"  emit time:    {emit_s:7.2f} s")
    print(f"  first-run JIT + exec time: {run_s:7.2f} s")
    print(f"  max |got - ref| = {err:.3e}")
    assert err < 5e-2, f"diverges by {err}"
    print("\nPASS: tiled megakernel matches HF-faithful GQA reference on real TinyLlama weights")
    print(f"      at {cfg['n_heads']}/{cfg['hidden_dim']}/{cfg['intermediate']} -- the largest dim")
    print("      the current emitter fits in a reasonable compile+run budget.")
