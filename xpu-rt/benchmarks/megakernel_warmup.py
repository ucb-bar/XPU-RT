"""Warmup-cost benchmark: CompGen AOT megakernel vs PyTorch JIT baselines.

Mirrors the structure of Table 1 of the Event Tensor Compiler paper
(``vLLM JIT 123 s`` / ``SGLang JIT 583 s`` / ``ETC AOT 35 s`` for
Qwen3-32B).  We can't reproduce those magnitudes on a TITAN RTX with a
sliced TinyLlama block, but the *shape* of the comparison is
reproducible and is what matters as a demonstration of the AOT model:

    cold AOT path  (paper's "ETC")
        = (emit Triton source) + (Triton compile) + (one launch)
        run once during build; subsequent inferences pay only the
        Triton kernel-cache hit + launch.

    JIT path (paper's vLLM/SGLang baselines, modelled by torch.compile)
        = (trace + lower + compile) on first call to the compiled fn
        every cold-start engine launch repeats this work.

The benchmark loads the same TinyLlama-1.1B layer-0 weight slice used
by ``tinyllama_layer_megakernel.py`` so the workload is real, then:

    1. Times the megakernel cold path (emit + compile + first launch)
       starting from an empty Triton cache.
    2. Times a warm reload that mimics the AOT model: re-import the
       previously emitted source, hit the Triton cache, launch.
    3. Times ``torch.compile`` cold compile + first call as the JIT
       baseline on the equivalent PyTorch eager block.
    4. Reports both wall-clock numbers and the AOT-vs-JIT speedup.

Run as::

    python -m benchmarks.megakernel_warmup
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from examples.event_tensor.tinyllama_layer_megakernel import (
    DEFAULT_SEQ_LEN,
    compile_for_tinyllama,
    load_tinyllama_layer0,
    project_qkv,
    slice_weights_for_megakernel,
)
from examples.event_tensor.transformer_block_megakernel import (
    reference_block,
    run_transformer_block_megakernel,
)


@dataclass
class WarmupResult:
    label: str
    cold_seconds: float
    warm_seconds: float
    description: str


def _purge_triton_cache() -> None:
    """Wipe ~/.triton/cache so the next compile is genuinely cold."""
    cache = Path(os.path.expanduser("~/.triton/cache"))
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)


def _now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def measure_megakernel_aot(
    weights,
    sliced_cfg,
    x,
    *,
    label: str = "megakernel_aot",
) -> WarmupResult:
    """Cold = (emit source + Triton compile + first launch)
    Warm = (re-import previously emitted source + Triton cache hit + launch)"""
    _purge_triton_cache()
    gc.collect()
    torch.cuda.empty_cache()

    t0 = _now()
    compiled_cold = compile_for_tinyllama(seq_len=DEFAULT_SEQ_LEN)
    q, k, v = project_qkv(x, weights, sliced_cfg)
    _ = run_transformer_block_megakernel(
        compiled_cold,
        q,
        k,
        v,
        x,
        weights.w_gate,
        weights.w_up,
        weights.w_down,
    )
    cold = _now() - t0

    # Warm path: don't purge Triton cache; recompile the emitter (cheap)
    # and run again -- Triton sees a cache hit on the kernel hash.
    gc.collect()
    torch.cuda.empty_cache()
    t0 = _now()
    compiled_warm = compile_for_tinyllama(seq_len=DEFAULT_SEQ_LEN)
    q, k, v = project_qkv(x, weights, sliced_cfg)
    _ = run_transformer_block_megakernel(
        compiled_warm,
        q,
        k,
        v,
        x,
        weights.w_gate,
        weights.w_up,
        weights.w_down,
    )
    warm = _now() - t0

    return WarmupResult(
        label=label,
        cold_seconds=cold,
        warm_seconds=warm,
        description=(
            "Cold = emit Triton source + compile + first launch (no cache). Warm = re-emit + Triton cache hit + launch."
        ),
    )


def measure_torch_compile_jit(
    weights,
    sliced_cfg,
    x,
    *,
    label: str = "torch.compile_jit",
) -> WarmupResult:
    """Cold = first call into torch.compile'd block (compiles on first call).
    Warm = subsequent call (cached)."""

    def block(x_in, q, k, v, wg, wu, wd):
        return reference_block(q, k, v, x_in, wg, wu, wd)

    gc.collect()
    torch.cuda.empty_cache()
    q, k, v = project_qkv(x, weights, sliced_cfg)

    # Reset torch's compile cache for an apples-to-apples cold start.
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    compiled_block = torch.compile(block, mode="reduce-overhead", dynamic=False)

    t0 = _now()
    _ = compiled_block(x, q, k, v, weights.w_gate, weights.w_up, weights.w_down)
    cold = _now() - t0

    t0 = _now()
    _ = compiled_block(x, q, k, v, weights.w_gate, weights.w_up, weights.w_down)
    warm = _now() - t0

    return WarmupResult(
        label=label,
        cold_seconds=cold,
        warm_seconds=warm,
        description=(
            "Cold = first call (Dynamo trace + Inductor lower + Triton compile). "
            "Warm = second call (Dynamo cache hit + Inductor cache hit)."
        ),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    print("Loading TinyLlama-1.1B layer-0 weights ...")
    full = load_tinyllama_layer0()
    sliced, sliced_cfg = slice_weights_for_megakernel(full)
    print(
        f"  workload: H={sliced_cfg['n_heads']}, "
        f"D={sliced_cfg['hidden_dim']}, I={sliced_cfg['intermediate']}, "
        f"S={DEFAULT_SEQ_LEN}"
    )

    torch.manual_seed(123)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )

    print("\n[1/2] Measuring CompGen megakernel AOT path ...")
    aot = measure_megakernel_aot(sliced, sliced_cfg, x)
    print(f"  cold: {aot.cold_seconds:7.3f} s")
    print(f"  warm: {aot.warm_seconds:7.3f} s")

    print("\n[2/2] Measuring torch.compile JIT baseline ...")
    jit = measure_torch_compile_jit(sliced, sliced_cfg, x)
    print(f"  cold: {jit.cold_seconds:7.3f} s")
    print(f"  warm: {jit.warm_seconds:7.3f} s")

    print("\n=== summary ===")
    print(f"{'path':28}  {'cold (s)':>10}  {'warm (s)':>10}")
    print(f"{aot.label:28}  {aot.cold_seconds:10.3f}  {aot.warm_seconds:10.3f}")
    print(f"{jit.label:28}  {jit.cold_seconds:10.3f}  {jit.warm_seconds:10.3f}")
    if aot.cold_seconds > 0:
        speedup = jit.cold_seconds / aot.cold_seconds
        print(f"\nCompGen-AOT cold-start speedup vs torch.compile JIT: {speedup:.2f}x")
        if speedup >= 1.0:
            print("AOT wins on cold-start.")
        else:
            print(
                "AOT lost on cold-start (likely because the megakernel JITs through "
                "Triton just like torch.compile does on this small workload). "
                "Wall-clock parity is expected here; the AOT advantage compounds with "
                "graph size and is the bedrock of the paper's headline 35 s vs 583 s."
            )


if __name__ == "__main__":
    main()
