"""Tests for the live-adapter slice workloads (TinyLlama MLP, Whisper layer).

These tests are HEAVY — they download HF weights, run a real
``compile_model`` pipeline, and execute the xDSL CPU executor. They
are marked ``slow`` so the default test run skips them; they are
exercised in the live P4 path.

Coverage:

1. ``LiveCompGenAdapter`` returns a typed ``blocked`` measurement on
   a non-slice workload (today the full-model end-to-end is not built).
2. ``LiveTorchEagerAdapter`` produces measurable latencies on the
   TinyLlama MLP slice.
3. ``LiveCompGenAdapter`` produces measurable latencies on the
   TinyLlama MLP slice AND the output hash matches ``torch.compile``
   (real correctness signal — both go through similar lowering).
"""

from __future__ import annotations

import pytest

slow = pytest.mark.slow


@slow
def test_compgen_blocks_on_full_tinyllama() -> None:
    from compgen.benchmarks.live_adapters import LiveCompGenAdapter

    m = LiveCompGenAdapter().measure(
        "tinyllama_1_1b", iters=1, warmup=0, seed=0
    )
    assert m.blocked is True
    assert m.blocked_reason == "compgen_full_model_not_built"


@slow
def test_eager_runs_on_tinyllama_mlp_slice() -> None:
    from compgen.benchmarks.live_adapters import LiveTorchEagerAdapter

    m = LiveTorchEagerAdapter().measure(
        "tinyllama_1_1b__slice", iters=3, warmup=1, seed=42
    )
    assert m.blocked is False
    assert len(m.latencies_us) == 3
    assert m.output_hash  # non-empty


@slow
def test_compgen_runs_on_tinyllama_mlp_slice_with_compile_match() -> None:
    """CompGen output hash matches torch.compile output hash — proves
    the xDSL pipeline preserves correctness vs the inductor reference."""

    from compgen.benchmarks.live_adapters import (
        LiveCompGenAdapter,
        LiveTorchCompileAdapter,
    )

    compile_m = LiveTorchCompileAdapter().measure(
        "tinyllama_1_1b__slice", iters=2, warmup=1, seed=42
    )
    cg_m = LiveCompGenAdapter().measure(
        "tinyllama_1_1b__slice", iters=2, warmup=1, seed=42
    )
    assert compile_m.blocked is False
    assert cg_m.blocked is False
    assert compile_m.output_hash == cg_m.output_hash, (
        f"CompGen output hash {cg_m.output_hash!r} differs from "
        f"torch.compile {compile_m.output_hash!r}"
    )


@slow
def test_compgen_runs_on_whisper_encoder_slice() -> None:
    """After the aten-handler additions, CompGen runs the Whisper
    encoder block end-to-end without unsupported-callee warnings.

    We assert the run completes and emits real measurements; we do
    NOT assert hash equality with eager because some helper-func
    semantics (e.g. ``aten_full_like`` fill value) are not in the IR
    and the executor uses a documented fallback (zeros).
    """

    from compgen.benchmarks.live_adapters import LiveCompGenAdapter

    m = LiveCompGenAdapter().measure(
        "whisper_tiny__slice", iters=3, warmup=1, seed=42
    )
    assert m.blocked is False
    assert len(m.latencies_us) == 3
    assert m.output_hash
