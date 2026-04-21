"""Tests for ``compgen.bench.diagnosis`` + ``refinement`` + ``iterate``.

Uses mocked codegen / bench callables so these run fast and CPU-only.
The real GPU refinement is a separate smoke (``test_turing_refinement_demo``).
"""

from __future__ import annotations

import pytest

from compgen.bench.diagnosis import (
    Bottleneck,
    KernelDiagnosis,
    diagnose,
    format_diagnosis,
)
from compgen.bench.iterate import (
    IterationAttempt,
    IterationOutcome,
    iterate_kernel,
)
from compgen.bench.kernel_bench import BenchResult
from compgen.bench.refinement import build_refinement_prompt
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    NumericsSpec,
    OrchestrationSpec,
    ShapeClass,
    StaticAttr,
    TensorIO,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matmul_contract() -> KernelContractV3:
    hw = HardwareEnvelope(
        target_name="test-sm75",
        vector_lanes=72,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16", "f32"),
        peak_bandwidth_gbps=672.0,
    )
    lhs = TensorIO(name="lhs", shape=ShapeClass(dims=(None, None)),
                   dtype_class=("f16",))
    rhs = TensorIO(name="rhs", shape=ShapeClass(dims=(None, None)),
                   dtype_class=("f16",))
    out = TensorIO(name="out", shape=ShapeClass(dims=(None, None)),
                   dtype_class=("f16",))
    return KernelContractV3(
        op_name="linalg.matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(inputs=(lhs, rhs), outputs=(out,),
                      numerics=NumericsSpec(accumulator_dtype="f32")),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=hw),
        ),
    )


def _softmax_contract() -> KernelContractV3:
    hw = HardwareEnvelope(
        target_name="test-sm75",
        vector_lanes=72, scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f32",), peak_bandwidth_gbps=672.0,
    )
    inp = TensorIO(name="x", shape=ShapeClass(dims=(None, None)),
                   dtype_class=("f32",))
    out = TensorIO(name="y", shape=ShapeClass(dims=(None, None)),
                   dtype_class=("f32",))
    return KernelContractV3(
        op_name="softmax",
        archetype=KernelArchetype.REDUCE,
        io=IOContract(
            inputs=(inp,), outputs=(out,),
            attributes=(StaticAttr(name="axis", value=-1),),
        ),
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=hw),
        ),
    )


def _bench(
    name: str, our_us: float, eager_us: float,
    passed: bool = True, max_abs_err: float = 0.0,
    input_shapes: list[list[int]] | None = None,
) -> BenchResult:
    return BenchResult(
        name=name, device="test",
        dtype_in="f16", dtype_out="f16",
        max_abs_err=max_abs_err, max_rel_err=max_abs_err,
        passed=passed,
        our_us=our_us, eager_us=eager_us, torch_compile_us=None,
        us_ratio_vs_eager=(our_us / eager_us) if eager_us > 0 else 0.0,
        us_ratio_vs_torch_compile=None,
        warmup_iters=10, timed_iters=100,
        input_shapes=input_shapes or [[512, 1024], [1024, 512]],
    )


# ---------------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------------


def test_correctness_failure_dominates_diagnosis() -> None:
    """When correctness fails, diagnosis is correctness-bound, no perf hypos."""
    d = diagnose(_matmul_contract(),
                 _bench("m1", our_us=100, eager_us=40, passed=False, max_abs_err=0.5))
    assert d.primary_bottleneck is Bottleneck.CORRECTNESS_BOUND
    assert "numerically wrong" in d.hypotheses[0]
    assert "0.5" in d.hypotheses[0] or "5.00e-01" in d.hypotheses[0]


def test_compute_tiled_slow_kernel_suggests_autotune() -> None:
    """matmul at 194μs vs 37μs eager → 5x gap → should suggest autotune."""
    d = diagnose(_matmul_contract(),
                 _bench("m2", our_us=194, eager_us=37))
    joined = " ".join(d.hypotheses).lower()
    assert "autotune" in joined or "tl.dot" in joined
    # Should be either compute-bound (low eff) or bandwidth-bound; not correctness.
    assert d.primary_bottleneck is not Bottleneck.CORRECTNESS_BOUND


def test_reduce_archetype_gets_reduce_specific_hypotheses() -> None:
    """softmax slow → hypotheses about row batching or BLOCK_N power-of-2."""
    d = diagnose(_softmax_contract(),
                 _bench("s1", our_us=120, eager_us=40,
                        input_shapes=[[2048, 1024]]))
    joined = " ".join(d.hypotheses).lower()
    assert any(tok in joined for tok in ("reduction", "row", "block_n", "num_warps"))


def test_diagnosis_reports_roofline_efficiency_in_0_1() -> None:
    d = diagnose(_matmul_contract(), _bench("m3", our_us=100, eager_us=40))
    assert 0.0 <= d.roofline_efficiency <= 1.0


def test_format_diagnosis_is_human_readable() -> None:
    d = diagnose(_matmul_contract(), _bench("m4", our_us=100, eager_us=40))
    text = format_diagnosis(d)
    assert "DIAGNOSIS" in text
    assert "bottleneck" in text
    assert "hypotheses" in text


# ---------------------------------------------------------------------------
# build_refinement_prompt()
# ---------------------------------------------------------------------------


def test_refinement_prompt_contains_prior_source_and_metrics() -> None:
    contract = _matmul_contract()
    bench = _bench("m5", our_us=194, eager_us=37)
    d = diagnose(contract, bench)
    prior_src = "def matmul_kernel(...): pass"

    prompt = build_refinement_prompt(contract, prior_src, d, perf_target_us=60)

    assert "matmul_kernel" in prompt
    assert "our_us" in prompt
    assert "vs_eager" in prompt
    assert "top hypotheses" in prompt.lower()
    assert "contract reminder" in prompt.lower()
    assert "60" in prompt        # perf target surfaced


def test_refinement_prompt_trims_oversized_prior_source() -> None:
    huge = "x" * 50_000
    contract = _matmul_contract()
    d = diagnose(contract, _bench("m6", our_us=100, eager_us=40))
    prompt = build_refinement_prompt(contract, huge, d)

    assert "[trimmed for prompt size]" in prompt
    assert len(prompt) < 15_000


# ---------------------------------------------------------------------------
# iterate_kernel — converges when codegen improves; escalates when it doesn't
# ---------------------------------------------------------------------------


def test_iterate_kernel_converges_when_attempt_2_beats_target() -> None:
    """Mock codegen: attempt 1 is 194μs (misses target), attempt 2 is 50μs (hits)."""
    contract = _matmul_contract()
    calls: list[str | None] = []
    sources = [
        "# slow matmul\ndef matmul_kernel(...): # BLOCK_M=64, BLOCK_N=64 ...",
        "# refined w/ 128 tiles\ndef matmul_kernel(...): # BLOCK_M=128 ...",
    ]
    benches = [
        _bench("slow", our_us=194, eager_us=37),
        _bench("fast", our_us=50, eager_us=37),
    ]

    def codegen(_c, prev_src, prev_diag, refinement_prompt):
        calls.append(refinement_prompt)
        return sources.pop(0)

    def bench_fn(_src, _c):
        return benches.pop(0)

    outcome = iterate_kernel(
        contract, codegen, bench_fn,
        perf_target_us=60, max_attempts=3,
    )

    assert outcome.converged
    assert len(outcome.attempts) == 2
    assert outcome.best_attempt_idx == 1
    assert calls[0] is None                  # first attempt = cold prompt
    assert calls[1] is not None              # second attempt = refinement prompt
    assert "previous attempt" in calls[1].lower()


def test_iterate_kernel_escalates_when_nothing_closes_gap() -> None:
    """All 3 attempts stay 5× slower → outcome flags escalate_to_autocomp."""
    contract = _matmul_contract()

    def codegen(_c, _ps, _pd, _rp):
        return "# another slow kernel\n"

    def bench_fn(_src, _c):
        return _bench("still_slow", our_us=190, eager_us=37)

    outcome = iterate_kernel(
        contract, codegen, bench_fn,
        perf_target_us=60, max_attempts=3,
    )

    assert not outcome.converged
    assert len(outcome.attempts) == 3
    # Best is still 190μs/37μs ≈ 5× over eager → escalate.
    assert outcome.escalate_to_autocomp


def test_iterate_kernel_returns_best_passing_attempt() -> None:
    """When no perf target: loop runs all attempts; best-so-far (lowest us
    among correctness-passing attempts) is reported."""
    contract = _matmul_contract()
    benches = [
        _bench("fail_but_fast", our_us=30, eager_us=40, passed=False, max_abs_err=0.5),
        _bench("pass_medium",   our_us=80, eager_us=40, passed=True),
        _bench("pass_slower",   our_us=95, eager_us=40, passed=True),
    ]
    srcs = ["k1", "k2", "k3"]

    def codegen(_c, _ps, _pd, _rp):
        return srcs.pop(0)
    def bench_fn(_src, _c):
        return benches.pop(0)

    outcome = iterate_kernel(
        contract, codegen, bench_fn,
        perf_target_us=None, max_attempts=3,
    )
    assert len(outcome.attempts) == 3
    # Best = k2 (fastest that PASSED correctness).
    assert outcome.best_attempt_idx == 1
    assert outcome.best.kernel_source == "k2"


def test_iterate_kernel_summary_is_readable() -> None:
    contract = _matmul_contract()

    def codegen(_c, _ps, _pd, _rp):
        return "src"
    def bench_fn(_src, _c):
        return _bench("x", our_us=100, eager_us=40)

    outcome = iterate_kernel(contract, codegen, bench_fn, max_attempts=1)
    text = outcome.summary()
    assert "linalg.matmul" in text
    assert "100.0" in text or "100" in text
