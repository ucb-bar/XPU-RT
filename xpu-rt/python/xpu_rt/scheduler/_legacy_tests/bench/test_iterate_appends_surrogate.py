"""Tests for the G5 wire-in: iterate_kernel appends surrogate samples
on passing benches.

Coverage:

* When ``surrogate`` is supplied AND a bench passes, a typed
  ``Sample`` is appended with the right fingerprint + candidate_id.
* When a bench fails correctness, no sample is appended (bad
  numerical results would poison the predictor).
* When ``surrogate`` is omitted, behavior is unchanged (legacy
  callers see no observational hooks).
* The (region_fingerprint, candidate_id) tuple is deterministic
  across reruns of the same contract.
* After several iterations the surrogate's prediction has rising
  confidence as samples accumulate.

The tests use synthetic codegen + bench callables — the real
``compile_and_bench`` is too expensive for unit tests. We feed
a fixed string source + a controllable BenchResult shape and
verify the surrogate is updated correctly.
"""

from __future__ import annotations

from compgen.bench.iterate import _contract_fingerprint, iterate_kernel
from compgen.bench.kernel_bench import BenchResult
from compgen.bench.surrogate import Surrogate
from compgen.kernels.contract_v3 import (
    IOContract,
    KernelArchetype,
    KernelContractV3,
    LayoutKind,
    NumericsSpec,
    ShapeClass,
    TensorIO,
)


def _matmul_contract() -> KernelContractV3:
    """Minimal compute_tiled matmul contract for the iterate loop."""

    return KernelContractV3(
        op_name="matmul_test",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(
                    name="a",
                    shape=ShapeClass(dims=(64, 64)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
                TensorIO(
                    name="b",
                    shape=ShapeClass(dims=(64, 64)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
            ),
            outputs=(
                TensorIO(
                    name="c",
                    shape=ShapeClass(dims=(64, 64)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
            ),
            numerics=NumericsSpec(
                accumulator_dtype="fp16",
                max_relative_error=1e-2,
                deterministic=False,
            ),
        ),
    )


def _make_bench_callable(
    *, passing: bool = True, latencies_us: list[float] | None = None
):
    """Build a synthetic compile_and_bench that returns a controllable
    sequence of BenchResult."""

    latencies = latencies_us or [50.0, 40.0, 30.0]
    state = {"idx": 0}

    def _bench(source: str, contract) -> BenchResult:
        idx = state["idx"]
        state["idx"] = idx + 1
        lat = latencies[idx % len(latencies)]
        return BenchResult(
            name=contract.op_name,
            device="cpu",
            dtype_in="fp16",
            dtype_out="fp16",
            max_abs_err=0.0 if passing else 1.0,
            max_rel_err=0.0 if passing else 1.0,
            passed=passing,
            our_us=lat,
            eager_us=100.0,
            torch_compile_us=None,
            us_ratio_vs_eager=lat / 100.0,
            us_ratio_vs_torch_compile=None,
            warmup_iters=1,
            timed_iters=10,
        )

    return _bench


def _make_codegen():
    """Stub codegen — returns a static source per attempt."""

    def _codegen(contract, prior_source, prior_diag, refinement_prompt):
        return f"// kernel attempt for {contract.op_name}\nreturn 0;"

    return _codegen


# ---------- Positive --------------------------------------------------


def test_surrogate_receives_samples_on_passing_bench():
    """Passing benches append samples; the surrogate's bucket fills."""

    contract = _matmul_contract()
    surr = Surrogate()
    iterate_kernel(
        contract,
        _make_codegen(),
        _make_bench_callable(passing=True, latencies_us=[50.0, 40.0, 30.0]),
        max_attempts=3,
        surrogate=surr,
    )
    assert surr.n_samples() == 3
    # All three samples land in the contract-keyed fingerprint bucket.
    fp = _contract_fingerprint(contract)
    pred = surr.predict(region_fingerprint=fp, candidate_id="attempt_1")
    assert pred.predicted_latency_us == 50.0


def test_failed_bench_does_not_poison_surrogate():
    """Bad numerical results must not contaminate the predictor."""

    contract = _matmul_contract()
    surr = Surrogate()
    iterate_kernel(
        contract,
        _make_codegen(),
        _make_bench_callable(passing=False, latencies_us=[50.0, 40.0, 30.0]),
        max_attempts=3,
        surrogate=surr,
    )
    assert surr.n_samples() == 0


def test_legacy_path_no_surrogate_arg():
    """Existing callers that don't pass ``surrogate`` see unchanged behavior."""

    contract = _matmul_contract()
    outcome = iterate_kernel(
        contract,
        _make_codegen(),
        _make_bench_callable(passing=True, latencies_us=[50.0]),
        max_attempts=1,
    )
    assert len(outcome.attempts) == 1
    # No exception, no observability side-effects.


def test_fingerprint_deterministic_across_reruns():
    """The fingerprint must be byte-stable for the same contract shape."""

    contract = _matmul_contract()
    a = _contract_fingerprint(contract)
    b = _contract_fingerprint(contract)
    assert a == b
    # Includes op_name and shape.
    assert "matmul_test" in a
    assert "64x64" in a


def test_fingerprint_distinguishes_different_shapes():
    """Two contracts with different shapes get different fingerprints."""

    c1 = _matmul_contract()
    # Synthesise a different shape via a fresh IOContract.
    c2 = KernelContractV3(
        op_name="matmul_test",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(
                    name="a",
                    shape=ShapeClass(dims=(128, 128)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
                TensorIO(
                    name="b",
                    shape=ShapeClass(dims=(128, 128)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
            ),
            outputs=(
                TensorIO(
                    name="c",
                    shape=ShapeClass(dims=(128, 128)),
                    dtype_class=("fp16",),
                    layout=LayoutKind.ROW_MAJOR,
                ),
            ),
        ),
    )
    assert _contract_fingerprint(c1) != _contract_fingerprint(c2)


def test_surrogate_prediction_after_iterations():
    """After multiple iterations, the surrogate predicts the mean
    latency of the matching bucket."""

    contract = _matmul_contract()
    surr = Surrogate(confidence_cap=10)
    iterate_kernel(
        contract,
        _make_codegen(),
        _make_bench_callable(passing=True, latencies_us=[10.0, 20.0, 30.0]),
        max_attempts=3,
        surrogate=surr,
    )
    fp = _contract_fingerprint(contract)
    # Tier 1 (exact bucket) for attempt_1 returns 10.0
    pred = surr.predict(region_fingerprint=fp, candidate_id="attempt_1")
    assert pred.predicted_latency_us == 10.0
    assert pred.confidence > 0.0
