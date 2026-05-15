"""Tests for the P4.1 headline benchmark runner.

The live workload runs need the ``[demo]`` + ``[benchmarks]`` extras
(transformers + matplotlib). These tests exercise the runner +
write_results + correctness gate against the
:class:`TimedSyntheticAdapter` so the full pipeline is testable
without GPU compute.
"""

from __future__ import annotations

import json

from compgen.benchmarks.headline import (
    ADAPTERS,
    AdapterMeasurement,
    TimedSyntheticAdapter,
    run_benchmark,
    write_results,
)


def _adapters():
    return {
        "torch_eager": TimedSyntheticAdapter(adapter_name="torch_eager", base_latency_us=100.0),
        "torch_compile": TimedSyntheticAdapter(adapter_name="torch_compile", base_latency_us=50.0),
        "compgen": TimedSyntheticAdapter(adapter_name="compgen", base_latency_us=45.0),
    }


def test_run_benchmark_produces_three_measurements_per_workload():
    results = run_benchmark(
        workloads=["tinyllama_1_1b", "smolvla_inference"],
        adapters=_adapters(),
        iters=20,
        warmup=2,
    )
    assert len(results) == 2
    for wr in results:
        assert set(wr.measurements.keys()) == set(_adapters().keys())


def test_measurement_stats_well_formed():
    a = TimedSyntheticAdapter(adapter_name="x", base_latency_us=10.0)
    m = a.measure("wl", iters=100, warmup=10, seed=0)
    stats = m.stats()
    assert stats["p50"] >= 10.0
    assert stats["p99"] >= stats["p50"]
    assert stats["mean"] > 0.0


def test_correctness_ok_when_hashes_match():
    """Synthetic adapters share a fixed output_hash per workload."""

    results = run_benchmark(
        workloads=["wl"],
        adapters=_adapters(),
        iters=5,
        warmup=1,
    )
    assert results[0].correctness_ok() is True


def test_correctness_fail_when_hashes_disagree():
    class _BadAdapter:
        adapter_name = "bad"

        def measure(self, workload_id: str, *, iters: int, warmup: int, seed: int):
            return AdapterMeasurement(
                adapter_name="bad",
                workload_id=workload_id,
                latencies_us=(10.0, 11.0),
                output_hash="DIFFERENT",
            )

    adapters = _adapters()
    adapters["bad"] = _BadAdapter()  # type: ignore[assignment]
    results = run_benchmark(workloads=["wl"], adapters=adapters, iters=5, warmup=1)
    assert results[0].correctness_ok() is False


def test_adapter_exception_is_typed_blocked():
    """An adapter that raises gets recorded as blocked, not silently
    dropped — honest failure surfaces in the evidence pack."""

    class _Crasher:
        adapter_name = "crash"

        def measure(self, *a, **k):
            raise RuntimeError("intentional")

    results = run_benchmark(
        workloads=["wl"],
        adapters={"torch_eager": _adapters()["torch_eager"], "crash": _Crasher()},  # type: ignore[arg-type]
        iters=5,
        warmup=1,
    )
    crash_m = results[0].measurements["crash"]
    assert crash_m.blocked is True
    assert "intentional" in crash_m.blocked_reason


def test_write_results_emits_csv_and_summary(tmp_path):
    results = run_benchmark(
        workloads=["wl_a"], adapters=_adapters(), iters=10, warmup=1
    )
    summary_path = write_results(results, tmp_path)
    assert summary_path.is_file()
    body = json.loads(summary_path.read_text(encoding="utf-8"))
    assert body["schema_version"] == "headline_benchmark_summary_v1"
    # Per-adapter latency CSV exists.
    csv = tmp_path / "wl_a" / "torch_eager" / "latency_us.csv"
    assert csv.is_file()
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "iter,latency_us"
    assert len(lines) == 11  # header + 10 iters


def test_byte_deterministic_across_reruns():
    a1 = run_benchmark(
        workloads=["wl"], adapters=_adapters(), iters=10, warmup=1, seed=42
    )
    a2 = run_benchmark(
        workloads=["wl"], adapters=_adapters(), iters=10, warmup=1, seed=42
    )
    assert a1[0].measurements["compgen"].latencies_us == a2[0].measurements["compgen"].latencies_us


def test_adapters_constant():
    assert ADAPTERS == ("torch_eager", "torch_compile", "compgen")
