"""Tests for the BenchmarkResult envelope and JSON round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.benchmarks.results import BenchmarkResult, compare_results, read_json


def _make_result(**overrides: object) -> BenchmarkResult:
    """Return a BenchmarkResult with sensible defaults, overridden by *overrides*."""
    defaults: dict[str, object] = {
        "suite": "torchbench",
        "workload": "resnet50",
        "capture_ok": True,
        "export_ok": True,
        "correctness_ok": True,
        "compile_time_s": 12.5,
        "latency_ms_p50": 3.2,
        "throughput": 312.5,
        "peak_memory_mb": 1024.0,
        "unsupported_ops": 0,
        "auto_translations_added": 2,
        "generated_kernels": 5,
        "generated_passes": 3,
        "generated_guards": 1,
        "promoted_artifacts": 4,
        "run_id": "run-001",
        "timestamp": "2025-06-01T00:00:00Z",
        "tags": ("nightly", "ci"),
        "source_commit": "abc1234",
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)  # type: ignore[arg-type]


# -- JSON round-trip -----------------------------------------------------------


def test_json_round_trip(tmp_path: Path) -> None:
    """write_json -> read_json should reconstruct an equal object."""
    original = _make_result()
    out = tmp_path / "result.json"
    original.write_json(out)
    restored = read_json(out)
    assert restored == original


def test_json_round_trip_nested_dir(tmp_path: Path) -> None:
    """write_json should create intermediate directories automatically."""
    original = _make_result()
    out = tmp_path / "a" / "b" / "result.json"
    original.write_json(out)
    assert out.exists()
    restored = read_json(out)
    assert restored == original


def test_all_fields_present_in_json(tmp_path: Path) -> None:
    """Every dataclass field must appear as a top-level JSON key."""
    result = _make_result()
    out = tmp_path / "result.json"
    result.write_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    expected_keys = {
        "suite",
        "workload",
        "capture_ok",
        "export_ok",
        "correctness_ok",
        "compile_time_s",
        "latency_ms_p50",
        "throughput",
        "peak_memory_mb",
        "unsupported_ops",
        "auto_translations_added",
        "generated_kernels",
        "generated_passes",
        "generated_guards",
        "promoted_artifacts",
        "run_id",
        "timestamp",
        "tags",
        "source_commit",
    }
    assert set(payload.keys()) == expected_keys


def test_tags_serialised_as_list(tmp_path: Path) -> None:
    """Tags tuple should be stored as a JSON array."""
    result = _make_result(tags=("a", "b", "c"))
    out = tmp_path / "result.json"
    result.write_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload["tags"], list)
    assert payload["tags"] == ["a", "b", "c"]


def test_empty_tags_round_trip(tmp_path: Path) -> None:
    """An empty tags tuple should round-trip correctly."""
    result = _make_result(tags=())
    out = tmp_path / "result.json"
    result.write_json(out)
    restored = read_json(out)
    assert restored.tags == ()


# -- default metadata values ---------------------------------------------------


def test_default_metadata_values() -> None:
    """Metadata fields should fall back to empty defaults."""
    result = BenchmarkResult(
        suite="s",
        workload="w",
        capture_ok=False,
        export_ok=False,
        correctness_ok=False,
        compile_time_s=0.0,
        latency_ms_p50=0.0,
        throughput=0.0,
        peak_memory_mb=0.0,
        unsupported_ops=0,
        auto_translations_added=0,
        generated_kernels=0,
        generated_passes=0,
        generated_guards=0,
        promoted_artifacts=0,
    )
    assert result.run_id == ""
    assert result.timestamp == ""
    assert result.tags == ()
    assert result.source_commit == ""


def test_default_metadata_round_trip(tmp_path: Path) -> None:
    """A result created with all defaults should survive JSON round-trip."""
    result = BenchmarkResult(
        suite="s",
        workload="w",
        capture_ok=False,
        export_ok=False,
        correctness_ok=False,
        compile_time_s=0.0,
        latency_ms_p50=0.0,
        throughput=0.0,
        peak_memory_mb=0.0,
        unsupported_ops=0,
        auto_translations_added=0,
        generated_kernels=0,
        generated_passes=0,
        generated_guards=0,
        promoted_artifacts=0,
    )
    out = tmp_path / "result.json"
    result.write_json(out)
    restored = read_json(out)
    assert restored == result


# -- compare_results -----------------------------------------------------------


def test_compare_results_deltas() -> None:
    """compare_results should report correct signed deltas."""
    baseline = _make_result(latency_ms_p50=10.0, throughput=100.0, peak_memory_mb=512.0)
    candidate = _make_result(latency_ms_p50=8.0, throughput=125.0, peak_memory_mb=480.0)
    deltas = compare_results(baseline, candidate)
    assert deltas["latency_delta"] == -2.0
    assert deltas["throughput_delta"] == 25.0
    assert deltas["memory_delta"] == -32.0


def test_compare_results_identical() -> None:
    """Comparing identical results should yield zero deltas."""
    result = _make_result()
    deltas = compare_results(result, result)
    assert deltas["latency_delta"] == 0.0
    assert deltas["throughput_delta"] == 0.0
    assert deltas["memory_delta"] == 0.0


def test_compare_results_regression() -> None:
    """Positive latency delta / negative throughput delta signal a regression."""
    baseline = _make_result(latency_ms_p50=5.0, throughput=200.0, peak_memory_mb=256.0)
    candidate = _make_result(latency_ms_p50=7.0, throughput=180.0, peak_memory_mb=300.0)
    deltas = compare_results(baseline, candidate)
    assert deltas["latency_delta"] == 2.0
    assert deltas["throughput_delta"] == -20.0
    assert deltas["memory_delta"] == 44.0


# -- frozen guarantee ---------------------------------------------------------


def test_frozen() -> None:
    """BenchmarkResult must be immutable."""
    result = _make_result()
    with pytest.raises(AttributeError):
        result.suite = "other"  # type: ignore[misc]
