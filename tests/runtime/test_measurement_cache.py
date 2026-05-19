"""Tests for the append-only on-board measurement cache."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xpu_rt.runtime.measurement_cache import (
    CacheEntry,
    CacheKey,
    MeasuredStats,
    append_entry,
    load_cache,
)
from xpu_rt.scheduling.measurement_driven import (
    CandidateSchedule,
    evaluate_with_cache,
    resolve_candidate,
)


def _stats(p50: float, *, n: int = 100, source: str = "test.csv") -> MeasuredStats:
    return MeasuredStats(
        mean_us=p50,
        p50_us=p50,
        p99_us=p50 * 1.1,
        stdev_us=p50 * 0.01,
        n_iters=n,
        deadline_met_rate=1.0,
        captured_at=datetime.now(UTC).isoformat(),
        source=source,
    )


def test_cache_key_canonicalises_techniques() -> None:
    k1 = CacheKey.make("qrb5165", "yolov8n", "DSP", ["b", "a", "a", "c"])
    k2 = CacheKey.make("qrb5165", "yolov8n", "DSP", ("a", "b", "c"))
    assert k1.deployment_techniques == ("a", "b", "c")
    assert k1 == k2


def test_append_load_round_trip(tmp_path: Path) -> None:
    entries = [
        CacheEntry(
            key=CacheKey.make("qrb5165", "w1", "DSP", ["a"], period_us=0),
            stats=_stats(100.0),
        ),
        CacheEntry(
            key=CacheKey.make("qrb5165", "w2", "GPU", ["a"], period_us=3333),
            stats=_stats(2000.0),
        ),
        CacheEntry(
            key=CacheKey.make("qrb5165", "w1", "CPU", ["a", "b"]),
            stats=_stats(150.0),
        ),
    ]
    for e in entries:
        append_entry(tmp_path, "qrb5165", e)
    cache = load_cache(tmp_path, "qrb5165")
    assert len(cache.entries) == 3
    found = cache.get(entries[1].key)
    assert found is not None
    assert found.stats.p50_us == 2000.0


def test_get_best_returns_lowest_p50(tmp_path: Path) -> None:
    k1 = CacheKey.make("qrb5165", "yolov8n", "DSP", ["a"])
    k2 = CacheKey.make("qrb5165", "yolov8n", "DSP", ["b"])
    append_entry(tmp_path, "qrb5165", CacheEntry(k1, _stats(100.0)))
    append_entry(tmp_path, "qrb5165", CacheEntry(k2, _stats(80.0)))
    cache = load_cache(tmp_path, "qrb5165")
    best = cache.get_best("qrb5165", "yolov8n", "DSP")
    assert best is not None
    assert best.stats.p50_us == 80.0


def test_get_best_prefers_techniques_filter(tmp_path: Path) -> None:
    k_bare = CacheKey.make("qrb5165", "yolov8n", "DSP", [])
    k_warm = CacheKey.make("qrb5165", "yolov8n", "DSP", ["a", "b"])
    # Bare has lower p50 but lacks the required techniques.
    append_entry(tmp_path, "qrb5165", CacheEntry(k_bare, _stats(40.0)))
    append_entry(tmp_path, "qrb5165", CacheEntry(k_warm, _stats(100.0)))
    cache = load_cache(tmp_path, "qrb5165")
    best = cache.get_best(
        "qrb5165", "yolov8n", "DSP", prefer_techniques=("a", "b"),
    )
    assert best is not None
    assert best.stats.p50_us == 100.0


def test_resolve_candidate_partial_miss_returns_not_fully_measured(
    tmp_path: Path,
) -> None:
    k = CacheKey.make("qrb5165", "yolov8n", "DSP", ["a"])
    append_entry(tmp_path, "qrb5165", CacheEntry(k, _stats(55000.0)))
    cache = load_cache(tmp_path, "qrb5165")
    candidate = CandidateSchedule(
        target_id="qrb5165",
        placements=(
            ("yolov8n", "DSP", 0, ("a",)),
            ("dronet", "GPU", 0, ("a",)),
        ),
    )
    res = resolve_candidate(candidate, cache)
    assert res.fully_measured is False
    assert res.aggregate_makespan_us is None
    assert res.per_placement_p50_us[0] == 55000.0
    assert res.per_placement_p50_us[1] is None


def test_evaluate_with_cache_full_hit_aggregates_max(tmp_path: Path) -> None:
    k_yolo = CacheKey.make("qrb5165", "yolov8n", "DSP", ["a"])
    k_drone = CacheKey.make("qrb5165", "dronet", "GPU", ["a"])
    append_entry(tmp_path, "qrb5165", CacheEntry(k_yolo, _stats(55000.0)))
    append_entry(tmp_path, "qrb5165", CacheEntry(k_drone, _stats(1700.0)))
    cache = load_cache(tmp_path, "qrb5165")
    candidate = CandidateSchedule(
        target_id="qrb5165",
        placements=(
            ("yolov8n", "DSP", 0, ("a",)),
            ("dronet", "GPU", 0, ("a",)),
        ),
    )
    eval_ = evaluate_with_cache(candidate, cache)
    assert eval_["source"] == "measurement_cache"
    assert eval_["makespan_us"] == pytest.approx(55000.0)
    assert eval_["deadlines_met"] is True
    assert eval_["missing"] == []


def test_evaluate_with_cache_full_miss_uses_fallback(tmp_path: Path) -> None:
    cache = load_cache(tmp_path, "qrb5165")
    candidate = CandidateSchedule(
        target_id="qrb5165",
        placements=(("yolov8n", "DSP", 0, ("a",)),),
    )
    eval_ = evaluate_with_cache(
        candidate, cache, predicted_makespan_fallback_us=12345.0,
    )
    assert eval_["source"] == "predicted"
    assert eval_["makespan_us"] == pytest.approx(12345.0)
    assert eval_["missing"] == [("yolov8n", "DSP", ("a",))]
