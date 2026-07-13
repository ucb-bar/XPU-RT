#!/usr/bin/env python3
"""
Self-contained tests for granularity_advisor.py -- plain asserts, no pytest
dependency (this repo doesn't have a test framework set up yet). Run with:

    python3 xpu-rt/tests/test_granularity_advisor.py

Fixtures here are small and synthetic, hand-built to mirror the real
schedule JSON schema (see schedules/*.json), rather than the real
multi-hundred-dispatch files -- those are untracked local run outputs, not
committed to the repo.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from granularity_advisor import (
    DispatchRecord,
    analyze_granularity,
    from_schedule_json,
    group_by_periodicity,
)


def _periodic_records(base: str, period: float, n_instances: int, dispatch_duration: float, jitter=0.0):
    """n_instances periodic instances of `base`, each a single dispatch of
    `dispatch_duration`, spaced `period` apart (with optional per-instance
    start-time jitter, to exercise the median-based period inference)."""
    records = []
    for i in range(n_instances):
        start = i * period + jitter[i] if jitter else i * period
        records.append(DispatchRecord(
            instance_id=f"{base}{i}",
            base_id=base,
            is_periodic=True,
            start_time=start,
            duration=dispatch_duration,
        ))
    return records


def _non_periodic_records(base: str, durations: list[float]):
    return [
        DispatchRecord(
            instance_id=base,
            base_id=base,
            is_periodic=False,
            start_time=float(i * 10),
            duration=d,
        )
        for i, d in enumerate(durations)
    ]


def test_period_inferred_from_instance_spacing():
    records = _periodic_records("dronet", period=55.0, n_instances=5, dispatch_duration=2.0)
    periods, non_periodic = group_by_periodicity(records)
    assert non_periodic == {}
    assert abs(periods["dronet"] - 55.0) < 1e-9, periods


def test_period_inference_robust_to_jitter():
    # Small per-instance jitter shouldn't move the median period much.
    jitter = [0.0, 0.4, -0.3, 0.2, -0.1]
    records = _periodic_records("dronet", period=55.0, n_instances=5, dispatch_duration=2.0, jitter=jitter)
    periods, _ = group_by_periodicity(records)
    assert abs(periods["dronet"] - 55.0) < 1.0, periods


def test_non_periodic_uses_max_duration():
    records = _non_periodic_records("mobilenet", durations=[9.29, 3.88, 2.04, 406.26])
    _, non_periodic = group_by_periodicity(records)
    group = non_periodic["mobilenet"]
    assert max(r.duration for r in group) == 406.26


def test_finer_recommended_when_coarse_dispatch_exceeds_period():
    # Motivating case: yolov8_nano-style single ~406ms block vs. dronet's
    # tight ~40ms period (unnamed.png).
    periodic = _periodic_records("dronet", period=40.0, n_instances=6, dispatch_duration=2.0)
    non_periodic = _non_periodic_records("yolov8_nano", durations=[406.26])
    advice = analyze_granularity(periodic + non_periodic)
    assert len(advice) == 1
    a = advice[0]
    assert a.subject == "yolov8_nano"
    assert a.recommended == "finer"
    assert a.conflicting_periodic_job == "dronet"
    assert "finer" in a.reason


def test_coarser_recommended_when_dispatches_far_smaller_than_period():
    periodic = _periodic_records("dronet", period=1000.0, n_instances=4, dispatch_duration=2.0)
    non_periodic = _non_periodic_records("mlp_control", durations=[0.5, 0.8, 1.0])
    advice = analyze_granularity(periodic + non_periodic)
    assert len(advice) == 1
    assert advice[0].recommended == "coarser"


def test_unchanged_when_duration_fits_comfortably():
    periodic = _periodic_records("dronet", period=55.0, n_instances=5, dispatch_duration=2.0)
    non_periodic = _non_periodic_records("mobilenet", durations=[9.0, 12.0, 20.0])
    advice = analyze_granularity(periodic + non_periodic)
    assert len(advice) == 1
    assert advice[0].recommended == "unchanged"


def test_no_periodic_job_means_no_advice():
    non_periodic = _non_periodic_records("mobilenet", durations=[9.0, 406.0])
    advice = analyze_granularity(non_periodic)
    assert advice == []


def test_from_schedule_json_recovers_instance_ids_from_dispatch_keys():
    # Mirrors the real (inconsistent) schema: job_name field is sometimes
    # the shared base ("dronet"), sometimes the per-instance id
    # ("mobilenet") -- the dispatch dict *key* is the reliable source.
    schedule = {
        "dispatches": {
            "dronet0_dispatch_0": {"start_time": 0.0, "duration": 2.0, "job_name": "dronet"},
            "dronet1_dispatch_0": {"start_time": 55.0, "duration": 2.0, "job_name": "dronet"},
            "dronet2_dispatch_0": {"start_time": 110.0, "duration": 2.0, "job_name": "dronet"},
            "mobilenet_dispatch_22_4": {"start_time": 908.8, "duration": 406.0, "job_name": "mobilenet"},
        },
        "metadata": {},
    }
    records = from_schedule_json(schedule)
    ids = {(r.instance_id, r.base_id) for r in records}
    assert ("dronet0", "dronet") in ids
    assert ("mobilenet", "mobilenet") in ids

    advice = analyze_granularity(records)
    assert len(advice) == 1
    assert advice[0].subject == "mobilenet"
    assert advice[0].recommended == "finer"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL: {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} failed: {failures}")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
