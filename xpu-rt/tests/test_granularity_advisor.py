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
    _critical_path_ms,
    _is_linear_chain,
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


def _periodic_chain_records(base: str, period: float, n_instances: int, chain_durations: list[float]):
    """n_instances periodic instances of `base`, each a *serial chain* of
    len(chain_durations) dispatches (dispatch j depends on dispatch j-1),
    spaced `period` apart. Exercises critical-path-based utilization (F_p),
    as opposed to `_periodic_records`' single-dispatch instances where
    critical path trivially equals that one dispatch's duration."""
    records = []
    for i in range(n_instances):
        instance_start = i * period
        prev_key = None
        for j, d in enumerate(chain_durations):
            key = f"{base}{i}_dispatch_{j}"
            records.append(DispatchRecord(
                instance_id=f"{base}{i}",
                base_id=base,
                is_periodic=True,
                start_time=instance_start,  # only the instance's earliest matters for period inference
                duration=d,
                dispatch_key=key,
                dependencies=[prev_key] if prev_key else [],
            ))
            prev_key = key
    return records


def _branching_non_periodic_records(base: str, root_duration: float, branch_durations: list[float]):
    """One non-periodic job whose dispatches branch: a root dispatch with
    multiple direct dependents (out-degree > 1) -- not a linear chain."""
    records = [DispatchRecord(
        instance_id=base, base_id=base, is_periodic=False,
        start_time=0.0, duration=root_duration, dispatch_key=f"{base}_dispatch_0",
        dependencies=[],
    )]
    for i, d in enumerate(branch_durations, start=1):
        records.append(DispatchRecord(
            instance_id=base, base_id=base, is_periodic=False,
            start_time=float(i * 10), duration=d, dispatch_key=f"{base}_dispatch_{i}",
            dependencies=[f"{base}_dispatch_0"],
        ))
    return records


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


def test_critical_path_is_serial_sum_for_a_chain():
    chain = _periodic_chain_records("dronet", period=100.0, n_instances=1, chain_durations=[20.0, 20.0, 20.0])
    assert _critical_path_ms(chain) == 60.0


def test_critical_path_is_not_overcounted_for_a_branch():
    branch = _branching_non_periodic_records("widget", root_duration=10.0, branch_durations=[5.0, 30.0])
    # root (10) -> longest branch (30) = 40, NOT root + sum(branches) = 45.
    assert _critical_path_ms(branch) == 40.0


def test_is_linear_chain_true_for_a_chain():
    chain = _periodic_chain_records("dronet", period=100.0, n_instances=1, chain_durations=[20.0, 20.0, 20.0])
    assert _is_linear_chain(chain) is True


def test_is_linear_chain_false_for_a_branch():
    branch = _branching_non_periodic_records("widget", root_duration=10.0, branch_durations=[5.0, 5.0])
    assert _is_linear_chain(branch) is False


def test_free_slot_accounts_for_periodic_jobs_own_utilization():
    # dronet's own 3-dispatch chain (20+20+20=60ms) eats 60% of its 100ms
    # period, leaving only a 40ms free slot -- so mobilenet's 50ms dispatch
    # should be flagged "finer" even though 50 < the *raw* 100ms period
    # (which the old period-only heuristic would have called "unchanged").
    periodic = _periodic_chain_records("dronet", period=100.0, n_instances=4, chain_durations=[20.0, 20.0, 20.0])
    non_periodic = _non_periodic_records("mobilenet", durations=[50.0])
    advice = analyze_granularity(periodic + non_periodic)
    assert len(advice) == 1
    a = advice[0]
    assert a.period_ms == 100.0
    assert abs(a.free_slot_ms - 40.0) < 1e-9, a.free_slot_ms
    assert a.recommended == "finer"


def test_coarser_capped_to_unchanged_when_job_branches():
    periodic = _periodic_records("dronet", period=1000.0, n_instances=4, dispatch_duration=2.0)
    non_periodic = _branching_non_periodic_records("widget", root_duration=1.0, branch_durations=[0.5, 0.5])
    advice = analyze_granularity(periodic + non_periodic)
    assert len(advice) == 1
    assert advice[0].recommended == "unchanged"
    assert "branch" in advice[0].reason


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
