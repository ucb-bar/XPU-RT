"""Unit tests for the split / coarsen / deltas heuristics."""

from __future__ import annotations

from xpu_rt.targets.backends.qnn.granularity import (
    compute_coarsen_candidates,
    compute_split_candidates,
    predicted_vs_measured_table,
)


def _schedule_with(ops):
    return {
        "makespan_us": max(o["finish_us"] for o in ops),
        "ops": ops,
        "dispatches": {o["name"]: o for o in ops},
    }


def _profile_with(measured):
    return {
        "dispatches": {
            name: {target: {"mean_us": value}
                   for target, value in by_t.items()}
            for name, by_t in measured.items()
        }
    }


def test_split_flags_high_ratio_dominant_region():
    schedule = _schedule_with([
        {"name": "yolo.backbone", "machine": "HTA",
         "start_us": 0.0, "finish_us": 1000.0, "predicted_us": 1000.0,
         "workload": "yolo"},
        {"name": "yolo.head", "machine": "CPU",
         "start_us": 0.0, "finish_us": 200.0, "predicted_us": 200.0,
         "workload": "yolo"},
    ])
    profile = _profile_with({
        "yolo.backbone": {"qnn_hta": 1800.0},   # ratio 1.8, share 1.0
        "yolo.head": {"cpu": 220.0},            # ratio 1.1, share 0.2
    })

    splits = compute_split_candidates(
        dossier=None, profile=profile, schedule=schedule,
    )
    assert len(splits) == 1
    assert splits[0].dispatch_id == "yolo.backbone"
    assert splits[0].ratio > 1.3
    assert splits[0].region_share >= 0.10


def test_split_ignores_small_share_even_when_slow():
    schedule = _schedule_with([
        {"name": "yolo.big", "machine": "HTA",
         "start_us": 0.0, "finish_us": 10_000.0, "predicted_us": 10_000.0,
         "workload": "yolo"},
        {"name": "yolo.tiny", "machine": "CPU",
         "start_us": 0.0, "finish_us": 50.0, "predicted_us": 50.0,
         "workload": "yolo"},
    ])
    profile = _profile_with({
        "yolo.tiny": {"cpu": 500.0},  # ratio 10, but share only 0.5%
        "yolo.big": {"qnn_hta": 10_100.0},  # ratio 1.01
    })
    splits = compute_split_candidates(
        dossier=None, profile=profile, schedule=schedule,
    )
    assert splits == []


def test_coarsen_flags_dominant_transfer_same_backend():
    schedule = _schedule_with([
        {"name": "a", "machine": "HTA",
         "start_us": 0.0, "finish_us": 100.0, "predicted_us": 100.0},
        {"name": "b", "machine": "HTA",
         "start_us": 150.0, "finish_us": 220.0, "predicted_us": 70.0},
    ])
    # transfer = start_b - finish_a = 50µs; compute = 100+70 = 170µs;
    # ratio = 50/170 ≈ 0.294 — above the 0.20 coarsen threshold.
    cs = compute_coarsen_candidates(schedule=schedule)
    assert len(cs) == 1
    assert cs[0].first_dispatch_id == "a"
    assert cs[0].second_dispatch_id == "b"
    assert cs[0].ratio >= 0.20


def test_coarsen_skips_different_backends():
    schedule = _schedule_with([
        {"name": "a", "machine": "HTA",
         "start_us": 0.0, "finish_us": 100.0, "predicted_us": 100.0},
        {"name": "b", "machine": "GPU",
         "start_us": 200.0, "finish_us": 300.0, "predicted_us": 100.0},
    ])
    assert compute_coarsen_candidates(schedule=schedule) == []


def test_deltas_handles_missing_measurements():
    schedule = _schedule_with([
        {"name": "x", "machine": "CPU",
         "start_us": 0.0, "finish_us": 200.0, "predicted_us": 200.0},
        {"name": "y", "machine": "HTA",
         "start_us": 0.0, "finish_us": 100.0, "predicted_us": 100.0},
    ])
    profile = _profile_with({"x": {"cpu": 220.0}})
    rows = predicted_vs_measured_table(profile=profile, schedule=schedule)
    by_name = {r["dispatch"]: r for r in rows}
    assert by_name["x"]["measured_us"] == 220.0
    assert by_name["x"]["ratio"] is not None
    assert by_name["y"]["measured_us"] is None
    assert by_name["y"]["ratio"] is None
