"""Tests for the schedule translation-validation harness."""

from __future__ import annotations

import pytest

from xpu_rt.solve.schedule_tv import translation_validate_schedule


def _clean_three_op() -> dict:
    """A clean 3-op schedule on 2 devices.

    Layout (µs):
        device 0: [a 0..10][b 10..25]
        device 1: [c 25..35]
        dependencies: c depends on b
    """
    return dict(
        partition_ids=["a", "b", "c"],
        durations_us_by_device={
            "a": [10.0, 10.0],
            "b": [15.0, 15.0],
            "c": [12.0, 10.0],
        },
        dependencies={"b": [], "c": ["b"]},
        num_devices=2,
        start_times={"a": 0.0, "b": 10.0, "c": 25.0},
        end_times={"a": 10.0, "b": 25.0, "c": 35.0},
        device_assignments={"a": 0, "b": 0, "c": 1},
        makespan_us=35.0,
        transfer_us=[[0.0, 0.0], [0.0, 0.0]],
    )


def test_clean_schedule_proves() -> None:
    res = translation_validate_schedule(**_clean_three_op())
    assert res.proved, res.violations
    assert res.z3_time_ms > 0  # Z3 path ran
    assert res.python_time_ms >= 0
    assert res.n_deps_checked == 1


def test_dep_violation_caught() -> None:
    kwargs = _clean_three_op()
    # Shift c backwards so it starts before b ends.
    kwargs["start_times"] = {**kwargs["start_times"], "c": 20.0}
    kwargs["end_times"] = {**kwargs["end_times"], "c": 30.0}
    kwargs["makespan_us"] = 30.0
    res = translation_validate_schedule(**kwargs)
    assert not res.proved
    kinds = {v.kind for v in res.violations}
    assert "dep_violated" in kinds


def test_device_overlap_caught() -> None:
    kwargs = _clean_three_op()
    # Move c onto device 0 overlapping b.
    kwargs["device_assignments"] = {**kwargs["device_assignments"], "c": 0}
    # Now c on dev 0 still starts at 25 = end of b, but its duration on
    # dev 0 is 12 not 10, so reported end (35) is inconsistent. Fix the
    # end so duration matches dev 0, then deliberately overlap.
    kwargs["start_times"] = {**kwargs["start_times"], "c": 20.0}
    kwargs["end_times"] = {**kwargs["end_times"], "c": 32.0}
    kwargs["makespan_us"] = 32.0
    res = translation_validate_schedule(**kwargs)
    assert not res.proved
    kinds = {v.kind for v in res.violations}
    assert "device_overlap" in kinds


def test_unassigned_partition_caught() -> None:
    kwargs = _clean_three_op()
    kwargs["device_assignments"] = {"a": 0, "b": 0}  # c missing
    res = translation_validate_schedule(**kwargs)
    assert not res.proved
    kinds = {v.kind for v in res.violations}
    assert "unassigned" in kinds


def test_makespan_mismatch_caught() -> None:
    kwargs = _clean_three_op()
    kwargs["makespan_us"] = 999.0
    res = translation_validate_schedule(**kwargs)
    assert not res.proved
    kinds = {v.kind for v in res.violations}
    assert "makespan_mismatch" in kinds


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
