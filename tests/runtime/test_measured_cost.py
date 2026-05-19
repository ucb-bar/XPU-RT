"""Unit tests for :mod:`xpu_rt.runtime.measured_cost`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xpu_rt.runtime.measured_cost import (
    SCHEMA_VERSION,
    MeasuredCost,
    MeasurementSample,
    aggregate_correctness,
)


def test_measured_cost_round_trip(tmp_path: Path) -> None:
    cost = MeasuredCost(
        executor="spike_gemmini",
        hardware_key="gemmini",
        target_id="gemmini",
        cycles_total=1_234_567,
        correctness_vs_eager="pass",
        samples=(
            MeasurementSample(
                region_id="matmul_0",
                op_family="matmul",
                cycles=1_234_567,
                correctness="pass",
                mismatches=0,
                total_elements=512,
                extras={"score": 1.0},
            ),
        ),
        run_id="bundle_abc",
        notes="",
    )
    path = tmp_path / "measured_cost.json"
    cost.write_json(path)

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["executor"] == "spike_gemmini"
    assert payload["cycles_total"] == 1_234_567
    assert payload["correctness_vs_eager"] == "pass"
    assert payload["samples"][0]["region_id"] == "matmul_0"

    restored = MeasuredCost.read_json(path)
    assert restored == cost


def test_measured_cost_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "measured_cost.json"
    path.write_text(json.dumps({"schema_version": "from_the_future"}))
    with pytest.raises(ValueError, match="schema_version"):
        MeasuredCost.read_json(path)


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ((), "skipped"),
        (("pass",), "pass"),
        (("pass", "pass"), "pass"),
        (("pass", "skipped"), "skipped"),
        (("pass", "fail"), "fail"),
        (("pass", "fail", "skipped"), "fail"),
        (("fail", "error"), "error"),
        (("error", "pass"), "error"),
    ],
)
def test_aggregate_correctness(outcomes: tuple[str, ...], expected: str) -> None:
    samples = tuple(
        MeasurementSample(region_id=f"r{i}", correctness=o)  # type: ignore[arg-type]
        for i, o in enumerate(outcomes)
    )
    assert aggregate_correctness(samples) == expected


def test_measured_cost_tolerates_partial_fields() -> None:
    """Old bundles or hand-written stubs may omit optional fields."""
    restored = MeasuredCost.from_json_dict({
        "executor": "spike_gemmini",
        "correctness_vs_eager": "pass",
        "samples": [{"region_id": "r0", "cycles": 100}],
    })
    assert restored.executor == "spike_gemmini"
    assert restored.cycles_total is None
    assert restored.samples[0].cycles == 100
    # Default rolls back to "skipped" — caller can choose to recompute
    # via aggregate_correctness if they want it derived.
    assert restored.samples[0].correctness == "skipped"
