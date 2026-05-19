"""Promotion gate consumes bundle-level ``measured_cost.json``.

Closes the outer measurement loop: a successful
:class:`xpu_rt.runtime.executor.XpuRtExecutor` run that writes a
``measured_cost.json`` with ``correctness_vs_eager=pass`` and at least
one numeric measurement advances ``characterized`` even when the
heavier ``compiled_bottleneck`` / ``profiler_evidence`` pipelines did
not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xpu_rt.promotion.gates import (
    PromotionLevel,
    _check_characterized,
    evaluate_gate,
)


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def _seed_analytical(run_dir: Path) -> None:
    _write_json(
        run_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json",
        {"summary": {"candidates_modeled": 1}},
    )


def _seed_verified_fx(run_dir: Path) -> None:
    rp = run_dir / "03_recipe_planning"
    _write_json(rp / "candidate_selection.json", {"selected_candidate_id": "c0"})
    _write_json(rp / "real_transform_differential_report.json", {"status": "pass"})


def _seed_verified_kernel(run_dir: Path) -> None:
    _write_json(
        run_dir / "02_graph_analysis" / "kernel_execution"
        / "kernel_execution_report.json",
        {"status": "pass"},
    )


def _seed_promoted(run_dir: Path) -> None:
    ga = run_dir / "02_graph_analysis"
    _write_json(
        ga / "readiness" / "graph_analysis_readiness_matrix.json",
        {"overall": "pass"},
    )
    _write_json(
        ga / "kernel_readiness" / "kernel_section_readiness_matrix.json",
        {"overall": "pass"},
    )
    _write_json(
        run_dir / "04_promotion" / "verification_report.json",
        {"passed": True},
    )


def test_characterized_passes_on_bundle_measured_cost(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    _seed_analytical(run_dir)
    # No compiled_bottleneck, no profiler_evidence — only the bundle.
    _write_json(
        bundle / "measured_cost.json",
        {
            "schema_version": "measured_cost_v1",
            "executor": "spike_gemmini",
            "correctness_vs_eager": "pass",
            "cycles_total": 12345,
            "samples": [{"region_id": "r0", "cycles": 12345, "correctness": "pass"}],
        },
    )
    ok, reason, summary = _check_characterized(run_dir, bundle)
    assert ok, f"expected characterized but reason={reason!r}"
    assert summary["bundle_measured_cost"] == "present"


def test_characterized_rejects_failed_executor_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    _seed_analytical(run_dir)
    _write_json(
        bundle / "measured_cost.json",
        {
            "schema_version": "measured_cost_v1",
            "executor": "spike_gemmini",
            "correctness_vs_eager": "fail",
            "samples": [{"region_id": "r0", "correctness": "fail"}],
        },
    )
    ok, reason, summary = _check_characterized(run_dir, bundle)
    assert not ok
    assert "measured cost not present" in reason
    assert summary["bundle_measured_cost"] == "missing"


def test_characterized_requires_at_least_one_numeric(tmp_path: Path) -> None:
    """A ``pass`` with no cycles or latency is not strong enough."""
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    _seed_analytical(run_dir)
    _write_json(
        bundle / "measured_cost.json",
        {
            "schema_version": "measured_cost_v1",
            "executor": "spike_gemmini",
            "correctness_vs_eager": "pass",
            "cycles_total": None,
            "latency_us_p50_total": None,
            "samples": [{"region_id": "r0", "correctness": "pass"}],
        },
    )
    ok, _reason, _summary = _check_characterized(run_dir, bundle)
    assert not ok


def test_evaluate_gate_threads_bundle_dir_through(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)

    _seed_verified_fx(run_dir)
    _seed_verified_kernel(run_dir)
    _seed_analytical(run_dir)
    _write_json(
        bundle / "measured_cost.json",
        {
            "schema_version": "measured_cost_v1",
            "executor": "spike_gemmini",
            "correctness_vs_eager": "pass",
            "cycles_total": 99,
            "samples": [{"region_id": "r0", "cycles": 99, "correctness": "pass"}],
        },
    )

    eval_without_bundle = evaluate_gate(run_dir)
    eval_with_bundle = evaluate_gate(run_dir, bundle_dir=bundle)

    # Without the bundle we cap at verified_kernel (no measured cost from M-22).
    assert eval_without_bundle.level == PromotionLevel.VERIFIED_KERNEL
    # With the bundle the executor's measurement counts for characterized.
    assert eval_with_bundle.level == PromotionLevel.CHARACTERIZED


@pytest.mark.parametrize("with_readiness", [False, True])
def test_full_ladder_with_executor_cost(
    tmp_path: Path, with_readiness: bool,
) -> None:
    run_dir = tmp_path / "run"
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    _seed_verified_fx(run_dir)
    _seed_verified_kernel(run_dir)
    _seed_analytical(run_dir)
    _write_json(
        bundle / "measured_cost.json",
        {
            "schema_version": "measured_cost_v1",
            "executor": "spike_gemmini",
            "correctness_vs_eager": "pass",
            "cycles_total": 4242,
            "samples": [{"region_id": "r0", "cycles": 4242, "correctness": "pass"}],
        },
    )
    if with_readiness:
        _seed_promoted(run_dir)

    result = evaluate_gate(run_dir, bundle_dir=bundle)
    if with_readiness:
        assert result.level == PromotionLevel.PROMOTED
    else:
        assert result.level == PromotionLevel.CHARACTERIZED
