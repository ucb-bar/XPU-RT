"""Tests for benchmark record system."""

from __future__ import annotations

import json
import tempfile

from benchmarks.record import (
    AgenticMetrics,
    CaptureMetrics,
    EqSatMetrics,
    LLMMetrics,
    PerformanceMetrics,
    RecipeMetrics,
    RunRecord,
    SolverMetrics,
    VerificationMetrics,
)


def test_run_record_defaults() -> None:
    record = RunRecord()
    assert record.run_id  # non-empty UUID
    assert record.timestamp  # non-empty ISO timestamp
    assert record.model_name == ""
    assert record.readiness == "full_pipeline"
    assert record.promotion_status == "pending"


def test_run_record_to_dict() -> None:
    record = RunRecord(model_name="mlp", target_name="cuda_a100")
    d = record.to_dict()
    assert d["model_name"] == "mlp"
    assert d["target_name"] == "cuda_a100"
    assert "capture" in d
    assert "suite" in d
    assert "eqsat" in d
    assert "recipe" in d
    assert "solver" in d
    assert "kernels" in d
    assert "verification" in d
    assert "performance" in d
    assert "baselines" in d
    assert "llm" in d
    assert "agentic" in d
    assert "profiling" in d


def test_run_record_save_load_round_trip() -> None:
    record = RunRecord(
        model_name="test_model",
        target_name="test_target",
        objective="latency",
    )
    record.capture.export_success = True
    record.capture.op_coverage = 0.95
    record.eqsat.ops_before = 42
    record.eqsat.ops_after = 30
    record.eqsat.changed = True
    record.agentic.iterations_run = 5
    record.agentic.iteration_costs = [100.0, 90.0, 85.0, 83.0, 82.0]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = record.save(tmpdir)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = RunRecord.load(path)
        assert loaded.model_name == "test_model"
        assert loaded.capture.export_success is True
        assert loaded.capture.op_coverage == 0.95
        assert loaded.eqsat.ops_before == 42
        assert loaded.eqsat.changed is True
        assert loaded.agentic.iterations_run == 5
        assert len(loaded.agentic.iteration_costs) == 5


def test_capture_metrics_defaults() -> None:
    m = CaptureMetrics()
    assert m.export_success is False
    assert m.capture_mode == "torch_export"
    assert m.auto_translations_added == 0
    assert m.op_coverage == 0.0
    assert m.unsupported_ops == []


def test_eqsat_metrics() -> None:
    m = EqSatMetrics(ops_before=100, ops_after=75, ops_reduction_pct=25.0, changed=True)
    assert m.ops_reduction_pct == 25.0


def test_solver_metrics() -> None:
    m = SolverMetrics(placement_feasible=True, placement_gap=0.0, placement_time_ms=15.3)
    assert m.placement_feasible is True
    assert m.placement_gap == 0.0


def test_recipe_metrics() -> None:
    m = RecipeMetrics(total_recipe_ops=46, candidate_ops=14, fact_ops=8)
    assert m.total_recipe_ops == 46


def test_performance_metrics() -> None:
    m = PerformanceMetrics(
        latency_median_us=150.0,
        latency_p99_us=200.0,
        per_run_us=[140.0, 150.0, 160.0],
    )
    assert len(m.per_run_us) == 3


def test_verification_metrics() -> None:
    m = VerificationMetrics(structural_pass=True, differential_pass=True, overall_status="pass")
    assert m.overall_status == "pass"


def test_llm_metrics() -> None:
    m = LLMMetrics(total_calls=5, total_tokens=10000, total_cost_usd=0.15)
    assert m.total_cost_usd == 0.15


def test_agentic_metrics_convergence() -> None:
    m = AgenticMetrics(
        iterations_run=5,
        iteration_costs=[100.0, 80.0, 70.0, 68.0, 68.0],
        iteration_improvements=[0.0, 20.0, 12.5, 2.9, 0.0],
    )
    assert m.iterations_run == 5
    assert m.iteration_costs[1] == 80.0


def test_json_serialization() -> None:
    record = RunRecord(model_name="mlp")
    record.eqsat.rules_applied = {"commute": 3, "reassociate": 1}
    d = record.to_dict()
    text = json.dumps(d, default=str)
    loaded = json.loads(text)
    assert loaded["eqsat"]["rules_applied"]["commute"] == 3


def test_multiple_records_in_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        r1 = RunRecord(model_name="mlp", target_name="cuda")
        r2 = RunRecord(model_name="transformer", target_name="cuda")
        r1.save(tmpdir)
        r2.save(tmpdir)

        from benchmarks.compare import load_all_results

        loaded = load_all_results(tmpdir)
        assert len(loaded) == 2
