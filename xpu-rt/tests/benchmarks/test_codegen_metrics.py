"""Tests for codegen metric dataclasses and collector functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.collector import (
    collect_codegen_funnel,
    collect_codegen_region_detail,
    collect_fallback_pressure,
    collect_kernel_rollups,
    collect_layout_friction,
)
from benchmarks.record import (
    CodegenFunnel,
    CodegenRegionDetail,
    EqSatMetrics,
    FallbackPressure,
    KernelMetrics,
    LayoutFriction,
    RunRecord,
    SolverMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_decision(
    strategy_value: str = "native",
    op_name: str = "linalg.matmul",
    reason: str = "",
    library_name: str | None = None,
    flops: int = 1000,
) -> SimpleNamespace:
    cost = SimpleNamespace(flops=flops)
    contract = SimpleNamespace(op_name=op_name, cost=cost)
    spec = SimpleNamespace(contract=contract)
    strategy = SimpleNamespace(value=strategy_value)
    return SimpleNamespace(spec=spec, strategy=strategy, reason=reason, library_name=library_name)


# ---------------------------------------------------------------------------
# TestCodegenDataclassDefaults
# ---------------------------------------------------------------------------

class TestCodegenDataclassDefaults:
    """Verify new codegen dataclasses instantiate with correct defaults."""

    def test_codegen_region_detail_defaults(self) -> None:
        d = CodegenRegionDetail()
        assert d.region_id == ""
        assert d.op_family == ""
        assert d.selected_strategy == ""
        assert d.candidate_backends == []
        assert d.selected_backend == ""
        assert d.search_budget == 0
        assert d.search_iterations_used == 0
        assert d.generated_kernel_count == 0
        assert d.compile_success is False
        assert d.numeric_pass is False
        assert d.perf_target_us == 0.0
        assert d.measured_latency_us == 0.0
        assert d.speedup_vs_reference == 0.0
        assert d.fallback_reason == ""
        assert d.layout_contract == ""
        assert d.prepack_applied is False
        assert d.transpose_materializations == 0
        assert d.opaque_boundary is False

    def test_fallback_pressure_defaults(self) -> None:
        fp = FallbackPressure()
        assert fp.fallback_region_count == 0
        assert fp.fallback_flop_share == 0.0
        assert fp.fallback_latency_share == 0.0
        assert fp.fallback_reasons_histogram == {}

    def test_layout_friction_defaults(self) -> None:
        lf = LayoutFriction()
        assert lf.materialized_transposes == 0
        assert lf.bytes_on_relayout == 0
        assert lf.prepacked_operands == 0
        assert lf.regions_in_propagated_layout == 0
        assert lf.opaque_boundaries_forcing_materialization == 0

    def test_codegen_funnel_defaults(self) -> None:
        cf = CodegenFunnel()
        assert cf.eligible == 0
        assert cf.attempted == 0
        assert cf.compiled == 0
        assert cf.verified == 0
        assert cf.faster == 0
        assert cf.promoted == 0
        assert cf.geo_mean_speedup == 0.0
        assert cf.median_time_to_first_valid_ms == 0.0
        assert cf.median_time_to_first_faster_ms == 0.0
        assert cf.budget_utilization == 0.0


# ---------------------------------------------------------------------------
# TestExtendedDataclassBackcompat
# ---------------------------------------------------------------------------

class TestExtendedDataclassBackcompat:
    """Ensure extended dataclasses remain backward-compatible."""

    def test_kernel_metrics_new_fields_defaults(self) -> None:
        km = KernelMetrics()
        assert km.region_details == []
        assert km.pct_native == 0.0
        assert km.pct_library == 0.0
        assert km.pct_fallback == 0.0
        assert km.pct_generated == 0.0
        assert km.pct_opaque == 0.0
        assert km.pct_verified_numerically == 0.0
        assert km.compile_ms_per_region == 0.0
        assert km.roofline_gap == 0.0
        # Pre-existing fields still work.
        assert km.total_kernel_specs == 0
        assert km.strategy_histogram == {}

    def test_eqsat_metrics_new_fields_defaults(self) -> None:
        em = EqSatMetrics()
        assert em.pct_regions_touched == 0.0
        assert em.top_rewrite_families == []
        assert em.speedup_delta_from_eqsat == 0.0
        # Pre-existing fields still work.
        assert em.ops_before == 0

    def test_solver_metrics_new_fields_defaults(self) -> None:
        sm = SolverMetrics()
        assert sm.distinct_devices_used == 0
        assert sm.total_bytes_transferred == 0
        assert sm.pct_time_solver == 0.0
        assert sm.pct_time_codegen == 0.0
        assert sm.pct_time_verification == 0.0
        # Pre-existing fields still work.
        assert sm.placement_feasible is False

    def test_run_record_new_nested_fields(self) -> None:
        rr = RunRecord()
        assert isinstance(rr.fallback_pressure, FallbackPressure)
        assert isinstance(rr.layout_friction, LayoutFriction)
        assert isinstance(rr.codegen_funnel, CodegenFunnel)


# ---------------------------------------------------------------------------
# TestRunRecordSerialization
# ---------------------------------------------------------------------------

class TestRunRecordSerialization:
    """Round-trip and backward-compat serialization tests."""

    def test_save_load_roundtrip_with_new_fields(self, tmp_path: Path) -> None:
        rr = RunRecord(model_name="test_model", target_name="gpu0")
        rr.fallback_pressure = FallbackPressure(
            fallback_region_count=3,
            fallback_flop_share=0.15,
            fallback_latency_share=0.10,
            fallback_reasons_histogram={"unsupported_dtype": 2, "no_backend": 1},
        )
        rr.layout_friction = LayoutFriction(
            materialized_transposes=4,
            bytes_on_relayout=8192,
            prepacked_operands=2,
            regions_in_propagated_layout=6,
            opaque_boundaries_forcing_materialization=1,
        )
        rr.codegen_funnel = CodegenFunnel(
            eligible=10, attempted=8, compiled=7, verified=6,
            faster=5, promoted=4, geo_mean_speedup=1.35,
        )
        rr.kernels.pct_native = 50.0
        rr.kernels.roofline_gap = 1.2

        saved = rr.save(tmp_path)
        loaded = RunRecord.load(saved)

        assert loaded.fallback_pressure.fallback_region_count == 3
        assert loaded.fallback_pressure.fallback_flop_share == pytest.approx(0.15)
        assert loaded.fallback_pressure.fallback_reasons_histogram == {"unsupported_dtype": 2, "no_backend": 1}
        assert loaded.layout_friction.materialized_transposes == 4
        assert loaded.layout_friction.prepacked_operands == 2
        assert loaded.codegen_funnel.eligible == 10
        assert loaded.codegen_funnel.geo_mean_speedup == pytest.approx(1.35)
        assert loaded.kernels.pct_native == pytest.approx(50.0)
        assert loaded.kernels.roofline_gap == pytest.approx(1.2)

    def test_load_legacy_json_without_new_fields(self, tmp_path: Path) -> None:
        legacy = {
            "run_id": "legacy01",
            "model_name": "old_model",
            "target_name": "cpu",
            "status": "pass",
            "kernels": {"total_kernel_specs": 5, "strategy_histogram": {"native": 5}},
        }
        path = tmp_path / "legacy01_compgen_old_model_cpu.json"
        path.write_text(json.dumps(legacy))

        loaded = RunRecord.load(path)
        assert loaded.run_id == "legacy01"
        # New nested fields fall back to defaults.
        assert loaded.fallback_pressure.fallback_region_count == 0
        assert loaded.layout_friction.materialized_transposes == 0
        assert loaded.codegen_funnel.eligible == 0
        # Pre-existing nested field loaded correctly.
        assert loaded.kernels.total_kernel_specs == 5


# ---------------------------------------------------------------------------
# TestCollectors
# ---------------------------------------------------------------------------

class TestCollectors:
    """Tests for the codegen-specific collector functions."""

    def test_collect_codegen_region_detail_basic(self) -> None:
        decision = _mock_decision(strategy_value="autocomp", op_name="linalg.matmul")
        result = collect_codegen_region_detail(decision, region_id="r0")
        assert result["region_id"] == "r0"
        assert result["op_family"] == "matmul"
        assert result["selected_strategy"] == "autocomp"
        assert result["fallback_reason"] == ""

    def test_collect_codegen_region_detail_fallback(self) -> None:
        decision = _mock_decision(
            strategy_value="fallback",
            op_name="custom.weird_op",
            reason="no_backend_available",
        )
        result = collect_codegen_region_detail(decision)
        assert result["selected_strategy"] == "fallback"
        assert result["fallback_reason"] == "no_backend_available"

    def test_collect_fallback_pressure_no_fallbacks(self) -> None:
        decisions = [_mock_decision("native") for _ in range(4)]
        fp = collect_fallback_pressure(decisions)
        assert fp.fallback_region_count == 0
        assert fp.fallback_flop_share == pytest.approx(0.0)

    def test_collect_fallback_pressure_mixed(self) -> None:
        decisions = [
            _mock_decision("native", flops=2000),
            _mock_decision("fallback", flops=3000, reason="unsupported_dtype"),
            _mock_decision("native", flops=5000),
        ]
        fp = collect_fallback_pressure(decisions)
        assert fp.fallback_region_count == 1
        assert fp.fallback_flop_share == pytest.approx(3000 / 10000)
        assert fp.fallback_reasons_histogram == {"unsupported_dtype": 1}

    def test_collect_layout_friction_empty(self) -> None:
        lf = collect_layout_friction()
        assert lf.materialized_transposes == 0
        assert lf.prepacked_operands == 0
        assert lf.regions_in_propagated_layout == 0
        assert lf.opaque_boundaries_forcing_materialization == 0

    def test_collect_layout_friction_with_prepacks(self) -> None:
        plans = {
            "r0": SimpleNamespace(prepack_candidates=["a", "b"], tile_encoding="tile4x4"),
            "r1": SimpleNamespace(prepack_candidates=[], tile_encoding=None),
        }
        details = [
            {"transpose_materializations": 1, "opaque_boundary": True},
            {"transpose_materializations": 0, "opaque_boundary": False},
        ]
        lf = collect_layout_friction(layout_plans=plans, region_details=details)
        assert lf.prepacked_operands == 2
        assert lf.regions_in_propagated_layout == 1
        assert lf.materialized_transposes == 1
        assert lf.opaque_boundaries_forcing_materialization == 1

    def test_collect_codegen_funnel_empty(self) -> None:
        cf = collect_codegen_funnel()
        assert cf.eligible == 0
        assert cf.attempted == 0
        assert cf.compiled == 0
        assert cf.verified == 0
        assert cf.faster == 0
        assert cf.promoted == 0

    def test_collect_codegen_funnel_full_pipeline(self) -> None:
        details = [
            {
                "selected_strategy": "autocomp",
                "search_iterations_used": 5,
                "search_budget": 10,
                "compile_success": True,
                "numeric_pass": True,
                "speedup_vs_reference": 1.5,
                "generated_kernel_count": 1,
            },
            {
                "selected_strategy": "exo",
                "search_iterations_used": 3,
                "search_budget": 10,
                "compile_success": True,
                "numeric_pass": True,
                "speedup_vs_reference": 0.9,
                "generated_kernel_count": 1,
            },
            {
                "selected_strategy": "native",
                "search_iterations_used": 0,
                "search_budget": 0,
                "compile_success": False,
                "numeric_pass": False,
                "speedup_vs_reference": 0.0,
                "generated_kernel_count": 0,
            },
        ]
        cf = collect_codegen_funnel(details)
        assert cf.eligible == 2  # native is skipped
        assert cf.attempted == 2
        assert cf.compiled == 2
        assert cf.verified == 2
        assert cf.faster == 1  # only speedup > 1.0
        assert cf.promoted == 2  # generated_kernel_count > 0 and numeric_pass

    def test_collect_kernel_rollups_basic(self) -> None:
        histogram = {"native": 5, "library": 3, "autocomp": 2}
        result = collect_kernel_rollups(histogram)
        total = 10
        assert result["pct_native"] == pytest.approx(5 / total * 100)
        assert result["pct_library"] == pytest.approx(3 / total * 100)
        assert result["pct_fallback"] == pytest.approx(0.0)
        assert result["pct_generated"] == pytest.approx(2 / total * 100)
        assert result["pct_opaque"] == pytest.approx(3 / total * 100)  # library + unsupported
