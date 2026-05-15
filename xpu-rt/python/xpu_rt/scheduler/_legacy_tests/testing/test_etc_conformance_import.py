"""API-surface tests for the conformance harness.

CPU-only. Locks the public shape of
:mod:`compgen.testing.etc_conformance` so the remote agent can rely
on it stably across releases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestPublicAPI:
    def test_top_level_imports(self) -> None:
        """Every name we promised the remote agent must import."""
        from compgen.testing.etc_conformance import (
            ConformanceReport,
            ConformanceWorkload,
            PassGate,
            run_conformance,
            summarize_reports,
        )

        assert callable(run_conformance)
        assert callable(summarize_reports)
        assert ConformanceReport is not None
        assert ConformanceWorkload is not None
        assert PassGate is not None

    def test_workload_enum_has_six_paper_workloads(self) -> None:
        from compgen.testing.etc_conformance import ConformanceWorkload

        names = {w.value for w in ConformanceWorkload}
        # Paper anchor workloads.
        assert names == {
            "gemm_rs",
            "ag_gemm",
            "moe_fwd",
            "shape_dynamic_mlp",
            "decoder_layer",
            "diamond_dag",
        }

    def test_workload_enum_is_string_subclass(self) -> None:
        """Subclassing str makes JSON serialisation + literal compare
        ergonomic for the MCP tools."""
        from compgen.testing.etc_conformance import ConformanceWorkload

        assert ConformanceWorkload("diamond_dag") == "diamond_dag"

    def test_default_pass_gate_matches_plan(self) -> None:
        from compgen.testing.etc_conformance import PassGate

        gate = PassGate()
        assert gate.correctness_atol == 1e-3
        assert gate.correctness_rtol == 1e-3
        assert gate.max_launches_static == 1
        assert gate.max_launches_dynamic == 2
        assert gate.require_atomics is True
        assert gate.min_speedup_vs_eager == 1.2

    def test_pass_gate_overridable(self) -> None:
        """Bring-up workflow: relax the speedup floor."""
        from compgen.testing.etc_conformance import PassGate

        gate = PassGate(min_speedup_vs_eager=0.5)
        assert gate.min_speedup_vs_eager == 0.5
        # other defaults preserved
        assert gate.correctness_atol == 1e-3


class TestConformanceReport:
    def test_to_dict_round_trips_through_json(self) -> None:
        from compgen.testing.etc_conformance import (
            ConformanceReport,
            ConformanceWorkload,
            PassGate,
        )

        rep = ConformanceReport(
            workload=ConformanceWorkload.DIAMOND_DAG,
            dtype="bf16",
            device="cuda:0",
            compute_capability=(12, 0),
            passed=False,
            correctness={"max_abs_err": 0.0},
            timing={"speedup_vs_eager": 0.0},
            launch_profile={"num_launches": 0},
            bundle_dir=None,
            gate=PassGate(),
            errors=["test error"],
            metadata={"sm_count": 188},
        )
        d = rep.to_dict()
        # Must be JSON-serialisable (no enum / Path leakage).
        round_trip = json.loads(json.dumps(d))
        assert round_trip["workload"] == "diamond_dag"
        assert round_trip["compute_capability"] == [12, 0]
        assert round_trip["gate"]["correctness_atol"] == 1e-3

    def test_write_json_lands_under_workload_filename(self, tmp_path: Path) -> None:
        from compgen.testing.etc_conformance import (
            ConformanceReport,
            ConformanceWorkload,
            PassGate,
        )

        rep = ConformanceReport(
            workload=ConformanceWorkload.DIAMOND_DAG,
            dtype="bf16",
            device="cpu",
            compute_capability=None,
            passed=False,
            correctness={},
            timing={},
            launch_profile={},
            bundle_dir=None,
            gate=PassGate(),
            errors=[],
        )
        path = rep.write_json(tmp_path)
        assert path.name == "diamond_dag.conformance_report.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["workload"] == "diamond_dag"


class TestRunConformanceCpuFallback:
    """On a CPU host the harness must report passed=False with a
    clear reason — never silently pretend a CPU run is a Blackwell
    PASS."""

    def test_cpu_host_reports_clean_failure(self, tmp_path: Path) -> None:
        import torch
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            run_conformance,
        )

        # Skip when an actual CUDA device is reachable — that's a
        # different code path covered by the GPU-marked tests.
        if torch.cuda.is_available():
            pytest.skip("This test exercises the CPU-fallback path")

        rep = run_conformance(
            ConformanceWorkload.DIAMOND_DAG,
            dtype="bf16",
            output_dir=tmp_path,
        )
        assert rep.passed is False
        assert rep.errors, "CPU-fallback must enumerate why it can't pass"
        # The first error must be an environment / routing reason,
        # not a bare exception trace.
        assert any(
            "cuda" in e.lower() or "torch.cuda" in e.lower() or "etc dispatch" in e.lower() for e in rep.errors
        ), f"unexpected error shape: {rep.errors}"

    def test_report_lands_on_disk_even_when_failing(self, tmp_path: Path) -> None:
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            run_conformance,
        )

        rep = run_conformance(
            ConformanceWorkload.DIAMOND_DAG,
            dtype="bf16",
            output_dir=tmp_path,
        )
        path = tmp_path / "diamond_dag.conformance_report.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["workload"] == "diamond_dag"
        assert loaded["passed"] is False


class TestSummarizeReports:
    def test_empty_dir_is_handled(self, tmp_path: Path) -> None:
        from compgen.testing.etc_conformance import summarize_reports

        out = summarize_reports(tmp_path)
        assert "No conformance reports found" in out

    def test_table_lists_each_workload(self, tmp_path: Path) -> None:
        from compgen.testing.etc_conformance import (
            run_conformance,
            summarize_reports,
        )

        # Run two workloads to populate the dir.
        for name in ("diamond_dag", "decoder_layer"):
            run_conformance(name, dtype="bf16", output_dir=tmp_path)

        table = summarize_reports(tmp_path)
        assert "| Workload" in table
        assert "diamond_dag" in table
        assert "decoder_layer" in table
        # Every row is FAIL on CPU host, so there should be ❌ markers.
        # On GPU runners the markers may differ; just assert presence
        # of one of the two valid status glyphs.
        assert "❌" in table or "✅" in table


class TestCli:
    def test_summary_only_flag_runs_without_a_gpu(self, tmp_path: Path) -> None:
        # Pre-create one report so summary has something to read.
        from compgen.testing.etc_conformance import (
            _cli,
            run_conformance,
        )

        run_conformance("diamond_dag", dtype="bf16", output_dir=tmp_path)

        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "compgen-run-conformance",
                "--summary-only",
                "--output-dir",
                str(tmp_path),
            ]
            rc = _cli()
        finally:
            sys.argv = old_argv
        assert rc == 0
