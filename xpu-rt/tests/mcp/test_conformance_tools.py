"""MCP-tool wiring tests for the conformance harness.

CPU-only. Verifies the three new tools are registered + their
handlers behave (return a structured response on a clean failure).
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestToolRegistration:
    def test_three_tools_registered(self) -> None:
        from compgen.mcp.tools import ALL_TOOLS, CONFORMANCE_TOOLS

        names = {t["name"] for t in CONFORMANCE_TOOLS}
        assert names == {
            "etc_conformance_run",
            "etc_conformance_summarize",
            "etc_megakernel_inspect",
        }
        # And they're all in the global ALL_TOOLS list.
        all_names = {t["name"] for t in ALL_TOOLS}
        assert names <= all_names

    def test_each_tool_has_required_descriptor_fields(self) -> None:
        from compgen.mcp.tools import CONFORMANCE_TOOLS

        for tool in CONFORMANCE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "phase" in tool
            assert callable(tool["handler"])
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestEtcConformanceRunHandler:
    def test_unknown_workload_lands_in_failed_list(self, tmp_path: Path) -> None:
        from compgen.mcp.tools.conformance import etc_conformance_run

        result = etc_conformance_run(
            workload="not_a_workload",
            output_dir=str(tmp_path),
        )
        assert result["status"] in {"fail", "partial"}
        assert "not_a_workload" in result["failed"]
        assert result["first_error"] is not None
        assert "unknown workload" in result["first_error"].lower()

    def test_diamond_on_cpu_returns_structured_failure(self, tmp_path: Path) -> None:
        """On a CPU host the harness reports the workload as failed
        with a clean reason — and the report file lands on disk."""
        import torch
        from compgen.mcp.tools.conformance import etc_conformance_run

        if torch.cuda.is_available():
            pytest.skip("This test exercises the CPU-fallback path")

        result = etc_conformance_run(
            workload="diamond_dag",
            output_dir=str(tmp_path),
        )
        assert result["status"] in {"fail", "partial"}
        assert "diamond_dag" in result["failed"]
        assert any("diamond_dag" in p for p in result["report_paths"])
        # Markdown summary has the table header (or "No conformance reports
        # found" if for some reason nothing landed — but it should land).
        assert "Workload" in result["summary_md"] or "No conformance reports" in result["summary_md"]

    def test_min_speedup_override_is_threaded_into_gate(self, tmp_path: Path) -> None:
        """Relax the gate to 0.5×; the resulting report's gate field
        must reflect the override."""
        import json

        from compgen.mcp.tools.conformance import etc_conformance_run

        etc_conformance_run(
            workload="diamond_dag",
            output_dir=str(tmp_path),
            min_speedup_vs_eager=0.5,
        )
        report = json.loads((tmp_path / "diamond_dag.conformance_report.json").read_text())
        assert report["gate"]["min_speedup_vs_eager"] == 0.5


class TestEtcConformanceSummarizeHandler:
    def test_missing_dir_is_reported_cleanly(self, tmp_path: Path) -> None:
        from compgen.mcp.tools.conformance import etc_conformance_summarize

        result = etc_conformance_summarize(str(tmp_path / "does_not_exist"))
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    def test_summary_after_one_run(self, tmp_path: Path) -> None:
        from compgen.mcp.tools.conformance import (
            etc_conformance_run,
            etc_conformance_summarize,
        )

        etc_conformance_run(workload="diamond_dag", output_dir=str(tmp_path))
        summary = etc_conformance_summarize(str(tmp_path))
        assert summary["n_total"] == 1
        # On CPU host, the run fails — but the summary still counts it.
        assert summary["n_pass"] + summary["n_fail"] == 1


class TestEtcMegakernelInspectHandler:
    def test_missing_bundle_dir_returns_error_field(self, tmp_path: Path) -> None:
        from compgen.mcp.tools.conformance import etc_megakernel_inspect

        result = etc_megakernel_inspect(str(tmp_path / "no_such_bundle"))
        assert result["manifest_present"] is False
        assert any("does not exist" in e for e in result["errors"])

    def test_bundle_without_megakernel_subdir(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text("{}")
        from compgen.mcp.tools.conformance import etc_megakernel_inspect

        result = etc_megakernel_inspect(str(bundle))
        assert result["manifest_present"] is False
        assert any("megakernel" in e for e in result["errors"])

    def test_bundle_with_manifest(self, tmp_path: Path) -> None:
        """Manifest is loaded; ptx_files list is populated."""
        bundle = tmp_path / "bundle"
        mk = bundle / "megakernel"
        mk.mkdir(parents=True)
        (mk / "manifest.yaml").write_text("graph_name: test\nsm_count: 188\nlaunch_config:\n  cooperative: true\n")
        # Two fake PTX files (no actual PTX content needed for this layer).
        (mk / "wrapper.ptx").write_text("// PTX placeholder\n")
        device_funcs = mk / "device_funcs"
        device_funcs.mkdir()
        (device_funcs / "matmul.ptx").write_text("// PTX placeholder\n")

        from compgen.mcp.tools.conformance import etc_megakernel_inspect

        result = etc_megakernel_inspect(str(bundle))
        assert result["manifest_present"] is True
        assert result["manifest"]["graph_name"] == "test"
        assert result["manifest"]["sm_count"] == 188
        assert len(result["ptx_files"]) == 2
        # tooling field always present (None when cuobjdump missing).
        assert "cuobjdump" in result["tooling"]
        assert "nvdisasm" in result["tooling"]
