"""Tests for the generalizable quantized model pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.quantization.pipeline import PipelineReport, QuantizedModelPipeline

torchao = pytest.importorskip("torchao")
yaml = pytest.importorskip("yaml")


def _simple_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).eval()


class TestPipelineNoQuant:
    """Test pipeline without quantization (analysis only)."""

    def test_run_produces_report(self) -> None:
        model = _simple_mlp()
        x = (torch.randn(4, 32),)
        pipeline = QuantizedModelPipeline(model=model, sample_inputs=x, model_name="test_mlp")
        report = pipeline.run()
        assert isinstance(report, PipelineReport)
        assert report.model_name == "test_mlp"
        assert report.param_count > 0
        assert report.quantization_applied is False

    def test_capture_produces_graphs(self) -> None:
        model = _simple_mlp()
        x = (torch.randn(4, 32),)
        pipeline = QuantizedModelPipeline(model=model, sample_inputs=x)
        report = pipeline.run()
        assert report.capture_artifact is not None
        assert report.capture_artifact.graph_count >= 1

    def test_graph_analysis_populated(self) -> None:
        model = _simple_mlp()
        x = (torch.randn(4, 32),)
        pipeline = QuantizedModelPipeline(model=model, sample_inputs=x)
        report = pipeline.run()
        assert report.graph_analysis is not None
        assert report.graph_analysis.total_ops > 0

    def test_timings_recorded(self) -> None:
        model = _simple_mlp()
        x = (torch.randn(4, 32),)
        pipeline = QuantizedModelPipeline(model=model, sample_inputs=x)
        report = pipeline.run()
        assert "capture" in report.timings
        assert "total" in report.timings

    def test_summary_string(self) -> None:
        model = _simple_mlp()
        x = (torch.randn(4, 32),)
        pipeline = QuantizedModelPipeline(model=model, sample_inputs=x)
        report = pipeline.run()
        s = report.summary()
        assert "test" not in s or "model" in s  # Default name
        assert "params=" in s


class TestPipelineWithQuant:
    """Test pipeline with FP8 quantization."""

    def test_quantization_applied(self) -> None:
        from compgen.capture.torchao_pipeline import QuantizationConfig

        model = _simple_mlp().to(torch.bfloat16)
        x = (torch.randn(4, 32, dtype=torch.bfloat16),)
        config = QuantizationConfig(scheme="fp8_e4m3_po2")
        pipeline = QuantizedModelPipeline(
            model=model,
            sample_inputs=x,
            quant_config=config,
        )
        report = pipeline.run()
        assert report.quantization_applied is True

    def test_alignment_checked(self) -> None:
        from compgen.capture.torchao_pipeline import QuantizationConfig

        model = _simple_mlp().to(torch.bfloat16)
        x = (torch.randn(4, 32, dtype=torch.bfloat16),)
        config = QuantizationConfig(scheme="fp8_e4m3_po2")
        pipeline = QuantizedModelPipeline(
            model=model,
            sample_inputs=x,
            quant_config=config,
        )
        report = pipeline.run()
        assert report.alignment_result is not None
        assert report.alignment_result.passed is True


class TestPipelineArtifacts:
    """Test artifact output in standard contract format."""

    def test_artifacts_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _simple_mlp()
            x = (torch.randn(4, 32),)
            pipeline = QuantizedModelPipeline(
                model=model,
                sample_inputs=x,
                model_name="test_artifacts",
                output_dir=tmpdir,
            )
            report = pipeline.run()
            out = Path(tmpdir)

            assert (out / "golden_inputs.pt").exists()
            assert (out / "graph_analysis.json").exists()
            assert (out / "verification_report.json").exists()
            assert (out / "manifest.json").exists()

    def test_graph_analysis_json_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _simple_mlp()
            x = (torch.randn(4, 32),)
            pipeline = QuantizedModelPipeline(
                model=model,
                sample_inputs=x,
                output_dir=tmpdir,
            )
            pipeline.run()
            data = json.loads((Path(tmpdir) / "graph_analysis.json").read_text())
            assert "total_ops" in data
            assert "coverage_pct" in data

    def test_manifest_lists_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _simple_mlp()
            x = (torch.randn(4, 32),)
            pipeline = QuantizedModelPipeline(
                model=model,
                sample_inputs=x,
                output_dir=tmpdir,
            )
            pipeline.run()
            manifest = json.loads((Path(tmpdir) / "manifest.json").read_text())
            assert "artifacts" in manifest
            assert len(manifest["artifacts"]) > 0

    def test_quantized_artifacts(self) -> None:
        from compgen.capture.torchao_pipeline import QuantizationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            model = _simple_mlp().to(torch.bfloat16)
            x = (torch.randn(4, 32, dtype=torch.bfloat16),)
            config = QuantizationConfig(scheme="fp8_e4m3_po2")
            pipeline = QuantizedModelPipeline(
                model=model,
                sample_inputs=x,
                quant_config=config,
                output_dir=tmpdir,
            )
            report = pipeline.run()
            out = Path(tmpdir)

            assert (out / "alignment_report.json").exists()
            alignment = json.loads((out / "alignment_report.json").read_text())
            assert alignment["passed"] is True

            verification = json.loads((out / "verification_report.json").read_text())
            assert verification["quantization_applied"] is True
