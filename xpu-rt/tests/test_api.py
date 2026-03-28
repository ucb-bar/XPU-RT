"""Tests for the top-level CompGen Python API (device / compile_model)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.api import CompGenDevice, CompiledModel, compile_model, device
from compgen.runtime.local_executor import BenchmarkResult
from compgen.stages.registry import PipelineResult, TargetDialectStack
from compgen.targetgen.generate import GeneratedTarget
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile

# Use the existing exemplar YAML files that ``test_generate.py`` also exercises.
EXEMPLAR_DIR = Path(__file__).parent / "targetgen" / "exemplars"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Minimal model for capture tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# device()
# ---------------------------------------------------------------------------


class TestDevice:
    """Tests for ``compgen.device()``."""

    def test_device_returns_compgen_device(self, tmp_path: Path) -> None:
        """``device()`` must return a CompGenDevice."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert isinstance(dev, CompGenDevice)

    def test_device_loads_spec(self, tmp_path: Path) -> None:
        """The returned device must carry the loaded HardwareSpec."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert dev.spec.name == "test-gpu-simt"

    def test_device_has_profile(self, tmp_path: Path) -> None:
        """The returned device must have a TargetProfile."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert isinstance(dev.profile, TargetProfile)
        assert dev.profile.name == "test-gpu-simt"

    def test_device_has_capabilities(self, tmp_path: Path) -> None:
        """The returned device must have a CapabilitySpec."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert isinstance(dev.capabilities, CapabilitySpec)

    def test_device_has_dialect_stack(self, tmp_path: Path) -> None:
        """The returned device must carry a TargetDialectStack."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert isinstance(dev.dialect_stack, TargetDialectStack)
        assert len(dev.dialect_stack.stages) > 0

    def test_device_has_generated_target(self, tmp_path: Path) -> None:
        """The GeneratedTarget must be accessible."""
        spec_path = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dev = device(spec_path, output_dir=tmp_path / "out")
        assert isinstance(dev.generated_target, GeneratedTarget)
        assert dev.generated_target.output_dir.exists()

    def test_device_file_not_found(self, tmp_path: Path) -> None:
        """``device()`` must raise FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            device(tmp_path / "nonexistent.yaml")

    def test_device_default_output_dir(self, tmp_path: Path) -> None:
        """When output_dir is omitted, artifacts go next to the spec."""
        import shutil

        # Copy exemplar into tmp_path so default output doesn't pollute the source tree.
        src = EXEMPLAR_DIR / "test_gpu_simt.yaml"
        dest = tmp_path / "test_gpu_simt.yaml"
        shutil.copy2(src, dest)

        dev = device(dest)
        expected_parent = tmp_path / "compgen_output"
        assert dev.generated_target.output_dir.parent == expected_parent

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_device_all_exemplars(self, yaml_file: Path, tmp_path: Path) -> None:
        """Every exemplar YAML must produce a valid CompGenDevice."""
        dev = device(yaml_file, output_dir=tmp_path / yaml_file.stem)
        assert isinstance(dev, CompGenDevice)
        assert len(dev.dialect_stack.stages) > 0


# ---------------------------------------------------------------------------
# compile_model()
# ---------------------------------------------------------------------------


class TestCompileModel:
    """Tests for ``compgen.compile_model()``."""

    def test_compile_returns_compiled_model(self, tmp_path: Path) -> None:
        """``compile_model()`` must return a CompiledModel."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        model = _TinyMLP()
        compiled = compile_model(model, dev)
        assert isinstance(compiled, CompiledModel)

    def test_compile_preserves_model(self, tmp_path: Path) -> None:
        """The compiled result must hold the original model."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        model = _TinyMLP()
        compiled = compile_model(model, dev)
        assert compiled.model is model

    def test_compile_preserves_device(self, tmp_path: Path) -> None:
        """The compiled result must hold the target device."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        model = _TinyMLP()
        compiled = compile_model(model, dev)
        assert compiled.device is dev

    def test_compile_default_objective(self, tmp_path: Path) -> None:
        """Default objective must be 'latency'."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert compiled.objective == "latency"

    def test_compile_custom_objective(self, tmp_path: Path) -> None:
        """A custom objective must be propagated."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev, objective="throughput")
        assert compiled.objective == "throughput"

    def test_compile_has_pipeline_result(self, tmp_path: Path) -> None:
        """PipelineResult must be present."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert isinstance(compiled.pipeline_result, PipelineResult)

    def test_compile_has_eqsat_result(self, tmp_path: Path) -> None:
        """EqSatResult must be present."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert compiled.eqsat_result is not None

    def test_compile_has_import_diagnostics(self, tmp_path: Path) -> None:
        """Import diagnostics list must be present."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert isinstance(compiled.import_diagnostics, list)

    def test_compile_has_capture_artifact(self, tmp_path: Path) -> None:
        """The strict frontend capture artifact must be attached to the result."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert compiled.capture_artifact.validation.valid
        assert "torch" in compiled.capture_artifact.runtime_versions

    def test_compile_has_analysis_dossier(self, tmp_path: Path) -> None:
        """Compilation should produce a graph-analysis dossier before eqsat."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        assert compiled.analysis_dossier is not None
        assert compiled.analysis_dossier.total_regions >= 1

    def test_compile_with_explicit_sample_inputs(self, tmp_path: Path) -> None:
        """Explicit sample_inputs must be used instead of the default."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        inputs = (torch.randn(2, 64),)
        compiled = compile_model(_TinyMLP(), dev, sample_inputs=inputs)
        assert isinstance(compiled, CompiledModel)


# ---------------------------------------------------------------------------
# CompiledModel.__call__()
# ---------------------------------------------------------------------------


class TestCompiledModelCall:
    """Tests for benchmarking via ``CompiledModel.__call__``."""

    def test_call_returns_benchmark_result(self, tmp_path: Path) -> None:
        """Calling a CompiledModel must return a BenchmarkResult."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        result = compiled(torch.randn(1, 64), num_iterations=5, warmup=1)
        assert isinstance(result, BenchmarkResult)

    def test_call_measures_latency(self, tmp_path: Path) -> None:
        """The benchmark must produce a non-zero latency measurement."""
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "out")
        compiled = compile_model(_TinyMLP(), dev)
        result = compiled(torch.randn(1, 64), num_iterations=5, warmup=1)
        assert result.latency_median_us > 0
        assert result.num_iterations == 5
        assert result.warmup_iterations == 1


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify that the top-level ``compgen`` package re-exports the API."""

    def test_import_device_from_package(self) -> None:
        from compgen import device as dev_fn
        assert callable(dev_fn)

    def test_import_compile_model_from_package(self) -> None:
        from compgen import compile_model as cm_fn
        assert callable(cm_fn)

    def test_import_classes_from_package(self) -> None:
        from compgen import CompGenDevice, CompiledModel
        assert CompGenDevice is not None
        assert CompiledModel is not None
