"""Tests for YAML loading and TargetProfile extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.targetgen.hardware_spec import ExecutionModel
from compgen.targetgen.load import extract_target_profile, load_hardware_spec, load_spec_with_profile
from compgen.targetgen.validate_spec import validate_hardware_spec

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


class TestLoadExemplars:
    """Each exemplar YAML must load, validate, and produce a TargetProfile."""

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_load_exemplar(self, yaml_file: Path) -> None:
        spec = load_hardware_spec(yaml_file)
        assert spec.name
        assert spec.schema_version == "2.0"

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_validate_exemplar(self, yaml_file: Path) -> None:
        spec = load_hardware_spec(yaml_file)
        result = validate_hardware_spec(spec)
        assert result.valid, f"{yaml_file.name}: {[e.message for e in result.errors]}"

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_extract_profile(self, yaml_file: Path) -> None:
        spec = load_hardware_spec(yaml_file)
        profile = extract_target_profile(spec)
        assert profile.name == spec.name
        assert len(profile.devices) >= 1
        assert profile.devices[0].device_type in {"cpu", "gpu", "accelerator", "npu"}


class TestLoadSpecific:
    def test_rvv_cpu_execution_model(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rvv_cpu.yaml")
        assert spec.execution_model.model == ExecutionModel.SIMD_VECTOR

    def test_rocc_execution_model(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        assert spec.execution_model.model == ExecutionModel.ROCC_COPROCESSOR

    def test_gpu_execution_model(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        assert spec.execution_model.model == ExecutionModel.SIMT_GPU

    def test_npu_execution_model(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        assert spec.execution_model.model == ExecutionModel.TEXT_ISA_NPU

    def test_matrix_execution_model(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_matrix_ext.yaml")
        assert spec.execution_model.model == ExecutionModel.DECOUPLED_MATRIX


class TestExtractProfile:
    def test_gpu_maps_to_gpu_device_type(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        profile = extract_target_profile(spec)
        assert profile.devices[0].device_type == "gpu"

    def test_rocc_maps_to_accelerator(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        profile = extract_target_profile(spec)
        assert profile.devices[0].device_type == "accelerator"

    def test_npu_maps_to_npu(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        profile = extract_target_profile(spec)
        assert profile.devices[0].device_type == "npu"

    def test_rvv_maps_to_cpu(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rvv_cpu.yaml")
        profile = extract_target_profile(spec)
        assert profile.devices[0].device_type == "cpu"

    def test_supported_ops_from_native_families(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        profile = extract_target_profile(spec)
        assert "matmul" in profile.devices[0].supported_ops

    def test_load_spec_with_profile_convenience(self) -> None:
        spec, profile = load_spec_with_profile(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        assert spec.name == profile.name
