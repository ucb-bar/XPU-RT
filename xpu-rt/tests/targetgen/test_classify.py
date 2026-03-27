"""Tests for the 5-family decision engine."""

from __future__ import annotations

from pathlib import Path

from compgen.targetgen.classify import (
    IntegrationStyle,
    LoweringSurface,
    TargetFamily,
    classify_hardware,
)
from compgen.targetgen.load import load_hardware_spec
from compgen.targetgen.plan import generate_support_plan

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


class TestClassifyFamilies:
    def test_rvv_cpu(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rvv_cpu.yaml")
        c = classify_hardware(spec)
        assert c.family == TargetFamily.RVV_CPU_EXTENSION
        assert c.integration_style == IntegrationStyle.LLVM_BACKEND
        assert c.lowering_surface == LoweringSurface.LLVM_IR
        assert c.confidence >= 0.9

    def test_matrix_ext(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_matrix_ext.yaml")
        c = classify_hardware(spec)
        assert c.family == TargetFamily.RISCV_VENDOR_MATRIX
        assert c.integration_style == IntegrationStyle.CUSTOM_LOWERING

    def test_rocc_accel(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        c = classify_hardware(spec)
        assert c.family == TargetFamily.ROCC_ACCELERATOR
        assert c.lowering_surface == LoweringSurface.CUSTOM_DIALECT

    def test_npu_text_isa(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        c = classify_hardware(spec)
        assert c.family == TargetFamily.STRUCTURED_NPU_TEXT_ISA
        assert c.lowering_surface == LoweringSurface.UKERNEL_CALLS

    def test_gpu_simt(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        c = classify_hardware(spec)
        assert c.family == TargetFamily.SIMT_GPU_HAL
        assert c.integration_style == IntegrationStyle.HAL_DRIVER
        assert c.lowering_surface == LoweringSurface.TRITON_IR

    def test_classification_is_deterministic(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        c1 = classify_hardware(spec)
        c2 = classify_hardware(spec)
        assert c1 == c2

    def test_all_classifications_have_reasoning(self) -> None:
        for yaml_file in EXEMPLAR_DIR.glob("*.yaml"):
            spec = load_hardware_spec(yaml_file)
            c = classify_hardware(spec)
            assert c.reasoning, f"{yaml_file.name} has empty reasoning"


class TestSupportPlan:
    def test_gpu_plan_has_5_stages(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_gpu_simt.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        assert len(plan.required_stages) == 5
        names = [s.stage_name for s in plan.required_stages]
        assert names == ["encoding", "dispatch", "tiling", "codegen", "bundle"]

    def test_rocc_plan_has_7_stages(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        assert len(plan.required_stages) == 7
        assert plan.needs_accel_dialect

    def test_npu_plan_has_7_stages(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_npu_text_isa.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        assert len(plan.required_stages) == 7
        assert plan.needs_ukernel_dialect

    def test_matrix_plan_has_6_stages(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_matrix_ext.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        assert len(plan.required_stages) == 6
        stage_names = [s.stage_name for s in plan.required_stages]
        assert "matrix_lowering" in stage_names

    def test_plan_records_dialects_from_spec(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_rocc_accel.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        assert "test_accel" in plan.required_dialects

    def test_plan_detects_llvm_patches(self) -> None:
        spec = load_hardware_spec(EXEMPLAR_DIR / "test_matrix_ext.yaml")
        c = classify_hardware(spec)
        plan = generate_support_plan(spec, c)
        # Matrix ext with custom instructions needs LLVM patches
        assert plan.llvm_patches_needed

    def test_plan_kernel_backends(self) -> None:
        for yaml_file in EXEMPLAR_DIR.glob("*.yaml"):
            spec = load_hardware_spec(yaml_file)
            c = classify_hardware(spec)
            plan = generate_support_plan(spec, c)
            assert plan.kernel_backend in {"triton", "llvm", "accel", "ukernel"}
