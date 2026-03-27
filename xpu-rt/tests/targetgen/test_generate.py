"""Tests for the top-level target generator and family-specific stacks."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.stages.registry import StageRegistry
from compgen.targetgen.generate import GeneratedTarget, generate_target
from compgen.targetgen.verification_ladder import VerificationLevel
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


class TestGenerateTarget:
    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_generate_all_exemplars(self, yaml_file: Path, tmp_path: Path) -> None:
        """Every exemplar must generate successfully."""
        result = generate_target(yaml_file, tmp_path / yaml_file.stem)
        assert isinstance(result, GeneratedTarget)
        assert result.output_dir.exists()
        assert (result.output_dir / "classification.json").exists()
        assert (result.output_dir / "support_plan.json").exists()
        assert (result.output_dir / "verification_manifest.json").exists()

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_generated_stack_runs(self, yaml_file: Path, tmp_path: Path) -> None:
        """Generated TargetDialectStack must run on sample IR."""
        result = generate_target(yaml_file, tmp_path / yaml_file.stem)
        module = _make_module()

        registry = StageRegistry()
        result.dialect_stack.target_name = result.profile.name
        registry.register_target_stack(result.dialect_stack)

        pipeline_result = registry.run_pipeline(module, result.profile, result.capabilities)
        assert pipeline_result.passed, (
            f"{yaml_file.name}: Pipeline failed at {pipeline_result.first_failure}"
        )


class TestFamilyStacks:
    def test_gpu_stack_5_stages(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_gpu_simt.yaml", tmp_path / "gpu")
        assert len(result.dialect_stack.stages) == 5

    def test_rvv_stack_5_stages(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_rvv_cpu.yaml", tmp_path / "rvv")
        assert len(result.dialect_stack.stages) == 5

    def test_matrix_stack_6_stages(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_matrix_ext.yaml", tmp_path / "matrix")
        assert len(result.dialect_stack.stages) == 6

    def test_rocc_stack_7_stages(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_rocc_accel.yaml", tmp_path / "rocc")
        assert len(result.dialect_stack.stages) == 7

    def test_npu_stack_7_stages(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_npu_text_isa.yaml", tmp_path / "npu")
        assert len(result.dialect_stack.stages) == 7

    def test_variable_depth_across_families(self, tmp_path: Path) -> None:
        """Different families produce different stack depths."""
        depths = {}
        for yaml_file in sorted(EXEMPLAR_DIR.glob("*.yaml")):
            result = generate_target(yaml_file, tmp_path / yaml_file.stem)
            depths[yaml_file.stem] = len(result.dialect_stack.stages)
        # At least 2 distinct depths
        assert len(set(depths.values())) >= 2, f"All stacks same depth: {depths}"


class TestVerificationManifest:
    def test_manifest_has_base_tests(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_gpu_simt.yaml", tmp_path / "gpu")
        manifest = result.verification_manifest
        assert len(manifest.tests) > 10

    def test_manifest_has_plugin_tests(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_rocc_accel.yaml", tmp_path / "rocc")
        plugin_tests = result.verification_manifest.tests_at_level(
            VerificationLevel.L6_PLUGIN_PASSES
        )
        # RoCC has a plugin for accel_lowering
        assert len(plugin_tests) >= 1

    def test_manifest_levels(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_gpu_simt.yaml", tmp_path / "gpu")
        levels = result.verification_manifest.levels_with_tests()
        assert VerificationLevel.L0_SPEC_SANITY in levels
        assert VerificationLevel.L8_BUNDLE_VALID in levels

    def test_maturity_mapping(self, tmp_path: Path) -> None:
        result = generate_target(EXEMPLAR_DIR / "test_rocc_accel.yaml", tmp_path / "rocc")
        from compgen.targets.maturity import TargetMaturity
        # RoCC has simulator → can reach L9
        assert result.verification_manifest.maturity == TargetMaturity.L3_PROMOTED


class TestArtifactOutput:
    def test_classification_json(self, tmp_path: Path) -> None:
        import json
        result = generate_target(EXEMPLAR_DIR / "test_gpu_simt.yaml", tmp_path / "gpu")
        data = json.loads((result.output_dir / "classification.json").read_text())
        assert data["family"] == "simt_gpu_hal"
        assert data["confidence"] >= 0.9

    def test_support_plan_json(self, tmp_path: Path) -> None:
        import json
        result = generate_target(EXEMPLAR_DIR / "test_rocc_accel.yaml", tmp_path / "rocc")
        data = json.loads((result.output_dir / "support_plan.json").read_text())
        assert data["needs_accel_dialect"] is True
        assert len(data["required_stages"]) == 7
