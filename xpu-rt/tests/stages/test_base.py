"""Tests for the compilation stages framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from compgen.stages.base import (
    CompilationStage,
    IRInvariant,
    StageContract,
    StageResult,
    TargetStagePlugin,
)
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region

# -- Fixtures --

def _make_arith_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


class _DummyStage(CompilationStage):
    """Minimal concrete stage for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy stage for testing"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="dummy", preconditions=[])

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="dummy", postconditions=[])

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS.md"


class _DummyPlugin:
    """Minimal plugin for testing."""

    @property
    def target_name(self) -> str:
        return "test_target"

    @property
    def stage_name(self) -> str:
        return "dummy"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"test_artifact": "hello"}


# -- IRInvariant tests --

class TestIRInvariant:
    def test_invariant_with_required_ops(self) -> None:
        inv = IRInvariant(
            name="has_addi",
            description="Must have addi",
            required_ops=frozenset({"arith.addi"}),
        )
        assert inv.name == "has_addi"
        assert "arith.addi" in inv.required_ops

    def test_invariant_with_forbidden_ops(self) -> None:
        inv = IRInvariant(
            name="no_muli",
            description="Must not have muli",
            forbidden_ops=frozenset({"arith.muli"}),
        )
        assert "arith.muli" in inv.forbidden_ops

    def test_invariant_with_custom_check(self) -> None:
        inv = IRInvariant(
            name="custom",
            description="Custom check",
            custom_check=lambda m: True,
        )
        assert inv.custom_check is not None


# -- StageContract tests --

class TestStageContract:
    def test_empty_contract(self) -> None:
        c = StageContract(stage_name="test")
        assert c.preconditions == []
        assert c.postconditions == []

    def test_contract_with_invariants(self) -> None:
        inv = IRInvariant(name="test", description="test")
        c = StageContract(stage_name="test", preconditions=[inv])
        assert len(c.preconditions) == 1


# -- StageResult tests --

class TestStageResult:
    def test_passing_result(self) -> None:
        r = StageResult(stage_name="test", passed=True)
        assert r.passed
        assert r.contract_violations == []

    def test_failing_result(self) -> None:
        r = StageResult(
            stage_name="test",
            passed=False,
            contract_violations=["missing op"],
        )
        assert not r.passed


# -- CompilationStage tests --

class TestCompilationStage:
    def test_stage_properties(self) -> None:
        stage = _DummyStage()
        assert stage.name == "dummy"
        assert stage.description == "A dummy stage for testing"

    def test_run_without_plugin(self) -> None:
        stage = _DummyStage()
        module = _make_arith_module()
        from compgen.targets.schema import load_profile
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        from compgen.targets.capability import infer_capabilities
        caps = infer_capabilities(target)

        result = stage.run(module, target, caps)
        assert result.passed
        assert result.stage_name == "dummy"

    def test_run_with_plugin(self) -> None:
        stage = _DummyStage()
        plugin = _DummyPlugin()
        stage.register_plugin(plugin)
        assert stage.has_plugin

        module = _make_arith_module()
        from compgen.targets.schema import load_profile
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        from compgen.targets.capability import infer_capabilities
        caps = infer_capabilities(target)

        result = stage.run(module, target, caps)
        assert result.passed
        assert result.artifacts.get("test_artifact") == "hello"

    def test_plugin_name_mismatch_raises(self) -> None:
        stage = _DummyStage()

        class BadPlugin:
            @property
            def target_name(self) -> str:
                return "x"
            @property
            def stage_name(self) -> str:
                return "wrong_name"
            def configure(self, t, c): pass
            def transform(self, m): return m
            def get_artifacts(self): return {}

        with pytest.raises(ValueError, match="does not match"):
            stage.register_plugin(BadPlugin())

    def test_verify_contract_required_ops(self) -> None:
        stage = _DummyStage()
        module = _make_arith_module()
        contract = StageContract(
            stage_name="test",
            preconditions=[
                IRInvariant("has_addi", "needs addi", required_ops=frozenset({"arith.addi"})),
            ],
        )
        violations = stage.verify_contract(module, contract)
        assert violations == []

    def test_verify_contract_missing_ops(self) -> None:
        stage = _DummyStage()
        module = _make_arith_module()
        contract = StageContract(
            stage_name="test",
            preconditions=[
                IRInvariant("has_matmul", "needs matmul", required_ops=frozenset({"linalg.matmul"})),
            ],
        )
        violations = stage.verify_contract(module, contract)
        assert len(violations) == 1
        assert "linalg.matmul" in violations[0]

    def test_verify_contract_forbidden_ops(self) -> None:
        stage = _DummyStage()
        module = _make_arith_module()
        contract = StageContract(
            stage_name="test",
            preconditions=[
                IRInvariant("no_addi", "forbid addi", forbidden_ops=frozenset({"arith.addi"})),
            ],
        )
        violations = stage.verify_contract(module, contract)
        assert len(violations) == 1

    def test_verify_contract_custom_check(self) -> None:
        stage = _DummyStage()
        module = _make_arith_module()
        contract = StageContract(
            stage_name="test",
            preconditions=[
                IRInvariant("always_true", "always passes", custom_check=lambda m: True),
            ],
        )
        assert stage.verify_contract(module, contract) == []

        contract_fail = StageContract(
            stage_name="test",
            preconditions=[
                IRInvariant("always_false", "always fails", custom_check=lambda m: False),
            ],
        )
        violations = stage.verify_contract(module, contract_fail)
        assert len(violations) == 1

    def test_input_contract_violation_stops_stage(self) -> None:
        """If input contract fails, shared_passes should NOT run."""

        class StrictStage(_DummyStage):
            def input_contract(self) -> StageContract:
                return StageContract(
                    stage_name="strict",
                    preconditions=[
                        IRInvariant("needs_matmul", "needs matmul",
                                    required_ops=frozenset({"linalg.matmul"})),
                    ],
                )

        stage = StrictStage()
        module = _make_arith_module()
        from compgen.targets.schema import load_profile
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        from compgen.targets.capability import infer_capabilities
        caps = infer_capabilities(target)

        result = stage.run(module, target, caps)
        assert not result.passed
        assert any("INPUT" in v for v in result.contract_violations)


# -- TargetStagePlugin protocol --

class TestTargetStagePlugin:
    def test_plugin_satisfies_protocol(self) -> None:
        plugin = _DummyPlugin()
        assert isinstance(plugin, TargetStagePlugin)

    def test_plugin_properties(self) -> None:
        plugin = _DummyPlugin()
        assert plugin.target_name == "test_target"
        assert plugin.stage_name == "dummy"
