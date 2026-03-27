"""Tests for the stage registry and pipeline runner."""

from __future__ import annotations

from pathlib import Path

from compgen.stages.base import CompilationStage, IRInvariant, StageContract
from compgen.stages.registry import StageRegistry, TargetDialectStack
from compgen.targets.capability import CapabilitySpec, infer_capabilities
from compgen.targets.schema import TargetProfile, load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


class _StageA(CompilationStage):
    @property
    def name(self) -> str:
        return "stage_a"

    @property
    def description(self) -> str:
        return "First stage"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="stage_a")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="stage_a")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        return module

    def requirements_doc_path(self) -> Path:
        return Path("/dev/null")


class _StageB(CompilationStage):
    @property
    def name(self) -> str:
        return "stage_b"

    @property
    def description(self) -> str:
        return "Second stage"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="stage_b")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="stage_b")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        return module

    def requirements_doc_path(self) -> Path:
        return Path("/dev/null")


class _FailingStage(CompilationStage):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "Always fails"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="failing")

    def output_contract(self) -> StageContract:
        return StageContract(
            stage_name="failing",
            postconditions=[
                IRInvariant("impossible", "requires matmul",
                            required_ops=frozenset({"linalg.matmul"})),
            ],
        )

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        return module

    def requirements_doc_path(self) -> Path:
        return Path("/dev/null")


def _get_target_and_caps() -> tuple[TargetProfile, CapabilitySpec]:
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    caps = infer_capabilities(target)
    return target, caps


class TestTargetDialectStack:
    def test_stack_creation(self) -> None:
        stack = TargetDialectStack(
            target_name="test_gpu",
            stages=[_StageA(), _StageB()],
        )
        assert stack.target_name == "test_gpu"
        assert len(stack.stages) == 2

    def test_stack_bind_plugins(self) -> None:
        class PluginA:
            @property
            def target_name(self) -> str:
                return "test"
            @property
            def stage_name(self) -> str:
                return "stage_a"
            def configure(self, t, c): pass
            def transform(self, m): return m
            def get_artifacts(self): return {}

        stack = TargetDialectStack(
            target_name="test",
            stages=[_StageA(), _StageB()],
            plugins={"stage_a": PluginA()},
        )
        stack.bind_plugins()
        assert stack.stages[0].has_plugin
        assert not stack.stages[1].has_plugin


class TestStageRegistry:
    def test_register_shared_stage(self) -> None:
        registry = StageRegistry()
        registry.register_shared_stage(_StageA())
        assert registry.get_shared_stage("stage_a") is not None
        assert registry.get_shared_stage("nonexistent") is None

    def test_register_target_stack(self) -> None:
        registry = StageRegistry()
        stack = TargetDialectStack(target_name="gpu", stages=[_StageA()])
        registry.register_target_stack(stack)
        assert registry.get_target_stack("gpu") is not None
        assert "gpu" in registry.list_targets()

    def test_run_pipeline_success(self) -> None:
        registry = StageRegistry()
        target, caps = _get_target_and_caps()
        stack = TargetDialectStack(
            target_name=target.name,
            stages=[_StageA(), _StageB()],
        )
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_pipeline(module, target, caps)
        assert result.passed
        assert result.stages_run == 2
        assert len(result.stage_results) == 2

    def test_run_pipeline_stops_on_failure(self) -> None:
        registry = StageRegistry()
        target, caps = _get_target_and_caps()
        stack = TargetDialectStack(
            target_name=target.name,
            stages=[_StageA(), _FailingStage(), _StageB()],
        )
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_pipeline(module, target, caps)
        assert not result.passed
        assert result.first_failure == "failing"
        assert result.stages_run == 2  # A ran, failing ran (failed), B didn't run

    def test_run_pipeline_no_stack(self) -> None:
        registry = StageRegistry()
        target, caps = _get_target_and_caps()
        module = _make_module()
        result = registry.run_pipeline(module, target, caps)
        assert not result.passed

    def test_run_single_stage(self) -> None:
        registry = StageRegistry()
        target, caps = _get_target_and_caps()
        stack = TargetDialectStack(
            target_name=target.name,
            stages=[_StageA(), _StageB()],
        )
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_single_stage("stage_a", module, target, caps)
        assert result.passed

    def test_run_single_stage_missing(self) -> None:
        registry = StageRegistry()
        target, caps = _get_target_and_caps()
        module = _make_module()
        result = registry.run_single_stage("nonexistent", module, target, caps)
        assert not result.passed

    def test_variable_depth_stacks(self) -> None:
        """Different targets can have different numbers of stages."""
        registry = StageRegistry()
        target, caps = _get_target_and_caps()

        # Short stack (3 stages)
        short_stack = TargetDialectStack(
            target_name=target.name,
            stages=[_StageA(), _StageB()],
        )

        # Simulate a long stack by reusing stages (in practice different stages)
        long_target_name = "long_target"
        long_stack = TargetDialectStack(
            target_name=long_target_name,
            stages=[_StageA(), _StageB(), _StageA(), _StageB()],
        )

        registry.register_target_stack(short_stack)
        registry.register_target_stack(long_stack)

        module = _make_module()
        short_result = registry.run_pipeline(module.clone(), target, caps)
        assert short_result.stages_run == 2

        # Create a fake target for the long stack
        from compgen.targets.schema import DeviceSpec, TargetProfile
        long_target = TargetProfile(
            name=long_target_name,
            devices=[DeviceSpec(device_type="cpu", name="d0")],
            interconnects=[],
            constraints={},
        )
        long_caps = infer_capabilities(long_target)
        long_result = registry.run_pipeline(module.clone(), long_target, long_caps)
        assert long_result.stages_run == 4
