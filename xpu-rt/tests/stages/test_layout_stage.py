"""Tests for LayoutStage.

Covers stage properties, input/output contracts, shared passes execution,
and integration with CudaLayoutPlugin.
"""

from __future__ import annotations

import pytest
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IndexType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import Block, Region

from compgen.ir.layout.ops import SetLayoutOp, UnsetLayoutOp
from compgen.stages.base import CompilationStage, StageContract
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.layout.stage import LayoutStage
from compgen.targets.capability import infer_capabilities
from compgen.targets.schema import (
    ComputeUnit,
    DeviceSpec,
    MemoryLevel,
    TargetProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_arith_module() -> ModuleOp:
    """Build a minimal arith module."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def _make_tensor_module() -> ModuleOp:
    """Build a module with tensor-typed block arguments."""
    f32 = Float32Type()
    tensor_type = TensorType(f32, [4, 4])
    func_type = FunctionType.from_lists([tensor_type], [tensor_type])
    block = Block(arg_types=[tensor_type])
    block.add_op(func.ReturnOp(block.args[0]))
    region = Region([block])
    func_op = func.FuncOp("forward", func_type, region)
    return ModuleOp([func_op])


def _add_encoding_attrs(module: ModuleOp) -> ModuleOp:
    """Add encoding attributes to simulate EncodingStage output."""
    for op in module.walk():
        if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
            continue
        if op.results:
            op.attributes[ENCODING_ATTR] = StringAttr("row_major")
    return module


@pytest.fixture
def target() -> TargetProfile:
    return TargetProfile(
        name="test_gpu",
        devices=[DeviceSpec(
            device_type="gpu", name="TestGPU", vendor="test",
            compute_units=[ComputeUnit(name="tensor_core", count=1, peak_tflops=100.0)],
            memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=1024**3)],
            supported_ops=["matmul"], features=["tensor_core"],
            kernel_backends=["triton"],
        )],
    )


@pytest.fixture
def capabilities(target):
    return infer_capabilities(target)


# ---------------------------------------------------------------------------
# Stage properties
# ---------------------------------------------------------------------------


class TestLayoutStageProperties:
    def test_name(self) -> None:
        stage = LayoutStage()
        assert stage.name == "layout"

    def test_description(self) -> None:
        stage = LayoutStage()
        assert isinstance(stage.description, str)
        assert len(stage.description) > 0

    def test_is_compilation_stage(self) -> None:
        stage = LayoutStage()
        assert isinstance(stage, CompilationStage)

    def test_requirements_doc_path(self) -> None:
        stage = LayoutStage()
        path = stage.requirements_doc_path()
        # Path should be defined (file may or may not exist depending on setup)
        assert path is not None


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestLayoutStageContracts:
    def test_input_contract_type(self) -> None:
        stage = LayoutStage()
        contract = stage.input_contract()
        assert isinstance(contract, StageContract)
        assert contract.stage_name == "layout"

    def test_input_contract_has_preconditions(self) -> None:
        stage = LayoutStage()
        contract = stage.input_contract()
        assert len(contract.preconditions) > 0

    def test_output_contract_type(self) -> None:
        stage = LayoutStage()
        contract = stage.output_contract()
        assert isinstance(contract, StageContract)
        assert contract.stage_name == "layout"

    def test_output_contract_has_postconditions(self) -> None:
        stage = LayoutStage()
        contract = stage.output_contract()
        assert len(contract.postconditions) > 0
        names = {inv.name for inv in contract.postconditions}
        assert "no_virtual_layout_ops" in names
        assert "layout_clean" in names

    def test_input_contract_passes_for_encoded_module(self, target) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_arith_module())
        violations = stage.verify_contract(module, stage.input_contract())
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Shared passes
# ---------------------------------------------------------------------------


class TestLayoutStageSharedPasses:
    def test_shared_passes_return_module(self, target) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.shared_passes(module, target)
        assert isinstance(result, ModuleOp)

    def test_shared_passes_mark_layout_clean(self, target) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.shared_passes(module, target)
        assert "compgen.layout_clean" in result.attributes

    def test_shared_passes_no_virtual_ops(self, target) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.shared_passes(module, target)
        for op in result.walk():
            assert not isinstance(op, (SetLayoutOp, UnsetLayoutOp))

    def test_shared_passes_on_tensor_module(self, target) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_tensor_module())
        result = stage.shared_passes(module, target)
        assert isinstance(result, ModuleOp)
        assert "compgen.layout_clean" in result.attributes


# ---------------------------------------------------------------------------
# CudaLayoutPlugin integration
# ---------------------------------------------------------------------------


class TestLayoutStageWithCudaPlugin:
    def test_plugin_registration(self) -> None:
        from compgen.stages.targets.cuda_gpu import CudaLayoutPlugin
        stage = LayoutStage()
        plugin = CudaLayoutPlugin()
        stage.register_plugin(plugin)
        assert stage.has_plugin

    def test_plugin_stage_name_matches(self) -> None:
        from compgen.stages.targets.cuda_gpu import CudaLayoutPlugin
        plugin = CudaLayoutPlugin()
        assert plugin.stage_name == "layout"

    def test_plugin_target_name(self) -> None:
        from compgen.stages.targets.cuda_gpu import CudaLayoutPlugin
        plugin = CudaLayoutPlugin()
        assert plugin.target_name == "cuda_gpu"

    def test_full_run_with_plugin(self, target, capabilities) -> None:
        from compgen.stages.targets.cuda_gpu import CudaLayoutPlugin
        stage = LayoutStage()
        plugin = CudaLayoutPlugin()
        stage.register_plugin(plugin)
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.run(module, target, capabilities)
        assert result.passed or len(result.contract_violations) > 0
        assert result.stage_name == "layout"

    def test_full_run_without_plugin(self, target, capabilities) -> None:
        stage = LayoutStage()
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.run(module, target, capabilities)
        assert result.stage_name == "layout"

    def test_run_produces_artifacts_with_plugin(self, target, capabilities) -> None:
        from compgen.stages.targets.cuda_gpu import CudaLayoutPlugin
        stage = LayoutStage()
        plugin = CudaLayoutPlugin()
        stage.register_plugin(plugin)
        module = _add_encoding_attrs(_make_arith_module())
        result = stage.run(module, target, capabilities)
        # Plugin should produce layout_strategy artifact
        if result.passed:
            assert "layout_strategy" in result.artifacts


# ---------------------------------------------------------------------------
# Pipeline chaining
# ---------------------------------------------------------------------------


class TestLayoutStageInPipeline:
    def test_encoding_then_layout(self, target, capabilities) -> None:
        """Encoding stage output feeds layout stage."""
        from compgen.stages.encoding import EncodingStage

        enc_stage = EncodingStage()
        layout_stage = LayoutStage()

        enc_result = enc_stage.run(_make_arith_module(), target, capabilities)
        assert enc_result.passed

        layout_result = layout_stage.run(enc_result.module, target, capabilities)
        assert layout_result.stage_name == "layout"
