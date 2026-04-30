"""Tests for the CUDA GPU target dialect stack.

Proves the full stage architecture works end-to-end:
  encoding → layout → dispatch → tiling → codegen → bundle
with CUDA-specific plugins at each stage.
"""

from __future__ import annotations

import pytest
import torch
from compgen.stages.dispatch.stage import DISPATCH_ID_ATTR
from compgen.stages.registry import StageRegistry
from compgen.stages.targets.cuda_gpu import (
    CudaCodegenPlugin,
    CudaDispatchPlugin,
    CudaEncodingPlugin,
    CudaTilingPlugin,
    create_cuda_gpu_stack,
)
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR
from compgen.targets.capability import infer_capabilities
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, c)
    block.add_op(mul)
    sub = arith.SubiOp(mul.result, a)
    block.add_op(sub)
    block.add_op(func.ReturnOp(sub.result))
    return ModuleOp([func.FuncOp("compute", ([idx, idx, idx], [idx]), Region([block]))])


def _make_matmul_module() -> ModuleOp:
    """Module from FX import with real matmul."""
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 16)

        def forward(self, x):
            return self.linear(x)

    ep = capture_model(Linear(), (torch.randn(4, 32),))
    module, _ = fx_to_xdsl(ep)
    return module


@pytest.fixture
def target():
    return load_profile("examples/target_profiles/cuda_a100.yaml")


@pytest.fixture
def capabilities(target):
    return infer_capabilities(target)


# ============================================================================
# Individual plugin tests
# ============================================================================


class TestCudaPlugins:
    def test_encoding_plugin_protocol(self) -> None:
        from compgen.stages.base import TargetStagePlugin

        plugin = CudaEncodingPlugin()
        assert isinstance(plugin, TargetStagePlugin)
        assert plugin.target_name == "cuda_gpu"
        assert plugin.stage_name == "encoding"

    def test_dispatch_plugin_protocol(self) -> None:
        from compgen.stages.base import TargetStagePlugin

        plugin = CudaDispatchPlugin()
        assert isinstance(plugin, TargetStagePlugin)

    def test_tiling_plugin_protocol(self) -> None:
        from compgen.stages.base import TargetStagePlugin

        plugin = CudaTilingPlugin()
        assert isinstance(plugin, TargetStagePlugin)

    def test_codegen_plugin_protocol(self) -> None:
        from compgen.stages.base import TargetStagePlugin

        plugin = CudaCodegenPlugin()
        assert isinstance(plugin, TargetStagePlugin)


# ============================================================================
# Stack creation
# ============================================================================


class TestCudaStack:
    def test_create_stack(self, tmp_path) -> None:
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        assert stack.target_name == "cuda_a100"
        assert len(stack.stages) == 6
        assert len(stack.plugins) == 5

    def test_stack_stage_names(self, tmp_path) -> None:
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        names = [s.name for s in stack.stages]
        assert names == ["encoding", "layout", "dispatch", "tiling", "codegen", "bundle"]

    def test_create_stack_requires_output_dir(self) -> None:
        """Factory must reject None output_dir — no volatile /tmp fallback."""
        import pytest

        with pytest.raises(ValueError, match="output_dir"):
            create_cuda_gpu_stack(output_dir=None)  # type: ignore[arg-type]


# ============================================================================
# Full pipeline E2E
# ============================================================================


class TestCudaPipeline:
    def test_full_pipeline_arith(self, target, capabilities, tmp_path) -> None:
        """Run full CUDA pipeline on arith-only module."""
        registry = StageRegistry()
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        # Override target name to match our profile
        stack.target_name = target.name
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_pipeline(module, target, capabilities)

        assert result.passed, f"Pipeline failed: {result.first_failure}"
        assert result.stages_run == 6

        # Check artifacts
        assert (tmp_path / "out" / "manifest.json").exists()
        assert (tmp_path / "out" / "payload.mlir").exists()

        # Check all stages left their marks
        for op in result.final_module.walk():
            if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
                continue
            if op.results:
                assert DISPATCH_ID_ATTR in op.attributes
                assert CODEGEN_BACKEND_ATTR in op.attributes

    def test_full_pipeline_matmul(self, target, capabilities, tmp_path) -> None:
        """Run full CUDA pipeline on FX-imported module with matmul."""
        registry = StageRegistry()
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        stack.target_name = target.name
        registry.register_target_stack(stack)

        module = _make_matmul_module()
        result = registry.run_pipeline(module, target, capabilities)

        violations = [r.contract_violations for r in result.stage_results]
        assert result.passed, f"Pipeline failed at {result.first_failure}: {violations}"
        assert result.stages_run == 6

    def test_pipeline_produces_per_stage_results(self, target, capabilities, tmp_path) -> None:
        """Each stage should produce a StageResult."""
        registry = StageRegistry()
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        stack.target_name = target.name
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_pipeline(module, target, capabilities)

        assert len(result.stage_results) == 6
        for sr in result.stage_results:
            assert sr.passed
            assert sr.stage_name in {"encoding", "layout", "dispatch", "tiling", "codegen", "bundle"}
            assert len(sr.diagnostics) > 0  # Each stage should log something

    def test_pipeline_collects_artifacts(self, target, capabilities, tmp_path) -> None:
        """Pipeline should aggregate artifacts from all stages."""
        registry = StageRegistry()
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "out"))
        stack.target_name = target.name
        registry.register_target_stack(stack)

        module = _make_module()
        result = registry.run_pipeline(module, target, capabilities)

        # CUDA plugins produce strategy artifacts
        assert "encoding_strategy" in result.all_artifacts
        assert "layout_strategy" in result.all_artifacts
        assert "fusion_strategy" in result.all_artifacts
        assert "tiling_strategy" in result.all_artifacts
        assert "codegen_strategy" in result.all_artifacts

    def test_variable_depth_vs_fixed(self, target, capabilities, tmp_path) -> None:
        """CUDA stack has 6 stages; a simpler target could have 3."""
        cuda_stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "cuda_out"))
        assert len(cuda_stack.stages) == 6

        # A minimal stack with just encoding + dispatch + bundle
        from compgen.stages.bundle import BundleStage
        from compgen.stages.dispatch import DispatchStage
        from compgen.stages.encoding import EncodingStage
        from compgen.stages.registry import TargetDialectStack

        mini_stack = TargetDialectStack(
            target_name=target.name,
            stages=[EncodingStage(), DispatchStage(), BundleStage(output_dir=tmp_path / "mini_out")],
        )
        assert len(mini_stack.stages) == 3

        # Both should work
        registry = StageRegistry()
        registry.register_target_stack(mini_stack)
        module = _make_module()
        result = registry.run_pipeline(module, target, capabilities)
        assert result.passed
        assert result.stages_run == 3
