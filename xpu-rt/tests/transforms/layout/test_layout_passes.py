"""Integration tests for the layout transform passes.

Tests the full 10-pass layout pipeline and individual passes on
xDSL ModuleOp instances. Uses SimpleMLP capture when torch is
available, falls back to hand-built modules otherwise.
"""

from __future__ import annotations

import pytest
from compgen.ir.layout.ops import SetLayoutOp, UnsetLayoutOp
from compgen.stages.encoding.stage import ENCODING_ATTR
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

# ---------------------------------------------------------------------------
# Module builders
# ---------------------------------------------------------------------------


def _make_arith_module() -> ModuleOp:
    """Build a minimal arith-only module (no tensors)."""
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
    """Add encoding attributes to all ops (simulate EncodingStage)."""
    for op in module.walk():
        if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
            continue
        if op.results:
            op.attributes[ENCODING_ATTR] = StringAttr("row_major")
    return module


# ---------------------------------------------------------------------------
# Individual pass tests
# ---------------------------------------------------------------------------


class TestCanonicalizeTransposes:
    def test_no_crash_on_empty_module(self) -> None:
        from compgen.transforms.layout.canonicalize_transposes import canonicalize_transposes

        module = _make_arith_module()
        result = canonicalize_transposes(module)
        assert isinstance(result, ModuleOp)

    def test_no_crash_on_tensor_module(self) -> None:
        from compgen.transforms.layout.canonicalize_transposes import canonicalize_transposes

        module = _make_tensor_module()
        result = canonicalize_transposes(module)
        assert isinstance(result, ModuleOp)


class TestAttachLayoutHints:
    def test_no_crash_on_empty_plans(self) -> None:
        from compgen.transforms.layout.attach_layout_hints import attach_layout_hints

        module = _make_arith_module()
        result = attach_layout_hints(module, {})
        assert isinstance(result, ModuleOp)


class TestSetVirtualEncodings:
    def test_no_crash_on_arith_module(self) -> None:
        from compgen.transforms.layout.set_virtual_encodings import set_virtual_encodings

        module = _add_encoding_attrs(_make_arith_module())
        result = set_virtual_encodings(module)
        assert isinstance(result, ModuleOp)


class TestPropagateLayouts:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.propagate_layouts import propagate_layouts

        module = _make_arith_module()
        result = propagate_layouts(module)
        assert isinstance(result, ModuleOp)


class TestHoistLayoutOps:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.hoist_layout_ops import hoist_layout_ops

        module = _make_arith_module()
        result = hoist_layout_ops(module)
        assert isinstance(result, ModuleOp)


class TestFuseLayoutIntoProducers:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.fuse_layout_into_producers import fuse_layout_into_producers

        module = _make_arith_module()
        result = fuse_layout_into_producers(module)
        assert isinstance(result, ModuleOp)


class TestIntroducePrepacking:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.introduce_prepacking import introduce_prepacking

        module = _make_arith_module()
        result = introduce_prepacking(module)
        assert isinstance(result, ModuleOp)


class TestSpecializeLayouts:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.specialize_layouts import specialize_layouts

        module = _make_arith_module()
        result = specialize_layouts(module)
        assert isinstance(result, ModuleOp)


class TestMaterializeLayoutBoundaries:
    def test_no_crash(self) -> None:
        from compgen.transforms.layout.materialize_layout_boundaries import materialize_layout_boundaries

        module = _make_arith_module()
        result = materialize_layout_boundaries(module)
        assert isinstance(result, ModuleOp)


class TestCleanupLayoutArtifacts:
    def test_marks_module_clean(self) -> None:
        from compgen.transforms.layout.cleanup_layout_artifacts import cleanup_layout_artifacts

        module = _make_arith_module()
        result = cleanup_layout_artifacts(module)
        assert "compgen.layout_clean" in result.attributes

    def test_no_virtual_ops_remain(self) -> None:
        from compgen.transforms.layout.cleanup_layout_artifacts import cleanup_layout_artifacts

        module = _make_arith_module()
        result = cleanup_layout_artifacts(module)
        for op in result.walk():
            assert not isinstance(op, (SetLayoutOp, UnsetLayoutOp))


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestRunLayoutPipeline:
    def test_full_pipeline_arith_module(self) -> None:
        from compgen.transforms.layout import run_layout_pipeline

        module = _add_encoding_attrs(_make_arith_module())
        result = run_layout_pipeline(module)
        assert isinstance(result, ModuleOp)
        # cleanup pass should mark module as layout-clean
        assert "compgen.layout_clean" in result.attributes

    def test_no_virtual_layout_ops_remain(self) -> None:
        from compgen.transforms.layout import run_layout_pipeline

        module = _add_encoding_attrs(_make_arith_module())
        result = run_layout_pipeline(module)
        for op in result.walk():
            assert not isinstance(op, (SetLayoutOp, UnsetLayoutOp))

    def test_pipeline_with_tensor_module(self) -> None:
        from compgen.transforms.layout import run_layout_pipeline

        module = _add_encoding_attrs(_make_tensor_module())
        result = run_layout_pipeline(module)
        assert isinstance(result, ModuleOp)
        assert "compgen.layout_clean" in result.attributes

    def test_pipeline_with_plans(self) -> None:
        from compgen.analysis.layout.planner import LayoutPlan
        from compgen.transforms.layout import run_layout_pipeline

        plans = {
            "region_0": LayoutPlan(
                region_id="region_0",
                preferred_output_layout="tiled",
            ),
        }
        module = _add_encoding_attrs(_make_arith_module())
        result = run_layout_pipeline(module, plans=plans)
        assert isinstance(result, ModuleOp)

    def test_pipeline_with_cuda_resolver(self) -> None:
        from compgen.transforms.layout import run_layout_pipeline
        from compgen.transforms.layout.cuda_resolver import CudaLayoutResolver

        resolver = CudaLayoutResolver()
        module = _add_encoding_attrs(_make_arith_module())
        result = run_layout_pipeline(module, resolver=resolver)
        assert isinstance(result, ModuleOp)


# ---------------------------------------------------------------------------
# SimpleMLP capture integration (requires torch)
# ---------------------------------------------------------------------------


class TestLayoutPipelineWithCapture:
    """Integration tests using actual PyTorch model capture."""

    @pytest.fixture
    def captured_module(self):
        """Attempt to capture a SimpleMLP and import to xDSL."""
        try:
            import torch
            import torch.nn as nn
            from compgen.ir.import_fx import import_fx_to_xdsl

            class SimpleMLP(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc1 = nn.Linear(32, 64)
                    self.fc2 = nn.Linear(64, 16)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.fc2(torch.relu(self.fc1(x)))

            model = SimpleMLP()
            example_input = torch.randn(1, 32)
            exported = torch.export.export(model, (example_input,))
            module = import_fx_to_xdsl(exported)
            return module
        except Exception:
            pytest.skip("torch.export or xDSL import not available")

    def test_encoding_then_pipeline(self, captured_module) -> None:
        from compgen.stages.encoding.stage import EncodingStage
        from compgen.targets.schema import ComputeUnit, DeviceSpec, MemoryLevel, TargetProfile
        from compgen.transforms.layout import run_layout_pipeline

        target = TargetProfile(
            name="test_gpu",
            devices=[
                DeviceSpec(
                    device_type="gpu",
                    name="TestGPU",
                    vendor="test",
                    compute_units=[ComputeUnit(name="tensor_core", count=1, peak_tflops=100.0)],
                    memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=1024**3)],
                )
            ],
        )
        stage = EncodingStage()
        encoded = stage.shared_passes(captured_module, target)
        result = run_layout_pipeline(encoded)
        assert isinstance(result, ModuleOp)
        assert "compgen.layout_clean" in result.attributes

        # Verify no virtual layout ops remain
        for op in result.walk():
            assert not isinstance(op, (SetLayoutOp, UnsetLayoutOp))
