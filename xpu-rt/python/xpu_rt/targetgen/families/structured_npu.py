"""Structured NPU text ISA family stack generator.

Produces: encoding → dispatch → kernel_contract → isa_lowering → memory_plan → scheduling → bundle
7 stages — multi-dialect progressive lowering like Merlin's NPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from xpu_rt.stages.bundle import BundleStage
from xpu_rt.stages.dispatch import DispatchStage
from xpu_rt.stages.encoding import EncodingStage
from xpu_rt.stages.registry import TargetDialectStack
from xpu_rt.stages.templates.lowering import LoweringStage
from xpu_rt.stages.templates.memory_plan import MemoryPlanStage
from xpu_rt.stages.templates.scheduling import SchedulingStage
from xpu_rt.targetgen.hardware_spec import HardwareSpec
from xpu_rt.targets.capability import CapabilitySpec
from xpu_rt.targets.schema import TargetProfile


class NpuKernelPlugin:
    """Lower dispatch groups to NPU kernel contracts."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "kernel_contract"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and "xpu_rt.npu_kernel" not in op.attributes:
                op.attributes["xpu_rt.npu_kernel"] = StringAttr("ukernel_call")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"npu_kernel_lowering": "ukernel_contracts"}


class NpuIsaPlugin:
    """Lower kernel contracts to NPU text ISA."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "isa_lowering"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and "xpu_rt.npu_isa" not in op.attributes:
                op.attributes["xpu_rt.npu_isa"] = StringAttr("text_isa")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        instrs = list(self._spec.isa.custom_instructions.keys())
        return {"npu_isa_lowering": "text_isa", "instructions": instrs}


def create_npu_stack(spec: HardwareSpec, output_dir: str | Path) -> TargetDialectStack:
    """Create structured NPU pipeline (7 stages, multi-dialect lowering).

    ``output_dir`` is mandatory; see :func:`create_cuda_gpu_stack`.
    """
    if output_dir is None:
        raise ValueError("create_npu_stack requires output_dir; pass a session-scoped path")
    bundle_dir = Path(output_dir)
    return TargetDialectStack(
        target_name=spec.name,
        stages=[
            EncodingStage(),
            DispatchStage(),
            LoweringStage("kernel_contract", "Kernel contract lowering"),
            LoweringStage("isa_lowering", "ISA text lowering"),
            MemoryPlanStage(),
            SchedulingStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={
            "kernel_contract": NpuKernelPlugin(spec),
            "isa_lowering": NpuIsaPlugin(spec),
        },
    )
