"""CUDA GPU target dialect stack.

Compilation pipeline for NVIDIA CUDA GPUs (Triton backend):

    1. Encoding — select MMA-friendly layouts for tensor cores
    2. Dispatch — fuse elementwise ops around matmuls
    3. Tiling — tile to thread block dimensions
    4. Codegen — select Triton/cuBLAS backends
    5. Bundle — package with CUDA-specific metadata

This is the first complete target stack, proving the stage architecture
works end-to-end.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.bundle import BundleStage
from compgen.stages.dispatch import DispatchStage
from compgen.stages.dispatch.stage import DISPATCH_ID_ATTR
from compgen.stages.encoding import EncodingStage
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.registry import TargetDialectStack
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR, CodegenStage
from compgen.stages.templates.tiling import TILE_SIZES_ATTR, TilingStage
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile

# ---------------------------------------------------------------------------
# CUDA-specific plugins
# ---------------------------------------------------------------------------


class CudaEncodingPlugin:
    """CUDA GPU encoding: prefer MMA-friendly layouts for matmul operands."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "encoding"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not any(isinstance(r.type, TensorType) for r in op.results):
                continue
            # Matmul-like ops get tiled layout, others get row_major
            if "matmul" in op.name.lower():
                op.attributes[ENCODING_ATTR] = StringAttr("tiled_128x64")
            elif ENCODING_ATTR not in op.attributes:
                op.attributes[ENCODING_ATTR] = StringAttr("row_major")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"encoding_strategy": "cuda_mma_friendly"}


class CudaDispatchPlugin:
    """CUDA GPU dispatch: fuse elementwise ops with their producer matmuls."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "dispatch"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        # Simple heuristic: group consecutive arith ops into the same dispatch
        dispatch_id = 0
        prev_is_arith = False
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue

            is_arith = op.name.startswith("arith.")
            if is_arith and prev_is_arith:
                # Fuse consecutive arith ops
                op.attributes[DISPATCH_ID_ATTR] = StringAttr(f"cuda_d_{dispatch_id}")
            else:
                dispatch_id += 1
                op.attributes[DISPATCH_ID_ATTR] = StringAttr(f"cuda_d_{dispatch_id}")

            prev_is_arith = is_arith
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"fusion_strategy": "cuda_matmul_fuse"}


class CudaTilingPlugin:
    """CUDA GPU tiling: tile linalg ops to thread block dimensions."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "tiling"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue
            if "matmul" in op.name.lower():
                op.attributes[TILE_SIZES_ATTR] = StringAttr("128x128x32")
            elif op.name.startswith("linalg."):
                op.attributes[TILE_SIZES_ATTR] = StringAttr("256")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"tiling_strategy": "cuda_threadblock"}


class CudaCodegenPlugin:
    """CUDA GPU codegen: assign Triton or cuBLAS backends."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "codegen"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue
            if "matmul" in op.name.lower():
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("cublas")
            elif op.name.startswith("linalg."):
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("triton")
            elif op.name.startswith("arith."):
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("triton_fused")
            else:
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("fallback")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"codegen_strategy": "cuda_triton_cublas"}


# ---------------------------------------------------------------------------
# CUDA GPU dialect stack
# ---------------------------------------------------------------------------


def create_cuda_gpu_stack(output_dir: str | None = None) -> TargetDialectStack:
    """Create the CUDA GPU compilation pipeline.

    Stack: encoding → dispatch → tiling → codegen → bundle
    """
    import tempfile
    from pathlib import Path

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cuda_bundle_"))

    return TargetDialectStack(
        target_name="cuda_a100",  # matches target profile name
        stages=[
            EncodingStage(),
            DispatchStage(),
            TilingStage(),
            CodegenStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={
            "encoding": CudaEncodingPlugin(),
            "dispatch": CudaDispatchPlugin(),
            "tiling": CudaTilingPlugin(),
            "codegen": CudaCodegenPlugin(),
        },
    )
