"""ROCm (AMD HIP) GPU target dialect stack — Wave 5 skeleton.

Mirrors ``cuda_gpu.py`` but routes to AMD's stack:

  * Triton-ROCm path for kernel codegen (Triton supports ROCm since 3.x;
    same source as NVIDIA Triton with backend="rocm").
  * HIP for runtime dispatch (host-side launch, memory mgmt).
  * Bundle layout identical (manifest.json + payload.mlir + kernels/).

What's *real* in this skeleton:
  * ``RocmEncodingPlugin`` / ``RocmDispatchPlugin`` / ``RocmTilingPlugin`` /
    ``RocmCodegenPlugin`` — concrete classes that implement the
    ``StagePlugin`` protocol so the pipeline runs end-to-end without
    crashing.
  * ``create_rocm_gpu_stack()`` factory — returns a usable
    ``TargetDialectStack``.

What's *deferred* to production:
  * MMA shape selection per ROCm arch (CDNA1 vs CDNA2 vs CDNA3 — they
    have different matrix-core dimensions). Today this skeleton uses
    16×16×16 universally; production needs per-arch tables.
  * Async copy via ``buffer_load_dword_xN`` — ROCm-Triton's analog of
    ``cp.async``. Skeleton emits sync loads.
  * HIP launcher — gpu_executor.py has the CUDA-only path; production
    needs a sibling ``rocm_executor.py`` (W5+ work).

The skeleton is enough to satisfy the Wave 5 plan: a v3 contract for
target ``rocm-mi250`` flows through ``select_translator`` →
``TritonContractTranslator`` (with ``target_arch="rocm"``) → this
stack's codegen, which currently emits the same Triton source the CUDA
stack would (Triton handles the HIP backend selection at install time).
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
from compgen.stages.layout.stage import LayoutStage
from compgen.stages.registry import TargetDialectStack
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR, CodegenStage
from compgen.stages.templates.tiling import TILE_SIZES_ATTR, TilingStage
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile

# ---------------------------------------------------------------------------
# ROCm-specific plugins
# ---------------------------------------------------------------------------


class RocmEncodingPlugin:
    """ROCm GPU encoding: prefer matrix-core-friendly layouts.

    CDNA matrix cores want 16×16 tiles for fp16/bf16; RDNA + WMMA
    accept 16×16 too (with different MFMA opcodes). Today this plugin
    tags every result tensor with ``row_major`` (default) and
    matmul-like ops with ``cdna_mfma_16x16``. Production should branch
    on the target's CDNA generation.
    """

    @property
    def target_name(self) -> str:
        return "rocm_gpu"

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
            if "matmul" in op.name.lower():
                op.attributes[ENCODING_ATTR] = StringAttr("cdna_mfma_16x16")
            elif ENCODING_ATTR not in op.attributes:
                op.attributes[ENCODING_ATTR] = StringAttr("row_major")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class RocmDispatchPlugin:
    """ROCm dispatch: fuse pointwise ops around matmul-shaped boundaries.

    Mirrors the CUDA dispatch policy. ROCm-Triton supports the same
    fusion patterns as CUDA-Triton; the dispatch IDs are backend-agnostic.
    """

    @property
    def target_name(self) -> str:
        return "rocm_gpu"

    @property
    def stage_name(self) -> str:
        return "dispatch"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        next_id = 0
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if DISPATCH_ID_ATTR in op.attributes:
                continue
            op.attributes[DISPATCH_ID_ATTR] = StringAttr(f"d_{next_id}")
            next_id += 1
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class RocmTilingPlugin:
    """ROCm tiling: 64×64×16 default for matmul (CDNA-friendly), 1024
    for pointwise."""

    @property
    def target_name(self) -> str:
        return "rocm_gpu"

    @property
    def stage_name(self) -> str:
        return "tiling"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not any(isinstance(r.type, TensorType) for r in op.results):
                continue
            if "matmul" in op.name.lower():
                op.attributes[TILE_SIZES_ATTR] = StringAttr("64x64x16")
            else:
                op.attributes[TILE_SIZES_ATTR] = StringAttr("1024")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class RocmLayoutPlugin:
    """ROCm layout — currently a passthrough; CDNA layout-folding is a
    production task."""

    @property
    def target_name(self) -> str:
        return "rocm_gpu"

    @property
    def stage_name(self) -> str:
        return "layout"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class RocmCodegenPlugin:
    """ROCm codegen — selects Triton-ROCm backend.

    For matmul, future work could route to rocBLAS / hipBLASLt. Today
    everything goes through Triton; the kernel translator (W5.1)
    produces a Triton skeleton that the codegen stage instantiates.
    """

    @property
    def target_name(self) -> str:
        return "rocm_gpu"

    @property
    def stage_name(self) -> str:
        return "codegen"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not any(isinstance(r.type, TensorType) for r in op.results):
                continue
            # Tag with triton_rocm so the bundle stage knows the backend
            op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("triton_rocm")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_rocm_gpu_stack(
    output_dir: str | None = None,
    target_name: str = "rocm_mi250",
) -> TargetDialectStack:
    """Create the ROCm GPU compilation pipeline.

    Stack mirrors CUDA: encoding → layout → dispatch → tiling → codegen → bundle.
    """
    import tempfile
    from pathlib import Path

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="rocm_bundle_"))

    return TargetDialectStack(
        target_name=target_name,
        stages=[
            EncodingStage(),
            LayoutStage(),
            DispatchStage(),
            TilingStage(),
            CodegenStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={
            "encoding": RocmEncodingPlugin(),
            "layout": RocmLayoutPlugin(),
            "dispatch": RocmDispatchPlugin(),
            "tiling": RocmTilingPlugin(),
            "codegen": RocmCodegenPlugin(),
        },
    )


__all__ = [
    "RocmCodegenPlugin",
    "RocmDispatchPlugin",
    "RocmEncodingPlugin",
    "RocmLayoutPlugin",
    "RocmTilingPlugin",
    "create_rocm_gpu_stack",
]
