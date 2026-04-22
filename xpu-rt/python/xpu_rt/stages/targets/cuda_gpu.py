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

from compgen.agent.decisions import (
    DecisionCandidate,
    DecisionSite,
    get_active_registry,
)
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


def _op_site_key(op: Any) -> str:
    """Build a stable site id from an op. Prefers ``compgen.region_id`` —
    already stamped by the importer and FX-stable across stages —
    falling back to a name+position key when no region id exists.
    """
    attrs = getattr(op, "attributes", {}) or {}
    rid_attr = attrs.get("compgen.region_id") if attrs else None
    if rid_attr is not None and hasattr(rid_attr, "data"):
        return str(rid_attr.data)
    return f"{op.name}@{id(op):x}"


# ---------------------------------------------------------------------------
# CUDA-specific plugins
# ---------------------------------------------------------------------------


class CudaEncodingPlugin:
    """CUDA GPU encoding — surfaces every op's encoding as a decision site.

    The plugin no longer hardcodes ``matmul → tiled_128x64, else row_major``.
    Each tensor-producing op becomes a :class:`DecisionSite` with a
    non-binding oracle recommendation; the agent can override via MCP
    (``apply_decision``) before compilation or between compile phases.
    When no agent override exists, the oracle's pick is applied with
    ``source="fallback_oracle"`` so reviewers can tell the difference.
    """

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "encoding"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target

    def transform(self, module: ModuleOp) -> ModuleOp:
        registry = get_active_registry()
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not any(isinstance(r.type, TensorType) for r in op.results):
                continue

            is_matmul = "matmul" in op.name.lower()
            candidates = (
                DecisionCandidate(
                    id="tiled_128x64",
                    value="tiled_128x64",
                    source="oracle:encoding",
                    oracle_verdict="recommended" if is_matmul else "allowed",
                    oracle_reason="MMA-friendly 128x64 tile for tensor cores" if is_matmul else "non-matmul fallback",
                    oracle_confidence=0.7 if is_matmul else 0.3,
                ),
                DecisionCandidate(
                    id="row_major",
                    value="row_major",
                    source="oracle:encoding",
                    oracle_verdict="recommended" if not is_matmul else "allowed",
                    oracle_reason="default linear layout"
                    if not is_matmul
                    else "readable but non-MMA; may prevent tensor-core use",
                    oracle_confidence=0.6 if not is_matmul else 0.2,
                ),
                DecisionCandidate(
                    id="tiled_16x16x16",
                    value="tiled_16x16x16",
                    source="oracle:encoding",
                    oracle_verdict="allowed",
                    oracle_reason="MMA tile aligned to f16/bf16 warp-level shape",
                    oracle_confidence=0.5 if is_matmul else 0.1,
                ),
            )
            recommended_id = "tiled_128x64" if is_matmul else "row_major"
            site_id = f"cuda.encoding:{_op_site_key(op)}"
            context = {
                "op": op.name,
                "shapes": [list(r.type.get_shape()) for r in op.results if isinstance(r.type, TensorType)],
                "is_matmul": is_matmul,
            }

            if registry is None:
                # Batch / no-MCP path: apply the oracle pick directly
                # with a clear marker so the trace differentiates this
                # from an agent decision.
                op.attributes[ENCODING_ATTR] = StringAttr(recommended_id)
                continue

            site = DecisionSite(
                site_id=site_id,
                kind="encoding",
                context=context,
                candidates=candidates,
                oracle_recommended_id=recommended_id,
            )
            registry.enqueue(site)
            outcome = registry.resolve(site_id)
            op.attributes[ENCODING_ATTR] = StringAttr(str(outcome.chosen_value))
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
    """CUDA GPU tiling — surfaces each op's tile shape as a decision site."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "tiling"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        registry = get_active_registry()
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue
            is_matmul = "matmul" in op.name.lower()
            is_linalg = op.name.startswith("linalg.")
            if not (is_matmul or is_linalg):
                continue

            if is_matmul:
                recommended = "128x128x32"
                candidates = (
                    DecisionCandidate(
                        id="128x128x32",
                        value="128x128x32",
                        source="oracle:tile",
                        oracle_verdict="recommended",
                        oracle_reason="CUDA threadblock-shaped matmul tile; fits 48KB SMEM",
                        oracle_confidence=0.7,
                    ),
                    DecisionCandidate(
                        id="64x64x32",
                        value="64x64x32",
                        source="oracle:tile",
                        oracle_verdict="allowed",
                        oracle_reason="smaller tile, lower SMEM pressure",
                        oracle_confidence=0.5,
                    ),
                    DecisionCandidate(
                        id="128x256x64",
                        value="128x256x64",
                        source="oracle:tile",
                        oracle_verdict="allowed",
                        oracle_reason="larger tile, favours Hopper class",
                        oracle_confidence=0.4,
                    ),
                )
            else:
                recommended = "256"
                candidates = (
                    DecisionCandidate(
                        id="256",
                        value="256",
                        source="oracle:tile",
                        oracle_verdict="recommended",
                        oracle_reason="generic linalg tile size 256",
                        oracle_confidence=0.5,
                    ),
                    DecisionCandidate(
                        id="128",
                        value="128",
                        source="oracle:tile",
                        oracle_verdict="allowed",
                        oracle_reason="half tile; gentler on small shapes",
                        oracle_confidence=0.4,
                    ),
                )
            site_id = f"cuda.tiling:{_op_site_key(op)}"
            context = {
                "op": op.name,
                "is_matmul": is_matmul,
            }

            if registry is None:
                op.attributes[TILE_SIZES_ATTR] = StringAttr(recommended)
                continue

            site = DecisionSite(
                site_id=site_id,
                kind="tile",
                context=context,
                candidates=candidates,
                oracle_recommended_id=recommended,
            )
            registry.enqueue(site)
            outcome = registry.resolve(site_id)
            op.attributes[TILE_SIZES_ATTR] = StringAttr(str(outcome.chosen_value))
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"tiling_strategy": "cuda_threadblock"}


class CudaLayoutPlugin:
    """CUDA GPU layout: specialize layouts for tensor-core MMA tile shapes."""

    @property
    def target_name(self) -> str:
        return "cuda_gpu"

    @property
    def stage_name(self) -> str:
        return "layout"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        self._target = target
        self._caps = capabilities

    def transform(self, module: ModuleOp) -> ModuleOp:
        from compgen.transforms.layout.cuda_resolver import CudaLayoutResolver
        from compgen.transforms.layout.fuse_layout_into_producers import fuse_layout_into_producers
        from compgen.transforms.layout.specialize_layouts import specialize_layouts

        module = fuse_layout_into_producers(module)
        resolver = CudaLayoutResolver()
        module = specialize_layouts(module, resolver=resolver, capabilities=self._caps)
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"layout_strategy": "cuda_mma_tiled"}


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
            LayoutStage(),
            DispatchStage(),
            TilingStage(),
            CodegenStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={
            "encoding": CudaEncodingPlugin(),
            "layout": CudaLayoutPlugin(),
            "dispatch": CudaDispatchPlugin(),
            "tiling": CudaTilingPlugin(),
            "codegen": CudaCodegenPlugin(),
        },
    )
