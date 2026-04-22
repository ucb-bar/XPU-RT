"""SIMT GPU family stack generator.

Produces: encoding → dispatch → tiling → codegen → bundle
Same pattern as cuda_gpu.py but parameterized by HardwareSpec.
"""

from __future__ import annotations

from pathlib import Path
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
from compgen.stages.encoding import EncodingStage
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.registry import TargetDialectStack
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR, CodegenStage
from compgen.stages.templates.tiling import TilingStage
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


def _op_site_key(op: Any) -> str:
    attrs = getattr(op, "attributes", {}) or {}
    rid_attr = attrs.get("compgen.region_id") if attrs else None
    if rid_attr is not None and hasattr(rid_attr, "data"):
        return str(rid_attr.data)
    return f"{op.name}@{id(op):x}"


class GpuEncodingPlugin:
    """GPU encoding: prefer tiled layouts for tensor core ops."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "encoding"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        """Surface every op's encoding as a decision site.

        The oracle recommends a tiled layout for matmuls sized to the
        target's declared engine geometry, and ``row_major`` otherwise.
        When a :class:`DecisionRegistry` is active (MCP session path),
        the agent can ``apply_decision`` before this runs and the
        override propagates into the IR. Without a registry we fall
        back to the oracle's pick directly.
        """
        tile_str = "row_major"
        if self._spec.engine_geometry.tiles:
            dims = self._spec.engine_geometry.tiles[0].dimensions
            tile_str = f"tiled_{'x'.join(str(d) for d in dims)}"

        registry = get_active_registry()
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not any(isinstance(r.type, TensorType) for r in op.results):
                continue

            is_matmul = "matmul" in op.name.lower()
            recommended = tile_str if is_matmul else "row_major"
            candidates = (
                DecisionCandidate(
                    id=tile_str,
                    value=tile_str,
                    source="oracle:encoding",
                    oracle_verdict="recommended" if is_matmul else "allowed",
                    oracle_reason=(
                        f"engine geometry declares tile {tile_str}; MMA-friendly"
                        if is_matmul
                        else f"tile {tile_str} exists but op is not matmul"
                    ),
                    oracle_confidence=0.7 if is_matmul else 0.3,
                ),
                DecisionCandidate(
                    id="row_major",
                    value="row_major",
                    source="oracle:encoding",
                    oracle_verdict="recommended" if not is_matmul else "allowed",
                    oracle_reason=(
                        "default layout for non-matmul ops"
                        if not is_matmul
                        else "row_major on matmul prevents tensor-core use"
                    ),
                    oracle_confidence=0.6 if not is_matmul else 0.2,
                ),
            )
            # Dedup by value (tile_str can equal row_major)
            seen: set[str] = set()
            unique = tuple(c for c in candidates if not (c.id in seen or seen.add(c.id)))
            site_id = f"simt.encoding:{_op_site_key(op)}"
            context = {
                "op": op.name,
                "shapes": [list(r.type.get_shape()) for r in op.results if isinstance(r.type, TensorType)],
                "is_matmul": is_matmul,
                "engine_tile": tile_str,
            }

            if registry is None:
                if is_matmul:
                    op.attributes[ENCODING_ATTR] = StringAttr(tile_str)
                elif ENCODING_ATTR not in op.attributes:
                    op.attributes[ENCODING_ATTR] = StringAttr("row_major")
                continue

            site = DecisionSite(
                site_id=site_id,
                kind="encoding",
                context=context,
                candidates=unique,
                oracle_recommended_id=recommended,
            )
            registry.enqueue(site)
            outcome = registry.resolve(site_id)
            op.attributes[ENCODING_ATTR] = StringAttr(str(outcome.chosen_value))
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"encoding_strategy": "gpu_tiled"}


class GpuCodegenPlugin:
    """GPU codegen: assign Triton backend."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "codegen"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and CODEGEN_BACKEND_ATTR not in op.attributes:
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("triton")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"codegen_strategy": "gpu_triton"}


def create_gpu_stack(spec: HardwareSpec, output_dir: str | None = None) -> TargetDialectStack:
    """Create SIMT GPU compilation pipeline from spec."""
    import tempfile

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="gpu_bundle_"))
    return TargetDialectStack(
        target_name=spec.name,
        stages=[EncodingStage(), DispatchStage(), TilingStage(), CodegenStage(), BundleStage(output_dir=bundle_dir)],
        plugins={"encoding": GpuEncodingPlugin(spec), "codegen": GpuCodegenPlugin(spec)},
    )
