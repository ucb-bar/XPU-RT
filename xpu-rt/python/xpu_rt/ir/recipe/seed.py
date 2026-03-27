"""Seed Recipe IR generation from Payload IR analysis.

Generates a deterministic seed Recipe IR module before the LLM starts
editing. The seed contains:
    - RegionOp for each significant payload op
    - Facts about backends, kernel contracts, fusibility
    - Default candidate ops
    - Verification obligations
"""

from __future__ import annotations

from typing import Any

import structlog
from xdsl.dialects import builtin, func, linalg
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region

from compgen.ir.recipe.attrs import (
    DeviceRefAttr,
    EffectClassAttr,
    ProvenanceAttr,
    ShapeSummaryAttr,
)
from compgen.ir.recipe.ops_candidate import PlaceOnDeviceOp, TileOp
from compgen.ir.recipe.ops_fact import BackendAvailableOp, KernelContractOp
from compgen.ir.recipe.ops_provenance import FromTemplateOp
from compgen.ir.recipe.ops_scope import RecipeRegionOp
from compgen.ir.recipe.ops_verify import RequireDiffTestOp

log = structlog.get_logger()


def generate_seed_recipe(
    payload_module: ModuleOp,
    target_profile: Any = None,  # TargetProfile when available
    objective: str = "latency",
) -> ModuleOp:
    """Generate a seed Recipe IR from Payload IR analysis.

    Steps:
        1. Walk payload → extract regions per significant op
        2. Emit BackendAvailableOp facts per device
        3. Emit KernelContractOp facts per region
        4. Generate default candidates (tile matmuls, place on device 0)
        5. Add RequireDiffTestOp for all candidates
        6. Add FromTemplateOp provenance

    Args:
        payload_module: xDSL ModuleOp containing Payload IR.
        target_profile: Optional TargetProfile for device info.
        objective: Optimization objective.

    Returns:
        ModuleOp containing seed Recipe IR.
    """
    block = Block()
    region_counter = 0

    # Step 0: Provenance
    block.add_op(FromTemplateOp.build(properties={
        "template_name": StringAttr("seed_generator"),
        "template_version": IntegerAttr(1, IntegerType(64)),
    }))

    # Step 1: Extract regions from payload
    significant_ops = _extract_significant_ops(payload_module)

    for op_name, op_info in significant_ops.items():
        region_id = f"r_{region_counter}"
        region_counter += 1

        # Create RegionOp
        region_props: dict[str, object] = {
            "sym_name": StringAttr(region_id),
            "payload_region_id": StringAttr(op_name),
        }
        if op_info.get("shape"):
            region_props["shape_summary"] = ShapeSummaryAttr(
                op_info["shape"], op_info.get("dtype", "f32")
            )
        if op_info.get("effect"):
            region_props["effect_class"] = EffectClassAttr(op_info["effect"])
        if op_info.get("op_count"):
            region_props["op_count"] = IntegerAttr(
                op_info["op_count"], IntegerType(64)
            )

        block.add_op(RecipeRegionOp.build(properties=region_props))

        # Step 2: Backend availability
        backends = _infer_backends(op_info, target_profile)
        for backend in backends:
            block.add_op(BackendAvailableOp.build(properties={
                "region_ref": SymbolRefAttr(region_id),
                "backend": StringAttr(backend),
            }))

        # Step 3: Kernel contracts
        if op_info.get("is_compute"):
            contract_props: dict[str, object] = {
                "region_ref": SymbolRefAttr(region_id),
                "op_name": StringAttr(op_info.get("op_type", "unknown")),
            }
            if op_info.get("estimated_flops"):
                contract_props["estimated_flops"] = IntegerAttr(
                    op_info["estimated_flops"], IntegerType(64)
                )
            block.add_op(KernelContractOp.build(properties=contract_props))

        # Step 4: Default candidates
        if op_info.get("tileable"):
            default_sizes = _default_tile_sizes(op_info)
            block.add_op(TileOp.build(properties={
                "region_ref": SymbolRefAttr(region_id),
                "tile_sizes": ArrayAttr(
                    [IntegerAttr(s, IntegerType(64)) for s in default_sizes]
                ),
                "provenance": ProvenanceAttr("seed", 0),
            }))

        # Default device placement (device 0)
        block.add_op(PlaceOnDeviceOp.build(properties={
            "region_ref": SymbolRefAttr(region_id),
            "device": DeviceRefAttr(0, "default"),
            "provenance": ProvenanceAttr("seed", 0),
        }))

        # Step 5: Verification obligations
        block.add_op(RequireDiffTestOp.build(properties={
            "region_ref": SymbolRefAttr(region_id),
        }))

    log.info("seed.generated", regions=region_counter)
    return ModuleOp(Region(block))


def _extract_significant_ops(
    payload_module: ModuleOp,
) -> dict[str, dict[str, Any]]:
    """Walk payload IR and extract significant ops with metadata."""
    ops: dict[str, dict[str, Any]] = {}
    counter = 0

    for op in payload_module.walk():
        if isinstance(op, (builtin.ModuleOp, func.FuncOp, func.ReturnOp)):
            continue

        op_type = type(op).__name__
        op_name = f"{op_type}_{counter}"
        counter += 1

        info: dict[str, Any] = {
            "op_type": op.name if hasattr(op, "name") and isinstance(op.name, str) else op_type,
            "effect": "pure",
            "is_compute": False,
            "tileable": False,
            "op_count": 1,
        }

        # Check for compute ops
        if isinstance(op, (linalg.MatmulOp,)):
            info["is_compute"] = True
            info["tileable"] = True
            info["estimated_flops"] = 1000  # placeholder
        elif hasattr(op, "name") and "linalg" in str(getattr(op, "name", "")):
            info["is_compute"] = True
            info["tileable"] = True

        ops[op_name] = info

    return ops


def _infer_backends(
    op_info: dict[str, Any],
    target_profile: Any,
) -> list[str]:
    """Infer available backends for an op."""
    backends = ["fallback"]
    if op_info.get("is_compute"):
        backends.append("triton")
        backends.append("autocomp")
    return backends


def _default_tile_sizes(op_info: dict[str, Any]) -> list[int]:
    """Generate default tile sizes for a compute op."""
    return [64, 64, 32]


__all__ = ["generate_seed_recipe"]
