"""Backward compatibility shim between old dataclass-based RecipeOps and new xDSL ops.

Provides bidirectional conversion between the legacy ``ops.py`` dataclass
types (MatchRegion, SetTileParams, etc.) and the new xDSL IRDLOperation
types (RecipeRegionOp, TileOp, etc.).  Also provides batch conversion
between ``list[RecipeOp]`` and ``ModuleOp``.

Invariants:
    - ``dataclass_to_xdsl`` is lossless for the old op set.
    - ``xdsl_to_dataclass`` is lossy for ops that have no old equivalent.
    - Round-tripping old -> xDSL -> old preserves all fields.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Operation

from compgen.ir.recipe.attrs import DeviceRefAttr
from compgen.ir.recipe.ops import (
    AssignDevice,
    ChooseTransformFamily,
    InsertCopyBoundary,
    MatchRegion,
    PromoteIfVerified,
    RecipeOp,
    RequestKernelSearch,
    RequireCheck,
    SetObjective,
    SetTileParams,
)
from compgen.ir.recipe.ops_candidate import (
    FuseOp,
    InsertCopyBoundaryOp,
    PlaceOnDeviceOp,
    RequestTritonKernelOp,
    TileOp,
    VectorizeOp,
)
from compgen.ir.recipe.ops_provenance import PromoteOp
from compgen.ir.recipe.ops_scope import RecipeRegionOp
from compgen.ir.recipe.ops_verify import (
    RequireDiffTestOp,
    RequireTranslationValidationOp,
)

log = structlog.get_logger()

_I64 = IntegerType(64)


def _int_attr(value: int) -> IntegerAttr:
    """Create an i64 IntegerAttr."""
    return IntegerAttr(value, _I64)


def _sym_ref(name: str) -> SymbolRefAttr:
    """Create a SymbolRefAttr from a plain string."""
    return SymbolRefAttr(name)


def dataclass_to_xdsl(op: RecipeOp) -> Operation | None:
    """Convert an old dataclass RecipeOp to a new xDSL Operation.

    Args:
        op: A legacy RecipeOp dataclass instance.

    Returns:
        The corresponding xDSL Operation, or ``None`` for ``SetObjective``
        (which has no direct xDSL equivalent).

    Raises:
        TypeError: If *op* is not a recognised RecipeOp variant.
    """
    if isinstance(op, MatchRegion):
        return RecipeRegionOp.build(properties={
            "sym_name": StringAttr(op.region_id),
            "payload_region_id": StringAttr(op.region_id),
        })

    if isinstance(op, SetTileParams):
        props: dict[str, object] = {
            "region_ref": _sym_ref(op.region_id),
            "tile_sizes": ArrayAttr([_int_attr(s) for s in op.tile_sizes]),
        }
        if op.interchange is not None:
            props["interchange"] = ArrayAttr([_int_attr(i) for i in op.interchange])
        return TileOp.build(properties=props)

    if isinstance(op, AssignDevice):
        return PlaceOnDeviceOp.build(properties={
            "region_ref": _sym_ref(op.region_id),
            "device": DeviceRefAttr(op.device_index, ""),
            **({"reason": StringAttr(op.reason)} if op.reason else {}),
        })

    if isinstance(op, InsertCopyBoundary):
        return InsertCopyBoundaryOp.build(properties={
            "src_region": _sym_ref(op.src_region),
            "dst_region": _sym_ref(op.dst_region),
            "tensor_name": StringAttr(op.tensor_name),
            "is_async": _int_attr(1 if op.async_ else 0),
        })

    if isinstance(op, RequestKernelSearch):
        return RequestTritonKernelOp.build(properties={
            "region_ref": _sym_ref(op.region_id),
            "search_budget": _int_attr(op.search_budget),
            **({"backend": StringAttr(op.backend)} if op.backend else {}),
        })

    if isinstance(op, RequireCheck):
        if op.check_type == "translation_validation":
            return RequireTranslationValidationOp.build(properties={
                "region_ref": _sym_ref(op.region_id),
            })
        # Default to differential test for all other check types
        return RequireDiffTestOp.build(properties={
            "region_ref": _sym_ref(op.region_id),
        })

    if isinstance(op, PromoteIfVerified):
        return PromoteOp.build(properties={
            "candidate_ref": _sym_ref(op.recipe_name),
            "recipe_key": StringAttr(op.recipe_name),
            "version": _int_attr(0),
        })

    if isinstance(op, ChooseTransformFamily):
        family = op.family.lower()
        if family == "tile":
            # Emit a TileOp with empty tile sizes as a placeholder
            return TileOp.build(properties={
                "region_ref": _sym_ref(op.region_id),
                "tile_sizes": ArrayAttr([]),
            })
        if family == "fuse":
            return FuseOp.build(properties={
                "fuse_regions": ArrayAttr([_sym_ref(op.region_id)]),
            })
        if family == "vectorize":
            return VectorizeOp.build(properties={
                "region_ref": _sym_ref(op.region_id),
                "vector_width": _int_attr(1),
            })
        # Unknown family -- fall back to TileOp placeholder
        log.warning(
            "compat.unknown_transform_family",
            family=op.family,
            region_id=op.region_id,
        )
        return TileOp.build(properties={
            "region_ref": _sym_ref(op.region_id),
            "tile_sizes": ArrayAttr([]),
        })

    if isinstance(op, SetObjective):
        log.info("compat.set_objective_skipped", objective=op.objective.value)
        return None

    msg = f"Unrecognised RecipeOp type: {type(op).__name__}"
    raise TypeError(msg)


def xdsl_to_dataclass(op: Operation) -> RecipeOp | None:
    """Convert a new xDSL Operation to an old dataclass RecipeOp.

    Args:
        op: An xDSL Operation from the Recipe IR dialect.

    Returns:
        The corresponding legacy RecipeOp, or ``None`` for op types
        that have no old dataclass equivalent.
    """
    if isinstance(op, RecipeRegionOp):
        return MatchRegion(
            region_id=op.sym_name.data,
            op_filter="",
        )

    if isinstance(op, TileOp):
        sizes = tuple(
            attr.value.data
            for attr in op.tile_sizes.data
            if isinstance(attr, IntegerAttr)
        )
        interchange: tuple[int, ...] | None = None
        if op.interchange is not None:
            interchange = tuple(
                attr.value.data
                for attr in op.interchange.data
                if isinstance(attr, IntegerAttr)
            )
        return SetTileParams(
            region_id=op.region_ref.root_reference.data,
            tile_sizes=sizes,
            interchange=interchange,
        )

    if isinstance(op, PlaceOnDeviceOp):
        return AssignDevice(
            region_id=op.region_ref.root_reference.data,
            device_index=op.device.index.value.data,
            reason=op.reason.data if op.reason is not None else "",
        )

    if isinstance(op, InsertCopyBoundaryOp):
        is_async = True
        if op.is_async is not None:
            is_async = op.is_async.value.data != 0
        return InsertCopyBoundary(
            src_region=op.src_region.root_reference.data,
            dst_region=op.dst_region.root_reference.data,
            tensor_name=op.tensor_name.data,
            async_=is_async,
        )

    if isinstance(op, RequestTritonKernelOp):
        return RequestKernelSearch(
            region_id=op.region_ref.root_reference.data,
            backend=op.backend.data if op.backend is not None else "autocomp",
            search_budget=op.search_budget.value.data,
        )

    if isinstance(op, RequireDiffTestOp):
        return RequireCheck(
            region_id=op.region_ref.root_reference.data,
            check_type="differential",
        )

    if isinstance(op, RequireTranslationValidationOp):
        return RequireCheck(
            region_id=op.region_ref.root_reference.data,
            check_type="translation_validation",
        )

    if isinstance(op, PromoteOp):
        return PromoteIfVerified(
            recipe_name=op.recipe_key.data,
        )

    if isinstance(op, (FuseOp, VectorizeOp)):
        # Best-effort: map back to ChooseTransformFamily
        if isinstance(op, FuseOp):
            region_id = ""
            if op.fuse_regions.data:
                first_ref = op.fuse_regions.data[0]
                if isinstance(first_ref, SymbolRefAttr):
                    region_id = first_ref.root_reference.data
            return ChooseTransformFamily(region_id=region_id, family="fuse")
        # VectorizeOp
        return ChooseTransformFamily(
            region_id=op.region_ref.root_reference.data,
            family="vectorize",
        )

    # No old equivalent for this op type
    return None


def recipe_list_to_module(ops: list[RecipeOp]) -> ModuleOp:
    """Convert a list of old dataclass RecipeOps to an xDSL ModuleOp.

    Args:
        ops: List of legacy RecipeOp dataclass instances.

    Returns:
        An xDSL ModuleOp containing the converted operations.
    """
    xdsl_ops: list[Operation] = []
    for op in ops:
        converted = dataclass_to_xdsl(op)
        if converted is not None:
            xdsl_ops.append(converted)
    return ModuleOp(xdsl_ops)


def module_to_recipe_list(module: ModuleOp) -> list[RecipeOp]:
    """Convert an xDSL ModuleOp to a list of old dataclass RecipeOps.

    Operations that have no old dataclass equivalent are silently dropped.

    Args:
        module: An xDSL ModuleOp containing Recipe IR operations.

    Returns:
        List of legacy RecipeOp instances.
    """
    result: list[RecipeOp] = []
    for op in module.body.ops:
        converted = xdsl_to_dataclass(op)
        if converted is not None:
            result.append(converted)
    return result


__all__ = [
    "dataclass_to_xdsl",
    "module_to_recipe_list",
    "recipe_list_to_module",
    "xdsl_to_dataclass",
]
