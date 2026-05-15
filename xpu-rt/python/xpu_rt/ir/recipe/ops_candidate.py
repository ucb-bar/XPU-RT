"""Recipe IR Family C: Candidate Action operations.

These encode possible optimization moves. The LLM primarily generates
these. Each is a proposal that may be accepted or rejected after
verification. All are Pure with optional ProvenanceAttr.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr


@irdl_op_definition
class TileOp(IRDLOperation):
    """Tile a region with specified sizes and optional interchange."""

    name = "recipe.tile"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    tile_sizes = prop_def(ArrayAttr)  # ArrayAttr of IntegerAttr
    interchange = opt_prop_def(ArrayAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        for size_attr in self.tile_sizes.data:
            if isinstance(size_attr, IntegerAttr) and size_attr.value.data <= 0:
                raise VerifyException(f"Tile sizes must be positive, got {size_attr.value.data}")


@irdl_op_definition
class FuseOp(IRDLOperation):
    """Fuse multiple regions together."""

    name = "recipe.fuse"

    sym_name = opt_prop_def(StringAttr)
    fuse_regions = prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    fusion_kind = opt_prop_def(StringAttr)  # "producer_consumer", "horizontal"
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if len(self.fuse_regions.data) < 2:
            raise VerifyException("FuseOp requires at least 2 regions")


@irdl_op_definition
class VectorizeOp(IRDLOperation):
    """Vectorize a region with the specified vector width."""

    name = "recipe.vectorize"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    vector_width = prop_def(IntegerAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.vector_width.value.data <= 0:
            raise VerifyException(f"Vector width must be positive, got {self.vector_width.value.data}")


@irdl_op_definition
class ReassociateOp(IRDLOperation):
    """Reassociate operations in a region.

    Strategies: "left", "right", "balanced".
    """

    name = "recipe.reassociate"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    strategy = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        valid = {"left", "right", "balanced"}
        if self.strategy.data not in valid:
            raise VerifyException(f"Invalid reassociation strategy '{self.strategy.data}', expected one of {valid}")


@irdl_op_definition
class LayoutNormalizeOp(IRDLOperation):
    """Normalize the layout of tensors in a region."""

    name = "recipe.layout_normalize"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    target_layout = prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class LowerToAccelOp(IRDLOperation):
    """Lower a region to the accelerator dialect."""

    name = "recipe.lower_to_accel"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    accel_cluster = opt_prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequestTritonKernelOp(IRDLOperation):
    """Request a Triton/Autocomp kernel search for a region."""

    name = "recipe.request_triton_kernel"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    search_budget = prop_def(IntegerAttr)
    kernel_family = opt_prop_def(StringAttr)  # "matmul_epilogue", "reduction", etc.
    backend = opt_prop_def(StringAttr)  # "triton", "autocomp"
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.search_budget.value.data <= 0:
            raise VerifyException(f"Search budget must be positive, got {self.search_budget.value.data}")


@irdl_op_definition
class MaterializeUkernelOp(IRDLOperation):
    """Materialize a microkernel for a region."""

    name = "recipe.materialize_ukernel"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    kernel_name = prop_def(StringAttr)
    calling_convention = opt_prop_def(StringAttr)  # "c", "triton", "nki", "cuda"
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class PlaceOnDeviceOp(IRDLOperation):
    """Place a region on a specific device."""

    name = "recipe.place_on_device"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    device = prop_def(DeviceRefAttr)
    reason = opt_prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.device.index.value.data < 0:
            raise VerifyException(f"Device index must be non-negative, got {self.device.index.value.data}")


@irdl_op_definition
class InsertCopyBoundaryOp(IRDLOperation):
    """Insert a data copy boundary between two regions on different devices."""

    name = "recipe.insert_copy_boundary"

    sym_name = opt_prop_def(StringAttr)
    src_region = prop_def(SymbolRefAttr)
    dst_region = prop_def(SymbolRefAttr)
    tensor_name = prop_def(StringAttr)
    is_async = opt_prop_def(IntegerAttr)  # 1 = async, 0 = sync
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class SegmentBoundaryOp(IRDLOperation):
    """Insert a segment boundary after a region."""

    name = "recipe.segment_boundary"

    sym_name = opt_prop_def(StringAttr)
    after_region = prop_def(SymbolRefAttr)
    reason = opt_prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class BlackboxOp(IRDLOperation):
    """Mark a region as a blackbox (excluded from eqsat/optimization)."""

    name = "recipe.blackbox"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    blackbox_class = prop_def(StringAttr)  # "side_effect", "unsupported", "opaque"
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequestExoKernelOp(IRDLOperation):
    """Request an Exo kernel search for a region.

    When lowered, triggers the Exo adapter to generate a seed proc,
    optionally apply a schedule library, compile to C, and validate.
    """

    name = "recipe.request_exo_kernel"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    search_budget = prop_def(IntegerAttr)
    schedule_lib = opt_prop_def(StringAttr)
    target_kit = opt_prop_def(StringAttr)
    kernel_family = opt_prop_def(StringAttr)  # "matmul", "conv2d", "reduction"
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.search_budget.value.data <= 0:
            raise VerifyException(f"Search budget must be positive, got {self.search_budget.value.data}")


@irdl_op_definition
class SelectExoScheduleLibOp(IRDLOperation):
    """Select a specific Exo schedule library for a region.

    The schedule library contains reusable scheduling helpers
    (following Exo 2's growable scheduling pattern).
    """

    name = "recipe.select_exo_schedule_lib"

    sym_name = opt_prop_def(StringAttr)
    region_ref = prop_def(SymbolRefAttr)
    lib_name = prop_def(StringAttr)
    version = opt_prop_def(StringAttr)
    guard_refs = opt_prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


__all__ = [
    "BlackboxOp",
    "FuseOp",
    "InsertCopyBoundaryOp",
    "LayoutNormalizeOp",
    "LowerToAccelOp",
    "MaterializeUkernelOp",
    "PlaceOnDeviceOp",
    "ReassociateOp",
    "RequestExoKernelOp",
    "RequestTritonKernelOp",
    "SegmentBoundaryOp",
    "SelectExoScheduleLibOp",
    "TileOp",
    "VectorizeOp",
]
