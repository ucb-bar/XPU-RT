"""Recipe IR Family B: Fact/Evidence operations.

These encode what the compiler knows -- observations, not commands.
All fact ops are Pure: they do not mutate state.
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

from compgen.ir.recipe.attrs import CostAttr, DeviceRefAttr


@irdl_op_definition
class BackendAvailableOp(IRDLOperation):
    """Declares that a backend can handle a given region.

    Backends: "triton", "autocomp", "vendor", "accel_native",
              "ukernel", "fallback".
    """

    name = "recipe.fact.backend_available"

    region_ref = prop_def(SymbolRefAttr)
    backend = prop_def(StringAttr)
    confidence = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class KernelContractOp(IRDLOperation):
    """Kernel interface specification for a region.

    Records layout requirements, supported dtypes, and cost estimates
    extracted from the Payload IR by ``extract_contracts()``.
    """

    name = "recipe.fact.kernel_contract"

    region_ref = prop_def(SymbolRefAttr)
    op_name = prop_def(StringAttr)
    input_layouts = opt_prop_def(ArrayAttr)
    output_layouts = opt_prop_def(ArrayAttr)
    supported_dtypes = opt_prop_def(ArrayAttr)
    estimated_flops = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class TransferCostOp(IRDLOperation):
    """Data movement cost between two regions/devices."""

    name = "recipe.fact.transfer_cost"

    src_region = prop_def(SymbolRefAttr)
    dst_region = prop_def(SymbolRefAttr)
    cost = prop_def(CostAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class LocalMemFitOp(IRDLOperation):
    """Whether a region's data fits in a device's local memory."""

    name = "recipe.fact.local_mem_fit"

    region_ref = prop_def(SymbolRefAttr)
    device = prop_def(DeviceRefAttr)
    fits = prop_def(IntegerAttr)  # 1 = fits, 0 = does not fit

    traits = traits_def(Pure())


@irdl_op_definition
class FusibleWithOp(IRDLOperation):
    """Records a fusion opportunity between two regions."""

    name = "recipe.fact.fusible_with"

    region_a = prop_def(SymbolRefAttr)
    region_b = prop_def(SymbolRefAttr)
    fusion_kind = opt_prop_def(StringAttr)  # "producer_consumer", "horizontal"

    traits = traits_def(Pure())


@irdl_op_definition
class CalibrationOp(IRDLOperation):
    """Hardware calibration measurement for a region on a device."""

    name = "recipe.fact.calibration"

    region_ref = prop_def(SymbolRefAttr)
    measured_latency_us = prop_def(IntegerAttr)
    device = prop_def(DeviceRefAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class ExportIssueOp(IRDLOperation):
    """Records a torch.export issue encountered during capture."""

    name = "recipe.fact.export_issue"

    description = prop_def(StringAttr)
    severity = prop_def(StringAttr)  # "error", "warning", "info"

    traits = traits_def(Pure())


@irdl_op_definition
class GraphBreakOp(IRDLOperation):
    """Records a graph break detected during dynamo tracing."""

    name = "recipe.fact.graph_break"

    location = prop_def(StringAttr)
    reason = prop_def(StringAttr)

    traits = traits_def(Pure())


__all__ = [
    "BackendAvailableOp",
    "CalibrationOp",
    "ExportIssueOp",
    "FusibleWithOp",
    "GraphBreakOp",
    "KernelContractOp",
    "LocalMemFitOp",
    "TransferCostOp",
]
