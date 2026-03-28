"""Recipe-fact extraction and guard-evaluation environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation

from compgen.ir.recipe.ops_candidate import (
    FuseOp,
    InsertCopyBoundaryOp,
    PlaceOnDeviceOp,
)
from compgen.ir.recipe.ops_fact import (
    BackendAvailableOp,
    CalibrationOp,
    ExportIssueOp,
    FusibleWithOp,
    GraphBreakOp,
    KernelContractOp,
    LocalMemFitOp,
    TransferCostOp,
)


@dataclass
class RegionFacts:
    """Aggregated Recipe facts for one region symbol."""

    region_ref: str
    op_name: str = "unknown"
    backends: set[str] = field(default_factory=set)
    estimated_flops: int = 0
    input_layouts: tuple[str, ...] = ()
    output_layouts: tuple[str, ...] = ()
    supported_dtypes: tuple[str, ...] = ()
    local_mem_fit_by_device: dict[tuple[int, str], bool] = field(default_factory=dict)
    measured_latency_us_by_device: dict[tuple[int, str], int] = field(default_factory=dict)


@dataclass
class RecipeFactIndex:
    """Indexed view of Recipe facts for guard evaluation."""

    target_class: str = ""
    region_facts: dict[str, RegionFacts] = field(default_factory=dict)
    fusible_pairs: dict[tuple[str, str], str] = field(default_factory=dict)
    transfer_cost_us: dict[tuple[str, str], int] = field(default_factory=dict)
    graph_breaks: list[tuple[str, str]] = field(default_factory=list)
    export_issues: list[tuple[str, str]] = field(default_factory=list)

    def get_or_create_region(self, region_ref: str) -> RegionFacts:
        if region_ref not in self.region_facts:
            self.region_facts[region_ref] = RegionFacts(region_ref=region_ref)
        return self.region_facts[region_ref]


def _maybe_string_tuple(attr: Any) -> tuple[str, ...]:
    if attr is None:
        return ()
    values: list[str] = []
    for item in getattr(attr, "data", []):
        if isinstance(item, StringAttr):
            values.append(item.data)
        else:
            values.append(str(item))
    return tuple(values)


def build_fact_index(module: Any, target_class: str = "") -> RecipeFactIndex:
    """Build a fact index from a Recipe module."""

    index = RecipeFactIndex(target_class=target_class)
    for op in module.walk():
        if isinstance(op, BackendAvailableOp):
            region = index.get_or_create_region(op.region_ref.root_reference.data)
            region.backends.add(op.backend.data)
        elif isinstance(op, KernelContractOp):
            region = index.get_or_create_region(op.region_ref.root_reference.data)
            region.op_name = op.op_name.data
            if op.estimated_flops is not None:
                region.estimated_flops = op.estimated_flops.value.data
            region.input_layouts = _maybe_string_tuple(op.input_layouts)
            region.output_layouts = _maybe_string_tuple(op.output_layouts)
            region.supported_dtypes = _maybe_string_tuple(op.supported_dtypes)
        elif isinstance(op, LocalMemFitOp):
            region = index.get_or_create_region(op.region_ref.root_reference.data)
            key = (op.device.index.value.data, op.device.device_name.data)
            region.local_mem_fit_by_device[key] = op.fits.value.data != 0
        elif isinstance(op, CalibrationOp):
            region = index.get_or_create_region(op.region_ref.root_reference.data)
            key = (op.device.index.value.data, op.device.device_name.data)
            region.measured_latency_us_by_device[key] = op.measured_latency_us.value.data
        elif isinstance(op, FusibleWithOp):
            a = op.region_a.root_reference.data
            b = op.region_b.root_reference.data
            key = tuple(sorted((a, b)))
            index.fusible_pairs[key] = op.fusion_kind.data if op.fusion_kind is not None else ""
        elif isinstance(op, TransferCostOp):
            key = (op.src_region.root_reference.data, op.dst_region.root_reference.data)
            index.transfer_cost_us[key] = op.cost.value_us.value.data
        elif isinstance(op, GraphBreakOp):
            index.graph_breaks.append((op.location.data, op.reason.data))
        elif isinstance(op, ExportIssueOp):
            index.export_issues.append((op.description.data, op.severity.data))
    return index


def _base_env(index: RecipeFactIndex) -> dict[str, Any]:
    target = index.target_class
    return {
        "target_class": target,
        "target_is_triton_friendly": target == "TRITON_FRIENDLY",
        "target_is_accel_native": target == "ACCEL_NATIVE",
        "target_is_ukernel_runtime": target == "UKERNEL_RUNTIME",
        "target_is_hybrid": target == "HYBRID",
        "graph_break_count": len(index.graph_breaks),
        "graph_break_free": len(index.graph_breaks) == 0,
        "export_issue_count": len(index.export_issues),
        "export_issue_free": len(index.export_issues) == 0,
    }


def _merge_region_env(env: dict[str, Any], facts: RegionFacts, *, device_key: tuple[int, str] | None = None) -> None:
    env["op_name"] = facts.op_name
    env["estimated_flops"] = facts.estimated_flops
    env["backend_triton"] = "triton" in facts.backends
    env["backend_autocomp"] = "autocomp" in facts.backends
    env["backend_vendor"] = "vendor" in facts.backends
    env["backend_accel_native"] = "accel_native" in facts.backends
    env["backend_ukernel"] = "ukernel" in facts.backends
    if device_key is not None and device_key in facts.local_mem_fit_by_device:
        env["local_mem_fit"] = facts.local_mem_fit_by_device[device_key]
    else:
        env["local_mem_fit"] = any(facts.local_mem_fit_by_device.values())
    env["measured_latency_us"] = (
        facts.measured_latency_us_by_device.get(device_key, 0)
        if device_key is not None
        else next(iter(facts.measured_latency_us_by_device.values()), 0)
    )


def build_candidate_env(op: Operation, fact_index: RecipeFactIndex) -> dict[str, Any]:
    """Build a guard-evaluation environment for a candidate op."""

    env = _base_env(fact_index)

    if hasattr(op, "sym_name") and getattr(op, "sym_name") is not None:
        env["candidate_symbol"] = getattr(op, "sym_name").data

    if hasattr(op, "region_ref") and getattr(op, "region_ref") is not None:
        region_ref = op.region_ref.root_reference.data
        env["region_ref"] = region_ref
        facts = fact_index.get_or_create_region(region_ref)
        device_key = None
        if isinstance(op, PlaceOnDeviceOp):
            device_key = (op.device.index.value.data, op.device.device_name.data)
            env["device_index"] = device_key[0]
            env["device_name"] = device_key[1]
        _merge_region_env(env, facts, device_key=device_key)
        return env

    if isinstance(op, FuseOp):
        regions = [ref.root_reference.data for ref in op.fuse_regions.data]
        env["fusion_region_count"] = len(regions)
        env["fusible"] = all(
            tuple(sorted((lhs, rhs))) in fact_index.fusible_pairs
            for lhs, rhs in zip(regions, regions[1:])
        ) if len(regions) >= 2 else False
        region_facts = [fact_index.get_or_create_region(region) for region in regions]
        env["estimated_flops"] = sum(facts.estimated_flops for facts in region_facts)
        env["backend_triton"] = all("triton" in facts.backends for facts in region_facts)
        env["backend_autocomp"] = all("autocomp" in facts.backends for facts in region_facts)
        env["backend_vendor"] = all("vendor" in facts.backends for facts in region_facts)
        env["backend_accel_native"] = all("accel_native" in facts.backends for facts in region_facts)
        env["backend_ukernel"] = all("ukernel" in facts.backends for facts in region_facts)
        env["local_mem_fit"] = all(any(facts.local_mem_fit_by_device.values()) for facts in region_facts)
        env["transfer_cost_us"] = sum(
            fact_index.transfer_cost_us.get((lhs, rhs), fact_index.transfer_cost_us.get((rhs, lhs), 0))
            for lhs, rhs in zip(regions, regions[1:])
        )
        return env

    if isinstance(op, InsertCopyBoundaryOp):
        src = op.src_region.root_reference.data
        dst = op.dst_region.root_reference.data
        env["src_region"] = src
        env["dst_region"] = dst
        env["transfer_cost_us"] = fact_index.transfer_cost_us.get((src, dst), fact_index.transfer_cost_us.get((dst, src), 0))
        return env

    return env


__all__ = [
    "RecipeFactIndex",
    "RegionFacts",
    "build_candidate_env",
    "build_fact_index",
]
