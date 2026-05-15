"""Module-level helpers for CompilerEnv.

These compute derived views of the IR module (region extraction, legal
action enumeration, pass factories) and are shared between the env core
and tests.  They carry no mutable state themselves.
"""

from __future__ import annotations

import os
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.ir import Operation

from compgen.targets.schema import TargetProfile

from compgen.agent.env.actions import (
    Action,
    AnalyzeAction,
    ApplyPassAction,
    AssignDeviceAction,
    BenchmarkAction,
    CalibrateAction,
    CheckpointAction,
    CompileAndRunAction,
    ConfigureDispatchAction,
    ConfigureProfilingAction,
    DiscoverOpsAction,
    EqSatAction,
    FuseAction,
    GenerateLLVMPatchAction,
    GeneratePassAction,
    GenerateRuntimeHooksAction,
    GenerateXDSLDialectAction,
    GeneralizeAction,
    InsertCopyAction,
    InspectAction,
    InspectEGraphAction,
    LegalAction,
    NoopAction,
    ProposeRuleAction,
    RequestSemanticsAction,
    RequestTransferAnalysisAction,
    RequestVerificationAction,
    RollbackAction,
    SearchKernelAction,
    SetDtypeAction,
    SetExtractionObjectiveAction,
    SolveAction,
    TileAction,
)
from compgen.agent.env.observations import RegionInfo

# ============================================================================
# The Environment
# ============================================================================


def _extract_regions(module: ModuleOp, target: TargetProfile) -> list[RegionInfo]:
    """Extract structured region info from the IR module.

    Picks up ops with compgen.region_id AND significant unlabeled ops
    (like GenericOp after generalization). Auto-assigns region IDs to
    unlabeled ops so the agent always has a complete view.
    """
    from xdsl.dialects.builtin import StringAttr
    from xdsl.dialects.linalg import GenericOp, TransposeOp

    regions: list[RegionInfo] = []

    # Significant op types that should always be visible to the agent
    significant_types = (MatmulOp, GenericOp, TransposeOp, CallOp)

    # Collect ops with existing region_id AND significant unlabeled ops
    ops_with_rid: list[tuple[str, Operation]] = []
    counters: dict[str, int] = {}

    for op in module.walk():
        rid_attr = op.attributes.get("compgen.region_id")
        if rid_attr is not None:
            ops_with_rid.append((rid_attr.data, op))  # type: ignore[attr-defined]
        elif isinstance(op, significant_types):
            # Auto-assign a region_id
            op_name = type(op).__name__.lower().replace("op", "")
            count = counters.get(op_name, 0)
            rid = f"{op_name}_{count}"
            counters[op_name] = count + 1
            op.attributes["compgen.region_id"] = StringAttr(rid)
            ops_with_rid.append((rid, op))

    for rid, op in ops_with_rid:
        # Determine op type
        if isinstance(op, MatmulOp):
            op_type = "matmul"
        elif isinstance(op, CallOp):
            callee = op.callee.string_value() if hasattr(op, "callee") else ""
            if "gelu" in callee:
                op_type = "gelu"
            elif "add" in callee:
                op_type = "add"
            elif "mul" in callee:
                op_type = "mul"
            else:
                op_type = "call"
        else:
            op_type = type(op).__name__.lower().replace("op", "")

        # Extract shapes
        input_shapes = tuple(
            tuple(o.type.get_shape()) for o in op.operands if isinstance(o.type, TensorType)
        )
        output_shapes = tuple(
            tuple(r.type.get_shape()) for r in op.results if isinstance(r.type, TensorType)
        )

        # Estimate FLOPs and bytes
        flops = 0
        bytes_in = 0
        bytes_out = 0
        for shape in input_shapes:
            elem_count = 1
            for s in shape:
                elem_count *= s
            bytes_in += elem_count * 4  # assume f32
        for shape in output_shapes:
            elem_count = 1
            for s in shape:
                elem_count *= s
            bytes_out += elem_count * 4

        if op_type == "matmul" and len(input_shapes) >= 2:
            dim_m = input_shapes[0][0]
            dim_k = input_shapes[0][1]
            dim_n = input_shapes[1][1] if len(input_shapes[1]) > 1 else input_shapes[1][0]
            flops = 2 * dim_m * dim_k * dim_n

        total_bytes = bytes_in + bytes_out
        ai = flops / total_bytes if total_bytes > 0 else 0.0

        # Estimate latency from target profile
        latency_us = 0.0
        if target.devices:
            dev = target.devices[0]
            peak_flops_per_sec = 0.0
            peak_bw_bytes_per_sec = 0.0
            for cu in dev.compute_units:
                if cu.peak_tflops:
                    peak_flops_per_sec = max(peak_flops_per_sec, cu.peak_tflops * 1e12)
            for ml in dev.memory_hierarchy:
                if ml.bandwidth_gbps:
                    peak_bw_bytes_per_sec = max(peak_bw_bytes_per_sec, ml.bandwidth_gbps * 1e9)

            compute_time = flops / peak_flops_per_sec if peak_flops_per_sec > 0 else 0
            memory_time = total_bytes / peak_bw_bytes_per_sec if peak_bw_bytes_per_sec > 0 else 0
            latency_us = max(compute_time, memory_time) * 1e6

        is_compute_bound = flops > 0 and (ai > 10.0)  # rough threshold

        regions.append(RegionInfo(
            region_id=rid,
            op_type=op_type,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            flops=flops,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            arithmetic_intensity=ai,
            estimated_latency_us=latency_us,
            device_index=-1,
            is_compute_bound=is_compute_bound,
            dtype="f32",
            consumers=(),  # TODO: build data flow graph
            producers=(),
        ))

    return regions


def _make_pass_factory(pass_name: str) -> Any:
    """Lazy-import pass factories to avoid import-time dependency on xDSL transforms."""
    factories: dict[str, Any] = {}

    def _generalize_factory(args: dict[str, Any]) -> Any:
        from xdsl.transforms.linalg_generalize_named_ops import LinalgGeneralizeNamedOpsPass
        return LinalgGeneralizeNamedOpsPass()

    def _dce_factory(args: dict[str, Any]) -> Any:
        from xdsl.transforms.dead_code_elimination import DeadCodeElimination
        return DeadCodeElimination()

    def _canonicalize_factory(args: dict[str, Any]) -> Any:
        from xdsl.transforms.canonicalize import CanonicalizePass
        return CanonicalizePass()

    def _constant_fold_factory(args: dict[str, Any]) -> Any:
        from xdsl.transforms.constant_fold_interp import ConstantFoldInterpPass
        return ConstantFoldInterpPass()

    def _cse_factory(args: dict[str, Any]) -> Any:
        from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
        return CommonSubexpressionElimination()

    factories = {
        "generalize": _generalize_factory,
        "dce": _dce_factory,
        "canonicalize": _canonicalize_factory,
        "constant_fold": _constant_fold_factory,
        "cse": _cse_factory,
    }
    return factories.get(pass_name)


_PASS_MENU: dict[str, dict[str, Any]] = {
    "generalize": {
        "factory": _make_pass_factory("generalize"),
        "desc": "Named ops (matmul, transpose) → linalg.generic",
        "risk": "safe",
    },
    "dce": {
        "factory": _make_pass_factory("dce"),
        "desc": "Remove dead code",
        "risk": "safe",
    },
    "canonicalize": {
        "factory": _make_pass_factory("canonicalize"),
        "desc": "Canonicalize ops",
        "risk": "safe",
    },
    "constant_fold": {
        "factory": _make_pass_factory("constant_fold"),
        "desc": "Fold constants",
        "risk": "safe",
    },
    "cse": {
        "factory": _make_pass_factory("cse"),
        "desc": "Eliminate common subexpressions",
        "risk": "safe",
    },
}


_PACK_ACTION_APERTURES: dict[type[Action], tuple[str, ...]] = {
    SearchKernelAction: (
        "kernel_selection",
        "schedule_generation",
        "tile_schedule_generation",
        "tile_fusion_generation",
        "tile_layout_generation",
        "tiling_schedules",
        "software_pipelining",
        "attention_schedule_generation",
        "packing_layout_generation",
        "kernel_submission_plans",
        "backend_plan_generation",
    ),
    ConfigureDispatchAction: (
        "runtime_policies",
        "topology_plans",
        "dispatch_batching",
        "queueing_policies",
        "packet_schedule_generation",
        "overlap_plans",
        "double_buffering",
    ),
    GenerateRuntimeHooksAction: ("runtime_policies", "host_runtime_glue"),
    ConfigureProfilingAction: ("metric_mapping", "trace_correlation"),
    RequestSemanticsAction: (
        "unsupported_op_translation",
        "xdsl_dialect_generation",
        "llvm_patch_generation",
    ),
    GenerateXDSLDialectAction: ("xdsl_dialect_generation",),
    GenerateLLVMPatchAction: ("llvm_patch_generation",),
}

_PACK_DIRECT_SURFACES: dict[type[Action], tuple[str, ...]] = {
    RequestSemanticsAction: ("dialect_semantics", "llvm_intrinsics"),
    GenerateXDSLDialectAction: ("dialect_semantics",),
    GenerateLLVMPatchAction: ("llvm_intrinsics",),
}


def _compute_legal_actions(regions: list[RegionInfo], target: TargetProfile) -> list[LegalAction]:
    """Compute the set of legal actions from the current state."""
    actions: list[LegalAction] = []
    rank = 0

    for region in regions:
        if region.op_type == "matmul":
            # Legal tile sizes based on dimensions
            dim_m = region.input_shapes[0][0] if region.input_shapes else 8
            dim_k = region.input_shapes[0][1] if region.input_shapes and len(region.input_shapes[0]) > 1 else 1
            dim_n = region.output_shapes[0][1] if region.output_shapes and len(region.output_shapes[0]) > 1 else 1

            for tm in _legal_tiles(dim_m):
                for tn in _legal_tiles(dim_n):
                    for tk in _legal_tiles(dim_k):
                        tile = (tm, tn, tk)
                        goodness = min(tm, 256) * min(tn, 256) * min(tk, 64) / (256 * 256 * 64)
                        delta = -region.estimated_latency_us * goodness * 0.3  # up to 30% improvement

                        rank += 1
                        actions.append(LegalAction(
                            action=TileAction(region_id=region.region_id, tile_sizes=tile),
                            estimated_cost_delta_us=delta,
                            estimated_cost_after_us=region.estimated_latency_us + delta,
                            reason=f"Tile {region.region_id} [{tm},{tn},{tk}]",
                            risk="safe",
                            rank=rank,
                        ))

        # Device assignment (for all regions, if multiple devices)
        if len(target.devices) > 1:
            for dev_idx, dev in enumerate(target.devices):
                rank += 1
                actions.append(LegalAction(
                    action=AssignDeviceAction(region_id=region.region_id, device_index=dev_idx),
                    estimated_cost_delta_us=0.0,  # TODO: model per-device cost
                    estimated_cost_after_us=region.estimated_latency_us,
                    reason=f"Place {region.region_id} on {dev.name}",
                    risk="safe",
                    rank=rank,
                ))

    # Generalize: offer for named ops (matmul, transpose)
    named_ops = [r for r in regions if r.op_type in ("matmul", "transpose")]
    if named_ops:
        rank += 1
        actions.append(LegalAction(
            action=GeneralizeAction(region_id="all"),
            estimated_cost_delta_us=0.0,
            estimated_cost_after_us=0.0,
            reason=f"Generalize {len(named_ops)} named ops → linalg.generic",
            risk="safe",
            rank=rank,
        ))

    # Pass menu: always offer safe passes
    for pass_name, pass_info in _PASS_MENU.items():
        if pass_name == "generalize":
            continue  # already offered above
        rank += 1
        actions.append(LegalAction(
            action=ApplyPassAction(pass_name=pass_name),
            estimated_cost_delta_us=0.0,
            estimated_cost_after_us=0.0,
            reason=f"Apply {pass_name}: {pass_info['desc']}",
            risk=pass_info["risk"],
            rank=rank,
        ))

    # Checkpoint and rollback
    rank += 1
    actions.append(LegalAction(
        action=CheckpointAction(),
        estimated_cost_delta_us=0.0,
        estimated_cost_after_us=0.0,
        reason="Save state for speculative exploration",
        risk="safe",
        rank=rank,
    ))

    # Inspect: offer for each region
    for region in regions:
        rank += 1
        actions.append(LegalAction(
            action=InspectAction(region_id=region.region_id),
            estimated_cost_delta_us=0.0,
            estimated_cost_after_us=0.0,
            reason=f"Inspect {region.region_id} ({region.op_type})",
            risk="safe",
            rank=rank,
        ))

    # Analyze: always available
    rank += 1
    actions.append(LegalAction(
        action=AnalyzeAction(),
        estimated_cost_delta_us=0.0,
        estimated_cost_after_us=0.0,
        reason="Analyze network: detect patterns, bottlenecks, kernel opportunities",
        risk="safe",
        rank=rank,
    ))

    # Always include noop
    actions.append(LegalAction(
        action=NoopAction(),
        estimated_cost_delta_us=0.0,
        estimated_cost_after_us=0.0,
        reason="Do nothing",
        risk="safe",
        rank=rank + 1,
    ))

    # Sort by estimated improvement (best first)
    actions.sort(key=lambda a: a.estimated_cost_delta_us)
    for i, a in enumerate(actions):
        actions[i] = LegalAction(
            action=a.action, estimated_cost_delta_us=a.estimated_cost_delta_us,
            estimated_cost_after_us=a.estimated_cost_after_us,
            reason=a.reason, risk=a.risk, rank=i + 1,
        )

    return actions


def _legal_tiles(dim: int) -> list[int]:
    """Return legal tile sizes for a dimension."""
    candidates = [t for t in [16, 32, 64, 128, 256] if t <= dim]
    if not candidates:
        candidates = [dim]
    return candidates

