"""Recipe IR lowering to concrete outputs.

Lowers Recipe IR ops into five output categories:
    1. Transform Dialect scripts (for Payload IR rewrites)
    2. Kernel search jobs (for Autocomp/Triton)
    3. Execution plan fragments (for the solver/planner)
    4. Verification obligations (for the semantic layer)
    5. EqSat job specifications (for the equality saturation pipeline)

Invariants:
    - Lowering is deterministic given the same Recipe IR.
    - Invalid ops produce diagnostics, not crashes.
    - Each output category is independently serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Operation

from compgen.ir.recipe.ops_candidate import (
    FuseOp,
    InsertCopyBoundaryOp,
    LayoutNormalizeOp,
    LowerToAccelOp,
    MaterializeUkernelOp,
    PlaceOnDeviceOp,
    ReassociateOp,
    RequestExoKernelOp,
    RequestTritonKernelOp,
    SegmentBoundaryOp,
    SelectExoScheduleLibOp,
    TileOp,
    VectorizeOp,
)
from compgen.ir.recipe.ops_choice import (
    RequireEqsatOp,
    RequireSolverOp,
)
from compgen.ir.recipe.ops_propose import (
    ProposeDequantFusionOp,
    ProposeFusionOp,
    ProposeLayoutPlanOp,
    ProposeMegakernelSynthesisOp,
    ProposePayload,
)
from compgen.ir.recipe.ops_verify import (
    RequireCheckFileOp,
    RequireDiffTestOp,
    RequireLayoutInvariantOp,
    RequireMemoryBoundOp,
    RequireProfileBudgetOp,
    RequireTranslationValidationOp,
)
from compgen.ir.recipe.ops_scope import RecipeGuardOp
from compgen.semantic.synthesis.facts import RecipeFactIndex, build_candidate_env, build_fact_index
from compgen.semantic.synthesis.registry import GuardRegistry
from compgen.semantic.synthesis.runtime import GuardRuntime, GuardVerdict

log = structlog.get_logger()


CANDIDATE_OP_TYPES = (
    TileOp,
    FuseOp,
    VectorizeOp,
    ReassociateOp,
    LayoutNormalizeOp,
    LowerToAccelOp,
    RequestTritonKernelOp,
    RequestExoKernelOp,
    SelectExoScheduleLibOp,
    MaterializeUkernelOp,
    PlaceOnDeviceOp,
    InsertCopyBoundaryOp,
    SegmentBoundaryOp,
)


@dataclass(frozen=True)
class LoweringOutput:
    """Output from lowering a Recipe IR module."""

    transform_scripts: list[str] = field(default_factory=list)
    kernel_jobs: list[dict[str, Any]] = field(default_factory=list)
    plan_fragments: list[dict[str, Any]] = field(default_factory=list)
    verification_obligations: list[dict[str, Any]] = field(default_factory=list)
    eqsat_jobs: list[dict[str, Any]] = field(default_factory=list)
    guard_verdicts: list[dict[str, Any]] = field(default_factory=list)
    feedback_events: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def lower_recipe(
    module: ModuleOp,
    *,
    guard_registry: GuardRegistry | None = None,
    fact_index: RecipeFactIndex | None = None,
    target_class: str = "",
) -> LoweringOutput:
    """Lower a Recipe IR module to concrete outputs.

    Dispatches each op to its lowering handler based on op type.
    """
    transform_scripts: list[str] = []
    kernel_jobs: list[dict[str, Any]] = []
    plan_fragments: list[dict[str, Any]] = []
    verification_obligations: list[dict[str, Any]] = []
    eqsat_jobs: list[dict[str, Any]] = []
    guard_verdicts: list[dict[str, Any]] = []
    feedback_events: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    runtime = GuardRuntime(guard_registry) if guard_registry is not None else None
    guard_ops = {
        op.sym_name.data: op
        for op in module.walk()
        if isinstance(op, RecipeGuardOp)
    }
    resolved_fact_index = fact_index
    if resolved_fact_index is None and (runtime is not None or guard_ops):
        resolved_fact_index = build_fact_index(module, target_class=target_class)

    for op in module.body.block.ops:
        try:
            _lower_op(
                op,
                transform_scripts,
                kernel_jobs,
                plan_fragments,
                verification_obligations,
                eqsat_jobs,
                guard_ops,
                runtime,
                resolved_fact_index,
                guard_verdicts,
                feedback_events,
                diagnostics,
            )
        except Exception as e:
            diagnostics.append(f"Error lowering {op.name}: {e}")

    log.info(
        "recipe.lowered",
        transforms=len(transform_scripts),
        kernels=len(kernel_jobs),
        plans=len(plan_fragments),
        verifications=len(verification_obligations),
        eqsat=len(eqsat_jobs),
        guards=len(guard_verdicts),
        diagnostics=len(diagnostics),
    )

    return LoweringOutput(
        transform_scripts=transform_scripts,
        kernel_jobs=kernel_jobs,
        plan_fragments=plan_fragments,
        verification_obligations=verification_obligations,
        eqsat_jobs=eqsat_jobs,
        guard_verdicts=guard_verdicts,
        feedback_events=feedback_events,
        diagnostics=diagnostics,
    )


def _lower_op(
    op: Operation,
    transform_scripts: list[str],
    kernel_jobs: list[dict[str, Any]],
    plan_fragments: list[dict[str, Any]],
    verification_obligations: list[dict[str, Any]],
    eqsat_jobs: list[dict[str, Any]],
    guard_ops: dict[str, RecipeGuardOp],
    guard_runtime: GuardRuntime | None,
    fact_index: RecipeFactIndex | None,
    guard_verdicts: list[dict[str, Any]],
    feedback_events: list[dict[str, Any]],
    diagnostics: list[str],
) -> None:
    """Dispatch a single op to its lowering handler."""

    if isinstance(op, CANDIDATE_OP_TYPES) and not _candidate_guards_allow(
        op,
        guard_ops,
        guard_runtime,
        fact_index,
        guard_verdicts,
        feedback_events,
        diagnostics,
    ):
        return

    # --- Candidate ops → Transform scripts ---
    if isinstance(op, TileOp):
        _lower_tile(op, transform_scripts)
    elif isinstance(op, FuseOp):
        _lower_fuse(op, transform_scripts)
    elif isinstance(op, VectorizeOp):
        _lower_vectorize(op, transform_scripts)
    elif isinstance(op, ReassociateOp):
        _lower_reassociate(op, transform_scripts)
    elif isinstance(op, LayoutNormalizeOp):
        _lower_layout_normalize(op, transform_scripts)

    # --- Candidate ops → Kernel jobs ---
    elif isinstance(op, RequestTritonKernelOp):
        _lower_request_kernel(op, kernel_jobs)
    elif isinstance(op, RequestExoKernelOp):
        _lower_request_exo_kernel(op, kernel_jobs)
    elif isinstance(op, SelectExoScheduleLibOp):
        _lower_select_exo_schedule(op, kernel_jobs)
    elif isinstance(op, MaterializeUkernelOp):
        _lower_materialize_ukernel(op, kernel_jobs)
    elif isinstance(op, LowerToAccelOp):
        _lower_to_accel(op, kernel_jobs)

    # --- Placement/planning ops → Plan fragments ---
    elif isinstance(op, PlaceOnDeviceOp):
        _lower_place_on_device(op, plan_fragments)
    elif isinstance(op, InsertCopyBoundaryOp):
        _lower_copy_boundary(op, plan_fragments)
    elif isinstance(op, SegmentBoundaryOp):
        _lower_segment_boundary(op, plan_fragments)
    elif isinstance(op, RequireSolverOp):
        _lower_require_solver(op, plan_fragments)

    # --- Choice ops → EqSat ---
    elif isinstance(op, RequireEqsatOp):
        _lower_require_eqsat(op, eqsat_jobs)

    # --- Propose ops (LLM invent-slots) → transform scripts + sometimes kernel jobs ---
    elif isinstance(op, ProposeFusionOp):
        _lower_propose_fusion(op, transform_scripts, verification_obligations)
    elif isinstance(op, ProposeMegakernelSynthesisOp):
        _lower_propose_megakernel(op, kernel_jobs, verification_obligations)
    elif isinstance(op, ProposeLayoutPlanOp):
        _lower_propose_layout(op, transform_scripts, verification_obligations)
    elif isinstance(op, ProposeDequantFusionOp):
        _lower_propose_dequant(op, transform_scripts, verification_obligations)

    # --- Verification obligation ops ---
    elif isinstance(op, RequireDiffTestOp):
        _lower_require_diff_test(op, verification_obligations)
    elif isinstance(op, RequireTranslationValidationOp):
        _lower_require_tv(op, verification_obligations)
    elif isinstance(op, RequireLayoutInvariantOp):
        _lower_require_layout(op, verification_obligations)
    elif isinstance(op, RequireMemoryBoundOp):
        _lower_require_memory(op, verification_obligations)
    elif isinstance(op, RequireCheckFileOp):
        _lower_require_check_file(op, verification_obligations)
    elif isinstance(op, RequireProfileBudgetOp):
        _lower_require_profile(op, verification_obligations)

    # Scope, fact, provenance ops don't lower to anything — they are metadata


# ---- Transform script lowering ----


def _sym_ref_str(ref: SymbolRefAttr) -> str:
    return ref.root_reference.data


def _int_attr_val(attr: IntegerAttr) -> int:
    return attr.value.data


def _str_attr_val(attr: StringAttr) -> str:
    return attr.data


def _candidate_symbol(op: Operation) -> str:
    if hasattr(op, "sym_name") and getattr(op, "sym_name") is not None:
        return getattr(op, "sym_name").data
    return ""


def _guard_ref_names(op: Operation) -> list[str]:
    if not hasattr(op, "guard_refs") or getattr(op, "guard_refs") is None:
        return []
    guard_refs = getattr(op, "guard_refs")
    return [
        ref.root_reference.data
        for ref in guard_refs.data
        if isinstance(ref, SymbolRefAttr)
    ]


def _verdict_to_dict(verdict: GuardVerdict, *, guard_ref: str, candidate_ref: str) -> dict[str, Any]:
    return {
        "candidate_ref": candidate_ref,
        "guard_ref": guard_ref,
        "guard_key": verdict.guard_key,
        "allow": verdict.allow,
        "reason": verdict.reason,
        "fragments_evaluated": verdict.fragments_evaluated,
        "failed_fragment_index": verdict.failed_fragment_index,
        "details": verdict.details,
    }


def _candidate_guards_allow(
    op: Operation,
    guard_ops: dict[str, RecipeGuardOp],
    guard_runtime: GuardRuntime | None,
    fact_index: RecipeFactIndex | None,
    guard_verdicts: list[dict[str, Any]],
    feedback_events: list[dict[str, Any]],
    diagnostics: list[str],
) -> bool:
    guard_refs = _guard_ref_names(op)
    if not guard_refs:
        return True

    candidate_ref = _candidate_symbol(op)
    if guard_runtime is None:
        diagnostics.append(f"Guarded candidate skipped ({op.name}): no guard registry/runtime available")
        for guard_ref in guard_refs:
            payload = {
                "candidate_ref": candidate_ref,
                "guard_ref": guard_ref,
                "allow": False,
                "reason": "missing_guard_runtime",
            }
            guard_verdicts.append(payload)
            feedback_events.append(payload)
        return False

    if fact_index is None:
        diagnostics.append(f"Guarded candidate skipped ({op.name}): no fact index available")
        for guard_ref in guard_refs:
            payload = {
                "candidate_ref": candidate_ref,
                "guard_ref": guard_ref,
                "allow": False,
                "reason": "missing_fact_index",
            }
            guard_verdicts.append(payload)
            feedback_events.append(payload)
        return False

    env = build_candidate_env(op, fact_index)
    for guard_ref in guard_refs:
        guard_op = guard_ops.get(guard_ref)
        if guard_op is None:
            payload = {
                "candidate_ref": candidate_ref,
                "guard_ref": guard_ref,
                "allow": False,
                "reason": "unknown_guard_ref",
            }
            diagnostics.append(f"Guarded candidate skipped ({op.name}): unresolved guard @{guard_ref}")
            guard_verdicts.append(payload)
            feedback_events.append(payload)
            return False
        verdict = guard_runtime.evaluate(guard_op.guard_key.data, env)
        payload = _verdict_to_dict(verdict, guard_ref=guard_ref, candidate_ref=candidate_ref)
        guard_verdicts.append(payload)
        feedback_events.append(payload)
        if not verdict.allow:
            diagnostics.append(
                f"Guarded candidate rejected ({op.name}): guard @{guard_ref} ({guard_op.guard_key.data}) -> {verdict.reason}"
            )
            return False
    return True


def _lower_tile(op: TileOp, out: list[str]) -> None:
    region = _sym_ref_str(op.region_ref)
    sizes = [_int_attr_val(s) for s in op.tile_sizes.data if isinstance(s, IntegerAttr)]
    sizes_str = ", ".join(str(s) for s in sizes)
    script = (
        f'// Tile {region} with sizes [{sizes_str}]\n'
        f'transform.structured.tile_using_forall %{region}\n'
        f'  tile_sizes [{sizes_str}]'
    )
    if op.interchange is not None:
        ic = [_int_attr_val(i) for i in op.interchange.data if isinstance(i, IntegerAttr)]
        script += f'\n  interchange [{", ".join(str(i) for i in ic)}]'
    out.append(script)


def _lower_fuse(op: FuseOp, out: list[str]) -> None:
    regions = [_sym_ref_str(r) for r in op.fuse_regions.data if isinstance(r, SymbolRefAttr)]
    kind = _str_attr_val(op.fusion_kind) if op.fusion_kind else "producer_consumer"
    script = (
        f'// Fuse regions: {", ".join(regions)}\n'
        f'transform.structured.fuse_into_containing_op\n'
        f'  targets [{", ".join(f"%{r}" for r in regions)}]\n'
        f'  fusion_kind = "{kind}"'
    )
    out.append(script)


# ---- Propose-op lowerings (LLM invent-slot outputs) ------------------------
#
# Propose-ops are the LLM's accepted proposals, appended to the recipe
# module by :func:`compgen.agent.recipe_bridge_invent.proposal_to_recipe_op`.
# Here we turn them into the same downstream artifacts candidate ops produce
# (transform scripts / kernel jobs / verification obligations), plus a
# verification obligation specific to each proposal family so the semantic
# executor can gate the lowered transform.


def _lower_propose_fusion(
    op: ProposeFusionOp,
    transform_scripts: list[str],
    verification_obligations: list[dict[str, Any]],
) -> None:
    regions = [
        _sym_ref_str(r) for r in op.grouped_regions.data
        if isinstance(r, SymbolRefAttr)
    ]
    if not regions:
        return
    try:
        payload = ProposePayload.from_json(op.payload.data)
    except Exception:   # noqa: BLE001
        payload = ProposePayload()
    fusion_kind = str(payload.chosen.get("fusion_kind", "producer_consumer"))
    sym = _candidate_symbol(op) or "propose_fusion"
    script = (
        f"// propose_fusion[{sym}]: {', '.join(regions)} (kind={fusion_kind})\n"
        f"// target_feature_justification: {payload.target_feature_justification}\n"
        f"transform.structured.fuse_into_containing_op\n"
        f"  targets [{', '.join(f'%{r}' for r in regions)}]\n"
        f'  fusion_kind = "{fusion_kind}"'
    )
    transform_scripts.append(script)
    # Every LLM-invented fusion MUST be differential-tested before the
    # bundle is promoted. The obligation anchors to the first grouped
    # region; the executor walks the fused set from there.
    verification_obligations.append({
        "type": "differential",
        "region_id": regions[0],
        "kind": "propose_fusion",
        "grouped_regions": regions,
        "source_op": op.name,
    })


def _lower_propose_megakernel(
    op: ProposeMegakernelSynthesisOp,
    kernel_jobs: list[dict[str, Any]],
    verification_obligations: list[dict[str, Any]],
) -> None:
    regions = [
        _sym_ref_str(r) for r in op.fused_region_refs.data
        if isinstance(r, SymbolRefAttr)
    ]
    if not regions:
        return
    try:
        payload = ProposePayload.from_json(op.payload.data)
    except Exception:   # noqa: BLE001
        payload = ProposePayload()
    sym = _candidate_symbol(op) or "propose_megakernel"
    megakernel_name = str(payload.chosen.get("megakernel_name", sym))
    kernel_jobs.append({
        "type": "megakernel_synthesis",
        "kernel_name": megakernel_name,
        "fused_regions": regions,
        "event_tensor_decls": payload.chosen.get("event_tensor_decls", []),
        "task_partition": payload.chosen.get("task_partition", {}),
        "prefetch_annotations": payload.chosen.get("prefetch_annotations", []),
        "source_op": op.name,
        "target_feature_justification": payload.target_feature_justification,
    })
    verification_obligations.append({
        "type": "translation_validation",
        "region_id": regions[0],
        "kind": "propose_megakernel_synthesis",
        "fused_regions": regions,
        "source_op": op.name,
    })


def _lower_propose_layout(
    op: ProposeLayoutPlanOp,
    transform_scripts: list[str],
    verification_obligations: list[dict[str, Any]],
) -> None:
    region = _sym_ref_str(op.region_ref)
    try:
        payload = ProposePayload.from_json(op.payload.data)
    except Exception:   # noqa: BLE001
        payload = ProposePayload()
    layout = str(payload.chosen.get("layout", "row_major"))
    sym = _candidate_symbol(op) or "propose_layout"
    script = (
        f"// propose_layout[{sym}]: {region} -> {layout}\n"
        f"// target_feature_justification: {payload.target_feature_justification}\n"
        f"transform.structured.pack %{region}\n"
        f'  layout = "{layout}"'
    )
    transform_scripts.append(script)
    verification_obligations.append({
        "type": "layout_invariant",
        "region_id": region,
        "kind": "propose_layout_plan",
        "layout": layout,
        "source_op": op.name,
    })


def _lower_propose_dequant(
    op: ProposeDequantFusionOp,
    transform_scripts: list[str],
    verification_obligations: list[dict[str, Any]],
) -> None:
    region = _sym_ref_str(op.region_ref)
    try:
        payload = ProposePayload.from_json(op.payload.data)
    except Exception:   # noqa: BLE001
        payload = ProposePayload()
    pattern = str(payload.chosen.get("pattern", "generic_dequant"))
    sym = _candidate_symbol(op) or "propose_dequant"
    script = (
        f"// propose_dequant_fusion[{sym}]: {region} (pattern={pattern})\n"
        f"// Lowered as a placeholder; payload carries full scheme.\n"
        f"transform.structured.match ops{{[\"linalg.generic\"]}} in %{region}"
    )
    transform_scripts.append(script)
    verification_obligations.append({
        "type": "differential",
        "region_id": region,
        "kind": "propose_dequant_fusion",
        "pattern": pattern,
        "tolerance_hint": payload.chosen.get("tolerance_hint"),
        "source_op": op.name,
    })


def _lower_vectorize(op: VectorizeOp, out: list[str]) -> None:
    region = _sym_ref_str(op.region_ref)
    width = _int_attr_val(op.vector_width)
    script = (
        f'// Vectorize {region} with width {width}\n'
        f'transform.structured.vectorize %{region}\n'
        f'  vector_sizes [{width}]'
    )
    out.append(script)


def _lower_reassociate(op: ReassociateOp, out: list[str]) -> None:
    region = _sym_ref_str(op.region_ref)
    strategy = _str_attr_val(op.strategy)
    out.append(
        f'// Reassociate {region} ({strategy})\n'
        f'transform.apply_patterns.reassociate %{region}\n'
        f'  strategy = "{strategy}"'
    )


def _lower_layout_normalize(op: LayoutNormalizeOp, out: list[str]) -> None:
    region = _sym_ref_str(op.region_ref)
    layout = _str_attr_val(op.target_layout)
    out.append(
        f'// Normalize layout of {region} to {layout}\n'
        f'transform.apply_patterns.layout_normalize %{region}\n'
        f'  target_layout = "{layout}"'
    )


# ---- Kernel job lowering ----


def _lower_request_kernel(op: RequestTritonKernelOp, out: list[dict[str, Any]]) -> None:
    job: dict[str, Any] = {
        "type": "kernel_search",
        "region_id": _sym_ref_str(op.region_ref),
        "search_budget": _int_attr_val(op.search_budget),
        "backend": _str_attr_val(op.backend) if op.backend else "autocomp",
    }
    if op.kernel_family:
        job["kernel_family"] = _str_attr_val(op.kernel_family)
    out.append(job)


def _lower_materialize_ukernel(op: MaterializeUkernelOp, out: list[dict[str, Any]]) -> None:
    job: dict[str, Any] = {
        "type": "ukernel_materialize",
        "region_id": _sym_ref_str(op.region_ref),
        "kernel_name": _str_attr_val(op.kernel_name),
    }
    if op.calling_convention:
        job["calling_convention"] = _str_attr_val(op.calling_convention)
    out.append(job)


def _lower_to_accel(op: LowerToAccelOp, out: list[dict[str, Any]]) -> None:
    job: dict[str, Any] = {
        "type": "accel_lowering",
        "region_id": _sym_ref_str(op.region_ref),
    }
    if op.accel_cluster:
        job["accel_cluster"] = _str_attr_val(op.accel_cluster)
    out.append(job)


# ---- Plan fragment lowering ----


def _lower_place_on_device(op: PlaceOnDeviceOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "placement",
        "region_id": _sym_ref_str(op.region_ref),
        "device_index": op.device.index.value.data,
        "device_name": op.device.device_name.data,
        "reason": _str_attr_val(op.reason) if op.reason else "",
    })


def _lower_copy_boundary(op: InsertCopyBoundaryOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "copy_boundary",
        "src_region": _sym_ref_str(op.src_region),
        "dst_region": _sym_ref_str(op.dst_region),
        "tensor_name": _str_attr_val(op.tensor_name),
        "is_async": bool(op.is_async and _int_attr_val(op.is_async)),
    })


def _lower_segment_boundary(op: SegmentBoundaryOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "segment_boundary",
        "after_region": _sym_ref_str(op.after_region),
        "reason": _str_attr_val(op.reason) if op.reason else "",
    })


def _lower_require_solver(op: RequireSolverOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "solver",
        "solve_type": _str_attr_val(op.solve_type),
        "timeout_ms": _int_attr_val(op.timeout_ms) if op.timeout_ms else None,
    })


# ---- EqSat job lowering ----


def _lower_require_eqsat(op: RequireEqsatOp, out: list[dict[str, Any]]) -> None:
    job: dict[str, Any] = {
        "type": "eqsat",
        "region_id": _sym_ref_str(op.region_ref),
    }
    if op.rule_categories:
        job["rule_categories"] = [
            _str_attr_val(c) for c in op.rule_categories.data
            if isinstance(c, StringAttr)
        ]
    if op.max_iterations:
        job["max_iterations"] = _int_attr_val(op.max_iterations)
    out.append(job)


# ---- Verification obligation lowering ----


def _lower_require_diff_test(op: RequireDiffTestOp, out: list[dict[str, Any]]) -> None:
    obligation: dict[str, Any] = {
        "type": "differential",
        "region_id": _sym_ref_str(op.region_ref),
    }
    if op.tolerance:
        obligation["tolerance_ulps"] = _int_attr_val(op.tolerance)
    out.append(obligation)


def _lower_require_tv(op: RequireTranslationValidationOp, out: list[dict[str, Any]]) -> None:
    obligation: dict[str, Any] = {
        "type": "translation_validation",
        "region_id": _sym_ref_str(op.region_ref),
    }
    if op.source_op:
        obligation["source_op"] = _str_attr_val(op.source_op)
    if op.target_op:
        obligation["target_op"] = _str_attr_val(op.target_op)
    out.append(obligation)


def _lower_require_layout(op: RequireLayoutInvariantOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "layout_invariant",
        "region_id": _sym_ref_str(op.region_ref),
        "expected_layout": _str_attr_val(op.expected_layout),
    })


def _lower_require_memory(op: RequireMemoryBoundOp, out: list[dict[str, Any]]) -> None:
    obligation: dict[str, Any] = {
        "type": "memory_bound",
        "region_id": _sym_ref_str(op.region_ref),
        "max_bytes": _int_attr_val(op.max_bytes),
    }
    if op.device:
        obligation["device_index"] = op.device.index.value.data
    out.append(obligation)


def _lower_require_check_file(op: RequireCheckFileOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "check_file",
        "path": _str_attr_val(op.check_file_path),
    })


def _lower_require_profile(op: RequireProfileBudgetOp, out: list[dict[str, Any]]) -> None:
    obligation: dict[str, Any] = {
        "type": "profile_budget",
        "region_id": _sym_ref_str(op.region_ref),
        "max_latency_us": _int_attr_val(op.max_latency_us),
    }
    if op.device:
        obligation["device_index"] = op.device.index.value.data
    out.append(obligation)


# ---- Exo kernel job lowering ----


def _lower_request_exo_kernel(op: RequestExoKernelOp, out: list[dict[str, Any]]) -> None:
    job: dict[str, Any] = {
        "type": "exo_kernel_search",
        "region_id": _sym_ref_str(op.region_ref),
        "search_budget": _int_attr_val(op.search_budget),
        "backend": "exo",
    }
    if op.schedule_lib:
        job["schedule_lib"] = _str_attr_val(op.schedule_lib)
    if op.target_kit:
        job["target_kit"] = _str_attr_val(op.target_kit)
    if op.kernel_family:
        job["kernel_family"] = _str_attr_val(op.kernel_family)
    out.append(job)


def _lower_select_exo_schedule(op: SelectExoScheduleLibOp, out: list[dict[str, Any]]) -> None:
    out.append({
        "type": "exo_schedule_lib",
        "region_id": _sym_ref_str(op.region_ref),
        "lib_name": _str_attr_val(op.lib_name),
        "version": _str_attr_val(op.version) if op.version else None,
    })


# --- Backward compatibility ---

from compgen.ir.recipe.ops import RecipeOp  # noqa: E402


def lower_recipe_ops(ops: list[RecipeOp]) -> LoweringOutput:
    """DEPRECATED: Lower old dataclass RecipeOps.

    Converts to xDSL module first via compat.py, then lowers.
    """
    from compgen.ir.recipe.compat import recipe_list_to_module
    module = recipe_list_to_module(ops)
    return lower_recipe(module)


__all__ = ["LoweringOutput", "lower_recipe", "lower_recipe_ops"]
