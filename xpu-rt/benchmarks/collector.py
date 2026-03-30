"""Metric collectors — extract RunRecord fields from CompGen pipeline results."""

from __future__ import annotations

import logging
from typing import Any

from dataclasses import asdict

from benchmarks.record import (
    AgenticMetrics,
    CaptureMetrics,
    CodegenFunnel,
    CodegenRegionDetail,
    EqSatMetrics,
    FallbackPressure,
    IRMetrics,
    LayoutFriction,
    PerformanceMetrics,
    RecipeMetrics,
    SynthesisMetrics,
    SolverMetrics,
)

log = logging.getLogger(__name__)


def collect_capture_metrics(
    export_success: bool,
    graph_break_count: int = 0,
    graph_count: int = 0,
    auto_translations_added: int = 0,
    export_time_ms: float = 0.0,
    decomposition_coverage: float = 0.0,
    total_fx_nodes: int = 0,
    decomposed_ops: int = 0,
    opaque_ops: int = 0,
    analysis_success: bool = False,
    capture_mode: str = "torch_export",
    unsupported_ops: list[str] | None = None,
) -> CaptureMetrics:
    """Collect capture stage metrics."""
    op_coverage = decomposed_ops / max(total_fx_nodes, 1)
    return CaptureMetrics(
        export_success=export_success,
        graph_break_count=graph_break_count,
        graph_count=graph_count,
        auto_translations_added=auto_translations_added,
        op_coverage=op_coverage,
        unsupported_ops=unsupported_ops or [],
        export_time_ms=export_time_ms,
        decomposition_coverage=decomposition_coverage,
        total_fx_nodes=total_fx_nodes,
        decomposed_ops=decomposed_ops,
        opaque_ops=opaque_ops,
        analysis_success=analysis_success,
        capture_mode=capture_mode,
    )


def collect_ir_metrics(module: Any) -> IRMetrics:
    """Collect Payload IR metrics from an xDSL ModuleOp."""
    from xdsl.dialects import builtin, func

    metrics = IRMetrics()
    op_histogram: dict[str, int] = {}

    for op in module.walk():
        if isinstance(op, (builtin.ModuleOp, func.FuncOp, func.ReturnOp)):
            continue
        metrics.total_ops += 1
        op_name = op.name if isinstance(op.name, str) else type(op).__name__
        op_histogram[op_name] = op_histogram.get(op_name, 0) + 1

    metrics.op_type_histogram = op_histogram
    return metrics


def collect_eqsat_metrics(result: Any, eqsat_time_ms: float = 0.0) -> EqSatMetrics:
    """Collect eqsat metrics from EqSatResult."""
    ops_before = result.ops_before
    ops_after = result.ops_after
    reduction = ((ops_before - ops_after) / max(ops_before, 1)) * 100
    return EqSatMetrics(
        ops_before=ops_before,
        ops_after=ops_after,
        ops_reduction_pct=reduction,
        eclasses_initial=result.eclasses_initial,
        eclasses_after_rewrite=result.eclasses_after_rewrite,
        enodes_after_rewrite=result.enodes_after_rewrite,
        rules_applied=dict(result.rule_stats),
        total_rule_applications=sum(result.rule_stats.values()),
        changed=result.changed,
        eqsat_time_ms=eqsat_time_ms,
    )


def collect_solver_metrics(
    placement: Any | None = None,
    schedule: Any | None = None,
    memory: Any | None = None,
) -> SolverMetrics:
    """Collect solver metrics from solution objects."""
    metrics = SolverMetrics()
    if placement is not None:
        metrics.placement_feasible = placement.feasible
        metrics.placement_objective = placement.objective_value
        metrics.placement_gap = placement.gap
        metrics.placement_time_ms = placement.solve_time_ms
        metrics.placement_transfer_cost = placement.transfer_cost
    if schedule is not None:
        metrics.schedule_feasible = schedule.feasible
        metrics.schedule_makespan_us = schedule.makespan_us
        metrics.schedule_time_ms = schedule.solve_time_ms
        metrics.schedule_deadline_met = schedule.deadline_met
    if memory is not None:
        metrics.memory_feasible = memory.feasible
        metrics.memory_peak_bytes = max(memory.peak_per_device.values()) if memory.peak_per_device else 0
        metrics.memory_reuse_count = memory.reuse_count
        metrics.memory_time_ms = memory.solve_time_ms
    return metrics


def collect_recipe_metrics(module: Any) -> RecipeMetrics:
    """Collect Recipe IR metrics from a recipe ModuleOp."""
    from compgen.ir.recipe.ops_candidate import (
        BlackboxOp,
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
        AlternativesOp,
        DeferChoiceOp,
        PromoteCandidateOp,
        RankOp,
        RequireEqsatOp,
        RequireSolverOp,
        SearchBudgetOp,
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
    from compgen.ir.recipe.ops_provenance import (
        FeedbackOp,
        FromAgentOp,
        FromEqsatOp,
        FromTemplateOp,
        LineageOp,
        PromoteOp,
        RejectOp,
    )
    from compgen.ir.recipe.ops_scope import (
        AnchorOp,
        BindPayloadOp,
        RecipeGuardOp,
        RecipeRegionOp,
        SegmentOp,
    )
    from compgen.ir.recipe.ops_verify import (
        RequireCheckFileOp,
        RequireDiffTestOp,
        RequireLayoutInvariantOp,
        RequireMemoryBoundOp,
        RequireProfileBudgetOp,
        RequireTranslationValidationOp,
    )

    scope_types = (RecipeRegionOp, SegmentOp, AnchorOp, RecipeGuardOp, BindPayloadOp)
    fact_types = (BackendAvailableOp, KernelContractOp, TransferCostOp,
                  LocalMemFitOp, FusibleWithOp, CalibrationOp, ExportIssueOp, GraphBreakOp)
    candidate_types = (TileOp, FuseOp, VectorizeOp, ReassociateOp, LayoutNormalizeOp,
                       LowerToAccelOp, RequestTritonKernelOp, RequestExoKernelOp,
                       MaterializeUkernelOp, PlaceOnDeviceOp, InsertCopyBoundaryOp,
                       SegmentBoundaryOp, SelectExoScheduleLibOp, BlackboxOp)
    choice_types = (AlternativesOp, RankOp, SearchBudgetOp, RequireEqsatOp,
                    RequireSolverOp, DeferChoiceOp, PromoteCandidateOp)
    verify_types = (RequireDiffTestOp, RequireTranslationValidationOp,
                    RequireLayoutInvariantOp, RequireMemoryBoundOp,
                    RequireCheckFileOp, RequireProfileBudgetOp)
    provenance_types = (FromAgentOp, FromEqsatOp, FromTemplateOp,
                        FeedbackOp, RejectOp, PromoteOp, LineageOp)

    metrics = RecipeMetrics()
    for op in module.body.block.ops:
        metrics.total_recipe_ops += 1
        if isinstance(op, scope_types):
            metrics.scope_ops += 1
        elif isinstance(op, fact_types):
            metrics.fact_ops += 1
        elif isinstance(op, candidate_types):
            metrics.candidate_ops += 1
        elif isinstance(op, choice_types):
            metrics.choice_ops += 1
        elif isinstance(op, verify_types):
            metrics.verify_ops += 1
        elif isinstance(op, provenance_types):
            metrics.provenance_ops += 1
    return metrics


def collect_agentic_metrics(result: Any) -> AgenticMetrics:
    """Collect agentic loop metrics from CompilationResult."""
    metrics = AgenticMetrics(
        iterations_run=result.iterations_run,
        iterations_improved=result.iterations_improved,
        initial_cost_us=result.initial_cost_us,
        final_cost_us=result.final_cost_us,
        total_improvement_pct=result.total_improvement_pct,
    )
    for record in result.history:
        metrics.iteration_costs.append(record.cost_after_us)
        metrics.iteration_improvements.append(record.improvement_pct)
        metrics.iteration_actions.append(record.action_type)
    # Convergence: last iteration with positive improvement
    for i, imp in enumerate(metrics.iteration_improvements):
        if imp > 0:
            metrics.convergence_iteration = i
    return metrics


def collect_performance_metrics(result: Any) -> PerformanceMetrics:
    """Collect from BenchmarkResult."""
    p90 = 0.0
    per_run_us = list(result.per_run_us)
    if per_run_us:
        index = min(max(int(len(per_run_us) * 0.9), 0), len(per_run_us) - 1)
        p90 = sorted(per_run_us)[index]
    return PerformanceMetrics(
        latency_median_us=result.latency_median_us,
        latency_p90_us=p90,
        latency_p99_us=result.latency_p99_us,
        latency_mean_us=sum(per_run_us) / max(len(per_run_us), 1) if per_run_us else 0.0,
        latency_std_us=_std(per_run_us),
        per_run_us=per_run_us,
        throughput_samples_per_sec=result.throughput_samples_per_sec,
        peak_memory_bytes=result.peak_memory_bytes,
        device=result.device,
        mode=result.mode,
        num_iterations=result.num_iterations,
        warmup_iterations=result.warmup_iterations,
    )


def collect_synthesis_metrics(summary: dict[str, Any] | None = None) -> SynthesisMetrics:
    """Collect synthesized-guard metrics from a summary dictionary."""

    summary = summary or {}
    metrics = SynthesisMetrics(
        fragments_proposed=int(summary.get("fragments_proposed", 0)),
        sound_on_first_attempt=int(summary.get("sound_on_first_attempt", 0)),
        precise_unsound=int(summary.get("precise_unsound", 0)),
        repaired_by_guard=int(summary.get("repaired_by_guard", 0)),
        promoted=int(summary.get("promoted", 0)),
        average_guard_terms=float(summary.get("average_guard_terms", 0.0)),
        average_proof_time_ms=float(summary.get("average_proof_time_ms", 0.0)),
        legality_recall=float(summary.get("legality_recall", 0.0)),
        unsafe_accept_rate=float(summary.get("unsafe_accept_rate", 0.0)),
        profitable_opportunity_recall=float(summary.get("profitable_opportunity_recall", 0.0)),
        missed_opportunity_rate=float(summary.get("missed_opportunity_rate", 0.0)),
        additional_legal_fusions=int(summary.get("additional_legal_fusions", 0)),
        additional_profitable_fusions=int(summary.get("additional_profitable_fusions", 0)),
        additional_local_mem_placements=int(summary.get("additional_local_mem_placements", 0)),
        speedup_passes_only=float(summary.get("speedup_passes_only", 0.0)),
        speedup_guards_only=float(summary.get("speedup_guards_only", 0.0)),
        speedup_combined=float(summary.get("speedup_combined", 0.0)),
        families=dict(summary.get("families", {})),
    )
    return metrics


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


# ---------------------------------------------------------------------------
# Codegen-specific collectors
# ---------------------------------------------------------------------------


def collect_codegen_region_detail(
    decision: Any,
    spec: Any | None = None,
    layout_plan: Any | None = None,
    region_id: str = "",
) -> dict[str, Any]:
    """Build a per-region codegen detail dict from a StrategyDecision.

    Returns ``asdict(CodegenRegionDetail(...))``.
    """
    strategy = decision.strategy.value if hasattr(decision.strategy, "value") else str(decision.strategy)
    op_name = decision.spec.contract.op_name if hasattr(decision.spec, "contract") else ""
    op_family = op_name.split(".")[-1] if "." in op_name else op_name

    perf_target = 0.0
    if spec is not None and hasattr(spec, "perf_target_us") and spec.perf_target_us is not None:
        perf_target = spec.perf_target_us

    search_budget = 0
    backends: list[str] = []
    if spec is not None and hasattr(spec, "search_budget"):
        search_budget = spec.search_budget
    if spec is not None and hasattr(spec, "backends"):
        backends = list(spec.backends)

    prepack = False
    layout_contract = ""
    if layout_plan is not None:
        prepack = bool(getattr(layout_plan, "prepack_candidates", []))
        layout_contract = getattr(layout_plan, "preferred_output_layout", "")

    fallback_reason = ""
    if strategy in ("fallback", "unsupported"):
        fallback_reason = decision.reason

    detail = CodegenRegionDetail(
        region_id=region_id or op_name,
        op_family=op_family,
        selected_strategy=strategy,
        candidate_backends=backends,
        selected_backend=decision.library_name or "",
        search_budget=search_budget,
        perf_target_us=perf_target,
        fallback_reason=fallback_reason,
        layout_contract=layout_contract,
        prepack_applied=prepack,
    )
    return asdict(detail)


def collect_fallback_pressure(
    decisions: list[Any],
    specs: list[Any] | None = None,
) -> FallbackPressure:
    """Compute fallback pressure from strategy decisions."""
    total_flops = 0
    fallback_flops = 0
    fallback_count = 0
    reasons: dict[str, int] = {}

    for i, d in enumerate(decisions):
        strategy = d.strategy.value if hasattr(d.strategy, "value") else str(d.strategy)
        flops = 0
        if specs and i < len(specs):
            flops = getattr(specs[i].contract.cost, "flops", 0) if hasattr(specs[i], "contract") else 0
        elif hasattr(d.spec, "contract"):
            flops = getattr(d.spec.contract.cost, "flops", 0)
        total_flops += flops

        if strategy == "fallback":
            fallback_count += 1
            fallback_flops += flops
            reason = d.reason if hasattr(d, "reason") else "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1

    return FallbackPressure(
        fallback_region_count=fallback_count,
        fallback_flop_share=fallback_flops / max(total_flops, 1),
        fallback_latency_share=0.0,  # needs runtime measurement
        fallback_reasons_histogram=reasons,
    )


def collect_layout_friction(
    layout_plans: dict[str, Any] | None = None,
    region_details: list[dict[str, Any]] | None = None,
) -> LayoutFriction:
    """Compute layout friction from layout plans and region details."""
    plans = layout_plans or {}
    details = region_details or []

    prepacked = 0
    propagated = 0
    for plan in plans.values():
        candidates = getattr(plan, "prepack_candidates", [])
        prepacked += len(candidates)
        if getattr(plan, "tile_encoding", None):
            propagated += 1

    transpose_mats = 0
    opaque_bounds = 0
    for d in details:
        transpose_mats += d.get("transpose_materializations", 0)
        if d.get("opaque_boundary", False):
            opaque_bounds += 1

    return LayoutFriction(
        materialized_transposes=transpose_mats,
        bytes_on_relayout=0,  # needs runtime measurement
        prepacked_operands=prepacked,
        regions_in_propagated_layout=propagated,
        opaque_boundaries_forcing_materialization=opaque_bounds,
    )


def collect_codegen_funnel(
    region_details: list[dict[str, Any]] | None = None,
) -> CodegenFunnel:
    """Compute codegen success funnel from region details."""
    details = region_details or []

    eligible = 0
    attempted = 0
    compiled = 0
    verified = 0
    benchmarked = 0
    faster = 0
    promoted = 0
    speedups: list[float] = []

    for d in details:
        strategy = d.get("selected_strategy", "")
        if strategy in ("native",):
            continue
        eligible += 1
        if d.get("search_iterations_used", 0) > 0:
            attempted += 1
        if d.get("compile_success", False):
            compiled += 1
        if d.get("numeric_pass", False):
            verified += 1
        if d.get("measured_latency_us", 0.0) > 0:
            benchmarked += 1
        su = d.get("speedup_vs_reference", 0.0)
        if su > 1.0:
            faster += 1
            speedups.append(su)
        if d.get("generated_kernel_count", 0) > 0 and d.get("numeric_pass", False):
            promoted += 1

    geo_mean = 0.0
    if speedups:
        import math
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    budget_total = sum(d.get("search_budget", 0) for d in details if d.get("selected_strategy", "") not in ("native",))
    iters_total = sum(d.get("search_iterations_used", 0) for d in details)
    utilization = iters_total / max(budget_total, 1)

    return CodegenFunnel(
        eligible=eligible,
        attempted=attempted,
        compiled=compiled,
        verified=verified,
        benchmarked=benchmarked,
        faster=faster,
        promoted=promoted,
        geo_mean_speedup=geo_mean,
        budget_utilization=utilization,
    )


def collect_realized_backends(
    region_details: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Count regions by realized codegen backend (provider that produced the kernel)."""
    details = region_details or []
    backends: dict[str, int] = {}
    for d in details:
        backend = d.get("provider_backend", "none")
        backends[backend] = backends.get(backend, 0) + 1
    return backends


def collect_kernel_rollups(
    strategy_histogram: dict[str, int],
    region_details: list[dict[str, Any]] | None = None,
    total_compile_time_ms: float = 0.0,
) -> dict[str, float]:
    """Compute model-level kernel rollups from strategy histogram and region details.

    Returns a dict of field names -> values to set on KernelMetrics.
    """
    details = region_details or []
    total = sum(strategy_histogram.values()) or 1

    native = strategy_histogram.get("native", 0)
    library = strategy_histogram.get("library", 0)
    fallback = strategy_histogram.get("fallback", 0)
    unsupported = strategy_histogram.get("unsupported", 0)
    generated = strategy_histogram.get("autocomp", 0) + strategy_histogram.get("exo", 0) + strategy_histogram.get("ukernel", 0)
    opaque = library + unsupported

    verified_count = sum(1 for d in details if d.get("numeric_pass", False))
    region_count = len(details) or total

    # Roofline gap: median of measured / target for regions with both
    gaps = []
    for d in details:
        target = d.get("perf_target_us", 0.0)
        measured = d.get("measured_latency_us", 0.0)
        if target > 0 and measured > 0:
            gaps.append(measured / target)
    gaps.sort()
    roofline_gap = gaps[len(gaps) // 2] if gaps else 0.0

    return {
        "pct_native": native / total * 100,
        "pct_library": library / total * 100,
        "pct_fallback": fallback / total * 100,
        "pct_generated": generated / total * 100,
        "pct_opaque": opaque / total * 100,
        "pct_verified_numerically": verified_count / max(region_count, 1) * 100,
        "compile_ms_per_region": total_compile_time_ms / max(region_count, 1),
        "roofline_gap": roofline_gap,
    }


__all__ = [
    "collect_agentic_metrics",
    "collect_capture_metrics",
    "collect_codegen_funnel",
    "collect_codegen_region_detail",
    "collect_eqsat_metrics",
    "collect_fallback_pressure",
    "collect_ir_metrics",
    "collect_kernel_rollups",
    "collect_layout_friction",
    "collect_realized_backends",
    "collect_performance_metrics",
    "collect_recipe_metrics",
    "collect_synthesis_metrics",
    "collect_solver_metrics",
]
