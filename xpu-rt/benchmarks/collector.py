"""Metric collectors — extract RunRecord fields from CompGen pipeline results."""

from __future__ import annotations

import logging
from typing import Any

from benchmarks.record import (
    AgenticMetrics,
    CaptureMetrics,
    EqSatMetrics,
    IRMetrics,
    PerformanceMetrics,
    RecipeMetrics,
    SolverMetrics,
)

log = logging.getLogger(__name__)


def collect_capture_metrics(
    export_success: bool,
    graph_break_count: int = 0,
    export_time_ms: float = 0.0,
    decomposition_coverage: float = 0.0,
    total_fx_nodes: int = 0,
    decomposed_ops: int = 0,
    opaque_ops: int = 0,
    unsupported_ops: list[str] | None = None,
) -> CaptureMetrics:
    """Collect capture stage metrics."""
    op_coverage = decomposed_ops / max(total_fx_nodes, 1)
    return CaptureMetrics(
        export_success=export_success,
        graph_break_count=graph_break_count,
        op_coverage=op_coverage,
        unsupported_ops=unsupported_ops or [],
        export_time_ms=export_time_ms,
        decomposition_coverage=decomposition_coverage,
        total_fx_nodes=total_fx_nodes,
        decomposed_ops=decomposed_ops,
        opaque_ops=opaque_ops,
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

    scope_types = (RecipeRegionOp, SegmentOp, AnchorOp, BindPayloadOp)
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
    return PerformanceMetrics(
        latency_median_us=result.latency_median_us,
        latency_p99_us=result.latency_p99_us,
        latency_mean_us=sum(result.per_run_us) / max(len(result.per_run_us), 1) if result.per_run_us else 0.0,
        latency_std_us=_std(result.per_run_us),
        per_run_us=list(result.per_run_us),
        throughput_samples_per_sec=result.throughput_samples_per_sec,
        peak_memory_bytes=result.peak_memory_bytes,
        device=result.device,
        mode=result.mode,
        num_iterations=result.num_iterations,
        warmup_iterations=result.warmup_iterations,
    )


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


__all__ = [
    "collect_agentic_metrics",
    "collect_capture_metrics",
    "collect_eqsat_metrics",
    "collect_ir_metrics",
    "collect_performance_metrics",
    "collect_recipe_metrics",
    "collect_solver_metrics",
]
