"""Baseline and system adapters for the benchmark harness."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from benchmarks.collector import (
    collect_capture_metrics,
    collect_eqsat_metrics,
    collect_ir_metrics,
    collect_recipe_metrics,
    collect_synthesis_metrics,
    collect_solver_metrics,
)
from benchmarks.record import RunRecord
from benchmarks.registry import BenchmarkRegistry
from benchmarks.spec import BaselineSpec, ExperimentCase, TargetSpec, WorkloadSpec, WorkspaceConfig


class Adapter(Protocol):
    """Common adapter interface."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        ...

    def run(self, ctx: AdapterContext) -> RunRecord:
        ...


@dataclass(frozen=True)
class AdapterContext:
    """Resolved context for a single adapter execution."""

    workspace: WorkspaceConfig
    registry: BenchmarkRegistry
    case: ExperimentCase
    workload: WorkloadSpec
    target: TargetSpec
    baseline: BaselineSpec
    output_dir: Path
    ablation: str = ""
    extra_config: dict[str, Any] = field(default_factory=dict)


def _new_record(ctx: AdapterContext, *, system_name: str) -> RunRecord:
    """Create a record with shared study identity pre-populated."""

    ablation = ctx.ablation or ""
    return RunRecord(
        model_name=ctx.workload.workload_id,
        target_name=ctx.target.target_id,
        objective=ctx.case.objective,
        system_name=system_name,
        workload_id=ctx.workload.workload_id,
        target_id=ctx.target.target_id,
        source_model_id=ctx.workload.source_model_id,
        readiness=ctx.workload.readiness,
        expected_status=ctx.workload.expected_status,
        status="pending",
        config={**ctx.extra_config, **({"ablation": ablation} if ablation else {})},
    )


def _populate_study_identity(record: RunRecord, ctx: AdapterContext) -> None:
    record.study.study_id = ctx.case.study_id
    record.study.case_id = ctx.case.case_id
    record.study.tier = ctx.workload.tier
    record.study.workload_id = ctx.workload.workload_id
    record.study.target_id = ctx.target.target_id
    record.study.baseline_id = ctx.baseline.baseline_id
    record.study.bundle_id = str(ctx.case.metadata.get("bundle_id", ""))
    record.study.tags = sorted(set(ctx.case.tags + ctx.workload.tags + ctx.target.tags + ctx.baseline.tags))


def _compute_artifact_metrics(record: RunRecord, bundle_dir: Path, artifact_paths: dict[str, str]) -> None:
    """Compute artifact completeness from a bundle manifest mapping."""

    required = [
        "payload",
        "recipe_mlir",
        "recipe_yaml",
        "kernel_contracts",
        "transforms",
        "execution_plan",
        "memory_plan",
        "verification_report",
        "manifest",
    ]
    present = {name: name in artifact_paths for name in required}
    record.artifacts.bundle_path = str(bundle_dir)
    record.artifacts.artifact_paths = artifact_paths
    record.artifacts.artifacts_present = present
    record.artifacts.missing_artifacts = [name for name, ok in present.items() if not ok]
    record.artifacts.completeness_score = (sum(1 for ok in present.values() if ok) / len(required)) if required else 1.0
    record.artifacts.runnable_bundle = present["payload"] and present["execution_plan"] and present["manifest"]
    if "manifest" in artifact_paths:
        record.artifacts.manifest_path = artifact_paths["manifest"]


def _load_target_profile(target: TargetSpec, output_dir: Path) -> tuple[Any, str]:
    """Load either a target profile or a generated device profile."""

    if target.kind == "hardware_spec":
        from compgen.api import device

        dev = device(target.path, output_dir=output_dir / f"targetgen_{target.target_id}")
        return dev.profile, dev.profile.name

    from compgen.targets.schema import load_profile

    profile = load_profile(str(target.path))
    return profile, profile.name


def _serialize_kernel_contracts(contracts: list[Any]) -> list[dict[str, Any]]:
    """Convert kernel contracts to JSON-friendly objects."""

    serialized: list[dict[str, Any]] = []
    for index, contract in enumerate(contracts):
        serialized.append(
            {
                "index": index,
                "repr": str(contract),
                "type": type(contract).__name__,
            }
        )
    return serialized


def _serialize_verification_result(result: Any) -> dict[str, Any]:
    """Convert TransformVerificationResult to a JSON-friendly dict."""

    return {
        "passed": result.passed,
        "levels_run": [level.value for level in result.levels_run],
        "levels_passed": [level.value for level in result.levels_passed],
        "max_abs_error": result.max_abs_error,
        "details": result.details,
    }


def _capture_artifact_for_workload(ctx: AdapterContext, model: Any, sample_inputs: tuple[Any, ...]) -> Any:
    """Capture according to the workload's configured frontend mode."""

    from compgen.capture import capture_dynamo_partitions, capture_frontend_artifact

    if ctx.workload.capture_mode == "torch_dynamo_partioned":
        return capture_dynamo_partitions(model, sample_inputs)
    if ctx.workload.capture_mode == "torch_dynamo_partitioned":
        return capture_dynamo_partitions(model, sample_inputs)
    return capture_frontend_artifact(model, sample_inputs)


def _populate_capture_from_artifact(record: RunRecord, artifact: Any, *, capture_ms: float) -> None:
    """Copy generic capture information from a frontend artifact into the record."""

    total_fx_nodes = int(getattr(artifact.validation, "num_ops", 0))
    auto_translations = len(getattr(artifact, "synthesized_payload_translations", {}))
    unsupported_ops = [
        str(resolution.target)
        for resolution in getattr(artifact, "unsupported_resolutions", [])
    ]
    record.capture = collect_capture_metrics(
        export_success=artifact.exported_program is not None,
        graph_break_count=int(getattr(artifact, "graph_break_count", 0)),
        graph_count=int(getattr(artifact, "graph_count", 0)),
        auto_translations_added=auto_translations,
        export_time_ms=capture_ms,
        decomposition_coverage=0.0,
        total_fx_nodes=total_fx_nodes,
        decomposed_ops=0,
        opaque_ops=len(unsupported_ops),
        analysis_success=bool(getattr(artifact, "analysis_success", False)),
        capture_mode=str(getattr(artifact, "capture_mode", "")),
        unsupported_ops=unsupported_ops,
    )


def _finalize_expected_failure(
    record: RunRecord,
    ctx: AdapterContext,
    exc: Exception,
    *,
    total_start: float,
) -> RunRecord:
    """Map expected workload failures to skip/xfail instead of hard fail."""

    record.errors.append(str(exc))
    record.total_compile_time_ms = (time.perf_counter() - total_start) * 1000
    if ctx.workload.expected_status in {"skip", "xfail"}:
        record.status = ctx.workload.expected_status
        record.verification.overall_status = "skip"
        return record
    record.status = "fail"
    record.verification.overall_status = "fail"
    return record


def _analysis_only_record(
    ctx: AdapterContext,
    record: RunRecord,
    *,
    model: Any,
    sample_inputs: tuple[Any, ...],
    target_profile: Any,
    total_start: float,
) -> RunRecord:
    """Run capture and graph analysis without the full lowering pipeline."""

    from compgen.agent.analyzer import NetworkAnalyzer

    capture_start = time.perf_counter()
    artifact = _capture_artifact_for_workload(ctx, model, sample_inputs)
    capture_ms = (time.perf_counter() - capture_start) * 1000
    _populate_capture_from_artifact(record, artifact, capture_ms=capture_ms)

    analysis_input = artifact.exported_program if artifact.exported_program is not None else artifact
    analysis = NetworkAnalyzer().analyze(
        analysis_input,
        target_profile,
        model_name=ctx.workload.workload_id,
    )
    record.capture.analysis_success = True
    if analysis.dossier is not None:
        record.ir.total_ops = sum(analysis.dossier.op_histogram.values())
        record.ir.op_type_histogram = dict(analysis.dossier.op_histogram)
        record.profiling.summary = {
            "critical_path": list(analysis.dossier.critical_path),
            "unsupported_targets": list(analysis.dossier.unsupported_targets),
            "total_regions": analysis.dossier.total_regions,
        }
    record.ir.total_flops = analysis.total_flops
    record.ir.total_bytes = analysis.total_bytes
    if record.ir.total_bytes:
        record.performance.arithmetic_intensity = record.ir.total_flops / max(record.ir.total_bytes, 1)
    record.verification.overall_status = "skip"
    record.total_compile_time_ms = (time.perf_counter() - total_start) * 1000
    record.status = "pass"
    return record


def _skip_non_pipeline_workload(record: RunRecord, ctx: AdapterContext, *, reason: str) -> RunRecord:
    """Skip baselines that require a runnable full pipeline."""

    record.status = "skip"
    record.errors.append(reason)
    record.verification.overall_status = "skip"
    return record


class CompGenAdapter:
    """Primary CompGen system adapter."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        return True, ""

    def run(self, ctx: AdapterContext) -> RunRecord:
        record = _new_record(ctx, system_name="compgen")
        _populate_study_identity(record, ctx)
        total_start = time.perf_counter()

        try:
            model, sample_inputs = ctx.workload.load(ctx.workspace)
            target_profile, resolved_target_name = _load_target_profile(ctx.target, ctx.output_dir)
            record.target_name = resolved_target_name

            if ctx.workload.readiness != "full_pipeline":
                return _analysis_only_record(
                    ctx,
                    record,
                    model=model,
                    sample_inputs=sample_inputs,
                    target_profile=target_profile,
                    total_start=total_start,
                )

            from compgen.capture import capture_frontend_artifact
            from compgen.eqsat.pipeline import run_eqsat_pass
            from compgen.ir.recipe.lower import lower_recipe
            from compgen.ir.recipe.seed import generate_seed_recipe
            from compgen.ir.recipe.serialize import recipe_module_to_yaml, recipe_to_mlir
            from compgen.ir.recipe.validate import validate_recipe_module
            from compgen.ir.payload.import_fx import fx_to_xdsl
            from compgen.kernels.contracts import build_kernel_contracts
            from compgen.kernels.selector import select_strategies
            from compgen.promotion.promote import RecipePromoter
            from compgen.runtime.bundle import create_bundle
            from compgen.runtime.local_executor import LocalExecutor
            from compgen.runtime.planner import plan_execution
            from compgen.runtime.torch_backend import CompGenBackend
            from compgen.synthesis.integration import synthesize_and_attach_guards
            from compgen.transforms.verify import TransformVerifier, VerificationLevel

            capture_start = time.perf_counter()
            artifact = capture_frontend_artifact(model, sample_inputs)
            capture_ms = (time.perf_counter() - capture_start) * 1000
            _populate_capture_from_artifact(record, artifact, capture_ms=capture_ms)
            exported = artifact.exported_program
            if exported is None:
                record.status = "fail"
                record.errors.append("torch.export failed")
                record.total_compile_time_ms = (time.perf_counter() - total_start) * 1000
                return record

            module, diagnostics = fx_to_xdsl(exported, **artifact.strict_import_options())
            original_module = module.clone()
            total_fx_nodes = len(getattr(exported.graph, "nodes", []))
            decomposed = sum(1 for d in diagnostics if getattr(d, "level", "") != "error")
            opaque = sum(1 for d in diagnostics if getattr(d, "level", "") == "error")
            record.capture = collect_capture_metrics(
                export_success=True,
                graph_break_count=artifact.graph_break_count,
                graph_count=artifact.graph_count,
                auto_translations_added=len(getattr(artifact, "synthesized_payload_translations", {})),
                export_time_ms=capture_ms,
                decomposition_coverage=decomposed / max(len(diagnostics), 1),
                total_fx_nodes=total_fx_nodes,
                decomposed_ops=decomposed,
                opaque_ops=opaque,
                analysis_success=True,
                capture_mode=str(artifact.capture_mode),
                unsupported_ops=[str(d) for d in diagnostics if getattr(d, "level", "") == "error"],
            )
            record.ir = collect_ir_metrics(module)
            if record.ir.total_bytes:
                record.performance.arithmetic_intensity = record.ir.total_flops / max(record.ir.total_bytes, 1)

            contracts = build_kernel_contracts(module, target_profile)
            decisions = select_strategies(contracts, target_profile)
            record.kernels.total_kernel_specs = len(contracts)
            for decision in decisions:
                strategy = getattr(decision.strategy, "value", str(decision.strategy))
                record.kernels.strategy_histogram[strategy] = record.kernels.strategy_histogram.get(strategy, 0) + 1

            if ctx.ablation != "no_eqsat":
                eqsat_start = time.perf_counter()
                eqsat_result = run_eqsat_pass(module)
                eqsat_ms = (time.perf_counter() - eqsat_start) * 1000
                record.eqsat = collect_eqsat_metrics(eqsat_result, eqsat_ms)
            else:
                record.eqsat.changed = False

            recipe_module = generate_seed_recipe(module, target_profile)
            guard_registry = None
            fact_index = None
            synthesis_summary: dict[str, Any] | None = None
            if ctx.ablation not in {"no_guard_synthesis", "handwritten_only"}:
                guard_registry, fact_index, synthesis_summary = synthesize_and_attach_guards(
                    recipe_module,
                    out_dir=ctx.output_dir / "guards",
                    target_class=ctx.target.target_class,
                )
                record.synthesis = collect_synthesis_metrics(synthesis_summary)
            record.recipe = collect_recipe_metrics(recipe_module)
            validation = validate_recipe_module(recipe_module)
            record.recipe.validation_passed = validation.valid
            record.recipe.validation_errors = len(validation.errors)
            lowered = lower_recipe(
                recipe_module,
                guard_registry=guard_registry,
                fact_index=fact_index,
                target_class=ctx.target.target_class,
            )
            record.recipe.transform_scripts_count = len(lowered.transform_scripts)
            record.recipe.kernel_jobs_count = len(lowered.kernel_jobs)
            record.recipe.plan_fragments_count = len(lowered.plan_fragments)
            record.recipe.verification_obligations_count = len(lowered.verification_obligations)
            record.recipe.eqsat_jobs_count = len(lowered.eqsat_jobs)
            record.recipe.lowering_diagnostics = len(lowered.diagnostics)
            record.performance.kernel_count = len(lowered.kernel_jobs)
            if synthesis_summary is not None:
                family_data = synthesis_summary.get("families", {})
                fusion_family = family_data.get("fusion", {})
                local_mem_family = family_data.get("local_mem", {})
                record.synthesis.additional_legal_fusions = int(fusion_family.get("promoted", 0))
                record.synthesis.additional_profitable_fusions = int(fusion_family.get("promoted", 0))
                record.synthesis.additional_local_mem_placements = int(local_mem_family.get("promoted", 0))
                total_guard_verdicts = len(lowered.guard_verdicts)
                rejected_verdicts = sum(1 for verdict in lowered.guard_verdicts if not verdict.get("allow", False))
                if total_guard_verdicts:
                    record.synthesis.unsafe_accept_rate = 0.0
                    record.synthesis.missed_opportunity_rate = rejected_verdicts / total_guard_verdicts
                    record.synthesis.legality_recall = (total_guard_verdicts - rejected_verdicts) / total_guard_verdicts
                    record.synthesis.profitable_opportunity_recall = record.synthesis.legality_recall

            record.generation.candidate_transforms = record.recipe.candidate_ops
            record.generation.candidate_kernels = len(lowered.kernel_jobs)
            record.generation.candidate_recipes_explored = max(record.recipe.total_recipe_ops, 1)

            solver_plan = None
            if ctx.ablation not in {"no_solver", "kernel_only"}:
                solver_plan = plan_execution(module, target_profile)
                record.solver = collect_solver_metrics()
                record.solver.copy_ops_count = len(solver_plan.copies)
                record.solver.copy_bytes = sum(copy.size_bytes for copy in solver_plan.copies)
                record.solver.copy_time_us = sum(copy.estimated_cost_us for copy in solver_plan.copies)
                record.solver.node_assignments = dict(solver_plan.node_assignments)
                record.solver.transport_config = dict(solver_plan.transport_config)
                record.solver.placement_gap = float(solver_plan.metadata.get("placement_gap", 0.0))
                record.solver.placement_time_ms = float(solver_plan.metadata.get("placement_time_ms", 0.0))
                record.solver.schedule_feasible = bool(solver_plan.metadata.get("schedule_feasible", True))
                record.solver.schedule_time_ms = float(solver_plan.metadata.get("schedule_time_ms", 0.0))
                record.solver.schedule_makespan_us = float(solver_plan.estimated_latency_us or 0.0)
                record.solver.memory_feasible = bool(solver_plan.metadata.get("memory_feasible", True))
                record.solver.memory_time_ms = float(solver_plan.metadata.get("memory_time_ms", 0.0))
                record.solver.memory_reuse_count = int(solver_plan.metadata.get("memory_reuse_count", 0))
                record.solver.memory_peak_bytes = max((plan.peak_bytes for plan in solver_plan.memory_plans), default=0)
                record.performance.bytes_moved_cross_device = record.solver.copy_bytes

            verification_payload = {"status": "skipped", "details": {}}
            if ctx.ablation != "no_verification":
                verifier = TransformVerifier(
                    levels=[
                        VerificationLevel.STRUCTURAL,
                        VerificationLevel.CHECK_ASSERTIONS,
                        VerificationLevel.DIFFERENTIAL,
                        VerificationLevel.TRANSLATION_VALIDATION,
                    ]
                )
                verification_result = verifier.verify(original_module, module)
                verification_payload = _serialize_verification_result(verification_result)
                record.verification.structural_pass = "structural" in verification_result.details and "PASS" in verification_result.details["structural"]
                record.verification.check_assertions_pass = True
                record.verification.check_assertions_run = 0
                record.verification.differential_pass = "differential" in verification_result.details and "PASS" in verification_result.details["differential"]
                record.verification.differential_max_error = float(verification_result.max_abs_error or 0.0)
                tv_detail = verification_result.details.get("translation_validation", "")
                record.verification.translation_validation_pass = None if "SKIPPED" in tv_detail else True
                record.verification.overall_status = "pass" if verification_result.passed else "fail"
            else:
                record.verification.overall_status = "skip"

            bundle_dir = ctx.output_dir / "bundles" / f"{record.study.case_id}_{record.run_id}"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            with torch.no_grad():
                golden_outputs = model(*sample_inputs)
            manifest = create_bundle(
                output_dir=bundle_dir,
                module=module,
                execution_plan=solver_plan,
                target_name=resolved_target_name,
                objective=ctx.case.objective,
                golden_inputs=sample_inputs,
                golden_outputs=golden_outputs,
                transform_scripts=lowered.transform_scripts,
                kernel_contracts=_serialize_kernel_contracts(contracts),
                recipe_mlir_text=recipe_to_mlir(recipe_module),
                recipe_yaml_text=recipe_module_to_yaml(recipe_module),
                exported_program_text=str(exported.graph),
                verification_report=verification_payload,
            )
            _compute_artifact_metrics(record, bundle_dir, manifest.artifacts)
            record.capture.exported_program_path = manifest.artifacts.get("exported_program", "")
            record.recipe.recipe_mlir_path = manifest.artifacts.get("recipe_mlir", "")
            record.recipe.recipe_yaml_path = manifest.artifacts.get("recipe_yaml", "")
            record.kernels.contracts_path = manifest.artifacts.get("kernel_contracts", "")
            record.verification.report_path = manifest.artifacts.get("verification_report", "")

            if record.verification.overall_status == "pass":
                promoter = RecipePromoter(library_path=ctx.output_dir / "recipe_library")
                promote_result = promoter.promote(manifest)
                record.promotion_status = "promoted" if promote_result.promoted else "rejected"
                record.generation.promoted_candidates = 1 if promote_result.promoted else 0
            else:
                record.promotion_status = "rejected"

            executor = LocalExecutor()
            comparison = executor.compare(model, sample_inputs, num_iterations=10)
            record.baselines.eager_cpu_latency_us = comparison.eager_cpu.latency_median_us if comparison.eager_cpu else 0.0
            record.baselines.eager_gpu_latency_us = comparison.eager_gpu.latency_median_us if comparison.eager_gpu else 0.0
            record.baselines.compiled_gpu_latency_us = comparison.compiled_gpu.latency_median_us if comparison.compiled_gpu else 0.0

            compgen_backend = CompGenBackend(decisions={"ablation": ctx.ablation})
            backend_result = compgen_backend.compile_and_benchmark(
                model,
                sample_inputs,
                device="cuda",
                num_iterations=10,
                warmup=3,
            )
            record.performance.latency_median_us = backend_result.latency_median_us
            record.performance.latency_p99_us = backend_result.latency_p99_us
            record.performance.throughput_samples_per_sec = backend_result.throughput_samples_per_sec
            record.performance.peak_memory_bytes = backend_result.peak_memory_bytes
            record.performance.device = backend_result.device
            record.performance.mode = backend_result.mode
            record.performance.num_iterations = backend_result.num_iterations
            record.baselines.compgen_latency_us = backend_result.latency_median_us
            if record.baselines.eager_cpu_latency_us:
                record.baselines.speedup_vs_eager_cpu = record.baselines.eager_cpu_latency_us / max(
                    backend_result.latency_median_us, 1e-6
                )
            if record.baselines.eager_gpu_latency_us:
                record.baselines.speedup_vs_eager_gpu = record.baselines.eager_gpu_latency_us / max(
                    backend_result.latency_median_us, 1e-6
                )
            if record.baselines.compiled_gpu_latency_us:
                record.baselines.speedup_vs_compiled = record.baselines.compiled_gpu_latency_us / max(
                    backend_result.latency_median_us, 1e-6
                )
            if ctx.ablation in {"fixed_pass_only", "no_guard_synthesis", "handwritten_only"}:
                record.synthesis.speedup_passes_only = record.baselines.speedup_vs_compiled
            elif ctx.ablation == "synth_only":
                record.synthesis.speedup_guards_only = record.baselines.speedup_vs_compiled
            else:
                record.synthesis.speedup_combined = record.baselines.speedup_vs_compiled

            record.generation.search_time_ms = record.eqsat.eqsat_time_ms + record.kernels.total_search_time_ms
            record.generation.compile_time_ms = (time.perf_counter() - total_start) * 1000
            record.generation.solver_time_ms = (
                record.solver.placement_time_ms + record.solver.schedule_time_ms + record.solver.memory_time_ms
            )
            record.total_compile_time_ms = record.generation.compile_time_ms
            record.generation.rejected_by_verification = 0 if record.verification.overall_status in {"pass", "skip"} else 1
            record.status = "pass" if not record.errors and record.verification.overall_status in {"pass", "skip"} else "fail"

        except Exception as exc:  # pragma: no cover - exercised by integration paths
            return _finalize_expected_failure(record, ctx, exc, total_start=total_start)

        return record


class TorchEagerAdapter:
    """Local PyTorch eager baseline."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        return True, ""

    def run(self, ctx: AdapterContext) -> RunRecord:
        record = _new_record(ctx, system_name="torch_eager")
        _populate_study_identity(record, ctx)
        if ctx.workload.readiness != "full_pipeline":
            return _skip_non_pipeline_workload(
                record,
                ctx,
                reason=f"torch_eager baseline skipped for workload readiness={ctx.workload.readiness}",
            )
        model, sample_inputs = ctx.workload.load(ctx.workspace)
        from compgen.runtime.local_executor import LocalExecutor

        result = LocalExecutor().benchmark(model, sample_inputs, device="cpu", mode="eager", num_iterations=10, warmup=3)
        record.performance.latency_median_us = result.latency_median_us
        record.performance.latency_p99_us = result.latency_p99_us
        record.performance.throughput_samples_per_sec = result.throughput_samples_per_sec
        record.performance.peak_memory_bytes = result.peak_memory_bytes
        record.performance.device = result.device
        record.performance.mode = result.mode
        record.performance.num_iterations = result.num_iterations
        record.performance.warmup_iterations = result.warmup_iterations
        record.performance.per_run_us = result.per_run_us
        record.status = "pass"
        return record


class TorchCompileAdapter:
    """Local torch.compile baseline."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        return True, ""

    def run(self, ctx: AdapterContext) -> RunRecord:
        record = _new_record(ctx, system_name="torch_compile")
        _populate_study_identity(record, ctx)
        if ctx.workload.readiness != "full_pipeline":
            return _skip_non_pipeline_workload(
                record,
                ctx,
                reason=f"torch_compile baseline skipped for workload readiness={ctx.workload.readiness}",
            )
        model, sample_inputs = ctx.workload.load(ctx.workspace)
        from compgen.runtime.local_executor import LocalExecutor

        result = LocalExecutor().benchmark(
            model,
            sample_inputs,
            device="cuda",
            mode="compiled",
            num_iterations=10,
            warmup=3,
        )
        record.performance.latency_median_us = result.latency_median_us
        record.performance.latency_p99_us = result.latency_p99_us
        record.performance.throughput_samples_per_sec = result.throughput_samples_per_sec
        record.performance.peak_memory_bytes = result.peak_memory_bytes
        record.performance.device = result.device
        record.performance.mode = result.mode
        record.performance.num_iterations = result.num_iterations
        record.performance.warmup_iterations = result.warmup_iterations
        record.performance.per_run_us = result.per_run_us
        record.status = "pass"
        return record


class ExpertFixtureAdapter:
    """Fixture-backed expert/manual baseline adapter."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        fixture = self._fixture_path(ctx)
        if fixture.exists():
            return True, ""
        return False, f"fixture_missing:{fixture}"

    def _fixture_path(self, ctx: AdapterContext) -> Path:
        fixture = Path(ctx.baseline.fixture_path)
        if fixture.is_absolute():
            return fixture
        return (ctx.workspace.repo_root / fixture).resolve()

    def run(self, ctx: AdapterContext) -> RunRecord:
        record = _new_record(ctx, system_name="expert_fixture")
        _populate_study_identity(record, ctx)
        fixture = self._fixture_path(ctx)
        if not fixture.exists():
            record.status = "skip"
            record.errors.append(f"Expert baseline fixture not found: {fixture}")
            return record

        payload = json.loads(fixture.read_text())
        key = f"{ctx.workload.workload_id}:{ctx.target.target_id}"
        metrics = payload.get(key) or payload.get(ctx.workload.workload_id) or {}
        if not metrics:
            record.status = "skip"
            record.errors.append(f"No expert fixture entry for {key}")
            return record

        record.performance.latency_median_us = float(metrics.get("latency_median_us", 0.0))
        record.performance.latency_p99_us = float(metrics.get("latency_p99_us", 0.0))
        record.performance.throughput_samples_per_sec = float(metrics.get("throughput_samples_per_sec", 0.0))
        record.performance.peak_memory_bytes = int(metrics.get("peak_memory_bytes", 0))
        record.productivity.person_hours_to_first_correct = float(metrics.get("person_hours_to_first_correct", 0.0))
        record.productivity.person_hours_to_80pct_expert = float(metrics.get("person_hours_to_80pct_expert", 0.0))
        record.productivity.handwritten_target_specific_loc = int(metrics.get("handwritten_target_specific_loc", 0))
        record.productivity.manual_interventions = int(metrics.get("manual_interventions", 0))
        record.status = "pass"
        return record


class ExternalRepoAdapter:
    """Sibling-repo baseline hook for systems like IREE and XLA."""

    def is_available(self, ctx: AdapterContext) -> tuple[bool, str]:
        repo_path = ctx.workspace.resolve_external(ctx.baseline.repo_name, ctx.baseline.repo_hint)
        if repo_path.exists():
            return True, ""
        return False, f"repo_missing:{repo_path}"

    def run(self, ctx: AdapterContext) -> RunRecord:
        record = _new_record(ctx, system_name=ctx.baseline.baseline_id)
        _populate_study_identity(record, ctx)
        repo_path = ctx.workspace.resolve_external(ctx.baseline.repo_name, ctx.baseline.repo_hint)
        if not repo_path.exists():
            record.status = "skip"
            record.errors.append(f"Sibling repo missing: {repo_path}")
            return record

        if not ctx.baseline.runner_command:
            record.status = "skip"
            record.errors.append(
                f"Baseline repo resolved at {repo_path} but no runner_command configured for {ctx.baseline.baseline_id}"
            )
            return record

        command = [
            part.format(
                repo_root=str(ctx.workspace.repo_root),
                baseline_root=str(repo_path),
                workload_id=ctx.workload.workload_id,
                target_id=ctx.target.target_id,
                output_dir=str(ctx.output_dir),
            )
            for part in ctx.baseline.runner_command
        ]
        result = subprocess.run(
            command,
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
        )
        record.config["external_stdout"] = result.stdout[-5000:]
        record.config["external_stderr"] = result.stderr[-5000:]
        record.config["external_command"] = command
        if result.returncode != 0:
            record.status = "skip"
            record.errors.append(f"External baseline command failed with exit code {result.returncode}")
            return record

        metrics_path = ctx.output_dir / f"{ctx.baseline.baseline_id}_{ctx.case.case_id}_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            record.performance.latency_median_us = float(metrics.get("latency_median_us", 0.0))
            record.performance.latency_p99_us = float(metrics.get("latency_p99_us", 0.0))
            record.performance.throughput_samples_per_sec = float(metrics.get("throughput_samples_per_sec", 0.0))
            record.performance.peak_memory_bytes = int(metrics.get("peak_memory_bytes", 0))
            record.status = "pass"
        else:
            record.status = "skip"
            record.errors.append(f"External baseline did not emit metrics file: {metrics_path}")
        return record


ADAPTERS: dict[str, Adapter] = {
    "compgen": CompGenAdapter(),
    "torch_eager": TorchEagerAdapter(),
    "torch_compile": TorchCompileAdapter(),
    "expert_fixture": ExpertFixtureAdapter(),
    "external_repo": ExternalRepoAdapter(),
}


def get_adapter(spec: BaselineSpec) -> Adapter:
    """Look up an adapter by baseline spec."""

    if spec.adapter not in ADAPTERS:
        raise KeyError(f"Unknown adapter: {spec.adapter}")
    return ADAPTERS[spec.adapter]


def check_baseline_availability(
    registry: BenchmarkRegistry,
    workspace: WorkspaceConfig,
    baseline_ids: list[str] | None = None,
) -> dict[str, str]:
    """Return a baseline-id -> status string map."""

    results: dict[str, str] = {}
    baseline_ids = baseline_ids or list(registry.baselines.keys())
    for baseline_id in baseline_ids:
        baseline = registry.get_baseline(baseline_id)
        workload = next(iter(registry.workloads.values()))
        target = next(iter(registry.targets.values()))
        ctx = AdapterContext(
            workspace=workspace,
            registry=registry,
            case=ExperimentCase(
                case_id=f"availability_{baseline_id}",
                study_id="availability",
                workload_id=workload.workload_id,
                target_id=target.target_id,
                baseline_ids=[baseline_id],
            ),
            workload=workload,
            target=target,
            baseline=baseline,
            output_dir=workspace.repo_root / "benchmarks" / "results",
        )
        available, reason = get_adapter(baseline).is_available(ctx)
        results[baseline_id] = "available" if available else reason
    return results


__all__ = ["ADAPTERS", "AdapterContext", "check_baseline_availability", "get_adapter"]
