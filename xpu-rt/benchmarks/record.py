"""Benchmark run record and schema helpers.

The study harness stores one JSON file per executed system/workload/target run.
Records are intentionally verbose so plots, tables, and paper-specific summaries
can be generated without re-running experiments.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CaptureMetrics:
    """Frontend capture metrics."""

    export_success: bool = False
    graph_break_count: int = 0
    op_coverage: float = 0.0
    unsupported_ops: list[str] = field(default_factory=list)
    export_time_ms: float = 0.0
    decomposition_coverage: float = 0.0
    total_fx_nodes: int = 0
    decomposed_ops: int = 0
    opaque_ops: int = 0
    exported_program_path: str = ""


@dataclass
class IRMetrics:
    """Payload IR quality metrics."""

    total_ops: int = 0
    region_count: int = 0
    total_flops: int = 0
    total_bytes: int = 0
    compute_ops: int = 0
    memory_ops: int = 0
    op_type_histogram: dict[str, int] = field(default_factory=dict)
    opaque_fraction: float = 0.0


@dataclass
class RecipeMetrics:
    """Recipe IR generation and lowering metrics."""

    total_recipe_ops: int = 0
    scope_ops: int = 0
    fact_ops: int = 0
    candidate_ops: int = 0
    choice_ops: int = 0
    verify_ops: int = 0
    provenance_ops: int = 0
    seed_generation_time_ms: float = 0.0
    validation_passed: bool = False
    validation_errors: int = 0
    transform_scripts_count: int = 0
    kernel_jobs_count: int = 0
    plan_fragments_count: int = 0
    verification_obligations_count: int = 0
    eqsat_jobs_count: int = 0
    lowering_diagnostics: int = 0
    recipe_mlir_path: str = ""
    recipe_yaml_path: str = ""


@dataclass
class EqSatMetrics:
    """Equality saturation metrics."""

    ops_before: int = 0
    ops_after: int = 0
    ops_reduction_pct: float = 0.0
    eclasses_initial: int = 0
    eclasses_after_rewrite: int = 0
    enodes_after_rewrite: int = 0
    rules_applied: dict[str, int] = field(default_factory=dict)
    total_rule_applications: int = 0
    changed: bool = False
    eqsat_time_ms: float = 0.0


@dataclass
class SolverMetrics:
    """Solver-backed planning metrics."""

    placement_feasible: bool = False
    placement_objective: float = 0.0
    placement_gap: float = 0.0
    placement_time_ms: float = 0.0
    placement_transfer_cost: float = 0.0
    schedule_feasible: bool = False
    schedule_makespan_us: float = 0.0
    schedule_time_ms: float = 0.0
    schedule_deadline_met: bool = True
    memory_feasible: bool = False
    memory_peak_bytes: int = 0
    memory_reuse_count: int = 0
    memory_time_ms: float = 0.0
    copy_ops_count: int = 0
    copy_bytes: int = 0
    copy_time_us: float = 0.0
    node_assignments: dict[str, str] = field(default_factory=dict)
    transport_config: dict[str, str] = field(default_factory=dict)


@dataclass
class KernelMetrics:
    """Kernel generation and validation metrics."""

    total_kernel_specs: int = 0
    strategy_histogram: dict[str, int] = field(default_factory=dict)
    kernel_results: list[dict[str, Any]] = field(default_factory=list)
    kernels_searched: int = 0
    kernels_correct: int = 0
    kernels_pass_rate: float = 0.0
    best_speedup: float = 0.0
    total_search_tokens: int = 0
    total_search_time_ms: float = 0.0
    contracts_path: str = ""


@dataclass
class VerificationMetrics:
    """Verification ladder metrics."""

    structural_pass: bool = False
    structural_violations: int = 0
    check_assertions_pass: bool = False
    check_assertions_run: int = 0
    differential_pass: bool = False
    differential_max_error: float = 0.0
    translation_validation_pass: bool | None = None
    translation_validation_time_ms: float = 0.0
    overall_status: str = "pending"
    caught_by_level: dict[str, int] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)
    report_path: str = ""


@dataclass
class PerformanceMetrics:
    """Runtime performance measurements."""

    latency_median_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_mean_us: float = 0.0
    latency_std_us: float = 0.0
    per_run_us: list[float] = field(default_factory=list)
    throughput_samples_per_sec: float = 0.0
    peak_memory_bytes: int = 0
    bytes_moved_cross_device: int = 0
    energy_joules: float = 0.0
    arithmetic_intensity: float = 0.0
    kernel_count: int = 0
    device: str = ""
    mode: str = ""
    num_iterations: int = 0
    warmup_iterations: int = 0


@dataclass
class BaselineMetrics:
    """Legacy baseline summary for existing tests and compact exports."""

    eager_cpu_latency_us: float = 0.0
    eager_gpu_latency_us: float = 0.0
    compiled_gpu_latency_us: float = 0.0
    compgen_latency_us: float = 0.0
    speedup_vs_eager_cpu: float = 0.0
    speedup_vs_eager_gpu: float = 0.0
    speedup_vs_compiled: float = 0.0


@dataclass
class LLMMetrics:
    """LLM interaction metrics."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    calls_per_stage: dict[str, int] = field(default_factory=dict)
    model_id: str = ""


@dataclass
class AgenticMetrics:
    """Agentic compilation loop metrics."""

    iterations_run: int = 0
    iterations_improved: int = 0
    initial_cost_us: float = 0.0
    final_cost_us: float = 0.0
    total_improvement_pct: float = 0.0
    convergence_iteration: int = 0
    iteration_costs: list[float] = field(default_factory=list)
    iteration_improvements: list[float] = field(default_factory=list)
    iteration_actions: list[str] = field(default_factory=list)


@dataclass
class ProfilingMetrics:
    """Hardware profiling metrics."""

    compute_utilization: float = 0.0
    memory_utilization: float = 0.0
    dma_compute_overlap: float = 0.0
    idle_fraction: float = 0.0
    bottleneck_regions: list[dict[str, Any]] = field(default_factory=list)
    roofline_points: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactMetrics:
    """Artifact completeness and bundle contract metrics."""

    bundle_path: str = ""
    manifest_path: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    artifacts_present: dict[str, bool] = field(default_factory=dict)
    missing_artifacts: list[str] = field(default_factory=list)
    completeness_score: float = 0.0
    runnable_bundle: bool = False


@dataclass
class ProductivityMetrics:
    """Bring-up effort and manual intervention metrics."""

    person_hours_to_first_correct: float = 0.0
    person_hours_to_80pct_expert: float = 0.0
    manual_interventions: int = 0
    handwritten_target_specific_loc: int = 0
    handwritten_code_changes: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class GenerationMetrics:
    """Search and generation metrics beyond the raw LLM accounting."""

    candidate_recipes_explored: int = 0
    candidate_transforms: int = 0
    candidate_kernels: int = 0
    rejected_by_verification: int = 0
    promoted_candidates: int = 0
    search_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    solver_time_ms: float = 0.0


@dataclass
class DefectMetrics:
    """Verification red-team outcomes."""

    injected_count: int = 0
    caught_count: int = 0
    false_accept_count: int = 0
    false_reject_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StudyMetadata:
    """Study and case identity for the paper harness."""

    study_id: str = ""
    case_id: str = ""
    tier: str = ""
    workload_id: str = ""
    target_id: str = ""
    baseline_id: str = ""
    bundle_id: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class RunRecord:
    """Complete benchmark run record."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_name: str = ""
    target_name: str = ""
    objective: str = "latency"
    system_name: str = "compgen"
    workload_id: str = ""
    target_id: str = ""
    status: str = "pending"
    config: dict[str, Any] = field(default_factory=dict)
    study: StudyMetadata = field(default_factory=StudyMetadata)
    capture: CaptureMetrics = field(default_factory=CaptureMetrics)
    ir: IRMetrics = field(default_factory=IRMetrics)
    recipe: RecipeMetrics = field(default_factory=RecipeMetrics)
    eqsat: EqSatMetrics = field(default_factory=EqSatMetrics)
    solver: SolverMetrics = field(default_factory=SolverMetrics)
    kernels: KernelMetrics = field(default_factory=KernelMetrics)
    verification: VerificationMetrics = field(default_factory=VerificationMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    baselines: BaselineMetrics = field(default_factory=BaselineMetrics)
    llm: LLMMetrics = field(default_factory=LLMMetrics)
    agentic: AgenticMetrics = field(default_factory=AgenticMetrics)
    profiling: ProfilingMetrics = field(default_factory=ProfilingMetrics)
    artifacts: ArtifactMetrics = field(default_factory=ArtifactMetrics)
    productivity: ProductivityMetrics = field(default_factory=ProductivityMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    defects: DefectMetrics = field(default_factory=DefectMetrics)
    total_compile_time_ms: float = 0.0
    promotion_status: str = "pending"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""

        return asdict(self)

    def save(self, output_dir: str | Path) -> Path:
        """Save the record as JSON."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_id}_{self.system_name}_{self.model_name}_{self.target_name}.json"
        path = output_dir / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        """Load a record from disk."""

        data = json.loads(Path(path).read_text())
        record = cls()
        nested_types: dict[str, type[Any]] = {
            "study": StudyMetadata,
            "capture": CaptureMetrics,
            "ir": IRMetrics,
            "recipe": RecipeMetrics,
            "eqsat": EqSatMetrics,
            "solver": SolverMetrics,
            "kernels": KernelMetrics,
            "verification": VerificationMetrics,
            "performance": PerformanceMetrics,
            "baselines": BaselineMetrics,
            "llm": LLMMetrics,
            "agentic": AgenticMetrics,
            "profiling": ProfilingMetrics,
            "artifacts": ArtifactMetrics,
            "productivity": ProductivityMetrics,
            "generation": GenerationMetrics,
            "defects": DefectMetrics,
        }
        for key, val in data.items():
            if key in nested_types and isinstance(val, dict):
                setattr(record, key, nested_types[key](**val))
            else:
                setattr(record, key, val)
        return record


__all__ = [
    "AgenticMetrics",
    "ArtifactMetrics",
    "BaselineMetrics",
    "CaptureMetrics",
    "DefectMetrics",
    "EqSatMetrics",
    "GenerationMetrics",
    "IRMetrics",
    "KernelMetrics",
    "LLMMetrics",
    "PerformanceMetrics",
    "ProductivityMetrics",
    "ProfilingMetrics",
    "RecipeMetrics",
    "RunRecord",
    "SolverMetrics",
    "StudyMetadata",
    "VerificationMetrics",
]
