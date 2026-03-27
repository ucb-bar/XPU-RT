"""Built-in benchmark registry for the CompGen MLSys study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.spec import (
    BaselineSpec,
    DefectSpec,
    ExperimentCase,
    StudySpec,
    TargetSpec,
    WorkloadBundle,
    WorkloadSpec,
)
from benchmarks.workloads import get_loader


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES = [
    "compgen",
    "torch_eager",
    "torch_compile",
    "expert_fixture",
    "iree",
    "xla_pjrt",
]


@dataclass
class BenchmarkRegistry:
    """Mutable registry of workloads, targets, baselines, studies, and cases."""

    workloads: dict[str, WorkloadSpec] = field(default_factory=dict)
    bundles: dict[str, WorkloadBundle] = field(default_factory=dict)
    targets: dict[str, TargetSpec] = field(default_factory=dict)
    baselines: dict[str, BaselineSpec] = field(default_factory=dict)
    studies: dict[str, StudySpec] = field(default_factory=dict)
    cases: dict[str, ExperimentCase] = field(default_factory=dict)
    defects: dict[str, DefectSpec] = field(default_factory=dict)

    def register_workload(self, spec: WorkloadSpec) -> None:
        self.workloads[spec.workload_id] = spec

    def register_bundle(self, spec: WorkloadBundle) -> None:
        self.bundles[spec.bundle_id] = spec

    def register_target(self, spec: TargetSpec) -> None:
        self.targets[spec.target_id] = spec

    def register_baseline(self, spec: BaselineSpec) -> None:
        self.baselines[spec.baseline_id] = spec

    def register_study(self, spec: StudySpec) -> None:
        self.studies[spec.study_id] = spec

    def register_case(self, spec: ExperimentCase) -> None:
        self.cases[spec.case_id] = spec

    def register_defect(self, spec: DefectSpec) -> None:
        self.defects[spec.defect_id] = spec

    def get_workload(self, workload_id: str) -> WorkloadSpec:
        return self.workloads[workload_id]

    def get_target(self, target_id: str) -> TargetSpec:
        return self.targets[target_id]

    def get_baseline(self, baseline_id: str) -> BaselineSpec:
        return self.baselines[baseline_id]

    def get_case(self, case_id: str) -> ExperimentCase:
        return self.cases[case_id]

    def get_study(self, study_id: str) -> StudySpec:
        return self.studies[study_id]


def _register_workloads(registry: BenchmarkRegistry) -> None:
    for workload_id, tier, description, tags in [
        ("simple_mlp", "tier_b", "SimpleMLP block from examples", ["mlp", "paper_subset"]),
        ("transformer_block", "tier_b", "TransformerBlock from examples", ["transformer", "paper_subset"]),
        ("quantized_mlp", "tier_b", "Quantized MLP example", ["quantized", "paper_subset"]),
        ("simple_mlp_batch16", "tier_c", "SimpleMLP shape variant for bundle specialization", ["mlp", "bundle"]),
        ("simple_mlp_batch32", "tier_c", "Held-out SimpleMLP bundle variant", ["mlp", "bundle", "heldout"]),
        ("transformer_block_seq8", "tier_c", "Transformer block discovery variant", ["transformer", "bundle"]),
        ("transformer_block_seq32", "tier_c", "Transformer block held-out variant", ["transformer", "bundle", "heldout"]),
        ("quantized_mlp_batch16", "tier_c", "Quantized MLP held-out bundle variant", ["quantized", "bundle", "heldout"]),
        ("matmul_bias_gelu", "tier_a", "Matmul + bias + GELU microbenchmark", ["microbenchmark", "matmul"]),
        ("matmul_add_relu", "tier_a", "Matmul + residual add + ReLU microbenchmark", ["microbenchmark", "matmul"]),
        ("layernorm_chain", "tier_a", "LayerNorm chain microbenchmark", ["microbenchmark", "reduction"]),
        ("softmax_elemwise", "tier_a", "Softmax surrounded by elementwise ops", ["microbenchmark", "reduction"]),
        ("transpose_pingpong", "tier_a", "Transpose/layout ping-pong graph", ["microbenchmark", "layout"]),
        ("copy_boundary_heavy", "tier_a", "Copy-boundary heavy synthetic graph", ["microbenchmark", "hybrid"]),
        ("scan_small_kernels", "tier_a", "Small-kernel scan-like graph", ["microbenchmark", "loop"]),
        ("reduction_block", "tier_a", "Reduction-heavy block", ["microbenchmark", "reduction"]),
    ]:
        registry.register_workload(
            WorkloadSpec(
                workload_id=workload_id,
                tier=tier,
                description=description,
                loader=get_loader(workload_id),
                tags=tags,
            )
        )

    registry.register_bundle(
        WorkloadBundle(
            bundle_id="BundleT",
            description="Transformer-like workloads for discovery/held-out transfer",
            discovery_workloads=["transformer_block_seq8", "transformer_block"],
            heldout_workloads=["transformer_block_seq32"],
            tags=["transformer", "paper_subset"],
        )
    )
    registry.register_bundle(
        WorkloadBundle(
            bundle_id="BundleM",
            description="MLP and reduction workloads for discovery/held-out transfer",
            discovery_workloads=["simple_mlp", "simple_mlp_batch16", "quantized_mlp"],
            heldout_workloads=["simple_mlp_batch32", "quantized_mlp_batch16"],
            tags=["mlp", "paper_subset"],
        )
    )


def _register_targets(registry: BenchmarkRegistry) -> None:
    for target_id, rel_path, kind, description, target_class, tags in [
        (
            "cuda_a100",
            "examples/target_profiles/cuda_a100.yaml",
            "target_profile",
            "TRITON_FRIENDLY GPU target",
            "TRITON_FRIENDLY",
            ["gpu", "paper_subset"],
        ),
        (
            "riscv_soc",
            "examples/target_profiles/riscv_soc.yaml",
            "target_profile",
            "ACCEL_NATIVE / UKERNEL_RUNTIME bring-up target",
            "UKERNEL_RUNTIME",
            ["accelerator", "paper_subset"],
        ),
        (
            "multi_device",
            "examples/target_profiles/multi_device.yaml",
            "target_profile",
            "HYBRID CPU+GPU topology target",
            "HYBRID",
            ["hybrid", "paper_subset"],
        ),
        (
            "gpu_simt_demo",
            "examples/hardware_specs/gpu_simt_demo.yaml",
            "hardware_spec",
            "Target-generation hardware-spec demo",
            "TRITON_FRIENDLY",
            ["hardware_spec"],
        ),
    ]:
        registry.register_target(
            TargetSpec(
                target_id=target_id,
                path=REPO_ROOT / rel_path,
                kind=kind,
                description=description,
                target_class=target_class,
                tags=tags,
            )
        )


def _register_baselines(registry: BenchmarkRegistry) -> None:
    for baseline in [
        BaselineSpec(
            baseline_id="compgen",
            adapter="compgen",
            description="CompGen full pipeline run",
            tags=["primary"],
        ),
        BaselineSpec(
            baseline_id="torch_eager",
            adapter="torch_eager",
            description="PyTorch eager execution baseline",
            tags=["local"],
        ),
        BaselineSpec(
            baseline_id="torch_compile",
            adapter="torch_compile",
            description="PyTorch torch.compile baseline",
            tags=["local"],
        ),
        BaselineSpec(
            baseline_id="expert_fixture",
            adapter="expert_fixture",
            description="Expert/manual baseline fixture manifest",
            fixture_path="benchmarks/results/expert_fixture.json",
            tags=["fixture", "manual"],
        ),
        BaselineSpec(
            baseline_id="iree",
            adapter="external_repo",
            description="IREE sibling-repo baseline hook",
            repo_name="iree",
            repo_hint="iree",
            tags=["external", "iree"],
        ),
        BaselineSpec(
            baseline_id="xla_pjrt",
            adapter="external_repo",
            description="OpenXLA/PJRT sibling-repo baseline hook",
            repo_name="xla",
            repo_hint="xla",
            tags=["external", "xla"],
        ),
    ]:
        registry.register_baseline(baseline)


def _register_cases_and_studies(registry: BenchmarkRegistry) -> None:
    case_ids: dict[str, list[str]] = {}

    def add_case(spec: ExperimentCase) -> None:
        registry.register_case(spec)
        case_ids.setdefault(spec.study_id, []).append(spec.case_id)

    for workload_id in ["simple_mlp", "transformer_block", "quantized_mlp"]:
        for target_id in ["cuda_a100", "riscv_soc", "multi_device"]:
            add_case(
                ExperimentCase(
                    case_id=f"pipeline_{workload_id}_{target_id}",
                    study_id="pipeline_sanity",
                    workload_id=workload_id,
                    target_id=target_id,
                    baseline_ids=DEFAULT_BASELINES,
                    tags=["sanity", "paper_subset"],
                )
            )

    for workload_id in [
        "matmul_bias_gelu",
        "matmul_add_relu",
        "layernorm_chain",
        "softmax_elemwise",
        "transpose_pingpong",
        "copy_boundary_heavy",
        "scan_small_kernels",
        "reduction_block",
    ]:
        for target_id in ["cuda_a100", "riscv_soc"]:
            add_case(
                ExperimentCase(
                    case_id=f"pass_{workload_id}_{target_id}",
                    study_id="pass_discovery",
                    workload_id=workload_id,
                    target_id=target_id,
                    baseline_ids=DEFAULT_BASELINES,
                    ablations=["fixed_pass_only", "no_eqsat", "kernel_only", "triton_only"],
                    tags=["microbenchmark"],
                )
            )

    for workload_id, target_id in [
        ("simple_mlp", "cuda_a100"),
        ("transformer_block", "cuda_a100"),
        ("copy_boundary_heavy", "riscv_soc"),
    ]:
        add_case(
            ExperimentCase(
                case_id=f"single_{workload_id}_{target_id}",
                study_id="single_specialization",
                workload_id=workload_id,
                target_id=target_id,
                baseline_ids=DEFAULT_BASELINES,
                ablations=["seed_only", "fixed_pass_only", "no_eqsat", "no_solver"],
                tags=["specialization", "paper_subset"],
            )
        )

    for workload_id, bundle_id in [
        ("transformer_block", "BundleT"),
        ("simple_mlp", "BundleM"),
    ]:
        add_case(
            ExperimentCase(
                case_id=f"bundle_{workload_id}",
                study_id="bundle_specialization",
                workload_id=workload_id,
                target_id="cuda_a100",
                baseline_ids=DEFAULT_BASELINES,
                ablations=["single_workload_specialization", "bundle_specialization"],
                tags=["bundle", "paper_subset"],
                metadata={"bundle_id": bundle_id},
            )
        )

    for workload_id in ["transformer_block", "simple_mlp", "copy_boundary_heavy"]:
        add_case(
            ExperimentCase(
                case_id=f"hybrid_{workload_id}",
                study_id="hybrid_planning",
                workload_id=workload_id,
                target_id="multi_device",
                baseline_ids=DEFAULT_BASELINES,
                ablations=["no_solver"],
                tags=["hybrid", "paper_subset"],
            )
        )

    for workload_id, target_id in [("simple_mlp", "cuda_a100"), ("transformer_block", "multi_device")]:
        add_case(
            ExperimentCase(
                case_id=f"red_team_{workload_id}_{target_id}",
                study_id="verification_red_team",
                workload_id=workload_id,
                target_id=target_id,
                baseline_ids=["compgen"],
                tags=["verification", "paper_subset"],
            )
        )

    for study_id, description, tier, tags in [
        ("pipeline_sanity", "Pipeline sanity study across frozen workloads and targets", "study", ["sanity"]),
        ("pass_discovery", "Pass-discovery microbenchmark study", "study", ["microbenchmark"]),
        ("single_specialization", "Single-workload specialization study", "study", ["specialization"]),
        ("bundle_specialization", "Workload-bundle transfer study", "study", ["bundle"]),
        ("hybrid_planning", "Hybrid placement/scheduling study", "study", ["hybrid"]),
        ("verification_red_team", "Verification red-team study", "study", ["verification"]),
    ]:
        registry.register_study(
            StudySpec(
                study_id=study_id,
                description=description,
                case_ids=case_ids.get(study_id, []),
                tier=tier,
                tags=tags,
            )
        )

    paper_subset_ids = (
        case_ids["pipeline_sanity"]
        + case_ids["single_specialization"]
        + case_ids["bundle_specialization"]
        + case_ids["hybrid_planning"]
        + case_ids["verification_red_team"]
    )
    registry.register_study(
        StudySpec(
            study_id="paper_subset",
            description="Union study used for the first MLSys paper subset",
            case_ids=paper_subset_ids,
            tier="study",
            tags=["paper_subset"],
        )
    )


def _register_defects(registry: BenchmarkRegistry) -> None:
    for defect in [
        DefectSpec("wrong_tile_sizes", "recipe_validation", "Illegal or negative tile sizes", "recipe_validation"),
        DefectSpec("illegal_fusion", "recipe_validation", "Fusion across illegal boundaries", "recipe_validation"),
        DefectSpec("wrong_layout_assumption", "layout_invariant", "Layout invariant mismatch", "verification"),
        DefectSpec("wrong_device_placement", "solver", "Placement violates device assumptions", "solver"),
        DefectSpec("missing_copy_boundary", "planning", "Cross-device dependency without copy op", "planning"),
        DefectSpec("numerically_wrong_kernel", "differential", "Kernel produces numerically wrong results", "differential"),
        DefectSpec("malformed_transform", "structural", "Transform text cannot be verified structurally", "structural"),
        DefectSpec("infeasible_memory_budget", "memory_bound", "Memory budget is impossible on the target", "memory"),
        DefectSpec("overoptimistic_profile_budget", "profile_budget", "Profile budget below measured latency", "profile"),
    ]:
        registry.register_defect(defect)


def build_default_registry() -> BenchmarkRegistry:
    """Build the benchmark registry used by the first study harness."""

    registry = BenchmarkRegistry()
    _register_workloads(registry)
    _register_targets(registry)
    _register_baselines(registry)
    _register_cases_and_studies(registry)
    _register_defects(registry)
    return registry


__all__ = ["BenchmarkRegistry", "DEFAULT_BASELINES", "build_default_registry"]
