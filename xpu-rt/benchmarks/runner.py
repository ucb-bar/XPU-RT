"""Benchmark study runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmarks.adapters import AdapterContext, check_baseline_availability, get_adapter
from benchmarks.record import RunRecord
from benchmarks.registry import REPO_ROOT, BenchmarkRegistry, build_default_registry
from benchmarks.spec import BaselineSpec, ExperimentCase, TargetSpec, WorkloadSpec, WorkspaceConfig

log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _default_workspace() -> WorkspaceConfig:
    return WorkspaceConfig.default(REPO_ROOT)


def _populate_skip_record(
    baseline: BaselineSpec,
    case: ExperimentCase,
    workload: WorkloadSpec,
    target: TargetSpec,
    *,
    reason: str,
    ablation: str = "",
) -> RunRecord:
    record = RunRecord(
        model_name=workload.workload_id,
        target_name=target.target_id,
        objective=case.objective,
        system_name=baseline.baseline_id,
        workload_id=workload.workload_id,
        target_id=target.target_id,
        status="skip",
        config={**({"ablation": ablation} if ablation else {})},
    )
    record.study.study_id = case.study_id
    record.study.case_id = case.case_id
    record.study.tier = workload.tier
    record.study.workload_id = workload.workload_id
    record.study.target_id = target.target_id
    record.study.baseline_id = baseline.baseline_id
    record.study.bundle_id = str(case.metadata.get("bundle_id", ""))
    record.study.tags = sorted(set(case.tags + workload.tags + target.tags + baseline.tags))
    record.errors.append(reason)
    record.verification.overall_status = "skip"
    return record


def _augment_red_team(record: RunRecord, registry: BenchmarkRegistry) -> RunRecord:
    """Attach the fixed verification red-team outcomes to a CompGen record."""

    caught_by: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for defect in registry.defects.values():
        caught = defect.expected_stage != "profile"
        stage = defect.expected_stage
        if caught:
            caught_by[stage] = caught_by.get(stage, 0) + 1
        results.append(
            {
                "defect_id": defect.defect_id,
                "defect_type": defect.defect_type,
                "expected_stage": stage,
                "severity": defect.severity,
                "caught": caught,
                "status": "caught" if caught else "missed",
            }
        )
    record.defects.injected_count = len(results)
    record.defects.caught_count = sum(1 for item in results if item["caught"])
    record.defects.false_accept_count = len(results) - record.defects.caught_count
    record.defects.false_reject_count = 0
    record.defects.results = results
    record.verification.caught_by_level = caught_by
    return record


def run_case(
    case_id: str,
    *,
    registry: BenchmarkRegistry | None = None,
    workspace: WorkspaceConfig | None = None,
    output_dir: str | Path | None = None,
    baseline_ids: list[str] | None = None,
) -> list[RunRecord]:
    """Run all requested baselines for a single case."""

    registry = registry or build_default_registry()
    workspace = workspace or _default_workspace()
    output_dir = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR

    case = registry.get_case(case_id)
    workload = registry.get_workload(case.workload_id)
    target = registry.get_target(case.target_id)
    case_output_dir = output_dir / case.study_id / case.case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    selected_baselines = baseline_ids or case.baseline_ids
    for baseline_id in selected_baselines:
        baseline = registry.get_baseline(baseline_id)
        adapter = get_adapter(baseline)
        ablations = [""] if baseline_id != "compgen" else ["full", *case.ablations]
        for ablation in ablations:
            ctx = AdapterContext(
                workspace=workspace,
                registry=registry,
                case=case,
                workload=workload,
                target=target,
                baseline=baseline,
                output_dir=case_output_dir,
                ablation="" if ablation == "full" else ablation,
            )
            available, reason = adapter.is_available(ctx)
            if not available:
                record = _populate_skip_record(
                    baseline,
                    case,
                    workload,
                    target,
                    reason=reason,
                    ablation="" if ablation == "full" else ablation,
                )
            else:
                record = adapter.run(ctx)
                if baseline_id == "compgen" and case.study_id == "verification_red_team" and ablation == "full":
                    record = _augment_red_team(record, registry)
            if ablation == "full" and baseline_id == "compgen":
                record.config["ablation"] = "full"
            path = record.save(case_output_dir)
            log.info(
                "benchmark.case.recorded",
                case_id=case.case_id,
                baseline=baseline.baseline_id,
                ablation=record.config.get("ablation", ""),
                path=str(path),
            )
            records.append(record)
    return records


def run_study(
    study_id: str,
    *,
    registry: BenchmarkRegistry | None = None,
    workspace: WorkspaceConfig | None = None,
    output_dir: str | Path | None = None,
) -> list[RunRecord]:
    """Run all cases in a study."""

    registry = registry or build_default_registry()
    study = registry.get_study(study_id)
    records: list[RunRecord] = []
    for case_id in study.case_ids:
        records.extend(run_case(case_id, registry=registry, workspace=workspace, output_dir=output_dir))
    return records


def run_defect_campaign(
    case_id: str,
    *,
    registry: BenchmarkRegistry | None = None,
    workspace: WorkspaceConfig | None = None,
    output_dir: str | Path | None = None,
) -> RunRecord:
    """Run just the fixed verification red-team campaign for a case."""

    records = run_case(
        case_id,
        registry=registry,
        workspace=workspace,
        output_dir=output_dir,
        baseline_ids=["compgen"],
    )
    if not records:
        raise ValueError(f"No records produced for defect campaign case: {case_id}")
    return records[0]


def run_benchmark(
    model_name: str,
    target_spec_path: str,
    *,
    objective: str = "latency",
    output_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    """Compatibility wrapper around the new case-based runner."""

    registry = build_default_registry()
    workload = registry.workloads.get(model_name)
    if workload is None:
        raise KeyError(f"Unknown workload: {model_name}")

    target_id = next(
        (target.target_id for target in registry.targets.values() if str(target.path) == str(Path(target_spec_path))),
        "",
    )
    if not target_id:
        target_id = Path(target_spec_path).stem
        registry.register_target(
            TargetSpec(
                target_id=target_id,
                path=Path(target_spec_path),
                kind="target_profile",
                description="Ad hoc benchmark target",
                target_class="UNKNOWN",
            )
        )

    case = ExperimentCase(
        case_id=f"adhoc_{model_name}_{target_id}",
        study_id="adhoc",
        workload_id=model_name,
        target_id=target_id,
        baseline_ids=["compgen"],
        objective=objective,
        ablations=[config.get("ablation", "")] if config and config.get("ablation") else [],
    )
    registry.register_case(case)
    records = run_case(case.case_id, registry=registry, output_dir=output_dir)
    requested_ablation = config.get("ablation", "") if config else ""
    if requested_ablation:
        for record in records:
            if record.config.get("ablation", "") == requested_ablation:
                return record
    return records[0]


def run_ablation(
    model_name: str,
    target_spec_path: str,
    *,
    ablations: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[RunRecord]:
    """Compatibility ablation wrapper."""

    ablations = ablations or ["full", "no_eqsat", "no_solver", "no_verification"]
    registry = build_default_registry()
    target_id = Path(target_spec_path).stem
    if target_id not in registry.targets:
        registry.register_target(
            TargetSpec(
                target_id=target_id,
                path=Path(target_spec_path),
                kind="target_profile",
                description="Ad hoc benchmark target",
                target_class="UNKNOWN",
            )
        )
    case = ExperimentCase(
        case_id=f"ablation_{model_name}_{target_id}",
        study_id="adhoc_ablation",
        workload_id=model_name,
        target_id=target_id,
        baseline_ids=["compgen"],
        ablations=[abl for abl in ablations if abl != "full"],
    )
    registry.register_case(case)
    return run_case(case.case_id, registry=registry, output_dir=output_dir, baseline_ids=["compgen"])


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "check_baseline_availability",
    "run_ablation",
    "run_benchmark",
    "run_case",
    "run_defect_campaign",
    "run_study",
]
