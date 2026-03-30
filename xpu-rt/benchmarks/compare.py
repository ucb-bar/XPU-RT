"""Cross-run aggregation, summaries, and exports."""

from __future__ import annotations

from pathlib import Path

from benchmarks.record import RunRecord
from compgen.benchmarks.common.results import write_normalized_suite_results


def load_all_results(results_dir: str | Path) -> list[RunRecord]:
    """Load all JSON benchmark records recursively from a directory."""

    results_dir = Path(results_dir)
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            import json

            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or "run_id" not in payload:
                continue
            records.append(RunRecord.load(path))
        except Exception:
            continue
    return records


def summary_table(records: list[RunRecord]) -> str:
    """Generate a markdown summary table."""

    if not records:
        return "No records found."

    lines = [
        "| Study | Case | System | Model | Target | Status | Compile (ms) | Latency (μs) | Speedup | Artifacts |",
        "|-------|------|--------|-------|--------|--------|-------------|-------------|---------|-----------|",
    ]
    for r in records:
        latency = f"{r.performance.latency_median_us:.1f}" if r.performance.latency_median_us else "—"
        speedup = f"{r.baselines.speedup_vs_compiled:.2f}x" if r.baselines.speedup_vs_compiled else "—"
        completeness = f"{r.artifacts.completeness_score:.0%}" if r.artifacts.artifacts_present else "—"
        status = r.status if r.status != "pending" else r.verification.overall_status
        lines.append(
            f"| {r.study.study_id or '—'} | {r.study.case_id or '—'} | {r.system_name} | "
            f"{r.model_name} | {r.target_name} | {status} | {r.total_compile_time_ms:.0f} | "
            f"{latency} | {speedup} | {completeness} |"
        )
    return "\n".join(lines)


def ablation_table(records: list[RunRecord]) -> str:
    """Generate an ablation comparison table."""

    if not records:
        return "No ablation records found."

    lines = [
        "| Study | Case | Ablation | System | Compile (ms) | Verification | Latency (μs) |",
        "|-------|------|----------|--------|-------------|-------------|-------------|",
    ]
    for r in records:
        ablation = r.config.get("ablation", "full")
        latency = f"{r.performance.latency_median_us:.1f}" if r.performance.latency_median_us else "—"
        lines.append(
            f"| {r.study.study_id or '—'} | {r.study.case_id or '—'} | {ablation} | {r.system_name} | "
            f"{r.total_compile_time_ms:.0f} | {r.verification.overall_status} | {latency} |"
        )
    return "\n".join(lines)


def artifact_completeness_table(records: list[RunRecord]) -> str:
    """Generate an artifact completeness table."""

    if not records:
        return "No records found."

    lines = [
        "| System | Model | Target | Bundle | Completeness | Missing |",
        "|--------|-------|--------|--------|--------------|---------|",
    ]
    for r in records:
        bundle = Path(r.artifacts.bundle_path).name if r.artifacts.bundle_path else "—"
        missing = ", ".join(r.artifacts.missing_artifacts) if r.artifacts.missing_artifacts else "—"
        lines.append(
            f"| {r.system_name} | {r.model_name} | {r.target_name} | {bundle} | "
            f"{r.artifacts.completeness_score:.0%} | {missing} |"
        )
    return "\n".join(lines)


def coverage_table(records: list[RunRecord]) -> str:
    """Generate a capture/import coverage table."""

    if not records:
        return "No records found."

    lines = [
        "| System | Model | Target | Mode | Export | Analysis | Graphs | Graph Breaks | Decomposition | Opaque Ops |",
        "|--------|-------|--------|------|--------|----------|--------|--------------|---------------|------------|",
    ]
    for r in records:
        export = "yes" if r.capture.export_success else "no"
        analysis = "yes" if r.capture.analysis_success else "no"
        lines.append(
            f"| {r.system_name} | {r.model_name} | {r.target_name} | {r.capture.capture_mode or '—'} | {export} | "
            f"{analysis} | {r.capture.graph_count} | {r.capture.graph_break_count} | "
            f"{r.capture.decomposition_coverage:.2f} | {r.capture.opaque_ops} |"
        )
    return "\n".join(lines)


def export_csv(records: list[RunRecord], output_path: str | Path) -> Path:
    """Export key metrics to CSV."""

    path = Path(output_path)
    import csv

    fieldnames = [
        "run_id",
        "study_id",
        "case_id",
        "system_name",
        "model_name",
        "target_name",
        "status",
        "readiness",
        "expected_status",
        "ablation",
        "total_compile_time_ms",
        "capture_mode",
        "analysis_success",
        "graph_count",
        "export_success",
        "decomposition_coverage",
        "eqsat_ops_before",
        "eqsat_ops_after",
        "eqsat_reduction_pct",
        "recipe_total_ops",
        "solver_makespan_us",
        "solver_copy_bytes",
        "verification_status",
        "latency_median_us",
        "throughput_samples_per_sec",
        "artifact_completeness_score",
        "promotion_status",
        # Codegen-specific columns
        "pct_native",
        "pct_library",
        "pct_fallback",
        "pct_generated",
        "roofline_gap",
        "fallback_flop_share",
        "materialized_transposes",
        "codegen_eligible",
        "codegen_faster",
        "geo_mean_speedup",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "run_id": r.run_id,
                    "study_id": r.study.study_id,
                    "case_id": r.study.case_id,
                    "system_name": r.system_name,
                    "model_name": r.model_name,
                    "target_name": r.target_name,
                    "status": r.status,
                    "readiness": r.readiness,
                    "expected_status": r.expected_status,
                    "ablation": r.config.get("ablation", ""),
                    "total_compile_time_ms": r.total_compile_time_ms,
                    "capture_mode": r.capture.capture_mode,
                    "analysis_success": r.capture.analysis_success,
                    "graph_count": r.capture.graph_count,
                    "export_success": r.capture.export_success,
                    "decomposition_coverage": r.capture.decomposition_coverage,
                    "eqsat_ops_before": r.eqsat.ops_before,
                    "eqsat_ops_after": r.eqsat.ops_after,
                    "eqsat_reduction_pct": r.eqsat.ops_reduction_pct,
                    "recipe_total_ops": r.recipe.total_recipe_ops,
                    "solver_makespan_us": r.solver.schedule_makespan_us,
                    "solver_copy_bytes": r.solver.copy_bytes,
                    "verification_status": r.verification.overall_status,
                    "latency_median_us": r.performance.latency_median_us,
                    "throughput_samples_per_sec": r.performance.throughput_samples_per_sec,
                    "artifact_completeness_score": r.artifacts.completeness_score,
                    "promotion_status": r.promotion_status,
                    # Codegen-specific
                    "pct_native": r.kernels.pct_native,
                    "pct_library": r.kernels.pct_library,
                    "pct_fallback": r.kernels.pct_fallback,
                    "pct_generated": r.kernels.pct_generated,
                    "roofline_gap": r.kernels.roofline_gap,
                    "fallback_flop_share": r.fallback_pressure.fallback_flop_share,
                    "materialized_transposes": r.layout_friction.materialized_transposes,
                    "codegen_eligible": r.codegen_funnel.eligible,
                    "codegen_faster": r.codegen_funnel.faster,
                    "geo_mean_speedup": r.codegen_funnel.geo_mean_speedup,
                }
            )
    return path


def export_normalized_suite_json(records: list[RunRecord], output_dir: str | Path) -> list[Path]:
    """Export the normalized cross-suite JSON projection for records with suite metadata."""

    suite_records = [record for record in records if record.suite.suite_id]
    return write_normalized_suite_results(suite_records, output_dir)


__all__ = [
    "ablation_table",
    "artifact_completeness_table",
    "coverage_table",
    "export_csv",
    "export_normalized_suite_json",
    "load_all_results",
    "summary_table",
]
