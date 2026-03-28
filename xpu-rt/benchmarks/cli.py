"""CLI for the benchmark study harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.adapters import check_baseline_availability
from benchmarks.compare import (
    ablation_table,
    artifact_completeness_table,
    coverage_table,
    export_csv,
    load_all_results,
    summary_table,
)
from benchmarks.plots import generate_all_plots
from benchmarks.registry import build_default_registry
from benchmarks.runner import run_case, run_defect_campaign, run_study
from benchmarks.spec import WorkspaceConfig
from benchmarks.suite_runner import (
    export_suite_results,
    list_suite_workloads,
    list_suites,
    probe_suite,
    run_suite,
    run_suite_workload,
)
from compgen.benchmarks import SuiteRunConfig


def _workspace_from_args(args: argparse.Namespace) -> WorkspaceConfig:
    if args.workspace_config:
        return WorkspaceConfig.from_yaml(args.workspace_config)
    return WorkspaceConfig.default(Path(__file__).resolve().parent.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks", description="CompGen benchmark study harness")
    parser.add_argument("--workspace-config", default="", help="Optional YAML workspace config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_case_parser = subparsers.add_parser("run-case", help="Run one benchmark case")
    run_case_parser.add_argument("case_id")
    run_case_parser.add_argument("--output-dir", default="benchmarks/results")

    run_study_parser = subparsers.add_parser("run-study", help="Run a full study")
    run_study_parser.add_argument("study_id")
    run_study_parser.add_argument("--output-dir", default="benchmarks/results")

    aggregate_parser = subparsers.add_parser("aggregate", help="Summarize benchmark results")
    aggregate_parser.add_argument("results_dir")
    aggregate_parser.add_argument("--csv", default="")

    plots_parser = subparsers.add_parser("plot", help="Generate plots from result JSONs")
    plots_parser.add_argument("results_dir")
    plots_parser.add_argument("--output-dir", default="benchmarks/results/plots")

    defects_parser = subparsers.add_parser("red-team", help="Run the verification defect campaign")
    defects_parser.add_argument("case_id")
    defects_parser.add_argument("--output-dir", default="benchmarks/results")

    baselines_parser = subparsers.add_parser("check-baselines", help="Check baseline adapter availability")
    baselines_parser.add_argument("--baseline-id", action="append", dest="baseline_ids", default=[])

    list_suites_parser = subparsers.add_parser("list-suites", help="List recognized benchmark suites")
    list_suites_parser.add_argument("--device", default="cpu")
    list_suites_parser.add_argument("--dtype", default="float32")

    list_workloads_parser = subparsers.add_parser("list-suite-workloads", help="List workloads for a benchmark suite")
    list_workloads_parser.add_argument("suite_id")
    list_workloads_parser.add_argument("--blessed-only", action="store_true")

    probe_suite_parser = subparsers.add_parser("probe-suite", help="Probe one benchmark suite")
    probe_suite_parser.add_argument("suite_id")
    probe_suite_parser.add_argument("--device", default="cpu")
    probe_suite_parser.add_argument("--dtype", default="float32")

    run_suite_parser = subparsers.add_parser("run-suite", help="Run a benchmark suite")
    run_suite_parser.add_argument("suite_id")
    run_suite_parser.add_argument("--output-dir", default="benchmarks/results/suites")
    run_suite_parser.add_argument("--device", default="cpu")
    run_suite_parser.add_argument("--dtype", default="float32")
    run_suite_parser.add_argument("--batch-size", type=int, default=1)
    run_suite_parser.add_argument("--mode", default="inference")
    run_suite_parser.add_argument("--iterations", type=int, default=10)
    run_suite_parser.add_argument("--warmup", type=int, default=3)
    run_suite_parser.add_argument("--all-workloads", action="store_true")
    run_suite_parser.add_argument("--output-tag", default="")

    run_suite_workload_parser = subparsers.add_parser("run-suite-workload", help="Run one workload from a benchmark suite")
    run_suite_workload_parser.add_argument("suite_id")
    run_suite_workload_parser.add_argument("workload_id")
    run_suite_workload_parser.add_argument("--output-dir", default="benchmarks/results/suites")
    run_suite_workload_parser.add_argument("--device", default="cpu")
    run_suite_workload_parser.add_argument("--dtype", default="float32")
    run_suite_workload_parser.add_argument("--batch-size", type=int, default=1)
    run_suite_workload_parser.add_argument("--mode", default="inference")
    run_suite_workload_parser.add_argument("--iterations", type=int, default=10)
    run_suite_workload_parser.add_argument("--warmup", type=int, default=3)
    run_suite_workload_parser.add_argument("--output-tag", default="")

    export_suite_parser = subparsers.add_parser("export-suite-results", help="Export normalized cross-suite JSON files")
    export_suite_parser.add_argument("results_dir")
    export_suite_parser.add_argument("--output-dir", default="benchmarks/results/normalized")

    return parser


def _suite_run_config(args: argparse.Namespace) -> SuiteRunConfig:
    return SuiteRunConfig(
        mode=getattr(args, "mode", "inference"),
        device=getattr(args, "device", "cpu"),
        dtype=getattr(args, "dtype", "float32"),
        batch_size=getattr(args, "batch_size", 1),
        blessed_only=not getattr(args, "all_workloads", False),
        num_iterations=getattr(args, "iterations", 10),
        warmup_iterations=getattr(args, "warmup", 3),
        output_tag=getattr(args, "output_tag", ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = build_default_registry()
    workspace = _workspace_from_args(args)

    if args.command == "run-case":
        records = run_case(args.case_id, registry=registry, workspace=workspace, output_dir=args.output_dir)
        print(summary_table(records))
        return 0

    if args.command == "run-study":
        records = run_study(args.study_id, registry=registry, workspace=workspace, output_dir=args.output_dir)
        print(summary_table(records))
        print()
        print(ablation_table([record for record in records if record.system_name == "compgen"]))
        return 0

    if args.command == "aggregate":
        records = load_all_results(args.results_dir)
        print(summary_table(records))
        print()
        print(coverage_table(records))
        print()
        print(artifact_completeness_table(records))
        if args.csv:
            path = export_csv(records, args.csv)
            print()
            print(f"CSV written to {path}")
        return 0

    if args.command == "plot":
        records = load_all_results(args.results_dir)
        paths = generate_all_plots(records, args.output_dir)
        for path in paths:
            print(path)
        return 0

    if args.command == "red-team":
        record = run_defect_campaign(args.case_id, registry=registry, workspace=workspace, output_dir=args.output_dir)
        print(summary_table([record]))
        return 0

    if args.command == "check-baselines":
        results = check_baseline_availability(
            registry,
            workspace,
            baseline_ids=args.baseline_ids or None,
        )
        for baseline_id, status in results.items():
            print(f"{baseline_id}: {status}")
        return 0

    if args.command == "list-suites":
        statuses = list_suites(workspace=workspace, config=SuiteRunConfig(device=args.device, dtype=args.dtype))
        for suite_id, status in statuses.items():
            availability = "available" if status.available else f"unavailable ({status.reason})"
            source = f" root={status.source_root}" if status.source_root else ""
            print(f"{suite_id}: {availability}{source}")
        return 0

    if args.command == "list-suite-workloads":
        entries = list_suite_workloads(args.suite_id, workspace=workspace, blessed_only=args.blessed_only)
        for entry in entries:
            blessed = "blessed" if entry.blessed else "extra"
            print(f"{entry.workload_id}\t{blessed}\t{entry.description}")
        return 0

    if args.command == "probe-suite":
        status = probe_suite(args.suite_id, workspace=workspace, config=SuiteRunConfig(device=args.device, dtype=args.dtype))
        availability = "available" if status.available else f"unavailable ({status.reason})"
        source = f" root={status.source_root}" if status.source_root else ""
        print(f"{args.suite_id}: {availability}{source}")
        return 0

    if args.command == "run-suite":
        records = run_suite(
            args.suite_id,
            workspace=workspace,
            output_dir=args.output_dir,
            config=_suite_run_config(args),
        )
        print(summary_table(records))
        return 0

    if args.command == "run-suite-workload":
        records = run_suite_workload(
            args.suite_id,
            args.workload_id,
            workspace=workspace,
            output_dir=args.output_dir,
            config=_suite_run_config(args),
        )
        print(summary_table(records))
        return 0

    if args.command == "export-suite-results":
        paths = export_suite_results(args.results_dir, args.output_dir)
        for path in paths:
            print(path)
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
