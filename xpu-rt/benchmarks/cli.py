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

    return parser


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

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
