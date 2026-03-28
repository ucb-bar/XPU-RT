"""Tests for the benchmark CLI."""

from __future__ import annotations

from benchmarks.cli import build_parser, main


def test_parser_accepts_run_study() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-study", "paper_subset"])
    assert args.command == "run-study"
    assert args.study_id == "paper_subset"


def test_cli_check_baselines_runs(capsys) -> None:
    exit_code = main(["check-baselines", "--baseline-id", "expert_fixture"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "expert_fixture" in captured.out


def test_parser_accepts_run_suite() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-suite", "mlperf"])
    assert args.command == "run-suite"
    assert args.suite_id == "mlperf"


def test_cli_list_suites_runs(capsys) -> None:
    exit_code = main(["list-suites"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "torchbench" in captured.out
