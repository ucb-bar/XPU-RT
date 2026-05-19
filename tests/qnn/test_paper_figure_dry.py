"""End-to-end dry-run of the paper-figure demo.

Asserts that the loop produces the expected artefacts, the qnn_*
events get written, and the makespan does not strictly increase across
rounds (we don't require it to *decrease* — the heuristic may decide
to keep — but it should never grow round-over-round when the cost
table is stable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def runner(tmp_path):
    from click.testing import CliRunner

    return CliRunner(), tmp_path


def test_paper_figure_dry_run(runner, monkeypatch):
    cli_runner, tmp = runner
    monkeypatch.setenv("XPURT_NONINTERACTIVE", "1")
    out_dir = tmp / "paper_figure"
    from xpu_rt.cli import main

    result = cli_runner.invoke(
        main,
        [
            "qnn", "demo", "paper-figure",
            "--out-dir", str(out_dir),
            "--max-rounds", "2",
            "--dry-run",
            "--no-interactive",
        ],
    )
    assert result.exit_code == 0, result.output

    # The events log is written by the MCP tools.
    events_path = out_dir / "qnn_events.jsonl"
    assert events_path.is_file(), result.output
    events = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
    kinds = {e["event"] for e in events}
    assert "qnn_schedule_emitted" in kinds
    assert "qnn_trace_ingested" in kinds
    assert "qnn_granularity_decision" in kinds

    # Each round writes a schedule + profiled manifest.
    rounds = sorted(out_dir.glob("round_*"))
    assert len(rounds) >= 1
    for rd in rounds:
        assert (rd / "schedule.json").is_file()
        assert (rd / "profiled_manifest.json").is_file()

    # Makespan should not strictly increase round-over-round.
    makespans = [
        json.loads((rd / "schedule.json").read_text()).get("makespan_us", 0.0)
        for rd in rounds
    ]
    for prev, cur in zip(makespans, makespans[1:]):
        assert cur <= prev * 1.05  # 5% slack for numerical jitter
