"""Render tests for the Claude-Code-facing markdown blocks."""

from __future__ import annotations

import pytest

from xpu_rt.ui.markdown import (
    render_decision_markdown,
    render_deltas_markdown,
    render_gantt_markdown,
    render_round_summary_markdown,
)


@pytest.fixture()
def paper_round():
    schedule = {
        "makespan_us": 18000.0,
        "ops": [
            {"name": "yolov8n.backbone", "workload": "yolov8n", "machine": "HTA",
             "start_us": 0.0, "finish_us": 6000.0, "predicted_us": 6000.0},
            {"name": "dronet_0.body", "workload": "dronet_0", "machine": "HTA",
             "start_us": 6000.0, "finish_us": 11000.0, "predicted_us": 5000.0},
            {"name": "dronet_1.body", "workload": "dronet_1", "machine": "GPU",
             "start_us": 0.0, "finish_us": 4500.0, "predicted_us": 4500.0},
            {"name": "yolov8n.head", "workload": "yolov8n", "machine": "CPU",
             "start_us": 11000.0, "finish_us": 12500.0, "predicted_us": 1500.0},
        ],
    }
    profile = {"dispatches": {
        "yolov8n.backbone": {"qnn_hta": {"mean_us": 12500.0}},
        "yolov8n.head":     {"cpu":     {"mean_us": 1620.0}},
        "dronet_0.body":    {"qnn_hta": {"mean_us": 5300.0}},
        "dronet_1.body":    {"qnn_gpu": {"mean_us": 4800.0}},
    }}
    return schedule, profile


def test_gantt_renders_three_machines_and_legend(paper_round):
    schedule, _ = paper_round
    out = render_gantt_markdown(schedule, width=40, title="t")
    assert "**t**" in out
    assert "HTA" in out and "GPU" in out and "CPU" in out
    assert "legend:" in out
    # All four workloads have a marker in the legend.
    for wl in ("yolov8n", "dronet_0", "dronet_1"):
        assert wl in out


def test_deltas_marks_outliers():
    rows = [
        {"dispatch": "yolov8n.backbone", "machine": "HTA",
         "predicted_us": 6000.0, "measured_us": 12500.0,
         "delta_us": 6500.0, "ratio": 2.083},
        {"dispatch": "dronet_0.body", "machine": "HTA",
         "predicted_us": 5000.0, "measured_us": 5300.0,
         "delta_us": 300.0, "ratio": 1.06},
        {"dispatch": "missing.op", "machine": "CPU",
         "predicted_us": 100.0, "measured_us": None,
         "delta_us": None, "ratio": None},
    ]
    out = render_deltas_markdown(rows)
    assert "⚠️" in out         # the 2.08 ratio row
    assert "12,500" in out
    assert "_missing_" in out
    # The markdown header row is present.
    assert "| dispatch | dev |" in out


def test_decision_markdown_lists_split_candidate():
    out = render_decision_markdown(
        round_index=2,
        makespan_us=14_200,
        greedy_pick="split:yolov8n.backbone",
        split_candidates=[{
            "dispatch_id": "yolov8n.backbone", "machine": "HTA",
            "predicted_us": 6000, "measured_us": 12500,
            "ratio": 2.08, "region_share": 0.33,
            "rationale": "ratio=2.08; share=33%",
        }],
        coarsen_candidates=[],
        legal_candidate_ids=["split:yolov8n.backbone", "keep:all"],
        prev_makespan_us=18000.0,
    )
    assert "Round 2" in out
    assert "split:yolov8n.backbone" in out
    assert "2.08" in out
    assert "Δ -21.1%" in out or "Δ-21.1%" in out or "-21.1%" in out


def test_round_summary_stitches_three_blocks(paper_round):
    schedule, _ = paper_round
    rows = [
        {"dispatch": "yolov8n.backbone", "machine": "HTA",
         "predicted_us": 6000.0, "measured_us": 12500.0,
         "delta_us": 6500.0, "ratio": 2.083},
    ]
    out = render_round_summary_markdown(
        round_index=0, makespan_us=18_000.0,
        schedule=schedule, deltas=rows,
        greedy_pick="split:yolov8n.backbone",
        split_candidates=[{
            "dispatch_id": "yolov8n.backbone", "machine": "HTA",
            "predicted_us": 6000, "measured_us": 12500,
            "ratio": 2.083, "region_share": 0.33,
        }],
        legal_candidate_ids=["split:yolov8n.backbone", "keep:all"],
    )
    assert "Round 0 decision" in out
    assert "legend:" in out             # Gantt block present
    assert "| dispatch | dev |" in out  # Deltas block present


def test_pretty_markdown_in_dry_run_tool_returns(tmp_path):
    """End-to-end: every QNN MCP tool emits a pretty_markdown field."""
    from xpu_rt.mcp.tools.qnn_flow import (
        xpu_rt_qnn_decide_granularity,
        xpu_rt_qnn_profile_on_board,
        xpu_rt_qnn_schedule_round,
    )

    class _S:
        pass

    sess = _S()
    out_dir = tmp_path / "run"
    from pathlib import Path
    cost_table = (
        Path(__file__).resolve().parents[1]
        / "python" / "xpu_rt" / "targets" / "backends" / "qnn"
        / "qrb5165_costs.json"
    )
    sched = xpu_rt_qnn_schedule_round(
        sess, out_dir=str(out_dir), round_index=0,
        workload_id="yolov8n", cost_table=str(cost_table), dry_run=True,
    )
    assert "pretty_markdown" in sched and "legend:" in sched["pretty_markdown"]
    prof = xpu_rt_qnn_profile_on_board(
        sess, out_dir=str(out_dir), round_index=0,
        schedule_path=sched["schedule_path"],
        cost_table=str(cost_table), dry_run=True,
    )
    assert "pretty_markdown" in prof and "| dispatch | dev |" in prof["pretty_markdown"]
    decide = xpu_rt_qnn_decide_granularity(
        sess, out_dir=str(out_dir), round_index=0,
        schedule_path=sched["schedule_path"],
        profile_path=prof["profiled_manifest_path"],
    )
    assert "pretty_markdown" in decide
    assert "round_summary_markdown" in decide
    assert "Round 0 decision" in decide["pretty_markdown"]
