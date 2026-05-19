"""Unit tests for the proof writer."""

from __future__ import annotations

import json

from xpu_rt.targets.backends.qnn.proof import ProofWriter, RoundSummary


def test_record_appends_one_line_per_call(tmp_path):
    pw = ProofWriter(tmp_path)
    pw.record(
        tool_name="xpu_rt_qnn_schedule_round", round_index=0,
        args={"workload_id": "yolov8n", "dry_run": False},
        result={"ok": True, "makespan_us": 325_000.0},
        rationale="initial calibration",
    )
    pw.record(
        tool_name="xpu_rt_qnn_propose_split", round_index=2,
        args={"group_id": "yolov8n", "target_backend": "DSP"},
        result={"ok": True, "makespan_us": 249_466.0},
        rationale="block CPU contention; move yolov8n to DSP",
    )
    lines = (tmp_path / "agent_trace.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["tool"] == "xpu_rt_qnn_schedule_round"
    assert rec0["round"] == 0
    assert rec0["rationale"] == "initial calibration"
    assert rec1["result_summary"]["makespan_us"] == 249_466.0


def test_redact_drops_pretty_markdown(tmp_path):
    pw = ProofWriter(tmp_path)
    pw.record(
        tool_name="x", round_index=0,
        args={"pretty_markdown": "## huge\nblock\n…",
              "out_dir": "/tmp/run", "param": "x" * 500},
        result={},
        rationale="r",
    )
    rec = json.loads((tmp_path / "agent_trace.jsonl").read_text().strip())
    assert "pretty_markdown" not in rec["args"]
    # Long values get truncated with an ellipsis.
    assert rec["args"]["param"].endswith("…")
    assert rec["args"]["out_dir"] == "/tmp/run"


def test_final_report_renders_arc_and_decision_log(tmp_path):
    pw = ProofWriter(tmp_path)
    pw.record(
        tool_name="xpu_rt_qnn_schedule_round", round_index=0,
        args={}, result={"ok": True}, rationale="calibration",
    )
    pw.record(
        tool_name="xpu_rt_qnn_set_deadline_and_reschedule", round_index=4,
        args={"makespan_bound_us": 355000},
        result={"feasible": True, "makespan_us": 355000.0,
                "deadlines_met_count": 13, "deadlines_total": 13},
        rationale="MOSEK MILP solved feasibly",
    )

    rounds = [
        RoundSummary(round_index=0, granularity="coarse",
                     action="calibrate",
                     predicted_makespan_us=None,
                     measured_makespan_us=325000.0,
                     feasibility="pass"),
        RoundSummary(round_index=4, granularity="fine",
                     action="reschedule",
                     predicted_makespan_us=355000.0,
                     measured_makespan_us=355000.0,
                     feasibility="pass",
                     deadlines_met=13, deadlines_total=13,
                     rationale="MOSEK MILP found a feasible 12-dronet "
                               "assignment within yolov8n's makespan",
                     assignment={"yolov8n": "HTA",
                                 "dronet.0..11": "CPU"}),
    ]
    p = pw.write_final_report(rounds, target_makespan_us=355000.0,
                              n_dronet_copies=12)
    body = p.read_text()
    assert "## Optimization arc" in body
    assert "PASS" in body
    assert "yolov8n" in body
    assert "12× DroNet" in body
    # Tool-invocation table includes both tools we recorded.
    assert "xpu_rt_qnn_schedule_round" in body
    assert "xpu_rt_qnn_set_deadline_and_reschedule" in body
