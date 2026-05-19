"""Round-trip tests for propose_fusion / propose_split."""

from __future__ import annotations

import json

from xpu_rt.mcp.tools.qnn_granularity import (
    xpu_rt_qnn_inspect_island_variants,
    xpu_rt_qnn_propose_fusion,
    xpu_rt_qnn_propose_split,
)


class _S:
    pass


def _seed_schedule(tmp_path, ops):
    p = tmp_path / "round_0" / "schedule.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "schema_version": "qnn_native_schedule_v1",
        "makespan_us": max(o["finish_us"] for o in ops),
        "machines": ["HTA", "GPU", "CPU"],
        "ops": ops,
        "dispatches": {o["name"]: o for o in ops},
    }))
    return p


def test_inspect_returns_per_island_variants(tmp_path):
    p = _seed_schedule(tmp_path, [
        {"name": "yolov8n", "workload": "yolov8n", "machine": "HTA",
         "start_us": 0.0, "finish_us": 355_000.0,
         "predicted_us": 355_000.0, "deadline_us": 355_000.0},
        {"name": "dronet.0", "workload": "dronet", "machine": "CPU",
         "start_us": 0.0, "finish_us": 7_400.0,
         "predicted_us": 7_400.0, "deadline_us": 355_000.0},
    ])
    out = xpu_rt_qnn_inspect_island_variants(
        _S(), out_dir=str(tmp_path), round_index=0,
        schedule_path=str(p),
    )
    assert out["ok"]
    assert out["n_islands"] == 2
    assert "pretty_markdown" in out
    assert "yolov8n" in out["pretty_markdown"]
    assert "dronet.0" in out["pretty_markdown"]


def test_propose_fusion_merges_adjacent_same_backend(tmp_path):
    p = _seed_schedule(tmp_path, [
        {"name": "a", "workload": "yolov8n", "machine": "HTA",
         "start_us": 0.0, "finish_us": 100.0, "predicted_us": 100.0},
        {"name": "b", "workload": "yolov8n", "machine": "HTA",
         "start_us": 100.0, "finish_us": 250.0, "predicted_us": 150.0},
    ])
    out = xpu_rt_qnn_propose_fusion(
        _S(), out_dir=str(tmp_path), round_index=0,
        schedule_path=str(p), first_id="a", second_id="b",
        rationale="adjacent same-backend pair with high transfer",
    )
    assert out["ok"]
    assert out["makespan_us"] == 250.0
    sched = json.loads(open(out["schedule_path"]).read())
    names = {o["name"] for o in sched["ops"]}
    assert "a+b" in names
    assert "a" not in names and "b" not in names


def test_propose_fusion_rejects_cross_backend(tmp_path):
    p = _seed_schedule(tmp_path, [
        {"name": "a", "workload": "yolov8n", "machine": "HTA",
         "start_us": 0.0, "finish_us": 100.0, "predicted_us": 100.0},
        {"name": "b", "workload": "yolov8n", "machine": "GPU",
         "start_us": 0.0, "finish_us": 200.0, "predicted_us": 200.0},
    ])
    out = xpu_rt_qnn_propose_fusion(
        _S(), out_dir=str(tmp_path), round_index=0,
        schedule_path=str(p), first_id="a", second_id="b",
    )
    assert not out["ok"]
    assert "same backend" in out["error"]


def test_propose_split_re_places_island(tmp_path):
    p = _seed_schedule(tmp_path, [
        {"name": "yolov8n", "workload": "yolov8n", "machine": "CPU",
         "start_us": 0.0, "finish_us": 325_000.0, "predicted_us": 325_000.0},
        {"name": "dronet.0", "workload": "dronet", "machine": "CPU",
         "start_us": 325_000.0, "finish_us": 332_400.0, "predicted_us": 7_400.0},
    ])
    out = xpu_rt_qnn_propose_split(
        _S(), out_dir=str(tmp_path), round_index=0,
        schedule_path=str(p),
        group_id="yolov8n", target_backend="HTA",
        new_predicted_us=355_000.0,
        rationale="block CPU for dronet copies",
    )
    assert out["ok"]
    # The split places yolov8n on HTA at start 0; dronet.0 is now alone
    # on CPU at start 0 too → makespan = max(355000, 7400) = 355000.
    sched = json.loads(open(out["schedule_path"]).read())
    yolo = next(o for o in sched["ops"] if o["name"] == "yolov8n")
    assert yolo["machine"] == "HTA"
    assert yolo["predicted_us"] == 355_000.0
    assert out["makespan_us"] == 355_000.0
