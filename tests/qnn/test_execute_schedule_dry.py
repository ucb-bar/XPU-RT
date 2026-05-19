"""Translate-schedule tests for the on-board executor.

We never actually SSH in these tests — instead we exercise the
public helpers that build the per-lane shell scripts + parse
trace markers. The real SSH path is exercised at the
closed-loop proof stage.
"""

from __future__ import annotations

from xpu_rt.targets.backends.qnn.execute_schedule import (
    LaneInvocation, _lane_script, _parse_lane_output, _resolve_invocations,
)


def test_lane_script_contains_qnn_net_run_invocation():
    inv = LaneInvocation(
        op_id="yolov8n", workload_id="yolov8n",
        dlc_path="/root/models/yolov8n/yolov8n_quantized.dlc",
        context_path=None, backend="DSP",
        input_list="/root/models/yolov8n/input_list.txt", iters=10,
        predicted_us=300_000.0,
    )
    script = _lane_script("DSP", [inv])
    assert "qnn-net-run" in script
    assert "/root/qairt/lib/target/libQnnDsp.so" in script
    assert "START_NS yolov8n" in script
    assert "END_NS yolov8n" in script
    assert "--dlc_path /root/models/yolov8n/yolov8n_quantized.dlc" in script


def test_lane_script_picks_retrieve_context_when_set():
    inv = LaneInvocation(
        op_id="conv0", workload_id="yolov8n",
        dlc_path=None, context_path="/root/contexts/conv0_dsp.bin",
        backend="DSP", input_list="/root/inputs.txt", iters=1,
        predicted_us=1_800.0,
    )
    script = _lane_script("DSP", [inv])
    assert "--retrieve_context /root/contexts/conv0_dsp.bin" in script
    assert "--dlc_path" not in script


def test_parse_lane_output_extracts_per_op_timestamps():
    stdout = (
        "LANE_START DSP 1000\n"
        "START_NS conv0 2000\n"
        "END_NS conv0 5000\n"
        "START_NS conv1 5100\n"
        "END_NS conv1 8000\n"
        "LANE_END DSP 8100\n"
    )
    starts, ends = _parse_lane_output(stdout)
    assert starts == {"conv0": 2000, "conv1": 5100}
    assert ends == {"conv0": 5000, "conv1": 8000}


def test_resolve_invocations_groups_by_backend():
    sched = {
        "ops": [
            {"name": "yolov8n", "workload": "yolov8n", "machine": "DSP",
             "start_us": 0.0, "finish_us": 300000.0, "predicted_us": 300000.0},
            {"name": "dronet.0", "workload": "dronet", "machine": "CPU",
             "start_us": 0.0, "finish_us": 7400.0, "predicted_us": 7400.0},
            {"name": "dronet.1", "workload": "dronet", "machine": "CPU",
             "start_us": 7400.0, "finish_us": 14800.0, "predicted_us": 7400.0},
        ],
    }
    workload_specs = {
        "yolov8n": {"dlc_path": "/root/models/yolov8n/yolov8n_quantized.dlc",
                    "input_list": "/root/models/yolov8n/input_list.txt"},
        "dronet":  {"dlc_path": "/root/models/dronet/dronet.dlc",
                    "input_list": "/root/models/dronet/input_list.txt"},
    }
    lanes = _resolve_invocations(sched, workload_specs=workload_specs)
    assert set(lanes.keys()) == {"DSP", "CPU"}
    # Collapse merged the two dronet runs into one invocation.
    assert len(lanes["CPU"]) == 1
    assert lanes["CPU"][0].op_id == "dronet.0+dronet.1"
    assert lanes["CPU"][0].iters == 2
    assert lanes["DSP"][0].dlc_path.endswith("yolov8n_quantized.dlc")


def test_resolve_invocations_no_collapse_keeps_each_island_separate():
    sched = {
        "ops": [
            {"name": "dronet.0", "workload": "dronet", "machine": "CPU",
             "start_us": 0.0, "finish_us": 7400.0, "predicted_us": 7400.0},
            {"name": "dronet.1", "workload": "dronet", "machine": "CPU",
             "start_us": 7400.0, "finish_us": 14800.0, "predicted_us": 7400.0},
        ],
    }
    workload_specs = {
        "dronet":  {"dlc_path": "/root/models/dronet/dronet.dlc",
                    "input_list": "/root/models/dronet/input_list.txt"},
    }
    lanes = _resolve_invocations(sched, workload_specs=workload_specs, collapse=False)
    assert [inv.op_id for inv in lanes["CPU"]] == ["dronet.0", "dronet.1"]


def test_resolve_invocations_raises_when_artifact_missing():
    sched = {"ops": [{"name": "x", "workload": "ghost",
                       "machine": "CPU", "predicted_us": 1.0}]}
    import pytest

    with pytest.raises(KeyError):
        _resolve_invocations(sched, workload_specs={})
