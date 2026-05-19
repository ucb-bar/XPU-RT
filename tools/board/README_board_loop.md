# Board-side closed-loop measurement (QRB5165)

This directory contains the host ↔ board round-trip the user runs to feed
real on-board measurements back into the Stage-4 feedback loop.

```
host                                      QRB5165 (Linux on the board)
----                                      ---------------------------
emit_loop_plan.py  ──── plan.json ────▶
                                          run_loop_plan_on_qrb5165.sh
                       ◀─── measurement.json ────
xpu_rt_apply_measurement (Python)
```

The two scripts can be used independently:

* `emit_loop_plan.py` — host-side. Reads a persisted `LoopState`
  (e.g. `build/loops/yolov8n__qrb5165.json`) and writes a board-runnable
  `plan.json` (`loop_plan_board_v1`).
* `run_loop_plan_on_qrb5165.sh` — board-side. Runs `qnn-net-run` for each
  partition in `plan.json` and writes `measurement.json`
  (`measurement_record_board_v1`).

Neither script ssh's into the board; that step belongs to the operator
and is intentionally explicit.

## Honest limitations

* This script suite cannot ssh into the board on its own — the operator
  copies files and shells in.
* It cannot pre-build DLCs. `plan.json` only assembles DLC paths; the
  files must already exist on the board (the standard convention is
  `/root/models/<workload>/<workload>{,_fp16,_quantized}.dlc`).
* It cannot tell the loop where to find the result. After copying
  `measurement.json` back to the host, the user invokes
  `xpu_rt_apply_measurement` (snippet below).

## Step-by-step

### 1. Generate `plan.json` on the host

After at least one successful `xpu_rt_feedback_step` (so
`build/loops/yolov8n__qrb5165.json` exists):

```bash
uv run python scripts/board/emit_loop_plan.py \
    --loop-state build/loops/yolov8n__qrb5165.json \
    --output     build/loops/yolov8n__qrb5165__plan.json \
    --iters      10
```

### 2. Copy the artefacts to the board

```bash
scp scripts/board/run_loop_plan_on_qrb5165.sh \
    root@<board>:/data/local/tmp/
scp build/loops/yolov8n__qrb5165__plan.json \
    root@<board>:/data/local/tmp/plan.json
```

DLCs and `input_list.txt` are assumed to already live on the board (e.g.
under `/root/models/yolov8n/`). If you need a custom input list, push it
explicitly and pass `--input-list` to either the host generator or the
board runner.

### 3. SSH to the board and execute

```bash
ssh root@<board>

# QNN environment — same idiom as
# xpu-rt/python/xpu_rt/targets/backends/qnn/on_board_runner.py:131-132.
export QNN_SDK_ROOT=/root/qairt
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/target:${LD_LIBRARY_PATH:-}
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v66;/dsp/cdsp;/dsp"

cd /data/local/tmp
bash run_loop_plan_on_qrb5165.sh \
    --plan-json   plan.json \
    --output-json measurement.json \
    --input-list  /root/models/yolov8n/input_list.txt
```

Expected output:

```
[plan] running partition=p0 backend=DSP dlc=/root/models/yolov8n/yolov8n_quantized.dlc iters=10
... (one line per partition) ...
[plan] wrote measurement.json
[plan] per_backend_mean_us = {'DSP': 254800.0, 'CPU': 134300.0}
```

### 4. Copy `measurement.json` back to the host

```bash
scp root@<board>:/data/local/tmp/measurement.json ./measurement.json
```

### 5. Apply the measurement to the loop state on the host

The board's `measurement_record_board_v1` carries one mean per backend
across all partitions. The loop's
`xpu_rt_apply_measurement` consumes one
`(workload, backend, measured_us, per_op_sum_us, predicted_us)` record
at a time, so we adapt one per backend and apply them sequentially.

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from xpu_rt.mcp.tools.feedback_loop_tools import xpu_rt_apply_measurement
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    state_from_dict, state_to_dict, save_loop_state,
)

board = json.loads(Path("measurement.json").read_text())
state_path = Path("build/loops/yolov8n__qrb5165.json")
state_dict = json.loads(state_path.read_text())

cost_matrix = load_cost_matrix("xpu-rt/data/profiled/qnn_cost_matrix.json")

# Reconstruct per-(workload, backend) per_op_sum_us from the cost matrix.
def per_op_sum(workload, backend):
    total = 0.0
    for op, lanes in cost_matrix.get(workload, {}).items():
        if isinstance(lanes, dict) and backend in lanes:
            total += float(lanes[backend])
    return total

workload = board["workload_id"]
predicted = state_dict.get("current_predicted_makespan_us") or 0.0
for backend, mean_us in board["per_backend_mean_us"].items():
    measurement = {
        "workload_id":   workload,
        "backend":       backend,
        "measured_us":   float(mean_us),
        "per_op_sum_us": per_op_sum(workload, backend),
        "predicted_us":  float(predicted),
    }
    out = xpu_rt_apply_measurement(
        sm=None,                       # SessionManager is unused.
        loop_state_dict=state_dict,
        measurement_dict=measurement,
        persist=False,                 # we persist once at the end.
    )
    state_dict = out["state"]

# Persist the EMA-updated state once.
save_loop_state(state_from_dict(state_dict), state_path)
print(f"[apply] updated overheads: {state_dict['current_calibration']['overhead_us']}")
PY
```

After this, calling `xpu_rt_feedback_step` again will re-solve under the
updated calibration and produce the next round's plan.

## Automated path via MCP tools

Once the plan is on the board, the host side can also drive the loop
via MCP tools instead of inline python:

1. Agent calls
   ``xpu_rt_emit_board_plan(loop_state_dict=<state>)`` — returns a
   ``loop_plan_board_v1`` JSON dict. The caller writes it to disk
   (e.g. ``build/loops/yolov8n__qrb5165__plan.json``) and copies it to
   the board.
2. The user (or an SCP/SSH-capable harness) runs
   ``run_loop_plan_on_qrb5165.sh`` on the board and copies the
   resulting ``measurement.json`` back to the host.
3. Agent calls
   ``xpu_rt_ingest_board_measurement(measurement_json_path=...,
   loop_state_dict=<state>, cost_matrix_path=...)`` to validate the
   measurement record, build per-``(workload, backend)``
   :class:`MeasurementRecord`s, and absorb them via
   :func:`xpu_rt_apply_measurement`. The returned ``state`` is the
   EMA-updated loop state.
4. For unattended runs, ``xpu_rt_run_board_loop_step(...,
   wait_for_measurement=True)`` polls
   ``build/loops/measurements/<workload>__<target>__iter<NNN>.json``
   at ``measurement_poll_interval_s`` until it appears or
   ``measurement_max_wait_s`` elapses; on success it ingests
   automatically. The default (``wait_for_measurement=False``) returns
   ``status='awaiting_measurement'`` plus the plan dict so the agent
   can hand off to a human runner.

All three tools live in
``xpu_rt.mcp.tools.board_runner`` and appear under
``xpu_rt.mcp.tools.get_all_tools()``. They never ssh into the board on
their own — that step remains the operator's explicit responsibility.

## File map

| File | Side | Purpose |
|---|---|---|
| `emit_loop_plan.py` | host | LoopState → board plan JSON |
| `run_loop_plan_on_qrb5165.sh` | board | plan JSON → measurement JSON via `qnn-net-run` |
| `README_board_loop.md` | host | this how-to |

## Verification (host-only, no board needed)

```bash
bash scripts/board/run_loop_plan_on_qrb5165.sh --help
uv run python scripts/board/emit_loop_plan.py --help
```

Both must print usage and exit 0. The bash script's `qnn-net-run`
invocations are guarded behind partition execution, so `--help` does not
attempt to reach the board.
