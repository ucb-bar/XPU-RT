"""End-to-end demo of the board-runner MCP tools.

Drives the host-side half of the QRB5165 measurement round-trip without
needing a real board:

1. Build (or reload) a :class:`LoopState` for ``yolov8n`` on ``qrb5165``.
2. Call ``xpu_rt_emit_board_plan`` and write ``plan.json``.
3. Print the user-visible instructions for the real round-trip.
4. Simulate a board run by writing a synthetic ``measurement.json``
   and feeding it back through ``xpu_rt_ingest_board_measurement``.
5. Persist the resumed loop state and a short ``summary.md``.

Outputs (under ``build/experiments/exp24_board_mcp/``):
  - ``plan.json``               — board-runnable loop_plan_board_v1.
  - ``fake_measurement.json``    — synthetic measurement_record_board_v1.
  - ``resumed_state.json``       — loop state after ingest.
  - ``summary.md``               — narrative report.

Usage:
    uv run python scripts/experiments/exp24_board_mcp_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp24_board_mcp"
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
LOOP_STATE_PATH = REPO_ROOT / "build" / "loops" / "yolov8n__qrb5165.json"

WORKLOAD = "yolov8n"
TARGET = "qrb5165"


def _make_or_load_loop_state() -> dict[str, Any]:
    """Reload the persisted loop state if it exists; else bootstrap one."""

    from xpu_rt.runtime.calibration import (
        CALIBRATION_SCHEMA_VERSION,
        CalibrationModel,
    )
    from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
    from xpu_rt.scheduling.feedback_loop import init_loop_state, state_to_dict

    if LOOP_STATE_PATH.is_file():
        return json.loads(LOOP_STATE_PATH.read_text(encoding="utf-8"))

    cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    calibration = CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id=TARGET,
        overhead_us={WORKLOAD: {"CPU": 1000.0, "GPU": 500.0, "DSP": 800.0}},
        contention_factor={WORKLOAD: {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0}},
        history=(),
        created_at="2026-05-15T00:00:00+00:00",
    )
    state = init_loop_state(
        workload_id=WORKLOAD,
        target_id=TARGET,
        cost_matrix=cost_matrix,
        calibration=calibration,
    )
    return state_to_dict(state)


def _synthesize_fake_measurement(plan: dict[str, Any]) -> dict[str, Any]:
    """Build a plausible-looking measurement matching the plan."""

    per_backend_mean_us = {"DSP": 245_000.0, "CPU": 380_000.0, "GPU": 290_000.0}
    raw = []
    for part in plan["partitions"]:
        backend = part["backend"]
        raw.append(
            {
                "partition_id": part["partition_id"],
                "backend": backend,
                "mean_us": per_backend_mean_us.get(backend, 250_000.0),
                "iters": part["iters"],
                "ok": True,
                "error": "",
            }
        )
    return {
        "schema_version": "measurement_record_board_v1",
        "target_id": plan["target_id"],
        "workload_id": plan["workload_id"],
        "captured_at": "2026-05-15T00:00:00+00:00",
        "iters": plan["iters"],
        "per_backend_mean_us": per_backend_mean_us,
        "raw_per_partition_us": raw,
    }


def _user_instructions(plan_path: Path, measurement_path: Path) -> str:
    return f"""\
To run on a real QRB5165 board:

  scp scripts/board/run_loop_plan_on_qrb5165.sh root@<board>:/data/local/tmp/
  scp {plan_path.relative_to(REPO_ROOT)} root@<board>:/data/local/tmp/plan.json
  ssh root@<board> 'bash /data/local/tmp/run_loop_plan_on_qrb5165.sh \\
      --plan-json   /data/local/tmp/plan.json \\
      --output-json /data/local/tmp/measurement.json'
  scp root@<board>:/data/local/tmp/measurement.json {measurement_path.relative_to(REPO_ROOT)}

Then on the host, resume the loop:

  uv run python -c "
  import json
  from xpu_rt.mcp.tools.board_runner import xpu_rt_ingest_board_measurement
  state = json.loads(open('build/loops/yolov8n__qrb5165.json').read())
  out = xpu_rt_ingest_board_measurement(
      measurement_json_path='{measurement_path.relative_to(REPO_ROOT)}',
      loop_state_dict=state,
      cost_matrix_path='xpu-rt/data/profiled/qnn_cost_matrix.json',
  )
  json.dump(out['state'], open('build/loops/yolov8n__qrb5165.json', 'w'))
  "
"""


def main() -> int:
    from xpu_rt.mcp.tools.board_runner import (
        xpu_rt_emit_board_plan,
        xpu_rt_ingest_board_measurement,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[exp24] using out_dir={OUT_DIR}")
    state = _make_or_load_loop_state()
    print(
        f"[exp24] loop state: workload_id={state['workload_id']} "
        f"target_id={state['target_id']} iteration={state['iteration']} "
        f"n_chunks={len(state['current_chunks'])}"
    )

    # 1. Emit board plan.
    plan = xpu_rt_emit_board_plan(loop_state_dict=state, iters=10)
    plan_path = OUT_DIR / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[exp24] wrote plan: {plan_path}")
    print(f"[exp24] plan has {len(plan['partitions'])} partitions")

    # 2. Print user instructions for the real round-trip.
    measurement_path = OUT_DIR / "fake_measurement.json"
    print("\n" + _user_instructions(plan_path, measurement_path))

    # 3. Simulate the board side: write a synthetic measurement.json.
    fake_measurement = _synthesize_fake_measurement(plan)
    measurement_path.write_text(
        json.dumps(fake_measurement, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[exp24] wrote synthetic measurement: {measurement_path}")

    # 4. Ingest the synthetic measurement and resume the loop.
    ingested = xpu_rt_ingest_board_measurement(
        measurement_json_path=str(measurement_path),
        loop_state_dict=state,
        cost_matrix_path=str(COST_MATRIX_PATH),
        persist=False,
    )
    resumed_path = OUT_DIR / "resumed_state.json"
    resumed_path.write_text(
        json.dumps(ingested["state"], indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[exp24] ingested: ok={ingested['ok']} n_applied={ingested['n_applied']} "
        f"-> {resumed_path}"
    )

    new_overheads = (
        ingested["state"]["current_calibration"]["overhead_us"].get(WORKLOAD, {})
    )

    # 5. Summary.
    summary_lines = [
        "# exp24 — board-runner MCP demo",
        "",
        f"- workload: `{state['workload_id']}`",
        f"- target: `{state['target_id']}`",
        f"- starting iteration: `{state['iteration']}`",
        f"- chunks in plan: `{len(plan['partitions'])}`",
        f"- backends touched by synthetic measurement: "
        f"`{sorted(fake_measurement['per_backend_mean_us'])}`",
        f"- ingested n_applied: `{ingested['n_applied']}`",
        "",
        "## Updated overheads (post-ingest)",
        "",
        "| backend | overhead_us |",
        "|---|---|",
        *[f"| {b} | {v:.1f} |" for b, v in sorted(new_overheads.items())],
        "",
        "## Files",
        "",
        f"- `{plan_path.relative_to(REPO_ROOT)}`",
        f"- `{measurement_path.relative_to(REPO_ROOT)}`",
        f"- `{resumed_path.relative_to(REPO_ROOT)}`",
        "",
    ]
    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[exp24] wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
