"""Exp25 — warm-loop deployment-mode bridge.

Demonstrates that the v4 calibration model produces deployment-mode-aware
predictions: a single (workload, target) configuration now yields one
number for the cold-start regime (qnn-net-run, init included) and a
*different* number for the warm-loop regime (cached context binaries,
pre-allocated tensor buffers, SCHED_FIFO+mlockall — see
``realtime_qnn/REPLICATION.md``).

The headline check: when warm-mode prediction is in the same band as
the realtime_qnn bundle's measured p50 (55.5ms for yolov8n@DSP,
1.67ms for dronet@GPU), the feedback loop is no longer biased by the
~295ms graph-init overhead that dominates cold-start measurements.

Outputs under ``build/experiments/exp25_warm_loop/``:
  * ``summary.md`` — narrative side-by-side comparison.
  * ``results.json`` — typed dict with cold/warm predictions, lane
    breakdowns, and the bundle's measured p50 for cross-reference.
  * ``../../xpu-rt/data/calibration/qrb5165.json`` — re-bootstrapped v4
    calibration (cold + warm overhead) committed alongside.

Usage:
    uv run python scripts/experiments/exp25_warm_loop_bridge.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from xpu_rt.runtime.calibration import (
    DEPLOYMENT_MODE_COLD,
    DEPLOYMENT_MODE_WARM,
    TECHNIQUE_CACHED_CONTEXT,
    TECHNIQUE_FULL_BUFFER_REWRITE,
    TECHNIQUE_NO_FILE_IO,
    TECHNIQUE_PER_SENSOR_ROTATION,
    TECHNIQUE_PREALLOC_BUFFERS,
    TECHNIQUE_SCHED_FIFO,
    TECHNIQUE_TIMERFD_ABSTIME,
    bootstrap_contention_from_closed_loop,
    bootstrap_from_solo_measurements,
    bootstrap_warm_from_csv_traces,
    save,
)
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    LoopConfig,
    init_loop_state,
    step,
)

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
E2E_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_e2e" / "measurements.json"
RT_YOLO_CSV = REPO_ROOT / "realtime_qnn" / "rt_yolo_f.csv"
RT_DRONE_CSV = REPO_ROOT / "realtime_qnn" / "rt_drone_f.csv"
CALIBRATION_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp25_warm_loop"

TARGET_ID = "qrb5165"

WARM_TECHNIQUES_DEMO: tuple[str, ...] = (
    TECHNIQUE_CACHED_CONTEXT,
    TECHNIQUE_PREALLOC_BUFFERS,
    TECHNIQUE_NO_FILE_IO,
    TECHNIQUE_SCHED_FIFO,
    TECHNIQUE_TIMERFD_ABSTIME,
    TECHNIQUE_PER_SENSOR_ROTATION,
    TECHNIQUE_FULL_BUFFER_REWRITE,
)

# Hand-encoded yolov8n DSP closed-loop rounds (per-iter measured),
# matching the test fixture in xpu-rt/tests/runtime/test_calibration.py.
CLOSED_LOOP_ROUNDS: list[dict[str, Any]] = [
    {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0},
    {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 350900.0},
    {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 255600.0},
    {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 257300.0},
]


@dataclass(frozen=True)
class ModeRunOutcome:
    """One step() outcome captured for the side-by-side."""

    workload: str
    deployment_mode: str
    predicted_us: float
    baseline_us: float
    solver_choice: str
    n_partitions: int
    per_lane_overhead_us: dict[str, float]


def _per_lane_overhead(cal: dict[str, Any], workload: str) -> dict[str, float]:
    return dict(cal.get(workload, {}))


def _run_one(
    workload: str,
    cost_matrix: dict,
    calibration: Any,
    deployment_mode: str,
) -> ModeRunOutcome:
    cfg = LoopConfig(deployment_mode=deployment_mode)
    state = init_loop_state(
        workload_id=workload,
        target_id=TARGET_ID,
        cost_matrix=cost_matrix,
        calibration=calibration,
        config=cfg,
        memory_dir=OUT_DIR / "memory",
    )
    next_state = step(
        state,
        measurement=None,
        cost_matrix=cost_matrix,
        config=cfg,
        memory_dir=OUT_DIR / "memory",
    )
    predicted = float(next_state.current_predicted_makespan_us or 0.0)
    baseline = next_state.baseline_makespan_us
    if not math.isfinite(baseline):
        baseline = predicted
    if deployment_mode == DEPLOYMENT_MODE_WARM:
        per_lane_ovh = _per_lane_overhead(calibration.overhead_us_warm, workload)
    else:
        per_lane_ovh = _per_lane_overhead(calibration.overhead_us, workload)
    return ModeRunOutcome(
        workload=workload,
        deployment_mode=deployment_mode,
        predicted_us=predicted,
        baseline_us=float(baseline),
        solver_choice=next_state.current_solver_choice,
        n_partitions=len(next_state.current_chunks),
        per_lane_overhead_us=per_lane_ovh,
    )


def _format_us(us: float) -> str:
    if us > 1000.0:
        return f"{us / 1000.0:.2f} ms ({us:.0f} us)"
    return f"{us:.1f} us"


def _write_summary(results: dict[str, Any], path: Path) -> None:
    rt_yolo_p50_ms = results["bundle_measured_p50_ms"]["yolov8n_DSP"]
    rt_drone_p50_ms = results["bundle_measured_p50_ms"]["dronet_GPU"]
    yolo_cold = results["runs"]["yolov8n"]["cold_start"]["predicted_us"] / 1000.0
    yolo_warm = results["runs"]["yolov8n"]["warm_loop"]["predicted_us"] / 1000.0
    dro_cold = results["runs"]["dronet"]["cold_start"]["predicted_us"] / 1000.0
    dro_warm = results["runs"]["dronet"]["warm_loop"]["predicted_us"] / 1000.0
    yolo_band_ok = 50.0 <= yolo_warm <= 70.0
    yolo_cold_ok = 300.0 <= yolo_cold <= 360.0
    yolo_pct = abs(yolo_warm - rt_yolo_p50_ms) / rt_yolo_p50_ms * 100.0
    dro_pct = abs(dro_warm - rt_drone_p50_ms) / rt_drone_p50_ms * 100.0
    lines = [
        "# Exp25 — warm-loop deployment-mode bridge",
        "",
        f"Target: `{TARGET_ID}` | Calibration schema: `calibration_model_v4`",
        "",
        "## Side-by-side: feedback-loop predicted vs bundle measured",
        "",
        "| workload | deployment_mode | predicted (ms) | bundle p50 (ms) | abs err |",
        "|---|---|---:|---:|---:|",
        f"| yolov8n@DSP | cold_start | {yolo_cold:.2f} | — | — |",
        f"| yolov8n@DSP | warm_loop  | {yolo_warm:.2f} | {rt_yolo_p50_ms:.2f} | {yolo_pct:.1f}% |",
        f"| dronet@GPU  | cold_start | {dro_cold:.2f} | — | — |",
        f"| dronet@GPU  | warm_loop  | {dro_warm:.2f} | {rt_drone_p50_ms:.2f} | {dro_pct:.1f}% |",
        "",
        "## Validation",
        "",
        f"- yolov8n warm in [50, 70] ms: **{yolo_band_ok}**",
        f"- yolov8n cold in [300, 360] ms: **{yolo_cold_ok}**",
        f"- yolov8n warm error vs bundle p50: **{yolo_pct:.1f}%** (target ≤ 20%)",
        f"- dronet warm error vs bundle p50: **{dro_pct:.1f}%**",
        "",
        "## Deployment techniques in the warm-mode run",
        "",
    ]
    for t in WARM_TECHNIQUES_DEMO:
        lines.append(f"- `{t}`")
    lines.extend(
        [
            "",
            "## Implication",
            "",
            "Before v4, the calibration model lumped graph-init time (~295ms for",
            "yolov8n@DSP, dominating qnn-net-run measurements) into the same",
            "constant the loop applied to every per-iter prediction. The loop's",
            "regression guard, convergence rule, and decision rubric were all",
            "biased toward cold-start arithmetic. With v4, the loop selects the",
            "right per-(workload, backend) overhead based on the deployment",
            "techniques in effect; in warm-loop mode the regression guard now",
            "compares against an init-free baseline, and convergence can fire",
            "on per-iter ground truth in the 50ms regime instead of permanently",
            "thinking yolov8n needs 300+ ms.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    e2e = json.loads(E2E_PATH.read_text())
    warm_yolo_p50 = 55_461.0
    warm_drone_p50 = 1_668.0

    cal = bootstrap_from_solo_measurements(cost_matrix, e2e, target_id=TARGET_ID)
    cal = bootstrap_warm_from_csv_traces(
        cal, [RT_YOLO_CSV, RT_DRONE_CSV], cost_matrix, drop_warmup=5, aggregator="median"
    )
    cal = bootstrap_contention_from_closed_loop(cal, CLOSED_LOOP_ROUNDS, cost_matrix)
    save(cal, CALIBRATION_PATH)

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for workload in ("yolov8n", "dronet"):
        per_workload: dict[str, dict[str, Any]] = {}
        for mode in (DEPLOYMENT_MODE_COLD, DEPLOYMENT_MODE_WARM):
            outcome = _run_one(workload, cost_matrix, cal, mode)
            per_workload[mode] = {
                "predicted_us": outcome.predicted_us,
                "baseline_us": outcome.baseline_us,
                "solver_choice": outcome.solver_choice,
                "n_partitions": outcome.n_partitions,
                "per_lane_overhead_us": outcome.per_lane_overhead_us,
            }
            log.info(
                "exp25_run",
                workload=workload,
                mode=mode,
                predicted_ms=outcome.predicted_us / 1000.0,
                baseline_ms=outcome.baseline_us / 1000.0,
                per_lane_overhead_us=outcome.per_lane_overhead_us,
            )
        runs[workload] = per_workload

    results = {
        "target_id": TARGET_ID,
        "calibration_schema_version": cal.schema_version,
        "calibration_path": str(CALIBRATION_PATH.relative_to(REPO_ROOT)),
        "warm_techniques_demo": list(WARM_TECHNIQUES_DEMO),
        "bundle_measured_p50_ms": {
            "yolov8n_DSP": warm_yolo_p50 / 1000.0,
            "dronet_GPU": warm_drone_p50 / 1000.0,
        },
        "runs": runs,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    _write_summary(results, OUT_DIR / "summary.md")

    yolo_warm_ms = runs["yolov8n"]["warm_loop"]["predicted_us"] / 1000.0
    yolo_cold_ms = runs["yolov8n"]["cold_start"]["predicted_us"] / 1000.0
    dro_warm_ms = runs["dronet"]["warm_loop"]["predicted_us"] / 1000.0
    log.info(
        "exp25_complete",
        yolo_cold_ms=yolo_cold_ms,
        yolo_warm_ms=yolo_warm_ms,
        yolo_bundle_p50_ms=warm_yolo_p50 / 1000.0,
        dro_warm_ms=dro_warm_ms,
        dro_bundle_p50_ms=warm_drone_p50 / 1000.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
