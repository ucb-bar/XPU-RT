"""Exp26 — measurement-driven feedback loop.

When a real on-board measurement exists for a candidate schedule, the
loop short-circuits the calibration-driven predictor and acts on the
measurement directly. This demo:

  1. Imports the realtime_qnn bundle CSVs into the MeasurementCache
     (idempotent — re-running just appends a fresh snapshot).
  2. Runs the loop on yolov8n@DSP with ``measurement_first=True`` and
     ``deployment_mode='warm_loop'``: the predicted makespan must come
     from the cache (the 55.46 ms ground-truth bundle p50).
  3. Runs the loop again with ``measurement_first=False``: predicted
     reverts to the v4 calibration's number.
  4. Asserts the cache-driven prediction matches the bundle's measured
     p50 (55,461 us) within 10 %.

Output: ``build/experiments/exp26_measurement_driven/{summary.md,results.json}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xpu_rt.runtime.calibration import (
    DEPLOYMENT_MODE_WARM,
)
from xpu_rt.runtime.calibration import (
    load as load_calibration,
)
from xpu_rt.runtime.measurement_cache import (
    DEFAULT_CACHE_DIR,
    load_cache,
)
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    LoopConfig,
    init_loop_state,
    step,
)

# Local import so re-running exp26 also refreshes the cache.
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "board"))
from import_realtime_qnn_bundle import import_bundle  # noqa: E402

TARGET_ID = "qrb5165"
CALIBRATION_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp26_measurement_driven"

# Bundle-measured p50s (post-warmup), µs. Cross-reference:
# ``xpu-rt/data/profiled/qnn_warm/measurements.json``.
BUNDLE_YOLO_DSP_P50_US = 55_461.0
BUNDLE_DRONET_GPU_P50_US = 1_668.0


def _run_one(
    *,
    workload: str,
    cost_matrix: dict[str, Any],
    calibration: Any,
    measurement_first: bool,
    cache_dir: Path,
) -> dict[str, Any]:
    cfg = LoopConfig(
        measurement_first=measurement_first,
        measurement_cache_dir=cache_dir,
        deployment_mode=DEPLOYMENT_MODE_WARM,
    )
    state = init_loop_state(
        workload_id=workload,
        target_id=TARGET_ID,
        cost_matrix=cost_matrix,
        calibration=calibration,
        config=cfg,
        memory_dir=OUT_DIR / "memory",
    )
    state = step(
        state, measurement=None,
        cost_matrix=cost_matrix, config=cfg,
        memory_dir=OUT_DIR / "memory",
    )
    record = state.history[-1]
    return {
        "measurement_first": measurement_first,
        "predicted_makespan_us": float(state.current_predicted_makespan_us or 0.0),
        "prediction_source": record.prediction_source,
        "solver_choice": record.solver_choice,
        "n_partitions": record.n_partitions,
    }


def _format_us(us: float) -> str:
    return f"{us / 1000.0:.2f} ms ({us:.0f} us)"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp26] importing realtime_qnn bundle into MeasurementCache ...")
    cache_dir = DEFAULT_CACHE_DIR
    import_bundle(cache_dir)

    cache = load_cache(cache_dir, TARGET_ID)
    cache_entries = cache.all_for(TARGET_ID)
    assert len(cache_entries) >= 2, "expected at least two cache entries"
    keys_present = sorted(
        (e.key.workload_id, e.key.lane) for e in cache_entries
    )
    assert ("yolov8n", "DSP") in keys_present
    assert ("dronet", "GPU") in keys_present

    print("[exp26] loading cost matrix + v4 calibration ...")
    cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    calibration = load_calibration(CALIBRATION_PATH)

    print("[exp26] running loop with measurement_first=True (yolov8n) ...")
    with_cache = _run_one(
        workload="yolov8n",
        cost_matrix=cost_matrix,
        calibration=calibration,
        measurement_first=True,
        cache_dir=cache_dir,
    )
    print(
        f"  predicted={_format_us(with_cache['predicted_makespan_us'])}  "
        f"source={with_cache['prediction_source']}"
    )

    print("[exp26] running loop with measurement_first=False (yolov8n) ...")
    without_cache = _run_one(
        workload="yolov8n",
        cost_matrix=cost_matrix,
        calibration=calibration,
        measurement_first=False,
        cache_dir=cache_dir,
    )
    print(
        f"  predicted={_format_us(without_cache['predicted_makespan_us'])}  "
        f"source={without_cache['prediction_source']}"
    )

    delta = with_cache["predicted_makespan_us"] - BUNDLE_YOLO_DSP_P50_US
    pct = abs(delta) / BUNDLE_YOLO_DSP_P50_US * 100.0
    within_10pct = pct < 10.0

    headline = {
        "workload": "yolov8n",
        "lane": "DSP",
        "bundle_measured_p50_us": BUNDLE_YOLO_DSP_P50_US,
        "predictor_only_us": without_cache["predicted_makespan_us"],
        "measurement_cache_us": with_cache["predicted_makespan_us"],
        "delta_us": delta,
        "pct_diff_vs_bundle": pct,
        "within_10pct": within_10pct,
    }
    results = {
        "target_id": TARGET_ID,
        "cache_entries": [
            {
                "workload": e.key.workload_id,
                "lane": e.key.lane,
                "techniques": list(e.key.deployment_techniques),
                "p50_us": e.stats.p50_us,
                "n_iters": e.stats.n_iters,
                "source": e.stats.source,
            }
            for e in cache_entries
        ],
        "with_cache": with_cache,
        "without_cache": without_cache,
        "headline": headline,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    summary = "\n".join([
        "# Exp26 — measurement-driven feedback loop",
        "",
        "Loop run on `yolov8n@DSP`, `deployment_mode='warm_loop'`.",
        "Bundle ground-truth p50 (post-warmup):",
        f"  - yolov8n@DSP : {BUNDLE_YOLO_DSP_P50_US/1000:.2f} ms",
        f"  - dronet@GPU  : {BUNDLE_DRONET_GPU_P50_US/1000:.2f} ms",
        "",
        "## Headline",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| predictor-only (model-based) | {_format_us(without_cache['predicted_makespan_us'])} |",
        f"| measurement-cache short-circuit | {_format_us(with_cache['predicted_makespan_us'])} |",
        f"| bundle measured p50 | {_format_us(BUNDLE_YOLO_DSP_P50_US)} |",
        f"| delta (cache - bundle) | {delta:.0f} us |",
        f"| pct diff (cache vs bundle) | {pct:.3f} % |",
        f"| within 10 % of bundle | {'YES' if within_10pct else 'NO'} |",
        "",
        "## Cache contents",
        "",
    ])
    rows = "\n".join(
        f"  - ({e.key.workload_id}, {e.key.lane}) "
        f"p50={e.stats.p50_us:.1f}us  n_iters={e.stats.n_iters}  "
        f"source={e.stats.source}"
        for e in cache_entries
    )
    summary = summary + rows + "\n"
    (OUT_DIR / "summary.md").write_text(summary)

    print()
    print("=" * 60)
    print(f"  predictor-only        : {_format_us(without_cache['predicted_makespan_us'])}")
    print(f"  measurement-cache     : {_format_us(with_cache['predicted_makespan_us'])}")
    print(f"  bundle measured p50   : {_format_us(BUNDLE_YOLO_DSP_P50_US)}")
    print(f"  delta vs bundle       : {delta:+.0f} us  ({pct:.3f} %)")
    print(f"  within 10 % of bundle : {'YES' if within_10pct else 'NO'}")
    print("=" * 60)
    assert within_10pct, (
        f"cache short-circuit deviates from bundle p50 by {pct:.2f} % "
        f"(predicted={with_cache['predicted_makespan_us']:.0f} us, "
        f"bundle={BUNDLE_YOLO_DSP_P50_US:.0f} us)"
    )
    print(f"[exp26] artifacts → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
