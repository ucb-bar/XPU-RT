"""Stage-4 acceptance: end-to-end feedback loop on the closed-loop ground truth.

Loads the seeded calibration model and the QNN cost matrix, builds a
:class:`LoopState` for ``yolov8n`` on ``qrb5165``, and steps the loop
using the four whole-network DSP rounds from
``xpu-rt/data/profiled/qnn_closed_loop/final_report.md`` as the
ground-truth measurements.

The acceptance criteria are intentionally lenient (the seeded calibration
model only carries one overhead constant per backend; closing the gap
fully needs per-(workload, backend) terms that are out of scope here):

* PASS if the loop reaches ``status == "converged"`` within
  ``--max-iterations`` iterations, OR
* PASS if at least one of the last 3 iterations reports ``|prediction
  error| < 0.10``.

Otherwise the script exits non-zero with a diagnostic so CI catches a
real regression in the loop.

Usage:
    uv run python scripts/experiments/exp14_feedback_loop_e2e.py
    uv run python scripts/experiments/exp14_feedback_loop_e2e.py --max-iterations 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Whole-network DSP rounds from final_report.md (predicted_ms, measured_ms).
# Predicted is the closed-loop scheduler's number from that run; we use
# the measured column as ground truth here.
ROUNDS_MS: tuple[tuple[int, float, float], ...] = (
    (1, 354.9, 254.8),
    (2, 304.8, 350.9),
    (3, 356.7, 255.6),
    (4, 305.5, 257.3),
)


def main(argv: list[str] | None = None) -> int:
    from xpu_rt.runtime.calibration import (
        MeasurementRecord,
    )
    from xpu_rt.runtime.calibration import (
        load as load_calibration,
    )
    from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
    from xpu_rt.scheduling.feedback_loop import (
        LoopConfig,
        init_loop_state,
        step,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum number of feedback-loop iterations (default: 8).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="Convergence band for |pred-meas|/meas (default: 0.10).",
    )
    args = parser.parse_args(argv)

    cal_path = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
    cost_matrix_path = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"

    cal = load_calibration(cal_path)
    cost_matrix = load_cost_matrix(cost_matrix_path)

    cfg = LoopConfig(
        epsilon=args.epsilon,
        consecutive_required=2,
        max_iterations=args.max_iterations,
        outlier_threshold=2.0,
        transfer_dominance_threshold=0.30,
        max_chunk_ops=16,
        max_partitions=200,
    )

    print("=" * 78)
    print("Stage 4 acceptance harness — feedback loop on yolov8n / DSP / qrb5165")
    print("=" * 78)
    print(f"calibration overhead_us: {dict(cal.overhead_us)}")
    print(f"max iterations: {cfg.max_iterations}, epsilon: {cfg.epsilon}")
    print()
    print("NOTE: The closed-loop ground truth measured the DSP-only chain "
          "(whole-net DLC running on DSP alone), so we compare the calibrated "
          "DSP chain-sum + overhead against the measurement, NOT the CP-SAT "
          "multi-lane makespan (which would parallelise across CPU/GPU/DSP "
          "and isn't directly comparable to a DSP-only run).")
    print()

    state = init_loop_state(
        workload_id="yolov8n",
        target_id="qrb5165",
        cost_matrix=cost_matrix,
        calibration=cal,
        config=cfg,
    )

    # Use the four closed-loop measurements as ground truth, cycling if
    # the loop wants more iterations than we have rounds for. The
    # synthesized per_op_sum_us is the chain-sum of DSP per-op costs
    # (the same chain the closed-loop scheduler used for whole_net).
    per_op_sum_us = sum(
        float(c.get("DSP", 0.0))
        for c in cost_matrix["yolov8n"].values()
        if isinstance(c, dict)
    )
    print(f"per_op_sum_us (DSP, yolov8n): {per_op_sum_us:.1f}")
    print()

    print(f"{'iter':>4} {'dsp_pred_us':>12} {'meas_us':>10} {'err%':>7} "
          f"{'decision':>20} {'status':>10}")
    print("-" * 78)

    def dsp_chain_prediction_us(s) -> float:
        """Calibrated DSP chain-sum + per-backend overhead constant."""

        return per_op_sum_us + float(s.current_calibration.overhead_us.get("DSP", 0.0))

    last_3_errors: list[float] = []
    rounds = list(ROUNDS_MS)
    for i in range(cfg.max_iterations):
        # Iteration 0: plan-only step (no measurement available yet).
        if i == 0:
            new_state = step(state, None, cost_matrix=cost_matrix, config=cfg)
            dsp_pred = dsp_chain_prediction_us(new_state)
            print(f"{new_state.iteration:>4} {dsp_pred:>12.1f} {'—':>10} "
                  f"{'—':>7} {'(plan-only)':>20} {new_state.status:>10}")
            state = new_state
            continue

        round_idx = (i - 1) % len(rounds)
        _r, _pred_ms, measured_ms = rounds[round_idx]
        dsp_pred_now = dsp_chain_prediction_us(state)
        m = MeasurementRecord(
            workload_id="yolov8n",
            backend="DSP",
            measured_us=measured_ms * 1000.0,
            per_op_sum_us=per_op_sum_us,
            predicted_us=dsp_pred_now,
        )
        new_state = step(state, m, cost_matrix=cost_matrix, config=cfg)
        last_round = new_state.history[-1]
        # Re-compute the DSP-comparable prediction *after* the calibration
        # update, so the printed error reflects the post-EMA model.
        dsp_pred = dsp_chain_prediction_us(new_state)
        meas = last_round.measured_makespan_us or 0.0
        err = abs(dsp_pred - meas) / meas if meas > 0 else float("inf")
        last_3_errors.append(err)
        last_3_errors = last_3_errors[-3:]
        print(f"{new_state.iteration:>4} {dsp_pred:>12.1f} {meas:>10.1f} "
              f"{err * 100:>6.1f}% {last_round.decision_next:>20} "
              f"{new_state.status:>10}")
        state = new_state
        if state.status == "converged":
            break

    print("-" * 78)
    print(f"final status: {state.status}")
    print(f"final iteration: {state.iteration}")
    print(f"last-3 errors: {[f'{e * 100:.1f}%' for e in last_3_errors]}")

    converged = state.status == "converged"
    in_band_in_last_3 = any(e < 0.10 for e in last_3_errors)
    if converged:
        print(f"\nPASS — loop converged at iteration {state.iteration}")
        return 0
    if in_band_in_last_3:
        print("\nPASS — did not converge but at least one of last 3 errors < 10%")
        return 0
    print("\nFAIL — loop did not converge and last-3 errors all >= 10%")
    print("       (calibration model likely needs per-(workload, backend) terms)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
