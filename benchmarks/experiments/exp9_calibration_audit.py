"""Experiment 9 — Calibration audit of the closed-loop QNN scheduler.

Compares predicted vs measured makespans from the four-round closed-loop
proof (``xpu-rt/data/profiled/qnn_closed_loop/``), checks contention
factor convergence, cross-validates the per-op cost matrix against
whole-network E2E measurements, and produces a chain-sum prediction
baseline so contention-modeling error can be separated from per-op cost
calibration error.

Outputs are written to ``build/experiments/exp9_calibration/``:

* ``report.md``     -- single-page audit report
* ``summary.json``  -- machine-readable metrics
* ``contention_convergence.png`` (optional; only if matplotlib is available)

Usage:
    uv run python scripts/experiments/exp9_calibration_audit.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "xpu-rt" / "data" / "profiled"
CL_DIR = DATA_DIR / "qnn_closed_loop"
E2E_PATH = DATA_DIR / "qnn_e2e" / "measurements.json"
COST_MATRIX_PATH = DATA_DIR / "qnn_cost_matrix.json"
CALIBRATION_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp9_calibration"

# The four closed-loop rounds ran whole_net on the DSP lane (final_report.md
# per-iter DSP wall times match ROUNDS' measured_ms column to 0.1ms). When
# evaluating the calibration model we therefore use DSP overhead.
CLOSED_LOOP_WORKLOAD = "yolov8n"
CLOSED_LOOP_BACKEND = "DSP"

# Whole-network rounds copied verbatim from final_report.md (predicted ms / measured ms).
# Each entry: (round, predicted_ms, measured_ms). All rounds are granularity=whole_net.
ROUNDS: tuple[tuple[int, float, float], ...] = (
    (1, 354.9, 254.8),
    (2, 304.8, 350.9),
    (3, 356.7, 255.6),
    (4, 305.5, 257.3),
)

CONVERGENCE_THRESHOLD = 0.05  # |delta| < 0.05 for all factors == converged


@dataclass(frozen=True)
class RoundError:
    round: int
    pred_ms: float
    measured_ms: float
    error_ms: float       # pred - measured (signed)
    abs_error_ms: float
    error_pct: float      # (pred - measured) / measured * 100
    abs_error_pct: float
    direction: str        # "pessimistic" if pred>measured, "optimistic" otherwise


def compute_round_errors() -> list[RoundError]:
    out: list[RoundError] = []
    for r, pred, meas in ROUNDS:
        err = pred - meas
        pct = err / meas * 100.0
        out.append(
            RoundError(
                round=r,
                pred_ms=pred,
                measured_ms=meas,
                error_ms=err,
                abs_error_ms=abs(err),
                error_pct=pct,
                abs_error_pct=abs(pct),
                direction="pessimistic" if err > 0 else "optimistic",
            )
        )
    return out


def aggregate(errors: list[RoundError]) -> dict[str, float]:
    abs_pct = [e.abs_error_pct for e in errors]
    signed = [e.error_pct for e in errors]
    signed_ms = [e.error_ms for e in errors]
    return {
        "mean_abs_pct": statistics.fmean(abs_pct),
        "median_abs_pct": statistics.median(abs_pct),
        "max_abs_pct": max(abs_pct),
        "min_abs_pct": min(abs_pct),
        "bias_mean_signed_pct": statistics.fmean(signed),
        "bias_mean_signed_ms": statistics.fmean(signed_ms),
        "std_signed_pct": statistics.pstdev(signed) if len(signed) > 1 else 0.0,
        "std_signed_ms": statistics.pstdev(signed_ms) if len(signed_ms) > 1 else 0.0,
        "n_pessimistic": sum(1 for e in errors if e.direction == "pessimistic"),
        "n_optimistic": sum(1 for e in errors if e.direction == "optimistic"),
    }


def load_contention() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (CL_DIR / "contention.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convergence_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = rows[-1]
    deltas = final["last_delta"]
    converged = all(abs(v) < CONVERGENCE_THRESHOLD for v in deltas.values())
    return {
        "final_round": final["round"],
        "final_factors": final["factors"],
        "final_last_delta": deltas,
        "threshold": CONVERGENCE_THRESHOLD,
        "converged_per_threshold": converged,
        "stamp_in_report": "round 4: converged ✅",
        "stamp_matches_threshold": converged,
    }


def sum_per_op(cost_matrix: dict[str, Any], workload: str, backend: str) -> tuple[float, int, int]:
    """Return (sum_ms, op_count_with_cost, op_count_missing)."""
    ops = cost_matrix.get(workload, {})
    total_us = 0.0
    covered = 0
    missing = 0
    for op_name, costs in ops.items():
        if not isinstance(costs, dict):
            continue
        if backend in costs:
            total_us += float(costs[backend])
            covered += 1
        else:
            missing += 1
    return total_us / 1000.0, covered, missing


def sum_of_parts(cost_matrix: dict[str, Any], e2e: dict[str, Any]) -> dict[str, Any]:
    per_op: dict[str, dict[str, Any]] = {}
    for workload in ("yolov8n", "dronet"):
        per_op[workload] = {}
        wl_e2e = e2e["matrix"].get(workload, {})
        for backend in ("CPU", "GPU", "DSP"):
            sum_ms, covered, missing = sum_per_op(cost_matrix, workload, backend)
            be = wl_e2e.get(backend) or {}
            e2e_ms = float(be.get("total_ms", 0.0)) / 10.0 if be.get("total_ms") else None
            # qnn_e2e total_ms is across 10 iters in measurements.json; per-iter = total_ms/10
            ratio = (sum_ms / e2e_ms) if e2e_ms else None
            disagreement = ((sum_ms - e2e_ms) / e2e_ms * 100.0) if e2e_ms else None
            per_op[workload][backend] = {
                "sum_per_op_ms": round(sum_ms, 3),
                "covered_ops": covered,
                "missing_ops": missing,
                "e2e_per_iter_ms": round(e2e_ms, 3) if e2e_ms else None,
                "ratio_sum_over_e2e": round(ratio, 3) if ratio else None,
                "disagreement_pct": round(disagreement, 2) if disagreement is not None else None,
            }
    return per_op


def chain_sum_predict(cost_matrix: dict[str, Any]) -> dict[str, Any]:
    """Predicted solo-makespan = Σ per-op cost on chosen backend (no transfer, no contention)."""
    preds: dict[str, dict[str, Any]] = {}
    for workload in ("yolov8n", "dronet"):
        preds[workload] = {}
        for backend in ("CPU", "GPU", "DSP"):
            sum_ms, _, _ = sum_per_op(cost_matrix, workload, backend)
            preds[workload][backend] = round(sum_ms, 3)
    return preds


def chain_sum_vs_e2e(chain: dict[str, Any], e2e: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for workload in ("yolov8n", "dronet"):
        wl_e2e = e2e["matrix"].get(workload, {})
        for backend in ("CPU", "GPU", "DSP"):
            pred_ms = chain[workload][backend]
            be = wl_e2e.get(backend) or {}
            e2e_total = float(be.get("total_ms", 0.0))
            e2e_per_iter = e2e_total / 10.0 if e2e_total else None
            if e2e_per_iter:
                err = pred_ms - e2e_per_iter
                pct = err / e2e_per_iter * 100.0
            else:
                err = None
                pct = None
            rows.append(
                {
                    "workload": workload,
                    "backend": backend,
                    "chain_sum_pred_ms": pred_ms,
                    "e2e_measured_per_iter_ms": round(e2e_per_iter, 3) if e2e_per_iter else None,
                    "error_ms": round(err, 3) if err is not None else None,
                    "error_pct": round(pct, 2) if pct is not None else None,
                }
            )
    abs_pcts = [abs(r["error_pct"]) for r in rows if r["error_pct"] is not None]
    return {
        "rows": rows,
        "mean_abs_pct": statistics.fmean(abs_pcts) if abs_pcts else None,
        "median_abs_pct": statistics.median(abs_pcts) if abs_pcts else None,
    }


def maybe_plot_convergence(rows: list[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    rounds = [r["round"] for r in rows]
    cpu = [r["factors"]["CPU"] for r in rows]
    dsp = [r["factors"]["DSP"] for r in rows]
    delta_cpu = [abs(r["last_delta"]["CPU"]) for r in rows]
    delta_dsp = [abs(r["last_delta"]["DSP"]) for r in rows]
    fig, (ax_f, ax_d) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax_f.plot(rounds, cpu, "o-", label="CPU factor")
    ax_f.plot(rounds, dsp, "s-", label="DSP factor")
    ax_f.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax_f.set_xlabel("round")
    ax_f.set_ylabel("contention factor")
    ax_f.set_title("contention factors")
    ax_f.legend()
    ax_f.grid(alpha=0.3)
    ax_d.semilogy(rounds, delta_cpu, "o-", label="|delta CPU|")
    ax_d.semilogy(rounds, delta_dsp, "s-", label="|delta DSP|")
    ax_d.axhline(CONVERGENCE_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
                 label=f"threshold={CONVERGENCE_THRESHOLD}")
    ax_d.set_xlabel("round")
    ax_d.set_ylabel("|last_delta|")
    ax_d.set_title("convergence (log scale)")
    ax_d.legend()
    ax_d.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "contention_convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fmt_round_table(errors: list[RoundError]) -> str:
    lines = [
        "| round | pred (ms) | measured (ms) | error (ms) | error (%) | direction |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for e in errors:
        lines.append(
            f"| {e.round} | {e.pred_ms:.1f} | {e.measured_ms:.1f} | {e.error_ms:+.1f} | "
            f"{e.error_pct:+.2f}% | {e.direction} |"
        )
    return "\n".join(lines)


def fmt_contention_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| round | CPU factor | DSP factor | |Δ CPU| | |Δ DSP| | converged? |",
        "|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in rows:
        f = r["factors"]
        d = r["last_delta"]
        conv = all(abs(v) < CONVERGENCE_THRESHOLD for v in d.values())
        lines.append(
            f"| {r['round']} | {f['CPU']:.4f} | {f['DSP']:.4f} | "
            f"{abs(d['CPU']):.4f} | {abs(d['DSP']):.4f} | {'✅' if conv else '…'} |"
        )
    return "\n".join(lines)


def fmt_sum_of_parts(sop: dict[str, Any]) -> str:
    lines = [
        "| workload | backend | Σ per-op (ms) | E2E per-iter (ms) | ratio | disagreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for wl in ("yolov8n", "dronet"):
        for be in ("CPU", "GPU", "DSP"):
            row = sop[wl][be]
            lines.append(
                f"| {wl} | {be} | {row['sum_per_op_ms']:.2f} | "
                f"{row['e2e_per_iter_ms']:.2f} | {row['ratio_sum_over_e2e']:.3f} | "
                f"{row['disagreement_pct']:+.1f}% |"
            )
    return "\n".join(lines)


def fmt_chain_vs_e2e(payload: dict[str, Any]) -> str:
    lines = [
        "| workload | backend | chain-sum pred (ms) | E2E per-iter (ms) | error (ms) | error (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['workload']} | {row['backend']} | {row['chain_sum_pred_ms']:.2f} | "
            f"{row['e2e_measured_per_iter_ms']:.2f} | {row['error_ms']:+.2f} | "
            f"{row['error_pct']:+.2f}% |"
        )
    return "\n".join(lines)


def build_report(
    errors: list[RoundError],
    agg: dict[str, float],
    contention_rows: list[dict[str, Any]],
    conv: dict[str, Any],
    sop: dict[str, Any],
    chain: dict[str, Any],
    chain_eval: dict[str, Any],
    plot_path: Path | None,
) -> str:
    direction_verdict = (
        "systematically pessimistic"
        if agg["bias_mean_signed_pct"] > 5.0
        else "systematically optimistic"
        if agg["bias_mean_signed_pct"] < -5.0
        else "roughly balanced (no strong bias)"
    )
    convergence_line = (
        "✅ all |Δ| < 0.05 at round 4"
        if conv["converged_per_threshold"]
        else "❌ round-4 stamp incorrect — at least one |Δ| ≥ 0.05"
    )

    # Sum-of-parts agreement summary for yolov8n (the closed-loop workload).
    y_cpu = sop["yolov8n"]["CPU"]
    y_gpu = sop["yolov8n"]["GPU"]
    y_dsp = sop["yolov8n"]["DSP"]

    return f"""# Exp 9 — Closed-loop scheduler calibration audit

_Source: `xpu-rt/data/profiled/qnn_closed_loop/final_report.md` + companion JSON._

## 1. Per-round error table

{fmt_round_table(errors)}

## 2. Aggregate metrics (4 rounds, whole_net)

| metric | value |
|---|---:|
| mean abs % error | **{agg['mean_abs_pct']:.2f}%** |
| median abs % error | {agg['median_abs_pct']:.2f}% |
| max abs % error | {agg['max_abs_pct']:.2f}% |
| min abs % error | {agg['min_abs_pct']:.2f}% |
| bias (mean signed %) | **{agg['bias_mean_signed_pct']:+.2f}%** |
| bias (mean signed ms) | {agg['bias_mean_signed_ms']:+.2f} ms |
| std of signed % | {agg['std_signed_pct']:.2f}% |
| std of signed ms | {agg['std_signed_ms']:.2f} ms |
| pessimistic rounds | {agg['n_pessimistic']} / {agg['n_pessimistic']+agg['n_optimistic']:.0f} |
| direction verdict | **{direction_verdict}** |

## 3. Contention convergence

{fmt_contention_table(contention_rows)}

Threshold: |Δ| < {CONVERGENCE_THRESHOLD} for every tracked factor.

**Convergence verdict:** {convergence_line}
(Final round = {conv['final_round']}; final Δ = CPU {conv['final_last_delta']['CPU']:+.4f}, DSP {conv['final_last_delta']['DSP']:+.4f}.)

{f'Plot: `{plot_path.relative_to(REPO_ROOT)}`' if plot_path else '_(matplotlib not available; plot skipped.)_'}

## 4. Sum-of-parts sanity (per-op matrix vs solo E2E)

{fmt_sum_of_parts(sop)}

* `Σ per-op` = sum of `qnn_cost_matrix.json` entries / 1000 (matrix is μs).
* `E2E per-iter` = `qnn_e2e/measurements.json::total_ms / 10`.

**yolov8n agreement:** CPU Σ={y_cpu['sum_per_op_ms']:.1f}ms vs E2E {y_cpu['e2e_per_iter_ms']:.1f}ms ({y_cpu['disagreement_pct']:+.1f}%); GPU Σ={y_gpu['sum_per_op_ms']:.1f}ms vs E2E {y_gpu['e2e_per_iter_ms']:.1f}ms ({y_gpu['disagreement_pct']:+.1f}%); DSP Σ={y_dsp['sum_per_op_ms']:.1f}ms vs E2E {y_dsp['e2e_per_iter_ms']:.1f}ms ({y_dsp['disagreement_pct']:+.1f}%).

The CPU per-op profile is within ~20% of E2E (acceptable overhead band). The
DSP and GPU per-op profiles are **wildly low** vs the whole-network E2E —
the per-op-kernel timing on DSP/GPU is excluding offload setup, host↔device
transfer, and graph-launch overheads that dominate whole-network wall time
on those backends.

## 5. Chain-sum (per-op summed) prediction baseline vs solo E2E

{fmt_chain_vs_e2e(chain_eval)}

Mean abs % error of chain-sum baseline: **{chain_eval['mean_abs_pct']:.2f}%**
(median {chain_eval['median_abs_pct']:.2f}%).

The chain-sum baseline is an order of magnitude worse than the closed-loop
predictor on DSP/GPU specifically because the per-op matrix is missing the
fixed per-frame offload overhead. Pure chain summation is *not* a viable
solo-prediction baseline on DSP/GPU until that overhead is added back.

## 6. Direction of error

{agg['n_pessimistic']:.0f} of 4 rounds are pessimistic (pred > measured); the
single optimistic round (round 2) coincides with the DSP factor swinging
from 0.86 to 1.005 then back to 0.86 — an EMA over-correction in the
contention loop, not a per-op cost issue. Mean signed error is
{agg['bias_mean_signed_pct']:+.2f}% — the scheduler is **{direction_verdict}**.

---

## Verdict

**Mean abs % error:** {agg['mean_abs_pct']:.2f}% across 4 rounds.

**Bias:** {direction_verdict} ({agg['bias_mean_signed_pct']:+.2f}% signed mean).

**Sum-of-parts:** CPU Σ ≈ E2E (within {abs(y_cpu['disagreement_pct']):.0f}%);
DSP Σ off by {y_dsp['disagreement_pct']:+.1f}%; GPU Σ off by {y_gpu['disagreement_pct']:+.1f}%.
**Per-op DSP/GPU calibration data disagrees with whole-network E2E by an order of magnitude.**

**Convergence:** {convergence_line}

**Error source (single-sentence verdict):** **Contention modeling** — the
chain-sum CPU prediction matches whole-network CPU E2E to within ~20%, so
per-op CPU cost is fine; the 13–40% closed-loop swing is dominated by
EMA-over-correction on the DSP contention factor between rounds.

**Most actionable next step:** add a per-backend fixed-overhead term
(graph-launch / offload setup / transfer cost) to the cost model — without
it, the chain-sum baseline understates DSP wall by ~{abs(y_dsp['disagreement_pct']):.0f}% and
GPU wall by ~{abs(y_gpu['disagreement_pct']):.0f}%, and the EMA contention loop has to
absorb that gap, causing the round-to-round overshoot/undershoot pattern.
"""


def _compute_calibrated_rounds(cost_matrix: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float], float, float]:
    """Re-tabulate per-round error using the seeded calibration model.

    Returns a tuple of (per-round dicts, overhead_us map, mean abs % error
    before, mean abs % error after).
    """

    from xpu_rt.runtime.calibration import bootstrap_from_solo_measurements, load, save

    if CALIBRATION_PATH.exists():
        model = load(CALIBRATION_PATH)
    else:
        e2e = json.loads(E2E_PATH.read_text())
        model = bootstrap_from_solo_measurements(cost_matrix, e2e, target_id="qrb5165")
        save(model, CALIBRATION_PATH)

    per_op_sum_us, _ = sum_per_op_us(cost_matrix, CLOSED_LOOP_WORKLOAD, CLOSED_LOOP_BACKEND)
    overhead_us = float(model.overhead_us.get(CLOSED_LOOP_BACKEND, 0.0))
    predicted_with_calib_ms = (per_op_sum_us + overhead_us) / 1000.0

    rows: list[dict[str, Any]] = []
    abs_pct_before: list[float] = []
    abs_pct_after: list[float] = []
    for r, pred_ms, meas_ms in ROUNDS:
        before = abs(pred_ms - meas_ms) / meas_ms * 100.0
        after = abs(predicted_with_calib_ms - meas_ms) / meas_ms * 100.0
        rows.append(
            {
                "round": r,
                "measured_ms": meas_ms,
                "pred_orig_ms": pred_ms,
                "pred_calib_ms": round(predicted_with_calib_ms, 3),
                "abs_pct_before": round(before, 2),
                "abs_pct_after": round(after, 2),
            }
        )
        abs_pct_before.append(before)
        abs_pct_after.append(after)
    return (
        rows,
        dict(model.overhead_us),
        statistics.fmean(abs_pct_before),
        statistics.fmean(abs_pct_after),
    )


def sum_per_op_us(cost_matrix: dict[str, Any], workload: str, backend: str) -> tuple[float, int]:
    """Like sum_per_op() but in microseconds (the model's native unit)."""

    ops = cost_matrix.get(workload, {})
    total_us = 0.0
    covered = 0
    for op_name, costs in ops.items():
        if not isinstance(costs, dict):
            continue
        if backend in costs:
            total_us += float(costs[backend])
            covered += 1
    return total_us, covered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-calibration",
        action="store_true",
        help="Apply the seeded calibration model and report PASS/FAIL on the <15% gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = compute_round_errors()
    agg = aggregate(errors)
    contention_rows = load_contention()
    conv = convergence_diagnostic(contention_rows)
    cost_matrix = json.loads(COST_MATRIX_PATH.read_text())
    e2e = json.loads(E2E_PATH.read_text())
    sop = sum_of_parts(cost_matrix, e2e)
    chain = chain_sum_predict(cost_matrix)
    chain_eval = chain_sum_vs_e2e(chain, e2e)
    plot_path = maybe_plot_convergence(contention_rows)

    if args.with_calibration:
        rows, overhead_us, mean_before, mean_after = _compute_calibrated_rounds(cost_matrix)
        print("[exp9] --- calibrated re-tabulation (DSP lane) ---")
        print(f"[exp9] overhead_us seeded: {overhead_us}")
        print("[exp9] | round | measured (ms) | pred orig (ms) | pred+calib (ms) | |%| before | |%| after |")
        print("[exp9] |---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"[exp9] | {r['round']} | {r['measured_ms']:.1f} | {r['pred_orig_ms']:.1f} | "
                f"{r['pred_calib_ms']:.1f} | {r['abs_pct_before']:.2f}% | {r['abs_pct_after']:.2f}% |"
            )
        print(f"[exp9] mean abs % error before calibration: {mean_before:.2f}%")
        print(f"[exp9] mean abs % error after  calibration: {mean_after:.2f}%")
        passed = mean_after < 15.0
        if passed:
            print(f"[exp9] PASS — mean abs % error {mean_after:.2f}% < 15.00% (was {mean_before:.2f}%)")
        else:
            print(f"[exp9] FAIL — mean abs % error {mean_after:.2f}% ≥ 15.00%; residual {mean_after:.2f}%")
            print(
                "[exp9] suggest: add per-(workload, backend) transfer terms; "
                "the residual likely reflects per-frame DMA cost not captured by a single overhead constant."
            )
        # Persist calibrated summary for downstream stages.
        (OUT_DIR / "calibrated_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "exp9_calibrated_v1",
                    "workload": CLOSED_LOOP_WORKLOAD,
                    "backend": CLOSED_LOOP_BACKEND,
                    "overhead_us": overhead_us,
                    "rounds": rows,
                    "mean_abs_pct_before": mean_before,
                    "mean_abs_pct_after": mean_after,
                    "gate_pct": 15.0,
                    "passed": passed,
                },
                indent=2,
            )
        )

    report_md = build_report(errors, agg, contention_rows, conv, sop, chain, chain_eval, plot_path)
    (OUT_DIR / "report.md").write_text(report_md)

    summary: dict[str, Any] = {
        "schema_version": "exp9_calibration_v1",
        "rounds": [
            {
                "round": e.round,
                "pred_ms": e.pred_ms,
                "measured_ms": e.measured_ms,
                "error_ms": e.error_ms,
                "error_pct": e.error_pct,
                "direction": e.direction,
            }
            for e in errors
        ],
        "aggregate": agg,
        "contention": {
            "rows": contention_rows,
            "verdict": conv,
        },
        "sum_of_parts": sop,
        "chain_sum_predictions": chain,
        "chain_sum_vs_e2e": chain_eval,
        "plot_path": str(plot_path.relative_to(REPO_ROOT)) if plot_path else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Console digest.
    print(f"[exp9] mean abs % error : {agg['mean_abs_pct']:.2f}%")
    print(f"[exp9] bias signed %    : {agg['bias_mean_signed_pct']:+.2f}%")
    print(
        f"[exp9] round-4 |Δ| CPU={abs(conv['final_last_delta']['CPU']):.4f}, "
        f"DSP={abs(conv['final_last_delta']['DSP']):.4f}, "
        f"converged_per_threshold={conv['converged_per_threshold']}"
    )
    for wl in ("yolov8n",):
        for be in ("CPU", "GPU", "DSP"):
            row = sop[wl][be]
            print(
                f"[exp9] {wl}/{be}: Σ={row['sum_per_op_ms']:.1f}ms vs E2E={row['e2e_per_iter_ms']:.1f}ms "
                f"({row['disagreement_pct']:+.1f}%)"
            )
    print(f"[exp9] wrote {OUT_DIR / 'report.md'}")
    print(f"[exp9] wrote {OUT_DIR / 'summary.json'}")
    if plot_path:
        print(f"[exp9] wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
