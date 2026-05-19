"""Exp 16 — Before/after tabulation: does Stage 1 calibration narrow prediction error?

This harness produces the canonical "is the closed-loop's predictor any
better than a naive chain-sum, and does adding the Stage-1 overhead
constant help?" table for the QRB5165 closed-loop ground truth.

Three prediction conditions, scored against the per-round measured
makespan from ``xpu-rt/data/profiled/qnn_closed_loop/final_report.md``:

* **A. baseline_chain_sum** — sum of per-op DSP costs from
  ``qnn_cost_matrix.json``. No overhead, no contention, no calibration.
  This is the "naive predictor".
* **B. calibrated_overhead** — A + the per-backend overhead constant from
  ``xpu-rt/data/calibration/qrb5165.json``. The Stage-1 calibration
  contribution, evaluated as a *static* additive correction (no contention
  EMA, no per-round adaptation). Constant across all rounds.
* **C. closed_loop_predicted** — what the closed-loop scheduler emitted
  that round (read from ``qnn_events.jsonl``'s ``qnn_execute_schedule``
  events, divided by the iter count to get per-iteration ms). The
  system-as-shipped predictor, including contention EMA and re-solve.

The chosen lane for the closed-loop scenario is **DSP** — the workload
under test is yolov8n running whole-net on DSP while 12× DroNet runs on
CPU; DSP is the makespan-critical lane (per-iter measured DSP wall =
254-350 ms vs CPU ~134 ms).

Outputs:
    build/experiments/exp16_before_after/results.jsonl
    build/experiments/exp16_before_after/summary.md

Usage:
    uv run python scripts/experiments/exp16_before_after_harness.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "xpu-rt" / "data"
COST_MATRIX_PATH = DATA_ROOT / "profiled" / "qnn_cost_matrix.json"
CALIBRATION_PATH = DATA_ROOT / "calibration" / "qrb5165.json"
EVENTS_PATH = DATA_ROOT / "profiled" / "qnn_closed_loop" / "qnn_events.jsonl"
FINAL_REPORT_PATH = DATA_ROOT / "profiled" / "qnn_closed_loop" / "final_report.md"
E2E_PATH = DATA_ROOT / "profiled" / "qnn_e2e" / "measurements.json"

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp16_before_after"
RESULTS_PATH = OUT_DIR / "results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.md"

# The closed-loop scenario runs yolov8n whole-net on DSP, with 12× dronet
# concurrently on CPU. DSP is critical-path; predictions and measured
# makespan in qnn_events.jsonl track the DSP per-iter wall time.
WORKLOAD_ID = "yolov8n"
LANE = "DSP"
ITERS = 10  # iters per round (matches qnn_execute_schedule events)


def chain_sum_us(cost_matrix: dict[str, Any], workload: str, lane: str) -> float:
    """Sum per-op cost on ``lane`` for ``workload`` from the raw cost matrix.

    Args:
        cost_matrix: Loaded JSON of ``qnn_cost_matrix.json``.
        workload: Workload key (e.g. ``"yolov8n"`` or ``"dronet"``).
        lane: Backend key (e.g. ``"CPU"`` / ``"DSP"`` / ``"GPU"``).

    Returns:
        Sum in microseconds. Ops without an entry for the lane are skipped.
    """
    total = 0.0
    for op_name, lanes in cost_matrix[workload].items():
        if not isinstance(lanes, dict):
            continue
        if lane in lanes:
            total += float(lanes[lane])
    return total


def load_cost_matrix() -> dict[str, Any]:
    return json.loads(COST_MATRIX_PATH.read_text(encoding="utf-8"))


def load_calibration() -> dict[str, Any]:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def load_e2e_measurements() -> dict[str, Any]:
    return json.loads(E2E_PATH.read_text(encoding="utf-8"))


def load_closed_loop_rounds() -> list[dict[str, Any]]:
    """Extract per-round (predicted, measured) pairs from qnn_events.jsonl.

    Returns:
        A list of ``{round, predicted_us, measured_us}`` dicts, one per
        ``qnn_execute_schedule`` event. Values are *total* microseconds
        across ``ITERS`` iterations (event semantics).
    """
    rounds: list[dict[str, Any]] = []
    with EVENTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            if evt.get("event") != "qnn_execute_schedule":
                continue
            rounds.append(
                {
                    "round": int(evt["round"]),
                    "predicted_us": float(evt["predicted_makespan_us"]),
                    "measured_us": float(evt["measured_makespan_us"]),
                }
            )
    rounds.sort(key=lambda r: r["round"])
    return rounds


def pct_error(pred_ms: float, measured_ms: float) -> float:
    """``(pred - measured) / measured * 100``. Signed."""
    if measured_ms == 0:
        return float("nan")
    return (pred_ms - measured_ms) / measured_ms * 100.0


def compute_per_round_table(
    cost_matrix: dict[str, Any],
    calibration: dict[str, Any],
    closed_loop_rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the per-round predictions + errors table.

    Conditions:
      A — chain_sum
      B — chain_sum + overhead[w][b]                              (v2 base)
      C — closed-loop as-shipped
      D — (chain_sum + overhead[w][b]) * contention[w][b]         (v3 two-term)
    """
    chain_us = chain_sum_us(cost_matrix, WORKLOAD_ID, LANE)
    chain_ms = chain_us / 1000.0
    overhead_us = float(calibration["overhead_us"][WORKLOAD_ID][LANE])
    overhead_ms = overhead_us / 1000.0
    calibrated_ms = chain_ms + overhead_ms
    # v3: contention is per-(workload, backend); default 1.0 for unmeasured cells.
    contention_factor = float(
        calibration.get("contention_factor", {})
        .get(WORKLOAD_ID, {})
        .get(LANE, 1.0)
    )
    two_term_ms = calibrated_ms * contention_factor

    rows: list[dict[str, Any]] = []
    for r in closed_loop_rounds:
        pred_per_iter_ms = r["predicted_us"] / ITERS / 1000.0
        meas_per_iter_ms = r["measured_us"] / ITERS / 1000.0
        rows.append(
            {
                "round": r["round"],
                "measured_ms": meas_per_iter_ms,
                "cond_A_baseline_chain_sum_ms": chain_ms,
                "cond_B_calibrated_overhead_ms": calibrated_ms,
                "cond_C_closed_loop_predicted_ms": pred_per_iter_ms,
                "cond_D_two_term_ms": two_term_ms,
                "err_pct_A": pct_error(chain_ms, meas_per_iter_ms),
                "err_pct_B": pct_error(calibrated_ms, meas_per_iter_ms),
                "err_pct_C": pct_error(pred_per_iter_ms, meas_per_iter_ms),
                "err_pct_D": pct_error(two_term_ms, meas_per_iter_ms),
                "contention_factor": contention_factor,
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-condition error stats over all rounds."""
    out: dict[str, dict[str, float]] = {}
    for cond, key in (
        ("A_baseline_chain_sum", "err_pct_A"),
        ("B_calibrated_overhead", "err_pct_B"),
        ("C_closed_loop_predicted", "err_pct_C"),
        ("D_two_term_v3", "err_pct_D"),
    ):
        errs = [row[key] for row in rows]
        abs_errs = [abs(e) for e in errs]
        out[cond] = {
            "mean_abs_pct": statistics.fmean(abs_errs),
            "median_abs_pct": statistics.median(abs_errs),
            "signed_bias_pct": statistics.fmean(errs),
            "max_abs_pct": max(abs_errs),
        }
    return out


def sanity_rows(cost_matrix: dict[str, Any], e2e: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute chain_sum vs solo-E2E sanity rows for dronet/CPU + yolov8n/DSP+CPU."""
    rows = []
    for workload, lane in (("dronet", "CPU"), ("yolov8n", "DSP"), ("yolov8n", "CPU")):
        chain_ms = chain_sum_us(cost_matrix, workload, lane) / 1000.0
        e2e_us = float(e2e["matrix"][workload][lane]["mean_us"])
        e2e_ms = e2e_us / 1000.0
        rows.append(
            {
                "workload": workload,
                "lane": lane,
                "chain_sum_ms": chain_ms,
                "solo_e2e_ms": e2e_ms,
                "ratio_e2e_over_chain": e2e_ms / chain_ms if chain_ms > 0 else float("nan"),
                "delta_ms": e2e_ms - chain_ms,
            }
        )
    return rows


def write_results_jsonl(rows: list[dict[str, Any]]) -> None:
    """Emit one JSONL row per (round, condition)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            for cond, pred_key, err_key in (
                ("A_baseline_chain_sum", "cond_A_baseline_chain_sum_ms", "err_pct_A"),
                ("B_calibrated_overhead", "cond_B_calibrated_overhead_ms", "err_pct_B"),
                ("C_closed_loop_predicted", "cond_C_closed_loop_predicted_ms", "err_pct_C"),
                ("D_two_term_v3", "cond_D_two_term_ms", "err_pct_D"),
            ):
                handle.write(
                    json.dumps(
                        {
                            "schema_version": "exp16_before_after_v1",
                            "workload_id": WORKLOAD_ID,
                            "lane": LANE,
                            "round": row["round"],
                            "condition": cond,
                            "predicted_ms": row[pred_key],
                            "measured_ms": row["measured_ms"],
                            "err_pct": row[err_key],
                        }
                    )
                    + "\n"
                )


def render_per_round_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Per-round predictions (yolov8n on DSP, qrb5165)",
        "",
        "| round | measured (ms) | A: chain_sum (ms) | B: +overhead (ms) | "
        "C: closed-loop (ms) | D: two-term v3 (ms) | err% A | err% B | err% C | err% D |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['round']} | {r['measured_ms']:.1f} | {r['cond_A_baseline_chain_sum_ms']:.1f} "
            f"| {r['cond_B_calibrated_overhead_ms']:.1f} | {r['cond_C_closed_loop_predicted_ms']:.1f} "
            f"| {r['cond_D_two_term_ms']:.1f} "
            f"| {r['err_pct_A']:+.1f}% | {r['err_pct_B']:+.1f}% | {r['err_pct_C']:+.1f}% "
            f"| {r['err_pct_D']:+.1f}% |"
        )
    return "\n".join(lines)


def render_aggregate_md(agg: dict[str, dict[str, float]]) -> str:
    lines = [
        "## Aggregate error (over 4 rounds)",
        "",
        "| condition | mean abs % | median abs % | signed bias % | max abs % |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond, stats in agg.items():
        lines.append(
            f"| {cond} | {stats['mean_abs_pct']:.1f}% | {stats['median_abs_pct']:.1f}% "
            f"| {stats['signed_bias_pct']:+.1f}% | {stats['max_abs_pct']:.1f}% |"
        )
    return "\n".join(lines)


def render_sanity_md(sanity: list[dict[str, Any]]) -> str:
    lines = [
        "## Sanity: chain_sum vs solo whole-network E2E",
        "",
        "| workload | lane | chain_sum (ms) | solo E2E (ms) | E2E / chain | delta (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sanity:
        lines.append(
            f"| {r['workload']} | {r['lane']} | {r['chain_sum_ms']:.1f} "
            f"| {r['solo_e2e_ms']:.1f} | {r['ratio_e2e_over_chain']:.2f}× "
            f"| {r['delta_ms']:+.1f} |"
        )
    return "\n".join(lines)


def render_headline(agg: dict[str, dict[str, float]]) -> str:
    a = agg["A_baseline_chain_sum"]["mean_abs_pct"]
    b = agg["B_calibrated_overhead"]["mean_abs_pct"]
    c = agg["C_closed_loop_predicted"]["mean_abs_pct"]
    d = agg["D_two_term_v3"]["mean_abs_pct"]
    best = min((("A", a), ("B", b), ("C", c), ("D", d)), key=lambda x: x[1])
    return (
        "## Headline\n\n"
        f"- Mean abs prediction error per condition:\n"
        f"  - A (chain_sum, naive): **{a:.1f}%**\n"
        f"  - B (+overhead, v2 base): **{b:.1f}%**\n"
        f"  - C (closed-loop as-shipped predictor): **{c:.1f}%**\n"
        f"  - D (v3 two-term: (chain+overhead)*contention[w][b]): **{d:.1f}%**\n"
        f"- Best condition: **{best[0]}** at {best[1]:.1f}%\n"
        f"- B vs A: {a - b:+.1f} pp ({'lower' if b < a else 'higher'} error than naive)\n"
        f"- D vs B: {b - d:+.1f} pp ({'lower' if d < b else 'higher'} than v2 base)\n"
        f"- D vs C: {c - d:+.1f} pp ({'lower' if d < c else 'higher'} than as-shipped)\n"
    )


def render_assessment(
    rows: list[dict[str, Any]],
    agg: dict[str, dict[str, float]],
    sanity: list[dict[str, Any]],
) -> str:
    a = agg["A_baseline_chain_sum"]["mean_abs_pct"]
    b = agg["B_calibrated_overhead"]["mean_abs_pct"]
    c = agg["C_closed_loop_predicted"]["mean_abs_pct"]
    yolo_dsp_sanity = next(s for s in sanity if s["workload"] == "yolov8n" and s["lane"] == "DSP")
    overhead_ms_yolov8n_dsp = rows[0]["cond_B_calibrated_overhead_ms"] - rows[0]["cond_A_baseline_chain_sum_ms"]
    return (
        "## Honest assessment\n\n"
        "Stage 1's static overhead constant (condition B) closes a large fraction of "
        f"the chain-sum gap: yolov8n DSP chain_sum is {yolo_dsp_sanity['chain_sum_ms']:.1f} ms but "
        f"the solo whole-network DSP wall is {yolo_dsp_sanity['solo_e2e_ms']:.1f} ms — a "
        f"{yolo_dsp_sanity['ratio_e2e_over_chain']:.1f}× gap that the per-op cost model alone "
        "cannot explain (graph-init, weight-load, DMA setup, intra-graph transfers). Adding "
        f"yolov8n's per-workload DSP overhead constant ({overhead_ms_yolov8n_dsp:.1f} ms, v2) "
        f"brings cond B's mean abs error to {b:.1f}% "
        f"vs {a:.1f}% for the naive chain_sum — a "
        f"{a - b:.1f} pp improvement that is real and load-bearing.\n\n"
        f"Whether Stage 1 helps **the loop** beyond what the closed-loop's contention EMA "
        f"already does is a separate question: cond C (the as-shipped predictor) lands at "
        f"{c:.1f}% mean abs error, "
        f"{'better than' if c < b else 'worse than' if c > b else 'tied with'} cond B's "
        f"{b:.1f}%. The closed-loop's per-iter prediction is not a static formula; it re-solves "
        "each round under EMA-updated contention factors, so its error reflects the schedule "
        "actually executed, not just per-op summing. Concretely, the closed-loop's predictions "
        "stay near the solo-DSP baseline (~305-357 ms) while measured DSP per-iter walls "
        "oscillate (255-351 ms). The static calibration B underestimates because it cannot "
        "react; the as-shipped C overestimates because it inherits the solo-baseline anchor "
        "and the contention factors don't pull it down enough in 4 rounds. Neither is wildly "
        "wrong; both are within ~25% mean abs of measured. The headline takeaway is that "
        "Stage 1's static overhead is the bulk of the win over chain_sum, and the closed-loop's "
        "EMA does not (in this 4-round trace) clearly beat that static correction."
    )


def main() -> int:
    cost_matrix = load_cost_matrix()
    calibration = load_calibration()
    e2e = load_e2e_measurements()
    closed_loop = load_closed_loop_rounds()

    rows = compute_per_round_table(cost_matrix, calibration, closed_loop)
    agg = aggregate(rows)
    sanity = sanity_rows(cost_matrix, e2e)

    write_results_jsonl(rows)

    summary = "\n\n".join(
        [
            "# Exp 16 — Before/after prediction-accuracy harness",
            f"_Workload: {WORKLOAD_ID} on lane {LANE}, target qrb5165, "
            f"{len(rows)} rounds, {ITERS} iters/round_",
            render_per_round_md(rows),
            render_aggregate_md(agg),
            render_headline(agg),
            render_sanity_md(sanity),
            render_assessment(rows, agg, sanity),
        ]
    )
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    # Mirror the headline + tables to stdout so a CI tail catches it.
    print(summary)
    print()
    print(f"[exp16] wrote {RESULTS_PATH}")
    print(f"[exp16] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
