"""Multi-rate dominant-workload demo.

Loads the QRB5165 cost matrix + calibration, runs the multi-rate
analysis over ``[yolov8n, dronet]``, prints the recommendation, compares
it to the closed-loop's hand-tuned ``12× dronet`` value, and renders a
synthetic Gantt of one yolov8n cycle plus N×dronet cycles.

Outputs (under ``build/experiments/exp19_multi_rate/``):
  - ``summary.md``               — narrative + comparison.
  - ``multiplicity_gantt.png``   — synthetic Gantt of the recommendation.
  - ``results.json``             — full :class:`MultiRateAnalysis` payload.

Usage:
    uv run python scripts/experiments/exp19_multi_rate_demo.py

This is a static planning analysis. It assumes the dominant workload
occupies only its preferred lane and that all other lanes are fully
idle during the dominant period. Real schedules may reduce the
multiplicity if specialty chunking spills onto other lanes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp19_multi_rate"
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
CALIBRATION_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"

WORKLOADS: tuple[str, ...] = ("yolov8n", "dronet")
BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")
CLOSED_LOOP_HARDCODED_DRONET = 12

# Repo modules must import before matplotlib because matplotlib pulls in
# its own logging stack which can shadow structlog defaults.
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))

from xpu_rt.runtime.calibration import load as load_calibration  # noqa: E402
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix  # noqa: E402
from xpu_rt.scheduling.multi_rate import (  # noqa: E402
    MultiRateAnalysis,
    WorkloadRate,
    _best_backend_period,
    analyze,
    compute_multiplicity,
    estimate_lane_availability,
)


def _require_matplotlib() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.patches import Patch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        print(
            f"FATAL: matplotlib unavailable ({exc}); install with: "
            "uv run python -m pip install matplotlib",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _analysis_to_payload(a: MultiRateAnalysis) -> dict[str, Any]:
    return {
        "dominant_workload_id": a.dominant_workload_id,
        "dominant_period_us": a.dominant_period_us,
        "rates": [
            {
                "workload_id": r.workload_id,
                "period_us": r.period_us,
                "multiplicity": r.multiplicity,
                "preferred_lane": r.preferred_lane,
                "primary_lane_busy_us": r.primary_lane_busy_us,
                "achievable_frequency_hz": r.achievable_frequency_hz,
            }
            for r in a.rates
        ],
        "lane_availability": [
            {
                "lane": la.lane,
                "busy_us": la.busy_us_per_dominant_period,
                "idle_us": la.idle_us_per_dominant_period,
                "busy_fraction": la.busy_fraction,
            }
            for la in a.lane_availability
        ],
        "notes": list(a.notes),
    }


def _print_recommendation(a: MultiRateAnalysis) -> None:
    print(f"Dominant: {a.dominant_workload_id}  period_us = {a.dominant_period_us:.1f}")
    for r in a.rates:
        print(
            f"  {r.workload_id:>10s}: multiplicity = {r.multiplicity:>3d}  "
            f"lane = {r.preferred_lane:>4s}  "
            f"freq = {r.achievable_frequency_hz:>10.2f} Hz  "
            f"period = {r.period_us:.1f} us"
        )
    print("Lane availability (during one dominant period):")
    for la in a.lane_availability:
        print(
            f"  {la.lane:>4s}: busy = {la.busy_us_per_dominant_period:>10.1f} us  "
            f"idle = {la.idle_us_per_dominant_period:>10.1f} us  "
            f"busy_fraction = {la.busy_fraction:.3f}"
        )


def _compare_to_closed_loop(a: MultiRateAnalysis, n_hardcoded: int) -> str:
    dronet_rate = next(r for r in a.rates if r.workload_id == "dronet")
    n = dronet_rate.multiplicity
    if n > n_hardcoded:
        return (
            f"system would propose more aggressive multi-rate "
            f"({n}× dronet vs hand-tuned {n_hardcoded}×)"
        )
    if n == n_hardcoded:
        return f"system matches closed-loop's hand-tuned value ({n}×)"
    return (
        f"system is more conservative ({n}× dronet vs {n_hardcoded}×); "
        "the dominant's preferred lane is busier than the closed-loop "
        "assumed, or per-cycle cost is higher under this calibration"
    )


def _render_gantt(
    a: MultiRateAnalysis,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    dominant = next(r for r in a.rates if r.workload_id == a.dominant_workload_id)
    secondaries = [r for r in a.rates if r.workload_id != a.dominant_workload_id]

    fig, ax = plt.subplots(figsize=(14, 5))
    backends = list(BACKENDS)
    y_for = {b: i for i, b in enumerate(backends)}

    # Dominant on its preferred lane.
    dom_color = (0.86, 0.20, 0.20, 0.92)
    ax.broken_barh(
        [(0.0, max(dominant.primary_lane_busy_us, 1.0))],
        (y_for[dominant.preferred_lane] - 0.4, 0.8),
        facecolors=dom_color,
        edgecolors="black",
        linewidth=0.4,
    )

    # Secondaries: pack N copies back-to-back on the preferred lane.
    sec_palette = plt.get_cmap("viridis")
    handles = [Patch(facecolor=dom_color, edgecolor="black", label=dominant.workload_id)]
    for s_idx, sec in enumerate(secondaries):
        if sec.multiplicity <= 0:
            continue
        per_cycle = sec.primary_lane_busy_us
        color_t = (s_idx + 1) / max(len(secondaries), 1)
        sec_color = sec_palette(color_t)
        for i in range(sec.multiplicity):
            ax.broken_barh(
                [(i * per_cycle, max(per_cycle, 1.0))],
                (y_for[sec.preferred_lane] - 0.4, 0.8),
                facecolors=sec_color,
                edgecolors="black",
                linewidth=0.25,
            )
        handles.append(
            Patch(
                facecolor=sec_color,
                edgecolor="black",
                label=f"{sec.workload_id} ×{sec.multiplicity}",
            )
        )

    ax.set_yticks(list(range(len(backends))))
    ax.set_yticklabels(backends)
    ax.set_ylim(-0.7, len(backends) - 0.3)
    ax.set_xlim(0, a.dominant_period_us * 1.04)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("device")
    ax.axvline(
        a.dominant_period_us,
        color="black",
        linestyle="--",
        alpha=0.55,
        linewidth=1.0,
        label="dominant period",
    )
    ax.set_title(
        f"Multi-rate recommendation  -  dominant: {a.dominant_workload_id}  "
        f"period={a.dominant_period_us / 1000.0:.1f} ms"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    a: MultiRateAnalysis,
    comparison: str,
    out_md: Path,
) -> None:
    dronet_rate = next(r for r in a.rates if r.workload_id == "dronet")
    lines: list[str] = []
    lines.append("# Experiment 19 - Multi-rate dominant workload demo\n")
    lines.append("Static planning analysis of `[yolov8n, dronet]` on QRB5165.\n")
    lines.append("## Recommendation\n")
    lines.append(f"- Dominant workload: **{a.dominant_workload_id}**")
    lines.append(f"- Dominant period: **{a.dominant_period_us / 1000.0:.2f} ms**")
    lines.append(
        f"- Recommended dronet multiplicity: **{dronet_rate.multiplicity}×** "
        f"(closed-loop hand-tuned: {CLOSED_LOOP_HARDCODED_DRONET}×)"
    )
    lines.append(f"- Recommended dronet preferred lane: **{dronet_rate.preferred_lane}**")
    lines.append(
        f"- Achievable dronet frequency: "
        f"**{dronet_rate.achievable_frequency_hz:.1f} Hz**"
    )
    lines.append("")
    lines.append(f"_Comparison_: {comparison}.\n")

    lines.append("## Lane availability (during one dominant period)\n")
    lines.append("| Lane | Busy (us) | Idle (us) | Busy fraction |")
    lines.append("|------|-----------|-----------|---------------|")
    for la in a.lane_availability:
        lines.append(
            f"| {la.lane} | {la.busy_us_per_dominant_period:.1f} | "
            f"{la.idle_us_per_dominant_period:.1f} | {la.busy_fraction:.3f} |"
        )
    lines.append("")

    lines.append("## Per-workload rates\n")
    lines.append("| Workload | Period (us) | Multiplicity | Preferred lane | Freq (Hz) |")
    lines.append("|----------|-------------|--------------|----------------|-----------|")
    for r in a.rates:
        lines.append(
            f"| {r.workload_id} | {r.period_us:.1f} | {r.multiplicity} | "
            f"{r.preferred_lane} | {r.achievable_frequency_hz:.1f} |"
        )
    lines.append("")

    lines.append("## Honest framing\n")
    lines.append(
        "This is a **static analysis**. The recommended multiplicity is an "
        "upper bound assuming no contention beyond the dominant's preferred "
        "lane. The real achievable multiplicity may be lower if the "
        "dominant's specialty chunking spills onto other lanes."
    )
    lines.append(
        "Without a board run, the recommendation is a planning estimate, not "
        "a measured ceiling. The closed-loop's "
        f"`{CLOSED_LOOP_HARDCODED_DRONET}× dronet` value is the only "
        "data-grounded number we have to compare against.\n"
    )
    if a.notes:
        lines.append("### Analyzer notes\n")
        for n in a.notes:
            lines.append(f"- {n}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def _analyze_per_workload(
    workload_ids: tuple[str, ...],
    cost_matrix: dict,
    per_workload_overhead: dict[str, dict[str, float]],
) -> MultiRateAnalysis:
    """Multi-rate analysis that uses each workload's own per-backend overhead.

    The library ``analyze()`` takes a single per-backend overhead dict
    (the v1 contract). v2 calibration is per-(workload, backend), so we
    drive the analysis manually:

      1. Compute each workload's best-backend period using its own
         overhead.
      2. Identify the dominant (largest-period) workload.
      3. Estimate lane availability with the dominant's overhead.
      4. For each secondary, compute multiplicity with the secondary's
         own overhead.

    Returns a :class:`MultiRateAnalysis` with the same shape ``analyze()``
    would return, but with per-workload calibration applied throughout.
    Falls back to ``analyze()`` (no overhead) when ``per_workload_overhead``
    is empty.
    """
    import math

    if not per_workload_overhead:
        return analyze(workload_ids, cost_matrix, calibration_overhead_us=None)

    # Per-workload best backend + period.
    best_per_workload: dict[str, tuple[str, float, float]] = {}
    for w in workload_ids:
        wl_overhead = per_workload_overhead.get(w, {})
        best_per_workload[w] = _best_backend_period(
            cost_matrix, w, wl_overhead
        )
    finite = {
        w: triple
        for w, triple in best_per_workload.items()
        if math.isfinite(triple[1])
    }
    if not finite:
        raise ValueError(f"no feasible backend for any workload in {workload_ids!r}")
    dominant_id = max(finite.items(), key=lambda kv: (kv[1][1], kv[0]))[0]
    dominant_lane, dominant_period, _ = best_per_workload[dominant_id]

    lane_avail = estimate_lane_availability(
        dominant_workload_id=dominant_id,
        dominant_preferred_lane=dominant_lane,
        dominant_period_us=dominant_period,
        cost_matrix=cost_matrix,
        calibration_overhead_us=per_workload_overhead.get(dominant_id, {}),
    )

    rates: list[WorkloadRate] = []
    notes: list[str] = [
        "Per-workload calibration v2: overhead indexed by (workload, backend).",
        "Static upper-bound model: dominant assumed to occupy only its preferred lane.",
        "Without a board run, the recommendation is a planning estimate, not a ceiling.",
    ]

    for wid in workload_ids:
        if wid == dominant_id:
            primary_busy = next(
                (
                    la.busy_us_per_dominant_period
                    for la in lane_avail
                    if la.lane == dominant_lane
                ),
                0.0,
            )
            freq = (1e6 / dominant_period) if dominant_period > 0 else 0.0
            rates.append(
                WorkloadRate(
                    workload_id=wid,
                    period_us=dominant_period,
                    multiplicity=1,
                    preferred_lane=dominant_lane,
                    primary_lane_busy_us=primary_busy,
                    achievable_frequency_hz=freq,
                )
            )
            continue

        secondary_period = best_per_workload.get(wid, ("", math.inf, 0.0))[1]
        mult, lane, cost_per_cycle = compute_multiplicity(
            secondary_workload_id=wid,
            dominant_period_us=dominant_period,
            lane_availability=lane_avail,
            cost_matrix=cost_matrix,
            calibration_overhead_us=per_workload_overhead.get(wid, {}),
        )
        if mult > 0 and dominant_period > 0:
            freq = 1e6 / (dominant_period / mult)
        elif math.isfinite(secondary_period) and secondary_period > 0:
            freq = 1e6 / secondary_period
        else:
            freq = 0.0
        if mult == 0:
            notes.append(
                f"{wid}: no lane fits even one cycle under dominant {dominant_id}; "
                "reporting fallback lane and 0 multiplicity."
            )
        rates.append(
            WorkloadRate(
                workload_id=wid,
                period_us=secondary_period if math.isfinite(secondary_period) else 0.0,
                multiplicity=mult,
                preferred_lane=lane,
                primary_lane_busy_us=cost_per_cycle * mult,
                achievable_frequency_hz=freq,
            )
        )

    return MultiRateAnalysis(
        dominant_workload_id=dominant_id,
        dominant_period_us=dominant_period,
        rates=tuple(rates),
        lane_availability=lane_avail,
        notes=tuple(notes),
    )


def _scale_costs_by_contention(
    cost_matrix: dict,
    workload_id: str,
    per_backend_contention: dict[str, float],
) -> dict:
    """Return a per-workload-scaled view of ``cost_matrix``.

    v3 calibration applies contention as a multiplicative residual on the
    full partition cost. The library multi-rate analyzer takes per-op
    costs directly, so we fold contention into the per-op cells for
    ``workload_id`` only. Cells with no contention data have factor 1.0
    (no-op). Other workloads pass through unchanged.
    """
    if not per_backend_contention or all(v == 1.0 for v in per_backend_contention.values()):
        return cost_matrix
    scaled: dict = {}
    for k, v in cost_matrix.items():
        if k != workload_id or not isinstance(v, dict):
            scaled[k] = v
            continue
        scaled_ops: dict = {}
        for op_id, costs in v.items():
            if not isinstance(costs, dict):
                scaled_ops[op_id] = costs
                continue
            scaled_ops[op_id] = {
                b: float(c) * float(per_backend_contention.get(b, 1.0))
                for b, c in costs.items()
            }
        scaled[k] = scaled_ops
    return scaled


def main() -> int:
    _require_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    per_workload_overhead: dict[str, dict[str, float]] = {}
    per_workload_contention: dict[str, dict[str, float]] = {}
    if CALIBRATION_PATH.is_file():
        cal = load_calibration(CALIBRATION_PATH)
        per_workload_overhead = {
            wid: dict(per_b) for wid, per_b in cal.overhead_us.items()
        }
        per_workload_contention = {
            wid: dict(per_b) for wid, per_b in cal.contention_factor.items()
        }
        print(f"Calibration v3 loaded: overhead_us = {per_workload_overhead}")
        print(f"                       contention_factor = {per_workload_contention}")
    else:
        print(f"Calibration not found at {CALIBRATION_PATH}; running uncalibrated.")

    # Fold contention into per-op costs (one workload at a time) so the
    # multi-rate analyzer sees the v3 effective cost without a library
    # signature change.
    cost_matrix_v3 = cost_matrix
    for w in WORKLOADS:
        cost_matrix_v3 = _scale_costs_by_contention(
            cost_matrix_v3, w, per_workload_contention.get(w, {})
        )

    analysis = _analyze_per_workload(WORKLOADS, cost_matrix_v3, per_workload_overhead)
    _print_recommendation(analysis)

    comparison = _compare_to_closed_loop(analysis, CLOSED_LOOP_HARDCODED_DRONET)
    print(f"\nVS closed-loop's hand-tuned {CLOSED_LOOP_HARDCODED_DRONET}× dronet: {comparison}")

    payload = _analysis_to_payload(analysis)
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _render_gantt(analysis, OUT_DIR / "multiplicity_gantt.png")
    _write_summary(analysis, comparison, OUT_DIR / "summary.md")
    print(f"\nWrote {OUT_DIR / 'summary.md'}")
    print(f"Wrote {OUT_DIR / 'multiplicity_gantt.png'}")
    print(f"Wrote {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
