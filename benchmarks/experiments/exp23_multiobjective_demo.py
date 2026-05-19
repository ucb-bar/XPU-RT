"""Exp 23 — Multi-objective scheduling demo on yolov8n + 12 x dronet.

Drives one feedback-loop step under three different MultiObjectiveSpec
weightings and reports how each spec ranks the resulting schedule:

  * Spec A — makespan-only (the legacy default).
  * Spec B — 0.7 * makespan + 0.3 * deadline_violation_count.
  * Spec C — 0.5 * makespan + 0.3 * deadline_violation_count
            + 0.2 * peak_memory_bytes (chunk-count-based proxy).

The deadline is 40 ms per dronet instance (the closed-loop SLO). Peak
memory is modelled as ``n_chunks * 16 MiB`` (proxy documented in
summary.md); this is not a real planner output, just a stand-in until
the Stage-4 memory planner is wired into the loop.

Outputs:
    build/experiments/exp23_multiobjective/results.jsonl
    build/experiments/exp23_multiobjective/summary.md
    build/experiments/exp23_multiobjective/pareto_plot.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp23_multiobjective"
RESULTS_PATH = OUT_DIR / "results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.md"
PARETO_PATH = OUT_DIR / "pareto_plot.png"

CAL_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"

DRONET_DEADLINE_US = 40_000.0  # 40 ms per dronet inference.
PEAK_MEMORY_BYTES_PER_CHUNK = 16 * 1024 * 1024  # 16 MiB chunk-count proxy.


def _build_specs() -> list[tuple[str, dict[str, Any]]]:
    from xpu_rt.scheduling.objectives import (
        MultiObjectiveSpec,
        ObjectiveKind,
        ObjectiveWeight,
    )

    spec_a = MultiObjectiveSpec()  # makespan-only default
    spec_b = MultiObjectiveSpec(weights=(
        ObjectiveWeight(ObjectiveKind.MAKESPAN, 0.7, target_value=100_000.0),
        ObjectiveWeight(
            ObjectiveKind.DEADLINE_VIOLATION_COUNT, 0.3, target_value=1.0
        ),
    ))
    spec_c = MultiObjectiveSpec(weights=(
        ObjectiveWeight(ObjectiveKind.MAKESPAN, 0.5, target_value=100_000.0),
        ObjectiveWeight(
            ObjectiveKind.DEADLINE_VIOLATION_COUNT, 0.3, target_value=1.0
        ),
        ObjectiveWeight(
            ObjectiveKind.PEAK_MEMORY_BYTES,
            0.2,
            target_value=float(PEAK_MEMORY_BYTES_PER_CHUNK * 16),
        ),
    ))
    return [("A", spec_a), ("B", spec_b), ("C", spec_c)]


def _run_one_step(spec_name: str, spec, cost_matrix, calibration) -> dict[str, Any]:
    from xpu_rt.runtime.calibration import MeasurementRecord
    from xpu_rt.scheduling.feedback_loop import (
        LoopConfig,
        init_loop_state,
        step,
    )
    from xpu_rt.scheduling.objectives import compute_metrics, evaluate

    cfg = LoopConfig(
        epsilon=0.10,
        max_iterations=4,
        max_chunk_ops=16,
        max_partitions=200,
        multi_objective=spec,
    )
    state = init_loop_state(
        workload_id="yolov8n",
        target_id="qrb5165",
        cost_matrix=cost_matrix,
        calibration=calibration,
        config=cfg,
    )
    # Plan-only first step so we have a schedule to score against.
    new_state = step(state, None, cost_matrix=cost_matrix, config=cfg)
    last = new_state.history[-1]
    score = last.objective_score
    raw = score.raw_metrics if score is not None else None

    # Rescore the same schedule under deadline + memory side-info so the
    # demo shows differentiation across specs even before the solver
    # learns about deadlines. We rebuild the schedule_dict view from
    # ``current_predicted_makespan_us`` + chunk count.
    n_chunks = len(new_state.current_chunks)
    # Synthetic deadlines: every chunk inherits a 40 ms cap (proxy for
    # per-dronet SLO in the chain-of-chunks view).
    deadlines = {c.chunk_id: DRONET_DEADLINE_US for c in new_state.current_chunks}
    # Reconstruct start/end times from the chunk chain assuming
    # sequential greedy execution at the round's predicted makespan.
    pred_us = float(new_state.current_predicted_makespan_us or 0.0)
    per_chunk = pred_us / max(1, n_chunks)
    starts = {c.chunk_id: i * per_chunk for i, c in enumerate(new_state.current_chunks)}
    ends = {c.chunk_id: (i + 1) * per_chunk for i, c in enumerate(new_state.current_chunks)}
    buffer_specs = [
        (starts[c.chunk_id], ends[c.chunk_id], PEAK_MEMORY_BYTES_PER_CHUNK)
        for c in new_state.current_chunks
    ]

    metrics = compute_metrics(
        start_times=starts,
        end_times=ends,
        device_assignments={c.chunk_id: 0 for c in new_state.current_chunks},
        deadlines_us=deadlines,
        buffer_specs=buffer_specs,
    )
    rescored = evaluate(spec, metrics)

    return {
        "spec_name": spec_name,
        "weights": [
            {"kind": str(w.kind), "weight": w.weight, "target_value": w.target_value}
            for w in spec.weights
        ],
        "makespan_us": metrics.makespan_us,
        "deadline_violations": metrics.deadline_violations,
        "deadline_violation_total_us": metrics.deadline_violation_total_us,
        "peak_memory_bytes": metrics.peak_memory_bytes,
        "n_chunks": n_chunks,
        "solver_choice": new_state.current_solver_choice,
        "score": rescored.score,
        "component_scores": dict(rescored.component_scores),
        "raw_score_post_step": (score.score if score is not None else None),
        "raw_metrics": {
            "makespan_us": (raw.makespan_us if raw else None),
            "deadline_violations": (raw.deadline_violations if raw else None),
            "peak_memory_bytes": (raw.peak_memory_bytes if raw else None),
        },
    }


def _plot_pareto(rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping pareto plot", file=sys.stderr)
        return

    from xpu_rt.scheduling.objectives import (
        ObjectiveScore,
        ScheduleMetrics,
        pareto_frontier,
    )

    scores: list[ObjectiveScore] = []
    for r in rows:
        m = ScheduleMetrics(
            makespan_us=r["makespan_us"],
            deadline_violations=r["deadline_violations"],
            deadline_violation_total_us=r["deadline_violation_total_us"],
            peak_memory_bytes=r["peak_memory_bytes"],
            energy_proxy_joules=None,
            makespan_variance_us=0.0,
        )
        scores.append(
            ObjectiveScore(
                score=r["score"],
                component_scores={
                    "makespan": r["makespan_us"],
                    "deadline_violation_count": float(r["deadline_violations"]),
                },
                raw_metrics=m,
            )
        )
    front = set(id(s) for s in pareto_frontier(scores))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r, s in zip(rows, scores):
        on_front = id(s) in front
        ax.scatter(
            r["makespan_us"] / 1000.0,
            r["deadline_violations"],
            s=120,
            c=("#1f77b4" if on_front else "#aaaaaa"),
            edgecolor="black",
            zorder=3,
        )
        ax.annotate(
            f"Spec {r['spec_name']}",
            (r["makespan_us"] / 1000.0, r["deadline_violations"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("Makespan (ms)")
    ax.set_ylabel("Deadline violation count")
    ax.set_title("exp23 — multi-objective scoring (filled = on Pareto frontier)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PARETO_PATH, dpi=120)
    plt.close(fig)


def _write_summary(rows: list[dict[str, Any]], frontier_size: int) -> None:
    lines = [
        "# exp23 — Multi-objective scheduling demo",
        "",
        "## Setup",
        "",
        "- Workload: yolov8n + 12x dronet (closed-loop scenario).",
        "- Per-dronet deadline: 40 ms.",
        f"- Peak-memory proxy: n_chunks x {PEAK_MEMORY_BYTES_PER_CHUNK // (1024*1024)} MiB.",
        "- One feedback-loop step per spec; scoring is post-hoc on the",
        "  resulting schedule (CP-SAT still optimises makespan only).",
        "",
        "## Per-spec results",
        "",
        "| Spec | makespan (ms) | deadline_viol | peak_mem (MiB) | score |",
        "| ---- | ------------- | ------------- | -------------- | ----- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['spec_name']} | {r['makespan_us']/1000:.1f} | "
            f"{r['deadline_violations']} | "
            f"{r['peak_memory_bytes'] / (1024*1024):.1f} | {r['score']:.3f} |"
        )
    lines.extend([
        "",
        f"## Pareto frontier size: {frontier_size}",
        "",
        "## Headline",
        "",
        "Spec B and C change the *score* assigned to a given schedule",
        "(their weighted aggregates pull deadline violation and peak",
        "memory into the objective). The Stage-4 solver dispatch is",
        "currently invariant to the spec (CP-SAT still minimises pure",
        "makespan), so the *raw schedule* — and hence the chunk count and",
        "peak-memory proxy — is identical across A/B/C at iteration 1.",
        "What differs is the score the loop's convergence rule sees in",
        "subsequent iterations, which is what feeds",
        "recompile_finer / recompile_coarser / converged decisions.",
        "",
        "Lifting the multi-objective into CP-SAT's objective (so the",
        "solver itself trades makespan against deadline slack) is the",
        "next-iteration extension and is out of scope for this commit.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    from xpu_rt.runtime.calibration import load as load_calibration
    from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
    from xpu_rt.scheduling.objectives import (
        ObjectiveScore,
        ScheduleMetrics,
        pareto_frontier,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    calibration = load_calibration(CAL_PATH)

    print("=" * 78)
    print("exp23 — multi-objective scheduling demo (yolov8n + 12x dronet)")
    print("=" * 78)

    rows: list[dict[str, Any]] = []
    for name, spec in _build_specs():
        print(f"\n>> Running spec {name} ({len(spec.weights)} objective(s))")
        row = _run_one_step(name, spec, cost_matrix, calibration)
        rows.append(row)
        print(
            f"   makespan_us={row['makespan_us']:.0f}  "
            f"deadline_viol={row['deadline_violations']}  "
            f"peak_mem_MiB={row['peak_memory_bytes'] / (1024*1024):.1f}  "
            f"score={row['score']:.3f}"
        )

    # Pareto frontier across (makespan, deadline_violations).
    scores = [
        ObjectiveScore(
            score=r["score"],
            component_scores={
                "makespan": r["makespan_us"],
                "deadline_violation_count": float(r["deadline_violations"]),
            },
            raw_metrics=ScheduleMetrics(
                makespan_us=r["makespan_us"],
                deadline_violations=r["deadline_violations"],
                deadline_violation_total_us=r["deadline_violation_total_us"],
                peak_memory_bytes=r["peak_memory_bytes"],
                energy_proxy_joules=None,
                makespan_variance_us=0.0,
            ),
        )
        for r in rows
    ]
    front = pareto_frontier(scores)
    print(f"\nPareto frontier size: {len(front)} / {len(scores)}")

    with RESULTS_PATH.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")
    _write_summary(rows, frontier_size=len(front))
    _plot_pareto(rows)

    print(f"\nArtifacts written under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
