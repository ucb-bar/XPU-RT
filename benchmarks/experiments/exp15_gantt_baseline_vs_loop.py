"""Side-by-side Gantt: closed-loop whole_net baseline vs feedback-loop schedule.

Builds two schedules over the canonical (yolov8n + 12x dronet) workload mix
and renders them as a single PNG so the visual difference between
"whole_net per-DLC placement" (the scheme the closed-loop run executed across
its 4 rounds in `xpu-rt/data/profiled/qnn_closed_loop/`) and the calibrated
specialty + CP-SAT joint scheduler is immediate.

Outputs (under ``build/experiments/exp15_gantt/``):
  - ``baseline_vs_loop.png``     -- the headline figure.
  - ``baseline_schedule.json``   -- baseline partitions + start/end/device.
  - ``loop_schedule.json``       -- same shape, calibrated-loop schedule.
  - ``summary.md``               -- short narrative + headline numbers.

Usage:
    uv run python scripts/experiments/exp15_gantt_baseline_vs_loop.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp15_gantt"
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
CALIBRATION_PATH = REPO_ROOT / "xpu-rt" / "data" / "calibration" / "qrb5165.json"
E2E_MEASUREMENTS_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_e2e" / "measurements.json"

BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")
N_DRONET_INSTANCES = 12
CLOSED_LOOP_BASELINE_MS = 305.5  # YOLOv8n target makespan from final_report.md

# Round 4 closed-loop placement summary from final_report.md:
# yolov8n on DSP (whole_net DLC), dronet split across CPU and DSP lanes.
# We assign 6 dronet to CPU and 6 to DSP for the baseline (matches the
# "CPU+DSP lanes" wording while remaining deterministic).
BASELINE_DRONET_DSP_COUNT = 6


def _require_matplotlib():
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


def _device_idx(name: str) -> int:
    return BACKENDS.index(name)


def _color_for_workload(idx: int, n: int):
    import matplotlib.pyplot as plt

    if idx == 0:
        return (0.86, 0.20, 0.20, 0.95)  # yolov8n -> red
    cmap = plt.get_cmap("viridis")
    # spread the 12 dronet instances across the colormap.
    t = (idx - 1) / max(n - 2, 1)
    r, g, b, _a = cmap(t)
    return (r, g, b, 0.92)


def _load_e2e_measurements() -> dict[str, dict[str, float]]:
    """Return ``{workload_id: {backend: mean_us}}`` from the E2E DLC runs."""

    raw = json.loads(E2E_MEASUREMENTS_PATH.read_text())
    out: dict[str, dict[str, float]] = {}
    for w_id, by_backend in raw.get("matrix", {}).items():
        out[w_id] = {b: float(rec["mean_us"]) for b, rec in by_backend.items() if rec.get("ok")}
    return out


# ---------------------------------------------------------------------------
# Baseline (whole_net) schedule via list scheduling on per-DLC placements.
# ---------------------------------------------------------------------------


def build_baseline_schedule() -> dict[str, Any]:
    """Per-DLC whole-net schedule: yolov8n on DSP + 12x dronet split CPU/DSP.

    Uses the E2E measured-DLC mean per backend (matches closed-loop's
    round-4 strategy). EFT list scheduler over a single device per
    partition; no transfer cost (each whole_net runs independently on its
    chosen lane).
    """

    e2e = _load_e2e_measurements()
    yolov_us = e2e["yolov8n"]["DSP"]
    dronet_cpu_us = e2e["dronet"]["CPU"]
    dronet_dsp_us = e2e["dronet"]["DSP"]

    partitions: list[dict[str, Any]] = []

    # yolov8n on DSP at t=0.
    partitions.append(
        {
            "partition_id": "yolov8n_whole",
            "workload": "yolov8n",
            "device": "DSP",
            "start_us": 0.0,
            "end_us": yolov_us,
            "duration_us": yolov_us,
        }
    )

    # 12 dronet split between CPU and DSP. EFT: schedule each instance on
    # its lane immediately after the lane becomes free.
    cpu_busy_until = 0.0
    dsp_busy_until = yolov_us  # DSP is held by yolov8n.

    # Round-robin first BASELINE_DRONET_DSP_COUNT to DSP, the rest to CPU,
    # so the visualization shows both lanes in flight.
    for i in range(N_DRONET_INSTANCES):
        if i < BASELINE_DRONET_DSP_COUNT:
            device = "DSP"
            dur = dronet_dsp_us
            start = dsp_busy_until
            dsp_busy_until = start + dur
        else:
            device = "CPU"
            dur = dronet_cpu_us
            start = cpu_busy_until
            cpu_busy_until = start + dur
        partitions.append(
            {
                "partition_id": f"dronet_{i}_whole",
                "workload": f"dronet_{i}",
                "device": device,
                "start_us": start,
                "end_us": start + dur,
                "duration_us": dur,
            }
        )

    makespan = max(p["end_us"] for p in partitions)
    return {
        "kind": "baseline_whole_net",
        "n_partitions": len(partitions),
        "makespan_us": makespan,
        "partitions": partitions,
    }


# ---------------------------------------------------------------------------
# Calibrated loop schedule: specialty chunking + joint CP-SAT.
# ---------------------------------------------------------------------------


def build_loop_schedule() -> dict[str, Any]:
    from xpu_rt.runtime.calibration import apply as apply_calibration
    from xpu_rt.runtime.calibration import load as load_calibration
    from xpu_rt.scheduler.qnn_real_workload import (
        BACKENDS as QNN_BACKENDS,
    )
    from xpu_rt.scheduler.qnn_real_workload import (
        load_cost_matrix,
        make_chain_dag,
    )
    from xpu_rt.scheduling.granularity import (
        compute_specialty_matrix,
        propose_chunks,
    )
    from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint

    raw_cost_matrix = load_cost_matrix(COST_MATRIX_PATH)
    calibration = load_calibration(CALIBRATION_PATH)
    calibrated = apply_calibration(calibration, raw_cost_matrix)
    overhead_per_backend = calibrated.get("_calibration_overhead_us", {})

    partition_ids: list[str] = []
    durations_by_dev: dict[str, list[float | None]] = {}
    dependencies: dict[str, list[str]] = {}
    workload_for_partition: dict[str, str] = {}

    def add_workload_chunks(workload_key: str, label: str) -> None:
        """Build chunks for a single workload and register them globally."""

        dag = make_chain_dag(workload_key, cost_matrix=calibrated)
        specialty = compute_specialty_matrix(calibrated, workload_key)
        plan = propose_chunks(
            dag,
            calibrated,
            workload_key,
            specialty,
            max_chunk_ops=16,
            max_partitions=200,
        )

        # Track first chunk to add per-backend overhead exactly once per
        # workload-lane (overhead is "DLC load + first-call warmup", which
        # is paid once per workload-on-a-backend).
        for chunk_idx, chunk in enumerate(plan.chunks):
            unique_id = f"{label}__{chunk.chunk_id}"
            partition_ids.append(unique_id)
            workload_for_partition[unique_id] = label
            row: list[float | None] = []
            for b in QNN_BACKENDS:
                v = chunk.durations_us_by_backend.get(b, math.inf)
                if not math.isfinite(v):
                    row.append(None)
                else:
                    base = float(v)
                    if chunk_idx == 0:
                        base += float(overhead_per_backend.get(b, 0.0))
                    row.append(base)
            durations_by_dev[unique_id] = row
            if chunk_idx == 0:
                dependencies[unique_id] = []
            else:
                dependencies[unique_id] = [
                    f"{label}__{plan.chunks[chunk_idx - 1].chunk_id}"
                ]

    add_workload_chunks("yolov8n", "yolov8n")
    for i in range(N_DRONET_INSTANCES):
        add_workload_chunks("dronet", f"dronet_{i}")

    # Cross-backend transfer matrix: reuse DEFAULT_TRANSFER_US (100us).
    from xpu_rt.scheduler.qnn_real_workload import DEFAULT_TRANSFER_US

    n_dev = len(QNN_BACKENDS)
    transfer = [[0.0] * n_dev for _ in range(n_dev)]
    for i in range(n_dev):
        for j in range(n_dev):
            if i != j:
                transfer[i][j] = float(DEFAULT_TRANSFER_US)

    print(
        f"[loop] {len(partition_ids)} partitions across {n_dev} devices; "
        "calling solve_schedule_joint(timeout_ms=30000)..."
    )
    t0 = time.perf_counter()
    sol = solve_schedule_joint(
        partition_ids=partition_ids,
        durations_us_by_device=durations_by_dev,
        dependencies=dependencies,
        num_devices=n_dev,
        transfer_us=transfer,
        timeout_ms=30000,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[loop] solver status={sol.status} feasible={sol.feasible} "
        f"makespan_us={sol.makespan_us:.1f} solve_wall_ms={wall_ms:.1f}"
    )

    if not sol.feasible:
        raise RuntimeError(
            f"calibrated loop CP-SAT call did not produce a feasible schedule: {sol.status}"
        )

    partitions: list[dict[str, Any]] = []
    for pid in partition_ids:
        device_idx = sol.device_assignments[pid]
        device = QNN_BACKENDS[device_idx]
        start = float(sol.start_times[pid])
        end = float(sol.end_times[pid])
        partitions.append(
            {
                "partition_id": pid,
                "workload": workload_for_partition[pid],
                "device": device,
                "start_us": start,
                "end_us": end,
                "duration_us": end - start,
            }
        )

    return {
        "kind": "loop_specialty_calibrated",
        "n_partitions": len(partitions),
        "makespan_us": float(sol.makespan_us),
        "solver_status": sol.status,
        "solver_wall_ms": wall_ms,
        "solver_reported_solve_time_ms": sol.solve_time_ms,
        "calibration_overhead_us": dict(overhead_per_backend),
        "partitions": partitions,
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _workload_index(name: str) -> int:
    if name == "yolov8n":
        return 0
    if name.startswith("dronet_"):
        return 1 + int(name.split("_", 1)[1])
    return -1


def render_gantt(
    baseline: dict[str, Any],
    loop: dict[str, Any],
    out_png: Path,
) -> tuple[int, int]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    workloads = ["yolov8n"] + [f"dronet_{i}" for i in range(N_DRONET_INSTANCES)]
    n_workloads = len(workloads)
    color_map = {w: _color_for_workload(_workload_index(w), n_workloads) for w in workloads}

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    closed_loop_marker_us = CLOSED_LOOP_BASELINE_MS * 1000.0
    max_makespan = max(baseline["makespan_us"], loop["makespan_us"], closed_loop_marker_us)
    xmax = max_makespan * 1.04

    for ax, schedule, title_suffix in (
        (axes[0], baseline, "baseline whole_net (closed-loop round-4 placement)"),
        (axes[1], loop, "calibrated loop (specialty chunks + CP-SAT joint)"),
    ):
        for p in schedule["partitions"]:
            y = _device_idx(p["device"])
            color = color_map.get(p["workload"], (0.5, 0.5, 0.5, 0.9))
            ax.broken_barh(
                [(p["start_us"], max(p["duration_us"], 1.0))],
                (y - 0.4, 0.8),
                facecolors=color,
                edgecolors="black",
                linewidth=0.3,
            )
        ax.set_yticks(list(range(len(BACKENDS))))
        ax.set_yticklabels(list(BACKENDS))
        ax.set_ylim(-0.7, len(BACKENDS) - 0.3)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("time (us)")
        ax.set_ylabel("device")
        makespan_ms = schedule["makespan_us"] / 1000.0
        ax.set_title(
            f"{title_suffix}  -  makespan={makespan_ms:.1f} ms  -  "
            f"{schedule['n_partitions']} partitions"
        )
        ax.axvline(
            closed_loop_marker_us,
            color="black",
            linestyle="--",
            alpha=0.55,
            linewidth=1.0,
            label=f"closed-loop target {CLOSED_LOOP_BASELINE_MS:.1f} ms",
        )
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    legend_handles = [Patch(facecolor=color_map[w], edgecolor="black", label=w) for w in workloads]
    legend_handles.append(
        Patch(facecolor="white", edgecolor="black", label=f"dashed = {CLOSED_LOOP_BASELINE_MS:.1f}ms target")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        "Closed-loop whole_net baseline vs feedback-loop calibrated schedule",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    width_px, height_px = fig.get_size_inches() * 120
    return int(width_px), int(height_px)


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------


def _per_device_busy_us(schedule: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {b: 0.0 for b in BACKENDS}
    for p in schedule["partitions"]:
        out[p["device"]] += float(p["duration_us"])
    return out


def write_summary(baseline: dict[str, Any], loop: dict[str, Any], out_md: Path) -> None:
    base_busy = _per_device_busy_us(baseline)
    loop_busy = _per_device_busy_us(loop)
    base_dev_count: dict[str, int] = {b: 0 for b in BACKENDS}
    loop_dev_count: dict[str, int] = {b: 0 for b in BACKENDS}
    for p in baseline["partitions"]:
        base_dev_count[p["device"]] += 1
    for p in loop["partitions"]:
        loop_dev_count[p["device"]] += 1

    lines: list[str] = []
    lines.append("# exp15 Gantt summary -- baseline vs feedback-loop schedule\n")
    lines.append("## Headline\n")
    lines.append(
        f"- baseline whole_net makespan: **{baseline['makespan_us'] / 1000:.1f} ms** "
        f"({baseline['n_partitions']} partitions)"
    )
    lines.append(
        f"- calibrated loop makespan: **{loop['makespan_us'] / 1000:.1f} ms** "
        f"({loop['n_partitions']} partitions)"
    )
    lines.append(f"- closed-loop reference target: {CLOSED_LOOP_BASELINE_MS:.1f} ms")
    lines.append(
        f"- CP-SAT solver: status={loop.get('solver_status')!r}, "
        f"wall={loop.get('solver_wall_ms', 0.0):.1f} ms "
        f"(internal {loop.get('solver_reported_solve_time_ms', 0.0):.1f} ms)"
    )
    overhead = loop.get("calibration_overhead_us", {})
    if overhead:
        ov = ", ".join(f"{k}={v:.0f}us" for k, v in overhead.items())
        lines.append(f"- calibration overhead per backend: {ov}")
    lines.append("")

    lines.append("## Per-device busy time (us)\n")
    lines.append("| device | baseline busy | loop busy | baseline #parts | loop #parts |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in BACKENDS:
        lines.append(
            f"| {b} | {base_busy[b]:.0f} | {loop_busy[b]:.0f} | "
            f"{base_dev_count[b]} | {loop_dev_count[b]} |"
        )
    lines.append("")

    lines.append("## What changed visually\n")
    base_uses_gpu = base_dev_count["GPU"] > 0
    loop_uses_gpu = loop_dev_count["GPU"] > 0
    notes: list[str] = []
    if loop_uses_gpu and not base_uses_gpu:
        notes.append(
            "The loop activated the GPU lane that the baseline left idle (whole_net DLCs "
            "had no GPU placement). Specialty-chunked partitions surface families that "
            "argmin to GPU and the joint scheduler routes them there."
        )
    elif base_dev_count["DSP"] > 0 and loop_dev_count["DSP"] < base_dev_count["DSP"]:
        notes.append(
            "The loop pulled work off the DSP lane (which the baseline serialized) and "
            "scattered chunks across CPU/GPU, exposing intra-workload parallelism the "
            "whole_net partitioning could not."
        )
    notes.append(
        f"Baseline serializes yolov8n + 6 dronet on DSP (~{base_busy['DSP'] / 1000:.0f} ms "
        f"busy); the loop's per-chunk placement balances to "
        f"{loop_busy['DSP'] / 1000:.0f} / {loop_busy['CPU'] / 1000:.0f} / "
        f"{loop_busy['GPU'] / 1000:.0f} ms (DSP/CPU/GPU)."
    )
    diff_ms = (baseline["makespan_us"] - loop["makespan_us"]) / 1000.0
    if diff_ms > 0:
        notes.append(
            f"Net: loop makespan is {diff_ms:.1f} ms shorter than the baseline."
        )
    else:
        notes.append(
            f"Net: loop makespan is {-diff_ms:.1f} ms LONGER than baseline -- transfer or "
            "calibration overhead dominates the chunk-level parallelism gain at this scale."
        )
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    _require_matplotlib()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[exp15] output dir: {OUT_DIR}")
    print(f"[exp15] cost matrix: {COST_MATRIX_PATH}")
    print(f"[exp15] calibration: {CALIBRATION_PATH}")

    baseline = build_baseline_schedule()
    print(
        f"[baseline] {baseline['n_partitions']} partitions, "
        f"makespan_ms={baseline['makespan_us'] / 1000:.1f}"
    )

    loop = build_loop_schedule()
    print(
        f"[loop] {loop['n_partitions']} partitions, "
        f"makespan_ms={loop['makespan_us'] / 1000:.1f}"
    )

    (OUT_DIR / "baseline_schedule.json").write_text(json.dumps(baseline, indent=2))
    (OUT_DIR / "loop_schedule.json").write_text(json.dumps(loop, indent=2))

    out_png = OUT_DIR / "baseline_vs_loop.png"
    w, h = render_gantt(baseline, loop, out_png)
    print(f"[plot] wrote {out_png} (~{w}x{h} px)")

    summary_path = OUT_DIR / "summary.md"
    write_summary(baseline, loop, summary_path)
    print(f"[summary] wrote {summary_path}")

    print("[exp15] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
