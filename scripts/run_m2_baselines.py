"""
M2 driver: run all six fast list-scheduling baselines on a realistic-but-
synthetic heterogeneous workload and produce per-scheduler Gantts plus a
side-by-side composite.

The workload models a small robotic stack:
    - 3 jobs (perception, planning, control)
    - 22 operations total with chains + parallelism
    - 4 machine combinations: CPU_BIG, CPU_LITTLE, NPU, CPU_BIG+CPU_LITTLE
    - Per-op processing times vary across machines (NPU faster for matmul-like
      ops; preprocessing infeasible on NPU; control prefers CPU_BIG)
    - A handful of ops carry release-time windows and deadlines
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from workload import Operation, Workload
from schedulers import get_scheduler
from metrics import compute_metrics
from report import SchedulerResult, render_gantt, render_side_by_side, write_markdown_report
from postprocessing import validate_schedule, output_scheduled_json


MACHINES = ["CPU_BIG", "CPU_LITTLE", "NPU"]
COMBINATIONS = [
    ["CPU_BIG"],            # 0
    ["CPU_LITTLE"],         # 1
    ["NPU"],                # 2
    ["CPU_BIG", "CPU_LITTLE"],  # 3 (cooperative CPU pair)
]
N = len(MACHINES)
TRANSFER = np.array([
    # rows/cols ordered as MACHINES
    [0.0, 5.0, 30.0],
    [5.0, 0.0, 30.0],
    [30.0, 30.0, 0.0],
])

# Op categories drive per-combination cost vectors.
# Each list is indexed by COMBINATIONS position (0..3) in microseconds.
COST_BY_KIND = {
    # preprocess: scalar code → CPU_BIG decent, CPU_LITTLE okay, NPU infeasible,
    # paired-CPU close to CPU_BIG (no useful split for scalar work)
    "preprocess": [120.0, 200.0, np.inf, 110.0],
    # matmul: NPU dominates; paired-CPU gets ~1.6x over CPU_BIG via vector splits
    "matmul":     [800.0, 1400.0, 250.0, 500.0],
    # conv: NPU dominates but with launch overhead; CPU pair beats CPU_BIG modestly
    "conv":       [600.0, 1000.0, 320.0, 420.0],
    # postprocess: scalar; NPU infeasible
    "postprocess":[150.0, 250.0, np.inf, 130.0],
    # control: latency-critical scalar; CPU_BIG strongly preferred, paired hurts
    "control":    [80.0, 140.0, np.inf, 100.0],
}


def _make_op(name: str, kind: str, *, preds=None, deadline_us=None, min_start_t=None):
    costs = list(COST_BY_KIND[kind])
    infeasible = {k for k, c in enumerate(costs) if c == np.inf or c is np.inf}
    # Replace np.inf with a huge but finite stand-in so processing_times stays numeric.
    costs = [1e9 if c == np.inf else c for c in costs]
    return Operation(
        processing_times=costs,
        predecessors=preds or [],
        operation_name=name,
        deadline_us=deadline_us,
        min_start_t=min_start_t,
        infeasible_combinations=infeasible,
    )


def build_workload():
    # ---- Job 0: perception pipeline (chain of pre → 2x conv (parallel) → fuse → matmul → post)
    p0 = _make_op("percep_pre", "preprocess")
    c0a = _make_op("percep_conv_a", "conv", preds=[p0])
    c0b = _make_op("percep_conv_b", "conv", preds=[p0])
    c0c = _make_op("percep_conv_c", "conv", preds=[c0a])
    f0 = _make_op("percep_fuse", "postprocess", preds=[c0b, c0c])
    mm0 = _make_op("percep_matmul", "matmul", preds=[f0])
    pp0 = _make_op("percep_post", "postprocess", preds=[mm0], deadline_us=4500.0)

    # ---- Job 1: planning (smaller; depends on perception output)
    p1 = _make_op("plan_pre", "preprocess", preds=[pp0])
    m1a = _make_op("plan_mm_a", "matmul", preds=[p1])
    m1b = _make_op("plan_mm_b", "matmul", preds=[p1])
    f1 = _make_op("plan_fuse", "postprocess", preds=[m1a, m1b])
    pp1 = _make_op("plan_post", "postprocess", preds=[f1], deadline_us=7500.0)

    # ---- Job 2: control loop (latency-critical; runs in parallel with perception,
    #             must finish by t=1500us — tighter than perception)
    cp0 = _make_op("ctrl_in", "control", min_start_t=0.0)
    cp1 = _make_op("ctrl_calc", "control", preds=[cp0])
    cp2 = _make_op("ctrl_out", "control", preds=[cp1], deadline_us=1500.0)

    # ---- A few extra parallel "background" matmuls to add scheduling headroom
    bg0 = _make_op("bg_mm_0", "matmul")
    bg1 = _make_op("bg_mm_1", "matmul", preds=[bg0])
    bg2 = _make_op("bg_conv_0", "conv")
    bg3 = _make_op("bg_conv_1", "conv", preds=[bg2])
    bg4 = _make_op("bg_post_0", "postprocess", preds=[bg1, bg3])

    operations = [
        p0, c0a, c0b, c0c, f0, mm0, pp0,
        p1, m1a, m1b, f1, pp1,
        cp0, cp1, cp2,
        bg0, bg1, bg2, bg3, bg4,
    ]
    # 20 ops. Job ids: 0 (perception), 1 (planning), 2 (control), 3 (background).
    for op in [p0, c0a, c0b, c0c, f0, mm0, pp0]:
        op.job_id = 0
    for op in [p1, m1a, m1b, f1, pp1]:
        op.job_id = 1
    for op in [cp0, cp1, cp2]:
        op.job_id = 2
    for op in [bg0, bg1, bg2, bg3, bg4]:
        op.job_id = 3

    wl = Workload(
        operations,
        MACHINES,
        TRANSFER,
        job_names=["perception", "planning", "control", "background"],
        machine_combinations=COMBINATIONS,
    )
    return wl


SCHEDULERS = ["mosek", "heft", "critical_path", "edf", "fastest_device", "fifo", "random_list"]


def main():
    out_dir = REPO / "plots" / "m2_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir = REPO / "schedules" / "m2_baselines"
    schedules_dir.mkdir(parents=True, exist_ok=True)

    wl = build_workload()
    print(f"Built workload: {len(wl.operations)} ops, {len(wl.machine_combinations)} combos, "
          f"{len(wl.machines)} machines")

    results = []
    summary = {}
    for name in SCHEDULERS:
        print(f"\n--- {name} ---")
        sched = get_scheduler(name)
        kwargs = {}
        if name == "mosek":
            kwargs = dict(solver_verbosity=0, time_limit=10,
                          restrict_makespan_to_nonperiodic=True,
                          prune_cross_period_constraints=False)
        elif name == "random_list":
            kwargs = dict(random_seed=42)
        t0 = time.perf_counter()
        try:
            t, alpha, _, _ = sched(wl, **kwargs)
            wall = time.perf_counter() - t0
            ok, _details = validate_schedule(wl, t, alpha, original_json_data={"dispatches": {}})
        except Exception as exc:
            print(f"  scheduler {name} failed: {exc}")
            results.append(SchedulerResult(scheduler_name=name, feasible=False, note=str(exc)))
            continue

        m = compute_metrics(wl, t, alpha, scheduler_name=name, solver_wall_time_s=wall)
        print(f"  makespan={m['makespan_us']:.1f} us  "
              f"misses={m['deadline_miss_count']}  "
              f"util(CPU_BIG)={m['per_machine_utilization']['CPU_BIG']:.2f}  "
              f"util(NPU)={m['per_machine_utilization']['NPU']:.2f}  "
              f"valid={ok}  solver_s={wall:.3f}")

        gantt_path = str(out_dir / f"gantt_{name}.png")
        schedule_json_path = str(schedules_dir / f"scheduled_m2_synthetic_{name}.json")
        output_scheduled_json(wl, t, alpha, schedule_json_path)
        res = SchedulerResult(
            scheduler_name=name, workload=wl, t=t, alpha=alpha,
            metrics=m, feasible=True, schedule_json_path=schedule_json_path,
        )
        render_gantt(res, gantt_path, title=f"{name} schedule (makespan={m['makespan_us']:.0f}us)")
        results.append(res)
        # Sibling metrics JSON.
        metrics_json_path = schedule_json_path.replace(".json", "_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(m, f, indent=2)
        summary[name] = {
            "makespan_us": m["makespan_us"],
            "deadline_miss_count": m["deadline_miss_count"],
            "total_lateness_us": m["total_lateness_us"],
            "cross_device_transitions": m["cross_device_transitions"],
            "critical_path_us": m["critical_path_us"],
            "per_machine_utilization": m["per_machine_utilization"],
            "solver_wall_time_s": m["solver_wall_time_s"],
            "valid": True,
            "gantt": gantt_path,
        }

    # Side-by-side composite + markdown report.
    composite = str(out_dir / "side_by_side.png")
    render_side_by_side(results, composite, title="M2 baselines on synthetic heterogeneous workload")
    report_md = str(out_dir / "report.md")
    write_markdown_report(results, report_md,
                          title="M2 baselines — synthetic heterogeneous workload",
                          side_by_side_png=composite)

    # Headline summary JSON.
    headline = {
        "workload": {
            "num_ops": len(wl.operations),
            "machines": wl.machines,
            "machine_combinations": [list(c) for c in wl.machine_combinations],
        },
        "per_scheduler": summary,
        "heft_makespan_le_fifo_makespan": summary.get("heft", {}).get("makespan_us", float("inf"))
            <= summary.get("fifo", {}).get("makespan_us", float("inf")),
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(headline, f, indent=2)

    print("\n=== Headline ===")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_machine_utilization"}
                      for k, v in summary.items()}, indent=2))
    print(f"\nWrote: {report_md}")
    print(f"Composite: {composite}")
    print(f"HEFT makespan <= FIFO makespan: {headline['heft_makespan_le_fifo_makespan']}")


if __name__ == "__main__":
    main()
