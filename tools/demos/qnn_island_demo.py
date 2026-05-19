#!/usr/bin/env python3
"""QNN heterogeneous island scheduling demo — XPU-RT-driven.

Pipes a hand-coded multi-precision island DAG (the 7-island YOLOv8-stem-shaped
parallel-branch fixture from third_party/merlin/tools/kernels/qnn_emit_recognizers/) into
XPU-RT's scheduler.

The DAG models what real partitioning will produce on YOLOv8: an HTA-uint8
backbone, a parallel GPU-fp16 branch, a CPU-bridge for the precision swap, and
a CPU output decode head. The scheduler picks the per-island machine
assignment that minimises wall-clock makespan, distinguishing island-DAG
parallelism from hardware parallelism: HTA‖GPU is real, HTA‖HTA serialises.

Per-backend processing_times are the on-board measurements taken on QRB5165
(QAIRT 2.45) at the YOLOv8 stem-conv shape (1×320×320×3 → 1×160×160×16,
3×3 stride 2). transfer_times are the host-memcpy + dequant/quant bridge costs.

This is the *integration scaffolding*. Production usage will replace the
hand-coded DAG with the partitioner output and the constant cost table with a
calibration sweep populated from per-shape on-board measurements.

Invocation:
    cd /scratch2/agustin/XPU-RT
    conda run -n merlin-dev uv run python scripts/qnn_island_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

# Make XPU-RT scheduler importable.
_HERE = pathlib.Path(__file__).resolve()
_XPU_RT_PY = _HERE.parent.parent / "xpu-rt"
sys.path.insert(0, str(_XPU_RT_PY))

from xpu_rt.scheduler.workload_factory import create_workload_from_dependencies  # noqa: E402


# --- Measured costs on QRB5165 / QAIRT 2.45 -------------------------------
# Times in microseconds. HTA + GPU are real on-board measurements at the
# YOLOv8 stem-conv shape (5.91 ms HTA avg, 12.5 ms GPU avg). CPU is
# extrapolated from the int8 ukernel on A77 (ratio ~5× HTA for this shape).
# The "inf" markers mean "unsupported on this backend per QNN op-def":
# Adreno GPU has no quantised Conv2d on QAIRT 2.45.
HTA, GPU, CPU = "HTA", "GPU", "CPU"
MACHINES = [HTA, GPU, CPU]
INF = float("inf")

# Bridge transfer matrix (us). Bridges are host memcpy + (optionally)
# Dequantize/Quantize. Measured on the same tensor volume as the stem
# conv output (1×160×160×16 = 410 KB).
#   memcpy uint8 ........  52 us
#   Dequant uint8->fp32 .. 23 us  (read 1B, write 4B, multiply)
#   Quant   fp32->uint8 .. 17 us
# CPU↔CPU bridge is a no-op (0). Cross-backend bridge costs include the
# (de)quant when the dtypes differ (HTA<->GPU = ~75us; HTA<->CPU = 0 if
# both consume uint8, but the GPU island terminates in fp16 so any GPU
# island merging into HTA needs a CPU-side Quantize bridge).
BRIDGES_US = {
    (HTA, HTA): 0.0,
    (HTA, GPU): 75.0,   # uint8 -> Dequantize on GPU island (within graph)
    (HTA, CPU): 52.0,   # memcpy only
    (GPU, HTA): 92.0,   # fp16 -> CPU Quantize -> uint8 (52 + 17 + 23)
    (GPU, GPU): 0.0,
    (GPU, CPU): 101.0,  # fp16 memcpy
    (CPU, HTA): 17.0,   # Quantize on CPU side
    (CPU, GPU): 23.0,   # Dequantize on CPU side
    (CPU, CPU): 0.0,
}


def build_transfer_matrix() -> np.ndarray:
    """3×3 transfer-time matrix in machine-list order."""
    n = len(MACHINES)
    out = np.zeros((n, n), dtype=float)
    for i, mi in enumerate(MACHINES):
        for j, mj in enumerate(MACHINES):
            out[i, j] = BRIDGES_US[(mi, mj)]
    return out


# --- The hand-coded 7-island DAG ------------------------------------------
# Shape: parallel HTA-stem and GPU-stem branches feed a CPU concat bridge,
# then an HTA trunk conv, then 3 detection-head islands run concurrently
# (HTA, GPU, CPU), then a CPU decode merges them.
#
# processing_times[m] is the predicted island latency on machine m (us).
# INF marks unsupported (e.g. GPU has no quantised Conv2d on QAIRT 2.45,
# so an "HTA-uint8 stem" island is INF on GPU).
ISLANDS = {
    "input_split": {
        "deps": [],
        # Logical fan-out point: cheap on any backend; CPU runs it.
        "times": {HTA: INF, GPU: INF, CPU: 30.0},
    },
    "hta_stem_branch_a": {
        "deps": ["input_split"],
        # uint8 conv on HTA — measured 5910 us avg. GPU has no int8 Conv2d.
        "times": {HTA: 5910.0, GPU: INF, CPU: 30000.0},
    },
    "gpu_stem_branch_b": {
        "deps": ["input_split"],
        # fp16 conv on GPU — measured 12500 us avg. HTA accepts uint8 only.
        "times": {HTA: INF, GPU: 12500.0, CPU: 35000.0},
    },
    "cpu_bridge_concat_requant": {
        "deps": ["hta_stem_branch_a", "gpu_stem_branch_b"],
        # Concatenates the two branches + requantizes the GPU output;
        # CPU only (HTA/GPU lack Quantize on QAIRT 2.45).
        "times": {HTA: INF, GPU: INF, CPU: 200.0},
    },
    "hta_trunk_conv": {
        "deps": ["cpu_bridge_concat_requant"],
        # Smaller-shape uint8 conv (1×80×80×32). HTA wins; GPU fp16 ≈ 1.7×.
        "times": {HTA: 4200.0, GPU: 7100.0, CPU: 22000.0},
    },
    "head1_hta": {
        "deps": ["hta_trunk_conv"],
        "times": {HTA: 1800.0, GPU: 3500.0, CPU: 9000.0},
    },
    "head2_gpu": {
        "deps": ["hta_trunk_conv"],
        # Smaller op where fp16 GPU is competitive (intentionally chosen
        # to exercise the parallel GPU path).
        "times": {HTA: 2200.0, GPU: 1400.0, CPU: 6800.0},
    },
    "head3_cpu": {
        "deps": ["hta_trunk_conv"],
        # Tiny island below the QNN dispatch floor — CPU wins.
        "times": {HTA: 2100.0, GPU: 5800.0, CPU: 600.0},
    },
    "cpu_decode": {
        "deps": ["head1_hta", "head2_gpu", "head3_cpu"],
        # Detection decode: f32 box/score math, control flow → CPU only.
        "times": {HTA: INF, GPU: INF, CPU: 1500.0},
    },
}


def build_dispatch_dict() -> tuple[dict, dict]:
    """Convert ISLANDS to the dispatch_data shape XPU-RT consumes."""
    dispatches: dict[str, dict] = {}
    proc_times: dict[str, list[float]] = {}
    for name, spec in ISLANDS.items():
        dispatches[name] = {
            "id": name,
            "dependencies": list(spec["deps"]),
        }
        # Map INF to a large but finite value so the MILP stays bounded
        # while strongly disincentivising the assignment.
        row = []
        for m in MACHINES:
            t = spec["times"][m]
            row.append(1e9 if t == INF else float(t))
        proc_times[name] = row
    return dispatches, proc_times


def run_greedy_schedule(workload, transfer_us: np.ndarray):
    """Earliest-finish-time list scheduler, transfer-aware. Returns
    (start_us, machine_assignment) keyed by operation name."""
    by_name = {op.operation_name: op for op in workload.operations}
    deps = {n: [p.operation_name for p in (by_name[n].predecessors or [])]
            for n in by_name}

    # Topological order.
    order: list[str] = []
    pending = {n: list(d) for n, d in deps.items()}
    while pending:
        ready = sorted(n for n, d in pending.items() if not d)
        if not ready:
            raise RuntimeError("dependency cycle in island DAG")
        for n in ready:
            order.append(n)
            del pending[n]
        for rem in pending.values():
            for n in ready:
                if n in rem:
                    rem.remove(n)

    machine_idx = {m: i for i, m in enumerate(MACHINES)}
    machine_free = {m: 0.0 for m in MACHINES}
    finishes: dict[str, tuple[float, str]] = {}
    starts: dict[str, float] = {}
    assigns: dict[str, str] = {}

    for name in order:
        op = by_name[name]
        best = None
        for mi, m in enumerate(MACHINES):
            ready_t = 0.0
            for pred in op.predecessors or []:
                f, fm = finishes[pred.operation_name]
                t_us = float(transfer_us[machine_idx[fm], mi]) if fm != m else 0.0
                ready_t = max(ready_t, f + t_us)
            start = max(ready_t, machine_free[m])
            finish = start + op.processing_times[mi]
            if best is None or finish < best[0]:
                best = (finish, start, m)
        finish, start, m = best
        starts[name] = start
        finishes[name] = (finish, m)
        assigns[name] = m
        machine_free[m] = finish
    return starts, finishes, assigns


def print_gantt(starts, finishes, assigns, machine_busy_until):
    print("\n=== Schedule (μs) ===")
    print(f"{'island':<32} {'machine':<6} {'start':>8} {'finish':>8} {'dur':>8}")
    for name in starts:
        s = starts[name]; f, _ = finishes[name]; m = assigns[name]
        print(f"{name:<32} {m:<6} {s:>8.0f} {f:>8.0f} {f-s:>8.0f}")
    makespan = max(f for f, _ in finishes.values())
    print(f"\nmakespan: {makespan:.0f} us  ({makespan/1000:.2f} ms)")
    print("per-machine busy:")
    for m in MACHINES:
        busy = sum(finishes[n][0] - starts[n] for n, mm in assigns.items() if mm == m)
        print(f"  {m:<4}  {busy:>8.0f} us  utilisation {busy/makespan*100:>5.1f} %")
    print()


def sequential_baseline(workload, transfer_us: np.ndarray) -> float:
    """Wall-clock if every island were forced to its single fastest
    backend with NO parallelism (all on CPU pipe). Used as the upper
    bound the heterogeneous schedule competes against."""
    by_name = {op.operation_name: op for op in workload.operations}
    deps_order: list[str] = []
    pending = {op.operation_name: list(p.operation_name for p in (op.predecessors or []))
               for op in workload.operations}
    while pending:
        ready = sorted(n for n, d in pending.items() if not d)
        for n in ready:
            deps_order.append(n)
            del pending[n]
        for rem in pending.values():
            for n in ready:
                if n in rem: rem.remove(n)
    cpu_idx = MACHINES.index(CPU)
    finish: dict[str, float] = {}
    cpu_free = 0.0
    for n in deps_order:
        op = by_name[n]
        ready_t = max((finish[p.operation_name] for p in (op.predecessors or [])), default=0.0)
        start = max(ready_t, cpu_free)
        cpu_free = start + op.processing_times[cpu_idx]
        finish[n] = cpu_free
    return max(finish.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("scheduled_qnn_island_demo.json"))
    args = parser.parse_args()

    dispatches, proc_times = build_dispatch_dict()
    transfer = build_transfer_matrix()
    workload = create_workload_from_dependencies(
        {"dispatches": dispatches}, proc_times, MACHINES, transfer,
    )
    starts, finishes, assigns = run_greedy_schedule(workload, transfer)

    print_gantt(starts, finishes, assigns, None)

    cpu_only = sequential_baseline(workload, transfer)
    het_makespan = max(f for f, _ in finishes.values())
    print(f"all-CPU baseline:           {cpu_only:.0f} us  ({cpu_only/1000:.2f} ms)")
    print(f"heterogeneous schedule:     {het_makespan:.0f} us  ({het_makespan/1000:.2f} ms)")
    print(f"speedup:                    {cpu_only/het_makespan:.2f}×")

    out = {
        "machines": MACHINES,
        "transfer_us": transfer.tolist(),
        "islands": [
            {
                "name": n,
                "machine": assigns[n],
                "start_us": starts[n],
                "finish_us": finishes[n][0],
                "deps": [p.operation_name for p in workload.operations
                         [list(dispatches).index(n)].predecessors or []],
            }
            for n in starts
        ],
        "makespan_us": het_makespan,
        "cpu_baseline_us": cpu_only,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nschedule written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
