"""
M23 — Stress-test scenarios that push the schedulers HARD.

Five scenario families:

  1. dominator_packing
       One big model (yolov8n) defines the period envelope at a slow rate
       (1 Hz). Pack many smaller models (dronet/mlp/control) at their
       natural rates inside that envelope. Tests "fit small jobs in the
       gaps/parallelism of a dominant model."

  2. multi_granularity_dronet
       The SAME workload (dronet) is presented to the scheduler at three
       granularities concurrently:
         coarse: 1 fused op (cost = sum of all dispatches)
         medium: 3 ops (preprocess / main / postprocess; sum of subranges)
         fine  : full 15-op dispatch graph
       Tests whether finer granularity helps schedulers (the answer is
       'yes, until dispatch overhead dominates').

  3. mixed_size_stack
       5 models at their native granularity AND natural frequencies in a
       tight envelope. yolov8n (48) + dronet (15) + mlp (3) + control (3)
       + monitor (2). Tests realistic multi-model heterogeneity.

  4. solver_killer
       ~250 ops, tight deadlines, deep dependency chains, multi-rate
       mix. Designed to push CP-SAT past its tractable region and force
       MOSEK to time out. Shows where heuristics matter most.

  5. frequency_sweep_breaking_point
       Same workload, sweep ONE model's frequency upward. Find each
       scheduler's "breaking point" — the Hz beyond which it starts
       missing deadlines.

Built on real-data anchors (dronet / mlp_wide / yolov8n graphs +
qrb5165_costs.json measurements). Pseudo-grounded in reality.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload
from scenarios import MACHINES, COMBOS, TRANSFER


# ============================================================================
# Granularity helpers
# ============================================================================


def _remap_3machine(base_wl) -> List[Operation]:
    """Convert a multi-backend Workload (scalar/rvv/opu/gemmini or
    CPU/GPU/HTA) to our 3-machine SoC. Returns the list of new ops with
    rewired predecessors."""
    name_map = {m: i for i, m in enumerate(base_wl.machines)}
    cpu_idxs = [name_map[m] for m in base_wl.machines
                if m.lower() in ("cpu", "scalar", "rvv", "cpu_big", "cpu_little")]
    gpu_idxs = [name_map[m] for m in base_wl.machines
                if m.lower() in ("gpu", "opu")] or cpu_idxs
    npu_idxs = [name_map[m] for m in base_wl.machines
                if m.lower() in ("npu", "hta", "gemmini")] or gpu_idxs

    out_ops = []
    idx_of = {id(o): i for i, o in enumerate(base_wl.operations)}
    for src in base_wl.operations:
        pts = list(src.processing_times)
        cpu = min(pts[i] for i in cpu_idxs) if cpu_idxs else pts[0]
        gpu = min(pts[i] for i in gpu_idxs) if gpu_idxs else cpu
        npu = min(pts[i] for i in npu_idxs) if npu_idxs else gpu
        op = Operation(
            processing_times=[float(cpu), float(gpu), float(npu)],
            operation_name=src.operation_name,
            infeasible_combinations=set(),
        )
        op.output_bytes = getattr(src, "output_bytes", 0)
        out_ops.append(op)
    for i, src in enumerate(base_wl.operations):
        for p in src.get_predecessors():
            if id(p) in idx_of:
                out_ops[i].add_predecessor(out_ops[idx_of[id(p)]])
    return out_ops


def coarse_op(model: str, soc: str = "qrb5165",
              tag: str = "coarse") -> List[Operation]:
    """ONE op representing the whole model. Costs = sum of all dispatch costs
    on the locally-fastest machine, plus a small launch overhead."""
    from realistic_workloads import build_model_graph, build_workload_from_graph
    g = build_model_graph(model, soc)
    base = build_workload_from_graph(g)
    ops = _remap_3machine(base)
    if not ops:
        return []
    # Sum costs per machine.
    sums = [sum(op.processing_times[k] for op in ops) for k in range(3)]
    fused = Operation(
        processing_times=sums,
        operation_name=f"{model}_{tag}_fused",
        infeasible_combinations=set(),
    )
    fused.output_bytes = sum(getattr(op, "output_bytes", 0) for op in ops)
    return [fused]


def medium_op(model: str, soc: str = "qrb5165",
              n_stages: int = 3, tag: str = "med") -> List[Operation]:
    """``n_stages`` ops by grouping the model's dispatches into roughly-equal
    chunks. Cost of each stage = sum over its dispatches; predecessors form a
    chain across stages."""
    from realistic_workloads import build_model_graph, build_workload_from_graph
    g = build_model_graph(model, soc)
    base = build_workload_from_graph(g)
    ops = _remap_3machine(base)
    if not ops:
        return []
    if len(ops) <= n_stages:
        return ops  # already at or below medium granularity
    chunk = (len(ops) + n_stages - 1) // n_stages
    stages: List[Operation] = []
    for s in range(n_stages):
        sub = ops[s * chunk: (s + 1) * chunk]
        if not sub:
            continue
        sums = [sum(op.processing_times[k] for op in sub) for k in range(3)]
        stage = Operation(
            processing_times=sums,
            operation_name=f"{model}_{tag}_s{s}",
            infeasible_combinations=set(),
        )
        stage.output_bytes = sum(getattr(op, "output_bytes", 0) for op in sub)
        if stages:
            stage.add_predecessor(stages[-1])
        stages.append(stage)
    return stages


def fine_ops(model: str, soc: str = "qrb5165") -> List[Operation]:
    """Full dispatch graph (existing reconstruction)."""
    from realistic_workloads import build_model_graph, build_workload_from_graph
    g = build_model_graph(model, soc)
    base = build_workload_from_graph(g)
    return _remap_3machine(base)


# ============================================================================
# Periodic packing helper (shared with M17/M18)
# ============================================================================


def _pack_instances(role_specs: List[Tuple[str, float, str]],
                    envelope_us: float,
                    soc: str = "qrb5165") -> Workload:
    """Pack instances of (role, hz, granularity) into a Workload.

    granularity ∈ {"coarse", "medium", "fine", "control_chain", "monitor",
                   "planning_tail"}.
    """
    all_ops: List[Operation] = []
    job_names: List[str] = []
    job_id = 0

    def _builder(role: str, gran: str) -> List[Operation]:
        if gran == "coarse":
            return [_dup_op(o) for o in coarse_op(role, soc=soc, tag="coarse")]
        if gran == "medium":
            return [_dup_op(o) for o in medium_op(role, soc=soc, n_stages=3, tag="med")]
        if gran == "fine":
            return [_dup_op(o) for o in fine_ops(role, soc=soc)]
        if gran == "control_chain":
            # 3 small CPU-only ops at realistic elementwise cost
            ops = [
                Operation(processing_times=[260.0, 400.0, 600.0],
                          operation_name="ctrl_in", infeasible_combinations={2}),
                Operation(processing_times=[260.0, 400.0, 600.0],
                          operation_name="ctrl_calc", infeasible_combinations={2}),
                Operation(processing_times=[260.0, 400.0, 600.0],
                          operation_name="ctrl_out", infeasible_combinations={2}),
            ]
            ops[1].add_predecessor(ops[0])
            ops[2].add_predecessor(ops[1])
            return ops
        if gran == "monitor":
            ops = [
                Operation(processing_times=[1500.0, 800.0, 300.0],
                          operation_name="monitor_check"),
                Operation(processing_times=[200.0, 250.0, 400.0],
                          operation_name="monitor_log",
                          infeasible_combinations={2}),
            ]
            ops[1].add_predecessor(ops[0])
            return ops
        if gran == "planning_tail":
            # Last 15 ops of yolov8n's graph treated as a planning network
            full = fine_ops("yolov8n", soc=soc)
            tail = full[-15:]
            # Rewire: drop dangling predecessors
            old_to_new = {id(o): i for i, o in enumerate(tail)}
            for op in tail:
                op.predecessors = [p for p in op.predecessors if id(p) in old_to_new]
            return tail
        raise ValueError(f"unknown granularity: {gran}")

    for role, hz, gran in role_specs:
        period = 1e6 / hz
        n_inst = max(1, int(np.floor(envelope_us / period)))
        for inst in range(n_inst):
            release = inst * period
            deadline = (inst + 1) * period
            base = _builder(role, gran)
            if not base:
                continue
            # Identify sink(s).
            base_idx = {id(op): i for i, op in enumerate(base)}
            has_succ = set()
            for op in base:
                for p in op.get_predecessors():
                    pi = base_idx.get(id(p))
                    if pi is not None:
                        has_succ.add(pi)
            sink_ids = [i for i in range(len(base)) if i not in has_succ]

            job_names.append(f"{role}_{gran}_{hz:g}Hz_i{inst}")
            inst_ops = []
            for i, src in enumerate(base):
                op = Operation(
                    processing_times=list(src.processing_times),
                    operation_name=f"{role}_{gran}_{hz:g}Hz_i{inst}_{src.operation_name}",
                    infeasible_combinations=set(src.infeasible_combinations),
                    min_start_t=release,
                    deadline_us=deadline if i in sink_ids else None,
                )
                op.output_bytes = getattr(src, "output_bytes", 0)
                op.job_id = job_id
                inst_ops.append(op)
            for i, src in enumerate(base):
                for p in src.get_predecessors():
                    pi = base_idx.get(id(p))
                    if pi is not None:
                        inst_ops[i].add_predecessor(inst_ops[pi])
            all_ops.extend(inst_ops)
            job_id += 1

    return Workload(all_ops, MACHINES, np.array(TRANSFER),
                    job_names=job_names, machine_combinations=COMBOS)


def _dup_op(op: Operation) -> Operation:
    new = Operation(
        processing_times=list(op.processing_times),
        operation_name=op.operation_name,
        infeasible_combinations=set(op.infeasible_combinations),
    )
    new.output_bytes = getattr(op, "output_bytes", 0)
    return new


# ============================================================================
# Stress scenarios
# ============================================================================


def dominator_packing(soc: str = "qrb5165") -> Workload:
    """yolov8n at 1 Hz defines the envelope. dronet@5Hz + mlp@30Hz +
    control@50Hz packed inside. Tests fitting many small jobs in a big
    one's makespan window."""
    from realistic_workloads import e2e_envelope
    # yolov8n at 1 Hz means the envelope is 1 second
    envelope_us = 1_000_000.0
    return _pack_instances([
        ("yolov8n",  1, "fine"),
        ("dronet",   5, "fine"),
        ("mlp_wide", 30, "fine"),
        ("control",  50, "control_chain"),
    ], envelope_us, soc=soc)


def multi_granularity_dronet(soc: str = "qrb5165") -> Workload:
    """Same model (dronet) packed at THREE granularities concurrently.
    Tests whether finer granularity helps the scheduler."""
    envelope_us = 200_000.0  # 200ms envelope
    return _pack_instances([
        ("dronet",   2, "coarse"),
        ("dronet",   2, "medium"),
        ("dronet",   2, "fine"),
        ("control", 20, "control_chain"),  # extra workload so the choice matters
    ], envelope_us, soc=soc)


def mixed_size_stack(soc: str = "qrb5165") -> Workload:
    """All 5 robotic roles at native granularity and natural frequencies."""
    envelope_us = 500_000.0  # 500ms envelope
    return _pack_instances([
        ("yolov8n",   1,  "fine"),         # heavy planning at 1Hz
        ("dronet",    5,  "fine"),         # perception
        ("mlp_wide",  30, "fine"),         # IMU
        ("control",   50, "control_chain"),
        ("monitor",   5,  "monitor"),
    ], envelope_us, soc=soc)


def solver_killer(soc: str = "qrb5165") -> Workload:
    """Push the exact solvers past their tractable region. ~250+ ops with
    tight deadlines and lots of contention."""
    envelope_us = 400_000.0  # 400ms envelope
    return _pack_instances([
        ("yolov8n",   1,  "fine"),     # 48 ops
        ("dronet",    8,  "fine"),     # 8 * 15 = 120 ops
        ("mlp_wide",  100, "fine"),    # 40 * 3 = 120 ops
        ("control",   100, "control_chain"),  # 40 * 3 = 120 ops
    ], envelope_us, soc=soc)


def frequency_sweep_breaking_point(soc: str = "qrb5165",
                                    mlp_hz: float = 50) -> Workload:
    """Used by the driver to sweep ONE model's Hz upward and observe each
    scheduler's deadline-miss curve. The `mlp_hz` parameter is the swept
    variable; pass different values from the driver."""
    envelope_us = 500_000.0
    return _pack_instances([
        ("dronet",    5,       "fine"),
        ("mlp_wide",  mlp_hz,  "fine"),
        ("control",   50,      "control_chain"),
    ], envelope_us, soc=soc)


STRESS_SCENARIOS = {
    "dominator_packing":               dominator_packing,
    "multi_granularity_dronet":        multi_granularity_dronet,
    "mixed_size_stack":                mixed_size_stack,
    "solver_killer":                   solver_killer,
}
