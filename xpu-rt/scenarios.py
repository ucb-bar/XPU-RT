"""
Diagnostic synthetic scenarios that isolate single scheduling effects.

Each scenario builder returns ``(Workload, expected_winners, expected_failures)``
where ``expected_winners`` is a dict mapping a metric name to one or more
scheduler names that should perform best, and ``expected_failures`` lists
schedulers expected to perform poorly. The driver in
``scripts/run_scenarios.py`` cross-checks observed against expected and
embeds the table in the report.

Scenarios:
  1. wide_heft_enough              — embarrassingly parallel; HEFT == fastest_device
  2. transfer_diamond              — fastest-device fails; transfer-aware wins
  3. tight_periodic_multimodel     — small packed real-data analogue; EDF/CP-SAT win
  4. memory_fanout                 — large activation feeds multiple consumers
  5. fusion_win_tiny_chain         — tiny ops with high transfer overhead
  6. fusion_trap_parallel_branches — max-fusion would destroy parallelism
  7. split_win                     — qrb5165-parameterized coarse op hiding heterogeneity
  8. split_loss                    — splitting a single perfectly-mapped op hurts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


REPO_ROOT = Path(__file__).resolve().parent.parent
QRB5165_COSTS_JSON = REPO_ROOT / "qnn_scheduler" / "qrb5165_costs.json"


# Shared helper
def _three_machine_setup():
    machines = ["CPU", "GPU", "NPU"]
    combos = [[m] for m in machines]
    # transfer model — CPU↔GPU cheap, NPU expensive to/from anywhere
    transfer = np.array([
        [0.0, 5.0, 30.0],
        [5.0, 0.0, 30.0],
        [30.0, 30.0, 0.0],
    ])
    return machines, combos, transfer


# ----------------------------------------------------------------------------
# 1. wide_heft_enough
# ----------------------------------------------------------------------------


def wide_heft_enough() -> Tuple[Workload, Dict, Dict]:
    """8 independent ops, identical machines (homogeneous). Every scheduler
    should tie on makespan because there is no placement decision to make
    beyond load balancing."""
    machines, combos, transfer = _three_machine_setup()
    ops = []
    for i in range(8):
        # Same per-machine cost on every machine — no heterogeneity benefit.
        ops.append(Operation(processing_times=[100.0, 100.0, 100.0],
                             operation_name=f"wide_{i}"))
    wl = Workload(ops, machines, transfer, machine_combinations=combos)
    return wl, {"makespan_us": ["heft", "fifo", "critical_path", "edf",
                                 "fastest_device", "cpsat", "mosek"]}, {}


# ----------------------------------------------------------------------------
# 2. transfer_diamond
# ----------------------------------------------------------------------------


def transfer_diamond() -> Tuple[Workload, Dict, Dict]:
    """A → {B, C} → D. B is locally fastest on GPU, C on NPU; but the
    transfer between GPU and NPU is expensive, and the join D pulls from
    both. fastest_device should choose GPU and NPU and pay transfer cost;
    HEFT/CP-SAT should keep B and D on the same machine to avoid hops."""
    machines, combos, transfer = _three_machine_setup()
    a = Operation(processing_times=[40.0, 30.0, 50.0], operation_name="A")
    # B: GPU fastest, but only marginally; NPU very slow
    b = Operation(processing_times=[60.0, 30.0, 100.0],
                  operation_name="B", predecessors=[a])
    # C: NPU fastest, GPU slow
    c = Operation(processing_times=[60.0, 100.0, 30.0],
                  operation_name="C", predecessors=[a])
    # D: needs B and C; NPU fastest individually but transfers hurt
    d = Operation(processing_times=[40.0, 30.0, 25.0],
                  operation_name="D", predecessors=[b, c])
    ops = [a, b, c, d]
    wl = Workload(ops, machines, transfer, machine_combinations=combos)
    return (wl,
            {"cross_device_transitions": ["heft", "cpsat", "mosek"],
             "makespan_us": ["heft", "cpsat", "mosek"]},
            {"makespan_us": ["fastest_device"]})


# ----------------------------------------------------------------------------
# 3. tight_periodic_multimodel
# ----------------------------------------------------------------------------


def tight_periodic_multimodel() -> Tuple[Workload, Dict, Dict]:
    """Two short periodic models on 3 machines:
       - perception: 4-op chain, period 200us, deadline 200us
       - control:    3-op chain, period 80us, deadline 80us (tight!)
       NPU is fast for perception's heavy matmul, but slow for control.
       CPU_BIG is fast for control. fastest_device sends both to NPU and
       starves control; EDF/CP-SAT prioritize the tight control deadline."""
    machines, combos, transfer = _three_machine_setup()
    # perception (job 0): release 0, deadline 200
    p1 = Operation(processing_times=[40.0, 35.0, 25.0], operation_name="percep_pre",
                   min_start_t=0.0)
    p2 = Operation(processing_times=[80.0, 60.0, 30.0], operation_name="percep_mm",
                   predecessors=[p1])
    p3 = Operation(processing_times=[80.0, 60.0, 30.0], operation_name="percep_mm2",
                   predecessors=[p2])
    p4 = Operation(processing_times=[30.0, 30.0, 50.0], operation_name="percep_post",
                   predecessors=[p3], deadline_us=200.0)
    # control (job 1): release 0, deadline 80 (TIGHT)
    c1 = Operation(processing_times=[20.0, 30.0, 60.0], operation_name="ctrl_in",
                   min_start_t=0.0)
    c2 = Operation(processing_times=[20.0, 30.0, 60.0], operation_name="ctrl_calc",
                   predecessors=[c1])
    c3 = Operation(processing_times=[20.0, 30.0, 60.0], operation_name="ctrl_out",
                   predecessors=[c2], deadline_us=80.0)
    for op in (p1, p2, p3, p4):
        op.job_id = 0
    for op in (c1, c2, c3):
        op.job_id = 1
    wl = Workload([p1, p2, p3, p4, c1, c2, c3], machines, transfer,
                  machine_combinations=combos, job_names=["perception", "control"])
    return (wl,
            {"deadline_miss_count": ["edf", "cpsat", "mosek"]},
            {"deadline_miss_count": ["heft", "fastest_device"]})


# ----------------------------------------------------------------------------
# 4. memory_fanout
# ----------------------------------------------------------------------------


def memory_fanout() -> Tuple[Workload, Dict, Dict]:
    """One large producer feeding 4 parallel consumers. Without a memory
    planner the schedule still has to coordinate the 4 consumers on the
    available machines. fastest_device picks NPU for all consumers and
    serializes them; HEFT spreads them across CPU and GPU.

    Buffer-size annotations live on op.output_bytes for the future memory
    planner."""
    machines, combos, transfer = _three_machine_setup()
    producer = Operation(processing_times=[80.0, 50.0, 30.0],
                         operation_name="producer")
    producer.output_bytes = 16 * 1024 * 1024  # 16 MB hot activation
    consumers = []
    for i in range(4):
        c = Operation(processing_times=[60.0, 40.0, 25.0],
                      operation_name=f"consumer_{i}",
                      predecessors=[producer])
        c.output_bytes = 4 * 1024 * 1024  # 4 MB each
        consumers.append(c)
    join = Operation(processing_times=[30.0, 25.0, 40.0],
                     operation_name="join",
                     predecessors=consumers)
    ops = [producer, *consumers, join]
    wl = Workload(ops, machines, transfer, machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek"]},
            {"makespan_us": ["fastest_device"]})


# ----------------------------------------------------------------------------
# 5. fusion_win_tiny_chain
# ----------------------------------------------------------------------------


def fusion_win_tiny_chain() -> Tuple[Workload, Dict, Dict]:
    """6 tiny dispatches in a chain. Per-op compute is small (2-5us) but the
    NPU↔CPU transfer is 30us, so per-dispatch placement on different
    machines is catastrophic. A scheduler should keep the chain on ONE
    machine to avoid transfer cost. fastest_device naively picks the per-op
    locally-fastest (NPU for some, CPU for others) and pays transfers."""
    machines, combos, transfer = _three_machine_setup()
    ops: List[Operation] = []
    prev = None
    # Alternating "matmul-ish" (NPU-fast) and "elementwise" (CPU-fast).
    profile = [
        (10.0, 8.0, 3.0),   # NPU fastest (matmul-ish)
        (4.0, 6.0, 12.0),   # CPU fastest (elementwise)
        (10.0, 8.0, 3.0),
        (4.0, 6.0, 12.0),
        (10.0, 8.0, 3.0),
        (4.0, 6.0, 12.0),
    ]
    for i, (c, g, n) in enumerate(profile):
        op = Operation(processing_times=[c, g, n], operation_name=f"tiny_{i}",
                       predecessors=[prev] if prev is not None else [])
        ops.append(op)
        prev = op
    wl = Workload(ops, machines, transfer, machine_combinations=combos)
    return (wl,
            {"cross_device_transitions": ["heft", "cpsat", "mosek"]},
            {"cross_device_transitions": ["fastest_device"]})


# ----------------------------------------------------------------------------
# 6. fusion_trap_parallel_branches
# ----------------------------------------------------------------------------


def fusion_trap_parallel_branches() -> Tuple[Workload, Dict, Dict]:
    """A → {B1 → C1, B2 → C2} → join. B1/C1 prefer GPU; B2/C2 prefer NPU.
    If fused into one mega-op, both branches must run on a single machine
    and the parallelism is destroyed. A scheduler with branch-aware
    placement (HEFT, CP-SAT) places each branch on its preferred machine
    and runs them concurrently."""
    machines, combos, transfer = _three_machine_setup()
    a = Operation(processing_times=[20.0, 20.0, 30.0], operation_name="A")
    b1 = Operation(processing_times=[60.0, 25.0, 80.0],
                   operation_name="B1", predecessors=[a])
    c1 = Operation(processing_times=[60.0, 25.0, 80.0],
                   operation_name="C1", predecessors=[b1])
    b2 = Operation(processing_times=[80.0, 80.0, 25.0],
                   operation_name="B2", predecessors=[a])
    c2 = Operation(processing_times=[80.0, 80.0, 25.0],
                   operation_name="C2", predecessors=[b2])
    join = Operation(processing_times=[15.0, 15.0, 25.0],
                     operation_name="join", predecessors=[c1, c2])
    ops = [a, b1, c1, b2, c2, join]
    wl = Workload(ops, machines, transfer, machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek"]},
            {})


# ----------------------------------------------------------------------------
# 7. split_win (parameterized from qrb5165_costs.json)
# 8. split_loss (single pure-NPU op)
# ----------------------------------------------------------------------------


def _qrb5165_first_match(op_kind: str, target: str) -> float:
    """Pull a representative mean_us for ``op_kind`` on ``target`` from the
    QRB5165 cost table. Returns ``None`` if no match."""
    if not QRB5165_COSTS_JSON.exists():
        return None
    with open(QRB5165_COSTS_JSON) as f:
        data = json.load(f)
    aliases = {
        "matmul": ("matmul_like", "matmul"),
        "conv":   ("conv2d", "conv"),
        "elementwise": ("elementwise",),
    }.get(op_kind, (op_kind,))
    for key, entry in data.get("execute", {}).items():
        kl = key.lower()
        if not any(a in kl for a in aliases):
            continue
        if f"::{target.lower()}::" in kl:
            return float(entry.get("mean_us", entry.get("p50_us", 0)))
    return None


def split_win() -> Tuple[Workload, Dict, Dict]:
    """A coarse "MegaOp" that internally has 3 stages with different best
    targets: preprocess (CPU best), main compute (HTA best), postprocess
    (CPU best). Costs pulled from qrb5165_costs.json.

    If kept fused (modeled here as a single op that runs on one machine
    only): one combo per backend, taking the SUM of the 3 stages.

    If split (modeled here as 3 chained ops): each stage can choose its own
    machine. Splitting is expected to beat the fused variant."""
    # Fetch realistic costs.
    pre_cpu = _qrb5165_first_match("elementwise", "CPU") or 260.0
    pre_gpu = pre_cpu * 1.5  # not great for elementwise on GPU
    pre_npu = pre_cpu * 2.5  # NPU launch overhead dominates tiny pre

    main_cpu = _qrb5165_first_match("conv", "CPU") or 5283.0
    main_gpu = _qrb5165_first_match("conv", "GPU") or main_cpu * 0.5
    main_npu = _qrb5165_first_match("conv", "HTA") or main_cpu * 0.1

    post_cpu = pre_cpu
    post_gpu = pre_gpu
    post_npu = pre_npu

    machines = ["CPU", "GPU", "NPU"]
    combos = [[m] for m in machines]
    transfer = np.array([
        [0.0, 10.0, 25.0],
        [10.0, 0.0, 25.0],
        [25.0, 25.0, 0.0],
    ])

    # SPLIT version: 3 chained ops, each picks its own backend.
    o_pre = Operation(processing_times=[pre_cpu, pre_gpu, pre_npu],
                      operation_name="MegaOp_pre")
    o_main = Operation(processing_times=[main_cpu, main_gpu, main_npu],
                       operation_name="MegaOp_main", predecessors=[o_pre])
    o_post = Operation(processing_times=[post_cpu, post_gpu, post_npu],
                       operation_name="MegaOp_post", predecessors=[o_main])
    wl = Workload([o_pre, o_main, o_post], machines, transfer,
                  machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek"]},
            {})


def split_loss() -> Tuple[Workload, Dict, Dict]:
    """One pure HTA-best op (no internal stage variation). Splitting it
    would just introduce transfer overhead. With a single op, all schedulers
    should pick HTA. Used as a control: in the closed-loop optimizer, the
    rewrite generator should NOT propose splitting this op."""
    machines = ["CPU", "GPU", "NPU"]
    combos = [[m] for m in machines]
    transfer = np.array([
        [0.0, 10.0, 25.0],
        [10.0, 0.0, 25.0],
        [25.0, 25.0, 0.0],
    ])
    main_cpu = _qrb5165_first_match("conv", "CPU") or 5000.0
    main_gpu = _qrb5165_first_match("conv", "GPU") or 2000.0
    main_npu = _qrb5165_first_match("conv", "HTA") or 500.0
    op = Operation(processing_times=[main_cpu, main_gpu, main_npu],
                   operation_name="PureNPUOp")
    wl = Workload([op], machines, transfer, machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek", "fastest_device",
                             "edf", "critical_path", "fifo"]},
            {})


SCENARIOS = {
    "wide_heft_enough": wide_heft_enough,
    "transfer_diamond": transfer_diamond,
    "tight_periodic_multimodel": tight_periodic_multimodel,
    "memory_fanout": memory_fanout,
    "fusion_win_tiny_chain": fusion_win_tiny_chain,
    "fusion_trap_parallel_branches": fusion_trap_parallel_branches,
    "split_win": split_win,
    "split_loss": split_loss,
}
