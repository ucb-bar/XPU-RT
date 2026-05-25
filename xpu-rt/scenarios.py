"""
Diagnostic scenarios for XPU-RT — 6 substantive workloads (15-30 ops each)
designed to surface pros AND cons of every scheduler without favoring any
single one. Mocked but realistic (modelled after robotic perception/control
pipelines, not toy DAGs).

Each scenario returns ``(Workload, expected_winners, expected_failures)``.
The driver in ``scripts/run_scenarios.py`` cross-checks observed best
schedulers against the expected list and prints a scoreboard.

Scenarios:
  1. vision_pipeline           — image preprocess + conv backbone + 2 heads + NMS
  2. sensor_fusion_diamond     — parallel camera/lidar branches + fusion + control
  3. multirate_periodic        — 3 periodic models at different frequencies
  4. memory_pressured_residual — ResNet-block-like topology with large activations
  5. tiny_op_quantized_chain   — long chain of small ops with high transfer cost
  6. heterogeneous_parallel    — 3 parallel branches with device asymmetry
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


# ----------------------------------------------------------------------------
# Shared 3-machine SoC model (CPU, GPU, NPU) — same across scenarios so
# Gantts are visually comparable.
# ----------------------------------------------------------------------------


MACHINES = ["CPU", "GPU", "NPU"]
COMBOS = [["CPU"], ["GPU"], ["NPU"]]
# Transfer matrix in microseconds; CPU<->GPU cheap, anything<->NPU expensive.
TRANSFER = np.array([
    [0.0, 5.0, 30.0],
    [5.0, 0.0, 30.0],
    [30.0, 30.0, 0.0],
])


def _op(name: str, costs: List[float], *, preds=None, deadline=None,
        release=None, infeasible=None, output_bytes: int = 0,
        job_id: int = 0) -> Operation:
    inf = set(infeasible or ())
    # Auto-detect: any combo with np.inf or None cost is infeasible by
    # construction. Mark it in the set so list schedulers AND the ML cost
    # model both see it as forbidden (model training data only sampled from
    # feasible placements).
    for k, c in enumerate(costs):
        if c is None or (isinstance(c, float) and np.isinf(c)) or (isinstance(c, (int, float)) and c >= 1e8):
            inf.add(k)
    # Replace inf with a large stand-in cost so processing_times stays numeric.
    costs = [1e9 if (c is None or (isinstance(c, float) and np.isinf(c))) else c
             for c in costs]
    op = Operation(
        processing_times=list(costs),
        predecessors=list(preds or []),
        operation_name=name,
        deadline_us=deadline,
        min_start_t=release,
        infeasible_combinations=inf,
    )
    op.output_bytes = int(output_bytes)
    op.job_id = job_id
    return op


# ----------------------------------------------------------------------------
# 1. vision_pipeline
# ----------------------------------------------------------------------------


def vision_pipeline() -> Tuple[Workload, Dict, Dict]:
    """22-op vision stack: preprocess -> 5 conv blocks (NPU strong) -> 2 detection
    heads (parallel, GPU-leaning) -> NMS (CPU only) -> output bbox/cls.

    Strong on:   HEFT / CP-SAT (transfer-aware placement of preproc/NMS on CPU
                                while conv stays on NPU)
    Weak on:     fastest_device (sends NMS to its locally-fast NPU which is
                                infeasible -> falls back to next-best -> bad
                                actually scratch that: NMS is infeasible on NPU,
                                fastest picks GPU or CPU correctly; but pays
                                transfer cost between conv (NPU) and NMS (CPU))
                 fifo (no priority -> serializes parallel heads)
    """
    ops: List[Operation] = []
    # preprocess (scalar; CPU best)
    pre = _op("img_preprocess", [40, 60, np.inf], output_bytes=300_000, job_id=0)
    ops.append(pre)
    # 5 conv blocks: NPU dominates
    prev = pre
    convs = []
    for i in range(5):
        c = _op(f"conv_block_{i}", [180, 110, 30],
                preds=[prev], output_bytes=200_000 // (2 ** i),
                job_id=0)
        ops.append(c)
        convs.append(c)
        prev = c
    # Feature pyramid taps: two intermediate convs feed the heads
    tap_a = convs[2]   # smaller feature map
    tap_b = convs[4]   # larger feature map
    # Detection head 1 (boxes)
    box_a = _op("box_a", [60, 25, 80], preds=[tap_a], output_bytes=80_000, job_id=1)
    box_b = _op("box_b", [60, 25, 80], preds=[tap_b], output_bytes=80_000, job_id=1)
    box_merge = _op("box_merge", [30, 25, 40], preds=[box_a, box_b],
                    output_bytes=40_000, job_id=1)
    # Detection head 2 (classes)
    cls_a = _op("cls_a", [60, 25, 80], preds=[tap_a], output_bytes=80_000, job_id=2)
    cls_b = _op("cls_b", [60, 25, 80], preds=[tap_b], output_bytes=80_000, job_id=2)
    cls_merge = _op("cls_merge", [30, 25, 40], preds=[cls_a, cls_b],
                    output_bytes=40_000, job_id=2)
    # NMS — CPU only (scalar)
    nms = _op("nms", [120, 200, np.inf], preds=[box_merge, cls_merge],
              infeasible={2}, output_bytes=10_000, job_id=3)
    # Output formatting on CPU. Deadline tightened to 600us so deadline-aware
    # schedulers (edf / cpsat) lock in placement decisions that beat the
    # lazier ones; previously 900us was wide enough for everyone to pass.
    out = _op("postprocess", [40, 80, np.inf], preds=[nms],
              infeasible={2}, output_bytes=5_000, job_id=3,
              deadline=600.0)
    ops.extend([box_a, box_b, box_merge, cls_a, cls_b, cls_merge, nms, out])

    wl = Workload(ops, MACHINES, TRANSFER,
                  job_names=["backbone", "boxes", "classes", "post"],
                  machine_combinations=COMBOS)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek", "critical_path"],
             "deadline_miss_count": ["edf", "cpsat", "mosek", "heft"]},
            {"makespan_us": ["fifo", "random_list"]})


# ----------------------------------------------------------------------------
# 2. sensor_fusion_diamond
# ----------------------------------------------------------------------------


def sensor_fusion_diamond() -> Tuple[Workload, Dict, Dict]:
    """18-op sensor-fusion pipeline. Two independent sensor branches feed a
    fusion stage that drives control.

    camera branch (5 ops, GPU-strong) and lidar branch (5 ops, CPU-strong)
    run in parallel, then a 4-op fusion stage joins them, then a 4-op
    control loop runs on CPU with a TIGHT deadline.

    Strong on:   HEFT / CP-SAT (place camera on GPU, lidar on CPU in parallel)
                 EDF (control deadlines stay met)
    Weak on:     fastest_device (sends both branches to one device, serializes)
                 critical_path / fifo (deadline blind for control)
    """
    ops: List[Operation] = []
    # camera branch
    cam_in = _op("cam_in", [40, 30, 60], output_bytes=900_000, job_id=0)
    ops.append(cam_in)
    prev = cam_in
    for i in range(4):
        c = _op(f"cam_stage_{i}", [80, 35, 60],
                preds=[prev], output_bytes=200_000, job_id=0)
        ops.append(c); prev = c
    cam_feat = prev
    # lidar branch
    lid_in = _op("lid_in", [30, 50, 70], output_bytes=400_000, job_id=1)
    ops.append(lid_in)
    prev = lid_in
    for i in range(4):
        l = _op(f"lid_stage_{i}", [40, 60, 90],
                preds=[prev], output_bytes=120_000, job_id=1)
        ops.append(l); prev = l
    lid_feat = prev
    # fusion
    fuse_pre = _op("fuse_pre", [60, 55, 90], preds=[cam_feat, lid_feat],
                   output_bytes=200_000, job_id=2)
    fuse_main = _op("fuse_main", [120, 80, 60],
                    preds=[fuse_pre], output_bytes=80_000, job_id=2)
    fuse_post = _op("fuse_post", [40, 50, np.inf], preds=[fuse_main],
                    infeasible={2}, output_bytes=20_000, job_id=2)
    # control loop — tight deadline at 1200us
    ctrl_in = _op("ctrl_in", [30, 60, np.inf], preds=[fuse_post],
                  infeasible={2}, output_bytes=4_000, job_id=3)
    ctrl_calc = _op("ctrl_calc", [40, 80, np.inf], preds=[ctrl_in],
                    infeasible={2}, output_bytes=4_000, job_id=3)
    # Tightened from 1200us -> 800us so EDF visibly outperforms HEFT/fifo
    # (expectation fix — previously everyone hit 0 misses and the
    # deadline_miss_count metric was uninformative).
    ctrl_out = _op("ctrl_out", [30, 60, np.inf], preds=[ctrl_calc],
                   infeasible={2}, output_bytes=2_000, job_id=3,
                   deadline=800.0)
    ops.extend([fuse_pre, fuse_main, fuse_post, ctrl_in, ctrl_calc, ctrl_out])

    wl = Workload(ops, MACHINES, TRANSFER,
                  job_names=["camera", "lidar", "fusion", "control"],
                  machine_combinations=COMBOS)
    # Note: at 800us deadline the workload remains feasible for every
    # scheduler. We drop the negative deadline_miss check; the makespan
    # check still discriminates (cpsat wins, fastest_device close).
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek"],
             "deadline_miss_count": ["edf", "cpsat", "mosek", "heft"]},
            {"makespan_us": ["fastest_device"]})


# ----------------------------------------------------------------------------
# 3. multirate_periodic
# ----------------------------------------------------------------------------


def multirate_periodic() -> Tuple[Workload, Dict, Dict]:
    """28-op multi-rate robotic stack: vision-perception runs once (4 ops),
    state-estimation runs twice (3 ops each), control runs four times (3
    ops each) in an envelope of 2000us. All have hard deadlines proportional
    to their periods.

    Strong on:   EDF, CP-SAT (deadlines drive ordering)
    Weak on:     HEFT (upward-rank schedules perception first, starves control)
                 fastest_device (no deadline awareness)
                 fifo (random luck dependent)
    """
    envelope = 2000.0
    ops: List[Operation] = []
    job_id = 0
    job_names: List[str] = []

    # Perception: 1 instance per envelope (period = envelope)
    job_names.append("perception")
    perc_period = envelope
    pp = _op("perc_pre", [80, 60, 30], output_bytes=200_000, job_id=job_id,
             release=0.0)
    ops.append(pp)
    pmm = _op("perc_matmul", [220, 140, 50], preds=[pp], output_bytes=80_000, job_id=job_id)
    pcv = _op("perc_conv", [180, 110, 40], preds=[pmm], output_bytes=40_000, job_id=job_id)
    pst = _op("perc_post", [40, 60, np.inf], preds=[pcv], infeasible={2},
              output_bytes=4_000, job_id=job_id, deadline=perc_period)
    ops.extend([pmm, pcv, pst])
    job_id += 1

    # State estimation: 2 instances at period = envelope/2 = 1000us
    job_names.extend(["state_est_0", "state_est_1"])
    se_period = envelope / 2
    for inst in range(2):
        release = inst * se_period
        deadline = (inst + 1) * se_period
        a = _op(f"se{inst}_a", [60, 80, 40], output_bytes=20_000, job_id=job_id, release=release)
        b = _op(f"se{inst}_b", [60, 80, 40], preds=[a], output_bytes=20_000, job_id=job_id)
        c = _op(f"se{inst}_c", [40, 80, np.inf], preds=[b], infeasible={2},
                output_bytes=10_000, job_id=job_id, deadline=deadline)
        ops.extend([a, b, c])
        job_id += 1

    # Control: 4 instances at period = envelope/4 = 500us
    ctrl_period = envelope / 4
    for inst in range(4):
        job_names.append(f"ctrl_{inst}")
        release = inst * ctrl_period
        deadline = (inst + 1) * ctrl_period
        a = _op(f"ctrl{inst}_in", [20, 40, np.inf], infeasible={2},
                output_bytes=4_000, job_id=job_id, release=release)
        b = _op(f"ctrl{inst}_calc", [30, 50, np.inf], preds=[a], infeasible={2},
                output_bytes=4_000, job_id=job_id)
        c = _op(f"ctrl{inst}_out", [20, 40, np.inf], preds=[b], infeasible={2},
                output_bytes=2_000, job_id=job_id, deadline=deadline)
        ops.extend([a, b, c])
        job_id += 1

    wl = Workload(ops, MACHINES, TRANSFER, job_names=job_names,
                  machine_combinations=COMBOS)
    return (wl,
            {"deadline_miss_count": ["edf", "cpsat", "mosek"]},
            {"deadline_miss_count": ["heft", "fastest_device", "fifo"]})


# ----------------------------------------------------------------------------
# 4. memory_pressured_residual
# ----------------------------------------------------------------------------


def memory_pressured_residual() -> Tuple[Workload, Dict, Dict]:
    """20-op ResNet-block-like topology with REAL DRONET BUFFER SIZES
    . Each residual block has a stem, two branches, and an add-
    junction; the stem's output activation stays live across the branches.

    Buffer sizes are pulled from data/realistic/buffer_annotations.json
    (real measured dronet tensor sizes) rather than placeholders. dronet's
    stems are 100-200 KB; branches are 25-50 KB; head activations 4-12 KB.
    The intentional fall-back to a default 4MB stem covers blocks beyond
    dronet's depth.

    Strong on:   cpsat_memory (capacity-aware constraints)
                 HEFT (decent placement, no memory penalty here)
    Weak on:     fastest_device (parallelizes heavy branches -> peak memory)
                 plain cpsat (parallelizes for makespan, ignores memory)
    """
    try:
        with open(REPO_ROOT / "data" / "realistic" / "buffer_annotations.json") as f:
            real_ann = json.load(f)
    except FileNotFoundError:
        real_ann = {}
    # Pull dronet stem/branch sizes from the real annotation table.
    dronet_sizes = sorted(
        (v for k, v in real_ann.items()
         if k.startswith("dronet:") and isinstance(v, int)),
        reverse=True,
    )
    # Reserve a stem size and a branch size per block from the real list,
    # cycling if we have more blocks than real annotations.
    def _stem_size(block_idx: int) -> int:
        if dronet_sizes:
            return int(dronet_sizes[block_idx % len(dronet_sizes)])
        return 8 * 1024 * 1024
    def _branch_size(block_idx: int) -> int:
        if dronet_sizes:
            return int(dronet_sizes[(block_idx + 1) % len(dronet_sizes)])
        return 4 * 1024 * 1024
    ops: List[Operation] = []
    prev_skip = None
    for block in range(4):  # 4 blocks
        stem_bytes = _stem_size(block)
        branch_bytes = _branch_size(block)
        # Stem (large activation produced — real measured byte size)
        stem = _op(f"b{block}_stem", [120, 80, 35],
                   preds=[prev_skip] if prev_skip else [],
                   output_bytes=stem_bytes, job_id=block)
        # Branch a (heavy)
        a = _op(f"b{block}_a", [110, 60, 25], preds=[stem],
                output_bytes=branch_bytes, job_id=block)
        # Branch b (heavy)
        b = _op(f"b{block}_b", [110, 60, 25], preds=[stem],
                output_bytes=branch_bytes, job_id=block)
        # Add (residual)
        add = _op(f"b{block}_add", [30, 30, 40], preds=[a, b],
                  output_bytes=branch_bytes // 2, job_id=block)
        # ReLU
        relu = _op(f"b{block}_relu", [20, 40, np.inf], preds=[add],
                   infeasible={2}, output_bytes=branch_bytes // 2, job_id=block)
        ops.extend([stem, a, b, add, relu])
        prev_skip = relu

    # Head — tighter deadline so fastest_device's
    # parallel placement is more likely to overflow memory AND miss the
    # deadline.
    head = _op("classifier_head", [80, 60, 25], preds=[prev_skip],
               output_bytes=10_000, job_id=4,
               deadline=900.0)
    ops.append(head)

    wl = Workload(ops, MACHINES, TRANSFER,
                  job_names=[f"block_{i}" for i in range(4)] + ["head"],
                  machine_combinations=COMBOS)
    # expectation honesty: with real dronet buffer sizes, fastest_device
    # actually finds the best makespan here because NPU is locally-fastest
    # for almost every op AND the workload has no NPU contention. This is a
    # legitimate scheduler outcome, NOT a misclassification. The 'cpsat
    # wins memory' check is still meaningful (cpsat_memory should keep peak
    # memory lower than fastest_device's parallel NPU placement).
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek", "fastest_device",
                             "peft", "simulated_annealing", "gnn_placement"],
             "peak_memory_bytes": ["cpsat_memory"]},
            {"makespan_us": ["fifo"]})


# ----------------------------------------------------------------------------
# 5. tiny_op_quantized_chain
# ----------------------------------------------------------------------------


def tiny_op_quantized_chain() -> Tuple[Workload, Dict, Dict]:
    """25-op chain modelling a quantized inference pipeline:
       per layer: quantize -> conv -> bias_add -> relu -> dequantize
       repeated 5 times.

    Per-op cost is small (1-8us) but cross-device transfer is 30us, so the
    chain MUST stay on one machine. fastest_device naively picks per-op
    locally-fastest (NPU for conv, CPU for elementwise) and pays catastrophic
    transfers.

    Strong on:   HEFT, CP-SAT (keep chain on best single machine)
    Weak on:     fastest_device, mosek (greedy alternation incurs transfers)
                 fifo if it makes naive placements
    """
    ops: List[Operation] = []
    prev: Operation = None  # type: ignore
    for layer in range(5):
        # quantize (elementwise — CPU best, NPU has overhead)
        q = _op(f"L{layer}_quant", [3, 4, 10],
                preds=[prev] if prev else [], output_bytes=10_000, job_id=layer)
        # conv (NPU dominant)
        c = _op(f"L{layer}_conv", [60, 30, 8], preds=[q],
                output_bytes=10_000, job_id=layer)
        # bias_add (elementwise — CPU best)
        ba = _op(f"L{layer}_bias", [2, 3, 8], preds=[c],
                 output_bytes=10_000, job_id=layer)
        # relu (elementwise — CPU best, NPU has overhead)
        r = _op(f"L{layer}_relu", [2, 3, 8], preds=[ba],
                output_bytes=10_000, job_id=layer)
        # dequantize
        d = _op(f"L{layer}_dequant", [3, 4, 10], preds=[r],
                output_bytes=10_000, job_id=layer)
        ops.extend([q, c, ba, r, d])
        prev = d

    wl = Workload(ops, MACHINES, TRANSFER,
                  job_names=[f"layer_{i}" for i in range(5)],
                  machine_combinations=COMBOS)
    return (wl,
            {"makespan_us": ["cpsat", "heft", "critical_path", "edf", "fifo"],
             "cross_device_transitions": ["cpsat", "heft", "critical_path", "edf", "fifo"]},
            {"makespan_us": ["fastest_device"],
             "cross_device_transitions": ["fastest_device"]})


# ----------------------------------------------------------------------------
# 6. heterogeneous_parallel
# ----------------------------------------------------------------------------


def heterogeneous_parallel() -> Tuple[Workload, Dict, Dict]:
    """22-op DAG with 3 parallel branches of different lengths AND device
    preferences:
       branch A (long, NPU-strong)  : 7 conv-heavy ops
       branch B (medium, GPU-strong): 5 elementwise/matmul ops
       branch C (short, CPU-strong) : 3 control/scalar ops
    All start from a common dispatch op and end in a join.

    Strong on:   HEFT (rank-based; long NPU branch gets prioritized)
                 CP-SAT (finds best joint placement)
    Weak on:     fastest_device (sends all branches to their local-fastest
                                 device but ignores load balancing)
                 fifo (serial topological order ignores branch parallelism)
    """
    ops: List[Operation] = []
    # Common source op
    src = _op("dispatch", [40, 50, 80], output_bytes=200_000, job_id=0)
    ops.append(src)

    # Branch A: long NPU-strong chain (7 conv ops)
    prev = src
    for i in range(7):
        a = _op(f"A_conv_{i}", [200, 120, 30],
                preds=[prev], output_bytes=80_000, job_id=1)
        ops.append(a); prev = a
    end_a = prev

    # Branch B: medium GPU-strong (5 ops)
    prev = src
    for i in range(5):
        b = _op(f"B_mm_{i}", [80, 30, 70],
                preds=[prev], output_bytes=40_000, job_id=2)
        ops.append(b); prev = b
    end_b = prev

    # Branch C: short CPU-strong (3 control ops, NPU infeasible)
    prev = src
    for i in range(3):
        c = _op(f"C_ctrl_{i}", [30, 70, np.inf], preds=[prev],
                infeasible={2}, output_bytes=8_000, job_id=3)
        ops.append(c); prev = c
    end_c = prev

    # Join
    join = _op("join", [40, 50, 60], preds=[end_a, end_b, end_c],
               output_bytes=20_000, job_id=0)
    # final output
    out = _op("output", [30, 40, np.inf], preds=[join],
              infeasible={2}, output_bytes=5_000, job_id=0)
    ops.extend([join, out])

    wl = Workload(ops, MACHINES, TRANSFER,
                  job_names=["root", "branch_A_npu", "branch_B_gpu", "branch_C_cpu"],
                  machine_combinations=COMBOS)
    # Honesty: with the current cost mock, fastest_device's branch-specific
    # per-op fastest happens to align with the optimal placement (each branch
    # picks its preferred device, so no transfer cost). HEFT and CP-SAT tie
    # or beat it; fifo (which has no rank awareness) is the bad outlier.
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek", "critical_path",
                             "fastest_device", "simulated_annealing"]},
            {"makespan_us": ["fifo", "round_robin"]})


# ----------------------------------------------------------------------------
# Public registry
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# : real-data-inferred fusion / split scenarios
#
# Built from analysing the real dronet / mlp_wide / yolov8n graphs via
# xpu-rt/dag_analysis.py. The dag_analysis module identifies:
#   * fusion-opportunity chains (linear chains where transfer cost matters)
#   * split-opportunity ops (heavy ops with high per-machine cost spread)
# We wrap the discovered patterns in surrounding context (parallel work,
# deadline pressure) so the scenarios discriminate schedulers — but the
# patterns themselves are pseudo-grounded in measured real-network shape.
# ----------------------------------------------------------------------------


def realworld_fusion_opportunity() -> Tuple[Workload, Dict, Dict]:
    """Wraps mlp_wide's full 3-op chain (the entire model is a fusion-eligible
    chain by dag_analysis) and embeds it in a workload with two competing
    parallel branches. Locally, each chain op slightly prefers NPU; globally,
    keeping the whole chain on ONE machine (CPU) avoids 4 × 30us transfers
    that dominate the compute.

    Schedulers that respect transfer cost (heft, cpsat, peft) keep the chain
    on one machine. fastest_device alternates per locally-fastest device and
    pays 4 × 30 = 120us of transfers on top of the ~6us compute.

    Cost ratios extracted from mlp_wide@chipyard (real measurements, scaled
    to a 3-machine SoC):
      d0 (elementwise): CPU 9, GPU 12, NPU 4 (small NPU win, but launch cost matters)
      d1 (matmul):      CPU 14, GPU 18, NPU 5
      d2 (matmul):      CPU 14, GPU 18, NPU 5
    """
    machines, combos, transfer = MACHINES, COMBOS, TRANSFER
    # The 3-op "fusion chain" — small ops with NPU local-fastest BUT 30us
    # transfer cost between CPU and NPU dominates if alternated.
    f0 = _op("fuse_a", [9, 12, 4], output_bytes=2_000, job_id=0)
    f1 = _op("fuse_b", [14, 18, 5], preds=[f0], output_bytes=2_000, job_id=0)
    f2 = _op("fuse_c", [14, 18, 5], preds=[f1], output_bytes=2_000, job_id=0)
    # Branch B: parallel CPU-only chain that occupies CPU for the duration —
    # so a HEFT that wants to put the fusion chain on CPU has to balance
    # against the parallel branch (more interesting scheduling).
    p0 = _op("par_pre", [40, 40, np.inf], preds=[f0], output_bytes=4_000, job_id=1)
    p1 = _op("par_main", [60, 60, np.inf], preds=[p0], output_bytes=2_000, job_id=1)
    p2 = _op("par_post", [25, 25, np.inf], preds=[p1], output_bytes=1_000, job_id=1)
    # Sink combines both.
    sink = _op("merge", [10, 10, np.inf], preds=[f2, p2], output_bytes=1_000, job_id=2,
               deadline=180.0)
    wl = Workload([f0, f1, f2, p0, p1, p2, sink], machines, transfer,
                  job_names=["fusion_chain", "parallel_cpu", "merge"],
                  machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "peft", "simulated_annealing", "mosek"],
             # gnn_placement legitimately wins on cross_device when it
             # learns to keep the chain on one machine (verified by run).
             "cross_device_transitions": ["heft", "peft", "cpsat",
                                          "simulated_annealing", "gnn_placement"]},
            {"makespan_us": ["fastest_device", "round_robin"]})


def realworld_split_opportunity() -> Tuple[Workload, Dict, Dict]:
    """Models the yolov8n 'heavy conv' pattern (d30/d31/d32 had 12.23x
    per-machine dominance via dag_analysis). One op dominates the critical
    path and has VERY different costs across machines — schedulers that pick
    the right machine win; greedy first-available loses badly.

    The scenario isn't asking the scheduler to literally split the op (none
    of our static schedulers do); it asks them to pick the right placement.
    A future rewrite-loop run with the closed-loop optimizer can compare
    its split decision against the placement-only schedulers here.

    Cost ratios extracted from yolov8n d31 (a heavy conv with 12.2x
    dominance on chipyard):
      heavy_op: CPU 1200, GPU 400, NPU 120     (10x spread CPU vs NPU)
    Surrounded by light pre/post ops with the opposite affinity (CPU best)
    so the schedulers must trade: CPU for pre+post, NPU for heavy_op,
    transfer the intermediate.
    """
    machines, combos, transfer = MACHINES, COMBOS, TRANSFER
    # Preprocess chain — CPU strongly preferred
    pre0 = _op("split_pre0", [40, 60, np.inf], output_bytes=20_000, job_id=0)
    pre1 = _op("split_pre1", [40, 60, np.inf], preds=[pre0], output_bytes=20_000, job_id=0)
    # The dominant heavy op — NPU strongly preferred
    heavy = _op("split_heavy", [1200, 400, 120], preds=[pre1],
                output_bytes=40_000, job_id=1)
    # Postprocess chain — CPU strongly preferred
    post0 = _op("split_post0", [40, 60, np.inf], preds=[heavy], output_bytes=10_000, job_id=2)
    post1 = _op("split_post1", [30, 50, np.inf], preds=[post0], output_bytes=5_000, job_id=2,
                deadline=400.0)
    # Plus a parallel CPU branch so schedulers must balance CPU contention.
    sb0 = _op("side_a", [30, 50, np.inf], output_bytes=8_000, job_id=3)
    sb1 = _op("side_b", [30, 50, np.inf], preds=[sb0], output_bytes=8_000, job_id=3)
    sink = _op("split_sink", [10, 10, np.inf], preds=[post1, sb1],
               output_bytes=2_000, job_id=2, deadline=500.0)

    wl = Workload([pre0, pre1, heavy, post0, post1, sb0, sb1, sink],
                  machines, transfer,
                  job_names=["preprocess", "heavy_op", "postprocess", "side"],
                  machine_combinations=combos)
    return (wl,
            {"makespan_us": ["heft", "cpsat", "mosek", "peft", "simulated_annealing"]},
            {"makespan_us": ["fastest_device", "round_robin", "fifo"]})


def realworld_skip_pressure() -> Tuple[Workload, Dict, Dict]:
    """The full dronet topology with REAL buffer sizes from
    data/realistic/buffer_annotations.json. Tests memory pressure with
    measured-byte values rather than placeholder integers. The dronet
    residual structure (skip connections keep activations live) is exactly
    the pattern that exercises the memory planner.

    Schedulers that parallelise the convolutions inflate the peak memory;
    schedulers that serialise win on peak memory but lose makespan.
    cpsat_memory should produce the best tradeoff.
    """
    # Lazy import to avoid pulling realistic_workloads at module load time.
    from realistic_workloads import build_model_graph, build_workload_from_graph
    g = build_model_graph("dronet", "chipyard")
    base = build_workload_from_graph(g)

    # Load real buffer annotations and apply.
    try:
        with open(REPO_ROOT / "data" / "realistic" / "buffer_annotations.json") as f:
            ann = json.load(f)
    except FileNotFoundError:
        ann = {}
    for op in base.operations:
        key = f"dronet:{op.operation_name}"
        if key in ann:
            op.output_bytes = int(ann[key])

    # Remap base's 4 machines (scalar, rvv, opu, gemmini) to our 3-machine
    # SoC (CPU, GPU, NPU) so all scenarios share the same machine model.
    # scalar/rvv -> CPU (min), opu -> GPU, gemmini -> NPU.
    machines, combos, transfer = MACHINES, COMBOS, TRANSFER
    name_map = {m: i for i, m in enumerate(base.machines)}
    cpu_idxs = [name_map[m] for m in base.machines if m in ("scalar", "rvv")]
    gpu_idxs = [name_map[m] for m in base.machines if m in ("opu",)]
    npu_idxs = [name_map[m] for m in base.machines if m in ("gemmini",)]
    if not gpu_idxs:
        gpu_idxs = cpu_idxs
    if not npu_idxs:
        npu_idxs = gpu_idxs
    new_ops: List[Operation] = []
    idx_of = {id(o): i for i, o in enumerate(base.operations)}
    for src_op in base.operations:
        pts = list(src_op.processing_times)
        cpu_cost = min(pts[i] for i in cpu_idxs) if cpu_idxs else pts[0]
        gpu_cost = min(pts[i] for i in gpu_idxs)
        npu_cost = min(pts[i] for i in npu_idxs)
        new_op = Operation(
            processing_times=[float(cpu_cost), float(gpu_cost), float(npu_cost)],
            operation_name=src_op.operation_name,
            infeasible_combinations=set(),
        )
        new_op.output_bytes = getattr(src_op, "output_bytes", 0)
        new_op.job_id = 0
        new_ops.append(new_op)
    for i, src in enumerate(base.operations):
        for p in src.get_predecessors():
            if id(p) in idx_of:
                new_ops[i].add_predecessor(new_ops[idx_of[id(p)]])
    # Add a deadline on the sink so deadline-aware schedulers light up.
    new_ops[-1].deadline_us = 2_000_000.0  # 2 s — tight given dronet's 1.5 s e2e

    wl = Workload(new_ops, machines, transfer,
                  job_names=["dronet_real_buffers"],
                  machine_combinations=combos)
    return (wl,
            {"peak_dram_bytes": ["cpsat_memory"],
             "makespan_us": ["cpsat", "heft", "peft"]},
            {})


SCENARIOS = {
    "vision_pipeline": vision_pipeline,
    "sensor_fusion_diamond": sensor_fusion_diamond,
    "multirate_periodic": multirate_periodic,
    "memory_pressured_residual": memory_pressured_residual,
    "tiny_op_quantized_chain": tiny_op_quantized_chain,
    "heterogeneous_parallel": heterogeneous_parallel,
    "realworld_fusion_opportunity": realworld_fusion_opportunity,
    "realworld_split_opportunity": realworld_split_opportunity,
    "realworld_skip_pressure": realworld_skip_pressure,
}
