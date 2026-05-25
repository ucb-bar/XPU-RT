"""
M11 prerequisite: generate training data for the learned cost model (and
M12/M13 schedulers).

Two subcommands:

  synthesize   produce N synthetic + real-anchored workloads, save to
               data/training/workloads.pkl

  label        for each workload: run CP-SAT (proven-optimal label) AND
               sample K random feasible placements (each list-scheduled
               to measure makespan). Save to data/training/cpsat_labels.pkl
               and data/training/placements.pkl. Also write
               data/training/splits.json with train/val/test indices
               stratified by topology family.

Each workload entry is a plain dict serializable with pickle (avoid the
Workload object's transient state). We reconstruct a Workload at training/
eval time via ``_workload_from_dict``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from workload import Operation, Workload  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from realistic_workloads import build_model_graph, build_workload_from_graph  # noqa: E402


# ----------------------------------------------------------------------------
# Shared SoC model (same across synthetic + real-anchored for transferability)
# ----------------------------------------------------------------------------

MACHINES = ["CPU", "GPU", "NPU"]
COMBOS = [["CPU"], ["GPU"], ["NPU"]]
TRANSFER = np.array([
    [0.0, 5.0, 30.0],
    [5.0, 0.0, 30.0],
    [30.0, 30.0, 0.0],
])

# Per-op-kind machine affinities (matmul/conv: NPU strong; eltwise: CPU strong)
OPKIND_AFFINITIES = {
    "matmul":      [60.0, 35.0, 12.0],   # cost on [CPU, GPU, NPU]
    "conv":        [80.0, 40.0, 18.0],
    "eltwise":     [8.0,  12.0, 30.0],
    "transfer":    [20.0, 18.0, 25.0],
    "ctrl":        [15.0, 25.0, 1e9],    # NPU infeasible
}


# ----------------------------------------------------------------------------
# Workload serialization (dict <-> Workload)
# ----------------------------------------------------------------------------


def _workload_to_dict(wl: Workload, *, topology: str = "unknown",
                      meta: Optional[Dict] = None) -> Dict[str, Any]:
    """Serialize a Workload to a plain dict (numpy-safe)."""
    op_idx = {id(op): i for i, op in enumerate(wl.operations)}
    preds = []
    for op in wl.operations:
        preds.append([op_idx[id(p)] for p in op.get_predecessors() if id(p) in op_idx])
    return {
        "n_ops": len(wl.operations),
        "machines": list(wl.machines),
        "machine_combinations": [list(c) for c in wl.machine_combinations],
        "transfer_times": np.asarray(wl.transfer_times).tolist(),
        "operations": [
            {
                "name": op.operation_name,
                "processing_times": list(op.processing_times),
                "predecessors": preds[i],
                "deadline_us": op.deadline_us,
                "min_start_t": op.min_start_t,
                "infeasible_combinations": sorted(op.infeasible_combinations),
                "output_bytes": getattr(op, "output_bytes", 0),
                "job_id": op.job_id or 0,
            }
            for i, op in enumerate(wl.operations)
        ],
        "job_names": list(wl.job_names),
        "topology": topology,
        "meta": meta or {},
    }


def _workload_from_dict(d: Dict[str, Any]) -> Workload:
    """Rebuild a Workload from a dict produced by ``_workload_to_dict``."""
    ops: List[Operation] = []
    for o in d["operations"]:
        op = Operation(
            processing_times=list(o["processing_times"]),
            operation_name=o["name"],
            deadline_us=o["deadline_us"],
            min_start_t=o["min_start_t"],
            infeasible_combinations=set(o["infeasible_combinations"]),
        )
        op.output_bytes = int(o.get("output_bytes", 0))
        op.job_id = o.get("job_id", 0)
        ops.append(op)
    # Wire predecessors by index.
    for i, o in enumerate(d["operations"]):
        for j in o["predecessors"]:
            ops[i].add_predecessor(ops[j])
    transfer = np.asarray(d["transfer_times"])
    return Workload(ops, d["machines"], transfer,
                    job_names=d["job_names"],
                    machine_combinations=[list(c) for c in d["machine_combinations"]])


# ----------------------------------------------------------------------------
# Synthetic DAG generators
# ----------------------------------------------------------------------------


def _make_op(name: str, kind: str, *, preds=None, deadline=None,
             release=None, rng: Optional[np.random.Generator] = None,
             jitter_pct: float = 0.25) -> Operation:
    base = OPKIND_AFFINITIES.get(kind, OPKIND_AFFINITIES["eltwise"])
    if rng is None:
        rng = np.random.default_rng()
    # Per-machine cost with ±jitter_pct jitter.
    costs = [float(c) * float(rng.uniform(1 - jitter_pct, 1 + jitter_pct))
             for c in base]
    inf = {k for k, c in enumerate(base) if c >= 1e8}
    return Operation(
        processing_times=costs,
        predecessors=preds or [],
        operation_name=name,
        deadline_us=deadline,
        min_start_t=release,
        infeasible_combinations=inf,
    )


def gen_chain(n: int, rng: np.random.Generator) -> Workload:
    kinds = ["eltwise", "matmul", "conv", "eltwise"]
    ops = []
    prev = None
    for i in range(n):
        k = kinds[i % len(kinds)]
        ops.append(_make_op(f"chain_{i}", k, preds=[prev] if prev else [], rng=rng))
        prev = ops[-1]
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS)


def gen_tree(depth: int, branch: int, rng: np.random.Generator) -> Workload:
    ops: List[Operation] = []
    root = _make_op("root", "eltwise", rng=rng)
    ops.append(root)
    layer = [root]
    next_id = 1
    for d in range(depth - 1):
        new_layer: List[Operation] = []
        for p in layer:
            for _ in range(branch):
                kind = rng.choice(["matmul", "conv", "eltwise"])
                child = _make_op(f"t{next_id}", str(kind), preds=[p], rng=rng)
                ops.append(child)
                new_layer.append(child)
                next_id += 1
        layer = new_layer
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS)


def gen_layered(layers: int, width: int, rng: np.random.Generator) -> Workload:
    ops: List[Operation] = []
    prev_layer: List[Operation] = []
    for layer_idx in range(layers):
        new_layer: List[Operation] = []
        for w in range(width):
            kind = rng.choice(["matmul", "conv", "eltwise"])
            # Each new op connects to 1-2 random ops in prev layer
            if prev_layer:
                n_preds = int(min(len(prev_layer), rng.integers(1, 3)))
                preds = list(rng.choice(prev_layer, size=n_preds, replace=False))
            else:
                preds = []
            op = _make_op(f"L{layer_idx}_{w}", str(kind), preds=preds, rng=rng)
            ops.append(op)
            new_layer.append(op)
        prev_layer = new_layer
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS)


def gen_erdos_renyi(n: int, p: float, rng: np.random.Generator) -> Workload:
    """Random DAG: n nodes, edge i->j with probability p iff i<j (DAG guarantee)."""
    ops: List[Operation] = []
    for i in range(n):
        kind = rng.choice(["matmul", "conv", "eltwise"])
        op = _make_op(f"er_{i}", str(kind), rng=rng)
        ops.append(op)
    for j in range(n):
        for i in range(j):
            if rng.random() < p:
                ops[j].add_predecessor(ops[i])
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS)


def gen_diamond(n_branches: int, branch_len: int, rng: np.random.Generator) -> Workload:
    """One source → n_branches parallel chains of branch_len each → one join."""
    ops: List[Operation] = []
    src = _make_op("src", "eltwise", rng=rng)
    ops.append(src)
    sinks: List[Operation] = []
    for b in range(n_branches):
        prev = src
        for s in range(branch_len):
            kind = rng.choice(["matmul", "conv", "eltwise"])
            op = _make_op(f"b{b}_{s}", str(kind), preds=[prev], rng=rng)
            ops.append(op)
            prev = op
        sinks.append(prev)
    join = _make_op("join", "eltwise", preds=sinks, rng=rng)
    ops.append(join)
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS)


def synthesize_dataset(n_per_family: int = 20, seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    entries: List[Dict[str, Any]] = []
    families = {
        "chain":   lambda: gen_chain(int(rng.integers(5, 30)), rng),
        "tree":    lambda: gen_tree(int(rng.integers(3, 5)),
                                    int(rng.integers(2, 4)), rng),
        "layered": lambda: gen_layered(int(rng.integers(3, 6)),
                                       int(rng.integers(2, 5)), rng),
        "erdos":   lambda: gen_erdos_renyi(int(rng.integers(8, 25)),
                                           float(rng.uniform(0.15, 0.3)), rng),
        "diamond": lambda: gen_diamond(int(rng.integers(2, 5)),
                                       int(rng.integers(2, 5)), rng),
    }
    for family, gen in families.items():
        for _ in range(n_per_family):
            wl = gen()
            if len(wl.operations) < 3:
                continue
            entries.append(_workload_to_dict(wl, topology=family))
    return entries


def real_anchored_dataset(n_per_model: int = 20, seed: int = 0) -> List[Dict[str, Any]]:
    """Load dronet / mlp_wide / yolov8n graphs and emit `n_per_model` variants
    each with ±25 % per-op cost jitter."""
    rng = np.random.default_rng(seed + 100)
    entries: List[Dict[str, Any]] = []
    for model in ("dronet", "mlp_wide", "yolov8n"):
        try:
            g = build_model_graph(model, "chipyard")
        except Exception as exc:
            print(f"[warn] couldn't load {model}: {exc}")
            continue
        # The realistic graphs use 4-backend MACHINES; convert to our 3-backend
        # one by aggregating scalar+rvv → CPU and opu+gemmini → NPU.
        # For simplicity, just take their per-op costs and remap to our 3-machine SoC.
        base_wl = build_workload_from_graph(g)
        base_machines = base_wl.machines  # e.g. [scalar, rvv, opu, gemmini]
        # Map to our 3-machine SoC: scalar/rvv->CPU, gpu/opu->GPU, gemmini->NPU.
        # Heuristic: pick the fastest scalar/rvv as CPU; fastest opu/gemmini as NPU.
        for _ in range(n_per_model):
            ops: List[Operation] = []
            for src_op in base_wl.operations:
                # Find the min cost among "CPU-like" backends.
                pts = list(src_op.processing_times)
                cpu_idxs = [i for i, m in enumerate(base_machines)
                            if m.lower() in ("scalar", "rvv", "cpu")]
                npu_idxs = [i for i, m in enumerate(base_machines)
                            if m.lower() in ("opu", "gemmini", "npu", "hta")]
                gpu_idxs = [i for i, m in enumerate(base_machines)
                            if m.lower() in ("gpu",)] or npu_idxs
                cpu_cost = min((pts[i] for i in cpu_idxs), default=pts[0])
                gpu_cost = min((pts[i] for i in gpu_idxs), default=pts[0])
                npu_cost = min((pts[i] for i in npu_idxs), default=pts[0])
                # Jitter
                jitter = rng.uniform(0.75, 1.25, size=3)
                costs = [float(cpu_cost * jitter[0]),
                         float(gpu_cost * jitter[1]),
                         float(npu_cost * jitter[2])]
                op = Operation(
                    processing_times=costs,
                    operation_name=src_op.operation_name,
                    infeasible_combinations=set(),
                )
                op.output_bytes = getattr(src_op, "output_bytes", 0)
                ops.append(op)
            # Wire predecessors by source op identity.
            idx_of = {id(o): i for i, o in enumerate(base_wl.operations)}
            for i, src_op in enumerate(base_wl.operations):
                for p in src_op.get_predecessors():
                    if id(p) in idx_of:
                        ops[i].add_predecessor(ops[idx_of[id(p)]])
            wl = Workload(ops, MACHINES, TRANSFER,
                          job_names=[model], machine_combinations=COMBOS)
            entries.append(_workload_to_dict(wl, topology=f"real_{model}"))
    return entries


# ----------------------------------------------------------------------------
# CP-SAT labeling + random placement sampling
# ----------------------------------------------------------------------------


def _alpha_to_list(alpha: np.ndarray) -> List[int]:
    """Compress a one-hot alpha (N x K) into a list of K indices (N long)."""
    return [int(np.argmax(alpha[i])) for i in range(alpha.shape[0])]


def _list_schedule_alpha(wl: Workload, alpha_indices: List[int]) -> Optional[float]:
    """Given a fixed (op -> combo) assignment, simulate list scheduling to
    measure makespan. Returns None if infeasible."""
    n = len(wl.operations)
    combos = wl.get_machine_combinations()
    machines = list(wl.machines)
    op_idx = {id(op): i for i, op in enumerate(wl.operations)}

    # Topological sort.
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(wl.operations):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is None:
                continue
            indeg[i] += 1
            succ[pi].append(i)
    queue = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) < n:
        return None

    transfer = wl.get_transfer_times()
    name_to_idx = {m: i for i, m in enumerate(machines)}
    machine_busy = {m: 0.0 for m in machines}
    finish = [0.0] * n

    for i in order:
        op = wl.operations[i]
        k = alpha_indices[i]
        if k in op.infeasible_combinations:
            return None
        machines_here = combos[k]
        est = 0.0
        # Predecessor + transfer.
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is None:
                continue
            pred_combo = combos[alpha_indices[pi]]
            tx = 0.0
            for ma in pred_combo:
                for mb in machines_here:
                    ia, ib = name_to_idx.get(ma), name_to_idx.get(mb)
                    if ia is None or ib is None or ia == ib:
                        continue
                    tx = max(tx, float(transfer[ia][ib]))
            est = max(est, finish[pi] + tx)
        if op.min_start_t is not None:
            est = max(est, float(op.min_start_t))
        # Machine busy.
        est = max(est, max((machine_busy[m] for m in machines_here), default=0.0))
        dur = float(op.get_duration_for_combination(k, combos, machines))
        finish[i] = est + dur
        for m in machines_here:
            machine_busy[m] = finish[i]
    return max(finish)


def label_workload(d: Dict[str, Any], *, time_limit: float = 20.0,
                   n_placements: int = 8,
                   seed: int = 0) -> Tuple[Optional[Dict], List[Dict]]:
    """Run CP-SAT for the optimal placement label + sample random feasible
    placements. Returns (cpsat_record, placement_records)."""
    wl = _workload_from_dict(d)
    rng = np.random.default_rng(seed)
    # CP-SAT.
    cpsat_record = None
    try:
        cpsat = get_scheduler("cpsat")
        t0 = time.perf_counter()
        t, alpha, _, _ = cpsat(wl, time_limit=time_limit)
        wall = time.perf_counter() - t0
        if t is not None and alpha is not None:
            ms = _list_schedule_alpha(wl, _alpha_to_list(alpha))
            cpsat_record = {
                "alpha_indices": _alpha_to_list(alpha),
                "makespan_us": float(ms) if ms is not None else float("nan"),
                "solver_wall_time_s": wall,
            }
    except Exception as exc:
        print(f"  cpsat failed: {exc}")

    # Random placements.
    placement_records: List[Dict] = []
    n_combos = len(d["machine_combinations"])
    for s in range(n_placements):
        alpha_indices = []
        for op in d["operations"]:
            feasible = [k for k in range(n_combos) if k not in op["infeasible_combinations"]]
            if not feasible:
                alpha_indices.append(0)
            else:
                alpha_indices.append(int(rng.choice(feasible)))
        ms = _list_schedule_alpha(wl, alpha_indices)
        if ms is None:
            continue
        placement_records.append({
            "alpha_indices": alpha_indices,
            "makespan_us": float(ms),
        })
    return cpsat_record, placement_records


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    sp_syn = sub.add_parser("synthesize", help="generate workloads + save")
    sp_syn.add_argument("--n-per-family", type=int, default=20)
    sp_syn.add_argument("--n-per-real", type=int, default=10)
    sp_syn.add_argument("--seed", type=int, default=0)
    sp_syn.add_argument("--out", default=str(REPO / "data" / "training" / "workloads.pkl"))

    sp_lab = sub.add_parser("label", help="run CP-SAT + sample placements")
    sp_lab.add_argument("--workloads", default=str(REPO / "data" / "training" / "workloads.pkl"))
    sp_lab.add_argument("--time-limit", type=float, default=15.0)
    sp_lab.add_argument("--n-placements", type=int, default=8)
    sp_lab.add_argument("--seed", type=int, default=0)
    sp_lab.add_argument("--out-labels", default=str(REPO / "data" / "training" / "cpsat_labels.pkl"))
    sp_lab.add_argument("--out-placements", default=str(REPO / "data" / "training" / "placements.pkl"))
    sp_lab.add_argument("--out-splits", default=str(REPO / "data" / "training" / "splits.json"))

    args = ap.parse_args()

    out_dir = REPO / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "synthesize":
        print(f"Synthesizing workloads...")
        synth = synthesize_dataset(n_per_family=args.n_per_family, seed=args.seed)
        print(f"  synthetic: {len(synth)}")
        real = real_anchored_dataset(n_per_model=args.n_per_real, seed=args.seed)
        print(f"  real-anchored: {len(real)}")
        all_workloads = synth + real
        # tag with workload_id
        for i, w in enumerate(all_workloads):
            w["workload_id"] = i
        with open(args.out, "wb") as f:
            pickle.dump(all_workloads, f)
        print(f"Total {len(all_workloads)} workloads -> {args.out}")
        # Print topology distribution
        from collections import Counter
        topo_counts = Counter(w["topology"] for w in all_workloads)
        print(f"  by topology: {dict(topo_counts)}")
        op_counts = [w["n_ops"] for w in all_workloads]
        print(f"  op counts: min={min(op_counts)} max={max(op_counts)} mean={np.mean(op_counts):.1f}")

    elif args.stage == "label":
        with open(args.workloads, "rb") as f:
            workloads = pickle.load(f)
        print(f"Loaded {len(workloads)} workloads.")
        labels: List[Optional[Dict]] = []
        placements: List[Dict] = []
        for i, wd in enumerate(workloads):
            print(f"  [{i+1}/{len(workloads)}] {wd['topology']:<15s} n={wd['n_ops']}")
            t_start = time.perf_counter()
            cpsat, plac = label_workload(wd, time_limit=args.time_limit,
                                         n_placements=args.n_placements,
                                         seed=args.seed + i)
            t_used = time.perf_counter() - t_start
            if cpsat is not None:
                cpsat["workload_id"] = wd["workload_id"]
            for p in plac:
                p["workload_id"] = wd["workload_id"]
            labels.append(cpsat)
            placements.extend(plac)
            print(f"     cpsat={'ok' if cpsat else 'fail'} placements={len(plac)} t={t_used:.1f}s")

        with open(args.out_labels, "wb") as f:
            pickle.dump([x for x in labels if x is not None], f)
        with open(args.out_placements, "wb") as f:
            pickle.dump(placements, f)
        print(f"\nLabels: {sum(1 for x in labels if x is not None)} CP-SAT / {len(workloads)} workloads"
              f" -> {args.out_labels}")
        print(f"Placements: {len(placements)} -> {args.out_placements}")

        # Stratified train/val/test split (80/10/10) by topology family.
        rng = random.Random(args.seed)
        by_topo: Dict[str, List[int]] = {}
        for wd in workloads:
            by_topo.setdefault(wd["topology"], []).append(wd["workload_id"])
        train, val, test = [], [], []
        for topo, ids in by_topo.items():
            rng.shuffle(ids)
            n = len(ids)
            i_val = int(n * 0.8)
            i_test = int(n * 0.9)
            train.extend(ids[:i_val])
            val.extend(ids[i_val:i_test])
            test.extend(ids[i_test:])
        with open(args.out_splits, "w") as f:
            json.dump({"train": sorted(train), "val": sorted(val), "test": sorted(test)}, f, indent=2)
        print(f"Splits: train={len(train)} val={len(val)} test={len(test)} -> {args.out_splits}")


if __name__ == "__main__":
    main()
