"""
M11 — Learned cost model + cost_model scheduler.

Implements an MLP regressor over hand-crafted (workload, placement) features
that predicts log(makespan / lower_bound). Used as a fast oracle inside
``rewrite.score_candidates`` (M9) and as a search-based scheduler that
evaluates HEFT + perturbations and picks the lowest predicted makespan.

Layout:
  ``compute_features(workload, alpha_indices) -> np.ndarray`` —
      flat feature vector for one (workload, placement) pair
  ``CostModel`` — PyTorch nn.Module wrapping a small MLP
  ``cost_model_score(workload, t, alpha) -> float`` — single inference call
  ``cost_model_scheduler(workload)`` — registered scheduler entry

PyTorch imports are lazy so the registry stays usable without torch.

Future direction (M12/M13): if the MLP saturates below the ρ ≥ 0.7 target,
swap in a GNN encoder. The feature pipeline keeps both options open.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------

# Feature vector layout (per (workload, placement)):
#   workload-level (independent of placement):
#     [0]  n_ops_log
#     [1]  n_machines
#     [2]  n_combos
#     [3]  mean op cost (across feasible combos, log)
#     [4]  std  op cost
#     [5]  critical_path on fastest-device-per-op (log)
#     [6]  total compute on fastest-device-per-op (log)
#     [7]  avg in-degree
#     [8]  avg out-degree
#     [9]  max topological depth
#    [10]  fraction of ops with NPU infeasibility
#   placement-level:
#    [11]  per-machine total load on CPU-equivalent
#    [12]  per-machine total load on GPU-equivalent
#    [13]  per-machine total load on NPU-equivalent
#    [14]  load std across machines
#    [15]  max machine load
#    [16]  num cross-device transitions
#    [17]  total transfer cost (sum of edge transfer when src/dst differ)
#    [18]  fraction of ops on machine 0
#    [19]  fraction of ops on machine 1
#    [20]  fraction of ops on machine 2
FEATURE_DIM = 21


def _safe_log(x: float, eps: float = 1.0) -> float:
    return float(np.log(max(eps, x)))


def _lower_bound_makespan(workload: Workload) -> float:
    """Critical-path-on-fastest-device-per-op lower bound. Used as
    normalization denominator in log-ratio prediction."""
    n = len(workload.operations)
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}
    # Topological order.
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(workload.operations):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
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

    cp = [0.0] * n
    for u in order:
        op = workload.operations[u]
        feas = [c for c in op.processing_times if c < 1e8]
        own = float(min(feas)) if feas else 0.0
        max_pred = 0.0
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None and cp[pi] > max_pred:
                max_pred = cp[pi]
        cp[u] = max_pred + own
    return max(cp) if cp else 1.0


def compute_features(workload: Workload, alpha_indices: List[int]) -> np.ndarray:
    """Build the FEATURE_DIM-vector for one (workload, placement) pair.
    ``alpha_indices[i]`` is the chosen combination index for op i."""
    n = len(workload.operations)
    combos = workload.get_machine_combinations()
    machines = list(workload.machines)
    transfer = workload.get_transfer_times()
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}

    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    if n == 0:
        return feats

    # Workload-level.
    feats[0] = _safe_log(n)
    feats[1] = len(machines)
    feats[2] = len(combos)

    feasible_costs: List[float] = []
    npu_infeasible_count = 0
    in_deg: List[int] = []
    out_deg = [0] * n
    for i, op in enumerate(workload.operations):
        feas = [c for c in op.processing_times if c < 1e8]
        if feas:
            feasible_costs.extend(feas)
        if 2 in op.infeasible_combinations:  # NPU = combo index 2 (3-machine SoC convention)
            npu_infeasible_count += 1
        in_deg.append(len(op.get_predecessors()))
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                out_deg[pi] += 1
    feats[3] = _safe_log(float(np.mean(feasible_costs))) if feasible_costs else 0.0
    feats[4] = float(np.std(feasible_costs)) if feasible_costs else 0.0
    feats[5] = _safe_log(_lower_bound_makespan(workload))
    feats[6] = _safe_log(sum(min((c for c in op.processing_times if c < 1e8), default=0.0)
                             for op in workload.operations))
    feats[7] = float(np.mean(in_deg)) if in_deg else 0.0
    feats[8] = float(np.mean(out_deg)) if out_deg else 0.0
    # Max topological depth.
    depth = [0] * n
    indeg = [d for d in in_deg]
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(workload.operations):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                succ[pi].append(i)
    queue = [i for i in range(n) if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    feats[9] = max(depth) if depth else 0
    feats[10] = npu_infeasible_count / n

    # Placement-level.
    name_to_idx = {m: i for i, m in enumerate(machines)}
    per_machine_load = np.zeros(min(3, len(machines)))
    cross_device = 0
    transfer_cost_total = 0.0
    machine_op_counts = np.zeros(min(3, len(machines)))
    for i, op in enumerate(workload.operations):
        k = alpha_indices[i]
        dur = float(op.get_duration_for_combination(k, combos, machines))
        for m in combos[k]:
            mi = name_to_idx.get(m)
            if mi is not None and mi < len(per_machine_load):
                per_machine_load[mi] += dur
                machine_op_counts[mi] += 1
        # Cross-device transitions
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is None:
                continue
            pred_machines = combos[alpha_indices[pi]]
            cur_machines = combos[k]
            if not (set(pred_machines) & set(cur_machines)):
                cross_device += 1
                # Approximate transfer cost (worst-case)
                for ma in pred_machines:
                    for mb in cur_machines:
                        ia, ib = name_to_idx.get(ma), name_to_idx.get(mb)
                        if ia is None or ib is None or ia == ib:
                            continue
                        transfer_cost_total = max(transfer_cost_total, float(transfer[ia][ib]))

    # Pad / truncate to 3 machines.
    cpu_load = per_machine_load[0] if len(per_machine_load) > 0 else 0.0
    gpu_load = per_machine_load[1] if len(per_machine_load) > 1 else 0.0
    npu_load = per_machine_load[2] if len(per_machine_load) > 2 else 0.0
    feats[11] = _safe_log(cpu_load + 1)
    feats[12] = _safe_log(gpu_load + 1)
    feats[13] = _safe_log(npu_load + 1)
    feats[14] = float(np.std(per_machine_load))
    feats[15] = _safe_log(max(per_machine_load) + 1) if len(per_machine_load) > 0 else 0.0
    feats[16] = cross_device
    feats[17] = _safe_log(transfer_cost_total + 1)
    total_ops = max(1, machine_op_counts.sum())
    feats[18] = machine_op_counts[0] / total_ops if len(machine_op_counts) > 0 else 0
    feats[19] = machine_op_counts[1] / total_ops if len(machine_op_counts) > 1 else 0
    feats[20] = machine_op_counts[2] / total_ops if len(machine_op_counts) > 2 else 0
    return feats


def _alpha_to_indices(alpha) -> List[int]:
    arr = np.asarray(alpha)
    return [int(np.argmax(arr[i])) for i in range(arr.shape[0])]


# ----------------------------------------------------------------------------
# Cost model (lazy torch)
# ----------------------------------------------------------------------------


_MODEL_CACHE: Dict[str, Any] = {}


def _lazy_torch():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for the cost_model scheduler. "
            "Install via `pip install torch`."
        ) from exc


def _build_mlp(input_dim: int = FEATURE_DIM, hidden: int = 64):
    torch = _lazy_torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    )


def load_cost_model(path: Optional[str] = None):
    """Load a trained model. Caches by path to avoid repeated loads."""
    torch = _lazy_torch()
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "models" / "cost_model_v1.pt")
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not Path(path).exists():
        raise FileNotFoundError(f"cost model checkpoint not found: {path}")
    model = _build_mlp()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    _MODEL_CACHE[path] = model
    return model


def cost_model_score(workload: Workload, t, alpha, *,
                     model_path: Optional[str] = None) -> float:
    """Predicted makespan (microseconds) for the placement encoded by alpha.
    The model predicts log(makespan / lower_bound); we exponentiate and
    multiply by lower_bound to return absolute microseconds."""
    torch = _lazy_torch()
    model = load_cost_model(model_path)
    alpha_indices = _alpha_to_indices(alpha)
    feats = compute_features(workload, alpha_indices)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_log_ratio = float(model(x).item())
    lb = _lower_bound_makespan(workload)
    return float(np.exp(pred_log_ratio) * lb)


# ----------------------------------------------------------------------------
# Cost-model-guided scheduler
# ----------------------------------------------------------------------------


def cost_model_scheduler(workload: Workload, *,
                         n_candidates: int = 32,
                         model_path: Optional[str] = None,
                         random_seed: int = 0,
                         **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    """Generate HEFT + N perturbations, predict makespan for each, pick the
    best. Materialize the chosen alpha by list-scheduling.

    Each perturbation flips one randomly-chosen op's combination to a
    different feasible one.
    """
    from scheduler_heft import (
        heft, _list_schedule, _build_topo_order, _feasible_combinations,
        _earliest_start_on_combo,
    )

    rng = np.random.default_rng(random_seed)
    ops = workload.operations
    n = len(ops)
    combos = workload.get_machine_combinations()
    n_combos = len(combos)
    if n == 0:
        return np.zeros(0), np.zeros((0, n_combos)), None, None

    # Baseline = HEFT.
    base_t, base_alpha, _, _ = heft(workload)
    base_alpha_indices = _alpha_to_indices(base_alpha)

    # Try to load model; if missing, just return HEFT.
    try:
        load_cost_model(model_path)
    except Exception as exc:
        print(f"[cost_model_scheduler] no model loaded ({exc}); falling back to HEFT")
        return base_t, base_alpha, None, None

    candidates: List[Tuple[float, List[int]]] = []
    base_score = cost_model_score(workload, base_t, base_alpha, model_path=model_path)
    candidates.append((base_score, list(base_alpha_indices)))

    # Perturbations.
    for _ in range(n_candidates):
        new_indices = list(base_alpha_indices)
        # Pick a random op and a random different feasible combo.
        i = int(rng.integers(0, n))
        feas = _feasible_combinations(ops[i], n_combos)
        if len(feas) <= 1:
            continue
        choices = [k for k in feas if k != new_indices[i]]
        new_indices[i] = int(rng.choice(choices))
        # Construct alpha matrix for scoring.
        new_alpha = np.zeros((n, n_combos))
        for j, k in enumerate(new_indices):
            new_alpha[j, k] = 1.0
        # Use dummy t (model doesn't use t directly; features are placement-only).
        try:
            score = cost_model_score(workload, base_t, new_alpha, model_path=model_path)
        except Exception:
            continue
        candidates.append((score, new_indices))

    # Pick best (lowest predicted makespan).
    candidates.sort(key=lambda x: x[0])
    best_indices = candidates[0][1]

    # Materialize: list-schedule with the chosen assignment.
    machines = list(workload.machines)
    op_idx_map = {id(op): i for i, op in enumerate(ops)}
    order = _build_topo_order(ops)
    t_new = np.zeros(n)
    machine_busy: Dict[str, float] = {m: 0.0 for m in machines}
    pred_finish: Dict[int, float] = {}
    pred_combo: Dict[int, int] = {}
    for i in order:
        op = ops[i]
        k = best_indices[i]
        if k in op.infeasible_combinations:
            # Fall back to first feasible
            feas = _feasible_combinations(op, n_combos)
            k = feas[0] if feas else k
            best_indices[i] = k
        est = _earliest_start_on_combo(workload, op, k, pred_finish, pred_combo, machine_busy)
        t_new[i] = est
        dur = float(op.get_duration_for_combination(k, combos, machines))
        pred_finish[i] = est + dur
        pred_combo[i] = k
        for m in combos[k]:
            machine_busy[m] = est + dur

    alpha_new = np.zeros((n, n_combos))
    for i, k in enumerate(best_indices):
        alpha_new[i, k] = 1.0
    return t_new, alpha_new, None, None
