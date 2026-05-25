"""
M12 — GNN-based placement scheduler.

Architecture: small GraphSAGE (3 layers, hidden=64) over the op-graph with
per-node features. Per-op classification head produces logits over machine
combinations; infeasible combinations are masked to -inf before softmax.

Training (in scripts/train_ml.py --target gnn): supervised cross-entropy
against CP-SAT-optimal placements (Phase 1; REINFORCE phase deferred).

Inference: forward pass -> softmax argmax -> list-schedule pass to
materialise (t, alpha). HEFT fallback gate: if predicted alpha's
list-scheduled makespan > 2x HEFT makespan, fall back to HEFT.

Lazy torch / torch_geometric imports so the registry stays usable without
either library installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload


# ---------------------------------------------------------------------------
# Per-op feature scheme (shared with the trainer)
# ---------------------------------------------------------------------------

MAX_COMBOS = 3   # 3-machine SoC convention used across the project
FEAT_PER_NODE = 6 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # see compute_node_features


def _safe_log(x: float, eps: float = 1.0) -> float:
    return float(np.log(max(eps, float(x))))


def compute_node_features(workload: Workload) -> np.ndarray:
    """Return ``[N, FEAT_PER_NODE]`` array of per-op features for a workload.

    Layout per row:
      [0..MAX_COMBOS-1]              processing_times (log-scaled, padded to MAX_COMBOS)
      [MAX_COMBOS..2*MAX_COMBOS-1]   feasibility mask (1 = feasible)
      [2*MAX_COMBOS]                 output_bytes (log)
      [2*MAX_COMBOS+1]               has_deadline
      [2*MAX_COMBOS+2]               has_release
      [2*MAX_COMBOS+3]               topological depth (normalized)
      [2*MAX_COMBOS+4]               heft upward rank (log)
      [2*MAX_COMBOS+5]               heft downward rank (log)
      [2*MAX_COMBOS+6]               in-degree
      [2*MAX_COMBOS+7]               out-degree
      [2*MAX_COMBOS+8]               periodic (0/1)
    """
    from scheduler_heft import _upward_rank, _build_topo_order

    ops = workload.operations
    n = len(ops)
    op_idx = {id(op): i for i, op in enumerate(ops)}

    upward = _upward_rank(workload)

    # Compute downward rank (cost from a source to this op including own).
    # We just compute "earliest finish on fastest device" per op.
    indeg = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    for i, op in enumerate(ops):
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

    downward = [0.0] * n
    depth = [0] * n
    for u in order:
        own = min((c for c in ops[u].processing_times if c < 1e8), default=0.0)
        max_pred = 0.0
        max_pred_depth = -1
        for p in ops[u].get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                if downward[pi] > max_pred:
                    max_pred = downward[pi]
                if depth[pi] > max_pred_depth:
                    max_pred_depth = depth[pi]
        downward[u] = max_pred + own
        depth[u] = max_pred_depth + 1
    max_depth = max(depth) if depth else 1

    in_deg = [0] * n
    out_deg = [0] * n
    for i, op in enumerate(ops):
        in_deg[i] = len(op.get_predecessors())
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is not None:
                out_deg[pi] += 1

    feats = np.zeros((n, FEAT_PER_NODE), dtype=np.float32)
    for i, op in enumerate(ops):
        pts = list(op.processing_times)
        # Pad processing_times to MAX_COMBOS.
        for k in range(MAX_COMBOS):
            if k < len(pts):
                feats[i, k] = _safe_log(min(1e9, pts[k]))
                feats[i, MAX_COMBOS + k] = 0.0 if k in op.infeasible_combinations else 1.0
            else:
                feats[i, k] = 0.0
                feats[i, MAX_COMBOS + k] = 0.0
        col = 2 * MAX_COMBOS
        feats[i, col + 0] = _safe_log(getattr(op, "output_bytes", 0) + 1)
        feats[i, col + 1] = 1.0 if op.deadline_us is not None else 0.0
        feats[i, col + 2] = 1.0 if op.min_start_t is not None else 0.0
        feats[i, col + 3] = depth[i] / max(1, max_depth)
        feats[i, col + 4] = _safe_log(upward[i] + 1)
        feats[i, col + 5] = _safe_log(downward[i] + 1)
        feats[i, col + 6] = in_deg[i]
        feats[i, col + 7] = out_deg[i]
        feats[i, col + 8] = 1.0 if (op.min_start_t is not None or op.deadline_us is not None) else 0.0
    return feats


def compute_edge_index(workload: Workload) -> np.ndarray:
    """Return ``[2, E]`` edge_index in PyG convention (src row, dst row).

    Precedence pred -> op is encoded as a directed edge; we also add the
    reverse edge so message passing is bidirectional (typical GNN setup).
    """
    ops = workload.operations
    op_idx = {id(op): i for i, op in enumerate(ops)}
    src: List[int] = []
    dst: List[int] = []
    for i, op in enumerate(ops):
        for p in op.get_predecessors():
            pi = op_idx.get(id(p))
            if pi is None:
                continue
            src.append(pi); dst.append(i)
            src.append(i); dst.append(pi)
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def compute_infeasibility_mask(workload: Workload) -> np.ndarray:
    """``[N, MAX_COMBOS]`` boolean mask: True = infeasible (should be -inf'd
    in logits before softmax)."""
    ops = workload.operations
    n = len(ops)
    mask = np.zeros((n, MAX_COMBOS), dtype=bool)
    for i, op in enumerate(ops):
        for k in op.infeasible_combinations:
            if k < MAX_COMBOS:
                mask[i, k] = True
        # Also mask combos beyond the workload's actual combination count.
        n_real = len(workload.get_machine_combinations())
        for k in range(n_real, MAX_COMBOS):
            mask[i, k] = True
    return mask


# ---------------------------------------------------------------------------
# Model (lazy torch + torch_geometric)
# ---------------------------------------------------------------------------


_MODEL_CACHE: Dict[str, Any] = {}


def _lazy_pyg():
    try:
        import torch  # noqa
        from torch_geometric.nn import SAGEConv  # noqa
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "torch and torch_geometric are required for the gnn_placement scheduler."
        ) from exc


def build_gnn_model(in_dim: int = FEAT_PER_NODE,
                    hidden: int = 64, n_combos: int = MAX_COMBOS, n_layers: int = 3):
    torch = _lazy_pyg()
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv

    class GNNPlacement(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            d = in_dim
            for _ in range(n_layers):
                self.layers.append(SAGEConv(d, hidden, aggr="mean"))
                d = hidden
            self.head = nn.Linear(hidden, n_combos)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)

        def forward(self, x, edge_index):
            h = x
            for conv in self.layers:
                h = self.drop(self.act(conv(h, edge_index)))
            return self.head(h)  # [N, n_combos]

    return GNNPlacement()


def load_gnn_model(path: Optional[str] = None):
    torch = _lazy_pyg()
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "models" / "gnn_placement_v1.pt")
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not Path(path).exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_gnn_model(
        in_dim=ckpt.get("in_dim", FEAT_PER_NODE),
        hidden=ckpt.get("hidden", 64),
        n_combos=ckpt.get("n_combos", MAX_COMBOS),
        n_layers=ckpt.get("n_layers", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _MODEL_CACHE[path] = (model, ckpt.get("feature_mean"), ckpt.get("feature_std"))
    return _MODEL_CACHE[path]


def predict_placement(workload: Workload,
                      model_path: Optional[str] = None) -> Optional[List[int]]:
    """Return ``[N]`` list of combo indices predicted by the GNN, or None on
    error."""
    torch = _lazy_pyg()
    try:
        model, mean, std = load_gnn_model(model_path)
    except FileNotFoundError as exc:
        print(f"[gnn_placement] {exc} — falling back to HEFT")
        return None
    feats = compute_node_features(workload)
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32) + 1e-6
        feats = (feats - mean) / std
    edge_index = compute_edge_index(workload)
    mask = compute_infeasibility_mask(workload)

    x = torch.tensor(feats, dtype=torch.float32)
    e = torch.tensor(edge_index, dtype=torch.long)
    with torch.no_grad():
        logits = model(x, e).numpy()
    # Apply infeasibility mask: -inf at masked positions
    logits[mask] = -1e9
    return logits.argmax(axis=-1).tolist()


# ---------------------------------------------------------------------------
# Scheduler entry
# ---------------------------------------------------------------------------


def gnn_placement_scheduler(workload: Workload, *,
                            model_path: Optional[str] = None,
                            fallback_ratio: float = 1.3,
                            **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    """Run the GNN to predict alpha, materialise (t, alpha) via list-
    scheduling, and fall back to HEFT if the GNN's makespan is worse than
    ``fallback_ratio * heft_makespan``."""
    from scheduler_heft import (
        heft, _build_topo_order, _feasible_combinations,
        _earliest_start_on_combo,
    )

    # HEFT baseline (also used for fallback decision).
    heft_t, heft_alpha, _, _ = heft(workload)
    combos = workload.get_machine_combinations()
    machines = list(workload.machines)
    n_combos = len(combos)
    n = len(workload.operations)
    if n == 0:
        return heft_t, heft_alpha, None, None

    heft_ms = 0.0
    for i, op in enumerate(workload.operations):
        k = int(np.argmax(heft_alpha[i]))
        d = float(op.get_duration_for_combination(k, combos, machines))
        f = float(heft_t[i]) + d
        if f > heft_ms:
            heft_ms = f

    # GNN prediction.
    predicted = predict_placement(workload, model_path=model_path)
    if predicted is None:
        return heft_t, heft_alpha, None, None

    # Validate feasibility (mask should have prevented this, but guard).
    for i, op in enumerate(workload.operations):
        if predicted[i] >= n_combos or predicted[i] in op.infeasible_combinations:
            # Pick first feasible.
            feas = _feasible_combinations(op, n_combos)
            predicted[i] = feas[0] if feas else 0

    # Materialise (t, alpha) via list-scheduling.
    order = _build_topo_order(workload.operations)
    t_new = np.zeros(n)
    machine_busy: Dict[str, float] = {m: 0.0 for m in machines}
    pred_finish: Dict[int, float] = {}
    pred_combo: Dict[int, int] = {}
    for i in order:
        op = workload.operations[i]
        k = predicted[i]
        est = _earliest_start_on_combo(workload, op, k, pred_finish, pred_combo, machine_busy)
        t_new[i] = est
        d = float(op.get_duration_for_combination(k, combos, machines))
        pred_finish[i] = est + d
        pred_combo[i] = k
        for m in combos[k]:
            machine_busy[m] = est + d

    gnn_ms = max(pred_finish.values()) if pred_finish else 0.0

    # Fallback if GNN is much worse than HEFT.
    if gnn_ms > fallback_ratio * heft_ms:
        return heft_t, heft_alpha, None, None

    alpha_new = np.zeros((n, n_combos))
    for i, k in enumerate(predicted):
        alpha_new[i, k] = 1.0
    return t_new, alpha_new, None, None
