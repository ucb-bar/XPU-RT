"""
M11/M12/M13 unified trainer entry point.

Subcommands:
  --target cost_model   train MLP regressor on (workload, placement) -> log(makespan/lower_bound)
  --target gnn          (placeholder; populated by M12)
  --target rl           (placeholder; populated by M13)

Cost-model training reads:
  data/training/workloads.pkl
  data/training/cpsat_labels.pkl
  data/training/placements.pkl
  data/training/splits.json

Writes:
  data/models/cost_model_v1.pt
  data/models/cost_model_v1_eval.json   (held-out Spearman rho)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))
sys.path.insert(0, str(REPO / "scripts"))

from scheduler_ml import compute_features, FEATURE_DIM, _lower_bound_makespan, _build_mlp  # noqa: E402
from gen_training_data import _workload_from_dict  # noqa: E402


# ----------------------------------------------------------------------------
# Cost model training
# ----------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _build_dataset(workloads: List[dict], placements: List[dict],
                   cpsat_labels: List[dict], split_ids: set) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (X, y, workload_ids) where y = log(makespan / lower_bound)."""
    workloads_by_id = {w["workload_id"]: w for w in workloads}
    lb_cache: Dict[int, float] = {}

    samples: List[Tuple[np.ndarray, float, int]] = []
    # Include CP-SAT-labeled placements (optimal).
    sources = list(placements)
    sources.extend(cpsat_labels)

    for s in sources:
        wid = s["workload_id"]
        if wid not in split_ids:
            continue
        wd = workloads_by_id[wid]
        if wid not in lb_cache:
            wl = _workload_from_dict(wd)
            lb_cache[wid] = _lower_bound_makespan(wl)
            # Stash for feature extraction
            samples_workload = wl
        else:
            samples_workload = _workload_from_dict(wd)
        lb = lb_cache[wid]
        ms = s["makespan_us"]
        if not (ms > 0):
            continue
        feats = compute_features(samples_workload, s["alpha_indices"])
        y = float(np.log(ms / max(lb, 1.0)))
        samples.append((feats, y, wid))

    if not samples:
        return np.zeros((0, FEATURE_DIM)), np.zeros(0), []
    X = np.stack([s[0] for s in samples])
    y = np.array([s[1] for s in samples], dtype=np.float32)
    wids = [s[2] for s in samples]
    return X, y, wids


def train_cost_model(args):
    import torch

    print(f"=== train cost_model ===")
    data_dir = REPO / "data" / "training"
    with open(data_dir / "workloads.pkl", "rb") as f:
        workloads = pickle.load(f)
    with open(data_dir / "placements.pkl", "rb") as f:
        placements = pickle.load(f)
    cpsat_path = data_dir / "cpsat_labels.pkl"
    if cpsat_path.exists():
        with open(cpsat_path, "rb") as f:
            cpsat_labels = pickle.load(f)
    else:
        cpsat_labels = []
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    print(f"  workloads: {len(workloads)}")
    print(f"  placements: {len(placements)}")
    print(f"  cpsat labels: {len(cpsat_labels)}")
    print(f"  splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    X_tr, y_tr, wids_tr = _build_dataset(workloads, placements, cpsat_labels, set(splits["train"]))
    X_va, y_va, wids_va = _build_dataset(workloads, placements, cpsat_labels, set(splits["val"]))
    X_te, y_te, wids_te = _build_dataset(workloads, placements, cpsat_labels, set(splits["test"]))
    print(f"  samples: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Normalize features by train-set statistics
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    def _norm(X):
        return (X - mean) / std

    Xn_tr = _norm(X_tr)
    Xn_va = _norm(X_va)
    Xn_te = _norm(X_te)

    model = _build_mlp(input_dim=FEATURE_DIM, hidden=args.hidden)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss(delta=0.5)

    t_tr = torch.tensor(Xn_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
    t_va = torch.tensor(Xn_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(-1)

    n = len(t_tr)
    best_rho_va = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = t_tr[idx]; yb = y_tr_t[idx]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(idx)
        total_loss /= max(1, n)
        # Eval on val
        model.eval()
        with torch.no_grad():
            pred_va = model(t_va).squeeze(-1).cpu().numpy()
            pred_tr = model(t_tr).squeeze(-1).cpu().numpy()
        rho_va = _spearman(pred_va, y_va)
        rho_tr = _spearman(pred_tr, y_tr)
        if rho_va > best_rho_va:
            best_rho_va = rho_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3d}  train_loss={total_loss:.4f}  "
                  f"rho_tr={rho_tr:.3f}  rho_va={rho_va:.3f}  "
                  f"best_va={best_rho_va:.3f}")
        history.append({"epoch": epoch + 1, "loss": total_loss,
                        "rho_train": float(rho_tr), "rho_val": float(rho_va)})

    if best_state is not None:
        model.load_state_dict(best_state)
    # Test
    t_te = torch.tensor(Xn_te, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_te = model(t_te).squeeze(-1).cpu().numpy()
    rho_te = _spearman(pred_te, y_te)
    print(f"\nFinal: rho_val={best_rho_va:.3f}  rho_test={rho_te:.3f}")

    # Save
    models_dir = REPO / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "cost_model_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_dim": FEATURE_DIM,
        "rho_val_best": float(best_rho_va),
        "rho_test": float(rho_te),
    }, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    eval_path = models_dir / "cost_model_v1_eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "n_train": len(X_tr),
            "n_val": len(X_va),
            "n_test": len(X_te),
            "rho_val_best": float(best_rho_va),
            "rho_test": float(rho_te),
            "epochs": args.epochs,
            "hidden": args.hidden,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "history": history,
        }, f, indent=2)
    print(f"Eval -> {eval_path}")
    return best_rho_va, rho_te


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def train_gnn(args):
    """M12: GNN-based placement, supervised on CP-SAT-optimal alphas.

    Phase 1 only (cross-entropy); REINFORCE Phase 2 deferred.
    """
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGLoader

    from scheduler_gnn import (
        FEAT_PER_NODE, MAX_COMBOS, build_gnn_model,
        compute_node_features, compute_edge_index, compute_infeasibility_mask,
    )

    print("=== train gnn_placement ===")
    data_dir = REPO / "data" / "training"
    with open(data_dir / "workloads.pkl", "rb") as f:
        workloads = pickle.load(f)
    with open(data_dir / "cpsat_labels.pkl", "rb") as f:
        cpsat_labels = pickle.load(f)
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    print(f"  workloads={len(workloads)} cpsat_labels={len(cpsat_labels)}")
    print(f"  splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    by_id = {w["workload_id"]: w for w in workloads}
    label_by_id = {l["workload_id"]: l for l in cpsat_labels}

    def _make_data_list(ids):
        out = []
        for wid in ids:
            wd = by_id.get(wid)
            lbl = label_by_id.get(wid)
            if wd is None or lbl is None:
                continue
            wl = _workload_from_dict(wd)
            if len(wl.operations) == 0:
                continue
            feats = compute_node_features(wl)
            ei = compute_edge_index(wl)
            mask = compute_infeasibility_mask(wl)
            y = np.array(lbl["alpha_indices"], dtype=np.int64)
            data = Data(
                x=torch.tensor(feats, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long),
                y=torch.tensor(y, dtype=torch.long),
                inf_mask=torch.tensor(mask, dtype=torch.bool),
                num_nodes=feats.shape[0],
            )
            out.append(data)
        return out

    train_list = _make_data_list(splits["train"])
    val_list = _make_data_list(splits["val"])
    test_list = _make_data_list(splits["test"])
    print(f"  graphs: train={len(train_list)} val={len(val_list)} test={len(test_list)}")

    # Normalize features by train-set per-column statistics.
    train_X = np.concatenate([d.x.numpy() for d in train_list], axis=0)
    mean = train_X.mean(axis=0).astype(np.float32)
    std = (train_X.std(axis=0) + 1e-6).astype(np.float32)

    def _normalize(lst):
        for d in lst:
            d.x = (d.x - torch.tensor(mean)) / torch.tensor(std)
    _normalize(train_list); _normalize(val_list); _normalize(test_list)

    train_loader = PyGLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGLoader(val_list, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGLoader(test_list, batch_size=args.batch_size, shuffle=False)

    model = build_gnn_model(in_dim=FEAT_PER_NODE, hidden=args.hidden,
                            n_combos=MAX_COMBOS, n_layers=3)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    def _eval(loader):
        model.eval()
        n_correct = 0
        n_total = 0
        ws_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                logits = model(batch.x, batch.edge_index)
                logits = logits.clone()
                logits[batch.inf_mask] = -1e9
                pred = logits.argmax(dim=-1)
                n_correct += int((pred == batch.y).sum().item())
                n_total += int(batch.y.numel())
                ws_loss += float(nn.functional.cross_entropy(logits, batch.y).item())
                n_batches += 1
        return n_correct / max(1, n_total), ws_loss / max(1, n_batches)

    best_val_acc = 0.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            logits = model(batch.x, batch.edge_index)
            logits = logits.clone()
            logits[batch.inf_mask] = -1e9
            loss = nn.functional.cross_entropy(logits, batch.y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        train_acc, _ = _eval(train_loader)
        val_acc, val_loss = _eval(val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>3d}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"best_val={best_val_acc:.3f}")
        history.append({"epoch": epoch + 1, "loss": float(avg_loss),
                        "train_acc": float(train_acc), "val_acc": float(val_acc)})

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, _ = _eval(test_loader)
    print(f"\nFinal: val_acc={best_val_acc:.3f}  test_acc={test_acc:.3f}")

    # Save
    models_dir = REPO / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "gnn_placement_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": FEAT_PER_NODE,
        "hidden": args.hidden,
        "n_combos": MAX_COMBOS,
        "n_layers": 3,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "val_acc_best": float(best_val_acc),
        "test_acc": float(test_acc),
    }, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    eval_path = models_dir / "gnn_placement_v1_eval.json"
    with open(eval_path, "w") as f:
        json.dump({
            "n_train": len(train_list), "n_val": len(val_list), "n_test": len(test_list),
            "val_acc_best": float(best_val_acc), "test_acc": float(test_acc),
            "epochs": args.epochs, "hidden": args.hidden, "lr": args.lr,
            "history": history,
        }, f, indent=2)
    print(f"Eval -> {eval_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["cost_model", "gnn", "rl"], default="cost_model")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    if args.target == "cost_model":
        train_cost_model(args)
    elif args.target == "gnn":
        train_gnn(args)
    else:
        print(f"target={args.target} not implemented yet (M13 follow-up)")


if __name__ == "__main__":
    main()
