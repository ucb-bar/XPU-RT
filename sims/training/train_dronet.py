#!/usr/bin/env python3
"""Supervised training of DroNet's steering head on the IDSIA Trails corpus.

Uses ``qnn_models.dronet.DronetTorch`` (architecturally identical to the inlined
copies in ``sims/scripts/play_dronet*.py``) so the resulting ``state_dict.pt``
drops directly into the ``--dronet_weights`` arg of those scripts.

Only the steering head (``linear1``) is trained; the collision head
(``linear2`` + sigmoid) is left at its initialization. Targets are continuous
yaw-rate setpoints derived from camera identity in IDSIA (see
``dataset_idsia.py``).

Usage::

    python sims/training/train_dronet.py
    python sims/training/train_dronet.py --epochs 30 --batch_size 64 \\
        --img_size 112 --omega_max 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qnn_models.dronet import DronetTorch  # noqa: E402
from sims.training.dataset_idsia import IDSIAConfig, IDSIATrailDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path,
                   default=REPO_ROOT / "datasets/idsia/extracted",
                   help="Extracted IDSIA tree (output of extract_idsia.py).")
    p.add_argument("--out_dir", type=Path,
                   default=REPO_ROOT / "logs/dronet",
                   help="Directory to write checkpoints under, with a timestamped subdir.")

    # Model
    p.add_argument("--model_size", choices=["small", "large"], default="small",
                   help="'small' = 112x112 input, 2048-dim flatten; 'large' = 224x224, 6272-dim.")
    p.add_argument("--img_size", type=int, default=None,
                   help="Override input edge length. Default: 112 for small, 224 for large.")

    # Optimization
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    # Data
    p.add_argument("--omega_max", type=float, default=1.0,
                   help="Magnitude of the yaw-rate setpoint for lc/rc samples (sc => 0).")
    p.add_argument("--val_segments", nargs="*", default=["011", "012"],
                   help="IDSIA segment IDs to hold out for validation.")

    # Misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20, help="Steps between training-loss prints.")
    return p.parse_args()


def steering_only_loss(steer_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the steering head only (collision head dropped per agreed plan)."""
    return torch.nn.functional.mse_loss(steer_pred, target)


def evaluate(model: DronetTorch, loader: DataLoader, device: str) -> dict:
    """Return dict with val MSE and per-class mean prediction for debugging."""
    model.eval()
    total_mse = 0.0
    n = 0
    sums = {"lc": 0.0, "sc": 0.0, "rc": 0.0}
    counts = {"lc": 0, "sc": 0, "rc": 0}

    with torch.no_grad():
        for img, target in loader:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            steer_pred, _ = model(img)
            total_mse += torch.nn.functional.mse_loss(steer_pred, target, reduction="sum").item()
            n += target.numel()

            # Bin predictions by target sign — coarse, but tells us if the
            # network distinguishes the three classes at all.
            t = target.squeeze(1).cpu()
            s = steer_pred.squeeze(1).cpu()
            for ti, si in zip(t.tolist(), s.tolist()):
                key = "sc" if abs(ti) < 1e-6 else ("rc" if ti > 0 else "lc")
                sums[key] += si
                counts[key] += 1

    means = {k: (sums[k] / counts[k] if counts[k] else float("nan")) for k in sums}
    return {"mse": total_mse / max(1, n), "means_by_class": means, "counts": counts}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.img_size is None:
        args.img_size = 112 if args.model_size == "small" else 224

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] run dir: {run_dir}")
    with open(run_dir / "config.json", "w") as f:
        cfg_serializable = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        json.dump(cfg_serializable, f, indent=2)

    # --- data ---
    base_cfg = IDSIAConfig(
        root=args.data_root,
        img_size=args.img_size,
        omega_max=args.omega_max,
        augment=True,
        val_segments=tuple(args.val_segments),
        seed=args.seed,
    )
    train_cfg = IDSIAConfig(**{**base_cfg.__dict__, "split": "train"})
    val_cfg = IDSIAConfig(**{**base_cfg.__dict__, "split": "val", "augment": False})
    train_ds = IDSIATrailDataset(train_cfg)
    val_ds = IDSIATrailDataset(val_cfg)

    print(f"[info] train: n={len(train_ds)} counts={train_ds.class_counts()}")
    print(f"[info] val:   n={len(val_ds)} counts={val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- model ---
    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=3,
        output_dim=1,
        small=(args.model_size == "small"),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] DronetTorch: small={args.model_size == 'small'} input={args.img_size}x{args.img_size} params={n_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- train ---
    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_count = 0
        for step, (img, target) in enumerate(train_loader, 1):
            img = img.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            steer_pred, _collision = model(img)
            loss = steering_only_loss(steer_pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * target.size(0)
            train_count += target.size(0)
            if step % args.log_every == 0:
                running = train_loss_sum / max(1, train_count)
                print(f"  ep{epoch:02d} step {step:4d}/{len(train_loader)} "
                      f"loss={loss.item():.4f} avg={running:.4f}", flush=True)

        scheduler.step()
        train_mse = train_loss_sum / max(1, train_count)
        val = evaluate(model, val_loader, args.device)
        elapsed = time.time() - t0
        print(f"[ep{epoch:02d}] train_mse={train_mse:.4f}  val_mse={val['mse']:.4f}  "
              f"means_by_class={ {k: f'{v:+.3f}' for k, v in val['means_by_class'].items()} }  "
              f"({elapsed:.1f}s)")

        # Save plain state_dict — drops into sim scripts via --dronet_weights.
        torch.save(model.state_dict(), run_dir / "last.pt")
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  ^ new best val_mse={best_val:.4f}, saved best.pt")

        history.append({"epoch": epoch, "train_mse": train_mse, **val,
                        "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]})
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"[done] best val_mse={best_val:.4f}")
    print(f"[done] checkpoints in: {run_dir}")
    print(f"[done] use with: --dronet_weights {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
