#!/usr/bin/env python3
"""Finetune DroNet from an existing checkpoint on IDSIA forest-trail data.

Loads a pre-trained checkpoint and continues training with a lower learning
rate. Saves to a new timestamped run directory — the original checkpoint is
never modified.

Usage::

    python sims/training/finetune_dronet.py \
        --checkpoint logs/dronet/2026-04-27_17-10-41/best.pt \
        --epochs 20 --lr 1e-4

    # Then use the finetuned weights in the pilot:
    python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \
        --dronet_weights logs/dronet/<new_run>/best.pt ...
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
from sims.training.dataset_sim import SimDataConfig, SimTrailDataset  # noqa: E402
from sims.training.train_dronet import evaluate, steering_only_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the pre-trained DroNet state_dict .pt file.")
    p.add_argument("--data_root", type=Path,
                   default=REPO_ROOT / "datasets/idsia/extracted",
                   help="Extracted IDSIA tree (or sim data segment dir).")
    p.add_argument("--sim_data", action="store_true",
                   help="Treat --data_root as sim-collected data (labels.csv).")
    p.add_argument("--out_dir", type=Path,
                   default=REPO_ROOT / "logs/dronet",
                   help="Parent directory for the finetuned run.")

    # Model
    p.add_argument("--model_size", choices=["small", "large"], default="small")
    p.add_argument("--img_size", type=int, default=None)

    # Optimization — lower LR for finetuning
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default 1e-4, 10x lower than from-scratch).")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    # Data
    p.add_argument("--omega_max", type=float, default=1.0)
    p.add_argument("--val_segments", nargs="*", default=["011", "012"])

    # Misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.img_size is None:
        args.img_size = 112 if args.model_size == "small" else 224

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out_dir / f"{timestamp}_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] finetune run dir: {run_dir}")
    print(f"[info] base checkpoint: {args.checkpoint}")

    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg_dict["base_checkpoint"] = str(args.checkpoint)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # --- data ---
    if args.sim_data:
        train_ds = SimTrailDataset(SimDataConfig(
            root=args.data_root, img_size=args.img_size,
            augment=True, split="train", seed=args.seed,
        ))
        val_ds = SimTrailDataset(SimDataConfig(
            root=args.data_root, img_size=args.img_size,
            augment=False, split="val", seed=args.seed,
        ))
    else:
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

    # --- model (load from checkpoint) ---
    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=3,
        output_dim=1,
        small=(args.model_size == "small"),
    ).to(args.device)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] loaded checkpoint: {args.checkpoint}")
    print(f"[info] DronetTorch: small={args.model_size == 'small'} "
          f"input={args.img_size}x{args.img_size} params={n_params/1e6:.2f}M")

    # Evaluate baseline before finetuning
    val_baseline = evaluate(model, val_loader, args.device)
    print(f"[baseline] val_mse={val_baseline['mse']:.4f}  "
          f"means_by_class={ {k: f'{v:+.3f}' for k, v in val_baseline['means_by_class'].items()} }")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- finetune ---
    best_val = val_baseline["mse"]
    # Save the baseline as a starting point (so best.pt always exists)
    torch.save(model.state_dict(), run_dir / "best.pt")
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

        torch.save(model.state_dict(), run_dir / "last.pt")
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  ^ new best val_mse={best_val:.4f}, saved best.pt")

        history.append({"epoch": epoch, "train_mse": train_mse, **val,
                        "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]})
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[done] baseline val_mse={val_baseline['mse']:.4f} → best val_mse={best_val:.4f}")
    print(f"[done] checkpoints in: {run_dir}")
    print(f"[done] use with: --dronet_weights {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
