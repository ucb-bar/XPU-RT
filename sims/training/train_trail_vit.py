"""Train the ViT-backbone trail classifier/regressor on IDSIA (#56/#23 probe).

Head-to-head with greyscale DroNet on the SAME data, loader, held-out split
(011/012), loss, and eval metric — the ONLY thing that changes is the model
(vitfly MixVisionTransformer backbone instead of the DroNet CNN). This isolates
"backbone" as the variable, so a delta over DroNet's ~0.54 ceiling is
attributable to the architecture, not the pipeline.

    <py> sims/training/train_trail_vit.py --head classifier --epochs 30
    <py> sims/training/train_trail_vit.py --head regression --epochs 30

Reuses IDSIAConfig/IDSIATrailDataset + head_loss + evaluate from train_dronet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# vitfly model zoo lives outside XPU-RT
VITFLY_MODELS = REPO_ROOT.parent / "vitfly" / "models"
if str(VITFLY_MODELS) not in sys.path:
    sys.path.insert(0, str(VITFLY_MODELS))

from sims.training.dataset_idsia import IDSIAConfig, IDSIATrailDataset  # noqa: E402
from sims.training.train_dronet import evaluate, head_loss  # noqa: E402
from trail_vit import TrailViT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "datasets/idsia/extracted")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT.parent / "train_out/trail_vit")
    p.add_argument("--head", choices=["regression", "classifier"], default="classifier")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--omega_max", type=float, default=1.0)
    p.add_argument("--val_segments", nargs="*", default=["011", "012"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] run dir: {run_dir}")
    with open(run_dir / "config.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)

    base_cfg = IDSIAConfig(
        root=args.data_root, img_size=args.img_size, omega_max=args.omega_max,
        augment=True, val_segments=tuple(args.val_segments), seed=args.seed,
        greyscale=True, return_class=True,
    )
    train_ds = IDSIATrailDataset(IDSIAConfig(**{**base_cfg.__dict__, "split": "train"}))
    val_ds = IDSIATrailDataset(IDSIAConfig(**{**base_cfg.__dict__, "split": "val", "augment": False}))
    print(f"[info] train: n={len(train_ds)} counts={train_ds.class_counts()}")
    print(f"[info] val:   n={len(val_ds)} counts={val_ds.class_counts()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TrailViT(head=args.head, dropout=args.dropout).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] TrailViT: head={args.head} input={args.img_size}x{args.img_size} "
          f"params={n_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_select = float("-inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, count = 0.0, 0
        for step, (img, target, class_idx) in enumerate(train_loader, 1):
            img = img.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            class_idx = class_idx.to(args.device, non_blocking=True)
            out, _ = model(img)
            loss = head_loss(out, target, class_idx, args.head)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target.size(0)
            count += target.size(0)
            if step % args.log_every == 0:
                print(f"  ep{epoch:02d} step {step:4d}/{len(train_loader)} "
                      f"loss={loss.item():.4f} avg={loss_sum/max(1,count):.4f}", flush=True)
        scheduler.step()
        train_loss = loss_sum / max(1, count)
        val = evaluate(model, val_loader, args.device, head=args.head)
        elapsed = time.time() - t0
        if args.head == "classifier":
            print(f"[ep{epoch:02d}] train_ce={train_loss:.4f}  val_acc={val['accuracy']:.3f}  "
                  f"recall={ {k: f'{v:.2f}' for k, v in val['per_class_recall'].items()} }  ({elapsed:.1f}s)")
        else:
            print(f"[ep{epoch:02d}] train_mse={train_loss:.4f}  val_mse={val['mse']:.4f}  "
                  f"sign_agree={val['sign_agreement']:.3f}  ({elapsed:.1f}s)")

        torch.save(model.state_dict(), run_dir / "last.pt")
        if val["select"] > best_select:
            best_select = val["select"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            metric = "val_acc" if args.head == "classifier" else "sign_agree"
            print(f"  ^ new best {metric}={best_select:.4f}, saved best.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, **val,
                        "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]})
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    metric = "val_acc" if args.head == "classifier" else "sign_agree"
    print(f"[done] best {metric}={best_select:.4f}  in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
