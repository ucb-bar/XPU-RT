#!/usr/bin/env python3
"""Offline held-out eval for a trained DroNet head on the IDSIA test split.

Segment 014 is the IDSIA-designated test/holdout (never used for train/val), so
this is the honest acceptance metric for Workstream B's nav head:

  - classifier: top-1 accuracy, per-class recall, 3x3 confusion (rows=true).
  - regression: MSE + sign-agreement (fraction of non-straight frames whose
    predicted turn direction is correct) — the metric that predicts flight quality.

Usage::

    python sims/training/eval_dronet.py --head classifier \
        --checkpoint train_out/dronet_grey_cls/<ts>/best.pt
    python sims/training/eval_dronet.py --head regression \
        --checkpoint train_out/dronet_grey_reg/<ts>/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qnn_models.dronet import DronetTorch  # noqa: E402
from sims.training.dataset_idsia import IDSIAConfig, IDSIATrailDataset  # noqa: E402
from sims.training.train_dronet import evaluate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--head", choices=["regression", "classifier"], default="regression")
    p.add_argument("--rgb", action="store_true", help="Model uses 3-channel RGB (default greyscale).")
    p.add_argument("--model_size", choices=["small", "large"], default="small")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT / "datasets/idsia/extracted")
    p.add_argument("--omega_max", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = IDSIATrailDataset(IDSIAConfig(
        root=args.data_root, split=args.split, img_size=args.img_size, omega_max=args.omega_max,
        augment=False, greyscale=not args.rgb, return_class=True,
    ))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[eval] split={args.split} n={len(ds)} counts={ds.class_counts()}")

    model = DronetTorch(
        img_dims=(args.img_size, args.img_size),
        img_channels=3 if args.rgb else 1,
        output_dim=3 if args.head == "classifier" else 1,
        small=(args.model_size == "small"),
        head=args.head,
    ).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)

    res = evaluate(model, loader, args.device, head=args.head)
    print(json.dumps({"checkpoint": str(args.checkpoint), "head": args.head,
                      "split": args.split, **res}, indent=2, default=float))
    if args.head == "classifier":
        print(f"\n[eval] top-1 accuracy = {res['accuracy']:.3f}  (bar: >=0.75)")
    else:
        print(f"\n[eval] sign-agreement = {res['sign_agreement']:.3f}  (bar: >=0.85)   mse={res['mse']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
