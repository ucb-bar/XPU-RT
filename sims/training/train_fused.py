"""BC-train FusedSensorNet on expert sequences from collect_fused_data (task #56, M3).

Loads episode sequences (dict of (T,...) tensors + label (T,2)=[yaw_rate,
forward_speed]), cuts fixed-length windows, and rolls FusedSensorNet timestep-by-
timestep carrying the LSTM hidden state (truncated BPTT), regressing the expert
command. Pure PyTorch — no Isaac.

    <py> sims/training/train_fused.py --data <a.pt> [<b.pt> ...] --epochs 40

The trained head (out_dim=2) drops into eval_forest_nav's flight seam
(target_yaw_rate, target_velocity) exactly like DroNet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vitfly", "models")))
from fused_model import FusedSensorNet  # noqa: E402

KEYS = ("front_grey", "tof_cross", "optical_flow", "down_tof", "baro", "quat",
        "body_rates", "desired_vel", "flags")

# Mono-depth auxiliary task (task #76). TRAINING-ONLY: a small decoder taps the
# CNN vision feature map (B,64,4,6) and regresses a coarse inverse-depth map,
# forcing the greyscale encoder to learn geometry-aware features. The head is
# NOT part of FusedSensorNet — it is discarded after training, so the deployed
# checkpoint + inference latency + the Gemmini/int8 story are all unchanged.
DEPTH_AUX_HW = (16, 24)


class DepthHead(nn.Module):
    """Decoder from the CNN vision feature map (B,64,4,6) -> (B,1,16,24) inverse depth."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),   # 4x6 -> 8x12
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),   # 8x12 -> 16x24
            nn.Conv2d(16, 1, 3, padding=1),                # (B,1,16,24) logits
        )

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))


class WindowDataset(Dataset):
    """Fixed-length windows cut from episodes; each item = per-key (W,...) + label (W,2)."""

    def __init__(self, episodes, window=32, stride=16):
        self.items = []
        for ep in episodes:
            T = ep["label"].shape[0]
            if T < window:
                continue
            has_depth = "front_depth" in ep
            for s in range(0, T - window + 1, stride):
                it = {k: ep[k][s:s + window] for k in KEYS}
                it["label"] = ep["label"][s:s + window]
                # mono-depth aux GT (present only in depth-collected episodes)
                if has_depth:
                    it["front_depth"] = ep["front_depth"][s:s + window]
                    it["has_depth"] = torch.ones(1)
                else:
                    it["front_depth"] = torch.zeros(window, 1, *DEPTH_AUX_HW)
                    it["has_depth"] = torch.zeros(1)
                self.items.append(it)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        out = {k: it[k].float() for k in KEYS}
        out["label"] = it["label"].float()
        out["front_depth"] = it["front_depth"].float()
        out["has_depth"] = it["has_depth"].float()
        return out


def load_episodes(paths):
    eps = []
    for p in paths:
        d = torch.load(p, weights_only=False)
        eps.extend(d["episodes"])
    return eps


def run_window(model, batch, device, speed_weight=0.3, mask=None, depth_head=None):
    """Roll the model over the window carrying hidden; return (loss, aux, yaw_sign_agree, n).

    ``mask`` (dict[str,bool]) zero-skips modalities — e.g. {"desired_vel": False}
    for the Stage-2 vision-goal model, which must infer the goal from the camera
    (the gates are visible) instead of the privileged map goal. No extra network.

    ``depth_head`` (optional): when given, also computes the mono-depth auxiliary
    L1 loss on frames that carry a depth GT (``has_depth``), taps ``model.vision_cnn``
    per-frame (no LSTM/temporal). Returns 0 aux when absent.
    """
    W = batch["label"].shape[1]
    B = batch["label"].shape[0]
    hidden = None
    loss = 0.0
    sign_ok, sign_tot = 0, 0
    mse = nn.functional.mse_loss
    for t in range(W):
        step = {k: batch[k][:, t].to(device, non_blocking=True) for k in KEYS}
        tgt = batch["label"][:, t].to(device)                 # (B,2)
        cmd, hidden = model(step, hidden, mask=mask)          # (B,2)
        loss = loss + mse(cmd[:, 0], tgt[:, 0]) + speed_weight * mse(cmd[:, 1], tgt[:, 1])
        # yaw sign-agreement (ignore near-zero targets)
        m = tgt[:, 0].abs() > 0.05
        sign_ok += int(((cmd[m, 0] > 0) == (tgt[m, 0] > 0)).sum().item())
        sign_tot += int(m.sum().item())

    aux = torch.zeros((), device=device)
    if depth_head is not None:
        gv = batch["front_grey"].to(device)                   # (B,W,1,H,W')
        if gv.shape[-2:] != (60, 90):
            gv = nn.functional.interpolate(gv.flatten(0, 1), size=(60, 90),
                                           mode="bilinear", align_corners=False).view(*gv.shape[:2], 1, 60, 90)
        BW = B * W
        feat = model.vision_cnn(gv.reshape(BW, 1, 60, 90))    # (BW,64,4,6)
        pred = depth_head(feat)                               # (BW,1,16,24) in [0,1]
        dg = batch["front_depth"].to(device).reshape(BW, 1, *DEPTH_AUX_HW)
        hd = batch["has_depth"].to(device).reshape(B, 1).expand(B, W).reshape(BW)
        per = nn.functional.l1_loss(pred, dg, reduction="none").mean(dim=(1, 2, 3))  # (BW,)
        aux = (per * hd).sum() / hd.sum().clamp(min=1.0)
    return loss / W, aux, sign_ok, sign_tot


def evaluate(model, loader, device, mask=None):
    model.eval()
    tot_loss, nb, sok, stot = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            l, _aux, so, st = run_window(model, batch, device, mask=mask)
            tot_loss += float(l); nb += 1; sok += so; stot += st
    return {"val_loss": tot_loss / max(1, nb), "yaw_sign_agree": sok / max(1, stot), "select": sok / max(1, stot)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=Path("/scratch/agustin/projects/DIMA/train_out/fused_bc"))
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--vision_encoder", choices=["vit", "cnn"], default="vit",
                   help="Vision branch: 'vit' (shipped ViT-LSTM) or 'cnn' (Gemmini-friendly conv stem).")
    p.add_argument("--lstm_hidden", type=int, default=128,
                   help="Temporal-fusion LSTM hidden size (capacity knob; default 128).")
    p.add_argument("--lstm_layers", type=int, default=3,
                   help="Temporal-fusion LSTM layer count (capacity knob; default 3).")
    p.add_argument("--mask_desired_vel", action="store_true",
                   help="Stage 2: zero the map goal so the model infers it from the camera (no YOLO).")
    p.add_argument("--depth_aux_weight", type=float, default=0.0,
                   help="If >0, add the TRAINING-ONLY mono-depth auxiliary loss (weight) so the CNN "
                        "encoder learns geometry-aware features. Requires --vision_encoder cnn and "
                        "front_depth in the data. The aux head is discarded; deploy checkpoint unchanged.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    eps = load_episodes(args.data)
    n_val = max(1, int(len(eps) * args.val_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(eps), generator=g).tolist()
    val_eps = [eps[i] for i in perm[:n_val]]
    train_eps = [eps[i] for i in perm[n_val:]]
    train_ds = WindowDataset(train_eps, args.window, args.stride)
    val_ds = WindowDataset(val_eps, args.window, args.window)  # non-overlapping val windows
    print(f"[data] episodes train/val={len(train_eps)}/{len(val_eps)} windows train/val={len(train_ds)}/{len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FusedSensorNet(out_dim=2, vision_encoder=args.vision_encoder,
                           lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    mask = {"desired_vel": False} if args.mask_desired_vel else None
    print(f"[model] FusedSensorNet out_dim=2 params={n_params/1e6:.3f}M device={args.device} "
          f"mask={'desired_vel(Stage2-vision)' if mask else 'none(Stage1-mapped)'}", flush=True)
    depth_head = None
    if args.depth_aux_weight > 0.0:
        if args.vision_encoder != "cnn":
            raise SystemExit("--depth_aux_weight requires --vision_encoder cnn")
        depth_head = DepthHead().to(args.device)
        print(f"[aux] mono-depth auxiliary task ON (weight={args.depth_aux_weight}, "
              f"head params={sum(p.numel() for p in depth_head.parameters())/1e3:.1f}K, "
              f"TRAINING-ONLY — discarded at save)", flush=True)
    train_params = list(model.parameters()) + (list(depth_head.parameters()) if depth_head else [])
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    best_sel, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        if depth_head is not None:
            depth_head.train()
        t0 = time.time(); loss_sum, aux_sum, nb = 0.0, 0.0, 0
        for batch in train_loader:
            l, aux, _so, _st = run_window(model, batch, args.device, mask=mask, depth_head=depth_head)
            total = l + args.depth_aux_weight * aux
            opt.zero_grad(set_to_none=True); total.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            loss_sum += float(l); aux_sum += float(aux); nb += 1
        sched.step()
        val = evaluate(model, val_loader, args.device, mask=mask)
        aux_str = f" aux_depth={aux_sum/max(1,nb):.4f}" if depth_head is not None else ""
        print(f"[ep{ep:02d}] train_loss={loss_sum/max(1,nb):.4f} val_loss={val['val_loss']:.4f} "
              f"yaw_sign_agree={val['yaw_sign_agree']:.3f}{aux_str} ({time.time()-t0:.1f}s)", flush=True)
        torch.save(model.state_dict(), run_dir / "last.pt")
        if val["select"] > best_sel:
            best_sel = val["select"]; torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  ^ new best yaw_sign_agree={best_sel:.4f}", flush=True)
        hist.append({"epoch": ep, "train_loss": loss_sum/max(1,nb), **val})
        json.dump(hist, open(run_dir / "history.json", "w"), indent=2)
    print(f"[done] best yaw_sign_agree={best_sel:.4f} in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
