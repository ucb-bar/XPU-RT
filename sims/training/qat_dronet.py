"""Real int8 quantization of DroNet for the DSE (task #25) — measured, not estimated.

The DSE Pareto (dse_pareto.py) currently ESTIMATES int8 cost via a per-MAC energy
model + a latency speedup factor. This produces the REAL other half: the int8
*accuracy* cost, by actually quantizing greyscale DroNet to int8 and evaluating
on the held-out IDSIA split. Uses native torch.ao.quantization (torchao isn't
installed) — FX-graph PTQ (post-training, calibrated) and, if PTQ drops too much,
QAT (fake-quant fine-tune). DroNet is the clean CNN Pareto-frontier model; the
ViT/LSTM models need per-module handling (attention/LSTM FX-quant is fragile) — a
follow-up.

    <py> sims/training/qat_dronet.py --checkpoint <best.pt> --head classifier

Reports fp32 vs int8 top-1 (the real quantization quality cost) → feeds the DSE.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization import quantize_fx
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from qnn_models.dronet import DronetTorch  # noqa: E402
from sims.training.dataset_idsia import IDSIAConfig, IDSIATrailDataset  # noqa: E402
from sims.training.train_dronet import evaluate, head_loss  # noqa: E402

_SCRATCH = ("/tmp/claude-2621/-scratch-agustin-projects-DIMA/"
            "057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--head", choices=["classifier", "regression"], default="classifier")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--data_root", type=Path, default=REPO / "datasets/idsia/extracted")
    p.add_argument("--calib_batches", type=int, default=30)
    p.add_argument("--qat_epochs", type=int, default=3, help="QAT fine-tune epochs if PTQ drops >3pts.")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    out_dim = 3 if args.head == "classifier" else 1
    fp32 = DronetTorch(img_dims=(args.img_size, args.img_size), img_channels=1,
                       output_dim=out_dim, small=True, head=args.head)
    fp32.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    fp32.eval()

    def mk_loader(split, augment=False):
        cfg = IDSIAConfig(root=args.data_root, img_size=args.img_size, augment=augment,
                          val_segments=("011", "012"), greyscale=True, return_class=True, split=split)
        return DataLoader(IDSIATrailDataset(cfg), batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=4)

    val_loader = mk_loader("val")
    train_loader = mk_loader("train")

    # int8 quantized models run on the x86 CPU backend.
    fp32_res = evaluate(fp32, val_loader, "cpu", head=args.head)
    fp32_q = fp32_res["select"]
    metric = "accuracy" if args.head == "classifier" else "sign_agreement"
    print(f"[fp32] {metric}={fp32_q:.4f}", flush=True)

    example = torch.rand(1, 1, args.img_size, args.img_size)

    # ---- PTQ (post-training, calibrated) ----
    qmap = get_default_qconfig_mapping("x86")
    prepared = quantize_fx.prepare_fx(copy.deepcopy(fp32), qmap, example)
    with torch.no_grad():
        for i, (img, _t, _c) in enumerate(train_loader):
            prepared(img)
            if i + 1 >= args.calib_batches:
                break
    ptq = quantize_fx.convert_fx(prepared)
    ptq_q = evaluate(ptq, val_loader, "cpu", head=args.head)["select"]
    print(f"[int8-PTQ] {metric}={ptq_q:.4f}  (Δ={ptq_q-fp32_q:+.4f})", flush=True)

    result = {"model": "dronet-small", "head": args.head, "metric": metric,
              "fp32": round(fp32_q, 4), "int8_ptq": round(ptq_q, 4),
              "ptq_drop": round(fp32_q - ptq_q, 4)}

    # ---- QAT fine-tune if PTQ dropped > 3 pts ----
    if fp32_q - ptq_q > 0.03:
        print(f"[qat] PTQ dropped {fp32_q-ptq_q:.3f} > 0.03 — fine-tuning with fake-quant...", flush=True)
        qat_map = get_default_qat_qconfig_mapping("x86")
        qat_model = quantize_fx.prepare_qat_fx(copy.deepcopy(fp32).train(), qat_map, example)
        opt = torch.optim.AdamW(qat_model.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(args.qat_epochs):
            qat_model.train()
            for img, target, cls in train_loader:
                out, _ = qat_model(img)
                loss = head_loss(out, target, cls, args.head)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            qat_model.eval()
            q = quantize_fx.convert_fx(copy.deepcopy(qat_model))
            qat_q = evaluate(q, val_loader, "cpu", head=args.head)["select"]
            print(f"  [qat ep{ep+1}] {metric}={qat_q:.4f}", flush=True)
        result["int8_qat"] = round(qat_q, 4)
        result["qat_drop"] = round(fp32_q - qat_q, 4)

    out = os.path.join(_SCRATCH, f"qat_dronet_{args.head}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"\n[qat] RESULT: {json.dumps(result)}")
    print(f"[qat] wrote {out}  → feed int8 quality into dse_pareto.py")


if __name__ == "__main__":
    main()
