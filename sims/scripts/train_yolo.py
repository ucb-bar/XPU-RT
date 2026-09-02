"""Fine-tune YOLOv8-nano on the synthetic warehouse dataset (Agent B, Track 1e).

Starts from COCO yolov8n.pt, trains on the {gate, person, obstacle} greyscale-3ch dataset produced
by gen_yolo_dataset.py, reports mAP + per-class recall on val/test. Preserves the YOLOv8n operator
topology so ModelBlaster's existing yolov8_nano int8/fusion path applies to the fine-tuned weights.

    <env_isaaclab py> sims/scripts/train_yolo.py --data out/yolo_warehouse/dataset.yaml \
        --epochs 60 --imgsz 96 --out train_out/warehouse_yolov8n
"""
from __future__ import annotations
import argparse, os

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="dataset.yaml from gen_yolo_dataset.py")
p.add_argument("--weights", default="/scratch2/agustin/ModelBlaster/yolov8n.pt",
               help="pretrained init (fallback to ultralytics 'yolov8n.pt' if absent)")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--imgsz", type=int, default=96, help="long-side train size. With --rect on 90x60 (W x H) frames "
               "ultralytics letterboxes to 64x96 (h x w, aspect 3:2) with ~0 padding — matches the K1 rect build.")
p.add_argument("--rect", action="store_true", default=True, help="rectangular training (no square grey-bar padding)")
p.add_argument("--no-rect", dest="rect", action="store_false")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--out", default="/scratch/agustin/projects/DIMA/train_out/warehouse_yolov8n")
p.add_argument("--device", default="0")
args = p.parse_args()


def main():
    from ultralytics import YOLO
    init = args.weights if os.path.exists(args.weights) else "yolov8n.pt"
    print(f"[yolo] init from {init}")
    model = YOLO(init)
    os.makedirs(args.out, exist_ok=True)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, rect=args.rect,
                device=args.device, project=args.out, name="train", exist_ok=True,
                patience=20, verbose=True)
    # validate on val + test
    for split in ("val", "test"):
        try:
            m = model.val(data=args.data, split=split, imgsz=args.imgsz, rect=args.rect, device=args.device,
                          project=args.out, name=f"val_{split}", exist_ok=True)
            print(f"[{split}] mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f} "
                  f"per-class mAP50={[round(float(x),3) for x in m.box.maps]}")
        except Exception as e:
            print(f"[{split}] val skipped: {e}")
    # export best weights path
    best = os.path.join(args.out, "train", "weights", "best.pt")
    print(f"[yolo] best weights: {best}")


if __name__ == "__main__":
    main()
