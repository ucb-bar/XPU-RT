"""PyTorch dataset for sim-collected DroNet training data.

Reads images and continuous steering labels from the CSV produced by
``collect_sim_data.py``. Falls back to IDSIA-style class-based labels if no
CSV is available (for compatibility with mixed datasets).

The directory layout is identical to IDSIA's per-segment structure::

    <root>/videos/lc/*.jpg
    <root>/videos/sc/*.jpg
    <root>/videos/rc/*.jpg
    <root>/labels.csv

``labels.csv`` columns: filename, class, steering_label, x, y_off, heading_err_deg, height
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

CAMERA_TO_SIGN = {"lc": -1.0, "sc": 0.0, "rc": +1.0}


@dataclass
class SimDataConfig:
    root: Path
    img_size: int = 112
    augment: bool = True
    val_fraction: float = 0.15
    split: str = "train"
    seed: int = 0


def build_transform(img_size: int, augment: bool):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


class SimTrailDataset(Dataset):
    """Dataset for sim-collected forest trail images with continuous labels."""

    def __init__(self, cfg: SimDataConfig):
        self.cfg = cfg
        csv_path = cfg.root / "labels.csv"

        if csv_path.is_file():
            self.samples = self._load_from_csv(csv_path)
        else:
            self.samples = self._load_from_dirs(cfg.root)

        # Split
        rng = random.Random(cfg.seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n_val = int(len(indices) * cfg.val_fraction)
        if cfg.split == "val":
            keep = set(indices[:n_val])
        else:
            keep = set(indices[n_val:])
        self.samples = [self.samples[i] for i in sorted(keep)]

        self._tx = build_transform(cfg.img_size, augment=(cfg.augment and cfg.split == "train"))
        self._rng = random.Random(cfg.seed + 1)

    def _load_from_csv(self, csv_path: Path) -> list[tuple[Path, float, str]]:
        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = csv_path.parent / row["filename"]
                if img_path.is_file():
                    label = float(row["steering_label"])
                    cls = row["class"]
                    samples.append((img_path, label, cls))
        return samples

    def _load_from_dirs(self, root: Path) -> list[tuple[Path, float, str]]:
        samples = []
        for cam in ("lc", "sc", "rc"):
            cam_dir = root / "videos" / cam
            if not cam_dir.is_dir():
                continue
            sign = CAMERA_TO_SIGN[cam]
            for jpg in sorted(cam_dir.glob("*.jpg")):
                samples.append((jpg, sign * self.cfg.img_size, cam))  # fallback: ±1.0
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label, _cls = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            img = self._tx(im)

        target = label

        # Random horizontal flip with sign-flipped label
        if self.cfg.augment and self.cfg.split == "train" and self._rng.random() < 0.5:
            img = torch.flip(img, dims=[2])
            target = -target

        return img, torch.tensor([target], dtype=torch.float32)

    def class_counts(self) -> dict[str, int]:
        counts = {"lc": 0, "sc": 0, "rc": 0}
        for _, _, cls in self.samples:
            counts[cls] += 1
        return counts
