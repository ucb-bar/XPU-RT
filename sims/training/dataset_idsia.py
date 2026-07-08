"""PyTorch dataset for the IDSIA Forest Trails corpus.

The IDSIA convention is that the **camera identity** is the steering label:
the hiker walked the trail centerline with three head-mounted cameras at
``-30°`` (left, ``lc``), ``0°`` (centre, ``sc``), ``+30°`` (right, ``rc``). A
view from the left camera is "the trail is currently 30° to my right", so the
correct command is **turn right** — and analogously for the right camera.

We map the three classes to a continuous yaw-rate setpoint that matches the
velocity tracker's training range (``±omega_max``, default ``1.0 rad/s``):

================  =============================  =========================
Camera (sub-dir)  Implied agent state            Target yaw rate
================  =============================  =========================
``lc``            Heading too far left            ``-omega_max``  (turn right)
``sc``            Heading along trail             ``0.0``
``rc``            Heading too far right           ``+omega_max``  (turn left)
================  =============================  =========================

Sign convention: ``+omega_z`` = CCW = turn left, matching the velocity-tracker
policy and the reward function in ``mdp_rewards.steering_angle_tracking``.

The dataset is split per-segment (not per-frame) to avoid leakage between
near-duplicate frames in the same video. Segment ``014``'s ``info.txt``
explicitly says "Please do not use this images to train the nets", so we hard-
exclude it from train/val and route it to the test split.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# A handful of JPEGs in the IDSIA archive are truncated (e.g. last frame of a
# clip cut mid-write). Tolerate them rather than crashing a whole training run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


CAMERA_TO_SIGN = {"lc": -1.0, "sc": 0.0, "rc": +1.0}
"""Map from camera-id sub-directory to the sign of the target yaw rate."""

# Segments whose info.txt (or community convention) flags them as test-only or
# training-unsuitable. Routed to the 'test' split, never used for training/val.
HOLDOUT_SEGMENTS = {"014"}

# Segments worth excluding entirely (extremely shaky / blurry handheld phone
# footage). Small enough that the rest of the dataset isn't hurt.
LOW_QUALITY_SEGMENTS: set[str] = set()


@dataclass
class IDSIAConfig:
    root: Path
    split: str = "train"  # one of: "train", "val", "test", "all"
    img_size: int = 112
    omega_max: float = 1.0
    augment: bool = True
    val_segments: tuple[str, ...] = ("011", "012")  # held out from training
    seed: int = 0


def _enumerate_frames(root: Path) -> list[tuple[Path, str, str]]:
    """Walk an extracted IDSIA tree and return ``(jpg_path, segment, camera)``.

    Only paths shaped ``<root>/<NNN>/videos/<lc|sc|rc>/.../*.jpg`` are kept.
    Junk files (``.DS_Store``, ``__MACOSX``, ``._*``) are skipped.
    """
    samples: list[tuple[Path, str, str]] = []
    if not root.is_dir():
        return samples

    for segment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if segment_dir.name.startswith("__") or not segment_dir.name.isdigit():
            continue
        if segment_dir.name in LOW_QUALITY_SEGMENTS:
            continue
        videos_dir = segment_dir / "videos"
        if not videos_dir.is_dir():
            continue
        for camera in ("lc", "sc", "rc"):
            cam_dir = videos_dir / camera
            if not cam_dir.is_dir():
                continue
            for jpg in cam_dir.rglob("*.jpg"):
                base = jpg.name
                if base.startswith("._") or base == ".DS_Store":
                    continue
                samples.append((jpg, segment_dir.name, camera))
    return samples


def _split_segments(all_segments: set[str], cfg: IDSIAConfig) -> dict[str, set[str]]:
    """Partition segment IDs across train / val / test."""
    test = HOLDOUT_SEGMENTS & all_segments
    val = set(cfg.val_segments) & all_segments
    train = all_segments - test - val
    return {"train": train, "val": val, "test": test}


def build_transform(img_size: int, augment: bool) -> Callable:
    """Image transform pipeline. Outputs a ``[3, H, W]`` float tensor in [0, 1]."""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


class IDSIATrailDataset(Dataset):
    """PyTorch Dataset over the IDSIA Forest Trails corpus.

    Args:
        cfg: :class:`IDSIAConfig` describing data root, split, etc.

    Each sample is ``(image, target)`` where ``image`` is a ``[3, H, W]`` float
    tensor in [0, 1] and ``target`` is a 1-element tensor with the desired
    yaw-rate setpoint in rad/s.
    """

    def __init__(self, cfg: IDSIAConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test", "all"}:
            raise ValueError(f"unknown split: {cfg.split}")

        all_samples = _enumerate_frames(cfg.root)
        all_segments = {s for _, s, _ in all_samples}
        if not all_samples:
            raise FileNotFoundError(
                f"No IDSIA frames found under {cfg.root}. "
                "Did you run sims/training/extract_idsia.py?"
            )
        partitions = _split_segments(all_segments, cfg)

        if cfg.split == "all":
            keep = all_segments
        else:
            keep = partitions[cfg.split]
        self.samples = [s for s in all_samples if s[1] in keep]

        self._tx = build_transform(cfg.img_size, augment=(cfg.augment and cfg.split == "train"))
        self._rng = random.Random(cfg.seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, _segment, camera = self.samples[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                img = self._tx(im)
        except (OSError, SyntaxError) as e:
            # Some IDSIA JPEGs are unreadable beyond what LOAD_TRUNCATED_IMAGES
            # can salvage. Fall back to another random sample so a single bad
            # file doesn't kill an epoch.
            print(f"[dataset_idsia] WARN unreadable {path}: {e}; substituting another sample")
            return self.__getitem__((idx + 1) % len(self.samples))

        sign = CAMERA_TO_SIGN[camera]
        target = sign * self.cfg.omega_max

        # Horizontal flip with sign-flipped label. Important: the dataset has
        # a slight L/R imbalance and flipping doubles effective coverage. Only
        # done on the train split; val/test stay deterministic.
        if self.cfg.augment and self.cfg.split == "train" and self._rng.random() < 0.5:
            img = torch.flip(img, dims=[2])
            target = -target

        return img, torch.tensor([target], dtype=torch.float32)

    # Convenience: per-class counts, useful for debugging imbalance.
    def class_counts(self) -> dict[str, int]:
        counts = {"lc": 0, "sc": 0, "rc": 0}
        for _, _, cam in self.samples:
            counts[cam] += 1
        return counts


def make_loaders(
    cfg: IDSIAConfig,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader", IDSIATrailDataset, IDSIATrailDataset]:
    """Build (train_loader, val_loader, train_ds, val_ds)."""
    from torch.utils.data import DataLoader

    train_cfg = IDSIAConfig(**{**cfg.__dict__, "split": "train"})
    val_cfg = IDSIAConfig(**{**cfg.__dict__, "split": "val", "augment": False})
    train_ds = IDSIATrailDataset(train_cfg)
    val_ds = IDSIATrailDataset(val_cfg)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_ds, val_ds


if __name__ == "__main__":
    # Quick smoke test: count samples per split and sanity-check the first item.
    import argparse

    p = argparse.ArgumentParser(description="IDSIA dataset smoke test.")
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[2] / "datasets/idsia/extracted")
    p.add_argument("--img_size", type=int, default=112)
    args = p.parse_args()

    for split in ("train", "val", "test"):
        ds = IDSIATrailDataset(IDSIAConfig(root=args.root, split=split, img_size=args.img_size,
                                          augment=False))
        counts = ds.class_counts()
        print(f"{split:5s}: n={len(ds):6d}  lc={counts['lc']}  sc={counts['sc']}  rc={counts['rc']}")
        if len(ds) > 0 and split == "train":
            img, target = ds[0]
            print(f"  sample[0]: img={tuple(img.shape)} dtype={img.dtype} target={target.item():+.3f}")
