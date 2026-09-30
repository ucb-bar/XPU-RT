#!/usr/bin/env python3
"""Marey chronophotography of the moving warehouse cylinders from an overhead ov_seq dump.

Honest, image-based (no projection): the N overhead frames share ONE render pass, so the per-pixel
temporal MEDIAN is the static scene (transient movers wash out). Each frame minus the median, gated by
a pale-cylinder colour filter (patrol cylinders are whitish-blue; crates are saturated teal, boxes tan),
isolates just the moving cylinders. Their real image pixels are overlaid on the clean plate (ov_bg) with
a time-increasing alpha -> overlapped CAPTURES OF THE IMAGE of each cylinder stepping through the space.

Usage: chrono_compose.py <figure_data.npz> <out.png>
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def marey(npz_path, out_path, alpha0=0.5, pct=98.0, min_blob=120, sat_max=0.22, bright_min=150):
    d = np.load(npz_path, allow_pickle=True)
    seq = d["ov_seq"].astype(np.float32)
    bg = d["ov_bg"].astype(np.float32)
    n = len(seq)
    med = np.median(seq, axis=0)
    seqc = np.stack([seq[i] - np.median(seq[i] - med) for i in range(n)])  # per-frame offset correction
    diff = np.abs(seqc - med).sum(axis=3)
    R, G, B = seq[..., 0], seq[..., 1], seq[..., 2]
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    sat = (mx - mn) / (mx + 1e-3)
    pale = (mx > bright_min) & (sat < sat_max)
    thr = np.percentile(diff, pct)
    comp = bg.copy()
    for i in range(n):
        m = (diff[i] > thr) & pale[i]
        m = ndi.binary_opening(m, iterations=1)
        m = ndi.binary_closing(m, iterations=2)
        lbl, nl = ndi.label(m)
        if nl:
            sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, nl + 1))
            m = np.isin(lbl, 1 + np.where(sizes >= min_blob)[0])
        a = alpha0 + (1.0 - alpha0) * i / max(1, n - 1)
        comp[m] = (1 - a) * comp[m] + a * seq[i][m]
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    plt.figure(figsize=(16, comp.shape[0] / comp.shape[1] * 16)); plt.imshow(comp); plt.axis("off")
    plt.tight_layout(pad=0); plt.savefig(out_path, dpi=110, bbox_inches="tight", pad_inches=0)
    print(f"wrote {out_path}  ({n} exposures)")
    return comp


if __name__ == "__main__":
    marey(sys.argv[1], sys.argv[2])
