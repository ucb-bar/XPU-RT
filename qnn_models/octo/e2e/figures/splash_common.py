#!/usr/bin/env python3
"""Shared helper: keep the ROBOT at full saturation under the scrim.

The scrim is a flat white rectangle over the whole photo, so it dims the arm along with
everything else -- but the arm is the actor, and a splash frame reads better with it
present rather than washed out.

Segmenting the robot by colour ALONE does not work: it is achromatic in both scenes, so a
low-saturation test also catches shadow, the wall, the sink and the table. Measured on the
tick-0 frames, a naive dark test claims 14.5% of the eggplant frame and only 0.6% of the
coke frame -- wrong in both directions.

So gate colour by GEOMETRY. The robot's pose is known: project its link centres of mass
with the run's own camera matrices and keep low-saturation pixels only where they are near
a projected link. That cannot drift onto the background, and it needs no per-scene tuning.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
import PIL.Image as PILImage


def project(P, K, E):
    P = np.atleast_2d(np.asarray(P, float))
    cam = np.c_[P, np.ones(len(P))] @ E.T
    uvw = cam[:, :3] @ K.T
    w = uvw[:, 2:3].copy(); w[np.abs(w) < 1e-9] = np.nan
    uv = uvw[:, :2] / w
    uv[cam[:, 2] <= 0] = np.nan
    return uv


def robot_mask(bg_path, link_com_t0, K, E, radius=55, sat_max=0.30):
    """Low-saturation pixels within `radius` px of a projected link COM."""
    hsv = np.asarray(PILImage.open(bg_path).convert("HSV")).astype(float)
    H, W = hsv.shape[:2]
    S = hsv[..., 1] / 255.0
    uv = project(link_com_t0, K, E)
    ins = uv[(~np.isnan(uv[:, 0])) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
             & (uv[:, 1] >= 0) & (uv[:, 1] < H)]
    if not len(ins):
        return np.zeros((H, W), bool)
    seed = np.zeros((H, W), bool)
    for x, y in ins:
        seed[int(y), int(x)] = True
    near = ndimage.distance_transform_edt(~seed) < radius
    m = near & (S < sat_max)
    return ndimage.binary_closing(m, np.ones((5, 5)))


def unscrim_robot(ax, bg_rgb, mask, W, H, z=2):
    """Re-stamp the robot's own pixels over the scrim, at full saturation."""
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., :3] = np.asarray(bg_rgb)[..., :3] / 255.0
    rgba[..., 3] = mask.astype(float)
    ax.imshow(rgba, extent=[0, W, H, 0], zorder=z, interpolation="nearest")
    return int(mask.sum())
