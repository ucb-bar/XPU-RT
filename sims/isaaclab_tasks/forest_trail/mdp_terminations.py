# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the forest-trail testbed.

``off_trail`` handles straight trails (fast, axis-aligned corridor check).
``off_trail_curved`` handles polyline trails (nearest-segment distance,
pure PyTorch so it vectorises over any number of envs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def off_trail(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_margin: float = 3.0,
    end_overrun: float = 2.0,
    trail_length: float = 30.0,
) -> torch.Tensor:
    """Terminate when drone leaves the trail corridor.

    The trail runs from ``x = 0`` to ``x = trail_length`` along +x in the
    env-local frame, centred at ``y = 0``.

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        lateral_margin: Max ``|y_local|`` in metres before termination.
        end_overrun: How far past the trail end (``trail_length + this``) the
            drone may go before termination.
        trail_length: Length of the trail along +x. Pass via the term cfg from
            the layout used to build the scene.

    Returns:
        Bool tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins  # (N, 3)
    x = pos_local[:, 0]
    y_abs = torch.abs(pos_local[:, 1])

    too_lateral = y_abs > lateral_margin
    past_end = x > (trail_length + end_overrun)
    behind_start = x < -end_overrun
    return too_lateral | past_end | behind_start


def off_trail_curved(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = (),
    lateral_margin: float = 3.0,
    end_margin: float = 2.0,
) -> torch.Tensor:
    """Terminate when drone strays from a curved (polyline) trail.

    Computes the perpendicular distance from each drone to the nearest
    segment of the waypoint polyline.  Also terminates if the drone is
    behind the trail start or past the trail end (measured along arc length).

    All maths is pure PyTorch so the function scales to any ``num_envs``
    without CPU/GPU data transfers.

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        waypoints: Tuple of ``(x, y)`` floats — the 2-D polyline vertices
            from ``DEFAULT_CURVED_WAYPOINTS_2D``.  Converted to a device
            tensor on the first call.
        lateral_margin: Perpendicular distance (m) from the nearest segment
            beyond which the episode terminates.
        end_margin: Extra slack (m) allowed past each end of the trail before
            the arc-length termination fires.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    if len(waypoints) < 2:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins  # (N, 3)
    xy = pos_local[:, :2]  # (N, 2)

    # ── Build segment tensors (M-1 segments from M waypoints) ────────────────
    pts = torch.tensor(waypoints, dtype=torch.float32, device=env.device)  # (M, 2)
    p0 = pts[:-1]             # (M-1, 2)  segment starts
    p1 = pts[1:]              # (M-1, 2)  segment ends
    seg = p1 - p0             # (M-1, 2)  segment vectors
    seg_len = seg.norm(dim=1)                       # (M-1,)
    seg_len_sq = seg_len ** 2                       # (M-1,)  avoid repeated squaring

    # ── Project each drone onto each segment ─────────────────────────────────
    # diff[n, m] = drone_n position relative to segment_m start
    diff = xy[:, None, :] - p0[None, :, :]         # (N, M-1, 2)

    # Clamped projection parameter t ∈ [0, 1]
    t = ((diff * seg[None]).sum(dim=2) / seg_len_sq).clamp(0.0, 1.0)  # (N, M-1)

    # Closest point on each segment
    closest = p0[None] + t[:, :, None] * seg[None]  # (N, M-1, 2)

    # Perpendicular distance to each segment
    dist = (xy[:, None, :] - closest).norm(dim=2)   # (N, M-1)

    # ── Nearest segment + arc-length position ────────────────────────────────
    min_dist, best_seg = dist.min(dim=1)             # (N,), (N,)

    # Arc length at the start of each segment
    arc_starts = torch.zeros(len(seg), device=env.device)
    if len(seg) > 1:
        arc_starts[1:] = seg_len[:-1].cumsum(0)
    total_arc = seg_len.sum()

    t_best = t[torch.arange(len(xy), device=env.device), best_seg]   # (N,)
    arc_pos = arc_starts[best_seg] + t_best * seg_len[best_seg]       # (N,)

    return (
        (min_dist > lateral_margin)
        | (arc_pos > total_arc + end_margin)
        | (arc_pos < -end_margin)
    )
