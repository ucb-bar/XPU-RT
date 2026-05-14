#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-render N ViNT goal images from the IsaacLab forest-trail env.

ViNT's goal_encoder consumes a (1, 3, H, W) FPV snapshot at deployment
time. Calibrating PTQ activation scales against IDSIA still photographs
mismatches that distribution (different lighting / textures / geometry
than IsaacLab's 3-D renders), so the goal_encoder's int8 forward
produces essentially uncorrelated outputs (see
zephyr-chipyard-sw/agents/examples/vint/inspect_intermediates.py
findings). This script generates renders that match what the
deployment goal_encoder actually sees, saved as PNGs the
``agents.datasets.isaaclab_forest_render`` loader can consume.

Strategy:
* Build a forest-trail env with the same image_size ViNT expects.
* For each desired sample, teleport the drone to a varied trail
  position (arc-length fraction + lateral offset + heading delta)
  and capture the FPV camera output.
* Save as PNG in the configured out dir.

Usage:
    conda activate xpurt   # env with isaaclab + ViNT deps
    python sims/scripts/utils/render_vint_calibration.py \\
        --n-samples 32 \\
        --trail straight \\
        --out-dir datasets/isaaclab_forest_renders

Then re-extract ViNT — the goal_encoder calibration spec auto-picks
up the rendered dir if it exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

freshscheduler_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))
isaaclab_root = os.path.join(freshscheduler_root, "sims/IsaacLab/source")
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_rl"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_contrib"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n-samples", type=int, default=32)
parser.add_argument("--out-dir", type=str,
                    default=str(Path(freshscheduler_root) /
                                "datasets/isaaclab_forest_renders"))
parser.add_argument("--trail", choices=["straight", "curved"], default="straight")
parser.add_argument("--image-w", type=int, default=85,
                    help="Output image width — must match ViNT's image_size[0].")
parser.add_argument("--image-h", type=int, default=64,
                    help="Output image height — must match ViNT's image_size[1].")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-humans", action="store_true")
parser.add_argument("--settle-steps", type=int, default=10,
                    help="Sim ticks between teleport and capture to let the "
                         "camera render the new pose.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import gymnasium as gym
import numpy as np
import torch
from PIL import Image as PILImage

from sims.isaaclab_tasks.forest_trail.config import crazyflie as _register  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (
    ForestTrailEnvCfg_PLAY,
    ForestTrailEnvCfg_PLAY_WithHumans,
    ForestTrailEnvCfg_Curved_PLAY,
    ForestTrailEnvCfg_Curved_PLAY_WithHumans,
)


def _compute_trail_pose(env_cfg, trail_kind: str, arc_fraction: float,
                       lateral_offset: float, yaw_delta_rad: float):
    """Return (x, y, yaw) in env-local frame for a teleport target —
    a point at `arc_fraction` along the trail, offset laterally and
    rotated relative to the trail tangent."""
    if trail_kind == "straight":
        trail_len = float(env_cfg.terminations.off_trail.params.get(
            "trail_length", 30.0))
        x = trail_len * arc_fraction
        y = lateral_offset
        yaw = yaw_delta_rad
        return (x, y, yaw)
    # Curved: walk the polyline.
    wps = list(env_cfg.terminations.off_trail.params.get("waypoints", ()))
    seg_lens = [math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
                for i in range(len(wps) - 1)]
    total = sum(seg_lens)
    target = total * arc_fraction
    arc = 0.0
    for j, slen in enumerate(seg_lens):
        if arc + slen >= target or j == len(seg_lens) - 1:
            t = (target - arc) / max(slen, 1e-9)
            t = max(0.0, min(1.0, t))
            x0, y0 = wps[j]; x1, y1 = wps[j + 1]
            x = x0 + t * (x1 - x0); y = y0 + t * (y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0) + yaw_delta_rad
            # Lateral offset perpendicular to trail tangent.
            x += -math.sin(yaw) * lateral_offset
            y += math.cos(yaw) * lateral_offset
            return (x, y, yaw)
        arc += slen
    p = wps[-1]
    return (p[0], p[1], yaw_delta_rad)


def _teleport_and_capture(unwrapped_env, env_for_step, pose, settle_steps):
    robot = unwrapped_env.scene["robot"]
    env_origin = unwrapped_env.scene.env_origins[0]
    device = robot.data.root_pos_w.device

    goal_x, goal_y, goal_yaw = pose
    half = goal_yaw * 0.5
    pose_world = torch.zeros((1, 7), device=device)
    pose_world[0, 0] = env_origin[0] + goal_x
    pose_world[0, 1] = env_origin[1] + goal_y
    pose_world[0, 2] = 1.0
    pose_world[0, 3] = math.cos(half)
    pose_world[0, 6] = math.sin(half)
    vel_zero = torch.zeros((1, 6), device=device)
    robot.write_root_pose_to_sim(pose_world)
    robot.write_root_velocity_to_sim(vel_zero)
    n_act = env_for_step.action_space.shape[1]
    zero_act = torch.zeros((1, n_act), device=device)
    for _ in range(settle_steps):
        env_for_step.step(zero_act)
        robot.write_root_pose_to_sim(pose_world)
        robot.write_root_velocity_to_sim(vel_zero)
    rgb = unwrapped_env.scene["fpv_camera"].data.output["rgb"][0].cpu().numpy()
    rgb3 = rgb[:, :, :3]
    if rgb3.dtype != np.uint8:
        rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)
    return PILImage.fromarray(rgb3)


def main():
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_humans = not args_cli.no_humans
    curved = args_cli.trail == "curved"
    if curved:
        cfg_cls = (ForestTrailEnvCfg_Curved_PLAY_WithHumans if use_humans
                   else ForestTrailEnvCfg_Curved_PLAY)
        task_id = ("Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-v0"
                   if use_humans else
                   "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-v0")
    else:
        cfg_cls = (ForestTrailEnvCfg_PLAY_WithHumans if use_humans
                   else ForestTrailEnvCfg_PLAY)
        task_id = ("Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0"
                   if use_humans else
                   "Isaac-Forest-Trail-Vision-Crazyflie-Play-v0")
    env_cfg = cfg_cls()
    env_cfg.sim.device = args_cli.device
    # Match the camera output resolution to ViNT's input.
    env_cfg.scene.fpv_camera.height = args_cli.image_h
    env_cfg.scene.fpv_camera.width = args_cli.image_w
    env_cfg.commands.steering_command.resampling_time_range = (1e9, 1e9)

    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped

    rng = np.random.default_rng(args_cli.seed)
    print(f"[render] writing {args_cli.n_samples} samples to {out_dir}",
          flush=True)
    for i in range(args_cli.n_samples):
        # Sweep arc_fraction across [0.3, 0.95] so the goal pipeline
        # sees varied trail content (early/mid/late). Small lateral
        # jitter + heading delta keeps the renders from being
        # identical at the same arc point.
        arc = 0.3 + 0.65 * (i / max(1, args_cli.n_samples - 1))
        lateral = float(rng.uniform(-0.6, 0.6))
        yaw = float(rng.uniform(-0.25, 0.25))
        pose = _compute_trail_pose(env_cfg, args_cli.trail, arc, lateral, yaw)
        img = _teleport_and_capture(unwrapped_env, env, pose,
                                    args_cli.settle_steps)
        out_path = out_dir / f"vint_goal_{i:04d}_arc{arc:.2f}.png"
        img.save(out_path)
        if (i + 1) % 8 == 0 or i == 0:
            print(f"  [{i+1}/{args_cli.n_samples}] {out_path.name}",
                  flush=True)
    print(f"[render] done — wrote {args_cli.n_samples} renders",
          flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
