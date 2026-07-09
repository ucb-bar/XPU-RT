#!/usr/bin/env python3
"""Collect DroNet training data from IsaacLab forest-trail simulation.

Teleports the drone to randomized positions along the trail (varying lateral
offset and heading), renders FPV camera frames, and saves them with steering
labels compatible with the IDSIA convention used by ``finetune_dronet.py``.

The key idea: at each sample we place the drone at a known lateral offset
``y_off`` from the trail centre and heading error ``psi_err`` from the trail
tangent. The correct steering command is proportional to the error — larger
offset/heading error → stronger corrective turn. We discretize into three
classes (lc, sc, rc) for directory structure, but save continuous labels in a
companion CSV for regression training.

Usage::

    # Launch with Isaac Sim (headless for speed)
    python sims/training/collect_sim_data.py --headless --num_samples 5000

    # Then finetune:
    python sims/training/finetune_dronet.py \
        --checkpoint logs/dronet/2026-04-27_17-10-41/best.pt \
        --data_root datasets/sim_forest/extracted/000 \
        --sim_data --epochs 20 --lr 1e-4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Isaac Lab source paths (same as pilot scripts)
_isaaclab_root = REPO_ROOT / "sims" / "IsaacLab" / "source"
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    _p = str(_isaaclab_root / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- AppLauncher must be created before other isaaclab imports ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_samples", type=int, default=5000,
                    help="Total number of images to collect.")
parser.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "datasets/sim_forest/extracted/000",
                    help="Output directory (IDSIA-style segment structure).")
parser.add_argument("--img_size", type=int, default=480,
                    help="Camera resolution height (width = img_size * 4/3).")
# Trail
parser.add_argument("--trail_length", type=float, default=30.0)
parser.add_argument("--with_humans", action="store_true",
                    help="Include procedural humans in the scene.")
# Sampling distribution
parser.add_argument("--max_lateral_offset", type=float, default=1.2,
                    help="Max lateral offset from trail centre (m).")
parser.add_argument("--max_heading_error_deg", type=float, default=35.0,
                    help="Max heading error from trail tangent (degrees).")
parser.add_argument("--height_range", nargs=2, type=float, default=[0.8, 1.2],
                    help="Drone height range (m).")
parser.add_argument("--omega_max", type=float, default=1.0,
                    help="Max steering label magnitude (rad/s).")
# Sim
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle_steps", type=int, default=5,
                    help="Sim steps after teleport before capturing frame.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- All isaaclab / torch / gym imports go below AppLauncher ---

import torch
import gymnasium as gym
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

import sims.isaaclab_tasks.forest_trail.config.crazyflie as _forest_register  # noqa: F401
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (
    ForestTrailEnvCfg_PLAY,
    ForestTrailEnvCfg_PLAY_WithHumans,
)


def _steering_label(y_off: float, psi_err: float, max_lat: float,
                    max_psi: float, omega_max: float) -> float:
    """Compute continuous steering label from lateral offset and heading error.

    Convention: positive omega = turn left (CCW), negative = turn right.
    In the sim's z-up right-hand frame, +y is LEFT of the trail (+x).
    If y_off > 0 the drone drifted LEFT → needs to turn RIGHT → negative omega.
    If psi_err > 0 the drone is pointed LEFT (CCW) → needs RIGHT → negative omega.

    We blend lateral and heading contributions 50/50 and negate.
    """
    lat_component = (y_off / max_lat) if max_lat > 0 else 0.0
    psi_component = (psi_err / max_psi) if max_psi > 0 else 0.0
    raw = -(0.5 * lat_component + 0.5 * psi_component)
    return float(np.clip(raw * omega_max, -omega_max, omega_max))


def _class_from_label(label: float, omega_max: float) -> str:
    """Map continuous label to IDSIA-style camera class.

    IDSIA convention: lc (left cam) → target is -omega (turn right),
    rc (right cam) → target is +omega (turn left).
    """
    threshold = omega_max / 3.0
    if label < -threshold:
        return "lc"  # needs to turn right
    elif label > threshold:
        return "rc"  # needs to turn left
    return "sc"


def main() -> int:
    rng = random.Random(args_cli.seed)
    np.random.seed(args_cli.seed)

    # -- Setup env --
    cfg_cls = ForestTrailEnvCfg_PLAY_WithHumans if args_cli.with_humans else ForestTrailEnvCfg_PLAY
    env_cfg = cfg_cls()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.fpv_camera.update_period = 0.0
    env_cfg.scene.fpv_camera.height = args_cli.img_size
    env_cfg.scene.fpv_camera.width = int(args_cli.img_size * 4 / 3)

    task_id = (
        "Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0" if args_cli.with_humans
        else "Isaac-Forest-Trail-Vision-Crazyflie-Play-v0"
    )
    print(f"[collect] creating env: {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped

    # -- Output directories --
    out_dir = args_cli.out_dir
    for cam in ("lc", "sc", "rc"):
        (out_dir / "videos" / cam).mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"
    print(f"[collect] output dir: {out_dir}")

    # -- Trail geometry for pose sampling --
    trail_length = args_cli.trail_length
    max_lat = args_cli.max_lateral_offset
    max_psi = math.radians(args_cli.max_heading_error_deg)
    height_lo, height_hi = args_cli.height_range
    omega_max = args_cli.omega_max

    # Initial reset
    obs, _ = env.reset()
    for _ in range(10):
        obs, _, _, _, _ = env.step(torch.zeros(1, unwrapped.action_space.shape[-1],
                                               device=args_cli.device))

    camera = unwrapped.scene["fpv_camera"]
    robot = unwrapped.scene["robot"]
    env_origin = unwrapped.scene.env_origins[0]

    print(f"[collect] env ready. Collecting {args_cli.num_samples} samples...")
    t0 = time.time()
    counts = {"lc": 0, "sc": 0, "rc": 0}

    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "class", "steering_label", "x", "y_off", "heading_err_deg", "height"])

        for i in range(args_cli.num_samples):
            x = rng.uniform(2.0, trail_length - 2.0)
            y_off = rng.uniform(-max_lat, max_lat)
            psi_err = rng.uniform(-max_psi, max_psi)
            height = rng.uniform(height_lo, height_hi)
            yaw = psi_err

            pos = torch.tensor(
                [[x + env_origin[0].item(),
                  y_off + env_origin[1].item(),
                  height + env_origin[2].item()]],
                device=args_cli.device, dtype=torch.float32,
            )
            quat = math_utils.quat_from_euler_xyz(
                torch.tensor([0.0], device=args_cli.device),
                torch.tensor([0.0], device=args_cli.device),
                torch.tensor([yaw], device=args_cli.device),
            )
            pose = torch.cat([pos, quat], dim=-1)
            vel = torch.zeros(1, 6, device=args_cli.device)

            env_ids = torch.tensor([0], device=args_cli.device, dtype=torch.long)
            robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(vel, env_ids=env_ids)

            for _ in range(args_cli.settle_steps):
                unwrapped.sim.step(render=True)
            camera.update(dt=unwrapped.sim.get_physics_dt())

            rgb_data = camera.data.output["rgb"][0].cpu().numpy()
            rgb3 = rgb_data[:, :, :3]
            if rgb3.dtype != np.uint8:
                rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)

            label = _steering_label(y_off, psi_err, max_lat, max_psi, omega_max)
            cls = _class_from_label(label, omega_max)
            counts[cls] += 1

            fname = f"frame_{i:06d}.jpg"
            img_path = out_dir / "videos" / cls / fname
            Image.fromarray(rgb3).save(img_path, quality=95)

            writer.writerow([
                f"videos/{cls}/{fname}", cls, f"{label:.4f}",
                f"{x:.2f}", f"{y_off:.3f}", f"{math.degrees(psi_err):.1f}", f"{height:.2f}",
            ])

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{args_cli.num_samples}] {rate:.1f} samples/s  "
                      f"counts={counts}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n[done] {args_cli.num_samples} samples in {elapsed:.1f}s ({args_cli.num_samples/elapsed:.1f}/s)")
    print(f"[done] counts: {counts}")
    print(f"[done] saved to: {out_dir}")
    print(f"[done] finetune with:")
    print(f"  python sims/training/finetune_dronet.py \\")
    print(f"    --checkpoint logs/dronet/2026-04-27_17-10-41/best.pt \\")
    print(f"    --data_root {out_dir} --sim_data --epochs 20 --lr 1e-4")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
