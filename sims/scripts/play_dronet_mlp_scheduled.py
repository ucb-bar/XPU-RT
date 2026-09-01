#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script that integrates DroNet vision with trained MLP steering policy using scheduled timing.

This combines:
1. DroNet: Processes FPV camera → steering angle + collision probability
2. Command Generator: Converts DroNet output → target yaw rate + velocity
3. Trained MLP Policy: Takes state observations → thrust commands
4. **Scheduled Timing**: Executes models according to SW scheduler dispatch timing

The schedule JSON contains dispatches with keys like:
- "dronet_dispatch_0", "dronet_dispatch_1", ... → DroNet inference
- "mlp_dispatch_0", "mlp_dispatch_1", ... → MLP policy inference

Simulation steps through schedule segments, executing models only during their dispatch periods.

Usage:
    # Use default schedule and latest checkpoint
    conda run -n xpurt python sims/scripts/play_dronet_mlp_scheduled.py

    # Specify schedule and checkpoint
    conda run -n xpurt python sims/scripts/play_dronet_mlp_scheduled.py \
        --schedule_json schedules/my_schedule.json \
        --checkpoint logs/rsl_rl/crazyflie_steering_tracking/XXX/model_XXX.pt
"""

import argparse
import sys
import os
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Add paths
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
_isaaclab_root = os.environ.get("ISAACLAB_ROOT", "/scratch2/dima/IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, os.path.join(_isaaclab_root, "source", _pkg))

from isaaclab.app import AppLauncher

_DEFAULT_SCHEDULE = Path(__file__).resolve().parent.parent.parent / "schedules" / "scheduled_networks_periodic_profile_profiled.json"

# Add argparse arguments
parser = argparse.ArgumentParser(description="DroNet + MLP policy with scheduled timing.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained MLP policy checkpoint (default: auto-find latest).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_periods", type=int, default=5, help="Number of schedule periods to run.")
parser.add_argument("--dronet_weights", type=str, default=None,
                    help="Path to DroNet weights (default: random initialization).")
parser.add_argument("--model_size", type=str, default="small", choices=["small", "large"],
                    help="DroNet model size.")
parser.add_argument("--no_visualization", action="store_true", help="Disable matplotlib visualization.")
parser.add_argument("--max_yaw_rate", type=float, default=1.0, help="Max yaw rate (rad/s).")
parser.add_argument("--max_velocity", type=float, default=1.0, help="Max forward velocity (m/s).")
parser.add_argument(
    "--schedule_json",
    type=Path,
    default=_DEFAULT_SCHEDULE,
    help="Path to profiled schedule JSON (dispatches + metadata.makespan).",
)
parser.add_argument(
    "--merge_eps",
    type=float,
    default=1e-9,
    help="Merge schedule breakpoints closer than this (seconds, after unit conversion).",
)
parser.add_argument(
    "--schedule_time_unit",
    choices=("ms", "s"),
    default="ms",
    help="Unit for start_time, duration, and makespan in the schedule JSON (default: ms).",
)
parser.add_argument("--save_gif", type=str, default=None,
                    help="Path to save visualization as GIF (e.g., output.gif). Only works with visualization enabled.")
parser.add_argument("--gif_fps", type=int, default=30,
                    help="FPS for saved GIF (default: 30). Lower FPS = smaller file.")
parser.add_argument("--gif_capture_skip", type=int, default=10,
                    help="Capture every Nth frame for GIF (default: 10). With 1000 Hz, 10 = 100 Hz capture.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras (required for FPV camera)
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, GaussianDistribution

# Register custom environments
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie

# Import environment config
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (
    TrackSteeringEnvCfg_PLAY,
)

# Visualization
import matplotlib.pyplot as plt

##
# DroNet Model
##

class DronetTorch(nn.Module):
    """DroNet PyTorch implementation for trail navigation."""
    def __init__(self, img_dims, img_channels, output_dim, small=True):
        super(DronetTorch, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_dims = img_dims
        self.channels = img_channels
        self.output_dim = output_dim
        self.small = small
        self.conv_modules = nn.ModuleList()

        # Initialize convolution modules
        if small:
            self.conv_modules.append(nn.Conv2d(self.channels, 32, (3,3), stride=(2,2), padding=(1,1)))
        else:
            self.conv_modules.append(nn.Conv2d(self.channels, 32, (5,5), stride=(2,2), padding=(2,2)))

        filter_amt = np.array([32, 64, 128])
        for f in filter_amt:
            x1 = int(f/2) if f != 32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3,3), stride=(2,2), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3,3), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1,1), stride=(2,2)))

        self.maxpool1 = nn.MaxPool2d((3,3), (2,2))

        bn_amt = np.array([32, 32, 32, 64, 64, 128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())

        self.dropout1 = nn.Dropout()

        if small:
            self.linear1 = nn.Linear(2048, output_dim)
            self.linear2 = nn.Linear(2048, output_dim)
        else:
            self.linear1 = nn.Linear(6272, output_dim)
            self.linear2 = nn.Linear(6272, output_dim)

        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """Initialize weights with He initialization."""
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        """Forward pass through DroNet."""
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)

        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx+1](x2)
            x2 = self.relu_modules[relu_idx+1](x2)
            x2 = self.conv_modules[conv_idx+1](x2)
            x1 = self.conv_modules[conv_idx+2](x1)
            x3 = torch.add(x1, x2)
            x1 = x3
            bn_idx += 2
            relu_idx += 2
            conv_idx += 3

        if self.small:
            x4 = torch.flatten(x3).reshape(-1, 2048)
        else:
            x4 = torch.flatten(x3).reshape(-1, 6272)

        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        steer = self.linear1(x5)
        collision = self.linear2(x5)
        collision = self.sigmoid1(collision)

        return steer, collision


def preprocess_camera_frame(rgb_tensor: torch.Tensor, target_size: tuple, device: torch.device) -> torch.Tensor:
    """Preprocess camera frame for DroNet inference."""
    # Take first environment if batch
    if rgb_tensor.shape[0] > 0:
        img = rgb_tensor[0, :, :, :3]  # (H, W, 3)
    else:
        return torch.zeros(1, 3, target_size[0], target_size[1], device=device)

    # Convert to float and normalize to [0, 1]
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.clamp(0.0, 1.0)

    # Permute to (C, H, W) and add batch dimension
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Resize to target size
    img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)

    return img.to(device)


def _tensor_rgb_to_uint8_hwc(rgb: torch.Tensor, env_idx: int) -> np.ndarray:
    """Convert camera RGB tensor to uint8 numpy array for display."""
    img = rgb[env_idx, ..., :3].detach().cpu().numpy()
    if img.dtype == np.uint8:
        return img
    if img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    img_max = float(np.nanmax(img))
    if img_max <= 1.0:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def find_latest_checkpoint() -> str:
    """Find the latest trained checkpoint automatically from multiple locations."""
    import glob

    # Check both possible log directories
    log_dirs = [
        os.path.join(os.environ.get("ISAACLAB_ROOT", "/scratch2/dima/IsaacLab"),
                     "logs", "rsl_rl", "crazyflie_steering_tracking"),
        os.path.join(freshscheduler_root, "logs/rsl_rl/crazyflie_steering_tracking"),
    ]

    all_checkpoints = []

    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            continue

        # Find all run directories
        run_dirs = glob.glob(os.path.join(log_dir, "20*"))

        for run_dir in run_dirs:
            # Find all checkpoints in this run
            checkpoints = glob.glob(os.path.join(run_dir, "model_*.pt"))
            all_checkpoints.extend(checkpoints)

    if not all_checkpoints:
        raise FileNotFoundError(
            "No training checkpoints found!\n"
            f"Searched in:\n  - {log_dirs[0]}\n  - {log_dirs[1]}\n"
            "Have you run training yet?"
        )

    # Return the most recently modified checkpoint
    latest = max(all_checkpoints, key=os.path.getmtime)
    return latest


class PolicyActor(nn.Module):
    """Actor network for policy inference."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list, activation: str = 'elu'):
        super().__init__()
        self.mlp = MLP(obs_dim, action_dim, hidden_dims, activation=activation)
        self.distribution = GaussianDistribution(action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference (deterministic)."""
        return self.mlp(obs)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic actions for inference."""
        return self.forward(obs)


def load_mlp_policy(checkpoint_path: str, obs_shape: tuple, action_dim: int, device: str):
    """Load trained MLP policy from checkpoint."""
    from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (
        SteeringTrackingPPORunnerCfg,
    )

    print(f"[INFO]: Loading MLP policy from: {checkpoint_path}")

    agent_cfg = SteeringTrackingPPORunnerCfg()

    # Create actor model
    model = PolicyActor(
        obs_dim=obs_shape[0],
        action_dim=action_dim,
        hidden_dims=agent_cfg.actor.hidden_dims,
        activation=agent_cfg.actor.activation,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'ac' in checkpoint:
        model.load_state_dict(checkpoint['ac'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[INFO]: MLP policy loaded successfully")
    print(f"  Iteration: {checkpoint.get('iter', 'N/A')}")

    return model


def _model_from_dispatch_key(key: str) -> str:
    """Extract model name from dispatch key (e.g., 'dronet_dispatch_0' → 'dronet', 'mlp0_dispatch_0' → 'mlp').

    Normalizes model names by stripping trailing digits so that mlp0, mlp1, mlp2, etc.
    all map to 'mlp'.
    """
    if "_dispatch" not in key:
        raise ValueError(f"Dispatch key has no '_dispatch' segment: {key!r}")

    model_name = key.split("_dispatch", 1)[0]

    # Strip trailing digits to normalize names like 'mlp0', 'mlp1' → 'mlp'
    # This allows multiple instances of the same model to be treated as one logical model
    import re
    normalized_name = re.sub(r'\d+$', '', model_name)

    return normalized_name if normalized_name else model_name


def _merge_sorted_times(times: list[float], eps: float) -> list[float]:
    """Merge times closer than eps."""
    out: list[float] = []
    for t in sorted(times):
        if not out or abs(t - out[-1]) > eps:
            out.append(t)
    return out


def load_schedule(schedule_path: Path, merge_eps: float, time_unit: str) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    """Load schedule JSON and return period and dispatch intervals by model.

    Returns:
        period: Schedule period in seconds
        model_dispatches: Dict mapping model name to list of (start_time, end_time) tuples
    """
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

    to_seconds = 1e-3 if time_unit == "ms" else 1.0

    data = json.loads(schedule_path.read_text())
    dispatches = data.get("dispatches")
    if not isinstance(dispatches, dict):
        raise ValueError("Schedule JSON must contain a 'dispatches' object.")

    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    max_end = 0.0
    for key, d in dispatches.items():
        m = _model_from_dispatch_key(str(key))
        start = float(d["start_time"]) * to_seconds
        end = start + float(d["duration"]) * to_seconds
        by_model[m].append((start, end))
        max_end = max(max_end, end)

    meta = data.get("metadata") or {}
    makespan = float(meta["makespan"]) * to_seconds if "makespan" in meta else max_end
    period = max(makespan, max_end)

    return period, dict(by_model)


def is_model_active(model_name: str, time_in_period: float, model_dispatches: dict[str, list[tuple[float, float]]]) -> bool:
    """Check if a model should be executing at the given time within the period."""
    if model_name not in model_dispatches:
        return False

    for start, end in model_dispatches[model_name]:
        if start <= time_in_period < end:
            return True
    return False


def main():
    """Run DroNet + MLP policy with scheduled timing."""

    # Find checkpoint if not provided
    if args_cli.checkpoint is None:
        print("[INFO]: No checkpoint provided, searching for latest...")
        checkpoint_path = find_latest_checkpoint()
        print(f"[INFO]: Found latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args_cli.checkpoint
        print(f"[INFO]: Using specified checkpoint: {checkpoint_path}")

    # Setup environment with camera
    env_cfg = TrackSteeringEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Use much finer timestep for scheduler sensitivity
    # Schedule period is 32.682ms, so we need fine-grained control
    env_cfg.sim.dt = 0.001  # 1ms physics timestep (1000 Hz)
    env_cfg.decimation = 1  # No decimation → 1ms control timestep (1000 Hz)
    # This gives us ~33 steps per schedule period

    # Update camera to match control frequency (1000 Hz = 0.001s update period)
    env_cfg.scene.fpv_camera.update_period = 0.001  # 1ms = 1000 Hz
    print(f"[INFO]: Camera update period set to {env_cfg.scene.fpv_camera.update_period*1000:.1f}ms (1000 Hz)")

    print("[INFO]: Creating environment...")
    env = gym.make("Isaac-Track-Steering-Vision-Crazyflie-Play-v0", cfg=env_cfg, render_mode="rgb_array")

    # Initialize DroNet
    device = torch.device(args_cli.device)
    model_small = args_cli.model_size == "small"
    img_size = (112, 112) if model_small else (224, 224)

    print(f"[INFO]: Initializing DroNet (size: {args_cli.model_size}, input: {img_size})")
    dronet = DronetTorch(
        img_dims=img_size,
        img_channels=3,
        output_dim=1,
        small=model_small
    ).to(device)

    if args_cli.dronet_weights and os.path.exists(args_cli.dronet_weights):
        print(f"[INFO]: Loading DroNet weights from {args_cli.dronet_weights}")
        dronet.load_state_dict(torch.load(args_cli.dronet_weights, map_location=device))
    else:
        print("[INFO]: Using randomly initialized DroNet weights")

    dronet.eval()

    # Load trained MLP policy
    obs_shape = env.observation_space['policy'].shape[1:]
    action_dim = env.action_space.shape[1]
    mlp_policy = load_mlp_policy(checkpoint_path, obs_shape, action_dim, args_cli.device)

    # Load schedule
    period, model_dispatches = load_schedule(args_cli.schedule_json.resolve(), args_cli.merge_eps, args_cli.schedule_time_unit)

    print(f"\n[INFO]: Schedule loaded from {args_cli.schedule_json}")
    print(f"  Period: {period:.6f}s")
    print(f"  Models:")
    for model, intervals in model_dispatches.items():
        total_time = sum(end - start for start, end in intervals)
        print(f"    {model}: {len(intervals)} dispatches, {total_time:.6f}s total")

    # Visualization setup
    show_viz = not args_cli.headless and not args_cli.no_visualization
    fpv_fig = None
    im_rgb = im_processed = None
    text_steer = text_coll = text_cmd = text_schedule = None
    ax_schedule = None
    time_line = None
    processed_title = None

    # GIF recording setup
    save_gif = args_cli.save_gif is not None and show_viz
    gif_frames = [] if save_gif else None
    if save_gif:
        print(f"[INFO]: GIF recording enabled. Will save to: {args_cli.save_gif}")
        print(f"[INFO]: GIF FPS: {args_cli.gif_fps}")

    # Statistics
    steer_history = []
    coll_history = []
    reward_history = []
    dronet_exec_count = 0
    mlp_exec_count = 0

    # Calculate timing parameters
    control_dt = env_cfg.sim.dt * env_cfg.decimation  # Control timestep (0.02s = 50 Hz)

    print("\n[INFO]: Starting scheduled DroNet + MLP policy...")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Schedule periods to run: {args_cli.num_periods}")
    print(f"  Control frequency: {1.0/control_dt:.0f} Hz")
    print(f"  Max yaw rate: {args_cli.max_yaw_rate} rad/s")
    print(f"  Max velocity: {args_cli.max_velocity} m/s")
    print()

    # Reset environment
    obs_dict, _ = env.reset()

    # Cache last model outputs (used when models are not scheduled)
    # DroNet outputs
    steer_val = 0.0
    coll_val = 0.5
    target_yaw_rate = 0.0
    target_velocity = args_cli.max_velocity * 0.5

    # MLP output (cached action)
    cached_actions = torch.zeros(args_cli.num_envs, action_dim, device=device)

    # Cached processed frame for visualization (updated only when DroNet executes)
    cached_processed_vis = None

    # Run for specified number of periods
    sim_time = 0.0
    step = 0

    for period_idx in range(args_cli.num_periods):
        period_start_time = sim_time
        print(f"\n{'='*70}")
        print(f"PERIOD {period_idx + 1}/{args_cli.num_periods}")
        print(f"{'='*70}")

        # Run one period
        while (sim_time - period_start_time) < period:
            time_in_period = sim_time - period_start_time

            # Check if models should execute
            dronet_active = is_model_active("dronet", time_in_period, model_dispatches)
            mlp_active = is_model_active("mlp", time_in_period, model_dispatches)

            # Get camera image
            camera_data = env.unwrapped.scene["fpv_camera"].data.output["rgb"]

            # DroNet inference (only if scheduled)
            if dronet_active:
                processed_frame = preprocess_camera_frame(camera_data, img_size, device)
                with torch.no_grad():
                    steer, collision = dronet(processed_frame)

                steer_val = steer.item()
                coll_val = collision.item()
                dronet_exec_count += 1

                # Convert DroNet output to steering commands and cache them
                target_yaw_rate = steer_val * args_cli.max_yaw_rate
                target_velocity = (1.0 - coll_val) * args_cli.max_velocity

                # Cache processed frame for visualization (update only when DroNet executes)
                if show_viz:
                    cached_processed_vis = processed_frame[0].permute(1, 2, 0).cpu().numpy()
                    cached_processed_vis = (cached_processed_vis * 255).astype(np.uint8)
            # else: use cached steer_val, coll_val, target_yaw_rate, target_velocity

            # Set command in environment (always, uses cached values if DroNet not active)
            command_tensor = torch.tensor([[target_yaw_rate, target_velocity]],
                                         device=device, dtype=torch.float32)
            command_tensor = command_tensor.repeat(args_cli.num_envs, 1)
            env.unwrapped.command_manager._terms["steering_command"].target_yaw_rate[:] = command_tensor[:, 0]
            env.unwrapped.command_manager._terms["steering_command"].target_velocity[:] = command_tensor[:, 1]

            # Get observations
            obs = obs_dict["policy"]

            # MLP policy inference (only if scheduled)
            if mlp_active:
                with torch.no_grad():
                    actions = mlp_policy.act_inference(obs)
                # Cache the new action for future use
                cached_actions = actions.clone()
                mlp_exec_count += 1
            else:
                # Use cached action from last MLP execution (hold last valid output)
                actions = cached_actions

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Store statistics
            steer_history.append(steer_val)
            coll_history.append(coll_val)
            reward_history.append(rewards.mean().item())

            # Print progress
            if (step + 1) % 50 == 0:
                dronet_status = 'EXEC' if dronet_active else 'cached'
                mlp_status = 'EXEC' if mlp_active else 'cached'
                schedule_status = f"[DroNet: {dronet_status}, MLP: {mlp_status}]"
                print(f"  Step {step + 1} | t={time_in_period:.3f}s/{period:.3f}s {schedule_status} | "
                      f"Steer: {steer_val:+.3f} | Coll: {coll_val:.3f} | "
                      f"Cmd: yaw={target_yaw_rate:+.2f}, vel={target_velocity:.2f} | "
                      f"Reward: {np.mean(reward_history[-50:]):.2f}")

            # Visualization (update every frame for camera, use cached for DroNet processed)
            if show_viz and camera_data.shape[0] > 0:
                if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                    fpv_fig = None

                # Camera view updates every frame
                rgb_np = _tensor_rgb_to_uint8_hwc(camera_data, 0)

                # Use cached processed frame (only updated when DroNet executes)
                # Initialize with zeros if not yet available
                if cached_processed_vis is None:
                    cached_processed_vis = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
                processed_vis = cached_processed_vis

                if fpv_fig is None:
                    plt.ion()
                    # Create figure with 3 subplots: camera views on top, schedule on bottom
                    fpv_fig = plt.figure(num="Scheduled DroNet + MLP", figsize=(14, 8))
                    gs = fpv_fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.2)

                    # Camera views
                    ax_rgb = fpv_fig.add_subplot(gs[0, 0])
                    ax_processed = fpv_fig.add_subplot(gs[0, 1])

                    # Schedule visualization
                    ax_schedule = fpv_fig.add_subplot(gs[1, :])

                    im_rgb = ax_rgb.imshow(rgb_np)
                    ax_rgb.set_title("FPV Camera")
                    ax_rgb.axis("off")

                    im_processed = ax_processed.imshow(processed_vis)
                    processed_title = ax_processed.set_title(f"DroNet Input ({img_size[0]}x{img_size[1]}) - Initializing")
                    ax_processed.axis("off")

                    # Plot schedule dispatches
                    ax_schedule.set_ylim(-0.5, 1.5)
                    ax_schedule.set_xlim(0, period * 1000)  # Convert to ms for display
                    ax_schedule.set_xlabel("Time (ms)")
                    ax_schedule.set_yticks([0, 1])
                    ax_schedule.set_yticklabels(["MLP", "DroNet"])
                    ax_schedule.set_title(f"Schedule Timeline (Period: {period*1000:.3f}ms)")
                    ax_schedule.grid(True, alpha=0.3, axis='x')

                    # Plot DroNet dispatches (y=1)
                    for start, end in model_dispatches.get("dronet", []):
                        ax_schedule.broken_barh([(start * 1000, (end - start) * 1000)],
                                               (0.6, 0.8),
                                               facecolors='orange',
                                               edgecolor='black',
                                               linewidth=0.5)

                    # Plot MLP dispatches (y=0)
                    for start, end in model_dispatches.get("mlp", []):
                        ax_schedule.broken_barh([(start * 1000, (end - start) * 1000)],
                                               (-0.4, 0.8),
                                               facecolors='skyblue',
                                               edgecolor='black',
                                               linewidth=0.5)

                    # Add vertical line for current time (will be updated)
                    time_line = ax_schedule.axvline(x=0, color='red', linewidth=2, linestyle='-', alpha=0.8, label='Current Time')
                    ax_schedule.legend(loc='upper right')

                    text_steer = fpv_fig.text(0.25, 0.48, "", ha='center', fontsize=10, weight='bold')
                    text_coll = fpv_fig.text(0.25, 0.45, "", ha='center', fontsize=10, weight='bold')
                    text_cmd = fpv_fig.text(0.75, 0.48, "", ha='center', fontsize=10, weight='bold', color='blue')
                    text_schedule = fpv_fig.text(0.75, 0.45, "", ha='center', fontsize=10, weight='bold', color='purple')

                    # Adjust layout - suppress tight_layout warnings from figure text elements
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fpv_fig.tight_layout()
                else:
                    # Update camera view (every frame)
                    im_rgb.set_data(rgb_np)

                    # Update processed view (shows cached frame, updated only when DroNet executes)
                    im_processed.set_data(processed_vis)

                    # Update title to show DroNet execution status
                    if dronet_active:
                        processed_title.set_text(f"DroNet Input ({img_size[0]}x{img_size[1]}) - ✓ EXECUTING")
                        processed_title.set_color('green')
                    else:
                        processed_title.set_text(f"DroNet Input ({img_size[0]}x{img_size[1]}) - CACHED")
                        processed_title.set_color('gray')

                    steer_color = 'green' if abs(steer_val) < 0.3 else 'orange'
                    coll_color = 'green' if coll_val < 0.3 else 'red'
                    text_steer.set_text(f"DroNet Steering: {steer_val:+.3f}")
                    text_steer.set_color(steer_color)
                    text_coll.set_text(f"Collision Prob: {coll_val:.3f}")
                    text_coll.set_color(coll_color)
                    text_cmd.set_text(f"Command → Yaw: {target_yaw_rate:+.2f} rad/s, Vel: {target_velocity:.2f} m/s")

                    schedule_text = f"t={time_in_period:.6f}s ({time_in_period*1000:.3f}ms) | DroNet: {'ACTIVE' if dronet_active else 'cached'}, MLP: {'ACTIVE' if mlp_active else 'cached'}"
                    text_schedule.set_text(schedule_text)

                    # Update time line position
                    time_line.set_xdata([time_in_period * 1000])

                if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
                    fpv_fig.canvas.draw_idle()
                    fpv_fig.canvas.flush_events()
                    plt.pause(0.001)

                    # Capture frame for GIF if enabled (with frame skipping)
                    if save_gif and step % args_cli.gif_capture_skip == 0:
                        # Render the figure to a numpy array
                        fpv_fig.canvas.draw()
                        # Use buffer_rgba() which is available in modern matplotlib
                        frame_data = np.frombuffer(fpv_fig.canvas.buffer_rgba(), dtype=np.uint8)
                        frame_width, frame_height = fpv_fig.canvas.get_width_height()
                        # Reshape to (height, width, 4) for RGBA
                        frame = frame_data.reshape(frame_height, frame_width, 4)
                        # Convert RGBA to RGB (remove alpha channel)
                        frame_rgb = frame[:, :, :3].copy()
                        gif_frames.append(frame_rgb)

            # Handle terminations
            if terminated.any() or truncated.any():
                print(f"\n[INFO]: Episode terminated at step {step + 1}")
                obs_dict, _ = env.reset()

            sim_time += control_dt
            step += 1

        print(f"\n[INFO]: Period {period_idx + 1} complete")
        print(f"  Sim time: {sim_time:.3f}s")
        print(f"  DroNet executions: {dronet_exec_count}")
        print(f"  MLP executions: {mlp_exec_count}")

    print(f"\n{'='*70}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total periods: {args_cli.num_periods}")
    print(f"Total steps: {step}")
    print(f"Total sim time: {sim_time:.3f}s")
    print(f"\nExecution counts:")
    print(f"  DroNet: {dronet_exec_count} times")
    print(f"  MLP: {mlp_exec_count} times")
    print(f"\nStatistics:")
    print(f"  Mean steering: {np.mean(steer_history):+.3f}")
    print(f"  Mean collision prob: {np.mean(coll_history):.3f}")
    print(f"  Mean reward: {np.mean(reward_history):.2f}")

    # Save GIF if enabled
    if save_gif and len(gif_frames) > 0:
        print(f"\n[INFO]: Saving GIF with {len(gif_frames)} frames...")
        try:
            from PIL import Image

            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in gif_frames]

            # Calculate frame duration in milliseconds
            # gif_fps is frames per second, duration is ms per frame
            frame_duration = int(1000 / args_cli.gif_fps)

            # Save as GIF
            gif_path = args_cli.save_gif
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=frame_duration,
                loop=0,  # 0 means infinite loop
                optimize=False  # Set to True for smaller file size but slower
            )
            print(f"[INFO]: GIF saved to: {gif_path}")
            print(f"[INFO]: Total frames: {len(gif_frames)}, FPS: {args_cli.gif_fps}, Duration: {len(gif_frames)/args_cli.gif_fps:.2f}s")
        except ImportError:
            print("[ERROR]: PIL (Pillow) not installed. Cannot save GIF. Install with: pip install Pillow")
        except Exception as e:
            print(f"[ERROR]: Failed to save GIF: {e}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
