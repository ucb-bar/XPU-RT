#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script that integrates DroNet vision with trained MLP steering policy.

This combines:
1. DroNet: Processes FPV camera → steering angle + collision probability
2. Command Generator: Converts DroNet output → target yaw rate + velocity
3. Trained MLP Policy: Takes state observations → thrust commands

Usage:
    # Use latest checkpoint and random DroNet weights (default)
    conda run -n xpurt python sims/scripts/play_dronet_with_mlp_policy.py --num_envs 4

    # Specify checkpoint explicitly
    conda run -n xpurt python sims/scripts/play_dronet_with_mlp_policy.py \
        --checkpoint logs/rsl_rl/crazyflie_steering_tracking/YYYY-MM-DD_HH-MM-SS/model_XXX.pt

    # Use pretrained DroNet weights
    conda run -n xpurt python sims/scripts/play_dronet_with_mlp_policy.py \
        --dronet_weights path/to/dronet_weights.pth
"""

import argparse
import sys
import os
import numpy as np

# Add paths
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_assets")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_rl")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_contrib")

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="DroNet + MLP policy integration.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to trained MLP policy checkpoint (default: auto-find latest).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--episode_length", type=int, default=1000, help="Episode length in steps.")
parser.add_argument("--dronet_weights", type=str, default=None,
                    help="Path to DroNet weights (default: random initialization).")
parser.add_argument("--model_size", type=str, default="small", choices=["small", "large"],
                    help="DroNet model size.")
parser.add_argument("--no_visualization", action="store_true", help="Disable matplotlib visualization.")
parser.add_argument("--max_yaw_rate", type=float, default=1.0, help="Max yaw rate (rad/s).")
parser.add_argument("--max_velocity", type=float, default=2.0, help="Max forward velocity (m/s).")
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

# Import camera and scene configs
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Visualization
import matplotlib.pyplot as plt

##
# DroNet Model (from quadcopter_fpv_dronet.py)
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
    """Find the latest trained checkpoint automatically from multiple locations.

    Returns:
        Path to the latest checkpoint

    Raises:
        FileNotFoundError: If no checkpoints are found
    """
    import glob

    # Check both possible log directories
    log_dirs = [
        "/scratch2/dima/IsaacLab/logs/rsl_rl/crazyflie_steering_tracking",  # Old training location
        os.path.join(freshscheduler_root, "logs/rsl_rl/crazyflie_steering_tracking"),  # Current location
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
            "Have you run training yet? Try: ./sims/scripts/train_full.sh"
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


def main():
    """Run DroNet + MLP policy integration."""

    # Find checkpoint if not provided
    if args_cli.checkpoint is None:
        print("[INFO]: No checkpoint provided, searching for latest...")
        checkpoint_path = find_latest_checkpoint()
        print(f"[INFO]: Found latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args_cli.checkpoint
        print(f"[INFO]: Using specified checkpoint: {checkpoint_path}")

    # Setup environment with camera
    # NOTE: TrackSteeringEnvCfg_PLAY includes SteeringSceneCfg_WithCamera which has fpv_camera
    env_cfg = TrackSteeringEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Update camera frequency for DroNet
    # Control frequency: 50 Hz (dt=0.01s, decimation=2 → control_dt=0.02s)
    # Set camera to 25 Hz (0.04s) so it updates every 2 control steps (0.04/0.02 = 2)
    # DroNet will run at camera rate (25 Hz), MLP policy at control rate (50 Hz)
    env_cfg.scene.fpv_camera.update_period = 0.04  # 25 Hz

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

    # Visualization setup
    show_viz = not args_cli.headless and not args_cli.no_visualization
    fpv_fig = None
    im_rgb = im_processed = None
    text_steer = text_coll = text_cmd = None

    # Statistics
    steer_history = []
    coll_history = []
    reward_history = []

    # Calculate timing parameters
    # Camera updates at 25 Hz (every 0.04s)
    # Control runs at 50 Hz (every 0.02s = sim.dt * decimation)
    # DroNet runs every: 0.04s / 0.02s = 2 control steps
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    camera_update_interval = int(env_cfg.scene.fpv_camera.update_period / control_dt)

    print("\n[INFO]: Starting integrated DroNet + MLP policy...")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Episode length: {args_cli.episode_length}")
    print(f"  Max yaw rate: {args_cli.max_yaw_rate} rad/s")
    print(f"  Max velocity: {args_cli.max_velocity} m/s")
    print(f"  Camera update: {env_cfg.scene.fpv_camera.update_period}s ({1.0/env_cfg.scene.fpv_camera.update_period:.0f} Hz)")
    print(f"  Control frequency: {1.0/control_dt:.0f} Hz")
    print(f"  DroNet runs every {camera_update_interval} control steps ({1.0/(camera_update_interval*control_dt):.0f} Hz)")
    print()

    # Reset environment
    obs_dict, _ = env.reset()

    # Track camera update for DroNet inference
    last_dronet_step = -camera_update_interval  # Run DroNet on first step

    # Cache last DroNet outputs
    steer_val = 0.0
    coll_val = 0.5
    target_yaw_rate = 0.0
    target_velocity = args_cli.max_velocity * 0.5

    for step in range(args_cli.episode_length):
        # Get camera image
        camera_data = env.unwrapped.scene["fpv_camera"].data.output["rgb"]

        # Run DroNet inference only when camera updates (every ~5 steps at 50Hz control)
        if step - last_dronet_step >= camera_update_interval:
            # DroNet inference
            processed_frame = preprocess_camera_frame(camera_data, img_size, device)
            with torch.no_grad():
                steer, collision = dronet(processed_frame)

            steer_val = steer.item()
            coll_val = collision.item()
            last_dronet_step = step

            # Convert DroNet output to steering commands
            # steering angle (-1 to 1) → yaw rate (rad/s)
            target_yaw_rate = steer_val * args_cli.max_yaw_rate
            # collision prob (0 to 1) → velocity (slower if high collision risk)
            target_velocity = (1.0 - coll_val) * args_cli.max_velocity

        # Manually set command in the environment
        # The MLP policy will use this command as part of its observation
        command_tensor = torch.tensor([[target_yaw_rate, target_velocity]],
                                     device=device, dtype=torch.float32)
        command_tensor = command_tensor.repeat(args_cli.num_envs, 1)
        env.unwrapped.command_manager._terms["steering_command"].target_yaw_rate[:] = command_tensor[:, 0]
        env.unwrapped.command_manager._terms["steering_command"].target_velocity[:] = command_tensor[:, 1]

        # Get observations (includes the command we just set)
        obs = obs_dict["policy"]

        # MLP policy inference
        with torch.no_grad():
            actions = mlp_policy.act_inference(obs)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Store statistics
        steer_history.append(steer_val)
        coll_history.append(coll_val)
        reward_history.append(rewards.mean().item())

        # Print progress
        if (step + 1) % 100 == 0:
            dronet_status = "[DroNet updated]" if step == last_dronet_step else ""
            print(f"  Step {step + 1}/{args_cli.episode_length} {dronet_status} | "
                  f"Steer: {steer_val:+.3f} | "
                  f"Coll: {coll_val:.3f} | "
                  f"Cmd: yaw={target_yaw_rate:+.2f}, vel={target_velocity:.2f} | "
                  f"Reward: {np.mean(reward_history[-100:]):.2f}")

        # Visualization
        if show_viz and step % 4 == 0 and camera_data.shape[0] > 0:
            if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                fpv_fig = None

            rgb_np = _tensor_rgb_to_uint8_hwc(camera_data, 0)
            # Preprocess for visualization (same as DroNet input)
            processed_vis_frame = preprocess_camera_frame(camera_data, img_size, device)
            processed_vis = processed_vis_frame[0].permute(1, 2, 0).cpu().numpy()
            processed_vis = (processed_vis * 255).astype(np.uint8)

            if fpv_fig is None:
                plt.ion()
                fpv_fig, axes = plt.subplots(1, 2, num="DroNet + MLP Policy", figsize=(12, 4.5))

                im_rgb = axes[0].imshow(rgb_np)
                axes[0].set_title("FPV Camera")
                axes[0].axis("off")

                im_processed = axes[1].imshow(processed_vis)
                axes[1].set_title(f"DroNet Input ({img_size[0]}x{img_size[1]})")
                axes[1].axis("off")

                text_steer = fpv_fig.text(0.5, 0.02, "", ha='center', fontsize=11, weight='bold')
                text_coll = fpv_fig.text(0.5, 0.06, "", ha='center', fontsize=11, weight='bold')
                text_cmd = fpv_fig.text(0.5, 0.10, "", ha='center', fontsize=11, weight='bold', color='blue')

                plt.tight_layout(rect=[0, 0.12, 1, 1])
            else:
                im_rgb.set_data(rgb_np)
                im_processed.set_data(processed_vis)

                steer_color = 'green' if abs(steer_val) < 0.3 else 'orange'
                coll_color = 'green' if coll_val < 0.3 else 'red'
                text_steer.set_text(f"DroNet Steering: {steer_val:+.3f}")
                text_steer.set_color(steer_color)
                text_coll.set_text(f"Collision Prob: {coll_val:.3f}")
                text_coll.set_color(coll_color)
                text_cmd.set_text(f"Command → Yaw: {target_yaw_rate:+.2f} rad/s, Vel: {target_velocity:.2f} m/s")

            if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
                fpv_fig.canvas.draw_idle()
                fpv_fig.canvas.flush_events()
                plt.pause(0.001)

        # Handle terminations
        if terminated.any() or truncated.any():
            print(f"\n[INFO]: Episode ended at step {step + 1}")
            break

    print(f"\n[INFO]: Integration complete!")
    print(f"  Mean steering: {np.mean(steer_history):+.3f}")
    print(f"  Mean collision prob: {np.mean(coll_history):.3f}")
    print(f"  Mean reward: {np.mean(reward_history):.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
