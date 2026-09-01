#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL policy for steering angle tracking with Crazyflie."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# Add paths
# Add FreshScheduler root to path (contains sims/ directory)
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
# Add IsaacLab to path
_isaaclab_root = os.environ.get("ISAACLAB_ROOT", "/scratch2/dima/IsaacLab")
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, os.path.join(_isaaclab_root, "source", _pkg))

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL policy for steering angle tracking.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Track-Steering-Vision-Crazyflie-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

# Register custom environments
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.dict import print_dict

# Import configs
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import TrackSteeringEnvCfg
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (
    SteeringTrackingPPORunnerCfg,
)


def main():
    """Train with RSL-RL agent."""
    # Load configurations
    env_cfg = TrackSteeringEnvCfg()
    agent_cfg = SteeringTrackingPPORunnerCfg()

    # Override with command line arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    # Setup logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO]: Logging experiment in directory: {log_root_path}")
    print(f"[INFO]: Run directory: {log_dir}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO]: Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Print configuration
    print("\n[INFO]: Starting training with configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Device: {args_cli.device}")
    print(f"  Seed: {args_cli.seed}\n")

    # Build config dict manually to avoid deprecated fields
    runner_cfg = {
        "algorithm": agent_cfg.algorithm.to_dict(),
        "actor": {
            "class_name": agent_cfg.actor.class_name,
            "hidden_dims": agent_cfg.actor.hidden_dims,
            "activation": agent_cfg.actor.activation,
            "obs_normalization": agent_cfg.actor.obs_normalization,
            "distribution_cfg": agent_cfg.actor.distribution_cfg.to_dict() if agent_cfg.actor.distribution_cfg else None,
        },
        "critic": {
            "class_name": agent_cfg.critic.class_name,
            "hidden_dims": agent_cfg.critic.hidden_dims,
            "activation": agent_cfg.critic.activation,
            "obs_normalization": agent_cfg.critic.obs_normalization,
        },
        "obs_groups": agent_cfg.obs_groups,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "save_interval": agent_cfg.save_interval,
        "experiment_name": agent_cfg.experiment_name,
        "empirical_normalization": False,
    }

    # Create runner
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=args_cli.device)

    # Load checkpoint if resuming
    if args_cli.resume is not None:
        print(f"[INFO]: Resuming training from checkpoint: {args_cli.resume}")
        runner.load(args_cli.resume)
        print(f"[INFO]: Loaded checkpoint at iteration {runner.current_learning_iteration}")
        print(f"[INFO]: Will train for {agent_cfg.max_iterations - runner.current_learning_iteration} more iterations")

    # Run training
    print("[INFO]: Starting PPO training...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"\n[INFO]: Training complete!")
    print(f"[INFO]: Logs and checkpoints saved to: {log_dir}")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run training
    main()
    # Close simulation app
    simulation_app.close()
