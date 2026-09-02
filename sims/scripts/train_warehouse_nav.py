"""Train the warehouse navigation policy with rsl_rl PPO.

Resolves BOTH the env cfg and the PPO runner cfg from the gym task registration, so it works
for the warehouse task (and any task registered with rsl_rl_cfg_entry_point) without editing
this file. State-based by default (fast); add --enable_cameras for the vision variant.

    python sims/scripts/train_warehouse_nav.py --headless --num_envs 128 --max_iterations 3000
"""

import argparse
import importlib
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Nav-Crazyflie-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_note", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--init_noise_std", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from sims.isaaclab_tasks.warehouse_nav.config import crazyflie  # noqa: F401  (registers task)


def _from_entry(entry_point):
    mod, cls = entry_point.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    spec = gym.spec(args_cli.task)
    env_cfg = _from_entry(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _from_entry(spec.kwargs["rsl_rl_cfg_entry_point"])

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if args_cli.init_noise_std is not None:
        agent_cfg.actor.distribution_cfg.init_std = args_cli.init_noise_std
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_note:
        log_dir += f"_{args_cli.run_note}"
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] task={args_cli.task}  env_cfg={type(env_cfg).__name__}  run_dir={log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

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
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=args_cli.device)
    if args_cli.resume is not None:
        runner.load(args_cli.resume)
    print("[INFO] starting PPO...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] training complete: {log_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
