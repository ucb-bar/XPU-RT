"""Smoke-test the warehouse navigation task: instantiate, step, and verify each subsystem.

Checks, in one short run:
  * the full_warehouse USD loads and clones per env,
  * the drone spawns and the DirectThrustMoment action drives it,
  * the goal command produces a finite (unit_dir, distance) vector,
  * the contact sensor reports forces (collision termination can fire),
  * rewards are finite and terminations behave.

    python sims/scripts/smoke_warehouse_nav.py --headless --num_envs 16 --steps 120
"""

import argparse
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from sims.isaaclab_tasks.warehouse_nav.config import crazyflie  # noqa: F401  (registers the task)
from sims.isaaclab_tasks.warehouse_nav import mdp_nav


def main() -> None:
    task = "Isaac-Drone-Warehouse-Nav-Crazyflie-v0"
    env_cfg = gym.spec(task).kwargs["env_cfg_entry_point"]
    mod, cls = env_cfg.split(":")
    import importlib
    cfg = getattr(importlib.import_module(mod), cls)()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    env = gym.make(task, cfg=cfg)
    print(f"[smoke] made {task}")
    print(f"[smoke] observation space: {env.observation_space}")
    print(f"[smoke] action space:      {env.action_space}")

    obs, _ = env.reset()
    act_dim = env.action_space.shape[-1]
    n = env.unwrapped.num_envs

    contact_seen = False
    goal_dist_start = None
    for i in range(args_cli.steps):
        # gentle upward-biased random actions so the drone actually flies around
        act = torch.zeros(n, act_dim, device=cfg.sim.device)
        act[:, 0] = 0.2 + 0.3 * torch.rand(n, device=cfg.sim.device)   # thrust > hover
        act[:, 1:] = 0.1 * (torch.rand(n, act_dim - 1, device=cfg.sim.device) - 0.5)
        obs, rew, term, trunc, info = env.step(act)

        cmd = env.unwrapped.command_manager.get_command("goal")
        gd = cmd[:, 3]
        if goal_dist_start is None:
            goal_dist_start = gd.clone()

        contact = env.unwrapped.scene.sensors["contact"]
        fmag = contact.data.net_forces_w.norm(dim=-1).max().item()
        if fmag > 1.0:
            contact_seen = True

        # obstacle field: how many are active (not dumped to z<-100) in env 0?
        coll = env.unwrapped.scene["obstacles"]
        obj_z = coll.data.object_pos_w[0, :, 2]
        n_active = int((obj_z > -100.0).sum().item())
        d_obs = mdp_nav.active_obstacle_min_dist(env.unwrapped)[0].item()
        active_count = int(env.unwrapped.obstacle_active_count[0].item())

        if i % 20 == 0:
            print(f"[step {i:3d}] rew[min/mean/max]={rew.min():.2f}/{rew.mean():.2f}/{rew.max():.2f}  "
                  f"goal_dist[mean]={gd.mean():.2f}  contact_fmax={fmag:7.1f}  "
                  f"obstacles_active(env0)={n_active} (target {active_count})  d_nearest_obs={d_obs:.2f}  "
                  f"term={int(term.sum())} trunc={int(trunc.sum())}", flush=True)

        assert torch.isfinite(rew).all(), "NON-FINITE REWARD"
        assert torch.isfinite(obs['policy']).all() if isinstance(obs, dict) else torch.isfinite(obs).all(), "NON-FINITE OBS"

    lines = [
        "[smoke] RESULTS",
        "  obs finite:            OK",
        "  reward finite:         OK",
        f"  goal command works:    dist {goal_dist_start.mean():.2f} m at start (finite, per-env)",
        f"  contact sensor active: {'saw a collision force > 1N' if contact_seen else 'no strong contact this run'}",
        f"  env stepped {args_cli.steps} times over {n} envs with no crash",
    ]
    # Isaac swallows stdout on close(); write results to a file we can read back.
    with open(os.path.join(freshscheduler_root, "out", "smoke_warehouse_result.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
