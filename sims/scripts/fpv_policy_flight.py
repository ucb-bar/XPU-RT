"""Fly the physics drone with Dima's TRAINED locomotion (steering) policy and record onboard FPV.

Loads the steering checkpoint into the warehouse LOCOMOTION env (DirectThrustMoment action +
steering obs + route-steering command), runs the policy, and records the fpv + chase cameras.
This replaces the geometric-controller pursuit with the reused learned low-level controller.

    <xpurt python> sims/scripts/fpv_policy_flight.py --headless \
        --task Isaac-Drone-Warehouse-Grand-Locomotion-FPV-v0 \
        --checkpoint logs/rsl_rl/crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt \
        --frames 500 --tag grandloco
"""
import argparse, math, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Grand-Locomotion-FPV-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--frames", type=int, default=500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tag", type=str, default="loco")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import importlib
import numpy as np, torch
import gymnasium as gym
import imageio.v2 as imageio
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401  (register)


def grab(cam):
    rgb = cam.data.output["rgb"]
    if rgb is None or rgb.shape[0] == 0:
        return None
    return rgb[0, ..., :3].cpu().numpy().astype(np.uint8)


def main():
    spec = gym.spec(args_cli.task)
    mod, cls = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(mod), cls)()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    amod, acls = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(importlib.import_module(amod), acls)()
    # model_6998.pt (2026-04-13) was trained with std_type="scalar" (std_param); the cfg was later
    # switched to "log" (log_std_param). Match the checkpoint so the state_dict loads.
    try:
        agent_cfg.actor.distribution_cfg.std_type = "scalar"
    except Exception:
        pass

    env = gym.make(args_cli.task, cfg=cfg)
    uenv = env.unwrapped
    fpv, chase, robot = uenv.scene["fpv_camera"], uenv.scene["chase_camera"], uenv.scene["robot"]
    dev = uenv.device

    wrapped = RslRlVecEnvWrapper(env)
    runner_cfg = {
        "algorithm": agent_cfg.algorithm.to_dict(),
        "actor": {"class_name": agent_cfg.actor.class_name, "hidden_dims": agent_cfg.actor.hidden_dims,
                  "activation": agent_cfg.actor.activation, "obs_normalization": agent_cfg.actor.obs_normalization,
                  "distribution_cfg": agent_cfg.actor.distribution_cfg.to_dict()},
        "critic": {"class_name": agent_cfg.critic.class_name, "hidden_dims": agent_cfg.critic.hidden_dims,
                   "activation": agent_cfg.critic.activation, "obs_normalization": agent_cfg.critic.obs_normalization},
        "obs_groups": agent_cfg.obs_groups, "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": 1, "save_interval": 1, "experiment_name": agent_cfg.experiment_name,
        "empirical_normalization": False,
    }
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)
    print(f"[fpvpol] loaded {args_cli.checkpoint}; task={args_cli.task}", flush=True)

    obs, _ = wrapped.reset()
    fpv_frames, chase_frames = [], []
    for i in range(args_cli.frames):
        p = robot.data.root_pos_w[0]
        q = robot.data.root_quat_w[0]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]).item(), 1 - 2 * (q[2] * q[2] + q[3] * q[3]).item())
        fwd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=dev)
        eye = p + torch.tensor([0.0, 0.0, 0.9], device=dev) - fwd * 1.6
        chase.set_world_poses_from_view(eye.unsqueeze(0), p.unsqueeze(0))
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wrapped.step(act)
        f, c = grab(fpv), grab(chase)
        if f is not None:
            fpv_frames.append(f)
        if c is not None:
            chase_frames.append(c)
        if i % 50 == 0:
            print(f"[fpvpol] step {i}/{args_cli.frames} pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})", flush=True)

    outdir = os.path.join(_ROOT, "out", "fpv")
    os.makedirs(outdir, exist_ok=True)
    for name, frames in (("onboard", fpv_frames), ("chase", chase_frames)):
        if not frames:
            print(f"[fpvpol] WARNING: no {name} frames", flush=True); continue
        outp = os.path.join(outdir, f"fpv_{name}__{args_cli.tag}.mp4")
        imageio.mimwrite(outp, frames, fps=30, quality=7)
        print(f"[fpvpol] wrote {outp} ({len(frames)} frames)", flush=True)
    if fpv_frames:
        imageio.imwrite(os.path.join(outdir, f"fpv_still__{args_cli.tag}.png"), fpv_frames[len(fpv_frames) // 2])
        print(f"[fpvpol] wrote still", flush=True)


if __name__ == "__main__":
    main()
    print("[fpvpol] done; hard-exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
