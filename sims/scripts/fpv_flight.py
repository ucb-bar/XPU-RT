"""PHYSICS-driven drone flight in the warehouse with ONBOARD FPV capture (Dima's direction:
"get the drone dynamics in the environment, then validate what the FPV footage looks like").

Unlike preview_circuits.py (kinematic visual mesh, no collisions), this runs the REAL physics
stack: the CRAZYFLIE_CFG articulation + the verified geometric velocity controller + real rack/
obstacle colliders, with the FPV Crazyflie's nose camera parented to the body. A scripted
pure-pursuit route flies the gate course; we record the ONBOARD fpv view + a chase view so the
footage the perception model will consume can be validated before any training.

(Control = geometric velocity controller for this pass; the trained DirectThrustMoment locomotion
policy swaps in next — same physics drone, same cameras.)

    <xpurt python> sims/scripts/fpv_flight.py --headless --enable_cameras --frames 500
"""
import argparse, math, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Nav-Crazyflie-FPV-v0")
parser.add_argument("--frames", type=int, default=500, help="max control steps to record")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--speed", type=float, default=0.6, help="forward-speed scale [0,1] for clean footage")
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import numpy as np, torch
import imageio.v2 as imageio

from isaaclab_tasks.utils import parse_env_cfg
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401  (registers the gym ids)
from sims.isaaclab_tasks.warehouse_nav import mdp_nav


def pursuit_action(uenv, speed_scale):
    """[-1,1]^4 for VelocityCommandAction: yaw toward the current gate, climb to its height, fly
    forward. Gentle gains + capped speed = smooth footage (not the aggressive eval floor)."""
    g = mdp_nav.goal_vector_b(uenv)                     # (N,4) unit dir (yaw frame) + distance
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    a = torch.zeros(uenv.num_envs, 4, device=uenv.device)
    a[:, 2] = (torch.atan2(gy, gx) / (math.pi / 3)).clamp(-1, 1)   # yawrate
    a[:, 1] = (gz * 1.5).clamp(-1, 1)                              # inclination (climb)
    facing = gx.clamp(min=0.0)
    a[:, 0] = (speed_scale * (0.4 + 0.6 * facing)).clamp(0, 1)     # forward speed
    return a


def grab(cam):
    rgb = cam.data.output["rgb"]
    if rgb is None or rgb.shape[0] == 0:
        return None
    return rgb[0, ..., :3].cpu().numpy().astype(np.uint8)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    fpv = uenv.scene["fpv_camera"]
    chase = uenv.scene["chase_camera"]
    robot = uenv.scene["robot"]
    dev = uenv.device
    print(f"[fpv] task={args_cli.task} dt={uenv.step_dt:.4f}s frames<={args_cli.frames}", flush=True)

    obs, _ = env.reset()
    for _ in range(6):                      # let physics + cameras settle
        env.step(torch.zeros(1, 4, device=dev))

    fpv_frames, chase_frames = [], []
    for i in range(args_cli.frames):
        # chase cam: free prim, reposition behind + above the drone each step, looking at it
        p = robot.data.root_pos_w[0]
        q = robot.data.root_quat_w[0]
        # forward (world) from body x-axis via yaw only
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]).item(),
                         1 - 2 * (q[2] * q[2] + q[3] * q[3]).item())
        fwd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=dev)
        eye = p + torch.tensor([0.0, 0.0, 0.9], device=dev) - fwd * 1.6
        chase.set_world_poses_from_view(eye.unsqueeze(0), p.unsqueeze(0))

        action = pursuit_action(uenv, args_cli.speed)
        obs, _, terminated, truncated, _ = env.step(action)

        f, c = grab(fpv), grab(chase)
        if f is not None:
            fpv_frames.append(f)
        if c is not None:
            chase_frames.append(c)
        if i % 50 == 0:
            gp = uenv.command_manager.get_term("goal")
            gp_passed = int(gp.current_idx[0].item()) + int(gp.course_complete[0].item())
            print(f"[fpv] step {i}/{args_cli.frames} pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) gates_passed={gp_passed}", flush=True)
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            reason = "crash" if bool(terminated[0].item()) else "timeout/complete"
            print(f"[fpv] episode ended at step {i} ({reason}); stopping capture", flush=True)
            break

    outdir = os.path.join(_ROOT, "out", "fpv")
    os.makedirs(outdir, exist_ok=True)
    tag = f"__{args_cli.tag}" if args_cli.tag else ""
    for name, frames in (("onboard", fpv_frames), ("chase", chase_frames)):
        if not frames:
            print(f"[fpv] WARNING: no {name} frames captured", flush=True); continue
        outp = os.path.join(outdir, f"fpv_{name}{tag}.mp4")
        imageio.mimwrite(outp, frames, fps=30, quality=7)
        print(f"[fpv] wrote {outp} ({len(frames)} frames)", flush=True)
    # a single onboard still for a quick eyeball
    if fpv_frames:
        imageio.imwrite(os.path.join(outdir, f"fpv_still{tag}.png"), fpv_frames[len(fpv_frames) // 2])
        print(f"[fpv] wrote {os.path.join(outdir, f'fpv_still{tag}.png')}", flush=True)


if __name__ == "__main__":
    main()
    print("[fpv] done; hard-exiting to skip the hanging close()", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
