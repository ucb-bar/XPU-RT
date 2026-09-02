"""RICH physics-flight render — matches the kinematic-preview reference (full warehouse, ~800
obstacles, top-down overview) but with the REAL physics drone.

The physics env carries only the drone (+ forklift + people capsules); the dense ~800-obstacle
clutter is injected as cheap STATIC VISUAL prims (like preview_circuits) so we bypass the physics
RigidObjectCollection pool cap. Views: onboard (nose FPV), chase, overview (top-down, roof hidden).

    <xpurt python> sims/scripts/fpv_rich_flight.py --headless \
        --task Isaac-Drone-Warehouse-Grand-Rich-Crazyflie-FPV-v0 --circuit warehouse_grand \
        --view overview --frames 400 --speed 1.0 --tag grandrich
"""
import argparse, math, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Grand-Rich-Crazyflie-FPV-v0")
parser.add_argument("--circuit", type=str, default="warehouse_grand")
parser.add_argument("--view", type=str, default="onboard", choices=["onboard", "chase", "overview"])
parser.add_argument("--frames", type=int, default=400)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tag", type=str, default="rich")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import random
import numpy as np, torch
import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401  (register)
from sims.isaaclab_tasks.warehouse_nav import circuits as C
from sims.isaaclab_tasks.warehouse_nav import placement as P
from sims.isaaclab_tasks.warehouse_nav import mdp_nav

_SCALED = ("cone", "klt", "crate", "box", "pallet")


def _q(yaw):
    return tuple(quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw]))[0].tolist())


def inject_clutter(circuit_name, seed):
    """Spawn the FULL-density circuit layout as STATIC non-colliding visual prims (reuses the
    kinematic-preview approach) — the ~800 dense obstacles the physics collection can't hold."""
    circ = C.CIRCUITS_BY_NAME[circuit_name]
    rng = random.Random(seed)
    items = C.sample_obstacles(circ, rng, density=1.0)
    n = 0
    for i, o in enumerate(items):
        kind = o["kind"]
        if kind not in C.PROP_USD:
            continue
        s = P.PROP_SCALE if kind in _SCALED else 1.0
        x, y, z = o["pos"]
        # NON-colliding (disable any collider authored in the raw prop USD) so the drone flies THROUGH
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{C.PROP_USD[kind]}", scale=(s, s, s),
                                   collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
        cfg.func(f"/World/RICH/obs_{i}", cfg, translation=(x, y, z), orientation=_q(float(o["yaw"])))
        n += 1
    return n


def hide_roof(stage):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    nh = 0
    for pr in stage.Traverse():
        if pr.GetTypeName() != "Mesh":
            continue
        try:
            b = cache.ComputeWorldBound(pr).ComputeAlignedRange()
            if not b.IsEmpty() and b.GetMin()[2] > 6.3:
                UsdGeom.Imageable(pr).MakeInvisible(); nh += 1
        except Exception:
            pass
    return nh


def pursuit_action(uenv, speed_scale):
    g = mdp_nav.goal_vector_b(uenv)
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    a = torch.zeros(uenv.num_envs, 4, device=uenv.device)
    a[:, 2] = (torch.atan2(gy, gx) / (math.pi / 3)).clamp(-1, 1)
    a[:, 1] = (gz * 1.5).clamp(-1, 1)
    facing = gx.clamp(min=0.0)
    a[:, 0] = (speed_scale * (0.4 + 0.6 * facing)).clamp(0, 1)
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
    robot = uenv.scene["robot"]
    dev = uenv.device
    view = args_cli.view
    cam = uenv.scene[{"onboard": "fpv_camera", "chase": "chase_camera", "overview": "overview_camera"}[view]]

    obs, _ = env.reset()
    n = inject_clutter(args_cli.circuit, args_cli.seed)
    # hide the placeholder capsule "people" (the reference warehouse shots had no cylinders; real
    # animated people are the separate IRA clip). Injected posed human meshes can replace these.
    from pxr import UsdGeom as _UG
    for i in range(8):
        pr = uenv.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Person_{i}")
        if pr.IsValid():
            _UG.Imageable(pr).MakeInvisible()
    print(f"[rich] injected {n} visual props; view={view} task={args_cli.task}", flush=True)
    if view == "overview":
        nh = hide_roof(uenv.sim.stage)
        print(f"[rich] hid {nh} roof prims", flush=True)
        cam.set_world_poses_from_view(torch.tensor([[-10.0, 2.5, 46.0]], device=dev),
                                      torch.tensor([[-10.0, 3.0, 0.0]], device=dev))
    for _ in range(4):
        env.step(torch.zeros(1, 4, device=dev))

    frames = []
    for i in range(args_cli.frames):
        if view == "chase":
            p = robot.data.root_pos_w[0]; q = robot.data.root_quat_w[0]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]).item(), 1 - 2 * (q[2] * q[2] + q[3] * q[3]).item())
            fwd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=dev)
            eye = p + torch.tensor([0.0, 0.0, 1.0], device=dev) - fwd * 2.2
            cam.set_world_poses_from_view(eye.unsqueeze(0), p.unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(pursuit_action(uenv, args_cli.speed))
        f = grab(cam)
        if f is not None:
            frames.append(f)
        if i % 50 == 0:
            pp = robot.data.root_pos_w[0]
            print(f"[rich] step {i}/{args_cli.frames} pos=({pp[0]:.1f},{pp[1]:.1f},{pp[2]:.1f})", flush=True)
        if bool(terminated[0].item()):  # drone terminates only on ground/timeout here (props non-colliding)
            print(f"[rich] terminated at step {i}", flush=True); break

    outdir = os.path.join(_ROOT, "out", "fpv")
    os.makedirs(outdir, exist_ok=True)
    if frames:
        outp = os.path.join(outdir, f"rich_{view}__{args_cli.tag}.mp4")
        imageio.mimwrite(outp, frames, fps=30, quality=7)
        imageio.imwrite(os.path.join(outdir, f"rich_{view}__{args_cli.tag}.png"), frames[len(frames) // 2])
        print(f"[rich] wrote {outp} ({len(frames)} frames)", flush=True)
    else:
        print("[rich] WARNING: no frames", flush=True)


if __name__ == "__main__":
    main()
    print("[rich] done; hard-exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
