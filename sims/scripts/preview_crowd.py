"""Kinematic LOOK-preview of the crowd_nav scenario (for design approval before training).

Builds ONE warehouse, spawns a CROWD of Isaac People character meshes in the open loading hall,
and advances them with the SAME social-force update the task uses (``scenario_crowd.crowd_step``)
so the preview shows exactly the training-time motion: many people walking in DIFFERENT
directions, steering around each other, keeping a hard min-separation (NON-overlapping). A
scripted Crazyflie flies a PLANAR path (fixed altitude) south->north across the crowd; a chase
cam follows it. One mp4.

Motion is kinematic (approved as a look-preview). Per the verified render-sync note, moving a
plain visual prim in a bare SimulationContext does NOT propagate to the RTX render, so each
frame the drone and every character are DELETED + RESPAWNED at their new pose.

KNOWN LIMITATION: Isaac People characters are skinned only by the omni.anim.graph RUNTIME; a
static/USD-bound clip renders as a T-pose. So the walkers here glide in a fixed T-pose — correct
MOTION + spacing, no limb animation (animated walking is a separate follow-up). For training the
people are person-tagged capsule colliders (see scenario_crowd.make_crowd_collection_cfg); pass
--capsules to preview those instead of the character meshes.

    <xpurt python> sims/scripts/preview_crowd.py --headless --frames 220 --n 20
"""
import argparse, math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=220)
parser.add_argument("--n", type=int, default=20, help="crowd size (people)")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--settle", type=int, default=4, help="render sub-steps per frame (RTX converge)")
parser.add_argument("--dome", type=float, default=300.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--view", type=str, default="chase", help="chase | topdown")
parser.add_argument("--dt", type=float, default=1 / 25, help="crowd integration dt per frame (s)")
parser.add_argument("--capsules", action="store_true",
                    help="preview the training capsule colliders instead of character meshes")
parser.add_argument("--out", type=str, default="out/crowd_nav_preview.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from sims.isaaclab_tasks.warehouse_nav import scenario_crowd as SC

WH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
CF2X = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
_ROOT = "/World/CROWD"   # everything crowd-specific lives here

# a variety of Isaac People characters, cycled across the crowd
CHARS = ["F_Business_02", "male_adult_construction_01_new", "female_adult_police_01_new",
         "M_Medical_01", "male_adult_construction_05", "female_adult_business_02"]

_ZONE = SC.CROWD_ZONE
_Z = SC.Z_TARGET


def _q(yaw):
    return tuple(quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]),
                                     torch.tensor([float(yaw)]))[0].tolist())


def _cam(view, p, dev):
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    if view == "topdown":
        e, t = [px + 0.5, py - 0.8, pz + 12.0], [px, py, pz]
    else:  # chase: behind (south of) + above, looking north down the crowd
        e, t = [px - 0.4, py - 4.0, pz + 2.0], [px, py + 2.0, pz - 0.2]
    return torch.tensor([e], device=dev), torch.tensor([t], device=dev)


def main():
    dev = args_cli.device
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=dev))
    sim_utils.UsdFileCfg(usd_path=WH).func("/World/WH", sim_utils.UsdFileCfg(usd_path=WH))
    d = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95)); d.func("/World/Dome", d)
    dcfg = sim_utils.UsdFileCfg(usd_path=CF2X, scale=(14.0, 14.0, 14.0))  # respawned per frame

    cam = Camera(CameraCfg(prim_path="/World/CrowdCam", update_period=0.0, height=args_cli.height,
                           width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 400))))
    sim.reset()
    for _ in range(60):
        sim.step()

    # ---- init the crowd exactly like scenario_crowd.reset_crowd (B=1) ----------------------
    torch.manual_seed(args_cli.seed)
    C = args_cli.n
    lo = torch.tensor([_ZONE["x"][0], _ZONE["y"][0]], device=dev)
    hi = torch.tensor([_ZONE["x"][1], _ZONE["y"][1]], device=dev)
    centre = 0.5 * (lo + hi)
    pos = lo + (hi - lo) * torch.rand(1, C, 2, device=dev)
    pos = SC._separate(pos, lo, hi, iters=12)
    goal = (2 * centre[None, None, :] - pos) + (torch.rand(1, C, 2, device=dev) - 0.5) * 3.0
    goal = torch.max(torch.min(goal, hi[None, None, :]), lo[None, None, :])
    slo, shi = SC.WALK_SPEED_RANGE
    speed = slo + (shi - slo) * torch.rand(1, C, device=dev)

    # per-character USD (or a capsule primitive with --capsules)
    def _char(i):
        name = CHARS[i % len(CHARS)]
        return sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/{name}/{name}.usd")

    # ---- drone PLANAR path: south edge -> north edge, fixed altitude, gentle x weave --------
    y0, y1 = _ZONE["y"][0] + 1.0, _ZONE["y"][1] - 1.0
    wp = [(-3.0, y0, _Z), (-4.0, y0 + 0.30 * (y1 - y0), _Z), (-2.0, y0 + 0.60 * (y1 - y0), _Z),
          (-3.5, y0 + 0.85 * (y1 - y0), _Z), (-3.0, y1, _Z)]

    def _path(t):
        n = len(wp) - 1
        f = t * n
        i = min(int(f), n - 1)
        a = np.array(wp[i]); b = np.array(wp[i + 1])
        u = f - i
        u = u * u * (3 - 2 * u)
        return a + (b - a) * u

    import imageio.v2 as imageio
    frames = []
    yaw_prev = torch.zeros(1, C, device=dev)
    for i in range(args_cli.frames):
        # advance the crowd one step (same math as move_crowd)
        pos, goal, vel = SC.crowd_step(pos, goal, speed, lo, hi, args_cli.dt)
        yaw = torch.atan2(vel[..., 1], vel[..., 0])
        moving = vel.norm(dim=-1) > 1e-3
        yaw = torch.where(moving, yaw, yaw_prev)  # keep last facing when momentarily stopped
        yaw_prev = yaw

        # respawn every person at its new pose (feet on the floor at z=0)
        sim.stage.RemovePrim(f"{_ROOT}/people")
        simulation_app.update()
        for j in range(C):
            px, py = float(pos[0, j, 0]), float(pos[0, j, 1])
            if args_cli.capsules:
                pc = sim_utils.CapsuleCfg(radius=SC.PERSON_R, height=SC.PERSON_H - 2 * SC.PERSON_R,
                                          visual_material=sim_utils.PreviewSurfaceCfg(
                                              diffuse_color=(0.25, 0.45, 0.75)))
                pc.func(f"{_ROOT}/people/p_{j}", pc, translation=(px, py, SC.PERSON_Z),
                        orientation=_q(float(yaw[0, j])))
            else:
                cc = _char(j)
                cc.func(f"{_ROOT}/people/p_{j}", cc, translation=(px, py, 0.0),
                        orientation=_q(float(yaw[0, j])))

        # respawn the drone at its scripted planar pose
        t = i / (args_cli.frames - 1)
        dp = _path(t)
        nxt = _path(min(t + 0.02, 1.0))
        heading = math.atan2(nxt[1] - dp[1], nxt[0] - dp[0])
        sim.stage.RemovePrim("/World/DroneVis")
        dcfg.func("/World/DroneVis", dcfg, translation=tuple(float(v) for v in dp), orientation=_q(heading))

        simulation_app.update(); simulation_app.update()
        e, tg = _cam(args_cli.view, dp, dev)
        cam.set_world_poses_from_view(e, tg)
        for _ in range(args_cli.settle):
            sim.step(); cam.update(dt=sim.get_physics_dt())
        frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
        if i % 40 == 0:
            print(f"[crowd] frame {i}/{args_cli.frames}", flush=True)

    outp = os.path.join(freshscheduler_root, args_cli.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    imageio.mimwrite(outp, frames, fps=25, quality=7)
    print(f"[crowd] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
