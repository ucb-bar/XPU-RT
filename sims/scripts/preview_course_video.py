"""FAST environment-preview video: fly a SCRIPTED quad down the warehouse aisle course so we can
iterate on the LOOK of the environment (gates, real props, people, camera) WITHOUT training.

Deliberately lightweight for quick iteration:
  * ONE warehouse (not the 4-64 cloned envs of the training scene) -> small BVH, fast RTX,
  * a scripted Crazyflie flythrough (no physics / no policy) along the aisle through the gates,
  * real warehouse-prop USDs as the aisle obstacles + Isaac People characters,
  * low-ish res + few settle steps/frame.

Edit VIS_GATES / PROPS / PEOPLE / the path waypoints, re-run, watch the mp4. Once the env looks
right, train with train_warehouse_nav.py (same coordinates).

    <xpurt python> sims/scripts/preview_course_video.py --headless --video out/aisle_preview.mp4 \
        --frames 160 --width 960 --height 540
"""
import argparse, math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="out/aisle_preview.mp4")  # (kept; per-view names used)
parser.add_argument("--views", type=str, default="chase,fpv,topdown,side",
                    help="comma-separated camera views to render (chase/fpv/topdown/side)")
parser.add_argument("--frames", type=int, default=160)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--settle", type=int, default=4, help="render sub-steps per frame (RTX converge)")
parser.add_argument("--dome", type=float, default=250.0)
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
from pxr import UsdGeom, Gf
from sims.isaaclab_tasks.warehouse_nav import mdp_gates


def _set_pose(prim, pos, quat_wxyz):
    """Directly set a prim's translate+orient xformOps (robust; XFormPrim.set_world_poses was
    silently leaving the drone at the origin)."""
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    ops["xformOp:translate"].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    w, x, y, z = [float(v) for v in quat_wxyz]
    if "xformOp:orient" in ops:
        ops["xformOp:orient"].Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))

WH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
PROPS_DIR = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props"
CF2X = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"

# gates come straight from the task definition so the preview matches what we train on
GATES = mdp_gates.GATES  # [((x,y,z), yaw), ...]

# real warehouse-prop obstacles scattered in the aisle BETWEEN the gates (visual only here).
# (usd, x, y, z, yaw, scale)
PROPS = [
    (f"{PROPS_DIR}/SM_PaletteA_01.usd", -8.9, 11.0, 0.0, 0.3, 1.0),
    (f"{PROPS_DIR}/SM_CratePlastic_A_01.usd", -7.1, 15.0, 0.0, 0.8, 1.0),
    (f"{PROPS_DIR}/S_TrafficCone.usd", -8.6, 12.8, 0.0, 0.0, 1.0),
    (f"{PROPS_DIR}/S_TrafficCone.usd", -7.4, 18.6, 0.0, 0.0, 1.0),
    (f"{PROPS_DIR}/SM_CardBoxA_01.usd", -8.8, 19.2, 0.0, 0.5, 1.0),
    (f"{PROPS_DIR}/SM_CratePlastic_A_01.usd", -6.9, 10.2, 0.0, 0.0, 1.0),
]
# people: DISABLED in the preview. Isaac People characters are skinned only by the
# omni.anim.graph RUNTIME; a static/USD-bound walk clip is ignored (verified), so without the
# full anim.graph integration they render as an uncanny T-pose — worse than absent. Dynamic
# obstacles (animated people vs. a moving forklift/AMR) is a separate decision; left empty here.
PEOPLE = []


def _q(yaw):
    return quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw]))[0]


def _catmull(waypts, t):
    """piecewise-linear-with-smoothing position along a list of (x,y,z) at t in [0,1]."""
    n = len(waypts) - 1
    f = t * n
    i = min(int(f), n - 1)
    a = torch.tensor(waypts[i]); b = torch.tensor(waypts[i + 1])
    u = f - i
    u = u * u * (3 - 2 * u)  # smoothstep
    return a + (b - a) * u


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    sim_utils.UsdFileCfg(usd_path=WH).func("/World/WH", sim_utils.UsdFileCfg(usd_path=WH))
    d = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95)); d.func("/World/Dome", d)

    # gate frames (reuse the task's frame geometry, placed at absolute /World paths)
    for name, cfg in mdp_gates.make_gate_scene().items():
        pth = cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/env0")
        cfg.spawn.func(pth, cfg.spawn, translation=tuple(cfg.init_state.pos), orientation=tuple(cfg.init_state.rot))

    # real props
    for i, (usd, x, y, z, yaw, sc) in enumerate(PROPS):
        c = sim_utils.UsdFileCfg(usd_path=usd, scale=(sc, sc, sc))
        c.func(f"/World/prop_{i}", c, translation=(x, y, z), orientation=tuple(_q(yaw).tolist()))

    # people (static pose)
    for i, (char, x, y, yaw) in enumerate(PEOPLE):
        c = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/{char}/{char}.usd")
        c.func(f"/World/person_{i}", c, translation=(x, y, 0.0), orientation=tuple(_q(yaw).tolist()))

    # scaled Crazyflie mesh (visual drone) — bumped to x12 so the (small, dark) quad reads
    # against the dark aisle floor; the white props are its most visible signature.
    # NOTE: in a bare SimulationContext, moving a plain Xform prim (XFormPrim.set_world_poses OR
    # direct USD xformOps) does NOT propagate to the Fabric stage the RTX renderer reads once the
    # sim is playing — the mesh stays pinned to its SPAWN pose (verified). The only reliable way
    # to reposition a non-physics visual each frame is to DELETE + RESPAWN it at the new pose.
    dcfg = sim_utils.UsdFileCfg(usd_path=CF2X, scale=(14.0, 14.0, 14.0))  # respawned each frame below

    cam = Camera(CameraCfg(prim_path="/World/PreviewCam", update_period=0.0,
                           height=args_cli.height, width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 400))))
    sim.reset()
    for _ in range(60):
        sim.step()

    # drone flythrough waypoints: aisle mouth -> through each gate -> past last gate
    wp = [(-8.0, 6.0, 1.2)] + [(g[0][0], g[0][1], g[0][2]) for g in GATES] + [(-8.0, 24.0, 1.3)]
    # a MOVING forklift (dynamic obstacle): drives SOUTH down the east side of the aisle, so the
    # drone passes it head-on. (Warehouse-vehicle dynamics per the "both" decision; animated
    # people are a follow-up.) Respawned per frame like the drone.
    FORKLIFT = f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd"
    fcfg = sim_utils.UsdFileCfg(usd_path=FORKLIFT)
    fwp = [(-6.9, 23.0, 0.0), (-6.9, 8.0, 0.0)]  # north -> south along the east lane

    def cam_for(view, p, pdev):
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        if view == "fpv":       # nose cam just ahead of the drone, looking forward down the aisle
            e, t = [px, py + 0.2, pz + 0.12], [px, py + 4.0, pz - 0.25]
        elif view == "topdown":  # high overhead, slight offset to avoid the straight-down singularity
            e, t = [px + 0.4, py - 0.6, pz + 7.5], [px, py, pz]
        elif view == "side":     # 3/4 tracking from the east-behind-above (stays inside the aisle)
            e, t = [px + 1.3, py - 1.8, pz + 1.2], [px - 0.1, py + 0.4, pz]
        else:                    # chase (default)
            e, t = [px - 0.4, py - 2.8, pz + 1.1], [px, py + 0.6, pz - 0.2]
        return (torch.tensor([e], device=pdev), torch.tensor([t], device=pdev))

    views = [v.strip() for v in args_cli.views.split(",") if v.strip()]
    import imageio.v2 as imageio
    dev = sim.device
    for view in views:
        frames = []
        for i in range(args_cli.frames):
            t = i / (args_cli.frames - 1)
            pos = _catmull(wp, t).to(dev)
            nxt = _catmull(wp, min(t + 0.02, 1.0)).to(dev)
            heading = torch.atan2((nxt[1] - pos[1]), (nxt[0] - pos[0]))
            # respawn drone at new pose (only reliable per-frame move in a bare SimulationContext)
            sim.stage.RemovePrim("/World/DroneVis")
            pl = pos.tolist()
            dcfg.func("/World/DroneVis", dcfg, translation=(pl[0], pl[1], pl[2]),
                      orientation=tuple(_q(heading.item()).tolist()))
            # respawn the moving forklift (drives south, faces -y)
            sim.stage.RemovePrim("/World/Forklift")
            fp = [fwp[0][k] + (fwp[1][k] - fwp[0][k]) * t for k in range(3)]
            fcfg.func("/World/Forklift", fcfg, translation=tuple(fp),
                      orientation=tuple(_q(-math.pi / 2).tolist()))
            simulation_app.update(); simulation_app.update()
            e, tg = cam_for(view, pos, dev)
            cam.set_world_poses_from_view(e, tg)
            for _ in range(args_cli.settle):
                sim.step(); cam.update(dt=sim.get_physics_dt())
            frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            if i % 40 == 0:
                print(f"[preview:{view}] frame {i}/{args_cli.frames}", flush=True)
        outp = os.path.join(freshscheduler_root, "out", f"aisle_preview__{view}.mp4")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        imageio.mimwrite(outp, frames, fps=25, quality=7)
        print(f"[preview] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
