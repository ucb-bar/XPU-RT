"""Kinematic LOOK-preview of the ``dense_crossing`` scenario (design approval before training).

Builds the single photoreal warehouse, floods the south loading hall with a PACKED, non-clipping
clutter field (pallets/crates/boxes stacked 1-3 high, free-standing shelf units, poles, barrels,
cones) via ``scenario_dense.sample_dense_field``, then flies a scripted quad south->north along a
gentle weave corridor through the field and renders a chase flythrough. Motion is kinematic (a
look-preview; real banking comes from the trained-policy demo later).

KEY quirk (same as preview_circuits.py): in a bare SimulationContext, moving a plain prim does
NOT sync to the RTX render, so the drone visual is DELETE+RESPAWNED every frame.

    <xpurt python> sims/scripts/preview_dense.py --headless --frames 200
"""
import argparse, math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--settle", type=int, default=4)
parser.add_argument("--dome", type=float, default=250.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--density", type=float, default=1.0)
parser.add_argument("--view", type=str, default="chase")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import random
import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from sims.isaaclab_tasks.warehouse_nav import circuits as C
from sims.isaaclab_tasks.warehouse_nav import scenario_dense as SD

WH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
CF2X = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
_ROOT = "/World/DENSE"   # everything scenario-specific lives here


def _q(yaw):
    return tuple(quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw]))[0].tolist())


def _spawn_shelf(prim, x, y, z_base, yaw):
    """Free-standing shelf unit ~0.9 (x) x 0.5 (y) x 2.0 (z): 4 uprights + 3 decks (primitive)."""
    hx, hy, H = 0.45, 0.25, SD.KIND_H["shelf"]
    post = (0.05, 0.05, H)
    c, s = math.cos(yaw), math.sin(yaw)

    def place(name, off, size):
        ox = off[0] * c - off[1] * s
        oy = off[0] * s + off[1] * c
        cfg = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.32, 0.34, 0.4), metallic=0.6))
        cfg.func(f"{prim}_{name}", cfg, translation=(x + ox, y + oy, z_base + off[2]), orientation=_q(yaw))

    for i, (px, py) in enumerate([(-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)]):
        place(f"post{i}", (px, py, H / 2), post)
    for i, dz in enumerate((0.35, 1.1, 1.85)):
        place(f"deck{i}", (0.0, 0.0, dz), (2 * hx + 0.05, 2 * hy + 0.05, 0.04))


def _spawn_dense_obstacle(idx, o):
    kind = o["kind"]; x, y, zb = o["pos"]; prim = f"{_ROOT}/obs_{idx}"
    if kind == "pole":
        cfg = sim_utils.CylinderCfg(radius=0.12, height=SD.KIND_H["pole"],
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.22, 0.15)))
        cfg.func(prim, cfg, translation=(x, y, zb + SD.KIND_H["pole"] / 2), orientation=_q(0.0))
    elif kind == "barrel":
        cfg = sim_utils.CylinderCfg(radius=0.3, height=SD.KIND_H["barrel"],
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.6)))
        cfg.func(prim, cfg, translation=(x, y, zb + SD.KIND_H["barrel"] / 2), orientation=_q(o["yaw"]))
    elif kind == "shelf":
        _spawn_shelf(prim, x, y, zb, o["yaw"])
    else:  # real USD prop (pallet/crate/box/klt/cone) — base-pivoted, sits at translation z=zb
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{C.PROP_USD[kind]}")
        cfg.func(prim, cfg, translation=(x, y, zb), orientation=_q(o["yaw"]))


def _catmull(wp, t):
    n = len(wp) - 1
    f = t * n
    i = min(int(f), n - 1)
    a = np.array(wp[i]); b = np.array(wp[i + 1])
    u = f - i
    u = u * u * (3 - 2 * u)
    return a + (b - a) * u


def _cam(view, p, dev):
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    if view == "topdown":
        e, t = [px + 0.4, py - 0.6, pz + 8.0], [px, py, pz]
    else:  # chase: behind (-y) and above, looking forward (+y)
        e, t = [px - 0.5, py - 3.0, pz + 1.2], [px, py + 1.0, pz - 0.1]
    return torch.tensor([e], device=dev), torch.tensor([t], device=dev)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    sim_utils.UsdFileCfg(usd_path=WH).func("/World/WH", sim_utils.UsdFileCfg(usd_path=WH))
    d = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95)); d.func("/World/Dome", d)
    dcfg = sim_utils.UsdFileCfg(usd_path=CF2X, scale=(12.0, 12.0, 12.0))
    cam = Camera(CameraCfg(prim_path="/World/Cam", update_period=0.0, height=args_cli.height,
                           width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 400))))
    sim.reset()
    for _ in range(60):
        sim.step()
    dev = sim.device

    rng = random.Random(args_cli.seed)
    sim.stage.RemovePrim(_ROOT)
    simulation_app.update()
    field = SD.sample_dense_field(rng, density=args_cli.density)
    for idx, o in enumerate(field):
        _spawn_dense_obstacle(idx, o)
    print(f"[dense] spawned {len(field)} obstacle items (packed field)", flush=True)
    simulation_app.update(); simulation_app.update()

    wp = SD.DENSE_CROSSING["waypoints"]
    import imageio.v2 as imageio
    frames = []
    for i in range(args_cli.frames):
        t = i / (args_cli.frames - 1)
        pos = _catmull(wp, t)
        nxt = _catmull(wp, min(t + 0.02, 1.0))
        heading = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
        sim.stage.RemovePrim("/World/DroneVis")
        dcfg.func("/World/DroneVis", dcfg, translation=tuple(float(v) for v in pos), orientation=_q(heading))
        simulation_app.update(); simulation_app.update()
        e, tg = _cam(args_cli.view, pos, dev)
        cam.set_world_poses_from_view(e, tg)
        for _ in range(args_cli.settle):
            sim.step(); cam.update(dt=sim.get_physics_dt())
        frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
        if i % 40 == 0:
            print(f"[dense] frame {i}/{args_cli.frames}", flush=True)
    outp = os.path.join(freshscheduler_root, "out", "dense_crossing.mp4")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    imageio.mimwrite(outp, frames, fps=25, quality=7)
    print(f"[dense] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main(); simulation_app.close()
