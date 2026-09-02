"""Render canonical inspection views of ANY Isaac USD scene to PNGs.

A batch (non-interactive) companion to the isaac_repl_daemon: give it a USD path (or one of the
stock warehouse names) and it drops a camera at a few canonical viewpoints — top-down, a
drone-altitude eye-level pan, and a 3/4 overhead — so you can look at a scene without hand-
writing a camera script each time. For live/iterative inspection prefer the REPL daemon.

    <xpurt python> sims/scripts/inspect_scene.py --headless --usd <path-or-name> --out_dir out/inspect
    <xpurt python> sims/scripts/inspect_scene.py --headless --usd full_warehouse \
        --center -8 15 1.2 --look_dir 0 1 0    # look down the aisle
"""
import argparse, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True,
                    help="USD path, or a stock name (full_warehouse, warehouse, warehouse_with_forklifts)")
parser.add_argument("--out_dir", type=str, default="out/inspect")
parser.add_argument("--center", type=float, nargs=3, default=None, help="scene focus point x y z")
parser.add_argument("--dome", type=float, default=200.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

STOCK = {"full_warehouse", "warehouse", "warehouse_with_forklifts", "warehouse_multiple_shelves"}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    usd = args_cli.usd
    if usd in STOCK:
        usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/{usd}.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=usd)
    cfg.func("/World/Scene", cfg)
    dome = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95))
    dome.func("/World/InspectDome", dome)
    cam = Camera(CameraCfg(prim_path="/World/InspectCam", update_period=0.0,
                           height=args_cli.height, width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 800))))
    sim.reset()
    for _ in range(90):
        sim.step()

    # focus point: given, else scene bbox center at ~1.2 m
    if args_cli.center is not None:
        cx, cy, cz = args_cli.center
    else:
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(sim.stage.GetPrimAtPath("/World/Scene")).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        cx, cy, cz = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, 1.2

    views = {
        "topdown": ((cx + 0.1, cy, 45.0), (cx, cy, 0.0)),
        "eye": ((cx, cy - 6.0, 1.6), (cx, cy + 6.0, 1.4)),
        "diagonal": ((cx + 12.0, cy - 12.0, 8.0), (cx, cy, 1.0)),
    }
    os.makedirs(args_cli.out_dir, exist_ok=True)
    import imageio.v2 as imageio
    for name, (eye, tgt) in views.items():
        cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device),
                                      torch.tensor([tgt], device=sim.device))
        for _ in range(30):
            sim.step(); cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        p = os.path.join(args_cli.out_dir, f"inspect__{name}.png")
        imageio.imwrite(p, rgb)
        print(f"[ok] {p} mean={rgb.mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
