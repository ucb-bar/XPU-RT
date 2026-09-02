"""Preview Isaac's photoreal warehouse as a candidate drone-benchmark environment.

Loads the stock Isaac 5.1 ``Simple_Warehouse`` USD, drops a camera in at drone altitude,
and renders a handful of viewpoints to PNG so we can actually LOOK at the scene before
building a task on it.

Why this scene: it ships with Isaac (zero asset-authoring risk), is RTX-photoreal, has
natural aisles/shelves for obstacle avoidance, and contains genuinely detectable object
classes for YOLO (person, forklift, pallet, box, shelf) -- unlike the competitor sims,
whose "environments" are untextured cylinders (aerial_gym) or a flat-shaded hangar (CRL).

MUST be run with --headless: without it Isaac loads the viewport-dependent render kit and
the camera FREEZES on a display-less box (see run_forest_dronet_demo.sh for the full story).

    python sims/scripts/preview_warehouse.py --headless --out_dir out/warehouse_preview
"""

import argparse
import os
import sys

# Isaac Lab lives outside this repo; match the path setup the other sims/scripts use.
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_assets")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_rl")
sys.path.insert(0, "/scratch2/dima/IsaacLab/source/isaaclab_contrib")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out_dir", type=str, default="out/warehouse_preview")
parser.add_argument("--scene", type=str, default="warehouse_with_forklifts",
                    choices=["warehouse", "warehouse_with_forklifts", "warehouse_multiple_shelves", "full_warehouse"])
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Cameras are the whole point of this script.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below must come AFTER the app launches ---
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Camera poses (eye, target) chosen to show off the aisles, the shelving and the forklifts
# from roughly drone flight altitude (~1.5-2 m).
VIEWPOINTS = [
    # near-top-down (slightly offset to avoid up-vector singularity), low light -> see the floor+racks
    ("topdown_full",   (-9.9, -4.0, 38.0), (-10.0, -4.0, 0.0)),
    ("topdown_origin", (-4.9, 0.0, 22.0), (-5.0, 0.0, 0.0)),
]


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    )

    # The warehouse itself.
    warehouse_usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/{args_cli.scene}.usd"
    print(f"[info] loading warehouse: {warehouse_usd}")
    warehouse_cfg = sim_utils.UsdFileCfg(usd_path=warehouse_usd)
    warehouse_cfg.func("/World/Warehouse", warehouse_cfg)

    # The warehouse USD ships its own lighting, but add a dome so the aisles aren't pitch black.
    dome_cfg = sim_utils.DomeLightCfg(intensity=60.0, color=(0.9, 0.9, 0.95))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    # A free-floating camera we reposition per viewpoint (same trick as the chase cam).
    camera = Camera(
        CameraCfg(
            prim_path="/World/PreviewCam",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, clipping_range=(0.05, 200.0)
            ),
        )
    )

    sim.reset()
    # Let assets finish streaming in before we photograph them, otherwise we capture a
    # half-loaded warehouse.
    for _ in range(120):
        sim.step()

    os.makedirs(args_cli.out_dir, exist_ok=True)
    for name, eye, target in VIEWPOINTS:
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device),
            torch.tensor([target], device=sim.device),
        )
        # Several steps so the RTX renderer converges (denoiser needs a few frames).
        for _ in range(30):
            sim.step()
            camera.update(dt=sim.get_physics_dt())

        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = os.path.join(args_cli.out_dir, f"{args_cli.scene}__{name}.png")
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, rgb)
        except ImportError:
            from PIL import Image

            Image.fromarray(rgb).save(path)
        print(f"[ok] {path}   mean_pixel={rgb.mean():.1f}  (a frozen/black cam would be ~0)")

    print(f"\n[done] wrote {len(VIEWPOINTS)} previews to {args_cli.out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
