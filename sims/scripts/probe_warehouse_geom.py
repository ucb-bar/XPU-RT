"""Fast top-down render of full_warehouse to see the open aisle layout.
bbox (already probed): x[-28,8] y[-41.4,33.4] z[-0.01,9.3], center (-10,-4).

    python sims/scripts/probe_warehouse_geom.py --headless
"""
import argparse, os, sys
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab","isaaclab_assets","isaaclab_rl","isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(); args_cli.enable_cameras = True
app = AppLauncher(args_cli); simulation_app = app.app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60, device=args_cli.device))
    usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=usd); cfg.func("/World/Warehouse", cfg)
    dome = sim_utils.DomeLightCfg(intensity=500.0); dome.func("/World/L", dome)
    views = {
        "topdown": ((-10.0, -4.0, 42.0), (-10.0, -4.0, 0.0)),
        "topdown_near_origin": ((-5.0, -4.0, 20.0), (-5.0, -4.0, 0.0)),
    }
    cam = Camera(CameraCfg(prim_path="/World/TopCam", update_period=0.0, height=900, width=900,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, clipping_range=(0.1,500))))
    sim.reset()
    for _ in range(90): sim.step()
    os.makedirs(os.path.join(freshscheduler_root, "out"), exist_ok=True)
    for name, (eye, tgt) in views.items():
        cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device),
                                      torch.tensor([tgt], device=sim.device))
        for _ in range(24):
            sim.step(); cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        outp = os.path.join(freshscheduler_root, "out", f"warehouse_{name}.png")
        try:
            import imageio.v2 as imageio; imageio.imwrite(outp, rgb)
        except ImportError:
            from PIL import Image; Image.fromarray(rgb).save(outp)
        print(f"[ok] {outp} mean={rgb.mean():.1f}", flush=True)


if __name__ == "__main__":
    main(); simulation_app.close()
