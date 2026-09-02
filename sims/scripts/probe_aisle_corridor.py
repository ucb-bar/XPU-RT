"""Dump every geometry prim whose world AABB intrudes into the fused-nav flight corridor
down the x=-8 aisle, so we can see exactly what the drone is clipping.

    <env_isaaclab py> sims/scripts/probe_aisle_corridor.py --headless
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="full_warehouse")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

# flight corridor: aisle spawn/gate x-window, full traversal y, z around cruise altitude 2.0
CX = (-8.9, -7.1)
CY = (6.0, 22.0)
CZ = (0.3, 2.9)


def intersects(mn, mx):
    return (mn[0] <= CX[1] and mx[0] >= CX[0] and
            mn[1] <= CY[1] and mx[1] >= CY[0] and
            mn[2] <= CZ[1] and mx[2] >= CZ[0])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/{args_cli.scene}.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=usd)
    cfg.func("/World/Warehouse", cfg)
    sim.reset()
    for _ in range(60):
        sim.step()
    stage = sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    hits = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        try:
            b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            continue
        if b.IsEmpty():
            continue
        mn, mx = b.GetMin(), b.GetMax()
        mn = [float(mn[i]) for i in range(3)]; mx = [float(mx[i]) for i in range(3)]
        if intersects(mn, mx):
            hits.append({"name": prim.GetName(), "path": str(prim.GetPath()),
                         "min": [round(v, 2) for v in mn], "max": [round(v, 2) for v in mx]})
    hits.sort(key=lambda h: (h["min"][1], h["min"][0]))
    print(f"[corridor] x{CX} y{CY} z{CZ}: {len(hits)} intruding meshes", flush=True)
    for h in hits:
        print(f"  y[{h['min'][1]:6.2f},{h['max'][1]:6.2f}] x[{h['min'][0]:6.2f},{h['max'][0]:6.2f}] "
              f"z[{h['min'][2]:5.2f},{h['max'][2]:5.2f}]  {h['name']}  {h['path']}", flush=True)
    app.close()


main()
