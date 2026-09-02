"""Kinematic LOOK-preview of the circuit library (for design approval before training).

For each circuit: builds the scene (gate frames, a structured obstacle sample, vehicles), flies a
scripted quad along the route, and renders a chase flythrough. Motion is kinematic (approved as a
look-preview; real banking comes from the trained-policy demo later). One mp4 per circuit.

    <xpurt python> sims/scripts/preview_circuits.py --headless --circuits all --frames 120
"""
import argparse, math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--circuits", type=str, default="all")
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--settle", type=int, default=4)
parser.add_argument("--dome", type=float, default=250.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--seeds", type=str, default="", help="comma list of seeds to render in ONE boot (overrides --seed)")
parser.add_argument("--view", type=str, default="chase")
parser.add_argument("--tag", type=str, default="", help="extra output filename suffix (preserves prior renders)")
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

WH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
CF2X = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
_ROOT = "/World/CIRC"   # everything circuit-specific lives here; cleared between circuits

_GATE_OPEN, _GATE_BAR, _GATE_THICK = 1.5, 0.14, 0.12
_GATE_OUTER = _GATE_OPEN + 2 * _GATE_BAR
# ground-anchored centre-z per obstacle kind (props sit on the floor; pole is a 2 m cylinder)
_KIND_Z = {"pole": 1.0, "crate": 0.0, "pallet": 0.0, "box": 0.0, "cone": 0.0, "klt": 0.0}


def _q(yaw):
    return tuple(quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw]))[0].tolist())


def _spawn_gate(gi, center, gyaw):
    cx, cy, cz = center
    c, s = math.cos(gyaw), math.sin(gyaw)
    half = _GATE_OPEN / 2 + _GATE_BAR / 2
    bars = [((-half, 0.0, 0.0), (_GATE_BAR, _GATE_THICK, _GATE_OUTER)),
            ((+half, 0.0, 0.0), (_GATE_BAR, _GATE_THICK, _GATE_OUTER)),
            ((0.0, 0.0, +half), (_GATE_OPEN, _GATE_THICK, _GATE_BAR)),
            ((0.0, 0.0, -half), (_GATE_OPEN, _GATE_THICK, _GATE_BAR))]
    for bi, (off, size) in enumerate(bars):
        ox = off[0] * c - off[1] * s
        oy = off[0] * s + off[1] * c
        cfg = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.75, 0.05), emissive_color=(0.35, 0.28, 0.02)))
        cfg.func(f"{_ROOT}/gate_{gi}_{bi}", cfg, translation=(cx + ox, cy + oy, cz + off[2]), orientation=_q(gyaw))


def _spawn_obstacle(idx, o):
    kind = o["kind"]; x, y, z = o["pos"]  # z already correct (ground/stack) from placement.py
    if kind == "pole":
        cfg = sim_utils.CylinderCfg(radius=0.12, height=2.0, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.75, 0.22, 0.15)))
        cfg.func(f"{_ROOT}/obs_{idx}", cfg, translation=(x, y, z), orientation=_q(0.0))
    else:
        from sims.isaaclab_tasks.warehouse_nav import placement as _P
        s = _P.PROP_SCALE if kind in ("cone", "klt", "crate", "box", "pallet") else 1.0
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{C.PROP_USD[kind]}", scale=(s, s, s))
        cfg.func(f"{_ROOT}/obs_{idx}", cfg, translation=(x, y, z), orientation=_q(o["yaw"]))


def _catmull(wp, t):
    n = len(wp) - 1
    f = t * n
    i = min(int(f), n - 1)
    a = np.array(wp[i]); b = np.array(wp[i + 1])
    u = f - i
    u = u * u * (3 - 2 * u)
    return a + (b - a) * u


def _polyline(wp, s):
    """Constant-speed follow of a polyline: s in [0,1] is fraction of TOTAL arc length.
    Unlike _catmull this does NOT ease at interior waypoints, so a vehicle drives THROUGH
    corners at speed (turning) instead of stopping dead at every point."""
    pts = [np.asarray(p, dtype=float) for p in wp]
    segs = [np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)]
    total = sum(segs) or 1.0
    d = max(0.0, min(1.0, s)) * total
    for k, L in enumerate(segs):
        if d <= L or k == len(segs) - 1:
            u = (d / L) if L > 1e-6 else 0.0
            return pts[k] + (pts[k + 1] - pts[k]) * u
        d -= L
    return pts[-1]


def _cam(view, p, dev):
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    if view == "overview":   # FIXED straight-down top-down (roof hidden in main) of the packed floor
        e, t = [-10.0, 2.5, 46.0], [-10.0, 3.0, 0.0]
    elif view == "topdown":
        e, t = [px + 0.4, py - 0.6, pz + 8.0], [px, py, pz]
    else:
        e, t = [px - 0.5, py - 3.0, pz + 1.2], [px, py + 1.0, pz - 0.1]
    return torch.tensor([e], device=dev), torch.tensor([t], device=dev)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    sim_utils.UsdFileCfg(usd_path=WH).func("/World/WH", sim_utils.UsdFileCfg(usd_path=WH))
    d = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95)); d.func("/World/Dome", d)
    dcfg = sim_utils.UsdFileCfg(usd_path=CF2X, scale=(12.0, 12.0, 12.0))
    fcfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{C.PROP_USD['forklift']}")
    cam = Camera(CameraCfg(prim_path="/World/Cam", update_period=0.0, height=args_cli.height,
                           width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 400))))
    sim.reset()
    for _ in range(60):
        sim.step()
    dev = sim.device

    # OVERVIEW: hide the roof/ceiling/high beams so a straight-down camera sees the whole floor.
    if args_cli.view == "overview":
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        nh = 0
        for p in sim.stage.Traverse():
            if p.GetTypeName() != "Mesh":
                continue
            try:
                b = cache.ComputeWorldBound(p).ComputeAlignedRange()
                if not b.IsEmpty() and b.GetMin()[2] > 6.3:
                    UsdGeom.Imageable(p).MakeInvisible(); nh += 1
            except Exception:
                pass
        print(f"[overview] hid {nh} high (roof) prims", flush=True)

    names = [c["name"] for c in C.CIRCUITS] if args_cli.circuits == "all" else args_cli.circuits.split(",")
    seeds = [int(s) for s in args_cli.seeds.split(",")] if args_cli.seeds else [args_cli.seed]
    import imageio.v2 as imageio
    for seed in seeds:
      for name in names:
        circ = C.CIRCUITS_BY_NAME[name]
        rng = random.Random(seed)
        sim.stage.RemovePrim(_ROOT)
        simulation_app.update()
        # gates + structured obstacle sample + vehicles(initial)
        for gi, (gc, gyaw) in enumerate(circ["gates"]):
            _spawn_gate(gi, gc, gyaw)
        veh = circ["vehicles"][0] if circ["vehicles"] else None
        # pick ONE forklift lane per seed (randomized route); even seeds drive it reversed.
        # OPEN lanes are driven forward once (accel/cruise/decel); CLOSED lanes (start==end) are
        # driven as continuous patrol LAPS. Each lane is a distinct way of moving.
        flane, floop = None, False
        if veh and veh.get("moving") and veh.get("lanes"):
            lanes = veh["lanes"]
            flane = list(lanes[seed % len(lanes)])
            if seed % 2 == 0:
                flane = flane[::-1]
            floop = float(np.linalg.norm(np.asarray(flane[0], float) - np.asarray(flane[-1], float))) < 3.0
        # only the CHOSEN lane is carved clear -> the rest of the bay fills with obstacles.
        for idx, o in enumerate(C.sample_obstacles(circ, rng, density=1.0,
                                                   carve_lanes=[flane] if flane else None)):
            _spawn_obstacle(idx, o)
        fyaw = 0.0
        if flane is not None:
            _d0 = np.asarray(flane[1], float) - np.asarray(flane[0], float)
            fyaw = math.atan2(_d0[1], _d0[0])
        simulation_app.update(); simulation_app.update()

        wp = circ["waypoints"]
        frames = []
        for i in range(args_cli.frames):
            t = i / (args_cli.frames - 1)
            pos = _catmull(wp, t)
            nxt = _catmull(wp, min(t + 0.02, 1.0))
            heading = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
            sim.stage.RemovePrim("/World/DroneVis")
            dcfg.func("/World/DroneVis", dcfg, translation=tuple(float(v) for v in pos), orientation=_q(heading))
            if flane is not None:
                if floop:
                    # CLOSED route: continuous patrol laps at constant speed (wraps around).
                    s = (t * 1.5) % 1.0
                    fp = _polyline(flane, s)
                    fn = _polyline(flane, (s + 0.01) % 1.0)
                else:
                    # OPEN route: drive FORWARD once with a global ease-in/ease-out so it
                    # accelerates from rest and decelerates to a stop, turning through corners.
                    se = t * t * (3 - 2 * t)
                    fp = _polyline(flane, se)
                    fn = _polyline(flane, min(se + 0.01, 1.0))
                d = fn - fp
                if abs(d[0]) + abs(d[1]) > 1e-4:
                    fyaw = math.atan2(d[1], d[0])
                sim.stage.RemovePrim(f"{_ROOT}/veh")
                fcfg.func(f"{_ROOT}/veh", fcfg, translation=(float(fp[0]), float(fp[1]), 0.0), orientation=_q(fyaw))
            simulation_app.update(); simulation_app.update()
            e, tg = _cam(args_cli.view, pos, dev)
            cam.set_world_poses_from_view(e, tg)
            for _ in range(args_cli.settle):
                sim.step(); cam.update(dt=sim.get_physics_dt())
            frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            if i % 40 == 0:
                print(f"[circuit:{name} seed{seed}] frame {i}/{args_cli.frames}", flush=True)
        suffix = "" if args_cli.view == "chase" else f"__{args_cli.view}"
        seedtag = "" if seed == 7 else f"__seed{seed}"
        tagsfx = f"__{args_cli.tag}" if args_cli.tag else ""
        outp = os.path.join(freshscheduler_root, "out", f"circuit__{name}{suffix}{seedtag}{tagsfx}.mp4")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        imageio.mimwrite(outp, frames, fps=25, quality=7)
        print(f"[circuit] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
    # imageio has already flushed every mp4 to disk; SimulationApp.close() frequently HANGS on
    # this build (seen spinning for >1 h), so exit hard instead of waiting on a clean teardown.
    print("[circuit] all renders written; hard-exiting to skip the hanging close()", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
