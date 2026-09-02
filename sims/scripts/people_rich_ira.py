"""
UNIFIED SHOWCASE of the DIMA warehouse:
  * ~20 REAL animated WALKING humans (isaacsim.replicator.agent / IRA pipeline)
  * ~500 DENSE warehouse-prop obstacles (the full `warehouse_grand` no-clip layout:
    pallets, crate/box STACKS incl. tall 1.5-3 m columns, cones, KLT bins)
  * a DRONE flythrough (cf2x) along the warehouse_grand serpentine waypoints
  * a moving FORKLIFT driving one of the circuit's hall lanes
all in ONE Isaac Sim render.

This is people_walking_crowded_ira.py with three additions, none of which disturb the
IRA SimulationManager / character pipeline (that stays verbatim):
  1. the obstacle scatter is REPLACED by the full circuits.sample_obstacles(warehouse_grand)
     layout, spawned as static visual props BEFORE the navmesh bake (people route around
     them). People-route corridors are carved clear so the crowd keeps walking.
  2. a cf2x DRONE + a FORKLIFT are pre-spawned, then DELETE+RESPAWNED every frame INSIDE the
     data-generation update loop (the loop that pumps sim_app.update() while the replicator
     orchestrator plays the timeline). Moving plain prims do NOT sync to the render once the
     timeline is playing unless deleted+respawned each frame (verified full-kit quirk).
  3. mp4/stills are encoded in a post pass AFTER the app closes (SimulationApp.close() hard-
     exits on this build).

Run:
  /scratch2/dima/miniforge3/envs/xpurt/bin/python \
    /scratch/agustin/projects/DIMA/XPU-RT/sims/scripts/people_rich_ira.py \
    --frames 360 --num 20

Output: out/people_walking/  (people_rich.mp4 + still_rich_*.png)
"""

import argparse
import asyncio
import math
import os
import sys

# make `sims.isaaclab_tasks.warehouse_nav` importable (namespace pkg under XPU-RT)
_XPURT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _XPURT_ROOT not in sys.path:
    sys.path.insert(0, _XPURT_ROOT)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("IRA people rich showcase (people + drone + obstacles + forklift)")
parser.add_argument("--frames", type=int, default=360, help="simulation_length (frames @30fps)")
parser.add_argument("--num", type=int, default=20, help="number of characters")
parser.add_argument("--seed", type=int, default=20260715)
parser.add_argument("--obs_seed", type=int, default=7, help="seed for the warehouse_grand obstacle layout")
parser.add_argument("--out", default="/scratch/agustin/projects/DIMA/out/people_walking")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--drone_scale", type=float, default=16.0)
args = parser.parse_args()

OUT = args.out
# separate frames dir so we never clobber prior runs' frames
FRAMES_DIR = os.path.join(OUT, "frames_rich")
CFG_DIR = os.path.join(OUT, "_ira_rich")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

MP4_PATH = os.path.join(OUT, "people_rich.mp4")

# Stock full experience boots offline on this box; anim/replicator/scripting exts
# are enabled at runtime (resolved from the local extscache, no registry needed).
EXP = "/scratch2/dima/miniforge3/envs/xpurt/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit"

# ---------------------------------------------------------------------------
# ROUTES: mix of HALL-CROSSERS (open south hall, y<=7) and AISLE-WALKERS (down the
# rack aisles at the measured centrelines). Coords are warehouse-local metres
# (same frame the circuits/placement modules use; metersPerUnit==1).
#   aisle centrelines x: {-22.84,-17.89,-12.94,-7.98,-3.03,1.93}, aisle y[9,24.5]
#   south hall: x[-24,4]  y[-18,7.5]
# ---------------------------------------------------------------------------
ROUTES = [
    # ---- hall-crossers ----
    (-22.0, -6.0, [(-10.0, -6.0), (2.0, -5.0)]),
    (2.0,  -3.0, [(-10.0, -3.0), (-21.0, -4.0)]),
    (-20.0, 5.0, [(-8.0, -2.0), (1.0, -10.0)]),
    (-6.0, -16.0, [(-6.0, -8.0), (-5.0, 1.0)]),
    (0.0, -15.0, [(-2.0, -6.0), (-4.0, 4.0)]),
    (-23.0, -14.0, [(-14.0, -10.0), (-4.0, -6.0)]),
    (2.0, 6.0, [(-8.0, 2.0), (-18.0, -2.0)]),
    (-15.0, -16.0, [(-15.0, -6.0), (-13.0, 4.0)]),
    (-2.0, -16.0, [(-8.0, -11.0), (-16.0, -6.0)]),
    (-20.0, -2.0, [(-11.0, 3.0), (-2.0, 6.0)]),
    (1.0, -8.0, [(-6.0, -10.0), (-14.0, -14.0)]),
    (-24.0, 2.0, [(-16.0, -4.0), (-8.0, -12.0)]),
    # ---- aisle-walkers ----
    (-7.98, 9.5, [(-7.98, 16.0), (-7.98, 24.0)]),
    (-12.94, 24.0, [(-12.94, 16.0), (-12.94, 10.0)]),
    (-3.03, 10.0, [(-3.03, 17.0), (-3.03, 23.5)]),
    (-17.89, 23.5, [(-17.89, 16.0), (-17.89, 10.5)]),
    (-22.84, 10.5, [(-22.84, 17.0), (-22.84, 24.0)]),
    (1.93, 24.0, [(1.93, 16.0), (1.93, 10.5)]),
    (-7.98, 24.0, [(-7.98, 17.0), (-7.98, 10.5)]),
    (-12.94, 10.5, [(-12.94, 17.0), (-12.94, 24.0)]),
    (-3.03, 24.0, [(-3.03, 16.0), (-3.03, 10.5)]),
    (-17.89, 10.5, [(-17.89, 17.0), (-17.89, 23.5)]),
    (-22.84, 24.0, [(-22.84, 16.0), (-22.84, 10.5)]),
    (1.93, 10.5, [(1.93, 17.0), (1.93, 24.0)]),
]

# ---------------------------------------------------------------------------
# Launch app  (headless, RTX, cameras/rendering on)
# ---------------------------------------------------------------------------
from isaacsim.simulation_app import SimulationApp

CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": True,
    "width": args.width,
    "height": args.height,
}
sim_app = SimulationApp(launch_config=CONFIG, experience=EXP)


# ===========================================================================
async def run():
    import carb
    import numpy as np
    import omni.kit.app
    import omni.usd
    import omni.kit.commands
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom, Usd

    # obstacle sampler / prop catalogue (pure-python, no isaac deps)
    import random
    from sims.isaaclab_tasks.warehouse_nav import placement as P, circuits as C

    # ---- route-follow helpers (from preview_circuits.py) ----
    def _catmull(wp, t):
        n = len(wp) - 1
        f = t * n
        i = min(int(f), n - 1)
        a = np.array(wp[i], dtype=float); b = np.array(wp[i + 1], dtype=float)
        u = f - i
        u = u * u * (3 - 2 * u)
        return a + (b - a) * u

    def _polyline(wp, s):
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

    settings = carb.settings.get_settings()

    # ----- enable required extensions (mirrors IRA SDG scheduler) -----
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext in [
        "omni.kit.scripting",
        "omni.anim.timeline",
        "omni.anim.graph.core",
        "omni.anim.retarget.core",
        "omni.anim.navigation.schema",
        "omni.anim.navigation.core",
        "omni.anim.navigation.bundle",
        "omni.anim.people",
        "omni.kit.mesh.raycast",
        "isaacsim.replicator.agent.core",
    ]:
        ok_ext = ext_manager.set_extension_enabled_immediate(ext, True)
        print(f"[people] enable {ext}: {ok_ext}", flush=True)
    await omni.kit.app.get_app().next_update_async()

    # ----- global settings -----
    rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
    settings.set("/app/scripting/ignoreWarningDialog", True)
    settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
    settings.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", True)
    settings.set("/app/omni.graph.scriptnode/enable_opt_in", False)
    settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
    # capture every frame, no skipping so the video is contiguous
    settings.set("/persistent/exts/isaacsim.replicator.agent/skip_starting_frames", 0)
    settings.set("/persistent/exts/isaacsim.replicator.agent/frame_write_interval", 1)
    settings.set("/persistent/exts/isaacsim.replicator.agent/hide_visualization", True)
    await omni.kit.app.get_app().next_update_async()

    # ----- resolve asset root & scene path -----
    from isaacsim.replicator.agent.core.settings import AssetPaths, PrimPaths
    from isaacsim.replicator.agent.core.stage_util import StageUtil, CharacterUtil, CameraUtil
    from isaacsim.replicator.agent.core.simulation import SimulationManager

    AssetPaths.cache_isaac_sim_asset_root_path()
    scene_path = AssetPaths.default_scene_path()          # .../Simple_Warehouse/full_warehouse.usd
    character_root = AssetPaths.default_character_path()   # .../People/Characters/
    # nucleus root that C.PROP_USD paths are relative to (same root the scene lives under)
    nucleus_root = scene_path.split("/Environments/")[0]
    cf2x_usd = f"{nucleus_root}/Robots/Bitcraze/Crazyflie/cf2x.usd"
    forklift_usd = f"{nucleus_root}/{C.PROP_USD['forklift']}"
    print(f"[people] scene_path       = {scene_path}", flush=True)
    print(f"[people] character_root   = {character_root}", flush=True)
    print(f"[people] nucleus_root     = {nucleus_root}", flush=True)

    # ----- write IRA config + (empty for now) command file -----
    cmd_file = os.path.join(CFG_DIR, "command.txt")
    open(cmd_file, "w").close()
    cfg_path = os.path.join(CFG_DIR, "config.yaml")
    cfg_text = f"""isaacsim.replicator.agent:
  version: 0.7.0
  global:
    seed: {args.seed}
    simulation_length: {args.frames}
  scene:
    asset_path: {scene_path}
  sensor:
    camera_num: 1
  character:
    asset_path: {character_root}
    command_file: {cmd_file}
    filters: []
    num: {args.num}
  robot:
    command_file:
    nova_carter_num: 0
    iw_hub_num: 0
    write_data: false
  replicator:
    writer: IRABasicWriter
    parameters:
      output_dir: {FRAMES_DIR}
      rgb: true
      camera_params: true
      object_info_bounding_box_2d_tight: false
      object_info_bounding_box_2d_loose: false
      object_info_bounding_box_3d: false
      agent_info_skeleton_data: false
      semantic_filter_predicate: class:character;id:*
"""
    with open(cfg_path, "w") as f:
        f.write(cfg_text)

    sm = SimulationManager()
    if not sm.load_config_file(cfg_path):
        print("[people] FATAL: could not load config", flush=True)
        return False

    # ----- pre-open the warehouse ourselves so IRA skips its own re-open and
    #       bakes the navmesh with the NavMeshVolume + obstacles we add -----
    print("[people] opening warehouse stage ...", flush=True)
    StageUtil.open_stage(scene_path)
    # pump until assets loaded
    ctx = omni.usd.get_context()
    for _ in range(4000):
        await omni.kit.app.get_app().next_update_async()
        files_loaded, total = ctx.get_stage_loading_status()[1:3]
        if ctx.get_stage() is not None and files_loaded == 0 and total == 0:
            break
    stage = ctx.get_stage()
    print(f"[people] stage url = {ctx.get_stage_url()}", flush=True)

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"[people] metersPerUnit = {mpu}", flush=True)

    def m(v):  # meters -> stage units
        return v / mpu

    # =======================================================================
    # OBSTACLES (before the navmesh bake): the FULL warehouse_grand no-clip layout
    # (~500 prims) scattered across the whole footprint. People-route corridors are
    # carved clear so the crowd keeps walking. Spawned into the stage NOW so IRA's
    # navmesh bake treats them as holes.
    # =======================================================================
    GRAND = C.CIRCUITS_BY_NAME["warehouse_grand"]
    obstacles = C.sample_obstacles(GRAND, random.Random(args.obs_seed), density=1.0)

    # carve people-route corridors clear (route waypoints are NOT in the circuit's
    # exclusion set, so drop any obstacle within CARVE_R of a densified route point).
    route_pts = []
    for (sx0, sy0, wps) in ROUTES:
        pts = [(sx0, sy0)] + list(wps)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            nseg = max(1, int(seg / 1.0))
            for k in range(nseg + 1):
                tt = k / nseg
                route_pts.append((a[0] + (b[0] - a[0]) * tt, a[1] + (b[1] - a[1]) * tt))
    CARVE_R = 1.05

    def near_route(x, y):
        return any((x - rx) ** 2 + (y - ry) ** 2 < CARVE_R ** 2 for rx, ry in route_pts)

    before = len(obstacles)
    obstacles = [o for o in obstacles if not near_route(o["pos"][0], o["pos"][1])]
    ground_n = sum(1 for o in obstacles if not o.get("stack"))
    print(f"[people] warehouse_grand obstacles: {before} sampled -> {len(obstacles)} after "
          f"route-carve ({ground_n} ground + {len(obstacles) - ground_n} stacked)", flush=True)

    # visual scale for the scalable clutter kinds (matches placement.PROP_SCALE / the preview)
    PROP_SCALE = P.PROP_SCALE
    _SCALED = ("cone", "klt", "crate", "box", "pallet")

    OBS_ROOT = "/World/PeopleObstacles"
    stage.DefinePrim(OBS_ROOT, "Scope")
    for i, o in enumerate(obstacles):
        kind = o["kind"]
        x, y, z = o["pos"]                       # z already at ground/stack centre (metres)
        usd_path = f"{nucleus_root}/{C.PROP_USD[kind]}"
        path = f"{OBS_ROOT}/obs_{i:04d}"
        parent = UsdGeom.Xform.Define(stage, path)
        parent.AddTranslateOp().Set(Gf.Vec3d(m(x), m(y), m(z)))
        parent.AddRotateZOp().Set(math.degrees(o["yaw"]))
        if kind in _SCALED:
            parent.AddScaleOp().Set(Gf.Vec3f(PROP_SCALE, PROP_SCALE, PROP_SCALE))
        child = stage.DefinePrim(path + "/geo", "Xform")
        child.GetReferences().AddReference(usd_path)
    print(f"[people] spawned {len(obstacles)} obstacle prims under {OBS_ROOT}", flush=True)
    # pump so the referenced prop meshes finish loading BEFORE the navmesh bake
    for _ in range(9000):
        await omni.kit.app.get_app().next_update_async()
        files_loaded, total = ctx.get_stage_loading_status()[1:3]
        if files_loaded == 0 and total == 0:
            break
    print("[people] obstacle assets loaded.", flush=True)

    # ----- add a walkable NavMeshVolume covering hall + aisles (obstacles inside it
    #       bake as holes) -----
    nav_path = "/World/PeopleNavVolume"
    from pxr import UsdGeom as _UG
    vol = stage.DefinePrim(nav_path, "NavMeshVolume")
    vol.CreateAttribute("nav:area", Sdf.ValueTypeNames.String).Set("Walkable")
    ext = _UG.Boundable(vol).GetExtentAttr()
    if not ext:
        ext = vol.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array)
    ext.Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    cx, cy, cz = m(-10.0), m(3.5), m(1.75)
    sx, sy, sz = m(30.0), m(45.0), m(3.5)
    vol.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(cx, cy, cz))
    vol.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(sx, sy, sz))
    vol.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(["xformOp:translate", "xformOp:scale"])
    vol.ApplyAPI("NavMeshAreaAPI")
    print(f"[people] NavMeshVolume added: center=({cx:.1f},{cy:.1f},{cz:.1f}) scale=({sx:.1f},{sy:.1f},{sz:.1f})", flush=True)
    await omni.kit.app.get_app().next_update_async()

    # ----- run IRA setup (bakes navmesh + spawns chars + anim graph + scripts + camera) -----
    done = {"v": False}

    def _done_cb(e):
        done["v"] = True

    sub = sm.register_set_up_simulation_done_callback(_done_cb)
    print("[people] set_up_simulation_from_config_file() ...", flush=True)
    sm.set_up_simulation_from_config_file()
    for _ in range(8000):
        await omni.kit.app.get_app().next_update_async()
        if done["v"]:
            break
    sub = None
    if not done["v"]:
        print("[people] FATAL: setup did not complete (navmesh bake or spawn failed)", flush=True)
        return False
    print("[people] IRA setup complete.", flush=True)

    # ----- inspect what was spawned -----
    char_prims = CharacterUtil.get_characters_root_in_stage(count=-1, count_invisible=False)
    names = [CharacterUtil.get_character_name(p) for p in char_prims]
    print(f"[people] spawned {len(char_prims)} characters: {names}", flush=True)

    def yaw_deg(dx, dy):
        return float(np.degrees(np.arctan2(dy, dx)))

    # ----- reposition characters to route starts + orient toward first waypoint -----
    n = min(len(char_prims), len(ROUTES))
    cmd_lines = []
    for i in range(n):
        prim = char_prims[i]
        name = names[i]
        sx0, sy0, wps = ROUTES[i]
        fx, fy = wps[0]
        yw = yaw_deg(fx - sx0, fy - sy0)
        t = prim.GetAttribute("xformOp:translate")
        t.Set(Gf.Vec3d(m(sx0), m(sy0), 0.0))
        orient = prim.GetAttribute("xformOp:orient")
        q = Gf.Rotation(Gf.Vec3d(0, 0, 1), yw).GetQuat()
        if isinstance(orient.Get(), Gf.Quatf):
            orient.Set(Gf.Quatf(q))
        else:
            orient.Set(q)
        idle = round(min(0.3 * i, 5.0), 1)
        if idle > 0:
            cmd_lines.append(f"{name} Idle {idle}")
        prev = (sx0, sy0)
        for (wx, wy) in wps:
            yw2 = yaw_deg(wx - prev[0], wy - prev[1])
            cmd_lines.append(f"{name} GoTo {m(wx):.3f} {m(wy):.3f} 0 {yw2:.1f}")
            prev = (wx, wy)

    with open(cmd_file, "w") as f:
        f.write("\n".join(cmd_lines) + "\n")
    print(f"[people] wrote {len(cmd_lines)} command lines to {cmd_file}", flush=True)
    sm.setup_anim_people_command_from_config_file()
    await omni.kit.app.get_app().next_update_async()

    # ----- place the camera: elevated 3/4 view across the hall toward the aisle mouths -----
    def look_at(eye, target, up=Gf.Vec3d(0, 0, 1)):
        eye = Gf.Vec3d(*eye); target = Gf.Vec3d(*target)
        fwd = (target - eye).GetNormalized()
        right = Gf.Cross(fwd, up).GetNormalized()
        trueup = Gf.Cross(right, fwd).GetNormalized()
        mtx = Gf.Matrix4d(
            right[0], right[1], right[2], 0.0,
            trueup[0], trueup[1], trueup[2], 0.0,
            -fwd[0], -fwd[1], -fwd[2], 0.0,
            0.0, 0.0, 0.0, 1.0)
        return mtx.ExtractRotationQuat()

    cams = CameraUtil.get_cameras_in_stage()
    print(f"[people] cameras in stage: {[str(c.GetPath()) for c in cams]}", flush=True)
    eye_m = (3.5, -17.5, 4.8)
    tgt_m = (-9.0, 3.0, 1.3)
    try:
      if cams:
        cam = cams[0]
        q = look_at(eye_m, tgt_m)
        ct = cam.GetAttribute("xformOp:translate")
        if not ct:
            ct = cam.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
        ct.Set(Gf.Vec3d(m(eye_m[0]), m(eye_m[1]), m(eye_m[2])))
        co = cam.GetAttribute("xformOp:orient")
        if co and co.IsValid():
            if isinstance(co.Get(), Gf.Quatf):
                co.Set(Gf.Quatf(q))
            else:
                co.Set(q)
        else:
            co = cam.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
            co.Set(q)
            cam.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
                ["xformOp:translate", "xformOp:orient"])
        camg = UsdGeom.Camera(cam)
        if camg:
            camg.GetFocalLengthAttr().Set(16.0)
            camg.GetFocusDistanceAttr().Set(m(26.0))
        print("[people] camera placed.", flush=True)
    except Exception as ce:
        import traceback
        print("[people] camera placement FAILED (continuing):", ce, flush=True)
        traceback.print_exc()

    await omni.kit.app.get_app().next_update_async()

    # =======================================================================
    # MOVING PROPS: drone (cf2x) along the warehouse_grand serpentine waypoints,
    # forklift along one hall lane. Pre-spawn once (warms the USD cache so the
    # per-frame re-reference is cheap), then delete+respawn every frame inside the
    # data-generation loop below.
    # =======================================================================
    DRONE_PATH = "/World/RichDrone"
    FORK_PATH = "/World/RichForklift"
    wp = GRAND["waypoints"]                             # (x,y,z) metres
    lanes = GRAND["vehicles"][0]["lanes"]
    flane = list(lanes[0])                              # one hall lane, driven forward once
    ds = float(args.drone_scale)

    def _spawn_drone(pos, heading):
        stage.RemovePrim(DRONE_PATH)
        pr = UsdGeom.Xform.Define(stage, DRONE_PATH)
        pr.AddTranslateOp().Set(Gf.Vec3d(m(pos[0]), m(pos[1]), m(pos[2])))
        pr.AddRotateZOp().Set(math.degrees(heading))
        pr.AddScaleOp().Set(Gf.Vec3f(ds, ds, ds))
        ch = stage.DefinePrim(DRONE_PATH + "/geo", "Xform")
        ch.GetReferences().AddReference(cf2x_usd)

    def _spawn_forklift(pos, heading):
        stage.RemovePrim(FORK_PATH)
        pr = UsdGeom.Xform.Define(stage, FORK_PATH)
        pr.AddTranslateOp().Set(Gf.Vec3d(m(pos[0]), m(pos[1]), m(0.0)))
        pr.AddRotateZOp().Set(math.degrees(heading))
        ch = stage.DefinePrim(FORK_PATH + "/geo", "Xform")
        ch.GetReferences().AddReference(forklift_usd)

    # initial poses (t=0) + warm the cache
    d0 = _catmull(wp, 0.0); d1 = _catmull(wp, 0.02)
    _spawn_drone(d0, math.atan2(d1[1] - d0[1], d1[0] - d0[0]))
    f0 = _polyline(flane, 0.0); f1 = _polyline(flane, 0.01)
    fyaw = math.atan2(f1[1] - f0[1], f1[0] - f0[0])
    _spawn_forklift(f0, fyaw)
    for _ in range(400):
        await omni.kit.app.get_app().next_update_async()
        files_loaded, total = ctx.get_stage_loading_status()[1:3]
        if files_loaded == 0 and total == 0:
            break
    print(f"[people] drone ({cf2x_usd}) + forklift pre-spawned & cached.", flush=True)

    # ----- run data generation (plays timeline through replicator; captures RGB) -----
    print(f"[people] running data generation for {args.frames} frames ...", flush=True)
    dg_done = {"v": False}

    def _dg_cb(e):
        dg_done["v"] = True

    import glob as _glob
    rgb_dir = os.path.join(FRAMES_DIR, "_World_Cameras_Camera", "rgb")
    denom = max(1, args.frames - 1)

    dgsub = sm.register_data_generation_callback(_dg_cb)
    task = asyncio.ensure_future(sm.run_data_generation_async(will_wait_until_complete=True))
    steps = 0
    while not task.done():
        await omni.kit.app.get_app().next_update_async()
        steps += 1
        # ---- drive the drone + forklift by CAPTURE progress ----
        # The replicator orchestrator does NOT advance omni.timeline's current time in
        # this flow, so we key motion off how many RGB frames the writer has produced.
        nf = len(_glob.glob(os.path.join(rgb_dir, "rgb_*.png")))
        t = max(0.0, min(1.0, nf / denom))
        # drone: smooth catmull follow of the serpentine
        pos = _catmull(wp, t)
        nxt = _catmull(wp, min(t + 0.02, 1.0))
        dh = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])
        _spawn_drone(pos, dh)
        # forklift: OPEN lane driven forward once with global ease-in/ease-out
        se = t * t * (3 - 2 * t)
        fp = _polyline(flane, se)
        fn = _polyline(flane, min(se + 0.01, 1.0))
        d = fn - fp
        if abs(d[0]) + abs(d[1]) > 1e-4:
            fyaw = math.atan2(d[1], d[0])
        _spawn_forklift(fp, fyaw)
        if steps % 200 == 0:
            print(f"[people]   ... stepping ({steps}) t={t:.2f} drone=({pos[0]:.1f},{pos[1]:.1f})", flush=True)
    dgsub = None
    print("[people] data generation done.", flush=True)
    return True


ok = False
try:
    # Drive the app: sim_app.update() pumps the kit loop, which resolves the
    # next_update_async() / step_async() awaits inside run().
    task = asyncio.ensure_future(run())
    while not task.done():
        sim_app.update()
    ok = task.result()
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    sim_app.update()
    sim_app.close()

print(f"[people] FINISHED ok={ok}", flush=True)

# ---------------------------------------------------------------------------
# Encode mp4 + stills from the captured RGB frames (after the app is closed).
# ---------------------------------------------------------------------------
if ok:
    try:
        import glob
        import imageio.v2 as imageio

        rgb_dir = os.path.join(FRAMES_DIR, "_World_Cameras_Camera", "rgb")
        frame_files = sorted(glob.glob(os.path.join(rgb_dir, "rgb_*.png")))
        print(f"[people] encoding {len(frame_files)} frames -> {MP4_PATH}", flush=True)
        if frame_files:
            with imageio.get_writer(MP4_PATH, fps=30, macro_block_size=None) as w:
                for fp in frame_files:
                    w.append_data(imageio.imread(fp))
            import shutil
            # skip the very first frames (characters briefly in T-pose before the anim
            # graph engages) when choosing representative stills
            n = len(frame_files)
            idxs = {"start": min(45, n // 6), "mid": n // 2, "end": n - 1}
            for tag, ix in idxs.items():
                dst = os.path.join(OUT, f"still_rich_{tag}.png")
                shutil.copy(frame_files[ix], dst)
                print(f"[people] still -> {dst}", flush=True)
            print(f"[people] wrote {MP4_PATH}", flush=True)
        else:
            print("[people] WARNING: no RGB frames found to encode", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
