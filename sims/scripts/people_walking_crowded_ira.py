"""
CROWDED variant of people_walking_ira.py.

Renders ~20 REAL WALKING HUMANS in Isaac's full_warehouse using isaacsim.replicator.agent
(IRA), now with:
  1. MANY MORE PEOPLE (~20) on VARIED GoTo routes through BOTH the open south hall
     (hall-crossers) AND down the rack aisles (aisle-walkers), with staggered Idles so
     it reads as a busy working warehouse. Character variants are randomized by IRA for
     visual diversity.
  2. SCATTERED WAREHOUSE-PROP OBSTACLES (pallets, crate/box stacks incl. TALL 1.5-3 m
     columns, cones, KLT bins) placed with the verified no-clip sampler BEFORE the navmesh
     bake, so IRA's navmesh treats them as holes and the people walk AROUND them.

Everything that made the original work is preserved verbatim: stock isaacsim.exp.full.kit
booted headless, runtime ext-enable, a pre-authored NavMeshVolume added BEFORE
set_up_simulation_from_config_file, DataGeneration.run_async playing the timeline through
the replicator orchestrator, the sim_app.update() pump (not asyncio), and the
IRABasicWriter with camera_params:true. The ONLY additions are (a) more routes, (b) the
obstacle scatter (spawned before the bake), and (c) an mp4/stills encode at the end so the
output goes to a NEW filename (people_walking_crowded.mp4) and the original is kept.

Run:
  /scratch2/dima/miniforge3/envs/xpurt/bin/python \
    /scratch/agustin/projects/DIMA/XPU-RT/sims/scripts/people_walking_crowded_ira.py \
    --frames 360 --num 20

Output: out/people_walking/  (people_walking_crowded.mp4 + still_crowded_*.png)
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
parser = argparse.ArgumentParser("IRA people walking (crowded + obstacles)")
parser.add_argument("--frames", type=int, default=360, help="simulation_length (frames @30fps)")
parser.add_argument("--num", type=int, default=20, help="number of characters")
parser.add_argument("--seed", type=int, default=20260715)
parser.add_argument("--out", default="/scratch/agustin/projects/DIMA/out/people_walking")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
args = parser.parse_args()

OUT = args.out
# separate frames dir so we never clobber the original run's frames
FRAMES_DIR = os.path.join(OUT, "frames_crowded")
CFG_DIR = os.path.join(OUT, "_ira_crowded")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

MP4_PATH = os.path.join(OUT, "people_walking_crowded.mp4")

# Stock full experience boots offline on this box; anim/replicator/scripting exts
# are enabled at runtime (resolved from the local extscache, no registry needed).
EXP = "/scratch2/dima/miniforge3/envs/xpurt/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit"

# ---------------------------------------------------------------------------
# ROUTES: mix of HALL-CROSSERS (open south hall, y<=7) and AISLE-WALKERS (down the
# rack aisles at the measured centrelines). Coords are warehouse-local metres
# (same frame the original script + warehouse_nav modules use; metersPerUnit==1).
#   aisle centrelines x: {-22.84,-17.89,-12.94,-7.98,-3.03,1.93}, aisle y[9,24.5]
#   south hall: x[-24,4]  y[-18,7.5]
# Each route: (start_x, start_y, [ (wx, wy), ... ]).
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
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom, Usd

    # obstacle sampler / prop catalogue (pure-python, no isaac deps)
    import random
    from sims.isaaclab_tasks.warehouse_nav import placement as P, circuits as C

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
    # OBSTACLES (before the navmesh bake): scatter warehouse props across the
    # walkable hall floor with the verified no-clip sampler, keeping the people
    # start/goal/route waypoints clear. Spawned into the stage NOW so IRA's
    # navmesh bake treats them as holes and the crowd routes AROUND them.
    # =======================================================================
    rng = random.Random(args.seed)
    # keep every route start + waypoint clear
    exclusions = []
    for (sx0, sy0, wps) in ROUTES:
        exclusions.append((sx0, sy0, 1.3))
        for (wx, wy) in wps:
            exclusions.append((wx, wy, 1.3))
    # dense clutter across the open south hall (the on-camera area); TALL stacks included
    regions = [
        {"box": (-23.0, 3.0, -17.0, 7.0),
         "kinds": ["pallet", "crate", "box", "cone", "klt", "crate", "box", "pallet"],
         "count": (40, 50), "density": 1.0, "stack_prob": 0.6, "stack_h": (1.6, 3.2)},
        # a second, wider band toward the aisle mouths so clutter fills more of the floor
        {"box": (-24.5, 4.5, 5.0, 8.2),
         "kinds": ["crate", "box", "pallet", "cone", "klt"],
         "count": (10, 16), "density": 1.0, "stack_prob": 0.5, "stack_h": (1.4, 2.6)},
    ]
    obstacles = P.sample_no_clip(regions, rng, exclusions=exclusions)
    viol = P.verify_no_clip(obstacles, exclusions=exclusions)
    ground_n = sum(1 for o in obstacles if not o.get("stack"))
    print(f"[people] sampled {len(obstacles)} obstacle prims "
          f"({ground_n} ground + {len(obstacles) - ground_n} stacked); clip violations={len(viol)}",
          flush=True)
    if viol:
        for v in viol[:10]:
            print("   VIOL:", v, flush=True)

    OBS_ROOT = "/World/PeopleObstacles"
    stage.DefinePrim(OBS_ROOT, "Scope")
    for i, o in enumerate(obstacles):
        kind = o["kind"]
        x, y, z = o["pos"]                       # z already at ground/stack centre (metres)
        usd_path = f"{nucleus_root}/{C.PROP_USD[kind]}"
        path = f"{OBS_ROOT}/obs_{i:03d}"
        parent = UsdGeom.Xform.Define(stage, path)
        parent.AddTranslateOp().Set(Gf.Vec3d(m(x), m(y), m(z)))
        parent.AddRotateZOp().Set(math.degrees(o["yaw"]))
        child = stage.DefinePrim(path + "/geo", "Xform")
        child.GetReferences().AddReference(usd_path)
    print(f"[people] spawned {len(obstacles)} obstacle prims under {OBS_ROOT}", flush=True)
    # pump so the referenced prop meshes finish loading BEFORE the navmesh bake
    for _ in range(3000):
        await omni.kit.app.get_app().next_update_async()
        files_loaded, total = ctx.get_stage_loading_status()[1:3]
        if files_loaded == 0 and total == 0:
            break
    print("[people] obstacle assets loaded.", flush=True)

    # ----- add a walkable NavMeshVolume covering hall + aisles (obstacles are inside
    #       it, so they bake as holes) -----
    # walkable floor (meters): x[-25,5]  y[-19,26]  z[0,3.5]
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
    for _ in range(6000):
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
        # orient toward first waypoint
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
        # stagger starts with an Idle (capped) so the crowd trickles into motion
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
    # make sure omni.anim.people points at our (now-populated) command file
    sm.setup_anim_people_command_from_config_file()
    await omni.kit.app.get_app().next_update_async()

    # ----- place the camera: elevated 3/4 view across the hall toward the aisle mouths -----
    def look_at(eye, target, up=Gf.Vec3d(0, 0, 1)):
        eye = Gf.Vec3d(*eye); target = Gf.Vec3d(*target)
        fwd = (target - eye).GetNormalized()                     # camera looks along -Z
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
    # pulled back / slightly higher + wider lens so the busy hall + aisle mouths all fit
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
        # wide-ish lens
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

    # ----- run data generation (plays timeline through replicator; captures RGB) -----
    print(f"[people] running data generation for {args.frames} frames ...", flush=True)
    dg_done = {"v": False}

    def _dg_cb(e):
        dg_done["v"] = True

    dgsub = sm.register_data_generation_callback(_dg_cb)
    task = asyncio.ensure_future(sm.run_data_generation_async(will_wait_until_complete=True))
    steps = 0
    while not task.done():
        await omni.kit.app.get_app().next_update_async()
        steps += 1
        if steps % 200 == 0:
            print(f"[people]   ... stepping ({steps})", flush=True)
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
            # 3 stills: start / mid / end
            import shutil
            idxs = {"start": 0, "mid": len(frame_files) // 2, "end": len(frame_files) - 1}
            for tag, ix in idxs.items():
                dst = os.path.join(OUT, f"still_crowded_{tag}.png")
                shutil.copy(frame_files[ix], dst)
                print(f"[people] still -> {dst}", flush=True)
            print(f"[people] wrote {MP4_PATH}", flush=True)
        else:
            print("[people] WARNING: no RGB frames found to encode", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
