"""
Render REAL WALKING HUMANS in Isaac's full_warehouse using isaacsim.replicator.agent (IRA).

This uses IRA's SimulationManager standalone flow (the designed path for animated
People): load scene -> bake navmesh -> spawn characters -> attach AnimationGraph +
omni.anim.people CharacterBehavior script -> drive them with a GoTo command file ->
run the IRABasicWriter to capture RGB frames.

Why this works where a bare SimulationContext failed: the CharacterBehavior BehaviorScript
only registers (so ag.get_character() is non-None and the biped leaves T-pose) when
omni.kit.scripting is actually running AND the timeline is played through the replicator
orchestrator. IRA's experience + DataGeneration.run_async do exactly that.

Run:
  /scratch2/dima/miniforge3/envs/xpurt/bin/python \
    /scratch/agustin/projects/DIMA/XPU-RT/sims/scripts/people_walking_ira.py \
    --frames 480 --num 8

Output: out/people_walking/  (RGB frames + people_walking.mp4 + stills)
"""

import argparse
import asyncio
import os
import sys

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("IRA people walking")
parser.add_argument("--frames", type=int, default=480, help="simulation_length (frames @30fps)")
parser.add_argument("--num", type=int, default=8, help="number of characters")
parser.add_argument("--seed", type=int, default=20260715)
parser.add_argument("--out", default="/scratch/agustin/projects/DIMA/out/people_walking")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
args = parser.parse_args()

OUT = args.out
FRAMES_DIR = os.path.join(OUT, "frames")
CFG_DIR = os.path.join(OUT, "_ira")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

# Stock full experience boots offline on this box; anim/replicator/scripting exts
# are enabled at runtime (resolved from the local extscache, no registry needed).
EXP = "/scratch2/dima/miniforge3/envs/xpurt/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit"

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
    print(f"[people] scene_path       = {scene_path}", flush=True)
    print(f"[people] character_root   = {character_root}", flush=True)

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
    #       bakes the navmesh with the NavMeshVolume we are about to add -----
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

    # ----- add a walkable NavMeshVolume covering hall + aisles -----
    # walkable floor (meters): x[-25,5]  y[-19,26]  z[0,3.5]
    def m(v):  # meters -> stage units
        return v / mpu
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

    # ----- routes: aisle-walkers + hall-crossers (meters, z=0 floor) -----
    # aisle x centrelines: {-22.84,-17.89,-12.94,-7.98,-3.03,1.93}, aisles y[9,24.9]
    # south hall y[-18,7.5], x[-24,4]
    routes = [
        # (start_x, start_y, [ (wx, wy), ... ])   -- hall crossers
        (-22.0, -6.0, [(-4.0, -6.0), (2.0, -5.0)]),
        (2.0,  -2.5, [(-10.0, -2.5), (-20.0, -3.5)]),
        (-18.0, 4.0, [(-6.0, -2.0), (0.0, -12.0)]),
        (-6.0, -16.0, [(-6.0, -8.0), (-5.0, 0.0)]),
        # aisle walkers (stay inside aisle corridors)
        (-7.98, 10.0, [(-7.98, 17.0), (-7.98, 24.0)]),
        (-12.94, 24.0, [(-12.94, 17.0), (-12.94, 10.5)]),
        (-3.03, 10.5, [(-3.03, 17.0), (-3.03, 23.5)]),
        (-17.89, 23.5, [(-17.89, 16.0), (-17.89, 10.5)]),
    ]

    def yaw_deg(dx, dy):
        return float(np.degrees(np.arctan2(dy, dx)))

    # ----- reposition characters to route starts + orient toward first waypoint -----
    n = min(len(char_prims), len(routes))
    cmd_lines = []
    for i in range(n):
        prim = char_prims[i]
        name = names[i]
        sx0, sy0, wps = routes[i]
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
        # stagger starts a little with an Idle, then GoTo each waypoint
        idle = round(0.4 * i, 1)
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
        # rows are local axes in world: X=right, Y=up, Z=-forward (USD camera)
        m = Gf.Matrix4d(
            right[0], right[1], right[2], 0.0,
            trueup[0], trueup[1], trueup[2], 0.0,
            -fwd[0], -fwd[1], -fwd[2], 0.0,
            0.0, 0.0, 0.0, 1.0)
        return m.ExtractRotationQuat()

    cams = CameraUtil.get_cameras_in_stage()
    print(f"[people] cameras in stage: {[str(c.GetPath()) for c in cams]}", flush=True)
    # low-ish 3/4 view: below the ceiling trusses/lamps, looking down the hall
    # toward the aisle mouths so hall-crossers pass through the foreground.
    eye_m = (3.0, -16.5, 4.4)
    tgt_m = (-9.0, 2.0, 1.5)
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
            camg.GetFocalLengthAttr().Set(18.0)
            camg.GetFocusDistanceAttr().Set(m(25.0))
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
