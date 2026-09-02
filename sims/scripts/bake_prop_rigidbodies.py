"""One-time asset prep: bake KINEMATIC-rigid-body wrapper USDs for the Simple_Warehouse props.

The stock prop meshes (SM_PaletteA_01, SM_CratePlastic_A_01, ...) are plain scenery with no
physics schemas. Isaac Lab's UsdFileCfg only *modifies* rigid-body props if the prim already has
RigidBodyAPI (schemas.modify_rigid_body_properties returns False otherwise), so they never become
collidable rigid bodies at spawn. This bakes a local wrapper per kind: an Xform that references the
real mesh and APPLIES the kinematic RigidBodyAPI at the root + a triangle-mesh CollisionAPI on
every child Mesh. The training/FPV obstacle collection then references these wrappers -> real prop
meshes that actually collide, repositionable per episode.

    <xpurt python> sims/scripts/bake_prop_rigidbodies.py --headless
Outputs: out/rb_props/{pallet,crate,box,cone,klt}.usd
"""
import argparse, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli)
simulation_app = app.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from sims.isaaclab_tasks.warehouse_nav import circuits as C

KINDS = ["pallet", "crate", "box", "cone", "klt", "forklift"]
outdir = os.path.join(_ROOT, "out", "rb_props")
os.makedirs(outdir, exist_ok=True)


def bake(kind):
    url = f"{ISAAC_NUCLEUS_DIR}/{C.PROP_USD[kind]}"
    wp = os.path.join(outdir, f"{kind}.usd")
    if os.path.exists(wp):
        os.remove(wp)
    stage = Usd.Stage.CreateNew(wp)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Prop")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(url)   # compose the real mesh under /Prop
    # kinematic rigid body at the root
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    # triangle-mesh collider on every child mesh (fine for a kinematic obstacle)
    nmesh = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mca.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)  # triangle mesh
            nmesh += 1
    stage.GetRootLayer().Save()
    print(f"[bake] {kind}: {nmesh} mesh colliders -> {wp}", flush=True)
    return nmesh


def main():
    for k in KINDS:
        try:
            n = bake(k)
            if n == 0:
                print(f"[bake] WARNING: {k} had 0 meshes (check the referenced USD structure)", flush=True)
        except Exception as e:
            print(f"[bake] ERROR baking {k}: {e}", flush=True)


if __name__ == "__main__":
    main()
    print("[bake] done", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
