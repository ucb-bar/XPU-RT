"""Extract the REAL rack/aisle geometry of Isaac's full_warehouse.

Unlike probe_warehouse_geom.py (which only renders a camera view), this traverses the loaded
stage with a UsdGeom.BBoxCache, records per-prim world AABBs for the structural prims
(racks/shelves/pallets/forklift/walls), clusters the rack instances into rows, derives the
aisle centerlines + clear widths between rows, and dumps everything to out/warehouse_geom.json.
It also renders a few LEGIBLE top-down views (low dome light + long focal length) so racks read
as dark rectangles on the light floor instead of blowing out.

    <xpurt python> sims/scripts/probe_warehouse_aisles.py --headless

Read the JSON + the out/warehouse_aisles__*.png afterwards to route the course down real aisles.
"""
import argparse, json, os, re, sys
from collections import defaultdict

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="full_warehouse")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdGeom, Gf

# prim-name keywords that indicate a structural obstacle we care about for routing.
STRUCT_KEYS = ("rack", "shelf", "shelv", "pallet", "palette", "pile", "fork", "beam",
               "bin", "panel", "crate", "box", "wall", "column", "pillar", "cone", "barrel")


def _grp(name: str) -> str:
    """Collapse an instance name to a group key by stripping trailing indices/suffixes."""
    s = re.sub(r"[_\d]+$", "", name)
    return s or name


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/{args_cli.scene}.usd"
    print(f"[info] loading {usd}", flush=True)
    cfg = sim_utils.UsdFileCfg(usd_path=usd)
    cfg.func("/World/Warehouse", cfg)
    dome = sim_utils.DomeLightCfg(intensity=8.0, color=(0.9, 0.9, 0.95))
    dome.func("/World/ProbeDome", dome)

    cam = Camera(CameraCfg(prim_path="/World/ProbeCam", update_period=0.0, height=1000, width=1000,
                           data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=35.0, clipping_range=(0.1, 800))))
    sim.reset()
    for _ in range(120):
        sim.step()

    # ---- stage traversal: world AABBs of structural prims ----
    stage = sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    records = []
    n_seen = 0
    for prim in stage.Traverse():
        n_seen += 1
        name = prim.GetName()
        tname = prim.GetTypeName()
        lname = name.lower()
        is_mesh = (tname == "Mesh")
        kw = any(k in lname for k in STRUCT_KEYS)
        if not (kw or is_mesh):
            continue
        try:
            b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            continue
        if b.IsEmpty():
            continue
        mn, mx = b.GetMin(), b.GetMax()
        mn = [float(mn[0]), float(mn[1]), float(mn[2])]
        mx = [float(mx[0]), float(mx[1]), float(mx[2])]
        size = [mx[i] - mn[i] for i in range(3)]
        ctr = [(mx[i] + mn[i]) / 2 for i in range(3)]
        # keep prims that are sizeable structures (avoid tiny mesh fragments) OR keyword hits
        vol = size[0] * size[1] * size[2]
        if not kw and (size[2] < 0.8 or vol < 0.5):
            continue
        records.append({"name": name, "path": str(prim.GetPath()), "type": tname,
                        "kw": kw, "min": mn, "max": mx, "size": size, "center": ctr})
    print(f"[info] traversed {n_seen} prims; kept {len(records)} structural records", flush=True)

    # ---- group by collapsed name ----
    groups = defaultdict(list)
    for r in records:
        groups[_grp(r["name"])].append(r)
    group_summary = {}
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        cx = [r["center"][0] for r in rs]
        cy = [r["center"][1] for r in rs]
        cz = [r["center"][2] for r in rs]
        group_summary[g] = {
            "count": len(rs),
            "x_range": [min(cx), max(cx)], "y_range": [min(cy), max(cy)], "z_range": [min(cz), max(cz)],
            "typical_size": np.median(np.array([r["size"] for r in rs]), axis=0).round(3).tolist(),
        }

    # ---- identify rack instances (tall, sizeable) and cluster into rows ----
    rack_records = [r for r in records if r["size"][2] > 1.5 and
                    (("rack" in r["name"].lower()) or ("shelf" in r["name"].lower())
                     or ("shelv" in r["name"].lower()) or ("pile" in r["name"].lower()))]
    # fall back: if no name-matched racks, use all tall large prims
    if len(rack_records) < 4:
        rack_records = [r for r in records if r["size"][2] > 2.0 and (r["size"][0] * r["size"][1]) > 1.0]

    def cluster_1d(vals, gap):
        vals = sorted(vals)
        clusters, cur = [], [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] <= gap:
                cur.append(v)
            else:
                clusters.append(cur); cur = [v]
        clusters.append(cur)
        return clusters

    aisle_info = {}
    if rack_records:
        rc_x = [r["center"][0] for r in rack_records]
        rc_y = [r["center"][1] for r in rack_records]
        spread_x = max(rc_x) - min(rc_x)
        spread_y = max(rc_y) - min(rc_y)
        # rows separated along the axis with larger spread of rack centers
        axis = "x" if spread_x >= spread_y else "y"
        vals = rc_x if axis == "x" else rc_y
        rows = cluster_1d(vals, gap=1.5)
        row_centers = [float(np.mean(c)) for c in rows]
        # per-row extent along the OTHER axis + the row's footprint edges along `axis`
        row_desc = []
        for c in rows:
            members = [r for r in rack_records if (r["center"][0] if axis == "x" else r["center"][1]) in c
                       or abs((r["center"][0] if axis == "x" else r["center"][1]) - np.mean(c)) <= 1.5]
            lo = min((r["min"][0] if axis == "x" else r["min"][1]) for r in members)
            hi = max((r["max"][0] if axis == "x" else r["max"][1]) for r in members)
            row_desc.append({"center": float(np.mean(c)), "extent_lo": float(lo), "extent_hi": float(hi),
                             "n": len(members)})
        row_desc.sort(key=lambda d: d["center"])
        # aisles = gaps between consecutive rows' facing edges
        aisles = []
        for a, b in zip(row_desc[:-1], row_desc[1:]):
            clear = b["extent_lo"] - a["extent_hi"]
            aisles.append({"between_rows": [round(a["center"], 2), round(b["center"], 2)],
                           "centerline_" + axis: round((a["extent_hi"] + b["extent_lo"]) / 2, 2),
                           "clear_width": round(clear, 2)})
        aisle_info = {"row_separation_axis": axis, "n_rows": len(row_desc), "rows": row_desc,
                      "aisles": aisles,
                      "rack_top_z": round(max(r["max"][2] for r in rack_records), 2),
                      "rack_bottom_z": round(min(r["min"][2] for r in rack_records), 2)}

    world_min = [min(r["min"][i] for r in records) for i in range(3)] if records else None
    world_max = [max(r["max"][i] for r in records) for i in range(3)] if records else None
    out = {
        "scene": args_cli.scene,
        "world_bbox_min": world_min, "world_bbox_max": world_max,
        "n_structural_prims": len(records),
        "group_summary": group_summary,
        "aisle_analysis": aisle_info,
        "rack_instances": [{"name": r["name"], "center": [round(c, 2) for c in r["center"]],
                            "size": [round(s, 2) for s in r["size"]]} for r in rack_records],
    }
    os.makedirs(os.path.join(freshscheduler_root, "out"), exist_ok=True)
    jp = os.path.join(freshscheduler_root, "out", "warehouse_geom.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {jp}", flush=True)
    print(f"[summary] groups: " + ", ".join(f"{g}({v['count']})" for g, v in list(group_summary.items())[:15]),
          flush=True)
    if aisle_info:
        print(f"[aisles] axis={aisle_info['row_separation_axis']} rows={aisle_info['n_rows']} "
              f"aisles={[a['clear_width'] for a in aisle_info['aisles']]}", flush=True)

    # ---- legible renders ----
    cx = (world_min[0] + world_max[0]) / 2 if world_min else -10.0
    cy = (world_min[1] + world_max[1]) / 2 if world_min else -4.0
    views = {
        "topdown_lowlight": ((cx + 0.1, cy, 55.0), (cx, cy, 0.0)),
        "highangle": ((cx + 20.0, cy - 30.0, 22.0), (cx, cy, 1.0)),
    }
    for name, (eye, tgt) in views.items():
        cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device),
                                      torch.tensor([tgt], device=sim.device))
        for _ in range(30):
            sim.step(); cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        p = os.path.join(freshscheduler_root, "out", f"warehouse_aisles__{name}.png")
        import imageio.v2 as imageio
        imageio.imwrite(p, rgb)
        print(f"[ok] {p} mean={rgb.mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
