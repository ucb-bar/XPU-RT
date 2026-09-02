#!/usr/bin/env python3
"""
Offline 3D-inspection gallery renderer for the reference drone-RL repos.

Renders (no simulator) the environment geometry that ships in the reference
repos so we can visually reason about how each structures tasks/obstacles:

  * CRL-Drone-Racing : GLB stages + gate/obstacle props, and full assembled
                       race tracks parsed from scene_instance.json.
  * aerial_gym       : analytic URDF obstacle assets (trees/panels/objects/walls)
                       built from primitive geometry (cylinders/boxes).

Renderer: trimesh 4.x -> Scene.save_image() (pyglet/OpenGL). There is no
display on this host, so run the whole thing under xvfb-run, e.g.:

  xvfb-run -a -s "-screen 0 1600x1200x24" \
    /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python \
    /scratch/agustin/projects/DIMA/XPU-RT/sims/scripts/inspect_meshes_offline.py

Writes PNGs to /scratch/agustin/projects/DIMA/out/framework_gallery/.
Read-only w.r.t. the reference repos: only reads their assets.
"""
import os
import json
import math
import traceback

import numpy as np
import trimesh

OUT = "/scratch/agustin/projects/DIMA/out/framework_gallery"
CRL = "/scratch/agustin/projects/DIMA/CRL-Drone-Racing/datasets/spy_datasets"
AG = "/scratch/agustin/projects/DIMA/aerial_gym_simulator/resources/models/environment_assets"

RES = (1400, 1000)


# --------------------------------------------------------------------------
# Camera helpers
# --------------------------------------------------------------------------
def _look_at(scene, eye, target, up=(0, 1, 0)):
    """Set scene.camera_transform to a look-at matrix (camera looks down -Z)."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, fwd)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = true_up
    M[:3, 2] = -fwd  # camera looks along -Z
    M[:3, 3] = eye
    scene.camera_transform = M


def _exterior_views(scene):
    """Orbit views from OUTSIDE. Good for compact props/trees whose surfaces
    face outward. Y-up assets."""
    b = scene.bounds
    center = b.mean(axis=0)
    r = max(np.linalg.norm(b[1] - b[0]) / 2.0, 0.5)
    d = 2.1 * r
    n1 = np.array([1.0, 0.75, 1.0]); n1 /= np.linalg.norm(n1)
    n2 = np.array([0.25, 0.2, 1.0]); n2 /= np.linalg.norm(n2)
    return {
        "3q": (center + n1 * d, center, (0, 1, 0)),
        "eye": (center + n2 * d, center, (0, 1, 0)),
    }


def _interior_views(scene):
    """Views from INSIDE an enclosed stage (walls modeled with inward normals).
    Camera sits at the room CENTER (guaranteed open space) at eye height and
    looks toward each end of the longest horizontal axis. Assets Y-up."""
    b = scene.bounds
    center = b.mean(axis=0)
    size = b[1] - b[0]
    lo = b[0]
    axis = 0 if size[0] >= size[2] else 2  # longest horizontal: X(0) or Z(2)
    y_eye = lo[1] + 0.45 * size[1]
    eye = center.copy(); eye[1] = y_eye
    # nudge eye slightly back toward the near end so more of the room is ahead
    eyeA = eye.copy(); eyeA[axis] = lo[axis] + 0.20 * size[axis]
    tgtA = center.copy(); tgtA[axis] = lo[axis] + 1.0 * size[axis]; tgtA[1] = y_eye
    eyeB = eye.copy(); eyeB[axis] = lo[axis] + 0.80 * size[axis]
    tgtB = center.copy(); tgtB[axis] = lo[axis] + 0.0 * size[axis]; tgtB[1] = y_eye
    return {
        "interiorA": (eyeA, tgtA, (0, 1, 0)),
        "interiorB": (eyeB, tgtB, (0, 1, 0)),
    }


def _top_view(scene):
    """Plan (top-down) view. Use for stage-free layouts (no ceiling to occlude).
    Up vector chosen so +Z points 'down' the image."""
    b = scene.bounds
    center = b.mean(axis=0)
    r = max(np.linalg.norm(b[1] - b[0]) / 2.0, 0.5)
    eye = center + np.array([0.0, 2.3 * r, 0.0])
    return {"top": (eye, center, (0, 0, -1))}


def render_scene(scene, out_prefix, view_dict, bg=(245, 245, 248, 255)):
    """Render a trimesh Scene. view_dict maps name -> (eye, target, up)."""
    written = []
    for name, (eye, target, up) in view_dict.items():
        try:
            _look_at(scene, eye, target, up=up)
            png = scene.save_image(resolution=RES, visible=True, background=bg)
            fn = f"{out_prefix}__{name}.png"
            with open(fn, "wb") as f:
                f.write(png)
            written.append(fn)
        except Exception as e:
            print(f"  !! failed view {name} for {out_prefix}: {e}")
            traceback.print_exc()
    return written


def load_as_scene(path):
    g = trimesh.load(path, force="scene")
    if isinstance(g, trimesh.Trimesh):
        g = g.scene()
    return g


# --------------------------------------------------------------------------
# CRL: individual GLB props / stages
# --------------------------------------------------------------------------
def crl_individuals():
    written = []
    # (name, path, kind) kind: "stage" -> interior views, "prop" -> exterior
    items = [
        ("crl_stage_garage_v1", f"{CRL}/stages/garage_v1.glb", "stage"),
        ("crl_stage_songjiang", f"{CRL}/stages/songjiang.glb", "stage"),
        ("crl_stage_frl_apartment", f"{CRL}/stages/frl_apartment_stage.glb", "stage"),
        ("crl_prop_gate", f"{CRL}/self_define_objects/gate.glb", "prop"),
        ("crl_prop_cylinder_0.1", f"{CRL}/self_define_objects/0.1_0.1_2cylinder.glb", "prop"),
    ]
    for name, path, kind in items:
        if not os.path.exists(path):
            print(f"  skip (missing) {path}")
            continue
        try:
            sc = load_as_scene(path)
            print(f"  {name}: {len(sc.geometry)} geom, bounds {sc.bounds.tolist()}")
            vd = _interior_views(sc) if kind == "stage" else _exterior_views(sc)
            written += render_scene(sc, os.path.join(OUT, "crl", name), vd)
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            traceback.print_exc()
    return written


# --------------------------------------------------------------------------
# CRL: assembled race tracks from scene_instance.json
# --------------------------------------------------------------------------
def _quat_wxyz_to_matrix(q):
    # trimesh expects [w, x, y, z]
    return trimesh.transformations.quaternion_matrix(np.asarray(q, float))


# resolve template_name -> render_asset GLB. Some object_config.json files
# contain a stale absolute path from the dataset author's machine, so we
# resolve to the known local files by basename.
def _resolve_asset(template_name):
    # stage templates come as "stages/<name>"
    tn = template_name
    cand = []
    if tn.startswith("stages/"):
        cand.append(f"{CRL}/{tn}.glb")
    # self-defined props
    cand.append(f"{CRL}/self_define_objects/{tn}.glb")
    cand.append(f"{CRL}/objects/{tn}.glb")
    cand.append(f"{CRL}/stages/{tn}.glb")
    for c in cand:
        if os.path.exists(c):
            return c
    return None


def crl_track(track_name):
    """Assemble stage + gates + obstacles for one track and render it."""
    f = f"{CRL}/configs/{track_name}/racing_1.scene_instance.json"
    if not os.path.exists(f):
        print(f"  skip track {track_name}: no {f}")
        return []
    d = json.load(open(f))
    full = trimesh.Scene()   # stage + gates + obstacles (for interior view)
    props = trimesh.Scene()  # gates + obstacles only (for top-down plan / 3q)

    # stage
    stage_tn = d.get("stage_instance", {}).get("template_name")
    stage_path = _resolve_asset(stage_tn) if stage_tn else None
    if stage_path:
        st = load_as_scene(stage_path)
        for k, geom in st.geometry.items():
            full.add_geometry(geom, node_name=f"stage_{k}",
                              transform=st.graph.get(k)[0] if k in st.graph else None)
    else:
        print(f"  track {track_name}: stage '{stage_tn}' not resolved")

    # objects (gates + obstacles)
    counts = {}
    for i, obj in enumerate(d.get("object_instances", [])):
        tn = obj["template_name"]
        path = _resolve_asset(tn)
        if not path:
            print(f"    unresolved object template: {tn}")
            continue
        T = np.eye(4)
        T[:3, 3] = obj.get("translation", [0, 0, 0])
        T = T @ _quat_wxyz_to_matrix(obj.get("rotation", [1, 0, 0, 0]))
        s = obj.get("uniform_scale", 1.0)
        if s != 1.0:
            S = np.eye(4); S[:3, :3] *= s
            T = T @ S
        og = load_as_scene(path)
        for k, geom in og.geometry.items():
            local = og.graph.get(k)[0] if k in og.graph else np.eye(4)
            node = f"obj{i}_{tn}_{k}"
            full.add_geometry(geom, node_name=node, transform=T @ local)
            props.add_geometry(geom.copy(), node_name=node, transform=T @ local)
        counts[tn] = counts.get(tn, 0) + 1
    print(f"  track {track_name}: stage={stage_tn} objs={counts}")

    written = []
    pfx = os.path.join(OUT, "crl", f"crl_track_{track_name}")
    if len(full.geometry):
        # interior corridor views (stage visible from inside)
        written += render_scene(full, pfx, _interior_views(full))
    if len(props.geometry):
        # course layout: gates+obstacles only, from top and 3/4 (no ceiling to occlude)
        vd = {}
        vd.update(_top_view(props))
        vd.update({"props3q": _exterior_views(props)["3q"]})
        written += render_scene(props, pfx + "_layout", vd)
    return written


# --------------------------------------------------------------------------
# aerial_gym: URDF analytic assets -> trimesh primitives
# --------------------------------------------------------------------------
import xml.etree.ElementTree as ET


def _origin_matrix(elem):
    T = np.eye(4)
    if elem is None:
        return T
    xyz = elem.get("xyz", "0 0 0").split()
    rpy = elem.get("rpy", "0 0 0").split()
    xyz = [float(v) for v in xyz]
    rpy = [float(v) for v in rpy]
    R = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    R[:3, 3] = xyz
    return R


def _geom_to_mesh(geom):
    box = geom.find("box")
    cyl = geom.find("cylinder")
    sph = geom.find("sphere")
    if box is not None:
        ext = [float(v) for v in box.get("size").split()]
        return trimesh.creation.box(extents=ext)
    if cyl is not None:
        r = float(cyl.get("radius"))
        h = float(cyl.get("length"))
        return trimesh.creation.cylinder(radius=r, height=h, sections=24)
    if sph is not None:
        return trimesh.creation.icosphere(radius=float(sph.get("radius")))
    return None


def urdf_to_scene(path):
    """Build a trimesh Scene from a URDF made of primitives, doing FK over
    fixed joints. Returns (scene, n_visuals)."""
    tree = ET.parse(path)
    root = tree.getroot()

    # parse links: name -> list of (visual_origin_matrix, mesh)
    link_vis = {}
    for link in root.findall("link"):
        name = link.get("name")
        vis = []
        for v in link.findall("visual"):
            m = _geom_to_mesh(v.find("geometry"))
            if m is None:
                continue
            vis.append((_origin_matrix(v.find("origin")), m))
        link_vis[name] = vis

    # parse joints -> parent/child + origin
    parent_of = {}
    joint_origin = {}
    children = {}
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        parent_of[c] = p
        joint_origin[c] = _origin_matrix(j.find("origin"))
        children.setdefault(p, []).append(c)

    # root links = those never a child
    all_links = set(link_vis.keys())
    roots = [l for l in all_links if l not in parent_of]

    # FK: world transform per link
    world = {}

    def compute(link, acc):
        world[link] = acc
        for ch in children.get(link, []):
            compute(ch, acc @ joint_origin[ch])

    for r in roots:
        compute(r, np.eye(4))
    # any links not reached (disconnected) -> identity
    for l in all_links:
        if l not in world:
            world[l] = np.eye(4)

    scene = trimesh.Scene()
    n = 0
    # a pleasant brown-ish for wood-like assets, gray otherwise
    for link, vis in link_vis.items():
        for vi, (vo, mesh) in enumerate(vis):
            mesh = mesh.copy()
            mesh.visual.face_colors = [120, 90, 60, 255]
            scene.add_geometry(mesh, node_name=f"{link}_{vi}",
                               transform=world[link] @ vo)
            n += 1
    return scene, n


def aerial_assets():
    written = []
    items = [
        ("aerial_tree_0", f"{AG}/trees/tree_0.urdf"),
        ("aerial_tree_5", f"{AG}/trees/tree_5.urdf"),
        ("aerial_tree_12", f"{AG}/trees/tree_12.urdf"),
        ("aerial_panel", f"{AG}/panels/panel.urdf"),
        ("aerial_object_cuboidal_rod", f"{AG}/objects/cuboidal_rod.urdf"),
        ("aerial_object_small_cube", f"{AG}/objects/small_cube.urdf"),
        ("aerial_object_1x1_wall", f"{AG}/objects/1_x_1_wall.urdf"),
        ("aerial_wall_front", f"{AG}/walls/front_wall.urdf"),
    ]
    for name, path in items:
        if not os.path.exists(path):
            print(f"  skip (missing) {path}")
            continue
        try:
            sc, n = urdf_to_scene(path)
            print(f"  {name}: {n} primitive visuals, bounds {sc.bounds.tolist()}")
            if n == 0:
                continue
            written += render_scene(sc, os.path.join(OUT, "aerial_gym_assets", name),
                                    _exterior_views(sc))
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            traceback.print_exc()
    return written


def main():
    os.makedirs(os.path.join(OUT, "crl"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "aerial_gym_assets"), exist_ok=True)
    allw = []
    print("=== CRL individual GLBs ===")
    allw += crl_individuals()
    print("=== CRL assembled tracks ===")
    for t in ["demo1_straight_ob1", "demo3_U"]:
        allw += crl_track(t)
    print("=== aerial_gym URDF assets ===")
    allw += aerial_assets()
    print(f"\nTOTAL rendered PNGs: {len(allw)}")
    for f in allw:
        print("  ", f)


if __name__ == "__main__":
    main()
