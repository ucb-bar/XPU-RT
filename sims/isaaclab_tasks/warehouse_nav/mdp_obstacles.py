# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle course for the TRAINABLE warehouse-nav env — REAL warehouse-prop meshes (the same
pallets / crates / box-stacks / cones / KLT bins as the circuit-library preview), placed by the
shared collision-aware NO-CLIP sampler (placement.sample_no_clip). This replaces the old abstract
poles + capsule-crates so the physics/training env matches the rich scene we iterate on visually
(Dima's "make the env look like the videos" + FPV-footage direction).

The props are a fixed POOL of kinematic rigid bodies (real colliders + semantic tags): each reset
draws a no-clip layout for the episode and snaps pool slots of the matching KIND onto it, dumping
the rest below the floor (z=DUMP_Z). TALL crate/box stacks are just several slots stacked in z
(kinematic, so they hold position). People stay capsules — the training-time stand-in for the
frozen-pose humans (realistic WALKING people are the IRA render layer, per Dima).

Curriculum scales obstacle DENSITY (0.25 -> 1.0) with success rate (aerial_gym-style ramp).
"""

from __future__ import annotations

import math
import os
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import circuits as C
from . import placement as P
from . import mdp_gates

N_HUMANS = 4
# per-kind real-prop pool (sized for a dense single aisle incl. stacks; curriculum activates a
# subset via density). KIND_ORDER fixes the object-id layout used when writing poses.
KIND_ORDER = ["pallet", "crate", "box", "cone", "klt"]
# TRAINING pool: moderate, curriculum thins it further (heavy at high num_envs).
PROP_POOL = {"pallet": 12, "crate": 34, "box": 34, "cone": 14, "klt": 14}
# SHOWCASE/FPV pool: PACKED clutter for the single-env render (~490 bodies — as crowded as the
# kinematic preview, incl. obstacles in the drone's path). Showcase props are non-colliding so the
# drone flies THROUGH the crowd (like the preview); the trainable env uses PROP_POOL + colliders.
PROP_POOL_FULL = {"pallet": 34, "crate": 220, "box": 90, "cone": 18, "klt": 18}
N_STATIC = sum(PROP_POOL.values())          # total static prop slots (curriculum max_level)
N_FORKLIFT = 1                               # one driven forklift (moving obstacle, circuit lanes)
POOL = N_HUMANS + N_STATIC + N_FORKLIFT

# global object-id base for each kind's slots (people occupy [0, N_HUMANS); forklift is last)
SLOT_BASE = {}
_idx = N_HUMANS
for _k in KIND_ORDER:
    SLOT_BASE[_k] = _idx
    _idx += PROP_POOL[_k]
FORK_IDX = N_HUMANS + N_STATIC               # object-id of the forklift slot

DUMP_Z = -1000.0
PERSON_H = 1.7

# baked kinematic-rigid-body wrapper USDs (sims/scripts/bake_prop_rigidbodies.py) — the raw
# Simple_Warehouse prop meshes have no physics schemas, so UsdFileCfg.rigid_props no-ops on them.
_RB_PROP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "out", "rb_props")
_RB_PROP_DIR = os.path.abspath(_RB_PROP_DIR)
# The crowded-course rack/box prop meshes are a generated asset, not tracked in
# this repo. Fall back to the collaborator's read-only checkout if the local
# generated dir is absent (the clean gate course doesn't spawn these).
if not os.path.isdir(_RB_PROP_DIR):
    _ag_props = "/scratch/agustin/projects/DIMA/XPU-RT/out/rb_props"
    if os.path.isdir(_ag_props):
        _RB_PROP_DIR = _ag_props


def _prop_spawn(kind: str, collide: bool = True) -> sim_utils.UsdFileCfg:
    """Real prop mesh (via the baked kinematic-rigid-body wrapper) with a semantic tag. Clutter
    props are scaled (P.PROP_SCALE) to match the placer's footprints. ``collide`` False disables
    the collider (SHOWCASE: the drone flies THROUGH the packed clutter like the kinematic preview;
    the prop is still a kinematic body so it's repositionable per episode)."""
    s = P.PROP_SCALE if kind in ("cone", "klt", "crate", "box", "pallet") else 1.0
    kw = dict(
        usd_path=os.path.join(_RB_PROP_DIR, f"{kind}.usd"),
        scale=(s, s, s),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        semantic_tags=[("class", kind)],
    )
    if not collide:
        kw["collision_props"] = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    return sim_utils.UsdFileCfg(**kw)


def make_obstacle_collection_cfg(pool: dict | None = None, collide: bool = True) -> RigidObjectCollectionCfg:
    """Build the obstacle collection. ``pool`` = per-kind slot counts (defaults to the moderate
    training PROP_POOL; pass PROP_POOL_FULL for the single-env FPV showcase). Object insertion
    order is people -> props(by KIND_ORDER) -> forklift; reset derives indices from object_names,
    so the pool size is free to differ between the training and showcase scenes."""
    pool = pool or PROP_POOL
    objs: dict[str, RigidObjectCfg] = {}
    cprops = sim_utils.CollisionPropertiesCfg(collision_enabled=collide)
    # people: capsules, person-tagged, always active + patrolling (frozen-pose stand-in)
    for i in range(N_HUMANS):
        spawn = sim_utils.CapsuleCfg(
            radius=0.28, height=PERSON_H - 0.56,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=cprops,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.35, 0.7)),
            semantic_tags=[("class", "person")])
        objs[f"person_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Person_{i}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DUMP_Z)))
    # real props, grouped by kind in KIND_ORDER
    for kind in KIND_ORDER:
        for j in range(pool[kind]):
            objs[f"{kind}_{j}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Prop_{kind}_{j}", spawn=_prop_spawn(kind, collide),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DUMP_Z)))
    # the driven forklift (moving obstacle) — LAST object-id (FORK_IDX)
    objs["forklift_0"] = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Prop_forklift_0", spawn=_prop_spawn("forklift", collide),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DUMP_Z)))
    return RigidObjectCollectionCfg(rigid_objects=objs)


def _yaw_quat_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class reset_obstacle_field(ManagerTermBase):
    """reset: draw a per-env no-clip layout of REAL props (density from the curriculum) in the
    passed ``volume`` and snap pool slots onto it; place patrolling people; dump unused slots."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.volume = cfg.params.get("volume")
        self.half_density_prob = cfg.params.get("half_density_prob", 0.15)
        self.speed = cfg.params.get("walk_speed", 0.8)
        # optional: place obstacles across a named CIRCUIT's full regions/route (proper-B), instead
        # of the single aisle `volume`. When set, C.sample_obstacles owns start/goal/gate/lane clearance.
        self.circuit = cfg.params.get("circuit", None)
        self._circ = C.CIRCUITS_BY_NAME[self.circuit] if self.circuit else None
        if self._circ is not None:
            wps = self._circ["waypoints"]
            xs = [p[0] for p in wps]; ys = [p[1] for p in wps]
            self._people_box = (min(xs) - 2.0, max(xs) + 2.0, min(ys) - 2.0, max(ys) + 2.0)
        if not hasattr(env, "obstacle_active_count"):
            env.obstacle_active_count = torch.full((env.num_envs,), 2, dtype=torch.long, device=env.device)
        env.person_base = torch.zeros(env.num_envs, N_HUMANS, 3, device=env.device)
        env.person_dir = torch.zeros(env.num_envs, N_HUMANS, 2, device=env.device)
        env.person_half = torch.zeros(env.num_envs, N_HUMANS, device=env.device)
        env.person_speed = torch.zeros(env.num_envs, N_HUMANS, device=env.device)
        # per-env forklift drive state (populated on reset for circuits with a moving forklift)
        if not hasattr(env, "_fork"):
            env._fork = {}
        # gate openings + spawn kept clear of props (opening radius a touch over half the gate)
        self._excl = [(gc[0], gc[1], 1.0) for gc, _ in mdp_gates.GATES]
        self._excl.append((self.volume["x"][0] * 0.5 + self.volume["x"][1] * 0.5, self.volume["y"][0] - 1.5, 1.2))
        self._idx = None  # object-id map, derived lazily from the actual collection pool

    def _ensure_idx(self, env):
        """Derive slot indices from the ACTUAL collection object names, so this works with either
        the moderate training pool or the big showcase pool (pose-writing is pool-size-agnostic)."""
        if self._idx is not None:
            return
        names = list(env.scene["obstacles"].object_names)
        name2i = {n: i for i, n in enumerate(names)}
        self._pool_n = len(names)
        self._person_ids = sorted(name2i[n] for n in names if n.startswith("person_"))
        self._slot = {k: sorted(name2i[n] for n in names if n.startswith(k + "_")) for k in KIND_ORDER}
        self._fork_idx = name2i.get("forklift_0", None)
        env._person_ids = torch.tensor(self._person_ids, device=env.device, dtype=torch.long)
        env._fork_idx = self._fork_idx

    def __call__(self, env, env_ids, volume=None, half_density_prob=0.15, walk_speed=0.8,
                 circuit=None, full_density=False, prop_density=None):
        coll = env.scene["obstacles"]
        dev = env.device
        self._ensure_idx(env)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=dev)
        env_ids = env_ids.to(dev).long()
        n = len(env_ids)
        if n == 0:
            return
        origins = env.scene.env_origins[env_ids]
        vol = self.volume
        if self._circ is not None:
            pb = self._people_box
            lo = torch.tensor([pb[0], pb[2], vol["z"][0]], device=dev)
            hi = torch.tensor([pb[1], pb[3], vol["z"][1]], device=dev)
        else:
            lo = torch.tensor([vol["x"][0], vol["y"][0], vol["z"][0]], device=dev)
            hi = torch.tensor([vol["x"][1], vol["y"][1], vol["z"][1]], device=dev)

        pose = torch.zeros(n, self._pool_n, 7, device=dev)
        pose[..., 2] = DUMP_Z          # everything dumped by default
        pose[..., 3] = 1.0             # identity quat
        person_ids = torch.tensor(self._person_ids, device=dev, dtype=torch.long)

        # --- people: base in the aisle, capsule centre at PERSON_H/2; straight N-S patrol ---
        pbase = origins[:, None, :] + lo + (hi - lo) * torch.rand(n, N_HUMANS, 3, device=dev)
        pbase[..., 2] = origins[:, None, 2] + PERSON_H / 2
        sign = torch.where(torch.rand(n, N_HUMANS, device=dev) < 0.5, -1.0, 1.0)
        ang = sign * (math.pi / 2) + (torch.rand(n, N_HUMANS, device=dev) - 0.5) * 0.6
        env.person_dir[env_ids] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        env.person_base[env_ids] = pbase
        env.person_half[env_ids] = 1.5 + 1.5 * torch.rand(n, N_HUMANS, device=dev)
        env.person_speed[env_ids] = self.speed * (0.7 + 0.6 * torch.rand(n, N_HUMANS, device=dev))
        pose[:, person_ids, :3] = pbase

        # --- props: per-env no-clip layout snapped to pool slots. full_density (showcase) = the
        # FULL preview layout; else curriculum-scaled (training starts easy and ramps up). ---
        frac = float(getattr(env, "curriculum_progress_fraction", 0.0))
        # ``prop_density`` (when not None) overrides the curriculum-scaled density — used by the
        # fused Stage-1 gate-following scene, which sets it to 0.0 so the aisle carries ONLY the
        # gate course + (short, fly-over) patrolling people. The tall random prop stacks (1.5-3 m,
        # reaching through the 2.0 m cruise altitude) are the Stage-2 ToF-avoidance task, not a
        # gate-follow obstacle. density<=0 skips prop placement entirely (all slots stay dumped).
        if prop_density is not None:
            density = float(prop_density)
        else:
            density = 1.0 if full_density else (0.25 + 0.75 * frac)
        region_box = (vol["x"][0] - 0.1, vol["x"][1] + 0.1, vol["y"][0], vol["y"][1])
        step = int(getattr(env, "common_step_counter", 0))
        for row, eid in enumerate(env_ids.tolist() if density > 0.0 else []):
            rng = random.Random(step * 100003 + eid * 131 + 7)
            if self._circ is not None:
                layout = C.sample_obstacles(self._circ, rng, density=density)
            else:
                # Crowded fused course (per user): FEWER but TALL, THIN, stacked-box obstacles —
                # box/crate/klt-heavy (drop wide pallets + small cones), high stack_prob + taller
                # stacks so most props reach through the 2.0 m cruise as slender towers the ToF must
                # weave around, rather than a dense low carpet. Count kept modest; density scales it.
                tall_thin = ["box", "crate", "box", "klt", "crate", "box"]
                reg = [{"box": region_box, "kinds": tall_thin,
                        "count": (int(N_STATIC * 0.25), int(N_STATIC * 0.5)),
                        "density": density, "stack_prob": 0.85, "stack_h": (2.0, 3.5)}]
                layout = P.sample_no_clip(reg, rng, exclusions=self._excl)
            used = {k: 0 for k in KIND_ORDER}
            oz = float(origins[row, 2].item())
            for it in layout:
                k = it["kind"]
                if k not in self._slot or used[k] >= len(self._slot[k]):
                    continue
                gidx = self._slot[k][used[k]]; used[k] += 1
                x, y, z = it["pos"]
                pose[row, gidx, 0] = origins[row, 0] + x
                pose[row, gidx, 1] = origins[row, 1] + y
                pose[row, gidx, 2] = oz + z
                w, qx, qy, qz = _yaw_quat_wxyz(float(it["yaw"]))
                pose[row, gidx, 3:7] = torch.tensor([w, qx, qy, qz], device=dev)

        # --- forklift: for circuits with a moving forklift, pick a lane, stage at its start, and
        # record the drive state so move_forklift() can drive it each step. Else it stays dumped. ---
        veh = self._circ["vehicles"][0] if (self._circ and self._circ.get("vehicles")) else None
        if self._fork_idx is not None:
            for row, eid in enumerate(env_ids.tolist()):
                env._fork.pop(eid, None)
                if not (veh and veh.get("moving") and veh.get("lanes")):
                    continue
                lanes = veh["lanes"]
                lane = lanes[(step + eid) % len(lanes)]
                pts = [(float(p[0]), float(p[1])) for p in lane]
                seg = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)]
                total = sum(seg)
                loop = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 3.0
                base = (float(origins[row, 0]), float(origins[row, 1]), float(origins[row, 2]))
                env._fork[eid] = {"pts": pts, "seg": seg, "total": total, "loop": loop, "base": base, "speed": 1.5}
                x0, y0 = pts[0]
                heading = math.atan2(pts[1][1] - y0, pts[1][0] - x0)
                pose[row, self._fork_idx, 0] = base[0] + x0
                pose[row, self._fork_idx, 1] = base[1] + y0
                pose[row, self._fork_idx, 2] = base[2]
                w, qx, qy, qz = _yaw_quat_wxyz(heading)
                pose[row, self._fork_idx, 3:7] = torch.tensor([w, qx, qy, qz], device=dev)

        coll.write_object_pose_to_sim(pose, env_ids=env_ids, object_ids=torch.arange(self._pool_n, device=dev))


def move_people(env, env_ids=None):
    """interval event (per step): patrol each person back-and-forth along its straight path at
    CONSTANT speed (triangle wave in [-half, +half]); face the current travel direction."""
    if not hasattr(env, "person_base"):
        return
    coll = env.scene["obstacles"]
    dev = env.device
    t = (env.episode_length_buf.float() * env.step_dt)[:, None]
    half = env.person_half.clamp_min(1e-3)
    speed = env.person_speed
    period = 4 * half / speed.clamp_min(1e-3)
    ph = (t % period) / period
    tri = torch.where(ph < 0.5, 4 * ph - 1, 3 - 4 * ph)
    s = tri * half
    pos = env.person_base.clone()
    pos[..., 0] += env.person_dir[..., 0] * s
    pos[..., 1] += env.person_dir[..., 1] * s
    pose = torch.zeros(env.num_envs, N_HUMANS, 7, device=dev)
    pose[..., :3] = pos
    pose[..., 3] = 1.0
    pids = getattr(env, "_person_ids", torch.arange(N_HUMANS, device=dev))
    coll.write_object_pose_to_sim(pose, object_ids=pids)


def _poly_at(pts, seg, total, u):
    """Point at arc-length fraction u in [0,1] along a polyline (constant speed, drives THROUGH
    corners — no stop at interior waypoints)."""
    d = max(0.0, min(1.0, u)) * total
    for k, L in enumerate(seg):
        if d <= L or k == len(seg) - 1:
            t = (d / L) if L > 1e-6 else 0.0
            return (pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t,
                    pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t)
        d -= L
    return pts[-1]


def move_forklift(env, env_ids=None):
    """interval event (per step): drive each active forklift along its lane — continuous LAPS for
    a closed lane, smooth there-and-back for an open lane — at constant speed, facing travel dir.
    (Per-env python loop; fine for the single-env FPV/preview. Vectorize for large-scale training.)"""
    fk = getattr(env, "_fork", None)
    fork_idx = getattr(env, "_fork_idx", None)
    if not fk or fork_idx is None:
        return
    coll = env.scene["obstacles"]
    dev = env.device
    pose = torch.zeros(env.num_envs, 1, 7, device=dev)
    pose[..., 2] = DUMP_Z
    pose[..., 3] = 1.0
    t = (env.episode_length_buf.float() * env.step_dt).tolist()
    for eid, d in fk.items():
        pts, seg, total, loop, base = d["pts"], d["seg"], d["total"], d["loop"], d["base"]
        s = (t[eid] * d["speed"]) / max(total, 1e-3)
        if loop:
            u = s % 1.0
            un = (s + 0.02) % 1.0
        else:
            m = s % 2.0
            u = m if m <= 1.0 else 2.0 - m         # triangle: forward then back
            un = min(u + 0.02, 1.0)
        x, y = _poly_at(pts, seg, total, u)
        xn, yn = _poly_at(pts, seg, total, un)
        heading = math.atan2(yn - y, xn - x)
        w, qx, qy, qz = _yaw_quat_wxyz(heading)
        pose[eid, 0, 0] = base[0] + x
        pose[eid, 0, 1] = base[1] + y
        pose[eid, 0, 2] = base[2]
        pose[eid, 0, 3:7] = torch.tensor([w, qx, qy, qz], device=dev)
    coll.write_object_pose_to_sim(pose, object_ids=torch.tensor([fork_idx], device=dev))


def randomize_dome_light(env, env_ids, intensity_range=(150.0, 650.0), prim_path="/World/DomeLight"):
    """reset event: jitter the supplementary dome-light intensity for lighting domain
    randomization (the warehouse USD ships its own ceiling lights; this varies overall
    brightness so a depth/vision policy doesn't overfit one lighting condition)."""
    from pxr import UsdLux
    pr = env.scene.stage.GetPrimAtPath(prim_path) if hasattr(env.scene, "stage") else env.sim.stage.GetPrimAtPath(prim_path)
    if pr is None or not pr.IsValid():
        return
    lo, hi = intensity_range
    val = float(lo + (hi - lo) * torch.rand(1).item())
    UsdLux.LightAPI(pr).GetIntensityAttr().Set(val)


class obstacle_count_curriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.min_level = p.get("min_level", 2); self.max_level = p.get("max_level", N_STATIC)
        self.check_after = p.get("check_after", 512)
        self.up = p.get("up", 0.7); self.down = p.get("down", 0.6)
        self.inc = p.get("inc", 2); self.dec = p.get("dec", 1)
        self.level = float(self.min_level); self._succ = 0; self._total = 0
        if not hasattr(env, "obstacle_active_count"):
            env.obstacle_active_count = torch.full((env.num_envs,), self.min_level,
                                                   dtype=torch.long, device=env.device)
        env.curriculum_progress_fraction = 0.0

    def __call__(self, env, env_ids, min_level=2, max_level=N_STATIC, check_after=512,
                 up=0.7, down=0.6, inc=2, dec=1):
        succ = getattr(env, "_ep_success", None)
        if succ is not None:
            self._succ += int(succ[env_ids].sum().item())
        self._total += len(env_ids)
        if self._total >= self.check_after:
            rate = self._succ / max(self._total, 1)
            if rate > self.up: self.level += self.inc
            elif rate < self.down: self.level -= self.dec
            self.level = float(min(max(self.level, self.min_level), self.max_level))
            self._succ = 0; self._total = 0
        env.obstacle_active_count[env_ids] = int(self.level)
        env.curriculum_progress_fraction = (self.level - self.min_level) / max(self.max_level - self.min_level, 1)
        return self.level
