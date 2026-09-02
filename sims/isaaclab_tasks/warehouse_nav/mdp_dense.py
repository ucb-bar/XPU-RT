# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training-side obstacle mechanism for the ``dense_crossing`` scenario.

A large ``RigidObjectCollection`` pool of kinematic, real-collider clutter (crates, boxes,
pallets, poles, barrels, shelf blocks) that ``reset_dense_field`` teleports into a PACKED,
non-clipping grid-jitter layout across the south loading hall each reset — with a fraction
STACKED vertically. Mirrors the pattern of ``mdp_obstacles.py`` (kinematic colliders,
curriculum-gated active count, dump the rest at z=DUMP_Z), so it plugs into the same
``obstacle_count_curriculum`` and the ``nearest_obstacles_b`` / proximity reward terms.

Non-clipping is guaranteed BY CONSTRUCTION: each active base obstacle is assigned a distinct
grid cell (cell size > footprint), and jitter is bounded to keep it inside its cell.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import ManagerTermBase

from sims.isaaclab_tasks.warehouse_nav import scenario_dense as SD

DUMP_Z = -1000.0
N_DENSE = 96          # pool size (PACKED). Scale up for a denser field / down to save memory.

# Pool menu (cycled to build the pool). Each entry: kind + collider primitive spec.
#   "cuboid": size (x,y,z);  "cylinder": (radius, height)
_MENU = [
    {"kind": "crate",  "shape": "cuboid",   "size": (0.42, 0.42, 0.24), "color": (0.6, 0.42, 0.2)},
    {"kind": "box",    "shape": "cuboid",   "size": (0.5, 0.5, 0.5),    "color": (0.72, 0.6, 0.35)},
    {"kind": "pallet", "shape": "cuboid",   "size": (1.0, 1.2, 0.21),   "color": (0.5, 0.35, 0.18)},
    {"kind": "pole",   "shape": "cylinder", "size": (0.13, 2.0),         "color": (0.75, 0.22, 0.15)},
    {"kind": "box",    "shape": "cuboid",   "size": (0.5, 0.5, 0.5),    "color": (0.72, 0.6, 0.35)},
    {"kind": "barrel", "shape": "cylinder", "size": (0.3, 0.9),          "color": (0.15, 0.4, 0.6)},
    {"kind": "crate",  "shape": "cuboid",   "size": (0.42, 0.42, 0.24), "color": (0.6, 0.42, 0.2)},
    {"kind": "shelf",  "shape": "cuboid",   "size": (0.9, 0.5, 2.0),    "color": (0.32, 0.34, 0.4)},
]

_STACKABLE_TOP = ("crate", "box")   # kinds allowed to be raised onto another obstacle


def _entry(i):
    return _MENU[i % len(_MENU)]


def _height(i):
    e = _entry(i)
    return e["size"][2] if e["shape"] == "cuboid" else e["size"][1]


def _half_footprint(i):
    e = _entry(i)
    if e["shape"] == "cuboid":
        return 0.5 * max(e["size"][0], e["size"][1])
    return e["size"][0]  # cylinder radius


def make_dense_obstacle_collection_cfg(n: int = N_DENSE) -> RigidObjectCollectionCfg:
    """Build the pool of `n` kinematic, real-collider obstacles (dumped until a reset places them)."""
    objs: dict[str, RigidObjectCfg] = {}
    for i in range(n):
        e = _entry(i)
        rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        coll = sim_utils.CollisionPropertiesCfg()
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=e["color"])
        if e["shape"] == "cuboid":
            spawn = sim_utils.CuboidCfg(size=e["size"], rigid_props=rigid, collision_props=coll,
                                        visual_material=mat, semantic_tags=[("class", e["kind"])])
        else:
            spawn = sim_utils.CylinderCfg(radius=e["size"][0], height=e["size"][1], rigid_props=rigid,
                                          collision_props=coll, visual_material=mat,
                                          semantic_tags=[("class", e["kind"])])
        objs[f"dense_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Dense_{i}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DUMP_Z)))
    return RigidObjectCollectionCfg(rigid_objects=objs)


class reset_dense_field(ManagerTermBase):
    """reset: place ``k`` (curriculum) obstacles on a jittered grid across the hall, stack a
    fraction, dump the rest. ``k`` comes from ``env.obstacle_active_count`` (shared with
    ``mdp_obstacles.obstacle_count_curriculum``)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cell = cfg.params.get("cell", 1.7)
        self.stack_prob = cfg.params.get("stack_prob", 0.4)
        self.margin = cfg.params.get("margin", 0.15)
        self.n = len(env.scene["obstacles"].object_names)
        if not hasattr(env, "obstacle_active_count"):
            env.obstacle_active_count = torch.full((env.num_envs,), self.n, dtype=torch.long,
                                                   device=env.device)
        dev = env.device
        self.heights = torch.tensor([_height(i) for i in range(self.n)], device=dev)
        self.halffoot = torch.tensor([_half_footprint(i) for i in range(self.n)], device=dev)
        self.is_top = torch.tensor([1.0 if _entry(i)["kind"] in _STACKABLE_TOP else 0.0
                                    for i in range(self.n)], device=dev)
        # precompute grid cell centres over the field footprint (env-local)
        xlo, xhi, ylo, yhi = SD.DENSE_CROSSING["field"]
        xs = torch.arange(xlo + self.cell / 2, xhi, self.cell, device=dev)
        ys = torch.arange(ylo + self.cell / 2, yhi, self.cell, device=dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        self.cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)   # (C,2)
        self.n_cells = self.cells.shape[0]

    def __call__(self, env, env_ids, cell=1.7, stack_prob=0.4, margin=0.15):
        coll = env.scene["obstacles"]
        dev = env.device
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=dev)
        env_ids = env_ids.to(dev).long()
        n = len(env_ids)
        if n == 0:
            return
        origins = env.scene.env_origins[env_ids]                       # (n,3)
        k = env.obstacle_active_count[env_ids].clamp(min=0, max=self.n)  # (n,)

        pose = torch.zeros(n, self.n, 7, device=dev)
        pose[..., 3] = 1.0

        # assign each object a distinct grid cell (per-env random permutation)
        perm = torch.argsort(torch.rand(n, self.n_cells, device=dev), dim=1)[:, :self.n]  # (n,N)
        cell_xy = self.cells[perm]                                     # (n,N,2)
        # bounded jitter so the obstacle stays inside its cell (no clipping between cells)
        room = (self.cell / 2 - self.halffoot - self.margin).clamp_min(0.0)  # (N,)
        jit = (torch.rand(n, self.n, 2, device=dev) - 0.5) * 2 * room[None, :, None]
        xy = cell_xy + jit                                             # (n,N,2)

        pose[..., 0:2] = origins[:, None, :2] + xy
        pose[..., 2] = origins[:, None, 2] + self.heights[None, :] / 2  # centred primitive on floor

        # keep the spawn + goal points clear so the drone never spawns/arrives inside clutter
        sx, sy, _ = SD.DENSE_CROSSING["start"]
        gx, gy, _ = SD.DENSE_CROSSING["goal"]
        clear = self.cell + 0.6
        near_start = (xy[..., 0] - sx) ** 2 + (xy[..., 1] - sy) ** 2 < clear ** 2
        near_goal = (xy[..., 0] - gx) ** 2 + (xy[..., 1] - gy) ** 2 < clear ** 2

        idx = torch.arange(self.n, device=dev)
        active = (idx[None, :] < k[:, None]) & (~near_start) & (~near_goal)  # (n,N) object i placed
        base_active = torch.cat([active[:, :1], active[:, :-1]], dim=1)  # object i-1 is placed

        # stacking: raise a fraction of topper-kind objects onto the PREVIOUS object (same cell xy);
        # only when BOTH the topper and its base are active (so nothing floats over a dumped base).
        can_stack = (self.is_top[None, :] > 0) & (idx[None, :] > 0)
        can_stack = can_stack & (torch.rand(n, self.n, device=dev) < stack_prob)
        # base must be at least as wide as the topper (no overhang) and itself active
        base_wide = self.halffoot[None, :] >= self.halffoot.roll(1)[None, :] - 1e-3
        do = can_stack & base_wide & active & base_active
        # copy base xy + sit on base top: base_center_z + base_h/2 + top_h/2
        base_xy = torch.cat([xy[:, :1, :], xy[:, :-1, :]], dim=1)      # xy of object i-1
        base_h = self.heights.roll(1)[None, :]
        stacked_xy = torch.where(do[..., None], base_xy, xy)
        stacked_z = origins[:, None, 2] + base_h + self.heights[None, :] / 2
        pose[..., 0:2] = origins[:, None, :2] + stacked_xy
        pose[..., 2] = torch.where(do, stacked_z, pose[..., 2])

        # dump inactive objects (index >= k)
        pose[..., 2] = torch.where(active, pose[..., 2], torch.full_like(pose[..., 2], DUMP_Z))

        coll.write_object_pose_to_sim(pose, env_ids=env_ids, object_ids=torch.arange(self.n, device=dev))


def update_goal_success(env, command_name: str = "goal", radius: float = 1.2) -> torch.Tensor:
    """Success metric for the crossing (0 reward): reached the goal without crashing this episode.
    Stashes ``env._ep_success`` / ``env._ep_crash`` so ``obstacle_count_curriculum`` can ramp."""
    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term(command_name).goal_pos_w
    dist = torch.norm(goal_w - robot.data.root_pos_w, dim=1)
    truncated = env.termination_manager.time_outs
    crashed = env.termination_manager.terminated & (~truncated)
    reached = dist < radius
    if not hasattr(env, "_dense_reached"):
        env._dense_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._dense_reached = env._dense_reached | reached
    env._ep_success = env._dense_reached & (~crashed)
    env._ep_crash = crashed
    return env._ep_success.float() * 0.0


def reset_reached_flag(env, env_ids=None):
    """reset event: clear the per-episode goal-reached latch used by ``update_goal_success``."""
    if not hasattr(env, "_dense_reached"):
        env._dense_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if env_ids is None:
        env._dense_reached[:] = False
    else:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=env.device)
        env._dense_reached[env_ids] = False
