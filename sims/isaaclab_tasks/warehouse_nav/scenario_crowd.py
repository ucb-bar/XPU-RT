# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crowd-navigation scenario: a dense CROWD of walking people (person-tagged capsules) that
MUTUALLY AVOID each other while the drone crosses the open loading hall at a FIXED altitude.

This extends the ``mdp_obstacles`` mechanism (N capsule "person" RigidObjects in a
``RigidObjectCollection`` moved per-step by writing ``write_object_pose_to_sim``). The two
differences from ``mdp_obstacles.move_people``:

  * MANY more people (``N_CROWD`` ~ 15-30) walking in DIFFERENT directions across an open zone,
  * a per-step SOCIAL-FORCE / velocity-obstacle-lite update (``crowd_step``) so people steer
    around neighbours within a radius, reflect at the zone bounds, and keep a hard MIN_SEP so
    they never clip into each other. ``move_people`` uses an analytic triangle-wave patrol which
    cannot avoid collisions between walkers; the crowd needs real mutual avoidance.

Also here (kept with the scenario it belongs to):
  * ``PlanarVelocityCommandAction`` — the planar (horizontal-only) drone control constraint,
  * altitude-hold + people-proximity reward terms and a crowd navigation reward,
  * ``reached_goal`` termination + a success metric.

The capsules carry real colliders + the ("class","person") semantic tag, so drone-vs-person
contact triggers the collision termination and the people show up in detection ground truth —
identical to ``mdp_obstacles``' people. Animated limbs are a separate follow-up (Isaac People
characters are skinned only by the omni.anim.graph runtime; see ``preview_crowd.py``).
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import ManagerTermBase

from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import (
    VelocityCommandAction,
    VelocityCommandActionCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    normalize,
    yaw_quat,
    quat_apply,
)

# ------------------------------------------------------------------------------------------
# Crowd geometry + dynamics constants (warehouse-local metres). The open loading hall sits
# south of / near the origin (measured: y ~ 8 down to ~ -41, x ~ [-10, 5], clear floor,
# ceiling z ~ 9.3). We use a crossable sub-corridor of it as the crowd zone.
# ------------------------------------------------------------------------------------------
N_CROWD = 20                 # people in the crowd (configurable 15-30; see make_crowd_collection_cfg)
PERSON_H = 1.7               # capsule person height (m); matches mdp_obstacles.PERSON_H
PERSON_R = 0.28              # capsule radius
PERSON_Z = PERSON_H / 2.0    # capsule centre height (feet on the floor)

# social-force / VO-lite tuning
NEIGHBOR_R = 2.2             # a person only reacts to neighbours within this radius (m)
REP_GAIN = 1.4               # repulsion strength (soft steering away from neighbours)
MIN_SEP = 0.70              # HARD minimum centre-to-centre separation (m); ~2*PERSON_R + margin
SEP_ITERS = 4               # Jacobi de-overlap iterations per step (guarantees non-clipping)
SPEED_CAP = 1.6             # avoidance may push a person up to this * its preferred speed
ARRIVE_R = 0.9              # within this of its goal, a person picks a new goal (keeps walking)
WALK_SPEED_RANGE = (0.6, 1.3)  # per-person preferred walking speed (m/s)

# default crowd zone (env-local metres) + the fixed cruise altitude for the drone.
CROWD_ZONE = {"x": (-9.0, 4.0), "y": (-24.0, 4.0)}
Z_TARGET = 1.5


# ==========================================================================================
# 1) Crowd collection + reset + per-step mutual-avoidance update
# ==========================================================================================
def make_crowd_collection_cfg(n: int = N_CROWD) -> RigidObjectCollectionCfg:
    """A big person-capsule ``RigidObjectCollection`` (kinematic, person-tagged), one per walker.

    Colours are cycled so a dense crowd reads as many distinct individuals in a preview."""
    palette = [
        (0.20, 0.35, 0.70), (0.70, 0.30, 0.25), (0.25, 0.55, 0.30), (0.65, 0.55, 0.20),
        (0.45, 0.30, 0.60), (0.25, 0.55, 0.60), (0.60, 0.40, 0.30), (0.35, 0.35, 0.35),
    ]
    objs: dict[str, RigidObjectCfg] = {}
    for i in range(n):
        spawn = sim_utils.CapsuleCfg(
            radius=PERSON_R, height=PERSON_H - 2 * PERSON_R,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=palette[i % len(palette)]),
            semantic_tags=[("class", "person")])
        objs[f"person_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Crowd_{i}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PERSON_Z)))
    return RigidObjectCollectionCfg(rigid_objects=objs)


def _separate(pos: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
              min_sep: float = MIN_SEP, iters: int = SEP_ITERS) -> torch.Tensor:
    """Hard positional de-overlap (Jacobi): push every pair closer than ``min_sep`` apart by
    half the overlap each, then clamp back inside the zone. Batched over the leading dim.
    ``pos`` is (B, C, 2) env-local xy; ``lo``/``hi`` are (2,)."""
    C = pos.shape[1]
    eye = torch.eye(C, device=pos.device, dtype=torch.bool)
    for _ in range(iters):
        diff = pos[:, :, None, :] - pos[:, None, :, :]          # (B,C,C,2)  i - j
        d = diff.norm(dim=-1).masked_fill(eye, 1e6)             # (B,C,C)
        overlap = (min_sep - d).clamp_min(0.0)                  # >0 only when too close
        n = diff / d.clamp_min(1e-6)[..., None]                 # unit push dir i<-j
        push = (n * (0.5 * overlap)[..., None]).sum(dim=2)      # (B,C,2)
        pos = pos + push
        pos = torch.max(torch.min(pos, hi[None, None, :]), lo[None, None, :])
    return pos


def crowd_step(pos: torch.Tensor, goal: torch.Tensor, speed: torch.Tensor,
               lo: torch.Tensor, hi: torch.Tensor, dt: float):
    """One social-force / velocity-obstacle-lite crowd update. PURE torch, batched over the
    leading dim so both the env (B=num_envs) and the kinematic preview (B=1) call the SAME code.

    Args (all env-local, planar):
      pos   (B,C,2)  current person positions
      goal  (B,C,2)  current per-person goals
      speed (B,C)    per-person preferred walking speed
      lo,hi (2,)     zone bounds
      dt             timestep (s)

    Returns (pos, goal, vel):
      pos   (B,C,2)  new positions  (non-overlapping, inside the zone)
      goal  (B,C,2)  goals (resampled for anyone who arrived)
      vel   (B,C,2)  the velocity actually taken this step (for facing)
    """
    B, C, _ = pos.shape
    dev = pos.device
    eye = torch.eye(C, device=dev, dtype=torch.bool)

    # 1) preferred velocity: straight at the goal, at the preferred speed
    to_goal = goal - pos
    dg = to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v_pref = to_goal / dg * speed[..., None]

    # 2) neighbour repulsion: soft push away from everyone within NEIGHBOR_R (linear falloff)
    diff = pos[:, :, None, :] - pos[:, None, :, :]              # (B,C,C,2) i - j
    d = diff.norm(dim=-1).masked_fill(eye, 1e6)                 # (B,C,C)
    rep_dir = diff / d.clamp_min(1e-6)[..., None]
    strength = (NEIGHBOR_R - d).clamp_min(0.0)                  # >0 only within the radius
    rep = (rep_dir * strength[..., None]).sum(dim=2)           # (B,C,2)
    v = v_pref + REP_GAIN * rep

    # cap the resulting speed (avoidance can add to the preferred velocity)
    sp = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vmax = (speed * SPEED_CAP)[..., None]
    v = v * torch.minimum(torch.ones_like(sp), vmax / sp)

    # 3) integrate
    pos = pos + v * dt

    # 4) HARD de-overlap so people never clip into each other, then clamp inside the zone
    pos = _separate(pos, lo, hi)

    # 5) arrival -> pick a new goal so walkers keep crossing in varied directions
    arrived = (goal - pos).norm(dim=-1) < ARRIVE_R             # (B,C)
    if bool(arrived.any()):
        new_goal = lo + (hi - lo) * torch.rand(B, C, 2, device=dev)
        goal = torch.where(arrived[..., None], new_goal, goal)

    return pos, goal, v


class reset_crowd(ManagerTermBase):
    """reset event: (re)spawn the crowd with random start positions (de-overlapped), a goal on
    roughly the OPPOSITE side of the zone (so people cross the hall in many directions), and a
    random preferred walking speed. State lives on ``env`` and is advanced by ``move_crowd``."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.zone = cfg.params.get("zone", CROWD_ZONE)
        dev = env.device
        lo = torch.tensor([self.zone["x"][0], self.zone["y"][0]], device=dev)
        hi = torch.tensor([self.zone["x"][1], self.zone["y"][1]], device=dev)
        env.crowd_lo = lo
        env.crowd_hi = hi
        env.crowd_n = N_CROWD
        env.crowd_pos = torch.zeros(env.num_envs, N_CROWD, 2, device=dev)
        env.crowd_goal = torch.zeros(env.num_envs, N_CROWD, 2, device=dev)
        env.crowd_speed = torch.zeros(env.num_envs, N_CROWD, device=dev)

    def __call__(self, env, env_ids, zone=None, speed_range=WALK_SPEED_RANGE):
        dev = env.device
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=dev)
        env_ids = env_ids.to(dev).long()
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = env.crowd_lo, env.crowd_hi
        centre = 0.5 * (lo + hi)

        # random start positions, de-overlapped up front
        pos = lo + (hi - lo) * torch.rand(n, N_CROWD, 2, device=dev)
        pos = _separate(pos, lo, hi, iters=12)
        # goal ~ reflection of the start through the zone centre (+ jitter) => everyone crosses
        goal = (2 * centre[None, None, :] - pos) + (torch.rand(n, N_CROWD, 2, device=dev) - 0.5) * 3.0
        goal = torch.max(torch.min(goal, hi[None, None, :]), lo[None, None, :])
        lo_s, hi_s = speed_range
        speed = lo_s + (hi_s - lo_s) * torch.rand(n, N_CROWD, device=dev)

        env.crowd_pos[env_ids] = pos
        env.crowd_goal[env_ids] = goal
        env.crowd_speed[env_ids] = speed

        # write initial world pose (env-local -> world by adding the env origin)
        coll = env.scene["crowd"]
        origins = env.scene.env_origins[env_ids]
        pose = torch.zeros(n, N_CROWD, 7, device=dev)
        pose[..., 0:2] = pos + origins[:, None, :2]
        pose[..., 2] = origins[:, None, 2] + PERSON_Z
        pose[..., 3] = 1.0
        coll.write_object_pose_to_sim(pose, env_ids=env_ids,
                                      object_ids=torch.arange(N_CROWD, device=dev))


def move_crowd(env, env_ids=None):
    """interval event (every step): advance the crowd one social-force step and write the new
    poses. People steer around neighbours, keep a hard MIN_SEP (non-clipping), reflect at the
    zone bounds, and re-goal on arrival — so the crowd keeps walking in many directions."""
    if not hasattr(env, "crowd_pos"):
        return
    coll = env.scene["crowd"]
    dev = env.device
    pos, goal, vel = crowd_step(env.crowd_pos, env.crowd_goal, env.crowd_speed,
                                env.crowd_lo, env.crowd_hi, env.step_dt)
    env.crowd_pos = pos
    env.crowd_goal = goal

    C = pos.shape[1]
    yaw = torch.atan2(vel[..., 1], vel[..., 0])                 # face travel direction
    origins = env.scene.env_origins
    pose = torch.zeros(env.num_envs, C, 7, device=dev)
    pose[..., 0:2] = pos + origins[:, None, :2]                 # env-local -> world
    pose[..., 2] = origins[:, None, 2] + PERSON_Z
    pose[..., 3] = torch.cos(yaw / 2)                           # quat wxyz, yaw about +z
    pose[..., 6] = torch.sin(yaw / 2)
    coll.write_object_pose_to_sim(pose, object_ids=torch.arange(C, device=dev))


# ==========================================================================================
# 2) PLANAR drone control constraint (horizontal-only velocity setpoint)
# ==========================================================================================
class PlanarVelocityCommandAction(VelocityCommandAction):
    """Horizontal-only velocity command: the policy's 4-vector in [-1,1] maps to a PLANAR
    velocity setpoint ``[vx (fwd/back), vy (lateral), vz=0, yawrate]`` in the yaw ("vehicle")
    frame. vz is HARD-zeroed here (channel 1 is repurposed as lateral rather than climb), so
    the geometric velocity controller only ever damps vertical velocity to 0 and
    gravity-compensates => the drone station-keeps at whatever altitude it holds (its spawn
    z ~ Z_TARGET). This is how we enforce "no up/down"; an ``altitude_hold_penalty`` reward
    keeps z pinned to the target against any residual drift.

    NB: the base ``VelocityCommandAction`` is forward-only and never fills the lateral channel,
    so we override BOTH ``process_actions`` (to build the planar setpoint) and ``apply_actions``
    (to feed the lateral component into the controller). The controller law itself is unchanged.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        ms = self.cfg.max_speed
        self._vel_sp[:, 0] = a[:, 0] * ms                      # forward/back (yaw frame +x)
        self._vel_sp[:, 1] = a[:, 1] * ms                      # lateral / sideways (yaw frame +y)
        self._vel_sp[:, 2] = 0.0                               # PLANAR: no vertical velocity
        self._vel_sp[:, 3] = a[:, 2] * self.cfg.max_yawrate    # yaw rate (channel 2, aerial_gym)

    def apply_actions(self):
        # Identical geometric velocity controller as the parent, except the desired velocity is
        # built with a LATERAL component and ZERO vertical component (see process_actions).
        data = self._asset.data
        quat = data.root_quat_w
        R = matrix_from_quat(quat)
        v_w = data.root_lin_vel_w
        w_b = data.root_ang_vel_b

        yq = yaw_quat(quat)
        v_sp_yaw = torch.zeros_like(v_w)
        v_sp_yaw[:, 0] = self._vel_sp[:, 0]                    # forward/back in yaw frame
        v_sp_yaw[:, 1] = self._vel_sp[:, 1]                    # lateral in yaw frame  (<- planar)
        v_sp_yaw[:, 2] = 0.0                                   # no climb  (<- planar)
        v_des_w = quat_apply(yq, v_sp_yaw)

        v_err = v_des_w - v_w
        a_des = self._Kv * v_err - self._g_vec
        forces_w = self._mass * a_des

        body_z = R[:, :, 2]
        thrust = (forces_w * body_z).sum(dim=1).clamp(min=0.0)
        self._thrust[:, 0, 2] = thrust

        z_des = normalize(forces_w)
        from isaaclab.utils.math import euler_xyz_from_quat
        _, _, cy = euler_xyz_from_quat(quat)
        yaw_des = cy
        x_c = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=1)
        y_des = normalize(torch.cross(z_des, x_c, dim=1))
        x_des = torch.cross(y_des, z_des, dim=1)
        R_des = torch.stack([x_des, y_des, z_des], dim=2)

        errM = 0.5 * (torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des))
        e_R = torch.stack([errM[:, 2, 1], errM[:, 0, 2], errM[:, 1, 0]], dim=1)

        w_des = torch.zeros_like(w_b)
        w_des[:, 2] = self._vel_sp[:, 3]
        e_w = w_b - w_des

        moment = -self._KR * e_R - self._Kw * e_w
        moment = self._J.unsqueeze(0) * moment
        self._moment[:, 0, :] = moment

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )


@configclass
class PlanarVelocityCommandActionCfg(VelocityCommandActionCfg):
    class_type: type = PlanarVelocityCommandAction
    # inherits max_speed / max_yawrate / gains from VelocityCommandActionCfg.
    # (max_inclination is unused in the planar variant.)


# ==========================================================================================
# 3) Reward terms + termination/metric (env(...) -> tensor signature)
# ==========================================================================================
def _exp_reward(mag, exp, v):
    return mag * torch.exp(-(v ** 2) * exp)


def _exp_penalty(mag, exp, v):
    return mag * (torch.exp(-(v ** 2) * exp) - 1.0)  # in [-mag, 0]


def _local_z(env) -> torch.Tensor:
    return (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2]


def nearest_person(env):
    """(dist, unit_dir drone->person) to the nearest crowd member. Falls back to a large
    distance if the crowd collection is absent (keeps the reward finite)."""
    if "crowd" not in env.scene.rigid_object_collections:
        z = torch.zeros(env.num_envs, device=env.device)
        return torch.full((env.num_envs,), 1e3, device=env.device), torch.stack([z, z, z], dim=1)
    coll = env.scene["crowd"]
    obj = coll.data.object_pos_w                                # (N,C,3)
    drone = env.scene["robot"].data.root_pos_w[:, None, :]      # (N,1,3)
    rel = obj - drone
    d = torch.norm(rel, dim=-1)                                 # (N,C)
    dmin, idx = d.min(dim=1)
    rel_n = torch.gather(rel, 1, idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
    unit = rel_n / dmin.clamp_min(1e-6)[:, None]
    return dmin, unit


def nearest_person_dist(env) -> torch.Tensor:
    """(num_envs,) distance to the nearest person — the people-proximity signal for obs/rewards."""
    return nearest_person(env)[0]


def people_proximity_penalty(env, mag: float = 6.0, exp: float = 0.5) -> torch.Tensor:
    """Penalty for being close to the nearest person (wider/stronger the closer you are). Mirrors
    ``mdp_nav.navigation_reward``'s obstacle-proximity term but on the crowd collection."""
    d, _ = nearest_person(env)
    return -_exp_reward(mag, exp, d)


def altitude_hold_penalty(env, target_z: float = Z_TARGET) -> torch.Tensor:
    """Strong penalty on altitude error (enforces the PLANAR / fixed-height constraint in the
    reward, complementing the vz-zeroed action). Quadratic-in-error, bounded to [-1, 0]."""
    return _exp_penalty(1.0, 4.0, _local_z(env) - target_z)


def crowd_navigation_reward(env, command_name: str = "goal") -> torch.Tensor:
    """Cross-the-crowd navigation reward — VisFly-style shaping (learnable from scratch) toward
    a fixed goal on the far side, fused with a people-proximity penalty and an approach-speed
    penalty near people. Modelled on ``mdp_nav.navigation_reward``. The −100 collision override
    and the altitude-hold penalty are separate RewTerms in the env cfg.

      closing        = 2 * (v . goal_dir)          progress toward the goal (dominant signal)
      heading        = exp(-angle^2/0.5)           yaw-to-goal
      arrival        = 5 e^{-d^2/3.5} + 5 e^{-2 d^2}
      people_prox    = -6 e^{-0.5 d_person^2}      keep clear of people (see people_proximity_penalty)
      people_approach= -1.5 * near * (v . person_dir)_+   slow/veer near people
      smooth         = small action smoothness
    """
    robot = env.scene["robot"]
    term = env.command_manager.get_term(command_name)
    goal_w = term.goal_pos_w
    to_goal_w = goal_w - robot.data.root_pos_w
    dist = torch.norm(to_goal_w, dim=1)
    goal_dir_w = to_goal_w / to_goal_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
    v_w = robot.data.root_lin_vel_w

    closing = 2.0 * (v_w * goal_dir_w).sum(dim=1).clamp(-2.0, 2.0)

    cmdvec = env.command_manager.get_command(command_name)     # (N,4) [ux,uy,uz,dist] yaw-frame
    ang = torch.atan2(cmdvec[:, 1], cmdvec[:, 0])
    heading = 1.0 * torch.exp(-(ang ** 2) / 0.5)

    pos_reward = _exp_reward(5.0, 1.0 / 3.5, dist)
    very_close = _exp_reward(5.0, 2.0, dist)

    cmd = env.action_manager.get_term("velocity").processed_actions
    pcmd = getattr(env, "_prev_vel_cmd", None)
    if pcmd is None or pcmd.shape != cmd.shape:
        pcmd = cmd.clone()
    dcmd = cmd - pcmd
    env._prev_vel_cmd = cmd.detach().clone()
    smooth = _exp_penalty(0.2, 1.0, dcmd[:, 0]) + _exp_penalty(0.2, 1.0, dcmd[:, 1]) \
        + _exp_penalty(0.2, 1.0, dcmd[:, 3])

    d_p, p_dir = nearest_person(env)                            # p_dir points drone->person
    people_prox = -_exp_reward(6.0, 0.5, d_p)
    approach = (v_w * p_dir).sum(dim=1).clamp_min(0.0)
    near = (d_p < 1.5).float()
    people_approach = -1.5 * near * approach

    return closing + heading + pos_reward + very_close + smooth + people_prox + people_approach


def reached_goal(env, command_name: str = "goal", radius: float = 0.9) -> torch.Tensor:
    """(num_envs,) bool termination: drone is within ``radius`` of the goal (success)."""
    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term(command_name).goal_pos_w
    return torch.norm(goal_w - robot.data.root_pos_w, dim=1) < radius


def update_success_metric(env, command_name: str = "goal", radius: float = 0.9) -> torch.Tensor:
    """Success bookkeeping (0 reward): reached the far-side goal without crashing. Stashes
    ``env._ep_success`` / ``env._ep_crash`` (mirrors ``mdp_nav.update_success_metric``)."""
    truncated = env.termination_manager.time_outs
    crashed = env.termination_manager.terminated & (~truncated)
    env._ep_success = reached_goal(env, command_name, radius) & (~crashed)
    env._ep_crash = crashed
    return env._ep_success.float() * 0.0
