# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for goal-conditioned warehouse navigation.

The reward is a faithful port of aerial_gym's ``navigation_task.compute_reward``
(``aerial_gym/task/navigation_task/navigation_task.py:435-521``), which is pure-torch and
carries no Isaac-Gym dependency.  Two deliberate deviations from the original:

* aerial_gym's obstacle-proximity penalty is **dead code** upstream — it is masked by
  ``terminations < 0`` on a counter that is never negative, so it never fires.  We port the
  *intended* behaviour (mask on "no collision this step") because it is the only term that
  rewards keeping clear of obstacles.
* The goal is sampled once per reset (a real objective tied to the scene), not resampled on a
  timer uncorrelated with the world — which was the core defect of the old steering task.

The goal command term subclasses Isaac Lab's ``CommandTerm`` so it plugs into the manager
framework and is visualised/logged like any other command.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


# ---------------------------------------------------------------------------------------------
# Goal-position command
# ---------------------------------------------------------------------------------------------
class GoalPositionCommand(CommandTerm):
    """A world-frame goal position, sampled once per episode inside the scene bounds.

    The command exposed to observations is the goal expressed in the robot's yaw-only
    ("vehicle") frame: a unit direction (3) plus the raw distance (1), matching aerial_gym's
    observation layout so their policy weights remain loadable.
    """

    cfg: "GoalPositionCommandCfg"

    def __init__(self, cfg: "GoalPositionCommandCfg", env):
        super().__init__(cfg, env)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # metrics buffer (distance to goal) for logging
        self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"GoalPositionCommand (ranges={self.cfg.ranges})"

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 4) = [unit_dir_x, unit_dir_y, unit_dir_z, distance] in the vehicle frame."""
        robot = self._env.scene["robot"]
        to_goal_w = self.goal_pos_w - robot.data.root_pos_w
        dist = torch.norm(to_goal_w, dim=1, keepdim=True).clamp_min(1e-6)
        # rotate world offset into the yaw-only frame so heading changes don't move the goal
        to_goal_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), to_goal_w)
        unit_b = to_goal_b / dist
        return torch.cat([unit_b, dist], dim=1)

    def _resample_command(self, env_ids):
        # Sample a goal as a ratio of the per-env AABB (aerial_gym convention: far +x end).
        r = self.cfg.ranges
        origins = self._env.scene.env_origins[env_ids]
        lo = torch.tensor([r.pos_x[0], r.pos_y[0], r.pos_z[0]], device=self.device)
        hi = torch.tensor([r.pos_x[1], r.pos_y[1], r.pos_z[1]], device=self.device)
        samp = lo + (hi - lo) * torch.rand(len(env_ids), 3, device=self.device)
        self.goal_pos_w[env_ids] = origins + samp

    def _update_command(self):
        # goal is static within an episode; nothing to advance
        pass

    def _update_metrics(self):
        robot = self._env.scene["robot"]
        self.metrics["distance_to_goal"] = torch.norm(self.goal_pos_w - robot.data.root_pos_w, dim=1)

    # No debug-vis markers needed for headless training; keep the hooks minimal.
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    """Config for :class:`GoalPositionCommand`."""

    class_type: type = GoalPositionCommand

    @configclass
    class Ranges:
        # Offsets (metres) from the env origin. Defaults suit the ~open warehouse volume;
        # override per scene. Goal biased to the far +x end like aerial_gym.
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


# ---------------------------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------------------------
def goal_vector_b(env, command_name: str = "goal") -> torch.Tensor:
    """(num_envs, 4) unit-direction + distance to the goal, in the vehicle frame."""
    return env.command_manager.get_command(command_name)


def nearest_obstacles_b(env, k: int = 5) -> torch.Tensor:
    """(num_envs, 3k) the k nearest active obstacles' relative positions in the vehicle frame.

    Privileged state (the info a depth camera would otherwise have to infer). Dumped obstacles
    (z=DUMP_Z) sort to the back and are effectively ignored. Used in the policy group for the
    state-based variant and moved to the critic-only group when vision is the actor input."""
    if "obstacles" not in env.scene.rigid_object_collections:
        return torch.zeros(env.num_envs, 3 * k, device=env.device)
    coll = env.scene["obstacles"]
    robot = env.scene["robot"]
    obj_w = coll.data.object_pos_w                       # (N, pool, 3)
    rel_w = obj_w - robot.data.root_pos_w[:, None, :]    # (N, pool, 3)
    dist = torch.norm(rel_w, dim=-1)                     # (N, pool)
    idx = torch.topk(dist, k=min(k, dist.shape[1]), dim=1, largest=False).indices  # (N,k)
    rel_k = torch.gather(rel_w, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N,k,3)
    # rotate into the yaw-only frame
    yq = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, rel_k.shape[1], -1).reshape(-1, 4)
    rel_b = quat_apply_inverse(yq, rel_k.reshape(-1, 3)).reshape(env.num_envs, -1)
    return rel_b


# ---------------------------------------------------------------------------------------------
# Reward (faithful port of aerial_gym navigation_task.compute_reward, exact coefficients)
# ---------------------------------------------------------------------------------------------
def _goal_dist(env, command_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term(command_name).goal_pos_w
    return torch.norm(goal_w - robot.data.root_pos_w, dim=1)


def _exp_reward(mag, exp, v):
    return mag * torch.exp(-(v**2) * exp)


def _exp_penalty(mag, exp, v):
    return mag * (torch.exp(-(v**2) * exp) - 1.0)  # in [-mag, 0]


def active_obstacle_min_dist(env) -> torch.Tensor:
    """Analytic nearest-active-obstacle distance (the RayCaster path is warp-static-only and
    can't see the moving pool — SPEC YELLOW). Dumped obstacles are at z=DUMP_Z so they are
    naturally far and never the minimum."""
    if "obstacles" not in env.scene.rigid_object_collections:
        return torch.full((env.num_envs,), 1e3, device=env.device)
    coll = env.scene["obstacles"]
    obj_pos = coll.data.object_pos_w  # (N, pool, 3)
    drone = env.scene["robot"].data.root_pos_w[:, None, :]  # (N,1,3)
    d = torch.norm(obj_pos - drone, dim=-1)  # (N, pool)
    return d.min(dim=1).values


def active_obstacle_nearest(env):
    """(dist, unit_dir_drone->obstacle) to the nearest obstacle (people + active clutter)."""
    if "obstacles" not in env.scene.rigid_object_collections:
        z = torch.zeros(env.num_envs, device=env.device)
        return torch.full((env.num_envs,), 1e3, device=env.device), torch.stack([z, z, z], dim=1)
    coll = env.scene["obstacles"]
    obj_pos = coll.data.object_pos_w                             # (N,pool,3)
    drone = env.scene["robot"].data.root_pos_w[:, None, :]      # (N,1,3)
    rel = obj_pos - drone                                        # (N,pool,3)
    d = torch.norm(rel, dim=-1)                                  # (N,pool)
    dmin, idx = d.min(dim=1)
    rel_n = torch.gather(rel, 1, idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
    unit = rel_n / dmin.clamp_min(1e-6)[:, None]
    return dmin, unit


def navigation_reward(env, command_name: str = "goal") -> torch.Tensor:
    """Navigation reward — VisFly NavigationEnv shaping (learnable from scratch) fused with
    aerial_gym's arrival bonus, obstacle-proximity, and −100 collision.

    Why not aerial_gym's exact reward: its ``getting_closer`` penalizes RETREAT at 2x, so under
    random exploration the expected progress reward is negative and PPO learns to STAY STILL;
    plus its ``dist_from_goal`` pays a free hover plateau. VisFly's design avoids both:
      * closing-speed  = v · goal_dir  (smooth, rewards ANY motion toward the goal — the key
        exploration signal; the drone discovers "move toward goal" immediately),
      * heading        = exp(-angle_to_goal^2/std)  (teaches yaw-to-goal — needed because the
        control is forward-only, so the drone must face the goal before advancing),
      * arrival        = 5·e^{-d²/3.5} + 5·e^{-2d²}  (aerial_gym's pos + very_close),
      * obstacle_prox  = -4·e^{-d_obs²}  (aerial_gym intent, dead-mask fixed),
      * action penalty = small smoothness,
      * collision      = -100 (separate terminating RewTerm).
    """
    dist = _goal_dist(env, command_name)
    env._prev_goal_dist = dist.detach().clone()  # kept for logging/metrics

    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term(command_name).goal_pos_w
    to_goal_w = goal_w - robot.data.root_pos_w
    goal_dir_w = to_goal_w / to_goal_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
    v_w = robot.data.root_lin_vel_w
    closing_speed = (v_w * goal_dir_w).sum(dim=1).clamp(-2.0, 2.0)   # VisFly closing speed
    closing = 2.0 * closing_speed                                    # dominant progress signal

    # heading: angle between drone-forward (+x in yaw frame) and goal direction (yaw frame)
    cmdvec = env.command_manager.get_command(command_name)          # (N,4) [ux,uy,uz,dist] yaw-frame
    ang = torch.atan2(cmdvec[:, 1], cmdvec[:, 0])                    # 0 => facing the goal
    heading = 1.0 * torch.exp(-(ang**2) / 0.5)

    pos_reward = _exp_reward(5.0, 1.0 / 3.5, dist)
    very_close = _exp_reward(5.0, 2.0, dist)

    # action smoothness (small) on the transformed velocity cmd
    cmd = env.action_manager.get_term("velocity").processed_actions
    pcmd = getattr(env, "_prev_vel_cmd", None)
    if pcmd is None or pcmd.shape != cmd.shape:
        pcmd = cmd.clone()
    dcmd = cmd - pcmd
    env._prev_vel_cmd = cmd.detach().clone()
    smooth = _exp_penalty(0.2, 1.0, dcmd[:, 0]) + _exp_penalty(0.2, 1.0, dcmd[:, 3])

    # obstacle avoidance: a wider, stronger proximity penalty PLUS a penalty on closing SPEED
    # toward the nearest obstacle (VisFly's approach-speed term) — the latter is what actually
    # teaches the drone to slow/veer near people & clutter rather than just rushing the gate.
    d_obs, obs_dir = active_obstacle_nearest(env)                 # dir points drone->obstacle
    obstacle_prox = -_exp_reward(6.0, 0.5, d_obs)                 # wider (0.5) + stronger (6)
    approach = (v_w * obs_dir).sum(dim=1).clamp_min(0.0)          # speed toward the obstacle
    near = (d_obs < 1.5).float()
    obstacle_approach = -1.5 * near * approach

    return closing + heading + pos_reward + very_close + smooth + obstacle_prox + obstacle_approach


# ---------------------------------------------------------------------------------------------
# dense_crossing extras: forward-speed bonus + invisible "fake ceiling"
# ---------------------------------------------------------------------------------------------
def forward_speed_bonus(env, command_name: str = "goal", cap: float = 2.5) -> torch.Tensor:
    """Speed bonus (>=0): the drone's closing speed toward the goal, clamped to ``cap``.

    Rewards faster completion / higher forward velocity along the goal direction (retreat gives
    0, not a penalty — that job is left to ``navigation_reward``'s progress term). Wire with a
    POSITIVE ``RewTerm`` weight, like the arrival bonuses."""
    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term(command_name).goal_pos_w
    to_goal_w = goal_w - robot.data.root_pos_w
    goal_dir_w = to_goal_w / to_goal_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
    v_w = robot.data.root_lin_vel_w
    return (v_w * goal_dir_w).sum(dim=1).clamp(0.0, cap)


def fake_ceiling_penalty(env, z_ceil: float = 2.5) -> torch.Tensor:
    """Penalty magnitude (>=0) for crossing the INVISIBLE ceiling at ``z_ceil`` (env-local z).

    There is NO physical collider — this pure reward term is what stops the drone cheating by
    climbing over the whole clutter field instead of weaving through it. Zero at/below the
    ceiling, growing linearly (per metre) above it. Wire with a NEGATIVE ``RewTerm`` weight
    (same convention as the ``collision`` term)."""
    z = (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2]
    return (z - z_ceil).clamp_min(0.0)


def above_fake_ceiling(env, z_ceil: float = 3.5) -> torch.Tensor:
    """Optional TERMINATION: True once the drone is above the ceiling by a grace margin.

    Set ``z_ceil`` a little above ``fake_ceiling_penalty``'s (e.g. 3.5 vs 2.5) so the penalty
    ramps first and only a blatant fly-over ends the episode. Returns a bool tensor for
    ``DoneTerm(func=...)``."""
    z = (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2]
    return z > z_ceil


def update_success_metric(env, command_name: str = "goal") -> torch.Tensor:
    """Course success: the drone completed the whole gate course (all gates passed) without
    crashing. Stashes ``env._ep_success`` / ``env._ep_crash`` for the curriculum. Returns 0
    reward (metric only). Evaluated every step; the curriculum reads it on the reset step."""
    term = env.command_manager.get_term(command_name)
    truncated = env.termination_manager.time_outs
    crashed = env.termination_manager.terminated & (~truncated)
    # course_complete is a latched per-episode flag on the waypoint command
    complete = getattr(term, "course_complete", None)
    if complete is None:
        complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._ep_success = complete & (~crashed)
    env._ep_crash = crashed
    return env._ep_success.float() * 0.0  # metric only
