# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint/gate course: the drone must pass a sequence of gates in order.

Turns the point-to-point nav into a structured COURSE (user request). Per env, a sequence of
K waypoints is sampled at reset (ordered along +y so the course progresses through the open
warehouse band). The command exposes the CURRENT waypoint (as ``goal_pos_w`` + a yaw-frame
dir/dist vector), so the existing ``navigation_reward`` drives the drone toward it unchanged.
When the drone gets within ``success_radius`` of the current gate, the index advances and a
pass bonus fires (``waypoint_pass_bonus``). Passing the final gate sets a course-complete flag.

Gate frames are drawn as thin visual markers at each waypoint (a square frame) so the course
is visible in renders; they are non-colliding (the "gate" is the pass-radius, like CRL).
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


class WaypointCourseCommand(CommandTerm):
    """A sequence of K waypoints sampled per episode; exposes the current one as the goal."""

    cfg: "WaypointCourseCommandCfg"

    def __init__(self, cfg: "WaypointCourseCommandCfg", env):
        super().__init__(cfg, env)
        self.K = cfg.num_waypoints
        self.waypoints_w = torch.zeros(self.num_envs, self.K, 3, device=self.device)
        self.current_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.just_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["gates_passed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance_to_gate"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        return f"WaypointCourseCommand(K={self.K}, radius={self.cfg.success_radius})"

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """World position of the CURRENT waypoint (drop-in for GoalPositionCommand.goal_pos_w)."""
        return torch.gather(self.waypoints_w, 1,
                            self.current_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    @property
    def command(self) -> torch.Tensor:
        """(N,4) = unit dir + distance to the current waypoint, in the vehicle (yaw) frame."""
        robot = self._env.scene["robot"]
        to_g = self.goal_pos_w - robot.data.root_pos_w
        dist = torch.norm(to_g, dim=1, keepdim=True).clamp_min(1e-6)
        to_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), to_g)
        return torch.cat([to_b / dist, dist], dim=1)

    def _resample_command(self, env_ids):
        r = self.cfg.ranges
        origins = self._env.scene.env_origins[env_ids]
        n = len(env_ids)
        # sample K waypoints, then sort by y so the course runs in +y through the open band
        lo = torch.tensor([r.pos_x[0], r.pos_y[0], r.pos_z[0]], device=self.device)
        hi = torch.tensor([r.pos_x[1], r.pos_y[1], r.pos_z[1]], device=self.device)
        wp = lo + (hi - lo) * torch.rand(n, self.K, 3, device=self.device)
        wp[..., 1] = torch.sort(wp[..., 1], dim=1).values  # order gates by y
        self.waypoints_w[env_ids] = origins[:, None, :] + wp
        self.current_idx[env_ids] = 0
        self.course_complete[env_ids] = False

    def _update_command(self):
        # advance the gate index when within success_radius of the current gate
        robot = self._env.scene["robot"]
        dist = torch.norm(self.goal_pos_w - robot.data.root_pos_w, dim=1)
        reached = dist < self.cfg.success_radius
        self.just_passed = reached & (~self.course_complete)
        advance = self.just_passed & (self.current_idx < self.K - 1)
        self.current_idx = torch.where(advance, self.current_idx + 1, self.current_idx)
        # reaching the LAST gate completes the course
        self.course_complete = self.course_complete | (reached & (self.current_idx == self.K - 1))

    def _update_metrics(self):
        robot = self._env.scene["robot"]
        self.metrics["gates_passed"] = self.current_idx.float() + self.course_complete.float()
        self.metrics["distance_to_gate"] = torch.norm(self.goal_pos_w - robot.data.root_pos_w, dim=1)

    def _set_debug_vis_impl(self, debug_vis):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class WaypointCourseCommandCfg(CommandTermCfg):
    class_type: type = WaypointCourseCommand
    num_waypoints: int = 4
    success_radius: float = 1.0

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


class FixedCircuitCourseCommand(WaypointCourseCommand):
    """Like WaypointCourseCommand but the waypoints are the FIXED route of a named circuit
    (circuits.py) — so the physics env flies the exact serpentine/aisle route from the preview
    videos, not a random per-episode course. This is how each circuit becomes a physics env."""

    cfg: "FixedCircuitCourseCommandCfg"

    def __init__(self, cfg, env):
        from . import circuits as C
        wp = C.CIRCUITS_BY_NAME[cfg.circuit]["waypoints"]
        self._wp_local = torch.tensor(wp, dtype=torch.float32, device=env.device)  # (K,3) env-local
        cfg.num_waypoints = self._wp_local.shape[0]
        super().__init__(cfg, env)

    def _resample_command(self, env_ids):
        origins = self._env.scene.env_origins[env_ids]
        self.waypoints_w[env_ids] = origins[:, None, :] + self._wp_local[None, :, :]
        self.current_idx[env_ids] = 0
        self.course_complete[env_ids] = False


@configclass
class FixedCircuitCourseCommandCfg(WaypointCourseCommandCfg):
    class_type: type = FixedCircuitCourseCommand
    circuit: str = MISSING
    ranges: WaypointCourseCommandCfg.Ranges = None  # unused (waypoints come from the circuit)


class RouteSteeringCommand(FixedCircuitCourseCommand):
    """Route-driven steering command for Dima's trained LOCOMOTION policy: exposes a 2-vector
    [yaw_rate, velocity] (the SteeringCommand convention the policy was trained on), computed to
    steer toward the current circuit waypoint. So the low-level steering policy — reused verbatim —
    flies the drone along the route with its LEARNED dynamics instead of a geometric controller."""

    @property
    def command(self) -> torch.Tensor:
        robot = self._env.scene["robot"]
        to_g = self.goal_pos_w - robot.data.root_pos_w
        target_yaw = torch.atan2(to_g[:, 1], to_g[:, 0])
        q = robot.data.root_quat_w
        cur_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        err = torch.atan2(torch.sin(target_yaw - cur_yaw), torch.cos(target_yaw - cur_yaw))
        yaw_rate = (err * self.cfg.yaw_gain).clamp(-1.0, 1.0)
        vel = torch.full_like(yaw_rate, self.cfg.fwd_velocity)
        return torch.stack([yaw_rate, vel], dim=1)


@configclass
class RouteSteeringCommandCfg(FixedCircuitCourseCommandCfg):
    class_type: type = RouteSteeringCommand
    yaw_gain: float = 1.2
    fwd_velocity: float = 1.0  # max of the steering policy's trained velocity range (faster flight)


# --- reward: bonus for passing each gate ---
def waypoint_pass_bonus(env, command_name: str = "goal") -> torch.Tensor:
    """+1 (scaled by term weight) on the step the drone passes a gate."""
    term = env.command_manager.get_term(command_name)
    return term.just_passed.float()


def course_complete_bonus(env, command_name: str = "goal") -> torch.Tensor:
    """+1 (scaled by weight) while the whole course is complete (loiter-after-finish)."""
    term = env.command_manager.get_term(command_name)
    return term.course_complete.float()
