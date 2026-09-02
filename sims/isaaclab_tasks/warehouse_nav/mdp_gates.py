# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A real gate course (CRL-Drone-Racing style): a fixed sequence of square gate FRAMES the
drone must fly THROUGH in order.

Each gate is a square frame (~1 m clear opening, 0.11 m thick, 0.14 m bar) built from 4 thin
collidable cuboids, standing vertical with its center at 1.2 m, oriented to face the flight
direction — matching CRL's gate.glb (1.28 m outer / ~1.0 m opening) and its pass mechanics:
the frame is a real collider (mis-flying clips it -> collision termination), and a gate is
"passed" when the drone center comes within ``success_radius`` (CRL uses 0.3 m; we use a bit
more for learnability) of the gate center, which arms the next gate.

Gates are a FIXED per-env course (like CRL demo3_U) so they render as real static frames.
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

# Fixed gate course down a REAL warehouse aisle (warehouse-local metres). The full_warehouse
# has 7 rack rows along x with ~3.95 m clear aisles between them (probe: out/warehouse_geom.json);
# we fly the aisle centred on x=-8.0 (between rack rows at x=-10.46 and x=-5.5), heading north in
# +y from the aisle mouth (y~6) to the far end (y~24). Gates are a gentle x-weave within the
# aisle clear width (x in [-9.96,-6.0]) so a forward-only drone threads them while dodging the
# obstacle field between them. (pos_xyz, yaw_facing); yaw=0 => opening faces +y (flight dir).
GATES = [
    ((-8.5, 9.0, 1.2), 0.0),
    ((-7.5, 13.0, 1.2), 0.0),
    ((-8.5, 17.0, 1.2), 0.0),
    ((-7.6, 21.0, 1.2), 0.0),
]

# Gentler gate weave for the FUSED-SENSOR BC nav scene (±0.25 m about the aisle centre x=-8,
# vs the RL course's ±0.5 m). The aggressive RL weave needs sharp lateral REVERSALS in the 4 m
# aisle that a forward-only controller + BC student can't execute at a brisk 1.6 m/s (it overshoots
# into a rack at each reversal). This gentle weave is a near-straight thread through 4 gates —
# still a real photoreal aisle nav task (props + walking people to avoid), but flyable at speed.
# Kept SEPARATE so the RL GATES / trained warehouse policy stay byte-identical.
# Gate centres are raised to z=2.0 (opening spans z∈[1.25,2.75]) so the flight altitude that
# threads them (an altitude-hold autopilot at 2.0 m, see cmd_to_action in collect/eval) also
# clears the low warehouse dressing intruding near the aisle mouth — a rack-shelf edge at
# (x≈-8, y≈10.8, z≲1.7) that clipped the drone whenever it held a low spawn height. Flying at
# 2.0 m gives ~0.3 m clearance over that clutter while sitting dead-centre in the gate opening.
FUSED_GATES = [
    ((-8.05, 9.0, 2.0), 0.0),
    ((-8.30, 13.0, 2.0), 0.0),
    ((-7.75, 17.0, 2.0), 0.0),
    ((-8.05, 21.0, 2.0), 0.0),
]

# gate frame dimensions (metres). CRL gate.glb is outer 1.28 / opening ~1.0; we use a wider
# ~1.5 m opening so a from-scratch forward-only policy can reliably thread it.
_OPEN = 1.5
_BAR = 0.14
_THICK = 0.12
_OUTER = _OPEN + 2 * _BAR


def _yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def make_gate_scene(collidable: bool = False, gates=None) -> dict:
    """Return {name: AssetBaseCfg} of cuboid bars for every gate (4 bars each).

    ``gates`` defaults to the RL ``GATES`` course; pass ``FUSED_GATES`` for the gentler
    fused-nav weave. A gate facing +y lies in the x-z plane: left/right vertical bars + top/bottom
    horizontal bars. The whole gate is rotated by its yaw about the vertical axis.

    ``collidable=False`` (default, for from-scratch RL) makes the frames VISUAL-only so a
    forward-only policy learns to thread them without a catastrophic -100 for clipping a bar
    (which otherwise traps it hovering just outside). ``collidable=True`` (for the fused BC demo
    / honest eval, where the expert threads them cleanly) makes each bar a real static collider,
    so mis-flying clips it → collision termination — an honest "flew THROUGH the opening" test.
    """
    gates = GATES if gates is None else gates
    out: dict = {}
    half = _OPEN / 2 + _BAR / 2  # bar center offset from gate center
    # bar (local, gate facing +y): (offset_xyz, size_xyz)
    bars = [
        ((-half, 0.0, 0.0), (_BAR, _THICK, _OUTER)),   # left vertical
        ((+half, 0.0, 0.0), (_BAR, _THICK, _OUTER)),   # right vertical
        ((0.0, 0.0, +half), (_OPEN, _THICK, _BAR)),    # top horizontal
        ((0.0, 0.0, -half), (_OPEN, _THICK, _BAR)),    # bottom horizontal
    ]
    for gi, (gc, gyaw) in enumerate(gates):
        c, s = math.cos(gyaw), math.sin(gyaw)
        # brighter frame for the first gate so the course start reads clearly
        color = (0.9, 0.45, 0.1)
        for bi, (off, size) in enumerate(bars):
            # rotate the bar's local x-offset by the gate yaw (z unchanged)
            ox = off[0] * c - off[1] * s
            oy = off[0] * s + off[1] * c
            pos = (gc[0] + ox, gc[1] + oy, gc[2] + off[2])
            # VISUAL frame only (no collider): the drone flies through the opening, pulled to
            # the gate center, without a catastrophic -100 crash risk for clipping a bar — which
            # otherwise makes a from-scratch policy hover just outside instead of threading.
            # (A collidable "clip = crash" gate is a harder follow-up variant.)
            out[f"gate_{gi}_bar_{bi}"] = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Gate_{gi}_bar_{bi}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color,
                                                               emissive_color=(0.35, 0.15, 0.0)),
                    collision_props=(sim_utils.CollisionPropertiesCfg() if collidable else None),
                    semantic_tags=[("class", "gate")],
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=_yaw_quat(gyaw)),
            )
    return out


class FixedGateCourseCommand(CommandTerm):
    """Tracks progress through the fixed GATES course; exposes the CURRENT gate as the goal."""

    cfg: "FixedGateCourseCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.K = len(GATES)
        centers = torch.tensor([g[0] for g in GATES], device=self.device)  # (K,3) env-local
        self.gate_centers_local = centers
        self.current_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.just_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.course_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["gates_passed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance_to_gate"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        return f"FixedGateCourseCommand(K={self.K}, radius={self.cfg.success_radius})"

    @property
    def goal_pos_w(self) -> torch.Tensor:
        cur = self.gate_centers_local[self.current_idx]                 # (N,3) local
        return self._env.scene.env_origins + cur                       # world

    @property
    def command(self) -> torch.Tensor:
        robot = self._env.scene["robot"]
        to_g = self.goal_pos_w - robot.data.root_pos_w
        dist = torch.norm(to_g, dim=1, keepdim=True).clamp_min(1e-6)
        to_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), to_g)
        return torch.cat([to_b / dist, dist], dim=1)

    def _resample_command(self, env_ids):
        self.current_idx[env_ids] = 0
        self.course_complete[env_ids] = False

    def _update_command(self):
        robot = self._env.scene["robot"]
        dist = torch.norm(self.goal_pos_w - robot.data.root_pos_w, dim=1)
        reached = dist < self.cfg.success_radius
        self.just_passed = reached & (~self.course_complete)
        advance = self.just_passed & (self.current_idx < self.K - 1)
        self.current_idx = torch.where(advance, self.current_idx + 1, self.current_idx)
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
class FixedGateCourseCommandCfg(CommandTermCfg):
    class_type: type = FixedGateCourseCommand
    success_radius: float = 0.9  # CRL uses 0.3; a bit larger for from-scratch learnability


def gate_pass_bonus(env, command_name: str = "goal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).just_passed.float()


def course_complete_bonus(env, command_name: str = "goal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).course_complete.float()
