# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-command action term with a fixed geometric low-level controller.

This gives the warehouse-nav RL policy the SAME control interface as aerial_gym's
navigation_task: the policy emits a 4-vector in [-1,1] which a polar transform maps to a
body/vehicle-frame velocity setpoint ``[vx>=0, vy=0, vz, yawrate]`` (aerial_gym
``navigation_task_config.py:87-117``), and a FIXED geometric velocity controller converts
that setpoint to a collective-thrust + body-moment wrench.  RL learns *navigation*; the
controller handles stabilization — the same two-tier split we already use for DroNet.

The controller is a compact port of aerial_gym's ``LeeVelocityController`` control law
(``control/controllers/velocity_control.py`` + ``base_lee_controller.py``), rewritten
natively against Isaac's articulation data + ``isaaclab.utils.math`` conventions rather than
importing aerial_gym's state-dict-coupled stack:

  v_err   -> a_des = Kv * v_err + g z_hat            (PD on velocity + gravity comp)
  thrust  = m * a_des . R[:, :, 2]                   (project onto body-z)
  R_des   from a_des direction + desired yaw
  moment  = J * (Kr * e_R + Kw * (w_des - w))        (geometric attitude control)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, normalize, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VelocityCommandAction(ActionTerm):
    """4-D velocity-command action → geometric velocity controller → thrust+moment wrench."""

    cfg: "VelocityCommandActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "VelocityCommandActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._body_id = self._asset.find_bodies(cfg.body_name)[0]

        m = self._asset.root_physx_view.get_masses()[0].sum().item()
        self._mass = m
        g = torch.tensor(env.sim.cfg.gravity, device=self.device)
        self._g_vec = g  # (3,), typically (0,0,-9.81)
        self._g_mag = g.norm().item()

        # diagonal inertia (crazyflie is tiny/symmetric; read from physx)
        inertia = self._asset.root_physx_view.get_inertias()[0, self._body_id[0]].to(self.device)
        # get_inertias returns a 9-vec (row-major 3x3); take the diagonal
        self._J = torch.tensor([inertia[0], inertia[4], inertia[8]], device=self.device)

        self._raw = torch.zeros(self.num_envs, 4, device=self.device)
        self._vel_sp = torch.zeros(self.num_envs, 4, device=self.device)  # [vx,vy,vz,yawrate], stored for obs
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # gains (tuned for the crazyflie mass/inertia scale; velocity PD + attitude PD)
        self._Kv = cfg.vel_gain
        self._KR = cfg.att_gain
        self._Kw = cfg.rate_gain

    # -- properties required by ActionTerm --
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._vel_sp

    # -- polar transform: [-1,1]^4 -> velocity setpoint (aerial_gym nav_cfg.py:87-117) --
    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        max_speed = self.cfg.max_speed
        max_incl = self.cfg.max_inclination
        max_yawrate = self.cfg.max_yawrate
        speed = a[:, 0] + 1.0  # [0,2] forward-only
        vx = speed * torch.cos(max_incl * a[:, 1]) * (max_speed / 2.0)
        vz = speed * torch.sin(max_incl * a[:, 1]) * (max_speed / 2.0)
        yawrate = a[:, 2] * max_yawrate  # note: aerial_gym reads ch2 for yawrate; ch3 unused
        self._vel_sp[:, 0] = vx
        self._vel_sp[:, 1] = 0.0
        self._vel_sp[:, 2] = vz
        self._vel_sp[:, 3] = yawrate

    # -- fixed geometric velocity controller: setpoint -> wrench --
    def apply_actions(self):
        data = self._asset.data
        quat = data.root_quat_w                       # (N,4) wxyz
        R = matrix_from_quat(quat)                     # (N,3,3), body->world
        v_w = data.root_lin_vel_w                      # (N,3) world-frame velocity
        w_b = data.root_ang_vel_b                      # (N,3) body-frame angular velocity

        # desired velocity: the setpoint's vx is forward in the YAW frame, vz is world-up.
        yq = yaw_quat(quat)
        # build world-frame desired velocity from [vx(fwd), 0, vz(up)] in yaw frame
        v_sp_yaw = torch.zeros_like(v_w)
        v_sp_yaw[:, 0] = self._vel_sp[:, 0]
        v_sp_yaw[:, 2] = self._vel_sp[:, 2]
        # rotate yaw-frame fwd/up into world (quat_apply_inverse of yaw would be world->yaw;
        # we want yaw->world so apply the yaw quat)
        from isaaclab.utils.math import quat_apply
        v_des_w = quat_apply(yq, v_sp_yaw)

        v_err = v_des_w - v_w
        # desired acceleration (PD on velocity) minus gravity (g_vec points down -> subtract)
        a_des = self._Kv * v_err - self._g_vec       # world frame
        forces_w = self._mass * a_des                 # (N,3)

        # collective thrust = projection of desired force onto body-z
        body_z = R[:, :, 2]                           # (N,3)
        thrust = (forces_w * body_z).sum(dim=1).clamp(min=0.0)  # (N,)
        self._thrust[:, 0, 2] = thrust

        # desired attitude: body-z aligned with desired force dir. We command a yaw RATE, not a
        # yaw angle, so keep the desired yaw at the CURRENT yaw (zero yaw attitude error) and let
        # the yaw motion come from the angular-velocity feedforward w_des below. This is the
        # standard way to track a rate and avoids per-substep dt integration.
        z_des = normalize(forces_w)
        from isaaclab.utils.math import euler_xyz_from_quat
        _, _, cy = euler_xyz_from_quat(quat)
        yaw_des = cy
        # build desired rotation from z_des and yaw_des
        x_c = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=1)
        y_des = normalize(torch.cross(z_des, x_c, dim=1))
        x_des = torch.cross(y_des, z_des, dim=1)
        R_des = torch.stack([x_des, y_des, z_des], dim=2)  # columns = axes

        # attitude error e_R = 0.5 * vee(R_des^T R - R^T R_des)
        errM = 0.5 * (torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des))
        e_R = torch.stack([errM[:, 2, 1], errM[:, 0, 2], errM[:, 1, 0]], dim=1)  # (N,3)

        w_des = torch.zeros_like(w_b)
        w_des[:, 2] = self._vel_sp[:, 3]  # commanded yaw rate
        e_w = w_b - w_des

        moment = -self._KR * e_R - self._Kw * e_w
        moment = self._J.unsqueeze(0) * moment  # scale by inertia
        self._moment[:, 0, :] = moment

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )


@configclass
class VelocityCommandActionCfg(ActionTermCfg):
    class_type: type = VelocityCommandAction
    asset_name: str = "robot"
    body_name: str = "body"
    max_speed: float = 2.0            # aerial_gym max_speed (nav_cfg.py:89)
    max_inclination: float = 0.785398  # pi/4 (nav_cfg.py:96)
    max_yawrate: float = 1.047198      # pi/3 (nav_cfg.py:90)
    # Controller gains — CASCADE STABILITY: the attitude (inner) loop must be much faster
    # than the velocity (outer) loop, else the tilt diverges to a flip. Inner natural freq
    # sqrt(att_gain)=~14 rad/s >> outer bandwidth vel_gain=2 rad/s (>5x). rate_gain gives
    # attitude damping ratio ~0.7 (=rate_gain/(2*sqrt(att_gain))).
    vel_gain: float = 2.0
    att_gain: float = 200.0
    rate_gain: float = 20.0
