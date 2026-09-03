# Copyright (c) 2026 UC Berkeley
# SPDX-License-Identifier: BSD-3-Clause
"""Motor-thrust action term — makes an ON-SoC controller (TinyMPC) the real controller.

Stage-2 of the RoSE fused-nav co-sim moves the state estimator + TinyMPC ONTO the SoC guest,
which outputs 4 normalized per-rotor motor thrusts (TinyMPC's native output, ~0.583 at hover).
This term REPLACES the warehouse VelocityCommandAction (a host-side geometric Lee velocity
tracker): instead of the host stabilizing the drone, the guest's 4 thrusts are applied directly
as per-rotor forces + propeller-drag yaw torque — identical physics to
envs/isaac_crazyflie/crazyflie_mpc_env.py (_apply_wrench / _thrusts_to_forces), which is
already validated to fly the crazyflie under TinyMPC. The warehouse robot is the SAME
CRAZYFLIE_CFG articulation, so the per-rotor arm geometry reproduces roll/pitch/collective
naturally.

apply_actions() re-asserts the wrench every physics substep (the ActionTerm contract), matching
the crazyflie_mpc substep loop.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# crazyflie_mpc_env.py constants (gym-pybullet-drones CF2X scale that TinyMPC's thrusts assume).
_HOVER_THRUST = 0.583           # normalized thrust at hover (per motor)
_MAX_THRUST_N = 0.58 / 4.0      # N per motor at full normalized command
_KM_OVER_KF = 7.94e-12 / 3.16e-10
_SPIN_SIGN = (+1.0, -1.0, +1.0, -1.0)


class MotorThrustAction(ActionTerm):
    """4 normalized per-rotor thrusts (from the SoC/TinyMPC) → per-rotor force + yaw torque."""

    cfg: "MotorThrustActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "MotorThrustActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._prop_ids = self._asset.find_bodies(cfg.prop_regex)[0]
        assert len(self._prop_ids) == 4, f"expected 4 props, found {self._prop_ids}"
        # Base (frame) body: the propeller-drag YAW reaction must be applied HERE, not to the props.
        # The prop joints are free-spinning revolute-z (ImplicitActuator stiffness=damping=0), so a
        # z-torque on a prop body just spins the prop and never reaches the frame -> zero yaw
        # actuation. A real drone's frame feels the drag reaction; Stage-1's Lee controller likewise
        # applies its yaw torque to the base. So we sum the per-rotor drag torques onto the base.
        self._base_id = self._asset.find_bodies(cfg.base_body)[0]
        assert len(self._base_id) == 1, f"expected 1 base body, found {self._base_id}"
        self._raw = torch.zeros(self.num_envs, 4, device=self.device)
        self._forces_n = torch.zeros(self.num_envs, 4, device=self.device)  # per-rotor force (N)
        self._tz = torch.tensor([s * _KM_OVER_KF for s in _SPIN_SIGN], device=self.device)
        # Plant gain IDENTICAL to the VALIDATED crazyflie_mpc_env (which flies TinyMPC to a stable
        # hover): per-rotor force = (u+0.583)*_MAX_THRUST_N with the FIXED 0.145/4 N scale. TinyMPC's
        # model (quadrotor_50hz_params) was tuned against THIS plant gain — any other scale is a
        # model/plant mismatch that makes the model-based controller oscillate. (u=0 is NOT the
        # open-loop hover here; TinyMPC closes the loop and trims to the true hover u≈-0.106.)
        self._max_thrust_n = _MAX_THRUST_N
        mass = float(self._asset.root_physx_view.get_masses().sum())
        print(f"[MotorThrust] mass={mass:.5f}kg max_thrust={self._max_thrust_n:.5f}N/rotor "
              f"(crazyflie_mpc plant gain; TinyMPC trims to hover)", flush=True)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._forces_n

    def process_actions(self, actions: torch.Tensor):
        """Normalized thrusts u (~[-0.583,0.417]) → per-motor force N: (u+0.583)*max_thrust
        (max_thrust derived from robot mass so u=0 → exact per-rotor hover force)."""
        self._raw[:] = actions
        self._forces_n[:] = ((actions + _HOVER_THRUST) * self._max_thrust_n).clamp(min=0.0)

    def apply_actions(self):
        # per-rotor +z forces at the prop arm positions -> collective + roll/pitch (these transfer
        # through the joints to the frame). NO yaw z-torque on the props (it would be lost to the
        # free-spinning revolute-z joints).
        forces = torch.zeros(self._asset.num_instances, 4, 3, device=self.device)
        forces[:, :, 2] = self._forces_n
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torch.zeros_like(forces), body_ids=self._prop_ids,
        )
        # propeller-drag YAW reaction: sum the per-rotor drag torques and apply the total to the
        # BASE (frame) body — a real drone's frame feels this reaction (Stage-1's Lee controller
        # applies its yaw torque to the base too). Applying it to the free prop joints yields ZERO
        # base yaw actuation (the on-SoC yaw-failure root cause).
        base_wr = torch.zeros(self._asset.num_instances, 1, 3, device=self.device)
        base_wr[:, 0, 2] = (self._tz.unsqueeze(0) * self._forces_n).sum(dim=1)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros_like(base_wr), torques=base_wr, body_ids=self._base_id,
        )


@configclass
class MotorThrustActionCfg(ActionTermCfg):
    class_type: type = MotorThrustAction
    asset_name: str = "robot"
    prop_regex: str = "m.*_prop"
    base_body: str = "body"      # frame body that receives the summed propeller-drag yaw torque
