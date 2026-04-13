"""Custom action terms for steering tracking task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DirectThrustMomentAction(ActionTerm):
    """Direct thrust and moment control action term.

    This action term applies direct thrust (vertical force) and moments (roll, pitch, yaw torques)
    to the drone body. This is simpler and more direct than controlling individual propellers.

    Action space (4D):
        - action[0]: Total vertical thrust (normalized to [-1, 1])
        - action[1]: Roll moment (torque around x-axis)
        - action[2]: Pitch moment (torque around y-axis)
        - action[3]: Yaw moment (torque around z-axis)
    """

    cfg: DirectThrustMomentActionCfg
    _asset: Articulation
    _body_id: torch.Tensor

    def __init__(self, cfg: DirectThrustMomentActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get the robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Get body IDs for applying wrench (returns tensor of body indices)
        self._body_id = self._asset.find_bodies(cfg.body_name)[0]

        # Compute robot weight for thrust scaling
        robot_mass = self._asset.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        self._robot_weight = (robot_mass * gravity).item()

        # Storage for forces and torques
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Storage for raw and processed actions (required by ActionTerm base class)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        """Process actions and prepare wrench to apply.

        Args:
            actions: Actions from policy [num_envs, 4]
                action[:, 0]: Thrust (normalized [-1, 1])
                action[:, 1:4]: Moments [roll, pitch, yaw]
        """
        # Store raw actions (required by ActionTerm base class)
        self._raw_actions[:] = actions

        # Clamp actions to valid range
        actions = actions.clone().clamp(-1.0, 1.0)

        # Convert thrust action from [-1, 1] to [0, thrust_to_weight * weight]
        # action=-1 → 0% thrust, action=+1 → 100% thrust
        thrust_normalized = (actions[:, 0] + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * thrust_normalized

        # Moments are already in [-1, 1], just scale them
        self._moment[:, 0, :] = self.cfg.moment_scale * actions[:, 1:]

        # Store processed actions (required by ActionTerm base class)
        self._processed_actions[:] = actions

    def apply_actions(self):
        """Apply the computed wrench to the robot body."""
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    """
    Properties
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (4: thrust + 3 moments)."""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions after clamping."""
        return self._processed_actions


@configclass
class DirectThrustMomentActionCfg(ActionTermCfg):
    """Configuration for direct thrust and moment action term."""

    class_type: type[ActionTerm] = DirectThrustMomentAction

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    body_name: str = "body"
    """Name of the body to apply forces/torques to."""

    thrust_to_weight: float = 1.9
    """Thrust-to-weight ratio. Maximum thrust = thrust_to_weight * robot_weight."""

    moment_scale: float = 0.01
    """Scaling factor for moments. Final moment = moment_scale * action[1:4]."""
