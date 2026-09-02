"""Passthrough wrench action term (Agent B, HIL wrench mode).

For full compute-on-target HIL: the K1/target computes thrust + body moments (nav -> MLP controller)
and the host applies them DIRECTLY, with NO host-side controller recomputing the wrench. The env's
normal action terms (VelocityCommandAction / DirectThrustMoment) recompute + overwrite the wrench
each physics substep, so wrench-mode HIL needs this passthrough term instead: it interprets the 4-D
action as [thrust(N), moment_x, moment_y, moment_z] and applies it verbatim every substep.
"""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PassthroughWrenchAction(ActionTerm):
    cfg: "PassthroughWrenchActionCfg"
    _asset: Articulation

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._body_id = self._asset.find_bodies(cfg.body_name)[0]
        self._raw = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._thrust[:, 0, 2] = actions[:, 0].clamp(min=0.0)   # thrust along body-z
        self._moment[:, 0, :] = actions[:, 1:4]

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment)


@configclass
class PassthroughWrenchActionCfg(ActionTermCfg):
    class_type: type = PassthroughWrenchAction
    asset_name: str = "robot"
    body_name: str = "body"
