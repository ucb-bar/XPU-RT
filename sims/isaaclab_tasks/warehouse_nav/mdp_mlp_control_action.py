"""MLP low-level controller action term (Agent B, Track 1c closed-loop / eval E1).

Drop-in replacement for VelocityCommandAction that uses the DISTILLED MLP network instead of the
geometric Lee law: same 4-D velocity-command interface + polar transform (inherited), but
apply_actions() runs the trained MLP (state + velocity setpoint -> thrust + moments). Lets us fly
the warehouse with a fully-learned controller and compare to the classical baseline (E1 vs E0).
"""
from __future__ import annotations
import os, sys
import torch
from isaaclab.utils import configclass

from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import (
    VelocityCommandAction, VelocityCommandActionCfg)

_WRAP_DIR = "/scratch/agustin/projects/DIMA/coordination/k1_hil/shared/modelblaster"


class MLPVelocityCommandAction(VelocityCommandAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if _WRAP_DIR not in sys.path:
            sys.path.insert(0, _WRAP_DIR)
        from warehouse_mlp_control import get_model
        self._mlp = get_model().to(self.device).eval()
        # report env physx constants vs the teacher's assumed constants (distillation validity check)
        print(f"[MLPctrl] env mass={self._mass:.6f} kg  J_diag={[round(float(x),8) for x in self._J]}",
              flush=True)

    def apply_actions(self):
        data = self._asset.data
        quat = data.root_quat_w                       # (N,4) wxyz
        v_w = data.root_lin_vel_w                      # (N,3)
        w_b = data.root_ang_vel_b                      # (N,3)
        # obs = [vel_sp(vx_fwd, vz_up, yawrate), quat(4), v_w(3), w_b(3)]  (matches training layout)
        obs = torch.cat([self._vel_sp[:, 0:1], self._vel_sp[:, 2:3], self._vel_sp[:, 3:4],
                         quat, v_w, w_b], dim=1)       # (N,13)
        with torch.no_grad():
            out = self._mlp(obs.float())               # (N,4) thrust, mx, my, mz
        self._thrust[:, 0, 2] = out[:, 0].clamp(min=0.0)
        self._moment[:, 0, :] = out[:, 1:4]
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment)


@configclass
class MLPVelocityCommandActionCfg(VelocityCommandActionCfg):
    class_type: type = MLPVelocityCommandAction
