"""Isolation test for the geometric velocity controller (VelocityCommandAction).

Spawns ONLY the crazyflie in an empty world (fast) and drives fixed 4-D actions through the
velocity action term, checking the drone tracks the commanded velocity setpoint:
  * hover  (action 0,0,0,0 -> vx=cos(0)*1=1? no: speed=1, vx=1*cos(0)*1=1) ... see below
  * we command specific setpoints and confirm achieved vx / vz / yaw-rate converge.

    python sims/scripts/test_velocity_controller.py --headless
"""

import argparse
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import base_lin_vel, last_action, reset_root_state_uniform, time_out, is_terminated
from isaaclab_assets import CRAZYFLIE_CFG

from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg


@configclass
class _Scene(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=300.0))
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _Obs:
    @configclass
    class P(ObsGroup):
        v = ObsTerm(func=base_lin_vel)
        a = ObsTerm(func=last_action)
        def __post_init__(self): self.concatenate_terms = True
    policy: P = P()


@configclass
class _Actions:
    vel = VelocityCommandActionCfg()


@configclass
class _Events:
    reset = EventTerm(func=reset_root_state_uniform, mode="reset",
                      params={"pose_range": {"z": (1.5, 1.5)}, "velocity_range": {}})


@configclass
class _Rew:
    alive = RewTerm(func=is_terminated, weight=0.0)


@configclass
class _Term:
    t = DoneTerm(func=time_out, time_out=True)


@configclass
class _Cfg(ManagerBasedRLEnvCfg):
    scene: _Scene = _Scene(num_envs=4, env_spacing=4.0)
    observations: _Obs = _Obs()
    actions: _Actions = _Actions()
    rewards: _Rew = _Rew()
    terminations: _Term = _Term()
    events: _Events = _Events()
    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


def main():
    env = ManagerBasedRLEnv(_Cfg())
    env.reset()
    robot = env.scene["robot"]
    dev = env.device
    n = env.num_envs

    # Command schedule: (label, action[-1,1]^4, expected)
    # polar: speed=a0+1; vx=speed*cos(pi/4*a1); vz=speed*sin(pi/4*a1); yawrate=a2*pi/3
    tests = [
        ("hover-ish fwd (a=0)",   torch.tensor([0.0, 0.0, 0.0, 0.0]), "vx~1.0, vz~0, yaw~0"),
        ("climb    (a1=+1)",      torch.tensor([0.0, 1.0, 0.0, 0.0]), "vz>0"),
        ("yaw left (a2=+1)",      torch.tensor([-1.0, 0.0, 1.0, 0.0]), "yawrate~+1.05"),
    ]
    lines = []
    for label, act, expect in tests:
        a = act.to(dev).unsqueeze(0).repeat(n, 1)
        # settle for 3 s
        for _ in range(300):
            env.step(a)
        v = robot.data.root_lin_vel_w[0]
        w = robot.data.root_ang_vel_b[0]
        from isaaclab.utils.math import euler_xyz_from_quat
        rr, pp, yy = euler_xyz_from_quat(robot.data.root_quat_w[:1])
        import math as _m
        pitch_deg = float(pp[0]) * 180.0 / _m.pi
        roll_deg = float(rr[0]) * 180.0 / _m.pi
        z = float(robot.data.root_pos_w[0, 2])
        sp = env.action_manager.get_term("vel").processed_actions[0]
        line = (f"[{label:20s}] setpoint[vx,vy,vz,yawr]=({sp[0]:+.2f},{sp[1]:+.2f},{sp[2]:+.2f},{sp[3]:+.2f})  "
                f"achieved v_w=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) yawrate={w[2]:+.2f} "
                f"roll={roll_deg:+.1f}deg pitch={pitch_deg:+.1f}deg z={z:.2f}  expect: {expect}")
        lines.append(line)
        print(line, flush=True)
        env.reset()

    with open(os.path.join(freshscheduler_root, "out", "velctrl_test.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
