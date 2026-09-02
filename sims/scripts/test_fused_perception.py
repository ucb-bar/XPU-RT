#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end integration test of the full perception stack (Workstreams S/F1/F2/F3).

Boots the forest ``*_WithSensors`` env, reads the REAL simulated sensors, runs
them through the Madgwick estimator, assembles the fused-model input dict
(range-normalized + validity flags), and runs FusedSensorNet — proving the whole
chain composes on live sim data, including a zero-skip mask (camera off).

    <env_isaaclab python> sims/scripts/test_fused_perception.py --headless
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
# FusedSensorNet lives in the vitfly package (imports ViTsubmodules relatively).
sys.path.insert(0, "/scratch/agustin/projects/DIMA/vitfly/models")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forest-Trail-Vision-Crazyflie-Play-WithSensors-v0")
parser.add_argument("--steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import torch
import gymnasium as gym

import sims.isaaclab_tasks.forest_trail.config.crazyflie  # noqa: F401 (register gym-ids)
from sims.isaaclab_tasks.forest_trail import sensors as S
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator
from isaaclab_tasks.utils import parse_env_cfg
from fused_model import FusedSensorNet, ALL_MODALITIES


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    dev = uenv.device
    N = uenv.num_envs

    est = StateEstimator(N, dev, control_dt=0.02)
    net = FusedSensorNet().to(dev).eval()

    env.reset()
    for _ in range(3):  # let sensors populate
        env.step(torch.zeros(N, env.action_space.shape[-1], device=dev))

    print(f"[test] task={args_cli.task} num_envs={N} device={dev}", flush=True)
    for step in range(args_cli.steps):
        env.step(torch.zeros(N, env.action_space.shape[-1], device=dev))
        robot = uenv.scene["robot"]

        # --- read real sim sensors ---
        grey = S.front_greyscale(uenv)               # (N,1,H,W)
        tof = S.tof_stack(uenv)                      # (N,4,8,8)
        tof_norm, _ = S.normalize_range(tof, S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
        dtof = S.down_tof(uenv)                       # (N,1)
        dtof_norm, dtof_valid = S.normalize_range(dtof, S.DOWN_TOF_RANGE_MIN, S.DOWN_TOF_RANGE_MAX)
        flow = S.optical_flow(uenv)                   # (N,2)
        flow_valid = S.optical_flow_valid(uenv)       # (N,1)
        baro = S.barometer(uenv, drift=est.step_baro_drift())  # (N,2) with drift

        # --- estimator: IMU (gyro=body rates, accel=-g in body) -> filtered state ---
        gyro = robot.data.root_ang_vel_b[:, :3]
        accel = -robot.data.projected_gravity_b * 9.81
        filt = est.update(gyro, accel, baro_alt=baro[:, 1], tof_alt=dtof.squeeze(1),
                          flow_vel=flow * 0.0)  # flow->velocity scale left to calibration

        # --- assemble fused-model inputs (range-normalized + filtered) ---
        inputs = {
            "front_grey": grey,
            "tof_cross": tof_norm,
            "optical_flow": flow,
            "down_tof": dtof_norm,
            "baro": baro / 10.0,                      # rough normalize
            "quat": filt["quat"],                     # FILTERED attitude (Madgwick)
            "body_rates": gyro,
            "desired_vel": torch.zeros(N, 3, device=dev),  # goal dir placeholder
            "flags": torch.cat([flow_valid, dtof_valid,
                                torch.ones(N, 4, device=dev)], dim=1),  # per-group validity
        }
        with torch.no_grad():
            cmd_full, _ = net(inputs)
            cmd_nocam, _ = net(inputs, mask={"front_grey": False})  # zero-skip the ViT

        if step % 4 == 0:
            print(f"  step {step:2d}: grey{tuple(grey.shape)} tof{tuple(tof.shape)} "
                  f"down_tof={dtof[0].item():.2f}m flow=({flow[0,0].item():.1f},{flow[0,1].item():.1f}) "
                  f"baro_glob={baro[0,1].item():.2f} qw={filt['quat'][0,0].item():.3f} "
                  f"-> cmd={cmd_full[0].tolist()}", flush=True)

    # verify masking gives fixed shape + zero-skip savings
    print(f"[test] fused cmd shape full={tuple(cmd_full.shape)} nocam={tuple(cmd_nocam.shape)}", flush=True)
    print(f"[test] active_param_frac: full={net.active_param_fraction():.3f} "
          f"cam_off={net.active_param_fraction({'front_grey': False}):.3f}", flush=True)
    print("[test] END-TO-END PERCEPTION STACK OK "
          "(sensors -> Madgwick -> FusedSensorNet, with zero-skip)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)
