#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantitative body-rate tracking eval for the CTBR control policy (Workstream A).

Injects DETERMINISTIC reference signals (per-axis step responses + a frequency
sweep) into the ``body_rate_command`` and measures how well the achieved
``root_ang_vel_b`` tracks them. Works for two controllers:

  --controller fixed        : an analytic proportional rate controller (the
                              baseline the learned policy must match or beat).
  --controller checkpoint   : a trained PPO policy loaded from --checkpoint.

Metrics (written to CSV + PNG plots): per-axis rise time, overshoot, steady-
state error (step); tracking gain + phase lag (sweep); combined RMS error.

    <env_isaaclab python> sims/scripts/eval_bodyrate_tracking.py --headless \
        --controller fixed --out /path/to/out
    <env_isaaclab python> sims/scripts/eval_bodyrate_tracking.py --headless \
        --controller checkpoint --checkpoint <run>/model_XXXX.pt --out /path/to/out
"""

import argparse
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Body-rate tracking eval.")
parser.add_argument("--task", type=str, default="Isaac-Track-BodyRate-Crazyflie-v0")
parser.add_argument("--controller", type=str, default="fixed", choices=["fixed", "checkpoint"])
parser.add_argument("--checkpoint", type=str, default=None, help="PPO checkpoint (.pt) for --controller checkpoint.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--kp", type=float, default=8.0, help="Proportional gain (rad/s^2 per rad/s err) for the fixed controller.")
parser.add_argument("--out", type=str, default=None, help="Output dir (default: scratchpad).")
parser.add_argument("--seed", type=int, default=0)
# Fixed off-nominal PLANT overrides for the robustness / OOD-dynamics eval (task ROB).
# Leave as None to use the env's nominal Crazyflie plant.
parser.add_argument("--ood_t2w", type=float, default=None, help="Override thrust-to-weight (OOD dynamics test).")
parser.add_argument("--ood_mscale", type=float, default=None, help="Override moment scale (OOD dynamics test).")
parser.add_argument("--ood_tau", type=float, default=None, help="Override motor-lag tau in s (OOD dynamics test).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
import numpy as np
import torch
import gymnasium as gym

import sims.isaaclab_tasks.track_steering_vision.config.crazyflie  # noqa: F401 (register)
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.body_rate_env_cfg import TrackBodyRateEnvCfg

AXES = ("wx", "wy", "wz")


def build_env():
    cfg = TrackBodyRateEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.sim.device = args_cli.device
    # Freeze auto-resampling: we drive the command ourselves each control step.
    cfg.commands.body_rate_command.resampling_time_range = (1.0e9, 1.0e9)
    # Fixed off-nominal plant for the OOD robustness eval (deterministic, not randomized).
    if args_cli.ood_t2w is not None:
        cfg.actions.thrust_moment.thrust_to_weight = args_cli.ood_t2w
    if args_cli.ood_mscale is not None:
        cfg.actions.thrust_moment.moment_scale = args_cli.ood_mscale
    if args_cli.ood_tau is not None:
        cfg.actions.thrust_moment.motor_tau = args_cli.ood_tau
    env = gym.make(args_cli.task, cfg=cfg)
    return env, env.unwrapped


def load_policy(wrapped):
    """Load a trained PPO actor as a callable obs->action (checkpoint controller)."""
    from rsl_rl.runners import OnPolicyRunner
    from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (
        SteeringTrackingPPORunnerCfg,
    )
    acfg = SteeringTrackingPPORunnerCfg()
    runner_cfg = {
        "algorithm": acfg.algorithm.to_dict(),
        "actor": {"class_name": acfg.actor.class_name, "hidden_dims": acfg.actor.hidden_dims,
                  "activation": acfg.actor.activation, "obs_normalization": acfg.actor.obs_normalization,
                  "distribution_cfg": acfg.actor.distribution_cfg.to_dict()},
        "critic": {"class_name": acfg.critic.class_name, "hidden_dims": acfg.critic.hidden_dims,
                   "activation": acfg.critic.activation, "obs_normalization": acfg.critic.obs_normalization},
        "obs_groups": acfg.obs_groups, "num_steps_per_env": acfg.num_steps_per_env,
        "max_iterations": acfg.max_iterations, "save_interval": acfg.save_interval,
        "experiment_name": acfg.experiment_name, "empirical_normalization": False,
    }
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    return runner.get_inference_policy(device=args_cli.device)


class _EnvHandle:
    env = None


def set_command(uenv, ref):
    """Overwrite the body_rate_command buffers with ref = [wx, wy, wz, thrust_norm]."""
    term = uenv.command_manager.get_term("body_rate_command")
    dev = uenv.device
    term.target_rates[:] = torch.tensor(ref[:3], device=dev).unsqueeze(0).expand(uenv.num_envs, 3)
    term.target_thrust[:] = torch.tensor(ref[3], device=dev)


def get_obs(uenv):
    # rsl_rl inference policy expects the full obs dict ({"policy": tensor}), not a bare tensor.
    return uenv.observation_manager.compute()


def fixed_action(uenv, ref, moment_scale=0.01):
    """Analytic proportional body-rate controller expressed in the action space of
    DirectThrustMomentAction: action = [2*thrust-1, moment_xyz/moment_scale]."""
    dev = uenv.device
    robot = uenv.scene["robot"]
    w = robot.data.root_ang_vel_b[:, :3]
    w_des = torch.tensor(ref[:3], device=dev).unsqueeze(0).expand_as(w)
    # inertia diag from physx (per-env same); moment = J * Kp * (w_des - w)
    inertia = robot.root_physx_view.get_inertias()[0].reshape(-1)[[0, 4, 8]].to(dev)
    moment = inertia.unsqueeze(0) * (args_cli.kp * (w_des - w))
    a = torch.zeros(uenv.num_envs, 4, device=dev)
    a[:, 0] = 2.0 * ref[3] - 1.0
    a[:, 1:4] = (moment / moment_scale).clamp(-1.0, 1.0)
    return a


def rise_time(t, y, target, lo=0.1, hi=0.9):
    """10%->90% rise time toward target (returns nan if never reached)."""
    if abs(target) < 1e-6:
        return float("nan")
    frac = y / target
    try:
        i_lo = np.argmax(frac >= lo)
        i_hi = np.argmax(frac >= hi)
        if i_hi <= i_lo:
            return float("nan")
        return t[i_hi] - t[i_lo]
    except Exception:
        return float("nan")


def main():
    ctrl_dt = 0.02  # 50 Hz control (decimation 2 @ 100 Hz)
    hover = 1.0 / 1.9
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env, uenv = build_env()
    _EnvHandle.env = env
    # The trained policy consumes the obs produced by RslRlVecEnvWrapper (same path
    # the working pilot uses), NOT observation_manager.compute() directly.
    wrapped = RslRlVecEnvWrapper(env) if args_cli.controller == "checkpoint" else None
    policy = load_policy(wrapped) if args_cli.controller == "checkpoint" else None
    env.reset()

    # ---- build the reference schedule: per-axis step responses + a sweep ----
    segments = []  # (label, duration_s, ref_fn(t_local)->[wx,wy,wz,thrust])
    step_amps = {"wx": 2.0, "wy": 2.0, "wz": 1.5}
    for ai, ax in enumerate(AXES):
        A = step_amps[ax]
        def mk(ai=ai, A=A):
            def f(tl):
                r = [0.0, 0.0, 0.0, hover]
                r[ai] = A if tl >= 0.2 else 0.0  # step at 0.2 s
                return r
            return f
        segments.append((f"step_{ax}", 1.2, mk()))
    # frequency sweep on wx
    for freq in (0.5, 1.0, 2.0, 4.0):
        def mk(freq=freq):
            def f(tl):
                return [1.5 * math.sin(2 * math.pi * freq * tl), 0.0, 0.0, hover]
            return f
        segments.append((f"sweep_wx_{freq}Hz", max(2.0, 3.0 / freq), mk()))

    rows = []  # per control step: seg, t_local, cmd_wx.., ach_wx..
    for label, dur, ref_fn in segments:
        if policy is not None:
            obs, _ = wrapped.reset()
        else:
            env.reset()
        nsteps = int(dur / ctrl_dt)
        for k in range(nsteps):
            tl = k * ctrl_dt
            ref = ref_fn(tl)
            set_command(uenv, ref)  # inject the reference command this step
            if policy is not None:
                with torch.no_grad():
                    action = policy(obs)  # obs from the wrapper (prev step); 1-step lag is inherent
                obs, _, _, _ = wrapped.step(action)
            else:
                action = fixed_action(uenv, ref)
                env.step(action)
            ach = uenv.scene["robot"].data.root_ang_vel_b[0, :3].detach().cpu().numpy()
            rows.append([label, tl, ref[0], ref[1], ref[2], float(ach[0]), float(ach[1]), float(ach[2])])

    # ---- metrics ----
    rows_np = rows
    outdir = args_cli.out or os.path.join(
        "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad",
        f"bodyrate_eval_{args_cli.controller}")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "trace.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "t", "cmd_wx", "cmd_wy", "cmd_wz", "ach_wx", "ach_wy", "ach_wz"])
        w.writerows(rows_np)

    summary = {}
    # step metrics per axis
    for ai, ax in enumerate(AXES):
        seg = [r for r in rows_np if r[0] == f"step_{ax}"]
        if not seg:
            continue
        t = np.array([r[1] for r in seg])
        cmd = np.array([r[2 + ai] for r in seg])
        ach = np.array([r[5 + ai] for r in seg])
        A = step_amps[ax]
        mask = t >= 0.2
        t_s, ach_s = t[mask] - 0.2, ach[mask]
        rt = rise_time(t_s, ach_s, A)
        overshoot = (np.max(ach_s) - A) / A * 100 if len(ach_s) else float("nan")
        ss = ach_s[-max(1, len(ach_s) // 4):]  # last quarter
        ss_err = abs(np.mean(ss) - A) / abs(A)
        summary[f"step_{ax}"] = {"rise_time_s": round(float(rt), 4),
                                 "overshoot_pct": round(float(overshoot), 2),
                                 "ss_err_frac": round(float(ss_err), 4)}
    # sweep gain per freq (amplitude ratio)
    for freq in (0.5, 1.0, 2.0, 4.0):
        seg = [r for r in rows_np if r[0] == f"sweep_wx_{freq}Hz"]
        if not seg:
            continue
        cmd = np.array([r[2] for r in seg]); ach = np.array([r[5] for r in seg])
        # skip first half-period transient
        n0 = len(seg) // 3
        gain = (np.max(ach[n0:]) - np.min(ach[n0:])) / max(1e-6, (np.max(cmd[n0:]) - np.min(cmd[n0:])))
        summary[f"sweep_wx_{freq}Hz"] = {"gain": round(float(gain), 3)}
    # combined RMS over all segments
    allc = np.array([[r[2], r[3], r[4]] for r in rows_np])
    alla = np.array([[r[5], r[6], r[7]] for r in rows_np])
    rms = float(np.sqrt(np.mean((allc - alla) ** 2)))
    summary["combined_rms_rad_s"] = round(rms, 4)

    import json
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"controller": args_cli.controller, "kp": args_cli.kp,
                   "checkpoint": args_cli.checkpoint, **summary}, f, indent=2)

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(AXES), 1, figsize=(9, 7), sharex=False)
        for ai, ax in enumerate(AXES):
            seg = [r for r in rows_np if r[0] == f"step_{ax}"]
            t = [r[1] for r in seg]
            axes[ai].plot(t, [r[2 + ai] for r in seg], "k--", label="cmd")
            axes[ai].plot(t, [r[5 + ai] for r in seg], "b-", label="achieved")
            axes[ai].set_ylabel(f"{ax} (rad/s)"); axes[ai].legend(loc="lower right"); axes[ai].grid(alpha=0.3)
        axes[-1].set_xlabel("t (s)")
        fig.suptitle(f"Body-rate step response — {args_cli.controller}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "step_response.png"), dpi=110)
    except Exception as e:
        print(f"[eval] plot skipped: {e}", flush=True)

    print(f"[eval] controller={args_cli.controller}  combined_rms={rms:.4f} rad/s", flush=True)
    print(f"[eval] summary: {summary}", flush=True)
    print(f"[eval] wrote {csv_path} + summary.json + step_response.png in {outdir}", flush=True)


if __name__ == "__main__":
    main()
    # NOTE: simulation_app.close() hangs indefinitely in this Isaac build (it
    # blocks the ROB driver's subprocess.run forever). All outputs (summary.json,
    # trace.csv, step_response.png) are already flushed inside main(), so hard-exit
    # without closing — the OS reclaims the sim.
    os._exit(0)
