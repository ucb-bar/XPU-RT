#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manually pilot the trained steering policy with FPV camera preview.

The trained MLP policy is run as usual, but instead of sampling random
``(target_yaw_rate, target_velocity)`` commands every few seconds, the
SteeringCommand term is overwritten each step from a pilot input device.
This lets you drive the drone around and qualitatively judge whether the
policy is behaving as expected (smooth turns, stable hover, sensible
response to forward-velocity commands, etc.).

Input modes (--input_mode):
    slider   (default)  Drag matplotlib sliders inside the FPV window for
                        target_velocity and target_yaw_rate. Click "Hover" to
                        zero both commands. Recommended — does not depend on
                        Isaac Sim viewport focus.
    keyboard            Use Isaac Lab's Se2Keyboard via the viewport:
                        W/Up & S/Down for velocity, A/Left & D/Right for yaw,
                        L to reset to hover.

Usage:
    conda run -n xpurt python sims/scripts/pilot_steering_with_camera.py
    conda run -n xpurt python sims/scripts/pilot_steering_with_camera.py \\
        --checkpoint logs/rsl_rl/crazyflie_steering_tracking/XXX/model_XXX.pt
"""

import argparse
import os
import sys

# Add paths
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
isaaclab_root = os.path.join(freshscheduler_root, "sims/IsaacLab/source")
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_rl"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_contrib"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manually pilot the steering policy with FPV camera.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint (default: auto-find latest).",
)
parser.add_argument(
    "--max_velocity",
    type=float,
    default=1.0,
    help="Forward velocity command magnitude when W/Up is held (m/s). Matches the trained range by default.",
)
parser.add_argument(
    "--max_yaw_rate",
    type=float,
    default=1.0,
    help="Yaw rate command magnitude when A/Left is held (rad/s). Matches the trained range by default.",
)
parser.add_argument(
    "--no_fpv_plot",
    action="store_true",
    help="Do not open matplotlib window for live FPV display.",
)
parser.add_argument(
    "--save_fpv_frames",
    action="store_true",
    help="Save FPV frames to /tmp/fpv_pilot_frames for video stitching.",
)
parser.add_argument(
    "--history_window",
    type=int,
    default=200,
    help="Number of recent control steps to plot in the command/state history (default: 200).",
)
parser.add_argument(
    "--input_mode",
    type=str,
    choices=["slider", "keyboard"],
    default="slider",
    help=(
        "How to input pilot commands. 'slider' uses matplotlib sliders inside the FPV"
        " window (recommended; doesn't require Isaac Sim keyboard focus). 'keyboard' uses"
        " Isaac Lab's Se2Keyboard via the viewport."
    ),
)
AppLauncher.add_app_launcher_args(parser)
# FPV camera needs cameras enabled
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider
from rsl_rl.runners import OnPolicyRunner

# Register custom environments
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie  # noqa: F401

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (
    SteeringTrackingPPORunnerCfg,
)
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (
    TrackSteeringEnvCfg_PLAY,
)


def find_latest_checkpoint() -> str:
    """Find the most recently modified policy checkpoint."""
    log_dirs = [
        "/scratch2/dima/IsaacLab/logs/rsl_rl/crazyflie_steering_tracking",
        os.path.join(freshscheduler_root, "logs/rsl_rl/crazyflie_steering_tracking"),
    ]
    all_checkpoints = []
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            continue
        for run_dir in glob.glob(os.path.join(log_dir, "20*")):
            all_checkpoints.extend(glob.glob(os.path.join(run_dir, "model_*.pt")))
    if not all_checkpoints:
        raise FileNotFoundError(
            "No training checkpoints found. Searched:\n  - "
            + "\n  - ".join(log_dirs)
        )
    return max(all_checkpoints, key=os.path.getmtime)


def main():
    checkpoint_path = args_cli.checkpoint or find_latest_checkpoint()
    print(f"[INFO]: Using checkpoint: {checkpoint_path}")

    # Single-env piloting
    env_cfg = TrackSteeringEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    # ManagerBasedEnv.reset() otherwise does
    # `while SimulationManager.assets_loading(): self.sim.render()` waiting on
    # Kit's ASSETS_LOADING/ASSETS_LOADED stage events. That pair can race or
    # drop (observed: stuck forever, 0% GPU, on a scene streaming S3 assets)
    # — Isaac Lab's own tests disable this the same way (test_outdated_sensor.py).
    env_cfg.wait_for_textures = False

    # Disable automatic command resampling — we drive the command from the keyboard.
    # A very large window means the timer never fires during a session.
    env_cfg.commands.steering_command.resampling_time_range = (1.0e9, 1.0e9)

    agent_cfg = SteeringTrackingPPORunnerCfg()

    print("\n" + "=" * 70)
    print(f"MANUAL PILOTING — input_mode={args_cli.input_mode}")
    print("=" * 70)
    if args_cli.input_mode == "keyboard":
        print("  W / Up         target_velocity += {:.2f} m/s".format(args_cli.max_velocity))
        print("  S / Down       target_velocity -= {:.2f} m/s".format(args_cli.max_velocity))
        print("  A / Left       target_yaw_rate += {:.2f} rad/s  (turn left)".format(args_cli.max_yaw_rate))
        print("  D / Right      target_yaw_rate -= {:.2f} rad/s  (turn right)".format(args_cli.max_yaw_rate))
        print("  L              reset commands to 0 (hover)")
        print("Focus the Isaac Sim viewport for keys to register.")
    else:
        print(f"  target v slider:   [{-args_cli.max_velocity:+.2f}, {args_cli.max_velocity:+.2f}] m/s")
        print(f"  target ω slider:   [{-args_cli.max_yaw_rate:+.2f}, {args_cli.max_yaw_rate:+.2f}] rad/s")
        print("  Hover button       zero both commands")
    print("=" * 70 + "\n")

    env = gym.make("Isaac-Track-Steering-Vision-Crazyflie-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    # Isaac Lab's SimulationContext._app_control_on_stop_handle_fn spins on
    # `while not timeline.is_playing(): self.render()` when the app receives
    # a "stop" event (e.g. from simulation_app.close()) — the timeline never
    # resumes playing during shutdown, so this becomes an infinite loop that
    # wedges the process forever (0% GPU, one CPU core pegged). Isaac Lab's
    # own test suite disables this the same way ("prevent timeout").
    unwrapped_env.sim._disable_app_control_on_stop_handle = True

    # Input device. Se2Keyboard reports (vx, vy, omega_z); we use vx for forward
    # velocity and omega_z for yaw rate, and ignore vy. Slider state lives in a
    # mutable dict so the matplotlib widget callbacks can mutate it.
    keyboard = None
    slider_state = {"v": 0.0, "w": 0.0}
    if args_cli.input_mode == "keyboard":
        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args_cli.max_velocity,
                v_y_sensitivity=0.0,
                omega_z_sensitivity=args_cli.max_yaw_rate,
                sim_device=args_cli.device,
            )
        )
        print(keyboard)
    else:
        print("[INFO]: Slider input mode — drag the sliders in the FPV window.")

    # Build runner config and load policy.
    runner_cfg = {
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
        "algorithm": agent_cfg.algorithm.to_dict(),
        "actor": {
            "class_name": agent_cfg.actor.class_name,
            "hidden_dims": agent_cfg.actor.hidden_dims,
            "activation": agent_cfg.actor.activation,
            "obs_normalization": agent_cfg.actor.obs_normalization,
            "distribution_cfg": (
                agent_cfg.actor.distribution_cfg.to_dict() if agent_cfg.actor.distribution_cfg else None
            ),
        },
        "critic": {
            "class_name": agent_cfg.critic.class_name,
            "hidden_dims": agent_cfg.critic.hidden_dims,
            "activation": agent_cfg.critic.activation,
            "obs_normalization": agent_cfg.critic.obs_normalization,
        },
        "obs_groups": agent_cfg.obs_groups,
    }
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    # Grab the command term so we can overwrite its target tensors directly.
    steering_term = unwrapped_env.command_manager.get_term("steering_command")

    # FPV + command-history display setup
    has_fpv_camera = "fpv_camera" in unwrapped_env.scene.sensors
    show_fpv_plot = has_fpv_camera and not args_cli.headless and not args_cli.no_fpv_plot
    fpv_fig = None
    im_fpv = None
    overlay_text = None
    line_cmd_v = line_act_v = line_cmd_w = line_act_w = None
    ax_v = ax_w = None
    slider_v = slider_w = None
    hover_button = None

    # Rolling buffers for the command/state time-series. Time is plotted in
    # seconds derived from the env's control dt (dt * decimation).
    history_len = max(2, args_cli.history_window)
    control_dt = unwrapped_env.cfg.sim.dt * unwrapped_env.cfg.decimation
    t_buf = deque(maxlen=history_len)
    cmd_v_buf = deque(maxlen=history_len)
    act_v_buf = deque(maxlen=history_len)
    cmd_w_buf = deque(maxlen=history_len)
    act_w_buf = deque(maxlen=history_len)

    fpv_output_dir = None
    if has_fpv_camera and args_cli.save_fpv_frames:
        fpv_output_dir = "/tmp/fpv_pilot_frames"
        os.makedirs(fpv_output_dir, exist_ok=True)
        print(f"[INFO]: Saving FPV frames to {fpv_output_dir}")

    obs = env.get_observations()
    step = 0
    print("[INFO]: Starting pilot loop. Press Ctrl+C in this terminal to stop.\n")
    while simulation_app.is_running():
        # Read pilot input, write into the command term.
        if keyboard is not None:
            cmd = keyboard.advance()  # tensor: [vx, vy, omega_z]
            target_v = float(cmd[0].item())
            target_w = float(cmd[2].item())
        else:
            target_v = float(slider_state["v"])
            target_w = float(slider_state["w"])
        steering_term.target_velocity.fill_(target_v)
        steering_term.target_yaw_rate.fill_(target_w)

        # Run policy and step. RslRlVecEnvWrapper observations already reflect the
        # command we just wrote, since policy obs are computed *after* the previous
        # step; the next step will see the updated command in its reward/term logic.
        actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)

        # Telemetry
        robot = unwrapped_env.scene["robot"]
        height = robot.data.root_pos_w[0, 2].item()
        vel_x = robot.data.root_lin_vel_b[0, 0].item()
        yaw_rate = robot.data.root_ang_vel_b[0, 2].item()
        cmd_v = steering_term.target_velocity[0].item()
        cmd_w = steering_term.target_yaw_rate[0].item()

        # Append to rolling history (one sample per control step).
        t_buf.append(step * control_dt)
        cmd_v_buf.append(cmd_v)
        act_v_buf.append(vel_x)
        cmd_w_buf.append(cmd_w)
        act_w_buf.append(yaw_rate)

        if step % 20 == 0:
            print(
                f"  cmd v={cmd_v:+.2f} m/s  w={cmd_w:+.2f} rad/s  |  "
                f"actual v={vel_x:+.2f} m/s  w={yaw_rate:+.2f} rad/s  |  h={height:.2f} m"
            )

        if dones.any():
            print(f"  [reset] drone reset at step {step + 1} (probably crashed or timed out)")

        # FPV preview
        if has_fpv_camera and step % 5 == 0:
            try:
                camera = unwrapped_env.scene["fpv_camera"]
                rgb_data = camera.data.output["rgb"][0].cpu().numpy()
                # IsaacLab's "rgb" output is uint8 [0, 255] on this version. Older
                # paths returned float [0, 1]; handle both rather than blindly *255
                # (which would overflow uint8 and produce inverted/garbage colors).
                rgb3 = rgb_data[:, :, :3]
                if rgb3.dtype == np.uint8:
                    rgb_image = rgb3
                else:
                    rgb_image = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)

                overlay = (
                    f"cmd: v={cmd_v:+.2f} m/s   w={cmd_w:+.2f} rad/s\n"
                    f"act: v={vel_x:+.2f} m/s   w={yaw_rate:+.2f} rad/s\n"
                    f"h={height:.2f} m   step={step + 1}"
                )

                if show_fpv_plot:
                    if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                        fpv_fig = None
                        im_fpv = None
                        overlay_text = None
                        line_cmd_v = line_act_v = line_cmd_w = line_act_w = None
                        ax_v = ax_w = None
                        slider_v = slider_w = None
                        hover_button = None
                    if fpv_fig is None:
                        plt.ion()
                        fpv_fig = plt.figure(num="Drone FPV (manual pilot)", figsize=(13, 7))
                        # Leave space at the bottom for slider widgets in slider mode.
                        bottom_margin = 0.22 if args_cli.input_mode == "slider" else 0.08
                        gs = fpv_fig.add_gridspec(
                            2,
                            2,
                            width_ratios=[1.4, 1.0],
                            hspace=0.35,
                            wspace=0.25,
                            left=0.04,
                            right=0.98,
                            top=0.95,
                            bottom=bottom_margin,
                        )
                        ax_img = fpv_fig.add_subplot(gs[:, 0])
                        ax_v = fpv_fig.add_subplot(gs[0, 1])
                        ax_w = fpv_fig.add_subplot(gs[1, 1])

                        im_fpv = ax_img.imshow(rgb_image)
                        ax_img.axis("off")
                        overlay_text = ax_img.text(
                            10,
                            30,
                            overlay,
                            color="lime",
                            fontsize=10,
                            weight="bold",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                        )

                        (line_cmd_v,) = ax_v.plot([], [], color="tab:orange", lw=2, label="cmd")
                        (line_act_v,) = ax_v.plot([], [], color="tab:blue", lw=1.5, label="actual")
                        ax_v.set_ylabel("v (m/s)")
                        ax_v.set_title("forward velocity")
                        ax_v.grid(True, alpha=0.3)
                        ax_v.legend(loc="upper left", fontsize=8)
                        ax_v.axhline(0.0, color="k", lw=0.5, alpha=0.4)

                        (line_cmd_w,) = ax_w.plot([], [], color="tab:orange", lw=2, label="cmd")
                        (line_act_w,) = ax_w.plot([], [], color="tab:blue", lw=1.5, label="actual")
                        ax_w.set_ylabel("ω (rad/s)")
                        ax_w.set_xlabel("time (s)")
                        ax_w.set_title("yaw rate")
                        ax_w.grid(True, alpha=0.3)
                        ax_w.legend(loc="upper left", fontsize=8)
                        ax_w.axhline(0.0, color="k", lw=0.5, alpha=0.4)

                        if args_cli.input_mode == "slider":
                            ax_slider_v = fpv_fig.add_axes([0.10, 0.11, 0.78, 0.03])
                            ax_slider_w = fpv_fig.add_axes([0.10, 0.06, 0.78, 0.03])
                            ax_button = fpv_fig.add_axes([0.10, 0.005, 0.10, 0.04])

                            slider_v = Slider(
                                ax=ax_slider_v,
                                label="target v (m/s)",
                                valmin=-args_cli.max_velocity,
                                valmax=args_cli.max_velocity,
                                valinit=0.0,
                                valstep=0.05,
                                color="tab:orange",
                            )
                            slider_w = Slider(
                                ax=ax_slider_w,
                                label="target ω (rad/s)",
                                valmin=-args_cli.max_yaw_rate,
                                valmax=args_cli.max_yaw_rate,
                                valinit=0.0,
                                valstep=0.05,
                                color="tab:orange",
                            )

                            def _on_v(val, _state=slider_state):
                                _state["v"] = float(val)

                            def _on_w(val, _state=slider_state):
                                _state["w"] = float(val)

                            slider_v.on_changed(_on_v)
                            slider_w.on_changed(_on_w)

                            hover_button = Button(ax_button, "Hover (0,0)")

                            def _on_hover(event, sv=slider_v, sw=slider_w):
                                sv.set_val(0.0)
                                sw.set_val(0.0)

                            hover_button.on_clicked(_on_hover)
                    else:
                        im_fpv.set_data(rgb_image)
                        overlay_text.set_text(overlay)

                    t_arr = np.fromiter(t_buf, dtype=float)
                    line_cmd_v.set_data(t_arr, np.fromiter(cmd_v_buf, dtype=float))
                    line_act_v.set_data(t_arr, np.fromiter(act_v_buf, dtype=float))
                    line_cmd_w.set_data(t_arr, np.fromiter(cmd_w_buf, dtype=float))
                    line_act_w.set_data(t_arr, np.fromiter(act_w_buf, dtype=float))
                    if t_arr.size >= 2:
                        ax_v.set_xlim(t_arr[0], t_arr[-1])
                        ax_w.set_xlim(t_arr[0], t_arr[-1])
                    ax_v.relim()
                    ax_v.autoscale_view(scalex=False, scaley=True)
                    ax_w.relim()
                    ax_w.autoscale_view(scalex=False, scaley=True)

                    fpv_fig.canvas.draw_idle()
                    fpv_fig.canvas.flush_events()
                    plt.pause(0.001)

                if fpv_output_dir is not None:
                    import cv2

                    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    for i, line in enumerate(overlay.split("\n")):
                        cv2.putText(
                            bgr,
                            line,
                            (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    cv2.imwrite(f"{fpv_output_dir}/frame_{step:06d}.jpg", bgr)
            except Exception as e:
                if step == 0:
                    print(f"[WARNING]: FPV capture failed: {e}")

        step += 1

    if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
        plt.close(fpv_fig)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO]: Interrupted by user.")
    finally:
        simulation_app.close()
