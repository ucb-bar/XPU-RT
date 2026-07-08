#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the trained velocity tracker in the forest-trail env with DroNet driving.

Optionally runs YOLOv8-nano object detection on the same FPV feed and displays
detection results in an additional matplotlib panel.

Usage:
    # DroNet only
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet.py \\
        --dronet_weights logs/dronet/<run>/best.pt

    # DroNet + YOLOv8-nano
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet.py \\
        --dronet_weights logs/dronet/<run>/best.pt --yolo

    # YOLO with custom class filter and confidence
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet.py \\
        --dronet_weights logs/dronet/<run>/best.pt --yolo \\
        --yolo_conf 0.3 --yolo_classes 0 1 3 16 24
"""

import argparse
import glob
import os
import sys
from collections import deque

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
isaaclab_root = os.path.join(freshscheduler_root, "sims/IsaacLab/source")
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_rl"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_contrib"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run DroNet in the forest-trail env.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Velocity-tracker (rsl_rl) checkpoint. Default: latest under logs/.")
parser.add_argument("--dronet_weights", type=str, required=True,
                    help="Path to a DroNet state_dict.pt (e.g. logs/dronet/<run>/best.pt).")
parser.add_argument("--dronet_size", choices=["small", "large"], default="small",
                    help="DroNet variant — must match what was trained ('small' = 112x112).")
parser.add_argument("--forward_velocity", type=float, default=1.0,
                    help=(
                        "Constant target_velocity fed to the inner-loop (m/s). Default 1.0 — "
                        "top of the velocity tracker's training range. Values >1.0 are "
                        "out-of-distribution: the inner-loop policy tends to pitch hard "
                        "forward at startup (which makes the FPV point at the ground) and "
                        "may saturate thrust / drift in altitude. Push higher only as an "
                        "explicit OOD test."
                    ))
parser.add_argument("--omega_clamp", type=float, default=1.5,
                    help=(
                        "Hard clamp on DroNet's predicted yaw rate (rad/s). Default 1.5 "
                        "covers training (±1.0) plus headroom."
                    ))
parser.add_argument("--camera_update_period", type=float, default=0.1,
                    help=(
                        "FPV camera update period (s). Default 0.1 (10 Hz) matches the "
                        "scene's inherited setting and the scheduled script."
                    ))
parser.add_argument("--trail", choices=["straight", "curved"], default="straight",
                    help=(
                        "Trail geometry.  'straight' (default): classic axis-aligned corridor. "
                        "'curved': procedural polyline whose turns are controlled by the "
                        "--max_turn_deg / --num_segments / --segment_length / --curvature_seed flags."
                    ))
parser.add_argument("--max_turn_deg", type=float, default=None,
                    help=(
                        "Curved-trail only: max heading change per segment in degrees. "
                        "Default 35 (set in CurvedTrailLayout). Try 10-15 for gentle, "
                        "45-60 for aggressive turns."
                    ))
parser.add_argument("--num_segments", type=int, default=None,
                    help="Curved-trail only: number of polyline segments. Default 8.")
parser.add_argument("--segment_length", type=float, default=None,
                    help="Curved-trail only: length of each segment in metres. Default 4.")
parser.add_argument("--curvature_seed", type=int, default=None,
                    help=(
                        "Curved-trail only: RNG seed for the procedural turn sequence. "
                        "Sweep this to generate different trail shapes from the same params. "
                        "Default 1337."
                    ))
parser.add_argument("--episode_length_s", type=float, default=120.0,
                    help=(
                        "Per-episode time-out in seconds.  Default 120 s — much longer than "
                        "the env's built-in 10 s so the drone can fly a full curved trail "
                        "without getting reset mid-flight.  Set to a small value (e.g. 5 s) "
                        "to revert to the original training-style episode length."
                    ))
parser.add_argument("--no_episode_timeout", action="store_true",
                    help=(
                        "Disable the time_out termination entirely.  The drone will only "
                        "reset on crash / off-trail.  Useful for indefinite demo recordings."
                    ))
parser.add_argument("--no_humans", action="store_true",
                    help=(
                        "Disable the procedural humans on the trail. By default this script "
                        "uses the with-humans variant so DroNet sees humanoid figures along "
                        "the centerline as additional non-tree obstacles."
                    ))
parser.add_argument("--no_fpv_plot", action="store_true",
                    help="Do not open matplotlib window for live FPV display.")
parser.add_argument("--history_window", type=int, default=200,
                    help="Number of recent control steps to keep in the time-series plot.")
# ── YOLOv8 args ──────────────────────────────────────────────────────────────
parser.add_argument("--yolo", action="store_true",
                    help=(
                        "Run YOLOv8-nano object detection on each FPV frame and show "
                        "results in an extra matplotlib panel. Requires: pip install ultralytics"
                    ))
parser.add_argument("--yolo_conf", type=float, default=0.25,
                    help="YOLOv8 confidence threshold. Default 0.25.")
parser.add_argument("--yolo_classes", type=int, nargs="+",
                    default=[0, 1, 3, 16, 24],
                    help=(
                        "COCO class IDs to report. Defaults cover the most likely "
                        "forest-trail encounters: person=0, bicycle=1, motorcycle=3, "
                        "dog=16, backpack=24."
                    ))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

# Register tasks
from sims.isaaclab_tasks.forest_trail.config import crazyflie as _forest_register  # noqa: F401
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie as _track_register  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from qnn_models.dronet import DronetTorch

from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (
    SteeringTrackingPPORunnerCfg,
)
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (
    ForestTrailEnvCfg_PLAY,
    ForestTrailEnvCfg_PLAY_WithHumans,
    ForestTrailEnvCfg_Curved_PLAY,
    ForestTrailEnvCfg_Curved_PLAY_WithHumans,
    make_curved_env_cfg,
)
from sims.isaaclab_tasks.forest_trail.tree_layout import CurvedTrailLayout

# ── COCO class labels for the default filter set ──────────────────────────────
_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
    17: "horse", 24: "backpack", 26: "handbag", 28: "umbrella",
}

# Per-class colours for the detection bar chart (class_id → hex colour).
_CLASS_COLOURS = {
    0: "#e74c3c",   # person    — red
    1: "#3498db",   # bicycle   — blue
    3: "#e67e22",   # motorcycle— orange
    16: "#2ecc71",  # dog       — green
    24: "#9b59b6",  # backpack  — purple
}
_DEFAULT_COLOUR = "#95a5a6"


def find_latest_checkpoint() -> str:
    log_dirs = [
        "/scratch2/dima/IsaacLab/logs/rsl_rl/crazyflie_steering_tracking",
        os.path.join(freshscheduler_root, "logs/rsl_rl/crazyflie_steering_tracking"),
    ]
    cks = []
    for d in log_dirs:
        if os.path.exists(d):
            for run in glob.glob(os.path.join(d, "20*")):
                cks.extend(glob.glob(os.path.join(run, "model_*.pt")))
    if not cks:
        raise FileNotFoundError(
            "No velocity-tracker checkpoint found. Pass --checkpoint explicitly."
        )
    return max(cks, key=os.path.getmtime)


def preprocess_for_dronet(rgb_uint8: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    """``(H, W, 3) uint8 RGB`` → ``(1, 3, S, S) float32`` in [0, 1] on device."""
    img = torch.from_numpy(rgb_uint8).to(device, non_blocking=True)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return img


def _b(msg: str) -> None:
    """Breadcrumb print with explicit flush; helps localize silent exits."""
    print(msg, flush=True)


def main():
    inner_ckpt = args_cli.checkpoint or find_latest_checkpoint()
    _b(f"[info] inner-loop (velocity tracker) checkpoint: {inner_ckpt}")
    _b(f"[info] DroNet weights: {args_cli.dronet_weights}")

    # ── YOLOv8-nano (optional) ────────────────────────────────────────────────
    yolo_model = None
    if args_cli.yolo:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            yolo_model = YOLO("yolov8n.pt")
            yolo_model.to(args_cli.device)
            _b(f"[info] YOLOv8-nano loaded (conf={args_cli.yolo_conf}, "
               f"classes={args_cli.yolo_classes})")
        except ImportError:
            _b("[warn] 'ultralytics' not installed — YOLO disabled.")
            _b("       pip install ultralytics")

    # ── env (forest, num_envs=1) ──────────────────────────────────────────────
    use_humans = not args_cli.no_humans
    curved = args_cli.trail == "curved"
    has_curve_override = curved and any(
        v is not None for v in (
            args_cli.max_turn_deg, args_cli.num_segments,
            args_cli.segment_length, args_cli.curvature_seed,
        )
    )
    if curved:
        task_id = (
            "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-v0" if use_humans
            else "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-v0"
        )
        if has_curve_override:
            layout = CurvedTrailLayout()
            if args_cli.max_turn_deg is not None:
                layout.max_turn_deg = args_cli.max_turn_deg
            if args_cli.num_segments is not None:
                layout.num_segments = args_cli.num_segments
            if args_cli.segment_length is not None:
                layout.segment_length = args_cli.segment_length
            if args_cli.curvature_seed is not None:
                layout.seed = args_cli.curvature_seed
            _b(f"[bc] building custom curved env "
               f"(turns≤{layout.max_turn_deg}°, {layout.num_segments}×{layout.segment_length}m, "
               f"seed={layout.seed}, humans={use_humans})")
            env_cfg = make_curved_env_cfg(layout, with_humans=use_humans, play=True, num_envs=1)
        else:
            cfg_cls = ForestTrailEnvCfg_Curved_PLAY_WithHumans if use_humans else ForestTrailEnvCfg_Curved_PLAY
            _b(f"[bc] building {cfg_cls.__name__} (humans={use_humans})")
            env_cfg = cfg_cls()
    else:
        cfg_cls = ForestTrailEnvCfg_PLAY_WithHumans if use_humans else ForestTrailEnvCfg_PLAY
        task_id = (
            "Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0" if use_humans
            else "Isaac-Forest-Trail-Vision-Crazyflie-Play-v0"
        )
        _b(f"[bc] building {cfg_cls.__name__} (humans={use_humans})")
        env_cfg = cfg_cls()
    env_cfg.sim.device = args_cli.device
    # ManagerBasedEnv.reset() otherwise does
    # `while SimulationManager.assets_loading(): self.sim.render()` waiting on
    # Kit's ASSETS_LOADING/ASSETS_LOADED stage events. That pair can race or
    # drop (observed: stuck forever, 0% GPU, on a scene streaming S3 assets)
    # — Isaac Lab's own tests disable this the same way (test_outdated_sensor.py).
    env_cfg.wait_for_textures = False
    env_cfg.commands.steering_command.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.scene.fpv_camera.update_period = args_cli.camera_update_period
    # Stretch (or remove) the episode time-out so a single rollout can cover a
    # full long/curved trail.  The env defaults to 10 s for training; demo
    # runs typically want much more.
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_episode_timeout:
        # Drop the time_out term so the drone never resets on a timer.
        # Only crash / off_trail will end an episode.
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        _b("[info] episode time_out termination disabled — only crash/off_trail will reset")
    else:
        _b(f"[info] episode_length_s = {env_cfg.episode_length_s:.1f} s")

    _b(f"[bc] gym.make {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    _b("[bc] gym.make returned")
    env = RslRlVecEnvWrapper(env)
    _b("[bc] RslRlVecEnvWrapper applied")
    unwrapped_env = env.unwrapped
    # Isaac Lab's SimulationContext._app_control_on_stop_handle_fn spins on
    # `while not timeline.is_playing(): self.render()` when the app receives
    # a "stop" event (e.g. from simulation_app.close()) — the timeline never
    # resumes playing during shutdown, so this becomes an infinite loop that
    # wedges the process forever (0% GPU, one CPU core pegged). Isaac Lab's
    # own test suite disables this the same way ("prevent timeout").
    unwrapped_env.sim._disable_app_control_on_stop_handle = True

    # ── inner-loop policy ─────────────────────────────────────────────────────
    agent_cfg = SteeringTrackingPPORunnerCfg()
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
                agent_cfg.actor.distribution_cfg.to_dict()
                if agent_cfg.actor.distribution_cfg else None
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
    _b("[bc] constructing OnPolicyRunner")
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device)
    _b("[bc] loading inner-loop checkpoint")
    runner.load(inner_ckpt)
    inner_policy = runner.get_inference_policy(device=args_cli.device)
    _b("[bc] inner policy ready")

    # ── DroNet ────────────────────────────────────────────────────────────────
    img_size = 112 if args_cli.dronet_size == "small" else 224
    dronet = DronetTorch(
        img_dims=(img_size, img_size),
        img_channels=3,
        output_dim=1,
        small=(args_cli.dronet_size == "small"),
    ).to(args_cli.device)
    state = torch.load(args_cli.dronet_weights, map_location=args_cli.device, weights_only=True)
    dronet.load_state_dict(state, strict=True)
    dronet.eval()
    _b(f"[info] DroNet loaded ({args_cli.dronet_size}, input {img_size}x{img_size})")

    # ── Hooks into command term + camera ──────────────────────────────────────
    _b("[bc] resolving steering_command term")
    steering_term = unwrapped_env.command_manager.get_term("steering_command")
    has_fpv = "fpv_camera" in unwrapped_env.scene.sensors
    _b(f"[bc] has_fpv={has_fpv}; sensors={list(unwrapped_env.scene.sensors.keys())}")
    if not has_fpv:
        raise RuntimeError("Forest env has no fpv_camera in scene; cannot run DroNet.")

    # ── Plot state ────────────────────────────────────────────────────────────
    use_yolo = yolo_model is not None
    show_plot = not args_cli.headless and not args_cli.no_fpv_plot
    fpv_fig = None
    im_fpv = None
    overlay_text = None
    line_cmd_v = line_act_v = line_cmd_w = line_act_w = None
    ax_v = ax_w = None
    ax_det = ax_det_hist = None   # YOLO panels (None when YOLO is off)

    history_len = max(2, args_cli.history_window)
    control_dt = unwrapped_env.cfg.sim.dt * unwrapped_env.cfg.decimation
    t_buf = deque(maxlen=history_len)
    cmd_v_buf = deque(maxlen=history_len)
    act_v_buf = deque(maxlen=history_len)
    cmd_w_buf = deque(maxlen=history_len)
    act_w_buf = deque(maxlen=history_len)
    person_conf_buf = deque(maxlen=history_len)   # max person confidence per step

    # Last YOLO result (updated every step, displayed every 5 steps)
    last_yolo_results = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    _b("[bc] env.get_observations() ...")
    obs = env.get_observations()
    _b(f"[bc] obs ready; simulation_app.is_running()={simulation_app.is_running()}")
    step = 0
    _b("[info] starting forest pilot loop. Ctrl+C to stop.")

    while simulation_app.is_running():
        # 1) Read FPV camera
        camera = unwrapped_env.scene["fpv_camera"]
        rgb_data = camera.data.output["rgb"][0].cpu().numpy()
        rgb3 = rgb_data[:, :, :3]
        if rgb3.dtype != np.uint8:
            rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)

        # 2) DroNet inference
        with torch.no_grad():
            x = preprocess_for_dronet(rgb3, img_size, args_cli.device)
            steer_pred, _collision = dronet(x)
            target_w = float(steer_pred.item())
        target_w = max(-args_cli.omega_clamp, min(args_cli.omega_clamp, target_w))
        target_v = float(args_cli.forward_velocity)

        # 3) YOLOv8 inference (runs every step, display throttled below)
        max_person_conf = 0.0
        if use_yolo:
            last_yolo_results = yolo_model(
                rgb3,
                conf=args_cli.yolo_conf,
                classes=args_cli.yolo_classes,
                verbose=False,
            )
            boxes = last_yolo_results[0].boxes
            if len(boxes):
                for cid, conf in zip(boxes.cls.cpu().int().tolist(),
                                     boxes.conf.cpu().tolist()):
                    if cid == 0:
                        max_person_conf = max(max_person_conf, conf)

        # 4) Push command into the steering term
        steering_term.target_velocity.fill_(target_v)
        steering_term.target_yaw_rate.fill_(target_w)

        # 5) Inner-loop policy step
        actions = inner_policy(obs)
        obs, rewards, dones, info = env.step(actions)

        # 6) Telemetry
        robot = unwrapped_env.scene["robot"]
        height = robot.data.root_pos_w[0, 2].item()
        vel_x = robot.data.root_lin_vel_b[0, 0].item()
        yaw_rate = robot.data.root_ang_vel_b[0, 2].item()
        x_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[0].item()
        y_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[1].item()

        t_buf.append(step * control_dt)
        cmd_v_buf.append(target_v)
        act_v_buf.append(vel_x)
        cmd_w_buf.append(target_w)
        act_w_buf.append(yaw_rate)
        person_conf_buf.append(max_person_conf)

        if step % 20 == 0:
            det_str = ""
            if use_yolo and last_yolo_results is not None:
                n = len(last_yolo_results[0].boxes)
                det_str = f"  yolo_n={n}  person_conf={max_person_conf:.2f}"
            print(f"  step {step:5d}  pos=({x_local:+.2f},{y_local:+.2f})m  h={height:.2f}m  "
                  f"cmd w={target_w:+.2f}  act w={yaw_rate:+.2f}  v={vel_x:+.2f}{det_str}")

        if dones.any():
            print(f"  [reset] step {step + 1}: drone reset (off-trail / crash / timeout)")

        # 7) Live plot (every 5 steps)
        if show_plot and step % 5 == 0:
            try:
                if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                    fpv_fig = None

                if fpv_fig is None:
                    plt.ion()
                    if use_yolo:
                        fpv_fig = plt.figure(
                            num="DroNet Forest Pilot + YOLOv8", figsize=(19, 6))
                        gs = fpv_fig.add_gridspec(
                            2, 3, width_ratios=[1.4, 1.0, 1.0],
                            hspace=0.45, wspace=0.30,
                            left=0.04, right=0.98, top=0.95, bottom=0.08)
                        ax_img  = fpv_fig.add_subplot(gs[:, 0])
                        ax_v    = fpv_fig.add_subplot(gs[0, 1])
                        ax_w    = fpv_fig.add_subplot(gs[1, 1])
                        ax_det  = fpv_fig.add_subplot(gs[0, 2])
                        ax_det_hist = fpv_fig.add_subplot(gs[1, 2])
                    else:
                        fpv_fig = plt.figure(
                            num="DroNet Forest Pilot", figsize=(13, 6))
                        gs = fpv_fig.add_gridspec(
                            2, 2, width_ratios=[1.4, 1.0],
                            hspace=0.35, wspace=0.25,
                            left=0.04, right=0.98, top=0.95, bottom=0.08)
                        ax_img = fpv_fig.add_subplot(gs[:, 0])
                        ax_v   = fpv_fig.add_subplot(gs[0, 1])
                        ax_w   = fpv_fig.add_subplot(gs[1, 1])

                    im_fpv = ax_img.imshow(rgb3)
                    ax_img.axis("off")
                    overlay_text = ax_img.text(
                        10, 30, "",
                        color="lime", fontsize=10, weight="bold",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                    )

                    (line_cmd_v,) = ax_v.plot([], [], color="tab:orange", lw=2, label="cmd")
                    (line_act_v,) = ax_v.plot([], [], color="tab:blue",   lw=1.5, label="actual")
                    ax_v.set_ylabel("v (m/s)")
                    ax_v.set_title("forward velocity")
                    ax_v.grid(True, alpha=0.3)
                    ax_v.legend(loc="upper left", fontsize=8)
                    ax_v.axhline(0.0, color="k", lw=0.5, alpha=0.4)

                    (line_cmd_w,) = ax_w.plot([], [], color="tab:orange", lw=2,   label="cmd (DroNet)")
                    (line_act_w,) = ax_w.plot([], [], color="tab:blue",   lw=1.5, label="actual")
                    ax_w.set_ylabel("ω (rad/s)")
                    ax_w.set_xlabel("time (s)")
                    ax_w.set_title("yaw rate")
                    ax_w.grid(True, alpha=0.3)
                    ax_w.legend(loc="upper left", fontsize=8)
                    ax_w.axhline(0.0, color="k", lw=0.5, alpha=0.4)

                # ── update FPV image ──────────────────────────────────────────
                if use_yolo and last_yolo_results is not None:
                    # plot() preserves input channel order; input is RGB so output is RGB
                    display_frame = last_yolo_results[0].plot()
                else:
                    display_frame = rgb3
                im_fpv.set_data(display_frame)

                # ── DroNet telemetry overlay ──────────────────────────────────
                overlay_text.set_text(
                    f"DroNet ω={target_w:+.2f}  v_cmd={target_v:+.2f}\n"
                    f"actual ω={yaw_rate:+.2f}  v={vel_x:+.2f}\n"
                    f"pos=({x_local:+.2f},{y_local:+.2f})m  h={height:.2f}m"
                )

                # ── velocity / yaw-rate time series ───────────────────────────
                t_arr = np.fromiter(t_buf, dtype=float)
                line_cmd_v.set_data(t_arr, np.fromiter(cmd_v_buf, dtype=float))
                line_act_v.set_data(t_arr, np.fromiter(act_v_buf, dtype=float))
                line_cmd_w.set_data(t_arr, np.fromiter(cmd_w_buf, dtype=float))
                line_act_w.set_data(t_arr, np.fromiter(act_w_buf, dtype=float))
                if t_arr.size >= 2:
                    ax_v.set_xlim(t_arr[0], t_arr[-1])
                    ax_w.set_xlim(t_arr[0], t_arr[-1])
                ax_v.relim(); ax_v.autoscale_view(scalex=False, scaley=True)
                ax_w.relim(); ax_w.autoscale_view(scalex=False, scaley=True)

                # ── YOLO panels (only when active) ────────────────────────────
                if use_yolo and ax_det is not None and last_yolo_results is not None:
                    _update_detection_bar(ax_det, last_yolo_results[0],
                                          args_cli.yolo_conf, step)
                    _update_person_history(ax_det_hist, t_arr,
                                           np.fromiter(person_conf_buf, dtype=float),
                                           args_cli.yolo_conf)

                fpv_fig.canvas.draw_idle()
                fpv_fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception as e:
                if step == 0:
                    print(f"[warn] matplotlib FPV preview failed: {e}")

        step += 1

    if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
        plt.close(fpv_fig)
    env.close()


# ── YOLO plot helpers ─────────────────────────────────────────────────────────

def _update_detection_bar(ax, result, conf_thresh: float, step: int):
    """Horizontal bar chart: one bar per detected class, width = max confidence."""
    ax.cla()
    boxes = result.boxes
    if len(boxes):
        cls_ids = boxes.cls.cpu().int().tolist()
        confs   = boxes.conf.cpu().tolist()
        # Keep max confidence per class
        best: dict[int, float] = {}
        for cid, conf in zip(cls_ids, confs):
            if conf > best.get(cid, 0.0):
                best[cid] = conf
        # Sort descending by confidence
        sorted_items = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        labels = [_COCO_NAMES.get(cid, f"cls{cid}") for cid, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colours = [_CLASS_COLOURS.get(cid, _DEFAULT_COLOUR) for cid, _ in sorted_items]
        ax.barh(labels, values, color=colours, alpha=0.80, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5, "no detections", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_xlim(0, 1)

    ax.axvline(conf_thresh, color="gray", linestyle="--", lw=0.8, alpha=0.7, label=f"thresh={conf_thresh}")
    ax.set_xlabel("confidence")
    ax.set_title(f"YOLO detections  (step {step})", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()


def _update_person_history(ax, t_arr: np.ndarray, person_conf: np.ndarray, conf_thresh: float):
    """Rolling time-series of max person-detection confidence."""
    ax.cla()
    if t_arr.size >= 2:
        ax.plot(t_arr, person_conf, color=_CLASS_COLOURS[0], lw=1.5, label="person conf")
        ax.fill_between(t_arr, person_conf, alpha=0.15, color=_CLASS_COLOURS[0])
        ax.set_xlim(t_arr[0], t_arr[-1])
    ax.axhline(conf_thresh, color="gray", linestyle="--", lw=0.8, alpha=0.7,
               label=f"thresh={conf_thresh}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("confidence")
    ax.set_xlabel("time (s)")
    ax.set_title("person detection history", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] interrupted by user", flush=True)
    except Exception:
        print("[error] uncaught exception in main():", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[bc] entering finally; closing simulation_app", flush=True)
        simulation_app.close()
