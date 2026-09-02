#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run trained DroNet + velocity-tracker MLP in the forest-trail env, gated by an XPURT schedule.

What's combined here vs the standalone runners:
- Forest-trail env (straight or curved) — the trail-with-trees scene we built
  so DroNet's image input has actual structure.  ``--trail straight`` (default)
  uses the axis-aligned corridor; ``--trail curved`` switches to the procedural
  polyline with the same curvature flags as ``pilot_forest_with_dronet.py``
  (``--max_turn_deg`` / ``--num_segments`` / ``--segment_length`` /
  ``--curvature_seed``).
- Trained DroNet (steering head only) — the ``best.pt`` from
  ``sims/training/train_dronet.py``. The collision head was left at random init
  in v1, so by default we do *not* use it; ``target_velocity`` is fixed via
  ``--forward_velocity`` instead. Pass ``--use_collision_modulation`` to revert
  to the original ``play_dronet_mlp_scheduled.py`` behaviour
  (``target_velocity = (1 − coll) · max_velocity``) once the collision head is
  actually trained.
- Trained MLP velocity tracker — the existing rsl_rl checkpoint.
- XPURT schedule replay — DroNet runs only during its scheduled dispatches and
  the MLP runs only during its scheduled dispatches; outside those windows we
  hold the last commanded yaw rate / last action (zero-order hold).

Usage:

    # Straight trail (default), humans on, schedule replay
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \\
        --dronet_weights logs/dronet/2026-04-27_17-10-41/best.pt

    # Curved trail, custom curvature + seed, no humans
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \\
        --dronet_weights logs/dronet/2026-04-27_17-10-41/best.pt \\
        --trail curved --max_turn_deg 45 --curvature_seed 7 --no_humans
"""

import argparse
import glob
import atexit
import os
import signal
import sys
from collections import defaultdict, deque
from pathlib import Path

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
isaaclab_root = os.path.join(freshscheduler_root, "sims/IsaacLab/source")
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_rl"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_contrib"))

from isaaclab.app import AppLauncher
from sims.scripts.utils.schedule_dispatch import (
    did_model_complete, is_model_active, load_schedule, load_schedule_by_hw,
    load_schedule_jobs,
)

_DEFAULT_SCHEDULE = Path(freshscheduler_root) / "schedules" / "scheduled_networks_mlp_control_dronet_firesim_het_profiled.json"

parser = argparse.ArgumentParser(
    description="Scheduled DroNet + MLP runner in the forest-trail env.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Velocity-tracker (rsl_rl) checkpoint. Default: latest under logs/.")
parser.add_argument("--dronet_weights", type=str, required=True,
                    help="DroNet state_dict.pt (e.g. logs/dronet/<run>/best.pt).")
parser.add_argument("--dronet_size", choices=["small", "large"], default="small",
                    help="Must match what the DroNet checkpoint was trained as.")
parser.add_argument("--schedule_json", type=Path, default=_DEFAULT_SCHEDULE,
                    help="Schedule JSON (XPURT profiler output).")
parser.add_argument("--schedule_time_unit", choices=("ms", "s"), default="ms",
                    help="Unit for start_time / duration / makespan in the schedule JSON.")
parser.add_argument("--dronet_model_name", type=str, default=None,
                    help=(
                        "Normalized model-name key (post trailing-digit stripping) for the "
                        "DroNet dispatches in the schedule. Default: autodetected from the "
                        "loaded schedule (any key containing 'dronet')."
                    ))
parser.add_argument("--mlp_model_name", type=str, default=None,
                    help=(
                        "Normalized model-name key for the MLP velocity-tracker dispatches. "
                        "Default: autodetected from the loaded schedule (any key containing "
                        "'mlp'). Newer schedules use 'mlp_control', older ones use 'mlp'. "
                        "If the schedule has no MLP-like key at all, the MLP policy runs "
                        "every control step (ungated)."
                    ))
parser.add_argument("--yolo_model_name", type=str, default=None,
                    help=(
                        "Normalized model-name key for the YOLOv8 dispatches (e.g. "
                        "'yolov8_nano'). Default: autodetected from the loaded schedule "
                        "(any key containing 'yolo'). Used together with --yolo."
                    ))
parser.add_argument("--num_periods", type=int, default=0,
                    help=(
                        "Number of full schedule periods to run.  Default 0 means "
                        "run indefinitely until Ctrl+C or window close — useful for "
                        "long demo rollouts.  Set to a positive integer for a fixed "
                        "(reproducible) run length."
                    ))
parser.add_argument("--episode_length_s", type=float, default=120.0,
                    help=(
                        "Per-episode time-out in seconds.  Default 120 s so a single "
                        "rollout can cover a full curved trail without resetting on the "
                        "env's 10 s training-style timer."
                    ))
parser.add_argument("--no_episode_timeout", action="store_true",
                    help=(
                        "Disable the time_out termination entirely — the drone resets only "
                        "on crash / off-trail.  Useful for indefinite demo recordings."
                    ))
parser.add_argument("--forward_velocity", type=float, default=1.0,
                    help=(
                        "Fixed target_velocity (m/s) when --use_collision_modulation is off. "
                        "Default 1.0 — top of the velocity tracker's training range. "
                        "Push higher (e.g. 2.0) for an OOD test, but expect altitude "
                        "drift / thrust saturation."
                    ))
parser.add_argument("--use_collision_modulation", action="store_true",
                    help=(
                        "Use DroNet's collision head to modulate velocity "
                        "(target_velocity = (1 - coll) * max_velocity). Off by default "
                        "because our trained DroNet has the collision head still at init."
                    ))
parser.add_argument("--max_velocity", type=float, default=2.0,
                    help="Used only when --use_collision_modulation is set.")
parser.add_argument("--omega_clamp", type=float, default=1.5,
                    help="Hard clamp on DroNet's predicted yaw rate (rad/s).")
parser.add_argument("--sim_dt", type=float, default=None,
                    help=(
                        "Override the physics/control time step (s). Default is auto-tuned "
                        "from the schedule's shortest perception job (~4ms for typical "
                        "configs). Set explicitly to e.g. 0.002 for 2ms steps / 500Hz."
                    ))
parser.add_argument("--camera_update_period", type=float, default=0.1,
                    help=(
                        "FPV camera update period (s). Default 0.1 (10 Hz) matches the "
                        "scene's inherited setting and the direct (non-scheduled) script. "
                        "The original play_dronet_mlp_scheduled.py used 0.001 (1 kHz) on "
                        "the lighter track_steering scene — on the forest scene's many "
                        "prims that 1 kHz request can interact poorly with rendering and "
                        "destabilize the inner-loop policy. Bump it back up only if you've "
                        "verified your scene can sustain it."
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
# --- YOLOv8 inference args (only relevant when the schedule includes yolo) ---
parser.add_argument("--yolo", action="store_true",
                    help=(
                        "Run YOLOv8-nano object detection, gated by the schedule's yolo "
                        "dispatches (or every tick if the schedule has no yolo key but you "
                        "still want detections).  Requires: pip install ultralytics."
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
parser.add_argument("--no_humans", action="store_true",
                    help=(
                        "Disable the procedural humans on the trail.  Humans are on by "
                        "default to match pilot_forest_with_dronet.py; use this flag for a "
                        "tree-only scene that matches the older scheduled-pilot behaviour."
                    ))
parser.add_argument("--no_fpv_plot", action="store_true",
                    help="Don't open the matplotlib visualization window.")
parser.add_argument("--save_video", type=str, default=None,
                    help=(
                        "Path to write a video of the matplotlib panel (FPV + DroNet input + "
                        "schedule timeline + telemetry). Extension picks the format: '.mp4' "
                        "uses ffmpeg/libx264, '.gif' produces an animated GIF. Requires "
                        "matplotlib visualization (i.e. not --no_fpv_plot, not --headless)."
                    ))
parser.add_argument("--video_fps", type=float, default=None,
                    help=(
                        "Frames per second in the output video. Default: auto-computed "
                        "for real-time playback (1s sim = 1s video), i.e. "
                        "1 / (video_capture_skip × control_dt). Set explicitly to "
                        "speed up or slow down playback."
                    ))
parser.add_argument("--video_capture_skip", type=int, default=5,
                    help="Capture every Nth control step. Default 5.")
parser.add_argument("--video_flush_periods", type=int, default=20,
                    help="Flush video to disk every N periods so progress is viewable. Default 20.")
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

# Register both task families (forest reuses the track_steering agent cfg).
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

# Model→colour mapping for the schedule timeline.  We pick by substring on
# the normalized model name (matching the autodetect logic) so renaming a
# schedule's keys doesn't break the colours.  Falls back to a stable default.
_MODEL_COLOUR_RULES = (
    ("dronet", "orange"),
    ("yolo",   "#e74c3c"),
    ("mlp",    "skyblue"),
)
_MODEL_DEFAULT_COLOUR = "#bdc3c7"


def _colour_for_model(model: str) -> str:
    m = model.lower()
    for needle, colour in _MODEL_COLOUR_RULES:
        if needle in m:
            return colour
    return _MODEL_DEFAULT_COLOUR


def _b(msg: str) -> None:
    """Breadcrumb print with explicit flush."""
    print(msg, flush=True)


def find_latest_checkpoint() -> str:
    log_dirs = [
        os.path.join(os.environ.get("ISAACLAB_ROOT", "/scratch2/dima/IsaacLab"),
                     "logs", "rsl_rl", "crazyflie_steering_tracking"),
        os.path.join(freshscheduler_root, "logs/rsl_rl/crazyflie_steering_tracking"),
    ]
    cks = []
    for d in log_dirs:
        if os.path.exists(d):
            for run in glob.glob(os.path.join(d, "20*")):
                cks.extend(glob.glob(os.path.join(run, "model_*.pt")))
    if not cks:
        raise FileNotFoundError("No velocity-tracker checkpoint found. Pass --checkpoint explicitly.")
    return max(cks, key=os.path.getmtime)


def preprocess_for_dronet(rgb_uint8: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    """``(H, W, 3) uint8 RGB`` → ``(1, 3, S, S) float32`` in [0, 1] on device."""
    img = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device, non_blocking=True)
    img = img.float().div_(255.0)
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return img


_video_writer_ref = None
_video_path_ref = None
_video_period_ref = 0
_video_finalized = False


def _finalize_video(writer, path, period_idx):
    """Close the video writer and rename the in-progress file to a final chunk."""
    global _video_finalized
    if _video_finalized or writer is None:
        return
    _video_finalized = True
    try:
        writer.close()
    except Exception:
        pass
    if path is None:
        return
    base, ext = os.path.splitext(path)
    final_path = f"{base}_p{period_idx}{ext}"
    if os.path.isfile(path):
        os.rename(path, final_path)
        print(f"[info] video written to {final_path}", flush=True)
    else:
        print(f"[info] video finalized (no in-progress file to rename)", flush=True)


def _atexit_flush():
    _finalize_video(_video_writer_ref, _video_path_ref, _video_period_ref)


atexit.register(_atexit_flush)


def _sigint_handler(sig, frame):
    print("\n[info] SIGINT caught — flushing video...", flush=True)
    _finalize_video(_video_writer_ref, _video_path_ref, _video_period_ref)
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _sigint_handler)


def main():
    global _video_writer_ref, _video_path_ref, _video_period_ref
    inner_ckpt = args_cli.checkpoint or find_latest_checkpoint()
    _b(f"[info] inner-loop (velocity tracker) checkpoint: {inner_ckpt}")
    _b(f"[info] DroNet weights: {args_cli.dronet_weights}")
    _b(f"[info] schedule:       {args_cli.schedule_json}")

    # --- schedule (loaded early so we can auto-tune decimation) ---
    period, model_dispatches = load_schedule(
        args_cli.schedule_json, time_unit=args_cli.schedule_time_unit
    )
    _, hw_dispatches = load_schedule_by_hw(
        args_cli.schedule_json, time_unit=args_cli.schedule_time_unit
    )
    _, job_intervals = load_schedule_jobs(
        args_cli.schedule_json, time_unit=args_cli.schedule_time_unit
    )
    _b(f"[info] schedule: period={period * 1000:.3f}ms")
    for model, intervals in model_dispatches.items():
        total = sum(e - s for s, e in intervals)
        _b(f"[info]   {model}: {len(intervals)} dispatches, {total * 1000:.3f}ms total per period")
    for hw, intervals in sorted(hw_dispatches.items()):
        total = sum(end - start for start, end, _ in intervals)
        models_on_hw = sorted({m for _, _, m in intervals})
        _b(f"[info]   {hw}: {len(intervals)} dispatches, "
           f"{total * 1000:.3f}ms used per period, models={models_on_hw}")
    for model, jobs in sorted(job_intervals.items()):
        _b(f"[info]   {model}: {len(jobs)} job(s)/period, "
           f"span={jobs[0][1]*1000 - jobs[0][0]*1000:.3f}ms each")

    # --- env (forest, num_envs=1) ---
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

    # Set the sim control time step.  If --sim_dt is given, use it directly
    # (decimation=1, override sim.dt).  Otherwise auto-tune decimation from
    # the schedule's shortest perception job so start/end land on separate
    # ticks.
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = args_cli.sim_dt
        env_cfg.decimation = 1
        _b(f"[info] sim_dt override → {args_cli.sim_dt*1000:.2f}ms (decimation=1)")
    else:
        sim_dt = float(env_cfg.sim.dt)
        orig_decimation = int(env_cfg.decimation)
        perception_spans = [
            e - s
            for model, spans in job_intervals.items()
            for s, e in spans
            if "mlp" not in model.lower()
        ]
        if perception_spans:
            min_percep_span = min(perception_spans)
            target_dt = min_percep_span / 2.0
            needed_dec = max(1, int(target_dt / sim_dt))
            env_cfg.decimation = min(needed_dec, orig_decimation)
        _b(f"[info] decimation = {env_cfg.decimation} (was {orig_decimation}, "
           f"sim_dt={sim_dt*1000:.2f}ms → control_dt={sim_dt * env_cfg.decimation * 1000:.2f}ms)")

    # Match camera update rate to the control loop so DroNet/YOLO snapshots
    # always see fresh frames.  Default: same as control_dt.
    cam_period = args_cli.camera_update_period
    final_control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if cam_period > final_control_dt:
        cam_period = final_control_dt
        _b(f"[info] camera_update_period clamped to control_dt = {cam_period*1000:.2f}ms")
    env_cfg.scene.fpv_camera.update_period = cam_period
    # Stretch (or remove) the episode time-out so a single rollout can cover a
    # full long/curved trail without resetting mid-flight.
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_episode_timeout:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        _b("[info] episode time_out termination disabled — only crash/off_trail will reset")
    else:
        _b(f"[info] episode_length_s = {env_cfg.episode_length_s:.1f} s")

    # Diagnostic: verify scene tree/trail positions match the debug plot
    if curved and has_curve_override:
        _scene = env_cfg.scene
        _t0 = getattr(_scene, "tree_000", None)
        _s0 = getattr(_scene, "trail_seg_000", None)
        if _t0 is not None:
            _b(f"[diag] scene tree_000.pos = {_t0.init_state.pos}")
        if _s0 is not None:
            _b(f"[diag] scene trail_seg_000.pos = {_s0.init_state.pos}")
        _b(f"[diag] scene class = {type(_scene).__name__}, "
           f"num fields = {len([k for k in _scene.__dict__ if k.startswith('tree_')])}")

    _b(f"[bc] gym.make {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    # Isaac Lab's SimulationContext._app_control_on_stop_handle_fn spins on
    # `while not timeline.is_playing(): self.render()` when the app receives
    # a "stop" event (e.g. from simulation_app.close()) — the timeline never
    # resumes playing during shutdown, so this becomes an infinite loop that
    # wedges the process forever (0% GPU, one CPU core pegged). Isaac Lab's
    # own test suite disables this the same way ("prevent timeout").
    unwrapped_env.sim._disable_app_control_on_stop_handle = True
    _b("[bc] env ready")

    # --- inner-loop (MLP velocity tracker) policy ---
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
    runner.load(inner_ckpt)
    inner_policy = runner.get_inference_policy(device=args_cli.device)
    _b("[bc] inner policy ready")

    # --- DroNet ---
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

    # --- YOLOv8-nano (optional) ---
    yolo_model = None
    if args_cli.yolo:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            yolo_model = YOLO("yolov8n.pt")
            yolo_model.to(args_cli.device)
            _b(f"[info] YOLOv8-nano loaded (conf={args_cli.yolo_conf}, "
               f"classes={args_cli.yolo_classes})")
        except ImportError:
            _b("[warn] 'ultralytics' not installed — YOLO disabled. pip install ultralytics")

    # Resolve which normalized key in the schedule corresponds to DroNet / MLP.
    # The newer profiler outputs e.g. "mlp_control0..3" instead of "mlp0..N", so
    # we autodetect from the loaded schedule instead of hardcoding the names.
    def _autodetect(needle: str) -> str | None:
        candidates = [k for k in model_dispatches if needle.lower() in k.lower()]
        if not candidates:
            return None
        if len(candidates) > 1:
            _b(f"[warn] multiple keys contain {needle!r}: {candidates}; using {candidates[0]!r}. "
               f"Pass --{needle}_model_name to override.")
        return candidates[0]

    dronet_key = args_cli.dronet_model_name or _autodetect("dronet")
    mlp_key = args_cli.mlp_model_name or _autodetect("mlp")
    yolo_key = args_cli.yolo_model_name or _autodetect("yolo")
    # If MLP isn't in the schedule, run it ungated (every control step) — most
    # higher-level schedules (dronet+yolo) only budget the perception models
    # and assume the inner-loop controller runs continuously.
    mlp_runs_every_tick = mlp_key is None
    if dronet_key is None:
        _b("[warn] schedule has no DroNet-like dispatches; DroNet will never fire")
    else:
        _b(f"[info] DroNet schedule key: {dronet_key!r}")
    if mlp_runs_every_tick:
        _b("[info] schedule has no MLP-like dispatches; MLP will run every control step")
    else:
        _b(f"[info] MLP    schedule key: {mlp_key!r}")
    if yolo_model is not None:
        if yolo_key is None:
            _b("[info] schedule has no YOLO-like dispatches; YOLO will run every control step")
        else:
            _b(f"[info] YOLO   schedule key: {yolo_key!r}")
    elif yolo_key is not None:
        _b(f"[info] schedule includes YOLO key {yolo_key!r} but --yolo not set — YOLO inference disabled")
    yolo_runs_every_tick = (yolo_model is not None) and (yolo_key is None)

    # --- hooks into command term + camera ---
    steering_term = unwrapped_env.command_manager.get_term("steering_command")
    has_fpv = "fpv_camera" in unwrapped_env.scene.sensors
    if not has_fpv:
        raise RuntimeError("Forest env has no fpv_camera; cannot run DroNet.")

    # --- visualization ---
    show_plot = not args_cli.headless and not args_cli.no_fpv_plot
    fpv_fig = None
    im_rgb = im_processed = None
    text_overlay = None
    ax_schedule = None
    time_line = None

    # Optional video writer. Streamed (frame-by-frame), so long runs don't OOM.
    # For real-time playback (1s sim = 1s video), set fps = 1 / (skip * control_dt).
    video_writer = None
    video_path = args_cli.save_video
    if video_path is not None and not show_plot:
        _b("[warn] --save_video requested but matplotlib viz is disabled; ignoring.")
        video_path = None
    if video_path is not None:
        import imageio
        realtime_fps = 1.0 / (args_cli.video_capture_skip * final_control_dt)
        video_fps = args_cli.video_fps if args_cli.video_fps is not None else realtime_fps
        ext = os.path.splitext(video_path)[1].lower()
        if ext == ".mp4":
            video_writer = imageio.get_writer(
                video_path, fps=video_fps, codec="libx264",
                quality=8, macro_block_size=None,
            )
        elif ext == ".gif":
            video_writer = imageio.get_writer(
                video_path, mode="I", fps=video_fps, loop=0,
            )
        else:
            raise ValueError(f"--save_video extension must be .mp4 or .gif, got {ext!r}")
        _b(f"[info] streaming video to {video_path} (capture every {args_cli.video_capture_skip}"
           f" steps = {args_cli.video_capture_skip * final_control_dt * 1000:.1f}ms sim, "
           f"output {video_fps:.1f} fps {'(real-time)' if args_cli.video_fps is None else ''})")

    _video_writer_ref = video_writer  # noqa: F841 — global ref for interrupt handler
    _video_path_ref = video_path      # noqa: F841

    # --- run loop ---
    obs = env.get_observations()
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    _b(f"[info] control_dt = {control_dt * 1000:.2f} ms "
       f"(sim_dt={env_cfg.sim.dt * 1000:.1f} ms × decimation={env_cfg.decimation}), "
       f"steps/period ≈ {int(period / control_dt)}")

    # ZOH caches
    cached_target_w = 0.0
    cached_target_v = float(args_cli.forward_velocity)
    action_dim = env.action_space.shape[1] if hasattr(env, "action_space") else None
    cached_actions = None  # filled on first MLP execution
    cached_processed_vis = None
    last_yolo_results = None       # ultralytics Results object — held across YOLO-idle ticks
    last_max_person_conf = 0.0     # for the schedule timeline / overlay

    # Job-level input snapshots: captured at job start, consumed at job end.
    dronet_snapshot = None
    yolo_snapshot = None

    # Yaw-rate time series buffers (for the right-side panel)
    _HISTORY_LEN = 200
    cmd_w_buf = deque(maxlen=_HISTORY_LEN)
    act_w_buf = deque(maxlen=_HISTORY_LEN)
    t_buf = deque(maxlen=_HISTORY_LEN)

    sim_time = 0.0
    step = 0
    dronet_exec_count = 0
    mlp_exec_count = 0
    yolo_exec_count = 0
    _b("[info] starting scheduled forest pilot. Ctrl+C to stop.")

    # num_periods <= 0 means "run forever until Ctrl+C / window close".
    infinite_periods = args_cli.num_periods <= 0
    period_idx = 0
    if infinite_periods:
        _b("[info] num_periods=0 — running indefinitely until Ctrl+C / window close")
    while infinite_periods or period_idx < args_cli.num_periods:
        if not simulation_app.is_running():
            break
        period_start = sim_time
        if infinite_periods:
            _b(f"\n{'=' * 60}\nPERIOD {period_idx + 1} (∞ mode)\n{'=' * 60}")
        else:
            _b(f"\n{'=' * 60}\nPERIOD {period_idx + 1}/{args_cli.num_periods}\n{'=' * 60}")

        while (sim_time - period_start) < period and simulation_app.is_running():
            time_in_period = sim_time - period_start

            # Read FPV (always — cheap; we still render every step for visualization)
            camera = unwrapped_env.scene["fpv_camera"]
            rgb_data = camera.data.output["rgb"][0].cpu().numpy()
            rgb3 = rgb_data[:, :, :3]
            if rgb3.dtype != np.uint8:
                rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)
            rgb3 = np.ascontiguousarray(rgb3)

            # ── Job-level scheduling ─────────────────────────────────────
            # Correct hardware-in-the-loop semantics:
            #   • A model's input (camera frame) is sampled when the job STARTS
            #   • The model's output (yaw target) is applied when the job ENDS
            # Between start and end: ZOH on the previous output.
            #
            # "job started" = a job's start fell in (t - dt, t]
            # "job completed" = a job's end fell in (t - dt, t]

            def _job_started(model_key):
                if model_key is None or model_key not in job_intervals:
                    return False
                t_lo = time_in_period - control_dt
                return any(t_lo < s <= time_in_period for s, _e in job_intervals[model_key])

            def _job_completed(model_key):
                if model_key is None or model_key not in job_intervals:
                    return False
                t_lo = time_in_period - control_dt
                # On the last tick of the period, extend the check window to
                # cover the full remaining period so jobs ending at/near the
                # period boundary aren't missed.
                is_last_tick = (time_in_period + control_dt) >= period
                t_hi = period if is_last_tick else time_in_period
                return any(t_lo < e <= t_hi for _s, e in job_intervals[model_key])

            # DroNet: snapshot on job start, apply on job end
            if _job_started(dronet_key):
                dronet_snapshot = rgb3.copy()

            dronet_active = _job_completed(dronet_key)
            if dronet_active and dronet_snapshot is not None:
                with torch.no_grad():
                    x = preprocess_for_dronet(dronet_snapshot, img_size, args_cli.device)
                    steer_pred, coll_pred = dronet(x)
                steer_val = float(steer_pred.item())
                coll_val = float(coll_pred.item())

                if dronet_exec_count == 0:
                    _b(f"  [dronet] first exec: input shape={dronet_snapshot.shape} "
                       f"range=[{dronet_snapshot.min()}, {dronet_snapshot.max()}] "
                       f"steer={steer_val:.4f}")

                cached_target_w = max(-args_cli.omega_clamp,
                                      min(args_cli.omega_clamp, steer_val))
                if args_cli.use_collision_modulation:
                    cached_target_v = (1.0 - coll_val) * args_cli.max_velocity
                else:
                    cached_target_v = float(args_cli.forward_velocity)

                if show_plot:
                    cached_processed_vis = (
                        x[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
                    )
                dronet_exec_count += 1

            # YOLO: snapshot on job start, apply on job end
            if _job_started(yolo_key):
                yolo_snapshot = rgb3.copy()

            yolo_active = (yolo_model is not None) and (
                yolo_runs_every_tick or _job_completed(yolo_key)
            )
            if yolo_active:
                yolo_input = yolo_snapshot if (yolo_snapshot is not None and not yolo_runs_every_tick) else rgb3
                last_yolo_results = yolo_model(
                    yolo_input,
                    conf=args_cli.yolo_conf,
                    classes=args_cli.yolo_classes,
                    verbose=False,
                )
                last_max_person_conf = 0.0
                boxes = last_yolo_results[0].boxes
                if len(boxes):
                    for cid, conf in zip(boxes.cls.cpu().int().tolist(),
                                         boxes.conf.cpu().tolist()):
                        if cid == 0 and conf > last_max_person_conf:
                            last_max_person_conf = conf
                yolo_exec_count += 1

            # MLP: runs every tick (ungated) or on job completion
            mlp_active = mlp_runs_every_tick or _job_completed(mlp_key)

            # Push command (always — uses cached value when DroNet not active = ZOH)
            steering_term.target_velocity.fill_(cached_target_v)
            steering_term.target_yaw_rate.fill_(cached_target_w)

            # 2) MLP policy inference (only if scheduled)
            if mlp_active:
                actions = inner_policy(obs)
                cached_actions = actions.clone()
                mlp_exec_count += 1
            elif cached_actions is not None:
                actions = cached_actions
            else:
                # Before MLP has fired the first time, fall back to a zero action
                # (drone falls but the schedule will catch up quickly).
                actions = torch.zeros(env.num_envs, env.action_space.shape[1],
                                      device=args_cli.device)

            obs, rewards, dones, info = env.step(actions)
            sim_time += control_dt

            # Telemetry
            robot = unwrapped_env.scene["robot"]
            x_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[0].item()
            y_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[1].item()
            height = robot.data.root_pos_w[0, 2].item()
            vel_x = robot.data.root_lin_vel_b[0, 0].item()
            yaw_rate = robot.data.root_ang_vel_b[0, 2].item()

            t_buf.append(sim_time)
            cmd_w_buf.append(cached_target_w)
            act_w_buf.append(yaw_rate)

            if step % 50 == 0:
                ds = "EXEC" if dronet_active else "cache"
                ms = "EXEC" if mlp_active else "cache"
                ys = ""
                if yolo_model is not None:
                    yflag = "EXEC" if yolo_active else "cache"
                    ys = f" YOLO={yflag} pconf={last_max_person_conf:.2f}"
                _b(f"  step {step:5d}  t={time_in_period * 1000:6.2f}ms  [DroNet={ds} MLP={ms}{ys}]  "
                   f"pos=({x_local:+.2f},{y_local:+.2f})  cmd ω={cached_target_w:+.2f} "
                   f"v={cached_target_v:+.2f}  act ω={yaw_rate:+.2f} v={vel_x:+.2f}")

            if dones.any():
                _b(f"  [reset] step {step + 1}: drone reset")

            # Visualization (every 5 steps, but only every N periods for video)
            if show_plot and step % 5 == 0:
                try:
                    if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                        fpv_fig = None
                    if fpv_fig is None:
                        plt.ion()
                        fpv_fig = plt.figure(num="Scheduled DroNet (forest)", figsize=(16, 8))
                        gs = fpv_fig.add_gridspec(
                            2, 3, height_ratios=[2, 1], width_ratios=[1.2, 1.0, 1.0],
                            hspace=0.35, wspace=0.25,
                        )
                        ax_rgb = fpv_fig.add_subplot(gs[0, 0])
                        ax_proc = fpv_fig.add_subplot(gs[0, 1])
                        ax_yaw = fpv_fig.add_subplot(gs[0, 2])
                        ax_schedule = fpv_fig.add_subplot(gs[1, :])

                        if yolo_model is not None and last_yolo_results is not None:
                            last_yolo_results[0].orig_img = rgb3
                            initial_fpv = last_yolo_results[0].plot()
                        else:
                            initial_fpv = rgb3
                        im_rgb = ax_rgb.imshow(initial_fpv)
                        ax_rgb.set_title("FPV camera" + (" (YOLO boxes = stale)" if yolo_model else ""))
                        ax_rgb.axis("off")

                        proc_init = (cached_processed_vis if cached_processed_vis is not None
                                     else np.zeros((img_size, img_size, 3), dtype=np.uint8))
                        im_processed = ax_proc.imshow(proc_init)
                        ax_proc.set_title(f"DroNet input ({img_size}x{img_size})")
                        ax_proc.axis("off")

                        (line_cmd_w,) = ax_yaw.plot([], [], color="tab:orange", lw=2, label="cmd (DroNet)")
                        (line_act_w,) = ax_yaw.plot([], [], color="tab:blue", lw=1.5, label="actual")
                        ax_yaw.set_ylabel("ω (rad/s)")
                        ax_yaw.set_xlabel("time (s)")
                        ax_yaw.set_title("yaw rate")
                        ax_yaw.grid(True, alpha=0.3)
                        ax_yaw.legend(loc="upper left", fontsize=8)
                        ax_yaw.axhline(0.0, color="k", lw=0.5, alpha=0.4)
                        fpv_fig._line_cmd_w = line_cmd_w
                        fpv_fig._line_act_w = line_act_w
                        fpv_fig._ax_yaw = ax_yaw

                        # Schedule timeline — one row per hardware target,
                        # bars coloured by which model is occupying that HW
                        # in each interval.  This matches how a real-time HW
                        # utilization view looks (concurrent models on shared
                        # cores show up as colour changes in the same row).
                        if hw_dispatches:
                            hw_names = sorted(hw_dispatches.keys())
                            n_hw = len(hw_names)
                            ax_schedule.set_ylim(-0.5, n_hw - 0.5)
                            ax_schedule.set_yticks(list(range(n_hw)))
                            ax_schedule.set_yticklabels(hw_names)
                            for row, hw in enumerate(hw_names):
                                for s, e, model in hw_dispatches[hw]:
                                    ax_schedule.broken_barh(
                                        [(s * 1000, (e - s) * 1000)],
                                        (row - 0.4, 0.8),
                                        facecolors=_colour_for_model(model),
                                        edgecolor="black", linewidth=0.4,
                                    )
                            # Legend entries — one per distinct model present
                            # in the schedule.  Drawn as invisible bars so
                            # matplotlib's legend renders the colour swatches.
                            seen_models: list[str] = []
                            for hw in hw_names:
                                for _s, _e, model in hw_dispatches[hw]:
                                    if model not in seen_models:
                                        seen_models.append(model)
                            from matplotlib.patches import Patch as _Patch
                            legend_handles = [
                                _Patch(facecolor=_colour_for_model(m), edgecolor="black",
                                       label=m)
                                for m in seen_models
                            ]
                        else:
                            # Fallback for older schedules without hardware_target
                            ax_schedule.set_ylim(-0.5, 1.5)
                            ax_schedule.set_yticks([0, 1])
                            ax_schedule.set_yticklabels(["MLP", "DroNet"])
                            for s, e in (model_dispatches.get(dronet_key, []) if dronet_key else []):
                                ax_schedule.broken_barh(
                                    [(s * 1000, (e - s) * 1000)], (0.6, 0.8),
                                    facecolors="orange", edgecolor="black", linewidth=0.5,
                                )
                            for s, e in (model_dispatches.get(mlp_key, []) if mlp_key else []):
                                ax_schedule.broken_barh(
                                    [(s * 1000, (e - s) * 1000)], (-0.4, 0.8),
                                    facecolors="skyblue", edgecolor="black", linewidth=0.5,
                                )
                            legend_handles = []
                        ax_schedule.set_xlim(0, period * 1000)
                        ax_schedule.set_xlabel("time within period (ms)")
                        ax_schedule.set_title(
                            f"Schedule by hardware target (period = {period * 1000:.3f} ms)"
                        )
                        ax_schedule.grid(True, alpha=0.3, axis="x")
                        time_line = ax_schedule.axvline(
                            x=0, color="red", linewidth=2, linestyle="-", alpha=0.8,
                            label="current time",
                        )
                        # Combine model-colour legend with the current-time line.
                        if legend_handles:
                            legend_handles.append(time_line)
                        ax_schedule.legend(handles=legend_handles or None,
                                           loc="upper right", fontsize=8)

                        text_overlay = fpv_fig.text(
                            0.5, 0.49, "", ha="center", fontsize=10, weight="bold",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, pad=0.4),
                            color="white",
                        )
                    else:
                        # Always show the *current* FPV frame. When YOLO has
                        # stale results, swap orig_img so .plot() overlays the
                        # stale boxes on the live camera feed — matching what a
                        # real scheduled system would see between inferences.
                        if yolo_model is not None and last_yolo_results is not None:
                            last_yolo_results[0].orig_img = rgb3
                            im_rgb.set_data(last_yolo_results[0].plot())
                        else:
                            im_rgb.set_data(rgb3)
                        if cached_processed_vis is not None:
                            im_processed.set_data(cached_processed_vis)

                    if time_line is not None:
                        time_line.set_xdata([time_in_period * 1000, time_in_period * 1000])
                    if hasattr(fpv_fig, "_line_cmd_w") and len(t_buf) >= 2:
                        t_arr = np.fromiter(t_buf, dtype=float)
                        fpv_fig._line_cmd_w.set_data(t_arr, np.fromiter(cmd_w_buf, dtype=float))
                        fpv_fig._line_act_w.set_data(t_arr, np.fromiter(act_w_buf, dtype=float))
                        fpv_fig._ax_yaw.set_xlim(t_arr[0], t_arr[-1])
                        fpv_fig._ax_yaw.relim()
                        fpv_fig._ax_yaw.autoscale_view(scalex=False, scaley=True)
                    if text_overlay is not None:
                        yolo_str = ""
                        if yolo_model is not None:
                            yolo_str = f"  YOLO={yolo_exec_count} pconf={last_max_person_conf:.2f}"
                        text_overlay.set_text(
                            f"DroNet ω={cached_target_w:+.2f}  v_cmd={cached_target_v:+.2f}  |  "
                            f"actual ω={yaw_rate:+.2f}  v={vel_x:+.2f}\n"
                            f"pos=({x_local:+.2f},{y_local:+.2f}) m  h={height:.2f} m  "
                            f"DroNet={dronet_exec_count}  MLP={mlp_exec_count}{yolo_str}"
                        )

                    fpv_fig.canvas.draw_idle()
                    fpv_fig.canvas.flush_events()
                    plt.pause(0.001)

                    # Stream this frame to the video writer if requested.
                    if (video_writer is not None
                            and step % args_cli.video_capture_skip == 0):
                        fpv_fig.canvas.draw()
                        frame_rgba = np.asarray(fpv_fig.canvas.buffer_rgba())
                        video_writer.append_data(frame_rgba[:, :, :3])
                except Exception as e:
                    if step == 0:
                        _b(f"[warn] matplotlib FPV preview failed: {e}")

            step += 1
        # End of one schedule period — advance the outer counter.  In ∞ mode
        # we never compare it to num_periods, so the while-loop continues.
        period_idx += 1
        _video_period_ref = period_idx
        if (video_writer is not None
                and period_idx % args_cli.video_flush_periods == 0):
            # Finalize the current video so it's playable, then start a new
            # chunk.  This lets the user check progress mid-run.
            video_writer.close()
            base, ext = os.path.splitext(video_path)
            chunk_path = f"{base}_p{period_idx}{ext}"
            os.rename(video_path, chunk_path)
            _b(f"[info] video checkpoint → {chunk_path}")
            import imageio
            if ext.lower() == ".mp4":
                video_writer = imageio.get_writer(
                    video_path, fps=video_fps, codec="libx264",
                    quality=8, macro_block_size=None,
                )
            else:
                video_writer = imageio.get_writer(
                    video_path, mode="I", fps=video_fps, loop=0,
                )
            _video_writer_ref = video_writer
            _video_period_ref = period_idx

    _b(f"\n[done] DroNet executions: {dronet_exec_count}")
    _b(f"[done] MLP executions:    {mlp_exec_count}"
       f"{' (ungated — every tick)' if mlp_runs_every_tick else ''}")
    if yolo_model is not None:
        _b(f"[done] YOLO executions:   {yolo_exec_count}"
           f"{' (ungated — every tick)' if yolo_runs_every_tick else ''}")
    _b(f"[done] total control steps: {step}")
    _finalize_video(video_writer, video_path, period_idx)
    _video_writer_ref = None
    if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
        plt.close(fpv_fig)
    env.close()




if __name__ == "__main__":
    import traceback
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] interrupted by user — flushing video...", flush=True)
        _finalize_video(_video_writer_ref, _video_path_ref, _video_period_ref)
    except Exception:
        print("[error] uncaught exception in main():", flush=True)
        traceback.print_exc()
        _finalize_video(_video_writer_ref, _video_path_ref, _video_period_ref)
        raise
    finally:
        print("[bc] entering finally; closing simulation_app", flush=True)
        simulation_app.close()
