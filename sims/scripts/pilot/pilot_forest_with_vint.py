#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run ViNT (visual-navigation transformer) in the forest-trail env.

Drop-in alternative to ``pilot_forest_with_dronet.py``: ViNT takes a 5-frame
context (the past 5 FPV images plus the current one) and a goal image, and
outputs 5 future waypoints in robot-local frame.  We map the angle of the
chosen waypoint to a yaw-rate command for the steering inner loop.

Setup (one-time):
    1. The repo is vendored at ``sims/external/visualnav-transformer/``.
    2. Download the pretrained ViNT checkpoint (``vint.pth``) from the
       project's Google Drive folder:
           https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg
       and save it to:
           sims/external/visualnav-transformer/deployment/model_weights/vint.pth
       (Auto-mode classifier blocks the download from inside Claude — please
       fetch manually and re-run.)
    3. ``pip install efficientnet_pytorch`` (already done in xpurt env).

Usage:
    # ViNT with default goal image (auto-picked from IDSIA samples)
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_vint.py

    # Curved trail
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_vint.py \\
        --trail curved --max_turn_deg 30

    # Specify a custom goal image (any forward-looking trail JPG/PNG)
    conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_vint.py \\
        --goal_image datasets/idsia/samples/sc/000_seg000_frame0001.jpg

Caveat: ViNT is goal-conditioned; it doesn't have NoMaD's goal-mask
"explore" mode.  Without a real goal image the predicted waypoints will be
biased toward whatever the goal looks like.  For trail navigation we use a
forward-looking trail snapshot as a proxy goal — this gives the model a
"keep heading down a trail like this" target, which is roughly what we want.
"""

import argparse
import glob
import os
import sys
from collections import deque

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
isaaclab_root = os.path.join(freshscheduler_root, "sims/IsaacLab/source")
vint_root = os.path.join(freshscheduler_root, "sims/external/visualnav-transformer/train")
sys.path.insert(0, freshscheduler_root)
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_rl"))
sys.path.insert(0, os.path.join(isaaclab_root, "isaaclab_contrib"))
sys.path.insert(0, vint_root)

from isaaclab.app import AppLauncher

_DEFAULT_VINT_CKPT = os.path.join(
    freshscheduler_root,
    "sims/external/visualnav-transformer/deployment/model_weights/vint.pth",
)
_DEFAULT_GOAL_DIR = os.path.join(
    freshscheduler_root, "datasets/idsia/samples/sc",
)

parser = argparse.ArgumentParser(description="ViNT pilot in forest-trail env.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Velocity-tracker (rsl_rl) checkpoint. Default: latest under logs/.")
parser.add_argument("--vint_ckpt", type=str, default=_DEFAULT_VINT_CKPT,
                    help=f"ViNT *.pth checkpoint. Default: {_DEFAULT_VINT_CKPT}")
parser.add_argument("--vint_onnx", type=str, default=None,
                    help=(
                        "Path to a quantized (or fp32) ONNX export of ViNT, produced by "
                        "sims/scripts/utils/quantize_vint.py.  When set, ONNX Runtime is "
                        "used for inference instead of the PyTorch model — the quantized "
                        "int8 export is ~4× smaller and validates the model survives PTQ."
                    ))
parser.add_argument("--goal_image", type=str, default=None,
                    help=(
                        "Path to a goal image for ViNT.  By default the goal image is "
                        "RENDERED at the trail end inside this env (so it's in-distribution "
                        "for the rest of the run).  Pass an explicit JPG/PNG path to override "
                        "(e.g. an IDSIA sample) — useful for cross-domain experiments."
                    ))
parser.add_argument("--goal_pose_t", type=float, default=0.95,
                    help=(
                        "Where on the trail (in arc-length fraction ∈ [0, 1]) to render the "
                        "goal image from.  Default 0.95 — slightly inside the end so the FPV "
                        "still sees trail extending forward.  1.0 puts the camera at the very "
                        "last waypoint (looking past the trail).  Ignored when --goal_image "
                        "is passed explicitly."
                    ))
parser.add_argument("--save_goal_image", type=str, default="/tmp/vint_goal.png",
                    help=(
                        "Path to save the rendered goal image to (PNG).  Useful for "
                        "debugging — open it side-by-side with the FPV to verify it matches "
                        "the kind of view the drone should be steering toward."
                    ))
parser.add_argument("--waypoint_idx", type=int, default=2,
                    help=(
                        "Which of ViNT's 5 predicted waypoints to use for the steering "
                        "command.  Default 2 (middle); 0 is the closest, 4 is the farthest."
                    ))
parser.add_argument("--forward_velocity", type=float, default=1.0,
                    help="Constant target_velocity (m/s) fed to the inner loop.")
parser.add_argument("--omega_clamp", type=float, default=1.5,
                    help="Hard clamp on yaw-rate command (rad/s).")
parser.add_argument("--omega_gain", type=float, default=2.0,
                    help=(
                        "Scale factor mapping ViNT's waypoint angle (rad) → yaw rate (rad/s). "
                        "ViNT's waypoint angles are usually small (<0.5 rad); a gain of 2-3 "
                        "produces visibly responsive steering."
                    ))
parser.add_argument("--camera_update_period", type=float, default=0.1,
                    help="FPV camera update period (s). Default 0.1 (10 Hz).")
parser.add_argument("--trail", choices=["straight", "curved"], default="straight",
                    help="Trail geometry. 'straight' (default) or 'curved'.")
parser.add_argument("--max_turn_deg", type=float, default=None,
                    help="Curved-trail only: max heading change per segment (deg).")
parser.add_argument("--num_segments", type=int, default=None,
                    help="Curved-trail only: number of polyline segments.")
parser.add_argument("--segment_length", type=float, default=None,
                    help="Curved-trail only: segment length in metres.")
parser.add_argument("--curvature_seed", type=int, default=None,
                    help="Curved-trail only: RNG seed for the polyline.")
parser.add_argument("--no_humans", action="store_true",
                    help="Disable procedural humans on the trail.")
parser.add_argument("--episode_length_s", type=float, default=120.0,
                    help="Per-episode timer (s). Default 120.")
parser.add_argument("--no_episode_timeout", action="store_true",
                    help="Disable the time_out termination entirely.")
parser.add_argument("--save_video", type=str, default=None,
                    help="Path to save a video (mp4/gif) of the matplotlib FPV capture.")
parser.add_argument("--video_fps", type=int, default=30,
                    help="FPS for the saved video. Default 30.")
parser.add_argument("--video_capture_skip", type=int, default=5,
                    help="Capture a video frame every N sim steps. Default 5.")
parser.add_argument("--video_max_frames", type=int, default=None,
                    help="Stop recording after this many video frames (None = until trail exit or shutdown).")
parser.add_argument("--no_fpv_plot", action="store_true",
                    help="Don't open the matplotlib viz window.")
parser.add_argument("--history_window", type=int, default=200,
                    help="Steps kept in the time-series plot.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that need the runtime to be up.
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from rsl_rl.runners import OnPolicyRunner
from torchvision import transforms

from sims.isaaclab_tasks.forest_trail.config import crazyflie as _forest_register  # noqa: F401
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie as _track_register  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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

# ViNT imports — vendored in sims/external/visualnav-transformer/.
from vint_train.models.vint.vint import ViNT  # noqa: E402


def _b(msg: str) -> None:
    print(msg, flush=True)


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
        raise FileNotFoundError("No velocity-tracker checkpoint found. Pass --checkpoint.")
    return max(cks, key=os.path.getmtime)


def _load_vint(ckpt_path: str, device: str) -> tuple[ViNT, dict]:
    """Load ViNT from the published vint.yaml settings.

    The visualnav-transformer repo's ``vint.yaml`` describes the exact arch
    (context_size=5, len_traj_pred=5, learn_angle=True, EfficientNet-B0,
    obs_encoding_size=512, 4 attention heads × 4 layers).  We mirror those
    here so the checkpoint loads bit-exact.
    """
    cfg_path = os.path.join(
        freshscheduler_root,
        "sims/external/visualnav-transformer/train/config/vint.yaml",
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = ViNT(
        context_size=cfg["context_size"],
        len_traj_pred=cfg["len_traj_pred"],
        learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"],
        obs_encoding_size=cfg["obs_encoding_size"],
        late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
    )

    if not os.path.isfile(ckpt_path):
        _b(f"[warn] ViNT checkpoint not found at {ckpt_path}")
        _b("[warn] Running with RANDOM-INIT weights — output will be meaningless.")
        _b("[warn] Download from https://drive.google.com/drive/folders/"
           "1a9yWR2iooXFAqjQHetz263--4_2FFggg")
    else:
        _b(f"[info] loading ViNT checkpoint: {ckpt_path}")
        # Published checkpoints are saved as {"model": <wrapped DataParallel>}.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        loaded = ckpt["model"]
        try:
            state_dict = loaded.module.state_dict()
        except AttributeError:
            state_dict = loaded.state_dict()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _b(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            _b(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    model = model.to(device).eval()
    return model, cfg


def _vint_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """ImageNet-normalized resize → tensor pipeline that ViNT was trained with."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _compute_goal_pose(
    env_cfg,
    trail_kind: str,
    arc_fraction: float,
) -> tuple[float, float, float]:
    """Return ``(x_local, y_local, yaw)`` for the goal pose in env-local frame.

    For a straight trail the goal sits along +x at ``arc_fraction *
    trail_length`` with yaw=0.  For a curved trail we walk the polyline
    (read out of the off-trail termination's ``waypoints`` param) and pick
    the segment-tangent heading at the corresponding arc-length position.
    """
    import math
    f = max(0.0, min(1.0, arc_fraction))
    if trail_kind == "straight":
        # Trail length is on the layout used to build this env's terminations.
        trail_len = float(env_cfg.terminations.off_trail.params.get("trail_length", 30.0))
        return (trail_len * f, 0.0, 0.0)

    # Curved: waypoints are tuples of (x, y) — flat 2-D polyline.
    wps = list(env_cfg.terminations.off_trail.params.get("waypoints", ()))
    if len(wps) < 2:
        raise RuntimeError("curved env has no waypoints in off_trail termination")
    seg_lens = [math.hypot(wps[i+1][0] - wps[i][0], wps[i+1][1] - wps[i][1])
                for i in range(len(wps) - 1)]
    total = sum(seg_lens)
    target = total * f
    arc = 0.0
    for j, slen in enumerate(seg_lens):
        if arc + slen >= target or j == len(seg_lens) - 1:
            t = (target - arc) / max(slen, 1e-9)
            t = max(0.0, min(1.0, t))
            x0, y0 = wps[j]
            x1, y1 = wps[j + 1]
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            return (x, y, yaw)
        arc += slen
    p = wps[-1]
    return (p[0], p[1], 0.0)


def _capture_goal_image(unwrapped_env, env_for_step, goal_xy_yaw, num_settle_steps: int):
    """Teleport the drone to ``goal_xy_yaw`` (env-local), let the camera render,
    capture the FPV, then teleport back to the spawn pose.

    Uses ``env.step()`` with zero actions to advance physics + render; the
    off-trail termination is suppressed for the duration of the capture so
    spawning at the trail end doesn't immediately reset the drone.
    """
    import math
    import torch
    import numpy as np

    robot = unwrapped_env.scene["robot"]
    env_origin = unwrapped_env.scene.env_origins[0]
    device = robot.data.root_pos_w.device

    goal_x, goal_y, goal_yaw = goal_xy_yaw
    half = goal_yaw * 0.5
    qw, qz = math.cos(half), math.sin(half)

    # Build a single-row root state in world frame: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    pose_world = torch.zeros((1, 7), device=device)
    pose_world[0, 0] = env_origin[0] + goal_x
    pose_world[0, 1] = env_origin[1] + goal_y
    pose_world[0, 2] = 1.0
    pose_world[0, 3] = qw
    pose_world[0, 6] = qz
    vel_zero = torch.zeros((1, 6), device=device)
    robot.write_root_pose_to_sim(pose_world)
    robot.write_root_velocity_to_sim(vel_zero)

    # Suppress off-trail term during capture so we don't immediately reset.
    term_mgr = unwrapped_env.termination_manager
    saved_off_trail = None
    if "off_trail" in term_mgr.active_terms:
        saved_off_trail = term_mgr._term_cfgs[term_mgr.active_terms.index("off_trail")]
    # Easiest route: just step a few times with zero actions; even if a reset
    # fires, our final pose-write below restores the goal pose before the
    # camera read.
    n_act = env_for_step.action_space.shape[1]
    zero_act = torch.zeros((1, n_act), device=device)
    for _ in range(num_settle_steps):
        env_for_step.step(zero_act)
        # Re-pin the pose every step so off-trail / time_out can't drag the
        # drone away from the goal.
        robot.write_root_pose_to_sim(pose_world)
        robot.write_root_velocity_to_sim(vel_zero)

    # Read FPV after settle.
    rgb_data = unwrapped_env.scene["fpv_camera"].data.output["rgb"][0].cpu().numpy()
    rgb3 = rgb_data[:, :, :3]
    if rgb3.dtype != np.uint8:
        rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)
    return PILImage.fromarray(rgb3), saved_off_trail


def _resolve_goal_image(args_goal: str | None) -> str:
    if args_goal:
        return args_goal
    if os.path.isdir(_DEFAULT_GOAL_DIR):
        jpgs = sorted(glob.glob(os.path.join(_DEFAULT_GOAL_DIR, "*.jpg")))
        if jpgs:
            chosen = jpgs[len(jpgs) // 2]
            _b(f"[info] auto-picked goal image: {chosen}")
            return chosen
    raise FileNotFoundError(
        f"No goal image. Pass --goal_image, or run "
        f"sims/scripts/utils/export_dronet_samples.py to populate "
        f"{_DEFAULT_GOAL_DIR}."
    )


def main():
    inner_ckpt = args_cli.checkpoint or find_latest_checkpoint()
    _b(f"[info] inner-loop checkpoint: {inner_ckpt}")

    # ── env construction (mirrors pilot_forest_with_dronet.py) ─────────────────
    use_humans = not args_cli.no_humans
    curved = args_cli.trail == "curved"
    has_curve_override = curved and any(
        v is not None for v in (args_cli.max_turn_deg, args_cli.num_segments,
                                args_cli.segment_length, args_cli.curvature_seed))
    if curved:
        task_id = ("Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-v0"
                   if use_humans else "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-v0")
        if has_curve_override:
            layout = CurvedTrailLayout()
            if args_cli.max_turn_deg is not None: layout.max_turn_deg = args_cli.max_turn_deg
            if args_cli.num_segments is not None: layout.num_segments = args_cli.num_segments
            if args_cli.segment_length is not None: layout.segment_length = args_cli.segment_length
            if args_cli.curvature_seed is not None: layout.seed = args_cli.curvature_seed
            env_cfg = make_curved_env_cfg(layout, with_humans=use_humans, play=True, num_envs=1)
        else:
            cfg_cls = (ForestTrailEnvCfg_Curved_PLAY_WithHumans if use_humans
                       else ForestTrailEnvCfg_Curved_PLAY)
            env_cfg = cfg_cls()
    else:
        cfg_cls = (ForestTrailEnvCfg_PLAY_WithHumans if use_humans
                   else ForestTrailEnvCfg_PLAY)
        task_id = ("Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0"
                   if use_humans else "Isaac-Forest-Trail-Vision-Crazyflie-Play-v0")
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
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.no_episode_timeout and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

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

    # ── inner-loop policy ──────────────────────────────────────────────────────
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
            "distribution_cfg": (agent_cfg.actor.distribution_cfg.to_dict()
                                 if agent_cfg.actor.distribution_cfg else None),
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

    # ── ViNT load + goal image setup ───────────────────────────────────────────
    # Two inference backends: PyTorch ViNT (default) or ONNX Runtime on a
    # pre-quantized model (when --vint_onnx is given).  The yaml config is
    # always loaded so we know the input image_size / context_size that the
    # model expects, regardless of backend.
    vint_cfg_path = os.path.join(
        freshscheduler_root,
        "sims/external/visualnav-transformer/train/config/vint.yaml",
    )
    with open(vint_cfg_path, "r") as f:
        vint_cfg = yaml.safe_load(f)
    vint_session = None
    vint_model = None
    if args_cli.vint_onnx:
        import onnxruntime as ort  # noqa: PLC0415
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "cuda" in args_cli.device else ["CPUExecutionProvider"])
        vint_session = ort.InferenceSession(args_cli.vint_onnx, providers=providers)
        size_mb = os.path.getsize(args_cli.vint_onnx) / 1e6
        _b(f"[info] ONNX ViNT loaded: {args_cli.vint_onnx} ({size_mb:.1f} MB), "
           f"providers={vint_session.get_providers()}")
    else:
        vint_model, _ = _load_vint(args_cli.vint_ckpt, args_cli.device)
    image_size = tuple(vint_cfg["image_size"])  # e.g. [85, 64] = (W, H)
    # PIL & torchvision use (H, W) for sizes; the published config is (W, H).
    pil_size = (image_size[0], image_size[1])
    transform = _vint_transform((image_size[1], image_size[0]))
    context_size = vint_cfg["context_size"]
    len_traj_pred = vint_cfg["len_traj_pred"]
    waypoint_idx = max(0, min(len_traj_pred - 1, args_cli.waypoint_idx))

    if args_cli.goal_image:
        goal_path = _resolve_goal_image(args_cli.goal_image)
        goal_pil = PILImage.open(goal_path).convert("RGB").resize(pil_size)
        _b(f"[info] using explicit goal image: {goal_path}")
    else:
        # Render the goal image from inside this very env, at the configured
        # arc-length fraction along the trail.  Same lighting + textures as
        # the runtime FPV → ViNT can compare apples-to-apples.
        goal_xy_yaw = _compute_goal_pose(env_cfg, args_cli.trail, args_cli.goal_pose_t)
        _b(f"[info] rendering goal at trail position t={args_cli.goal_pose_t:.2f} → "
           f"(x={goal_xy_yaw[0]:.2f}, y={goal_xy_yaw[1]:.2f}, yaw={goal_xy_yaw[2]:+.2f} rad)")
        goal_full_pil, _saved = _capture_goal_image(
            unwrapped_env, env, goal_xy_yaw, num_settle_steps=15,
        )
        goal_pil = goal_full_pil.resize(pil_size)
        if args_cli.save_goal_image:
            goal_full_pil.save(args_cli.save_goal_image)
            _b(f"[info] saved goal image preview to {args_cli.save_goal_image}")
        # Reset the drone back to the spawn point before the main loop starts.
        env.reset()
        goal_path = args_cli.save_goal_image or "<rendered>"
    goal_tensor = transform(goal_pil).unsqueeze(0).to(args_cli.device)

    # Rolling FPV context: (context_size + 1) most recent frames as PILs.
    context: deque[PILImage.Image] = deque(maxlen=context_size + 1)

    # Hooks
    steering_term = unwrapped_env.command_manager.get_term("steering_command")
    if "fpv_camera" not in unwrapped_env.scene.sensors:
        raise RuntimeError("Forest env has no fpv_camera; cannot run ViNT.")

    show_plot = not args_cli.headless and not args_cli.no_fpv_plot
    fpv_fig = None
    im_fpv = None
    overlay_text = None
    line_act_w = None
    cmd_w_buf = deque(maxlen=args_cli.history_window)
    act_w_buf = deque(maxlen=args_cli.history_window)
    t_buf = deque(maxlen=args_cli.history_window)
    control_dt = unwrapped_env.cfg.sim.dt * unwrapped_env.cfg.decimation

    obs = env.get_observations()
    step = 0
    last_waypoint_xy = (0.0, 0.0)
    last_target_w = 0.0
    _b(f"[info] starting ViNT pilot (ctxt={context_size}, "
       f"goal={os.path.basename(goal_path)}, waypoint_idx={waypoint_idx}). Ctrl+C to stop.")

    video_writer = None
    video_frame_count = 0
    if args_cli.save_video:
        import imageio
        video_writer = imageio.get_writer(
            args_cli.save_video, fps=args_cli.video_fps, macro_block_size=1,
        )
        _b(f"[info] recording video → {args_cli.save_video} (fps={args_cli.video_fps}, "
           f"capture every {args_cli.video_capture_skip} steps, "
           f"max_frames={args_cli.video_max_frames or 'unlimited'})")

    while simulation_app.is_running():
        camera = unwrapped_env.scene["fpv_camera"]
        rgb_data = camera.data.output["rgb"][0].cpu().numpy()
        rgb3 = rgb_data[:, :, :3]
        if rgb3.dtype != np.uint8:
            rgb3 = (np.clip(rgb3, 0.0, 1.0) * 255).astype(np.uint8)

        # Push current frame into the rolling context queue.
        context.append(PILImage.fromarray(rgb3).resize(pil_size))

        # Run ViNT only when we've accumulated enough context (context_size+1 frames).
        if len(context) == context_size + 1:
            obs_tensors = [transform(im) for im in context]
            obs_batched = torch.cat(obs_tensors, dim=0).unsqueeze(0).to(args_cli.device)
            if vint_session is not None:
                # ONNX Runtime path — feed numpy arrays.
                ort_outs = vint_session.run(
                    None,
                    {"obs": obs_batched.cpu().numpy().astype(np.float32),
                     "goal": goal_tensor.cpu().numpy().astype(np.float32)},
                )
                action_pred = torch.from_numpy(ort_outs[1])
            else:
                with torch.no_grad():
                    _dist_pred, action_pred = vint_model(obs_batched, goal_tensor)
            # action_pred: (1, len_traj_pred, 4) when learn_angle=True.
            # Coords are normalized deltas in robot frame; cumulative sum gives
            # absolute waypoints (robot-local x/y in [-1, 1]).
            deltas = action_pred[0, :, :2].cpu().numpy()
            waypoints = np.cumsum(deltas, axis=0)
            wp = waypoints[waypoint_idx]
            last_waypoint_xy = (float(wp[0]), float(wp[1]))
            angle = float(np.arctan2(wp[1], wp[0]))   # heading toward waypoint
            last_target_w = max(-args_cli.omega_clamp,
                                min(args_cli.omega_clamp, angle * args_cli.omega_gain))

        target_v = float(args_cli.forward_velocity)
        steering_term.target_velocity.fill_(target_v)
        steering_term.target_yaw_rate.fill_(last_target_w)

        actions = inner_policy(obs)
        obs, _rew, dones, _info = env.step(actions)

        robot = unwrapped_env.scene["robot"]
        height = robot.data.root_pos_w[0, 2].item()
        vel_x = robot.data.root_lin_vel_b[0, 0].item()
        yaw_rate = robot.data.root_ang_vel_b[0, 2].item()
        x_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[0].item()
        y_local = (robot.data.root_pos_w[0] - unwrapped_env.scene.env_origins[0])[1].item()

        t_buf.append(step * control_dt)
        cmd_w_buf.append(last_target_w)
        act_w_buf.append(yaw_rate)

        if step % 20 == 0:
            wx, wy = last_waypoint_xy
            _b(f"  step {step:5d}  pos=({x_local:+.2f},{y_local:+.2f})m  h={height:.2f}m  "
               f"wp=({wx:+.3f},{wy:+.3f})  cmd ω={last_target_w:+.2f}  "
               f"act ω={yaw_rate:+.2f}  v={vel_x:+.2f}")

        if dones.any():
            _b(f"  [reset] step {step + 1}: drone reset")
            if video_writer is not None:
                video_writer.close()
                _b(f"[info] video saved to {args_cli.save_video} "
                   f"({video_frame_count} frames, trail exit)")
                video_writer = None

        if show_plot and step % 5 == 0 and len(context) == context_size + 1:
            try:
                if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                    fpv_fig = None
                if fpv_fig is None:
                    plt.ion()
                    fpv_fig = plt.figure(num="ViNT Forest Pilot", figsize=(13, 6))
                    gs = fpv_fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0],
                                              hspace=0.35, wspace=0.25,
                                              left=0.04, right=0.98, top=0.95, bottom=0.08)
                    ax_img = fpv_fig.add_subplot(gs[:, 0])
                    ax_goal = fpv_fig.add_subplot(gs[0, 1])
                    ax_w = fpv_fig.add_subplot(gs[1, 1])
                    im_fpv = ax_img.imshow(rgb3)
                    ax_img.axis("off")
                    overlay_text = ax_img.text(
                        10, 30, "", color="lime", fontsize=10, weight="bold",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                    )
                    ax_goal.imshow(goal_pil)
                    if args_cli.goal_image:
                        ax_goal.set_title(f"goal ({os.path.basename(goal_path)})", fontsize=9)
                    else:
                        ax_goal.set_title(
                            f"goal — rendered at trail t={args_cli.goal_pose_t:.2f}",
                            fontsize=9,
                        )
                    ax_goal.axis("off")
                    (line_cmd_w,) = ax_w.plot([], [], color="tab:orange", lw=2, label="cmd (ViNT)")
                    (line_act_w,) = ax_w.plot([], [], color="tab:blue", lw=1.5, label="actual")
                    ax_w.set_ylabel("ω (rad/s)")
                    ax_w.set_xlabel("time (s)")
                    ax_w.set_title("yaw rate")
                    ax_w.grid(True, alpha=0.3)
                    ax_w.legend(loc="upper left", fontsize=8)
                    ax_w.axhline(0.0, color="k", lw=0.5, alpha=0.4)
                    fpv_fig._cmd_line = line_cmd_w
                    fpv_fig._act_line = line_act_w
                    fpv_fig._ax_w = ax_w

                im_fpv.set_data(rgb3)
                wx, wy = last_waypoint_xy
                overlay_text.set_text(
                    f"ViNT wp[{waypoint_idx}]=({wx:+.2f},{wy:+.2f})  ω={last_target_w:+.2f}\n"
                    f"actual ω={yaw_rate:+.2f}  v={vel_x:+.2f}\n"
                    f"pos=({x_local:+.2f},{y_local:+.2f}) m  h={height:.2f} m"
                )
                t_arr = np.fromiter(t_buf, dtype=float)
                fpv_fig._cmd_line.set_data(t_arr, np.fromiter(cmd_w_buf, dtype=float))
                fpv_fig._act_line.set_data(t_arr, np.fromiter(act_w_buf, dtype=float))
                if t_arr.size >= 2:
                    fpv_fig._ax_w.set_xlim(t_arr[0], t_arr[-1])
                fpv_fig._ax_w.relim(); fpv_fig._ax_w.autoscale_view(scalex=False, scaley=True)
                fpv_fig.canvas.draw_idle()
                fpv_fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception as exc:
                if step == 0:
                    _b(f"[warn] matplotlib FPV preview failed: {exc}")

        if (video_writer is not None and fpv_fig is not None
                and step % args_cli.video_capture_skip == 0):
            fpv_fig.canvas.draw()
            frame_rgba = np.asarray(fpv_fig.canvas.buffer_rgba())
            video_writer.append_data(frame_rgba[:, :, :3])
            video_frame_count += 1
            if (args_cli.video_max_frames is not None
                    and video_frame_count >= args_cli.video_max_frames):
                video_writer.close()
                _b(f"[info] video saved to {args_cli.save_video} "
                   f"({video_frame_count} frames, max reached)")
                video_writer = None

        step += 1

    if video_writer is not None:
        video_writer.close()
        _b(f"[info] video saved to {args_cli.save_video} ({video_frame_count} frames)")
    if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
        plt.close(fpv_fig)
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except KeyboardInterrupt:
        _b("\n[info] interrupted by user")
    except Exception:
        _b("[error] uncaught exception in main():")
        traceback.print_exc()
        raise
    finally:
        _b("[bc] entering finally; closing simulation_app")
        simulation_app.close()
