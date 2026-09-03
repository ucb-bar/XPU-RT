"""Closed-loop eval for the fused-sensor nav model in the PHOTOREAL warehouse (task #64).

Warehouse port of ``eval_forest_nav_fused.py``. The FusedSensorNet student consumes the
SAME onboard sensor suite (front greyscale + 4x8x8 cross ToF + down-ToF + optical-flow +
baro + Madgwick attitude), carries the LSTM hidden across steps, emits (yaw_rate,
forward_speed), and — instead of the forest steering inner-loop — drives the warehouse's own
cascade-stable geometric ``VelocityCommandAction`` (mapped a0=fwd-1, a2=yaw/max_yawrate).

Honest metrics in the real aisle: collidable gate frames + rack rows + curriculum prop field
+ patrolling people are ALL real colliders, so a mis-fly ends the episode via the contact
(collision) termination. success = flew through all 4 gates; crash = collision / ground.

    <env_isaaclab py> sims/scripts/eval_fused_warehouse.py --headless \
        --weights <fused_bc_warehouse>/best.pt --episodes 6 [--mask_off desired_vel] \
        [--save_video out/warehouse_gate_chase.mp4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
# vitfly lives outside this repo, so let an env var override the sibling-checkout
# default (ported from rose-2-dev, whose copy hardcoded a per-user path).
for _vitfly_models in (os.environ.get("MODELBLASTER_VITFLY_MODELS", ""),
                       os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models"))):
    if _vitfly_models and os.path.isfile(os.path.join(_vitfly_models, "fused_model.py")):
        sys.path.insert(0, _vitfly_models)
        break
else:  # nothing found -- keep the historical default so the import error names it
    sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--weights", type=str,
                    default=os.path.join(freshscheduler_root, "sims/models/warehouse/nav_fused_v12_cnn.pt"),
                    help="FusedSensorNet BC checkpoint (out_dim=2). Used only for --nav fused.")
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--base_speed", type=float, default=1.4)
parser.add_argument("--obstacle_level", type=int, default=6, help="active prop count for the eval (honest difficulty).")
parser.add_argument("--prop_density", type=float, default=None,
                    help="Override aisle tall-thin stacked-prop density [0..1]; >0 = crowded ToF-avoidance course.")
parser.add_argument("--cam_mask", action="store_true", help="Zero-skip the camera (alias for --mask_off front_grey).")
parser.add_argument("--mask_off", type=str, default="",
                    help="Comma-list of modalities to zero-skip (#62 ablation / Stage-2 vision via "
                         "'desired_vel'). Valid: front_grey,tof_cross,optical_flow,down_tof,baro,"
                         "quat,body_rates,desired_vel.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--fixed_speed", type=float, default=0.0,
                    help="If >0, override the model's forward-speed output with this constant cruise "
                         "(m/s) and use ONLY the model's yaw. The speed head underfits (trainer "
                         "down-weights speed loss); the model's real job is steering.")
parser.add_argument("--hidden_reset_steps", type=int, default=0,
                    help="Reset the LSTM hidden every N steps (0=never). Tests whether long-horizon "
                         "hidden drift (train uses 32-step windows) degrades closed-loop control.")
parser.add_argument("--visual_gates", action="store_true",
                    help="Use VISUAL (non-collidable) gate frames. Default = collidable gates so a "
                         "100%% gate-pass rate is an honest 'flew through the opening' result.")
parser.add_argument("--save_video", type=str, default=None,
                    help="Record a demo .mp4 (prefers 3rd-person chase view). Default = first successful "
                         "weave-through (else deepest run). With --record_all, records EVERY episode.")
parser.add_argument("--record_all", action="store_true",
                    help="Record every episode (crashes included) to one continuous video, not just the "
                         "first success. Use with --save_video for the full-run recording.")
parser.add_argument("--dump_calib", type=str, default=None,
                    help="Capture the packed [N,5677] model-input vectors (wire layout: front_grey@60x90 | "
                         "tof_cross | flow | down_tof | baro | quat | body_rates | desired_vel | flags) to an "
                         ".npz for int8 calibration (Agent A / ModelBlaster). Subsampled to --calib_max samples.")
parser.add_argument("--calib_max", type=int, default=512, help="Max calibration samples to keep (evenly strided).")
parser.add_argument("--dump_golden", type=str, default=None,
                    help="Save one fresh-hidden (input[1,5677], output[1,2]) fp32 golden pair from the "
                         "first inference step (hidden=None) for the ModelBlaster exact-verify gate.")
parser.add_argument("--controller", type=str, default="geom", choices=["geom", "mlp"],
                    help="Low-level controller: 'geom' = classical VelocityCommandAction (E0 baseline); "
                         "'mlp' = the distilled MLP network (E1).")
parser.add_argument("--yolo", type=str, default=None, help="warehouse YOLOv8n best.pt (Stage-2 loop closure).")
parser.add_argument("--safety", action="store_true",
                    help="close the YOLO->nav safety loop (person/obstacle-ahead -> slow+steer via hil/safety_layer).")
# --- ViNT nav (Workstream F): drive the warehouse aisle with the pretrained ViNT foundation model ---
parser.add_argument("--nav", type=str, default="fused", choices=["fused", "vint"],
                    help="fused = our trained FusedSensorNet nav; vint = pretrained ViNT (outdoor-trained, "
                         "OOD indoors) steering by image-goal, with cross-ToF fusion for collision avoidance.")
parser.add_argument("--vint_ckpt", type=str,
                    default=os.path.join(freshscheduler_root,
                                         "sims/external/visualnav-transformer/deployment/model_weights/vint.pth"))
parser.add_argument("--waypoint_idx", type=int, default=2, help="which of ViNT's 5 waypoints to steer to.")
parser.add_argument("--omega_gain", type=float, default=2.5, help="ViNT waypoint angle (rad) -> yaw rate gain.")
parser.add_argument("--vint_fwd", type=float, default=0.9, help="forward speed for ViNT nav (m/s).")
parser.add_argument("--no_tof_fusion", action="store_true",
                    help="disable the cross-ToF collision-avoidance overlay on ViNT (pure ViNT).")
parser.add_argument("--screenshot", type=str, default=None,
                    help="save a PNG of the chase frame at the last gate / episode end (aisle-end shot).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math as _math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as _F  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (  # noqa: E402
    WarehouseNavEnvCfg_PLAY_WithSensors, WarehouseNavEnvCfg_PLAY_WithSensors_Coll,
)
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg  # noqa: E402
from fused_model import FusedSensorNet  # noqa: E402

_VCFG = VelocityCommandActionCfg()
MAX_SPEED, MAX_YAWRATE = _VCFG.max_speed, _VCFG.max_yawrate
TASK_ID = ("Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0" if args_cli.visual_gates
           else "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0")
GATE_CENTERS_2D = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)  # (K,2)
PASS_RADIUS = GW.FixedGateCourseCommandCfg().success_radius


def log(m):
    print(m, flush=True)


TARGET_H = 2.0        # altitude-hold setpoint (m); matches FUSED_GATES centre z, clears aisle clutter
_K_ALT = 1.2          # altitude P-gain (vz per metre of error)
_VZ_MAX = 0.8         # clamp on the commanded climb velocity (m/s)
_MAX_INCL = _VCFG.max_inclination


def cmd_to_action(yr, fwd, h, dev, N):
    """Map the nav model's (yaw_rate, forward_speed) + an altitude-hold loop → velocity action.

    Horizontal guidance is the model's job; altitude is a fixed low-level autopilot that drives
    the drone to TARGET_H regardless of spawn-height jitter. The controller decodes vz from the
    inclination channel a1: vz = speed·sin(max_incl·a1)·(max_speed/2), speed = a0+1. Invert that
    for the desired climb velocity vz_des = K·(TARGET_H − h)."""
    a0 = float(fwd) / (MAX_SPEED / 2.0) - 1.0
    a2 = float(yr) / MAX_YAWRATE
    speed = max(0.05, a0 + 1.0)                       # decoded speed scalar (== vx at zero incl)
    vz_des = max(-_VZ_MAX, min(_VZ_MAX, _K_ALT * (TARGET_H - float(h))))
    s = max(-1.0, min(1.0, vz_des / speed))           # sin(max_incl·a1) = vz_des/speed
    a1 = _math.asin(s) / _MAX_INCL
    return torch.tensor([[a0, a1, a2, 0.0]], device=dev, dtype=torch.float32).clamp(-1.0, 1.0).repeat(N, 1)


class GateGeometry:
    """Perpendicular distance / arc-progress along the gate-course polyline (start + gate centres)."""

    def __init__(self, start_xy):
        self.pts = np.concatenate([np.asarray(start_xy, dtype=np.float64)[None, :], GATE_CENTERS_2D], axis=0)
        seg = self.pts[1:] - self.pts[:-1]
        self.seg = seg
        self.seg_sq = np.maximum((seg ** 2).sum(1), 1e-9)

    def offset(self, xy):
        p0 = self.pts[:-1]
        diff = xy[None, :] - p0
        t = np.clip((diff * self.seg).sum(1) / self.seg_sq, 0.0, 1.0)
        closest = p0 + t[:, None] * self.seg
        return float(np.linalg.norm(xy[None, :] - closest, axis=1).min())


def _load_vint(ckpt_path, device):
    """Lazy ViNT loader (Workstream F). Mirrors pilot_forest_with_vint._load_vint: build ViNT from the
    published vint.yaml, load the {'model': DataParallel} checkpoint. Returns (model, cfg, transform)."""
    import yaml
    from torchvision import transforms
    vint_root = os.path.join(freshscheduler_root, "sims/external/visualnav-transformer/train")
    if vint_root not in sys.path:
        sys.path.insert(0, vint_root)
    # vint_train pulls training-only deps at import; stub warmup_scheduler (a package with a
    # .scheduler submodule exposing GradualWarmupScheduler) — inference doesn't need it.
    import types as _types
    if "warmup_scheduler" not in sys.modules:
        _GW = type("GradualWarmupScheduler", (object,), {"__init__": lambda self, *a, **k: None})
        _pkg = _types.ModuleType("warmup_scheduler"); _pkg.__path__ = []
        _sub = _types.ModuleType("warmup_scheduler.scheduler")
        _sub.GradualWarmupScheduler = _GW; _pkg.GradualWarmupScheduler = _GW; _pkg.scheduler = _sub
        sys.modules["warmup_scheduler"] = _pkg
        sys.modules["warmup_scheduler.scheduler"] = _sub
    from vint_train.models.vint.vint import ViNT  # noqa: PLC0415
    cfg = yaml.safe_load(open(os.path.join(vint_root, "config/vint.yaml")))
    model = ViNT(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
                 learn_angle=cfg["learn_angle"], obs_encoder=cfg["obs_encoder"],
                 obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
                 mha_num_attention_heads=cfg["mha_num_attention_heads"],
                 mha_num_attention_layers=cfg["mha_num_attention_layers"],
                 mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        loaded = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        sd = loaded.module.state_dict() if hasattr(loaded, "module") else loaded.state_dict()
        model.load_state_dict(sd, strict=False)
        log(f"[vint] loaded {ckpt_path}")
    else:
        log(f"[vint] WARNING checkpoint not found: {ckpt_path} (random init — meaningless output)")
    isz = cfg.get("image_size", [85, 64])
    tf = transforms.Compose([transforms.Resize((isz[1], isz[0])), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return model.to(device).eval(), cfg, tf


def _tof_fusion(yaw_rate, fwd, tof_cross, near_is_high=True):
    """Cross-ToF collision-avoidance overlay for ViNT: bias yaw toward the open side, slow when the
    front cell is close. tof_cross: (4,8,8) normalized [N,E,S,W]. Returns (yaw_rate, fwd, trig)."""
    import numpy as _np
    t = _np.asarray(tof_cross, dtype=_np.float32)
    prox = t if near_is_high else (1.0 - t)   # high = close
    n, e, s, w = [float(_np.nanmean(prox[i])) for i in range(4)]
    trig = None
    # steer away from the closer of E/W
    if abs(e - w) > 0.08:
        yaw_rate = yaw_rate + (w - e) * 1.6   # if E closer (e>w) -> turn left (- ), sign folded into (w-e)
        trig = "avoid"
    if n > 0.6:                                # obstacle straight ahead -> slow
        fwd = fwd * max(0.2, 1.0 - (n - 0.6) * 2.0)
        trig = "brake"
    return float(yaw_rate), float(fwd), trig


def main():
    cfg_cls = WarehouseNavEnvCfg_PLAY_WithSensors if args_cli.visual_gates \
        else WarehouseNavEnvCfg_PLAY_WithSensors_Coll
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.obstacle_count.params["min_level"] = args_cli.obstacle_level
    if args_cli.prop_density is not None:  # crowded course: enable the tall-thin stacked-prop field
        env_cfg.events.reset_obstacles.params["prop_density"] = args_cli.prop_density
    if args_cli.controller == "mlp":  # swap the classical controller for the distilled MLP (E1)
        from sims.isaaclab_tasks.warehouse_nav.mdp_mlp_control_action import MLPVelocityCommandActionCfg
        env_cfg.actions.velocity = MLPVelocityCommandActionCfg(asset_name="robot", body_name="body")
        log("[ctrl] using distilled MLP low-level controller (E1)")
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    out_dir = os.path.dirname(args_cli.out) if args_cli.out else \
        "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad"

    log(f"[env] gym.make {TASK_ID}")
    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device
    N = uenv.num_envs
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    est = StateEstimator(N, dev, control_dt=control_dt)
    from collections import deque as _deque
    from PIL import Image as _PILImage
    model = None
    vint_model = vint_cfg = vint_tf = None
    vint_ctx = None
    masked = {m for m in args_cli.mask_off.split(",") if m}
    if args_cli.cam_mask:
        masked.add("front_grey")
    mask = {m: False for m in masked} if masked else None
    if args_cli.nav == "vint":
        vint_model, vint_cfg, vint_tf = _load_vint(args_cli.vint_ckpt, dev)
        vint_ctx = _deque(maxlen=int(vint_cfg["context_size"]) + 1)
        log(f"[nav] ViNT foundation nav (image-goal) · context={vint_cfg['context_size']} · "
            f"tof_fusion={'off' if args_cli.no_tof_fusion else 'ON'}")
    else:
        _sd = torch.load(args_cli.weights, map_location=dev, weights_only=True)
        if isinstance(_sd, dict) and "model_state_dict" in _sd:
            _sd = _sd["model_state_dict"]
        _venc = "cnn" if any(k.startswith("vision_cnn.") for k in _sd) else "vit"  # auto-detect encoder
        # Infer LSTM dims from the checkpoint (ViT v19: 256/4) so it loads strict=True.
        _lh, _ll = 128, 3
        if "lstm.weight_hh_l0" in _sd:
            _lh = _sd["lstm.weight_hh_l0"].shape[1]
            _ll = sum(1 for k in _sd if k.startswith("lstm.weight_ih_l") and k.endswith(tuple("0123456789")))
            _ll = max(_ll, 1)
        model = FusedSensorNet(out_dim=2, vision_encoder=_venc, lstm_hidden=_lh, lstm_layers=_ll).to(dev).eval()
        model.load_state_dict(_sd, strict=True)
        log(f"[nav] FusedSensorNet out_dim=2 loaded: {args_cli.weights}  encoder={_venc} "
            f"lstm={_lh}/{_ll}  mask_off={sorted(masked) or 'none'}")

    _yolo = None; _safety = None
    if args_cli.yolo:
        from ultralytics import YOLO as _YOLO
        _yolo = _YOLO(args_cli.yolo)
        sys.path.insert(0, os.path.join(freshscheduler_root, "hil"))
        from safety_layer import apply_safety as _safety
        log(f"[yolo] detector loaded: {args_cli.yolo}  safety_loop={'ON' if args_cli.safety else 'OFF (observer)'}")

    def _run_yolo(grey_bchw):
        g = grey_bchw
        if g.shape[-2:] != (60, 90):
            g = _F.interpolate(g, size=(60, 90), mode="bilinear", align_corners=False)
        img = (g[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img3 = np.repeat(img[:, :, None], 3, axis=2)
        r = _yolo.predict(img3, imgsz=96, conf=0.30, verbose=False)[0]
        dets = []
        for b in r.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            dets.append({"cls": int(b.cls[0].item()), "conf": float(b.conf[0].item()),
                         "xywh": ((x0 + x1) / 2 / 90.0, (y0 + y1) / 2 / 60.0, (x1 - x0) / 90.0, (y1 - y0) / 60.0)})
        return dets

    robot = uenv.scene["robot"]
    origin = uenv.scene.env_origins
    K = len(GATE_CENTERS_2D)
    geom = GateGeometry(start_xy=(-8.0, 6.0))

    def _yaw_of(q):
        w, x, y, z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        return float(torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).item())

    def sense(desired_vel):
        grey = S.front_greyscale(uenv)
        tof_norm, _ = S.normalize_range(S.tof_stack(uenv), S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
        dtof = S.down_tof(uenv)
        dtof_norm, dtof_valid = S.normalize_range(dtof, S.DOWN_TOF_RANGE_MIN, S.DOWN_TOF_RANGE_MAX)
        flow = S.optical_flow(uenv); flow_valid = S.optical_flow_valid(uenv)
        baro = S.barometer(uenv, drift=est.step_baro_drift())
        gyro = robot.data.root_ang_vel_b[:, :3]
        accel = -robot.data.projected_gravity_b * 9.81
        filt = est.update(gyro, accel, baro_alt=baro[:, 1], tof_alt=dtof.squeeze(1), flow_vel=flow * 0.0)
        return {"front_grey": grey.float(), "tof_cross": tof_norm, "optical_flow": flow, "down_tof": dtof_norm,
                "baro": baro / 10.0, "quat": filt["quat"], "body_rates": gyro, "desired_vel": desired_vel,
                "flags": torch.cat([flow_valid, dtof_valid, torch.ones(N, 4, device=dev)], dim=1)}

    # optional demo recorder (first 2 episodes), prefer chase cam
    vwriter = None; demo_cam = None; chase = None
    if args_cli.save_video:
        import imageio
        vwriter = imageio.get_writer(args_cli.save_video, fps=30, codec="libx264", quality=8, macro_block_size=None)
        if "chase_camera" in uenv.scene.sensors:
            chase = uenv.scene["chase_camera"]; demo_cam = chase
            log(f"[demo] recording 3rd-person chase view → {args_cli.save_video}")
        else:
            demo_cam = uenv.scene["front_camera"]
            log(f"[demo] recording onboard FPV → {args_cli.save_video}")

    def _drive_chase(cam):
        p = robot.data.root_pos_w[0]; q = robot.data.root_quat_w[0]
        w, x, y, z = q[0], q[1], q[2], q[3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fx, fy = torch.cos(yaw), torch.sin(yaw)
        eye = torch.stack([p[0] - fx * 3.0, p[1] - fy * 3.0, p[2] + 1.6])
        tgt = torch.stack([p[0] + fx * 4.0, p[1] + fy * 4.0, p[2] - 0.1])
        cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))

    def _pack_calib(inp):
        """Pack the sense() dict into the wire-layout [1,5677] vector (front_grey resized to 60x90,
        exactly as the model consumes it). Order matches hil/hil_protocol.SENSOR_LAYOUT[:5677]."""
        g = inp["front_grey"]
        if g.shape[-2:] != (60, 90):
            g = _F.interpolate(g, size=(60, 90), mode="bilinear", align_corners=False)
        parts = [g.reshape(N, -1), inp["tof_cross"].reshape(N, -1), inp["optical_flow"],
                 inp["down_tof"], inp["baro"], inp["quat"], inp["body_rates"],
                 inp["desired_vel"], inp["flags"]]
        return torch.cat(parts, dim=1).detach().cpu().numpy().astype(np.float32)  # (N,5677)

    calib_buf = [] if args_cli.dump_calib else None
    _golden_saved = {"done": False}
    # ViNT image-goal: for the straight aisle we use a forward-view proxy goal (set from an early
    # frame). ViNT then contributes steering; the constant vint_fwd + cross-ToF fusion carry the
    # tunnel-following + collision avoidance (honest: ViNT is OOD indoors, ToF does the safety).
    _vint_goal = [None]
    _shot_saved = {"done": False}
    results = []
    best_frames, best_prog, captured = None, -1.0, False  # demo: keep first success, else deepest run
    for ep in range(args_cli.episodes):
        torch.manual_seed(args_cli.seed + ep)
        if hasattr(est, "reset"):
            est.reset()
        reset_out = env.reset()
        _obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        hidden = None
        ep_frames = []
        off_sum, off_n, off_max = 0.0, 0, 0.0
        prev_w, jerk_sum, outcome, last_h = None, 0.0, "timeout", 1.0
        goal_idx, gates_passed = 0, 0
        for t in range(args_cli.max_steps):
            xy_now = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
            h_now = float(robot.data.root_pos_w[0, 2].item())
            yaw_now = _yaw_of(robot.data.root_quat_w)
            goal_xy = GATE_CENTERS_2D[min(goal_idx, K - 1)]
            if np.linalg.norm(xy_now - goal_xy) < PASS_RADIUS:
                if goal_idx < K - 1:
                    goal_idx += 1; goal_xy = GATE_CENTERS_2D[goal_idx]
                else:
                    gates_passed = K
            gates_passed = max(gates_passed, goal_idx)
            dvec = goal_xy - xy_now
            cy, sy = _math.cos(-yaw_now), _math.sin(-yaw_now)
            bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
            nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
            desired_vel = torch.tensor([[bx / nrm * args_cli.base_speed, by / nrm * args_cli.base_speed, 0.0]],
                                       device=dev, dtype=torch.float32).repeat(N, 1)

            if args_cli.hidden_reset_steps > 0 and t > 0 and t % args_cli.hidden_reset_steps == 0:
                hidden = None
            inp = sense(desired_vel)
            if calib_buf is not None:
                calib_buf.append(_pack_calib(inp))
            if args_cli.nav == "vint":
                # feed the onboard RGB FPV into ViNT's rolling context (WithSensors env exposes the
                # front cam under S.FRONT_CAM_KEY = "front_camera", rendered RGB)
                _rgb = uenv.scene[S.FRONT_CAM_KEY].data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if _rgb.dtype != np.uint8:
                    _rgb = (np.clip(_rgb, 0, 1) * 255).astype(np.uint8)
                vint_ctx.append(vint_tf(_PILImage.fromarray(_rgb)))
                # set the forward-view proxy goal once the drone is a little into the aisle
                if _vint_goal[0] is None and t >= 15:
                    _vint_goal[0] = vint_tf(_PILImage.fromarray(_rgb)).unsqueeze(0).to(dev)
                    log(f"[vint] goal image set at t={t} (forward aisle view)")
                yaw_rate = 0.0
                if len(vint_ctx) == vint_ctx.maxlen and _vint_goal[0] is not None:
                    obs_b = torch.cat(list(vint_ctx), dim=0).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        _dp, action_pred = vint_model(obs_b, _vint_goal[0])
                    deltas = action_pred[0, :, :2].cpu().numpy()
                    wps = np.cumsum(deltas, axis=0)
                    wp = wps[min(args_cli.waypoint_idx, len(wps) - 1)]
                    angle = float(np.arctan2(wp[1], wp[0]))
                    yaw_rate = max(-MAX_YAWRATE, min(MAX_YAWRATE, angle * args_cli.omega_gain))
                fwd = args_cli.vint_fwd
                if not args_cli.no_tof_fusion:
                    # inp["tof_cross"] is normalized (far=1, near=0) -> near_is_high=False
                    yaw_rate, fwd, _trig = _tof_fusion(yaw_rate, fwd,
                                                       inp["tof_cross"][0].cpu().numpy(), near_is_high=False)
            else:
                _fresh_hidden = hidden is None
                with torch.no_grad():
                    cmd, hidden = model(inp, hidden, mask=mask)
                # Golden: first inference runs with hidden=None -> reproducible (zero LSTM state).
                if args_cli.dump_golden and _fresh_hidden and not _golden_saved["done"]:
                    gi = _pack_calib(inp)[:1]
                    go = cmd[:1].detach().cpu().numpy().astype(np.float32)
                    os.makedirs(os.path.dirname(os.path.abspath(args_cli.dump_golden)), exist_ok=True)
                    np.savez(args_cli.dump_golden, input=gi, output=go)
                    _golden_saved["done"] = True
                    log(f"[golden] wrote {args_cli.dump_golden}  input{gi.shape} output{go.shape} out={go.tolist()}")
                yaw_rate = float(cmd[0, 0].item())
                fwd = args_cli.fixed_speed if args_cli.fixed_speed > 0 else float(max(0.1, min(MAX_SPEED, cmd[0, 1].item())))
                if _yolo is not None and args_cli.safety:  # Stage-2: YOLO detections modulate the command
                    _det = _run_yolo(inp["front_grey"])
                    (yaw_rate, fwd), _ = _safety((yaw_rate, fwd), _det)
            if ep < 3 and t % 20 == 0:
                log(f"    [trace ep{ep} t={t}] xy=({xy_now[0]:.2f},{xy_now[1]:.2f}) h={h_now:.2f} yaw={yaw_now:.2f} "
                    f"pred_yr={yaw_rate:+.2f} pred_fwd={fwd:.2f} goal={goal_idx} dvel_b=({float(desired_vel[0,0]):+.2f},{float(desired_vel[0,1]):+.2f})")
            if chase is not None:
                _drive_chase(chase)
            obs, _r, dones, _i = env.step(cmd_to_action(yaw_rate, fwd, h_now, dev, N))

            xy = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
            last_h = float(robot.data.root_pos_w[0, 2].item())
            wz = float(robot.data.root_ang_vel_b[0, 2].item())
            off = geom.offset(xy)
            off_sum += off; off_n += 1; off_max = max(off_max, off)
            if prev_w is not None:
                jerk_sum += abs(wz - prev_w)
            prev_w = wz
            if vwriter is not None:
                rgb = demo_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                ep_frames.append(np.ascontiguousarray(rgb))
            # aisle-end screenshot: grab the chase frame the moment the drone reaches the last gate
            if args_cli.screenshot and gates_passed >= K and not _shot_saved["done"]:
                _scam = uenv.scene["chase_camera"] if "chase_camera" in uenv.scene.sensors else uenv.scene["fpv_camera"]
                _srgb = _scam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if _srgb.dtype != np.uint8:
                    _srgb = (np.clip(_srgb, 0, 1) * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(os.path.abspath(args_cli.screenshot)) or ".", exist_ok=True)
                _PILImage.fromarray(np.ascontiguousarray(_srgb)).save(args_cli.screenshot)
                _shot_saved["done"] = True
                log(f"[screenshot] aisle-end shot saved -> {args_cli.screenshot} (gate {K}/{K}, ep{ep} t={t})")
            if gates_passed >= K:
                outcome = "success"; break
            if bool(dones[0].item()):
                try:
                    tm = uenv.termination_manager
                    terms = {nm: bool(tm.get_term(nm)[0].item()) for nm in tm.active_terms}
                except Exception:
                    terms = {}
                if terms.get("time_out"):
                    outcome = "timeout"
                elif any("collision" in nm and v for nm, v in terms.items()) or \
                        any("crash" in nm and v for nm, v in terms.items()) or last_h < 0.2:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                active = [nm for nm, v in terms.items() if v]
                log(f"    [done ep{ep:02d} t={t}] {active} pre-step xy=({xy_now[0]:.2f},{xy_now[1]:.2f}) "
                    f"h={last_h:.2f} goal={goal_idx}")
                break
        else:
            outcome = "success" if gates_passed >= K else "timeout"
        rec = {"episode": ep, "outcome": outcome, "gates_passed": gates_passed, "progress": round(gates_passed / K, 4),
               "mean_offset": round(off_sum / max(1, off_n), 4), "max_offset": round(off_max, 4),
               "mean_jerk": round(jerk_sum / max(1, off_n), 5), "steps": off_n}
        results.append(rec)
        log(f"[ep{ep:02d}] outcome={outcome:9s} gates={gates_passed}/{K} mean_off={rec['mean_offset']:.2f} steps={off_n}")
        # fallback aisle-end shot: if we never reached the last gate, save the deepest episode's
        # final chase frame so there is always an end-of-run screenshot.
        if args_cli.screenshot and not _shot_saved["done"] and (ep == args_cli.episodes - 1 or gates_passed >= K):
            try:
                _scam = uenv.scene["chase_camera"] if "chase_camera" in uenv.scene.sensors else uenv.scene["fpv_camera"]
                _srgb = _scam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if _srgb.dtype != np.uint8:
                    _srgb = (np.clip(_srgb, 0, 1) * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(os.path.abspath(args_cli.screenshot)) or ".", exist_ok=True)
                _PILImage.fromarray(np.ascontiguousarray(_srgb)).save(args_cli.screenshot)
                _shot_saved["done"] = True
                log(f"[screenshot] end-of-run shot saved -> {args_cli.screenshot} (gates {gates_passed}/{K})")
            except Exception as _e:
                log(f"[screenshot] fallback failed: {_e}")

        # demo recording. --record_all: append every episode continuously.
        # default: keep the first successful weave-through (deepest run as fallback).
        if vwriter is not None and args_cli.record_all:
            for f in ep_frames:
                vwriter.append_data(f)
            captured = True  # suppress fallback flush below
        elif vwriter is not None and not captured:
            if outcome == "success":
                for f in ep_frames:
                    vwriter.append_data(f)
                captured = True
                log(f"[demo] captured successful ep{ep:02d} ({gates_passed}/{K}) → {args_cli.save_video}")
                break
            elif rec["progress"] > best_prog:
                best_prog, best_frames = rec["progress"], ep_frames

    if vwriter is not None:
        if not captured and best_frames:
            for f in best_frames:
                vwriter.append_data(f)
            log(f"[demo] no full success in {len(results)} eps; kept deepest run (progress={best_prog:.2f})")
        vwriter.close()
        log(f"[demo] wrote {args_cli.save_video}")

    n = len(results)
    agg = {"env": "warehouse_gate", "nav_arch": "fused", "mask_off": sorted(masked), "episodes": n,
           "success_rate": round(sum(r["outcome"] == "success" for r in results) / max(1, n), 3),
           "mean_progress": round(sum(r["progress"] for r in results) / max(1, n), 3),
           "mean_offset": round(sum(r["mean_offset"] for r in results) / max(1, n), 3),
           "mean_jerk": round(sum(r["mean_jerk"] for r in results) / max(1, n), 5),
           "outcomes": {k: sum(r["outcome"] == k for r in results) for k in ("success", "crash", "timeout")}}
    log("\n=== FUSED WAREHOUSE NAV EVAL ===")
    log(json.dumps(agg, indent=2))
    # Save the calibration set FIRST — it is an Agent-A deliverable and must
    # never be lost to a downstream results-JSON write failure.
    if calib_buf:
        allc = np.concatenate(calib_buf, axis=0)  # (T,5677)
        if allc.shape[0] > args_cli.calib_max:     # evenly strided subsample for diversity
            idx = np.linspace(0, allc.shape[0] - 1, args_cli.calib_max).astype(np.int64)
            allc = allc[idx]
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.dump_calib)), exist_ok=True)
        np.savez(args_cli.dump_calib, input=allc)
        log(f"[calib] wrote {args_cli.dump_calib}  shape={allc.shape} (from {sum(c.shape[0] for c in calib_buf)} steps)")

    tag = ("_off-" + "-".join(sorted(masked))) if masked else ""
    out = args_cli.out or os.path.join(out_dir, f"warehousenav_fused{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"agg": agg, "episodes": results, "args": vars(args_cli)}, open(out, "w"), indent=2)
    log(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
