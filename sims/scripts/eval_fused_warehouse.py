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
# FusedSensorNet model definition (fused_model.py + ViTsubmodules): prefer a
# vendored copy alongside this tree, else fall back to the collaborator's
# read-only vitfly checkout. Override with MODELBLASTER_VITFLY_MODELS.
_vitfly_candidates = [
    os.environ.get("MODELBLASTER_VITFLY_MODELS", ""),
    os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")),
    "/scratch/agustin/projects/DIMA/vitfly/models",
]
for _vm in _vitfly_candidates:
    if _vm and os.path.isfile(os.path.join(_vm, "fused_model.py")):
        sys.path.insert(0, _vm)
        break
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--weights", type=str, required=True, help="FusedSensorNet BC checkpoint (out_dim=2).")
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math as _math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
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


def main():
    cfg_cls = WarehouseNavEnvCfg_PLAY_WithSensors if args_cli.visual_gates \
        else WarehouseNavEnvCfg_PLAY_WithSensors_Coll
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.obstacle_count.params["min_level"] = args_cli.obstacle_level
    if args_cli.prop_density is not None:  # crowded course: enable the tall-thin stacked-prop field
        env_cfg.events.reset_obstacles.params["prop_density"] = args_cli.prop_density
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
    _sd = torch.load(args_cli.weights, map_location=dev, weights_only=True)
    _venc = "cnn" if any(k.startswith("vision_cnn.") for k in _sd) else "vit"  # auto-detect encoder
    model = FusedSensorNet(out_dim=2, vision_encoder=_venc).to(dev).eval()
    model.load_state_dict(_sd, strict=True)
    masked = {m for m in args_cli.mask_off.split(",") if m}
    if args_cli.cam_mask:
        masked.add("front_grey")
    mask = {m: False for m in masked} if masked else None
    log(f"[nav] FusedSensorNet out_dim=2 loaded: {args_cli.weights}  mask_off={sorted(masked) or 'none'}")

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
            with torch.no_grad():
                cmd, hidden = model(inp, hidden, mask=mask)
            yaw_rate = float(cmd[0, 0].item())
            fwd = args_cli.fixed_speed if args_cli.fixed_speed > 0 else float(max(0.1, min(MAX_SPEED, cmd[0, 1].item())))
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
    tag = ("_off-" + "-".join(sorted(masked))) if masked else ""
    out = args_cli.out or os.path.join(out_dir, f"warehousenav_fused{tag}.json")
    json.dump({"agg": agg, "episodes": results, "args": vars(args_cli)}, open(out, "w"), indent=2)
    log(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
