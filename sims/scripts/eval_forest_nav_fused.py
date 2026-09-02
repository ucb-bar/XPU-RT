"""Closed-loop eval for the fused-sensor nav model (task #56, M4).

The FusedSensorNet counterpart of eval_forest_nav: same seeded-episode /
progress / offset / outcome metrics, but the student consumes the FULL onboard
sensor suite (front greyscale + 4x8x8 cross ToF + down-ToF + optical-flow + baro
+ Madgwick-filtered attitude) assembled exactly as in collect_fused_data, runs
FusedSensorNet(out_dim=2) carrying the LSTM hidden across steps, and emits
(yaw_rate, forward_speed) into the steering inner-loop. Requires a *_WithSensors
env so the sensors exist.

    <env_isaaclab py> sims/scripts/eval_forest_nav_fused.py --headless \
        --weights <fused_bc>/best.pt --trail straight --episodes 6

Compare its offset table head-to-head with the DroNet/ViT rows from eval_forest_nav.
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
sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--weights", type=str, required=True, help="FusedSensorNet BC checkpoint (out_dim=2).")
parser.add_argument("--trail", choices=["straight", "curved", "slalom", "gate"], default="straight")
parser.add_argument("--with_humans", action="store_true")
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--success_frac", type=float, default=0.9)
parser.add_argument("--cam_mask", action="store_true", help="Zero-skip the camera (alias for --mask_off front_grey).")
parser.add_argument("--mask_off", type=str, default="",
                    help="Comma-list of modalities to zero-skip for the sensor-aggregation ablation "
                         "(#62), e.g. 'front_grey,down_tof,baro'. Valid: front_grey,tof_cross,"
                         "optical_flow,down_tof,baro,quat,body_rates,desired_vel.")
parser.add_argument("--inner_ckpt", type=str,
                    default="/scratch2/dima/misc_sw/FreshScheduler/logs/rsl_rl/"
                            "crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--gate_seed", type=int, default=0,
                    help="Gate-course DR seed (0=canonical) — eval on a held-out layout to test generalization.")
parser.add_argument("--save_video", type=str, default=None,
                    help="Record the onboard FPV (front camera) for the first 2 episodes to this .mp4 (demo).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
os.environ["FOREST_GATE_SEED"] = str(args_cli.gate_seed)  # DR gate layout (before forest imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.forest_trail.config.crazyflie  # noqa: E402,F401
import sims.isaaclab_tasks.track_steering_vision.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (  # noqa: E402
    ForestTrailEnvCfg_PLAY_WithSensors, ForestTrailEnvCfg_PLAY_WithHumans_WithSensors,
    ForestTrailEnvCfg_Curved_PLAY_WithSensors, ForestTrailEnvCfg_Curved_PLAY_WithHumans_WithSensors,
    ForestTrailEnvCfg_Slalom_PLAY_WithSensors,
    ForestTrailEnvCfg_Gates_PLAY_WithSensors,
)
from sims.isaaclab_tasks.forest_trail.gates import FOREST_GATE_CENTERS_2D, PASS_RADIUS  # noqa: E402
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    SteeringTrackingPPORunnerCfg,
)
from fused_model import FusedSensorNet  # noqa: E402


def log(m):
    print(m, flush=True)


# ---- trail geometry (identical to eval_forest_nav.Geometry) ----
class Geometry:
    def __init__(self, env_cfg, curved):
        self.curved = curved
        p = env_cfg.terminations.off_trail.params
        if curved:
            self.pts = np.asarray(p["waypoints"], dtype=np.float64)
            seg = self.pts[1:] - self.pts[:-1]
            self.seg_len = np.linalg.norm(seg, axis=1)
            self.total_arc = float(self.seg_len.sum())
            self.arc_starts = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
            self.lateral_margin = float(p.get("lateral_margin", 3.0))
        else:
            self.trail_length = float(p.get("trail_length", 30.0))
            self.lateral_margin = float(p.get("lateral_margin", 3.0))

    def progress_offset(self, xy):
        if not self.curved:
            return xy[0] / self.trail_length, abs(xy[1])
        p0, p1 = self.pts[:-1], self.pts[1:]
        seg = p1 - p0
        seg_sq = np.maximum((seg ** 2).sum(1), 1e-9)
        diff = xy[None, :] - p0
        t = np.clip((diff * seg).sum(1) / seg_sq, 0.0, 1.0)
        closest = p0 + t[:, None] * seg
        dist = np.linalg.norm(xy[None, :] - closest, axis=1)
        j = int(dist.argmin())
        arc = self.arc_starts[j] + t[j] * self.seg_len[j]
        return arc / self.total_arc, float(dist[j])


def load_inner_policy(env, device, ckpt, out_dir):
    agent_cfg = SteeringTrackingPPORunnerCfg()
    runner_cfg = {
        "num_steps_per_env": agent_cfg.num_steps_per_env, "max_iterations": agent_cfg.max_iterations,
        "algorithm": agent_cfg.algorithm.to_dict(),
        "actor": {"class_name": agent_cfg.actor.class_name, "hidden_dims": agent_cfg.actor.hidden_dims,
                  "activation": agent_cfg.actor.activation, "obs_normalization": agent_cfg.actor.obs_normalization,
                  "distribution_cfg": agent_cfg.actor.distribution_cfg.to_dict() if agent_cfg.actor.distribution_cfg else None},
        "critic": {"class_name": agent_cfg.critic.class_name, "hidden_dims": agent_cfg.critic.hidden_dims,
                   "activation": agent_cfg.critic.activation, "obs_normalization": agent_cfg.critic.obs_normalization},
        "obs_groups": agent_cfg.obs_groups,
    }
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=device)
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    asd = loaded.get("actor_state_dict", {})
    if "distribution.std_param" in asd and "distribution.log_std_param" not in asd:
        asd["distribution.log_std_param"] = asd.pop("distribution.std_param").clamp_min(1e-6).log()
        ckpt = os.path.join(out_dir, "_model_6998_logstd.pt")
        torch.save(loaded, ckpt)
    runner.load(ckpt)
    return runner.get_inference_policy(device=device)


def main():
    curved = args_cli.trail == "curved"
    if args_cli.trail == "gate":
        cfg_cls = ForestTrailEnvCfg_Gates_PLAY_WithSensors
        task_id = "Isaac-Forest-Gates-Vision-Crazyflie-Play-WithSensors-v0"
    elif args_cli.trail == "slalom":
        cfg_cls = ForestTrailEnvCfg_Slalom_PLAY_WithSensors
        task_id = "Isaac-Forest-Trail-Slalom-Vision-Crazyflie-Play-WithSensors-v0"
    elif curved:
        cfg_cls = (ForestTrailEnvCfg_Curved_PLAY_WithHumans_WithSensors if args_cli.with_humans
                   else ForestTrailEnvCfg_Curved_PLAY_WithSensors)
        task_id = ("Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-WithSensors-v0" if args_cli.with_humans
                   else "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithSensors-v0")
    else:
        cfg_cls = (ForestTrailEnvCfg_PLAY_WithHumans_WithSensors if args_cli.with_humans
                   else ForestTrailEnvCfg_PLAY_WithSensors)
        task_id = ("Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-WithSensors-v0" if args_cli.with_humans
                   else "Isaac-Forest-Trail-Vision-Crazyflie-Play-WithSensors-v0")
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = 1
    # extend the episode so a full traverse fits (default ~10 s time-out caps at ~10 m);
    # we bound episodes by --max_steps ourselves.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    geom = Geometry(env_cfg, curved)
    out_dir = os.path.dirname(args_cli.out) if args_cli.out else \
        "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad"

    log(f"[env] gym.make {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device
    N = uenv.num_envs
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    inner_policy = load_inner_policy(env, dev, args_cli.inner_ckpt, out_dir)
    est = StateEstimator(N, dev, control_dt=control_dt)
    model = FusedSensorNet(out_dim=2).to(dev).eval()
    model.load_state_dict(torch.load(args_cli.weights, map_location=dev, weights_only=True), strict=True)
    masked = {m for m in args_cli.mask_off.split(",") if m}
    if args_cli.cam_mask:
        masked.add("front_grey")
    mask = {m: False for m in masked} if masked else None
    log(f"[nav] FusedSensorNet out_dim=2 loaded: {args_cli.weights}  mask_off={sorted(masked) or 'none'}")
    goal = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(N, 1)

    steering_term = uenv.command_manager.get_term("steering_command")
    robot = uenv.scene["robot"]
    origin = uenv.scene.env_origins

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

    import math as _math

    def _yaw_of(q):  # q:(N,4) wxyz -> float yaw of env 0
        w, x, y, z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        return float(torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).item())

    is_gate = args_cli.trail == "gate"
    gate_centers = np.asarray(FOREST_GATE_CENTERS_2D, dtype=np.float64) if is_gate else None
    K = len(gate_centers) if is_gate else 0
    fwd_goal = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(N, 1)

    # optional demo recorder (first 2 episodes) — prefer the 3rd-person chase camera.
    vwriter = None; demo_cam = None; chase = None
    if args_cli.save_video:
        import imageio
        vwriter = imageio.get_writer(args_cli.save_video, fps=30, codec="libx264",
                                     quality=8, macro_block_size=None)
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
        eye = torch.stack([p[0] - fx * 2.4, p[1] - fy * 2.4, p[2] + 1.2])
        tgt = torch.stack([p[0] + fx * 4.0, p[1] + fy * 4.0, p[2] - 0.2])
        cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))

    results = []
    for ep in range(args_cli.episodes):
        torch.manual_seed(args_cli.seed + ep)
        if hasattr(est, "reset"):
            est.reset()
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        hidden = None
        max_prog, off_sum, off_n, off_max = 0.0, 0.0, 0, 0.0
        prev_w, jerk_sum, outcome, last_h = None, 0.0, "timeout", 1.0
        goal_idx, gates_passed = 0, 0
        for t in range(args_cli.max_steps):
            # --- goal-conditioned (gate) mode: compute body-frame goal dir BEFORE sensing ---
            if is_gate:
                xy_now = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
                yaw_now = _yaw_of(robot.data.root_quat_w)
                goal_xy = gate_centers[min(goal_idx, K - 1)]
                if np.linalg.norm(xy_now - goal_xy) < PASS_RADIUS:
                    if goal_idx < K - 1:
                        goal_idx += 1; goal_xy = gate_centers[goal_idx]
                    else:
                        gates_passed = K
                gates_passed = max(gates_passed, goal_idx)
                dvec = goal_xy - xy_now
                cy, sy = _math.cos(-yaw_now), _math.sin(-yaw_now)
                bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
                nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
                desired_vel = torch.tensor([[bx / nrm, by / nrm, 0.0]], device=dev, dtype=torch.float32).repeat(N, 1)
            else:
                desired_vel = fwd_goal

            inp = sense(desired_vel)
            with torch.no_grad():
                cmd, hidden = model(inp, hidden, mask=mask)
            yaw_rate = float(cmd[0, 0].item())
            fwd = float(max(0.1, min(1.5, cmd[0, 1].item())))
            steering_term.target_yaw_rate.fill_(yaw_rate)
            steering_term.target_velocity.fill_(fwd)
            if chase is not None and ep < 2:
                _drive_chase(chase)          # reposition the chase cam before this step's render
            with torch.no_grad():
                actions = inner_policy(obs)
            obs, _r, dones, _i = env.step(actions)
            xy = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
            last_h = float(robot.data.root_pos_w[0, 2].item())
            wz = float(robot.data.root_ang_vel_b[0, 2].item())
            prog, off = geom.progress_offset(xy)
            max_prog = max(max_prog, prog); off_sum += off; off_n += 1; off_max = max(off_max, off)
            if prev_w is not None:
                jerk_sum += abs(wz - prev_w)
            prev_w = wz
            if vwriter is not None and ep < 2:
                rgb = demo_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                vwriter.append_data(np.ascontiguousarray(rgb))
            if is_gate and gates_passed >= K:
                outcome = "success"; break            # flew the whole gate course
            if bool(dones[0].item()):
                try:
                    tm = uenv.termination_manager
                    terms = {nm: bool(tm.get_term(nm)[0].item()) for nm in tm.active_terms}
                except Exception:
                    terms = {}
                if is_gate:
                    outcome = ("off_trail" if (terms.get("off_trail") or off > geom.lateral_margin)
                               else "crash" if last_h < 0.2 else "timeout")
                elif max_prog >= args_cli.success_frac:
                    outcome = "success"
                elif terms.get("time_out"):
                    outcome = "timeout"
                elif terms.get("off_trail") or off > geom.lateral_margin:
                    outcome = "off_trail"
                elif any("crash" in nm and v for nm, v in terms.items()) or last_h < 0.2:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                break
        else:
            if is_gate:
                outcome = "success" if gates_passed >= K else "timeout"
            else:
                outcome = "success" if max_prog >= args_cli.success_frac else "timeout"
        prog_report = (gates_passed / K) if is_gate else max_prog
        rec = {"episode": ep, "outcome": outcome, "progress": round(prog_report, 4),
               "gates_passed": (gates_passed if is_gate else None),
               "mean_offset": round(off_sum / max(1, off_n), 4), "max_offset": round(off_max, 4),
               "mean_jerk": round(jerk_sum / max(1, off_n), 5), "steps": off_n}
        results.append(rec)
        _pr = f"gates={gates_passed}/{K}" if is_gate else f"progress={prog_report:.2f}"
        log(f"[ep{ep:02d}] outcome={outcome:9s} {_pr} mean_off={rec['mean_offset']:.2f} steps={off_n}")

    if vwriter is not None:
        vwriter.close()
        log(f"[demo] wrote {args_cli.save_video}")

    n = len(results)
    agg = {"trail": args_cli.trail, "nav_arch": "fused", "mask_off": sorted(masked), "episodes": n,
           "success_rate": round(sum(r["outcome"] == "success" for r in results) / max(1, n), 3),
           "mean_progress": round(sum(r["progress"] for r in results) / max(1, n), 3),
           "mean_offset": round(sum(r["mean_offset"] for r in results) / max(1, n), 3),
           "mean_jerk": round(sum(r["mean_jerk"] for r in results) / max(1, n), 5),
           "outcomes": {k: sum(r["outcome"] == k for r in results) for k in ("success", "off_trail", "crash", "timeout")}}
    log("\n=== FUSED NAV EVAL ===")
    log(json.dumps(agg, indent=2))
    tag = ("_off-" + "-".join(sorted(masked))) if masked else ""
    out = args_cli.out or os.path.join(out_dir, f"forestnav_fused_{args_cli.trail}{tag}.json")
    json.dump({"agg": agg, "episodes": results, "args": vars(args_cli)}, open(out, "w"), indent=2)
    log(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
