"""Fly the warehouse with the RL velocity-controller (Agent B, Track 1c-b / eval E1-RL).

Same nav model + course as eval_fused_warehouse, but the low-level controller is the PPO-trained
velocity controller (Isaac-Track-VelocityCtrl) instead of the classical law or the distilled MLP.
The RL actor (MLP 16->256/128/64->4 ELU, no obs-norm) consumes
[base_lin_vel(3), base_ang_vel(3), projected_gravity(3), base_height(1), steering_cmd(2), last_action(4)]
where steering_cmd = nav's (yaw_rate, forward_speed), and outputs the DirectThrustMoment action.

    <env_isaaclab py> sims/scripts/eval_rl_controller_warehouse.py --headless \
        --weights <v12>/best.pt --rl_checkpoint <run>/model_250.pt --episodes 8 --prop_density 0.3
"""
from __future__ import annotations
import argparse, json, math as _math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--rl_checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--base_speed", type=float, default=1.3)
parser.add_argument("--cruise_speed", type=float, default=1.3, help="fixed forward-speed command (nav speed-head underfits)")
parser.add_argument("--yaw_scale", type=float, default=1.0, help="scale nav's yaw command to the controller (damp yaw-feedback spin)")
parser.add_argument("--moment_scale", type=float, default=0.01, help="DirectThrustMoment moment authority (lower = less over-yaw; DR-trained policy handles 0.008-0.013)")
parser.add_argument("--prop_density", type=float, default=0.3)
parser.add_argument("--obstacle_level", type=int, default=8)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (  # noqa: E402
    WarehouseNavEnvCfg_PLAY_WithSensors_Coll)
from sims.isaaclab_tasks.track_steering_vision.mdp_actions import DirectThrustMomentActionCfg  # noqa: E402
from fused_model import FusedSensorNet  # noqa: E402

TASK_ID = "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0"
GATE_CENTERS_2D = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)
PASS_RADIUS = GW.FixedGateCourseCommandCfg().success_radius


def log(m): print(m, flush=True)


def build_rl_actor(ckpt_path, dev):
    """RL actor (ELU MLP, obs_normalization=False). Architecture INFERRED from the checkpoint's Linear
    weight shapes -> works for [256,128,64] or the bigger [512,512,512,256] without edits."""
    import re as _re
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    asd = ck["actor_state_dict"]  # keys: mlp.{even}.{weight,bias} (+ distribution.log_std_param)
    lin_idx = sorted({int(m.group(1)) for k in asd for m in [_re.match(r"mlp\.(\d+)\.weight", k)] if m})
    layers = []
    for j, i in enumerate(lin_idx):
        out_f, in_f = asd[f"mlp.{i}.weight"].shape
        layers.append(nn.Linear(in_f, out_f))
        if j < len(lin_idx) - 1:
            layers.append(nn.ELU(alpha=1.0))
    mlp = nn.Sequential(*layers).to(dev)
    stripped = {k[len("mlp."):]: v for k, v in asd.items() if k.startswith("mlp.")}
    missing, unexpected = mlp.load_state_dict(stripped, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    mlp.eval()
    _dims = [layers[0].in_features] + [l.out_features for l in layers if isinstance(l, nn.Linear)]
    log(f"[rl] actor arch inferred from ckpt: {_dims}")
    return mlp


def main():
    env_cfg = WarehouseNavEnvCfg_PLAY_WithSensors_Coll()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.obstacle_count.params["min_level"] = args_cli.obstacle_level
    env_cfg.events.reset_obstacles.params["prop_density"] = args_cli.prop_density
    env_cfg.actions.velocity = DirectThrustMomentActionCfg(  # swap classical ctrl -> RL thrust/moment iface
        asset_name="robot", body_name="body", thrust_to_weight=1.9, moment_scale=args_cli.moment_scale)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device; N = uenv.num_envs
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    est = StateEstimator(N, dev, control_dt=control_dt)
    _sd = torch.load(args_cli.weights, map_location=dev, weights_only=True)
    _venc = "cnn" if any(k.startswith("vision_cnn.") for k in _sd) else "vit"
    nav = FusedSensorNet(out_dim=2, vision_encoder=_venc).to(dev).eval(); nav.load_state_dict(_sd, strict=True)
    actor = build_rl_actor(args_cli.rl_checkpoint, dev)
    log(f"[rl] loaded RL controller {args_cli.rl_checkpoint}")
    robot = uenv.scene["robot"]; origin = uenv.scene.env_origins
    K = len(GATE_CENTERS_2D)

    def _yaw_of(q):
        w, x, y, z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
        return float(torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).item())

    def sense(desired_vel):
        grey = S.front_greyscale(uenv)
        tof_norm, _ = S.normalize_range(S.tof_stack(uenv), S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
        dtof = S.down_tof(uenv); dtof_norm, dtof_valid = S.normalize_range(dtof, S.DOWN_TOF_RANGE_MIN, S.DOWN_TOF_RANGE_MAX)
        flow = S.optical_flow(uenv); flow_valid = S.optical_flow_valid(uenv)
        baro = S.barometer(uenv, drift=est.step_baro_drift())
        gyro = robot.data.root_ang_vel_b[:, :3]; accel = -robot.data.projected_gravity_b * 9.81
        filt = est.update(gyro, accel, baro_alt=baro[:, 1], tof_alt=dtof.squeeze(1), flow_vel=flow * 0.0)
        return {"front_grey": grey.float(), "tof_cross": tof_norm, "optical_flow": flow, "down_tof": dtof_norm,
                "baro": baro / 10.0, "quat": filt["quat"], "body_rates": gyro, "desired_vel": desired_vel,
                "flags": torch.cat([flow_valid, dtof_valid, torch.ones(N, 4, device=dev)], dim=1)}

    results = []
    for ep in range(args_cli.episodes):
        os.environ["FOREST_GATE_SEED"] = str(args_cli.seed + ep)
        torch.manual_seed(args_cli.seed + ep)
        if hasattr(est, "reset"):
            est.reset()
        env.reset()
        hidden = None; goal_idx = 0; gates_passed = 0; outcome = "timeout"
        last_action = torch.zeros(N, 4, device=dev)
        start_xy = None; max_disp = 0.0
        for t in range(args_cli.max_steps):
            xy_now = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
            if start_xy is None:
                start_xy = xy_now.copy()
            max_disp = max(max_disp, float(np.linalg.norm(xy_now - start_xy)))
            yaw_now = _yaw_of(robot.data.root_quat_w)
            goal_xy = GATE_CENTERS_2D[min(goal_idx, K - 1)]
            if np.linalg.norm(xy_now - goal_xy) < PASS_RADIUS:
                if goal_idx < K - 1:
                    goal_idx += 1
                else:
                    gates_passed = K
            gates_passed = max(gates_passed, goal_idx)
            dvec = goal_xy - xy_now
            cy, sy = _math.cos(-yaw_now), _math.sin(-yaw_now)
            bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
            nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
            desired_vel = torch.tensor([[bx / nrm * args_cli.base_speed, by / nrm * args_cli.base_speed, 0.0]],
                                       device=dev, dtype=torch.float32).repeat(N, 1)
            inp = sense(desired_vel)
            with torch.no_grad():
                cmd, hidden = nav(inp, hidden, mask=None)
            yaw_rate = float(cmd[0, 0].item()) * args_cli.yaw_scale
            steer_cmd = torch.tensor([[yaw_rate, args_cli.cruise_speed]], device=dev, dtype=torch.float32)
            # assemble RL obs (matches velocity_control_env_cfg ObservationsCfg order)
            base_lin_vel = robot.data.root_lin_vel_b[:, :3]
            base_ang_vel = robot.data.root_ang_vel_b[:, :3]
            proj_grav = robot.data.projected_gravity_b
            base_h = (robot.data.root_pos_w - origin)[:, 2:3]
            obs = torch.cat([base_lin_vel, base_ang_vel, proj_grav, base_h, steer_cmd, last_action], dim=1)
            with torch.no_grad():
                action = actor(obs).clamp(-1.0, 1.0)
            last_action = action.detach()
            if os.environ.get("DBG_RL") and ep == 0 and t % 100 == 0:
                log(f"  [dbg t={t}] cmd_fwd={args_cli.cruise_speed:.2f} vx_b={float(base_lin_vel[0,0]):+.2f} "
                    f"vy_b={float(base_lin_vel[0,1]):+.2f} cmd_yaw={yaw_rate:+.2f} ach_yaw={float(base_ang_vel[0,2]):+.2f} "
                    f"h={float(base_h[0,0]):.2f} xy=({float((robot.data.root_pos_w-origin)[0,0]):.1f},{float((robot.data.root_pos_w-origin)[0,1]):.1f})")
            obs_o, _r, dones, _i = env.step(action)
            last_h = float(robot.data.root_pos_w[0, 2].item())
            if gates_passed >= K:
                outcome = "success"; break
            if bool(dones[0].item()):
                outcome = "crash" if last_h < 0.2 else "timeout"
                try:
                    tm = uenv.termination_manager
                    if any("collision" in nm and tm.get_term(nm)[0].item() for nm in tm.active_terms):
                        outcome = "crash"
                except Exception:
                    pass
                break
        rec = {"episode": ep, "outcome": outcome, "gates_passed": gates_passed, "steps": t + 1,
               "max_displacement_m": round(max_disp, 2)}
        results.append(rec)
        log(f"[ep{ep:02d}] outcome={outcome} gates={gates_passed}/{K} steps={t+1} max_disp={max_disp:.1f}m")

    n = len(results)
    agg = {"controller": "rl_velocity", "episodes": n,
           "success_rate": round(sum(r["outcome"] == "success" for r in results) / max(1, n), 3),
           "mean_progress": round(sum(r["gates_passed"] / K for r in results) / max(1, n), 3),
           "outcomes": {k: sum(r["outcome"] == k for r in results) for k in ("success", "crash", "timeout")}}
    log("\n=== RL CONTROLLER WAREHOUSE EVAL (E1-RL) ==="); log(json.dumps(agg, indent=2))
    out = args_cli.out or "/scratch/agustin/projects/DIMA/XPU-RT/runs/e1_rl/e1_rl.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"agg": agg, "episodes": results, "args": vars(args_cli)}, open(out, "w"), indent=2)
    log(f"[out] wrote {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
