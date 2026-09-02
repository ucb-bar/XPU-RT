"""Collect FusedSensorNet BC data by flying the privileged expert (task #56).

Rolls out a forest ``*_WithSensors`` env driven by the analytic expert
(:class:`forest_trail.expert.ForestExpert`, which sees GT geometry), reads the
real onboard sensors each step, assembles the exact FusedSensorNet input dict
(range-normalized + Madgwick-filtered + validity flags, per
``test_fused_perception``), and dumps per-episode SEQUENCES with the expert's
``(yaw_rate, forward_speed)`` as the BC label. The drone is driven by the
expert's command through the frozen steering inner-loop (model_6998), so the
collected trajectories are dynamically feasible and the sensors see realistic
motion.

    <env_isaaclab py> sims/training/collect_fused_data.py --headless \
        --trail straight --with_humans --episodes 30 --max_steps 1600 \
        --out <path>/fused_straight.pt

Output: a torch .pt holding a list[episode]; each episode is a dict of stacked
CPU tensors (T, ...) for every input key + ``label`` (T,2) + ``label_mask`` — a
drop-in for the fused BC loader.
"""

from __future__ import annotations

import argparse
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--trail", choices=["straight", "curved", "slalom", "gate"], default="straight")
parser.add_argument("--gate_seed", type=int, default=0,
                    help="Gate-course DR seed (0=canonical). Randomizes gate layout per collection run (#46).")
parser.add_argument("--with_humans", action="store_true")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=1600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base_speed", type=float, default=1.0)
parser.add_argument("--noise_std", type=float, default=0.4,
                    help="DART-style yaw-rate noise added to the DRIVING command (label stays the "
                         "clean expert cmd) so the drone visits off-centre states and the expert "
                         "demonstrates recovery — fixes BC covariate shift. 0 = pure expert.")
parser.add_argument("--inner_ckpt", type=str,
                    default="/scratch2/dima/misc_sw/FreshScheduler/logs/rsl_rl/"
                            "crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
# set the gate-course DR seed BEFORE the forest modules import (gates.py reads it at import →
# the scene bakes that course, and the expert/goal read the same seeded gate centres).
os.environ["FOREST_GATE_SEED"] = str(args_cli.gate_seed)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.forest_trail.config.crazyflie  # noqa: E402,F401 (register)
import sims.isaaclab_tasks.track_steering_vision.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.forest_trail.expert import ForestExpert  # noqa: E402
from sims.isaaclab_tasks.forest_trail import forest_scene as FS  # noqa: E402
from sims.isaaclab_tasks.forest_trail import tree_layout as TL  # noqa: E402
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (  # noqa: E402
    ForestTrailEnvCfg_PLAY_WithSensors,
    ForestTrailEnvCfg_PLAY_WithHumans_WithSensors,
    ForestTrailEnvCfg_Curved_PLAY_WithSensors,
    ForestTrailEnvCfg_Curved_PLAY_WithHumans_WithSensors,
    ForestTrailEnvCfg_Slalom_PLAY_WithSensors,
    ForestTrailEnvCfg_Gates_PLAY_WithSensors,
)
from sims.isaaclab_tasks.forest_trail.gates import FOREST_GATE_CENTERS_2D, PASS_RADIUS  # noqa: E402
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    SteeringTrackingPPORunnerCfg,
)


def log(m):
    print(m, flush=True)


def quat_to_yaw(q):  # q: (N,4) wxyz -> (N,)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_expert(trail, with_humans, base_speed):
    """GT-geometry expert. Obstacles in env-local frame from the layout constants."""
    if trail in ("straight", "slalom", "gate"):
        trees = list(FS.DEFAULT_STRAIGHT_POSITIONS)                       # [(x,y)]
        humans = [(x, y) for x, y, _ in TL.generate_human_positions(FS.DEFAULT_HUMAN_LAYOUT)] if with_humans else []
        # slalom adds IN-CORRIDOR obstacles the expert must actively weave around.
        slalom = list(FS.DEFAULT_SLALOM_OBSTACLES) if trail == "slalom" else []
        # gate mode pursues per-step goal_xy (gate centres); obstacles are the off-corridor scenery.
        obstacles = trees + humans + slalom
        return ForestExpert("straight", trail_length=FS.DEFAULT_STRAIGHT_LAYOUT.trail_length,
                            obstacles_xy=obstacles, base_speed=base_speed)
    wps = FS.DEFAULT_CURVED_WAYPOINTS_2D
    trees = [(x, y) for x, y in TL.generate_curved_trail_trees(FS.DEFAULT_CURVED_LAYOUT, FS.DEFAULT_CURVED_WAYPOINTS)]
    humans = ([(x, y) for x, y, _ in TL.generate_curved_human_positions(
        FS.DEFAULT_CURVED_HUMAN_LAYOUT, FS.DEFAULT_CURVED_WAYPOINTS)] if with_humans else [])
    return ForestExpert("curved", waypoints_2d=wps, obstacles_xy=trees + humans, base_speed=base_speed)


def load_inner_policy(env, device, ckpt):
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
        ckpt = os.path.join(os.path.dirname(args_cli.out), "_model_6998_logstd.pt")
        torch.save(loaded, ckpt)
    runner.load(ckpt)
    return runner.get_inference_policy(device=device)


def main():
    torch.manual_seed(args_cli.seed)
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
    # extend the episode so the expert flies the FULL trail (default ~10 s time-out
    # caps episodes at ~10 m → the model would only ever see the first third and drift
    # off once past its training coverage). Bounded by --max_steps in the loop.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    log(f"[env] gym.make {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device
    N = uenv.num_envs
    control_dt = float(uenv.step_dt) if hasattr(uenv, "step_dt") else float(uenv.cfg.sim.dt * uenv.cfg.decimation)

    inner_policy = load_inner_policy(env, dev, args_cli.inner_ckpt)
    expert = build_expert(args_cli.trail, args_cli.with_humans, args_cli.base_speed)
    est = StateEstimator(N, dev, control_dt=control_dt)
    steering_term = uenv.command_manager.get_term("steering_command")
    robot = uenv.scene["robot"]
    origin = uenv.scene.env_origins  # (N,3)
    goal = torch.tensor([args_cli.base_speed, 0.0, 0.0], device=dev).repeat(N, 1)  # naive forward goal hint
    is_gate = args_cli.trail == "gate"
    gate_centers = np.asarray(FOREST_GATE_CENTERS_2D, dtype=np.float64) if is_gate else None
    log(f"[collect] {task_id} control_dt={control_dt*1000:.1f}ms expert={args_cli.trail} humans={args_cli.with_humans}"
        + (f" gates={len(gate_centers)}" if is_gate else ""))

    def assemble(dtof, flow, flow_valid, baro, grey, tof_norm, gyro, quat, dtof_norm, dtof_valid, desired_vel):
        return {
            "front_grey": grey, "tof_cross": tof_norm, "optical_flow": flow, "down_tof": dtof_norm,
            "baro": baro / 10.0, "quat": quat, "body_rates": gyro, "desired_vel": desired_vel,
            "flags": torch.cat([flow_valid, dtof_valid, torch.ones(N, 4, device=dev)], dim=1),
        }

    episodes = []
    total_frames = 0
    for ep in range(args_cli.episodes):
        torch.manual_seed(args_cli.seed + 1000 + ep)
        est.reset() if hasattr(est, "reset") else None
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        buf = {k: [] for k in ("front_grey", "tof_cross", "optical_flow", "down_tof",
                               "baro", "quat", "body_rates", "desired_vel", "flags")}
        labels = []
        drive_noise = 0.0  # OU-ish random walk on the DRIVEN yaw so the drone wanders off-centre
        goal_idx = 0        # gate mode: index of the current target gate
        for t in range(args_cli.max_steps):
            # --- sensors ---
            grey = S.front_greyscale(uenv)
            tof_norm, _ = S.normalize_range(S.tof_stack(uenv), S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
            dtof = S.down_tof(uenv)
            dtof_norm, dtof_valid = S.normalize_range(dtof, S.DOWN_TOF_RANGE_MIN, S.DOWN_TOF_RANGE_MAX)
            flow = S.optical_flow(uenv)
            flow_valid = S.optical_flow_valid(uenv)
            baro = S.barometer(uenv, drift=est.step_baro_drift())
            gyro = robot.data.root_ang_vel_b[:, :3]
            accel = -robot.data.projected_gravity_b * 9.81
            filt = est.update(gyro, accel, baro_alt=baro[:, 1], tof_alt=dtof.squeeze(1), flow_vel=flow * 0.0)

            # --- drone pose (GT, env-local) ---
            local = (robot.data.root_pos_w - origin)
            xy = local[:, :2].cpu().numpy()
            yaw = quat_to_yaw(robot.data.root_quat_w).cpu().numpy()

            # --- goal-conditioned (gate) mode: track current gate, feed body-frame goal dir ---
            if is_gate:
                goal_xy = gate_centers[min(goal_idx, len(gate_centers) - 1)]        # (2,)
                if np.linalg.norm(xy[0] - goal_xy) < PASS_RADIUS and goal_idx < len(gate_centers) - 1:
                    goal_idx += 1
                    goal_xy = gate_centers[goal_idx]
                dvec = goal_xy - xy[0]
                cy, sy = np.cos(-yaw[0]), np.sin(-yaw[0])                            # world->body rot
                bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
                nrm = max(float(np.hypot(bx, by)), 1e-6)
                desired_vel = torch.tensor(
                    [[bx / nrm * args_cli.base_speed, by / nrm * args_cli.base_speed, 0.0]],
                    device=dev, dtype=torch.float32).repeat(N, 1)
                yr, sp = expert.command(xy, yaw, local[:, 2].cpu().numpy(), goal_xy=goal_xy[None, :])
            else:
                desired_vel = goal
                yr, sp = expert.command(xy, yaw, local[:, 2].cpu().numpy())

            inp = assemble(dtof, flow, flow_valid, baro, grey, tof_norm, gyro, filt["quat"],
                           dtof_norm, dtof_valid, desired_vel)

            # store greyscale at the model's internal size (60x90) as float16 to keep
            # the dataset ~30x smaller (the ViT interpolates to 60x90 anyway → lossless).
            grey_small = Fn.interpolate(inp["front_grey"], size=(60, 90), mode="bilinear", align_corners=False)
            for k in buf:
                if k == "front_grey":
                    buf[k].append(grey_small[0].detach().to(torch.float16).cpu())
                else:
                    buf[k].append(inp[k][0].detach().cpu())
            labels.append(torch.tensor([yr[0], sp[0]], dtype=torch.float32))

            # --- drive with expert cmd + DART noise; LABEL stays the clean expert cmd ---
            # (label was recorded above from the CURRENT off-centre state, so it is the
            # correct recovery action for whatever state the noise drove us into.)
            drive_noise = 0.85 * drive_noise + np.random.randn() * args_cli.noise_std
            steering_term.target_yaw_rate.fill_(float(yr[0] + drive_noise))
            steering_term.target_velocity.fill_(float(sp[0]))
            with torch.no_grad():
                actions = inner_policy(obs)
            obs, _r, dones, _i = env.step(actions)
            if bool(dones[0].item()):
                break
        T = len(labels)
        if T < 10:
            log(f"[ep{ep:02d}] too short ({T}) — skipped"); continue
        epdict = {k: torch.stack(v) for k, v in buf.items()}
        epdict["label"] = torch.stack(labels)              # (T,2) = (yaw_rate, forward_speed)
        episodes.append(epdict)
        total_frames += T
        log(f"[ep{ep:02d}] T={T} frames (total={total_frames})")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    torch.save({"episodes": episodes, "meta": {"trail": args_cli.trail, "with_humans": args_cli.with_humans,
                "control_dt": control_dt, "label_keys": ["yaw_rate", "forward_speed"],
                "n_episodes": len(episodes), "n_frames": total_frames}}, args_cli.out)
    log(f"[done] wrote {len(episodes)} episodes / {total_frames} frames -> {args_cli.out}")


if __name__ == "__main__":
    main()
    os._exit(0)  # Isaac close() hangs; data already saved


