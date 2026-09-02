"""Collect FusedSensorNet BC data in the PHOTOREAL warehouse gate course (task #64).

The warehouse port of ``collect_fused_data.py``. Rolls out the single-env
``Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0`` scene (real
full_warehouse aisle: collidable rack rows + gate frames + curriculum obstacle
field + patrolling people), driven by the privileged analytic expert
(:class:`forest_trail.expert.ForestExpert` in goal-pursuit mode, fed the warehouse
gate centres + LIVE obstacle positions each step), reads the exact same onboard
sensor suite as the forest pipeline (``forest_trail/sensors.py`` — env-agnostic),
and dumps per-episode SEQUENCES + the expert's ``(yaw_rate, forward_speed)`` label.

Crucially the FLIGHT SEAM is the warehouse's own cascade-stable geometric
``VelocityCommandAction`` (NOT the forest steering checkpoint): the expert command
maps to the 4-D velocity action directly ―

    a0 = forward_speed - 1           (vx = (a0+1) * max_speed/2 = forward_speed, max_speed=2)
    a2 = yaw_rate / max_yawrate      (yawrate channel)
    a1 = a3 = 0                      (level flight, unused)

so no frozen policy / checkpoint is needed and the model's (yaw_rate, forward_speed)
output contract is unchanged.

    <env_isaaclab py> sims/training/collect_fused_warehouse.py --headless \
        --episodes 24 --max_steps 1500 --noise_std 0.15 \
        --out <path>/fused_warehouse_gate.pt
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
parser.add_argument("--episodes", type=int, default=24)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base_speed", type=float, default=1.2,
                    help="expert cruise speed (m/s); warehouse velocity controller max is 2.0.")
parser.add_argument("--noise_std", type=float, default=0.15,
                    help="DART-style yaw-rate noise on the DRIVEN command (label stays clean expert "
                         "cmd) → visits off-centre states, expert demonstrates recovery. 0=pure.")
parser.add_argument("--no_avoid", action="store_true",
                    help="Expert ignores obstacles → PURE goal-pursuit, so yaw_rate is a clean "
                         "function of desired_vel (learnable to high fidelity like the forest, vs "
                         "the stateful ToF-dependent avoidance mapping that caps at ~0.86).")
parser.add_argument("--obstacle_level", type=int, default=4,
                    help="active prop count in the aisle (people/racks/gates are always present). "
                         "Lower → cleaner expert teacher trajectories; higher → harder eval.")
parser.add_argument("--avoid_radius", type=float, default=1.4, help="expert obstacle-avoidance lookahead (m).")
parser.add_argument("--k_avoid", type=float, default=1.2, help="expert repulsive-yaw strength.")
parser.add_argument("--min_speed_frac", type=float, default=0.4, help="expert speed floor when blocked ahead.")
parser.add_argument("--turn_slow", action="store_true", help="expert slows into sharp heading changes (dense clutter).")
parser.add_argument("--planned_expert", action="store_true",
                    help="Gate-PRIMARY planned expert: static props cleared by a minimal planned "
                         "sub-goal offset (label stays gate-directed); only people+walls are reactive. "
                         "Fixes the 'avoidance swamps goal-seeking / steers into nothingness' failure.")
parser.add_argument("--prop_density", type=float, default=None,
                    help="Override the aisle tall-thin stacked-prop density [0..1]. Default None keeps "
                         "the cfg value (0.0 = gate-follow only). >0 enables the crowded ToF-avoidance "
                         "course the expert must weave through.")
parser.add_argument("--tof_noise", type=float, default=0.0,
                    help="If >0, apply VL53L5CX-style noise to the ToF stack before normalization "
                         "(range_noise_pct=this, far-zone dropout_prob=0.05). Closes the clean-sim "
                         "gap and stops the net from distrusting/ignoring ToF.")
parser.add_argument("--cam_dropout_frac", type=float, default=0.0,
                    help="Fraction of episodes [0..1] on which the STORED greyscale camera is zeroed, "
                         "forcing the net to learn ToF->avoidance (desired_vel still carries the goal). "
                         "Ablation showed BC ignores ToF because camera+goal predict the label.")
parser.add_argument("--collect_depth", action="store_true",
                    help="Also store aligned front-cam ground-truth inverse-depth (16x24) per frame "
                         "as 'front_depth' for the TRAINING-ONLY mono-depth auxiliary task (#76). "
                         "No effect on the model inputs; consumed only by train_fused --depth_aux_weight.")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--collidable", action="store_true",
                    help="Collect in the COLLIDABLE-gate env (WithSensors_Coll, the honest eval/demo scene) "
                         "instead of the pass-through-gate training scene. REQUIRED for DAgger that transfers "
                         "to record_sensor_demo: with pass-through gates the nav learns to CLIP gates (fine "
                         "there, a CRASH in the collidable demo). Collidable gates label clean gate-passing.")
parser.add_argument("--drive_rl", type=str, default=None,
                    help="NAV CO-TRAINING: actuate the drone with the trained RL velocity controller "
                         "(this checkpoint) executing the expert's command, instead of the classical "
                         "VelocityCommandAction. Label stays the expert (yaw_rate, forward_speed) so nav "
                         "learns commands suited to the RL controller's closed-loop response.")
parser.add_argument("--moment_scale", type=float, default=0.006, help="RL controller moment authority (drive_rl).")
parser.add_argument("--rl_cruise", type=float, default=1.2, help="fixed forward command fed to the RL controller (drive_rl).")
parser.add_argument("--drive_nav", type=str, default=None,
                    help="DAgger: STEER the drone with this trained nav checkpoint (FusedSensorNet, encoder "
                         "auto-detected) instead of the expert, while the LABEL stays the clean expert "
                         "(yaw_rate, forward_speed). This visits the NAV POLICY's own drift states and labels "
                         "the expert's recovery there — the covariate-shift fix that plain DART cannot give.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: E402,F401 (register)
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.forest_trail.expert import ForestExpert  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (  # noqa: E402
    WarehouseNavEnvCfg_PLAY_WithSensors,
    WarehouseNavEnvCfg_PLAY_WithSensors_Coll,
)
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg  # noqa: E402
from sims.isaaclab_tasks.track_steering_vision.mdp_actions import DirectThrustMomentActionCfg  # noqa: E402


def _build_rl_actor(ckpt_path, dev):
    """RL velocity-controller actor MLP [16->256/128/64->4] ELU (no obs-norm)."""
    from torch import nn as _nn
    dims = [16, 256, 128, 64, 4]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(_nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(_nn.ELU(alpha=1.0))
    mlp = _nn.Sequential(*layers).to(dev)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    asd = ck["actor_state_dict"]
    stripped = {k[len("mlp."):]: v for k, v in asd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(stripped, strict=True)
    return mlp.eval()

# warehouse velocity-controller limits (for the (yr,sp) -> action mapping)
_VCFG = VelocityCommandActionCfg()
MAX_SPEED = _VCFG.max_speed          # 2.0 -> vx = (a0+1)*max_speed/2 = a0+1
MAX_YAWRATE = _VCFG.max_yawrate      # 1.047

TASK_ID = ("Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0"
           if args_cli.collidable else
           "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0")
GATE_CENTERS_2D = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)  # (K,2) env-local
PASS_RADIUS = GW.FixedGateCourseCommandCfg().success_radius                    # 0.9 m
AISLE_CX = -7.98        # aisle centreline x; clear width 3.96 m → rack faces ~x=-9.96/-6.00
AISLE_HALF = 1.9        # virtual-wall offset (just inside the rack face) for expert corridor-keeping
N_PEOPLE = 4            # obstacle slots [0, N_PEOPLE) are patrolling people (dynamic); props follow


def log(m):
    print(m, flush=True)


def quat_to_yaw(q):  # q: (N,4) wxyz -> (N,)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


TARGET_H = 2.0        # altitude-hold setpoint (m); matches FUSED_GATES centre z, clears aisle clutter
_K_ALT = 1.2          # altitude P-gain (vz per metre of error)
_VZ_MAX = 0.8         # clamp on the commanded climb velocity (m/s)
_MAX_INCL = _VCFG.max_inclination


def cmd_to_action(yr_drive, sp, h, dev, N):
    """Map (yaw_rate, forward_speed) + an altitude-hold loop -> warehouse 4-D velocity action.

    Steering/speed come from the expert; altitude is a fixed autopilot driving the drone to
    TARGET_H (inclination channel a1: vz = speed·sin(max_incl·a1)·(max_speed/2))."""
    import math
    a0 = float(sp) / (MAX_SPEED / 2.0) - 1.0          # vx = (a0+1)*max_speed/2 = sp
    a2 = float(yr_drive) / MAX_YAWRATE                # yawrate channel
    speed = max(0.05, a0 + 1.0)
    vz_des = max(-_VZ_MAX, min(_VZ_MAX, _K_ALT * (TARGET_H - float(h))))
    s = max(-1.0, min(1.0, vz_des / speed))
    a1 = math.asin(s) / _MAX_INCL
    a = torch.tensor([[a0, a1, a2, 0.0]], device=dev, dtype=torch.float32).clamp(-1.0, 1.0)
    return a.repeat(N, 1)


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = (WarehouseNavEnvCfg_PLAY_WithSensors_Coll() if args_cli.collidable
               else WarehouseNavEnvCfg_PLAY_WithSensors())
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.obstacle_count.params["min_level"] = args_cli.obstacle_level
    # Crowded course: turn the tall-thin stacked-prop field back ON (the WithSensors cfg zeroes it
    # for pure gate-following). >0 → the ToF-avoidance collection the expert must weave through.
    if args_cli.prop_density is not None:
        env_cfg.events.reset_obstacles.params["prop_density"] = args_cli.prop_density
    if args_cli.drive_rl:  # NAV CO-TRAINING: actuate via the RL velocity controller
        env_cfg.actions.velocity = DirectThrustMomentActionCfg(
            asset_name="robot", body_name="body", thrust_to_weight=1.9, moment_scale=args_cli.moment_scale)
        log(f"[drive_rl] actuating with RL controller {args_cli.drive_rl} (moment_scale={args_cli.moment_scale})")
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    log(f"[env] gym.make {TASK_ID}")
    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device
    N = uenv.num_envs
    control_dt = float(uenv.step_dt) if hasattr(uenv, "step_dt") else float(uenv.cfg.sim.dt * uenv.cfg.decimation)

    # goal-conditioned expert: pursue gate centres, avoid live obstacle positions. Clamp the
    # expert's yaw rate to the controller's executable max so labels are achievable. Tuned for a
    # brisk cruise in the tight aisle: LARGER lookahead + avoid_radius so it plans around props /
    # racks / people EARLY (short reaction at speed = clipping); slow a touch sooner into gates.
    # Smooth, imitable, BRISK expert. With the gentle FUSED_GATES weave there are no sharp
    # reversals, so heavy turn-slowdown is NOT needed (it made the expert creep at ~0.65 m/s — too
    # slow, and slow flight lets lateral error accumulate into a rack). Keep only a light turn-slow
    # (won't trigger on the gentle weave) + a tight goal-slow (brake only right at each gate) + the
    # yaw-rate EMA (low-jerk, imitable target). Result: cruise ~base_speed with brief gate braking.
    expert = ForestExpert("straight", trail_length=30.0, base_speed=args_cli.base_speed,
                          max_yaw_rate=MAX_YAWRATE, goal_slow_radius=0.6, k_head=1.8,
                          avoid_radius=args_cli.avoid_radius, k_avoid=args_cli.k_avoid,
                          min_speed_frac=args_cli.min_speed_frac,
                          turn_slow=args_cli.turn_slow, yaw_ema=(0.0 if args_cli.no_avoid else 0.6))
    est = StateEstimator(N, dev, control_dt=control_dt)
    robot = uenv.scene["robot"]
    obstacles = uenv.scene["obstacles"]
    origin = uenv.scene.env_origins  # (N,3)
    rl_actor = _build_rl_actor(args_cli.drive_rl, dev) if args_cli.drive_rl else None
    rl_last_action = torch.zeros(N, 4, device=dev)
    # DAgger: steer with a trained nav policy (label stays the expert's).
    nav_model = None
    if args_cli.drive_nav:
        from fused_model import FusedSensorNet as _FSN  # vitfly/models already on sys.path (line 38)
        _nsd = torch.load(args_cli.drive_nav, map_location=dev, weights_only=True)
        _nenc = "cnn" if any(k.startswith("vision_cnn.") for k in _nsd) else "vit"
        _nlh = int(_nsd["lstm.weight_hh_l0"].shape[1]) if "lstm.weight_hh_l0" in _nsd else 128
        _nll = sum(1 for k in _nsd if k.startswith("lstm.weight_ih_l") and "_reverse" not in k) or 3
        nav_model = _FSN(out_dim=2, vision_encoder=_nenc, lstm_hidden=_nlh, lstm_layers=_nll).to(dev).eval()
        nav_model.load_state_dict(_nsd, strict=True)
        log(f"[dagger] STEERING with nav policy ({_nenc}); label stays expert: {args_cli.drive_nav}")
    nav_hidden = None
    log(f"[collect] {TASK_ID} control_dt={control_dt*1000:.1f}ms gates={len(GATE_CENTERS_2D)} "
        f"pass_r={PASS_RADIUS} base_speed={args_cli.base_speed}")

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
        expert.reset_smoothing()
        nav_hidden = None                       # DAgger: fresh LSTM state per episode
        reset_out = env.reset()
        _obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        buf = {k: [] for k in ("front_grey", "tof_cross", "optical_flow", "down_tof",
                               "baro", "quat", "body_rates", "desired_vel", "flags")}
        depth_buf = []          # mono-depth aux GT (only when --collect_depth)
        labels = []
        drive_noise = 0.0
        goal_idx = 0
        # camera-dropout episodes: force ToF->avoidance learning (deterministic, interleaved, ~frac).
        cam_drop_ep = (args_cli.cam_dropout_frac > 0.0 and
                       ((ep * 7 + 3) % 10) < round(args_cli.cam_dropout_frac * 10))
        for t in range(args_cli.max_steps):
            # --- sensors (identical contract to the forest pipeline) ---
            grey = S.front_greyscale(uenv)
            if args_cli.collect_depth:
                depth_buf.append(S.front_depth(uenv, out_hw=(16, 24))[0].detach().to(torch.float16).cpu())
            tof_raw = S.tof_stack(uenv)
            if args_cli.tof_noise > 0.0:
                tof_raw = S.add_tof_noise(tof_raw, range_noise_pct=args_cli.tof_noise, dropout_prob=0.05)
            tof_norm, _ = S.normalize_range(tof_raw, S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
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

            # --- LIVE obstacle positions (people + ACTIVE props) → expert avoidance ---
            # CRITICAL: inactive/unused pool slots are dumped to z=DUMP_Z (-1000) but keep their
            # xy, so a naive 2-D projection feeds the expert ~100 PHANTOM obstacles → it swerves
            # into a rack avoiding ghosts. Filter to objects actually in the scene (z above floor).
            # live obstacle split: people occupy object slots [0, N_PEOPLE); props follow.
            obj_full = (obstacles.data.object_pos_w - origin.unsqueeze(1))[0]     # (num_obj,3)
            live = obj_full[:, 2] > -5.0
            people = obj_full[:N_PEOPLE][live[:N_PEOPLE], :2].cpu().numpy()
            props = obj_full[N_PEOPLE:][live[N_PEOPLE:], :2].cpu().numpy()
            wy = float(xy[0, 1])
            # aisle RACK FACES are the static warehouse shell (invisible to the obstacle collection)
            # → inject virtual wall points so the expert never gets shoved into a rack.
            walls = np.array([[AISLE_CX - AISLE_HALF, wy + dy] for dy in (-1.5, 0.0, 1.5)] +
                             [[AISLE_CX + AISLE_HALF, wy + dy] for dy in (-1.5, 0.0, 1.5)],
                             dtype=np.float64)

            # --- goal-conditioned (gate) tracking ---
            goal_xy = GATE_CENTERS_2D[min(goal_idx, len(GATE_CENTERS_2D) - 1)]
            if np.linalg.norm(xy[0] - goal_xy) < PASS_RADIUS and goal_idx < len(GATE_CENTERS_2D) - 1:
                goal_idx += 1
                goal_xy = GATE_CENTERS_2D[goal_idx]

            if args_cli.no_avoid:
                expert.obstacles = None
                sub_goal = goal_xy
            elif args_cli.planned_expert:
                # GATE-PRIMARY planned expert (fixes "steers into nothingness"): the gate stays the
                # attractor; STATIC props are cleared by a MINIMAL planned lateral offset on the
                # expert's sub-goal (not on desired_vel), so the yaw LABEL is gate-directed with a
                # brief weave. Only PEOPLE (dynamic) + walls go to the reactive avoider.
                expert.obstacles = np.concatenate([people, walls], axis=0) if len(people) else walls
                sub_goal = goal_xy.copy()
                if len(props):
                    ahead = props[(props[:, 1] > xy[0, 1] - 0.3) & (props[:, 1] < goal_xy[1] + 0.5) &
                                  (np.abs(props[:, 0] - xy[0, 0]) < 1.1)]
                    if len(ahead):
                        j = int(np.argmin(np.hypot(ahead[:, 0] - xy[0, 0], ahead[:, 1] - xy[0, 1])))
                        px, py = ahead[j]
                        side = 1.0 if px <= AISLE_CX else -1.0          # detour to the roomier side
                        sub_x = float(np.clip(px + side * 1.2, AISLE_CX - 1.3, AISLE_CX + 1.3))
                        sub_goal = np.array([sub_x, min(py, goal_xy[1])])  # beside prop, before gate
            else:
                expert.obstacles = np.concatenate([props, walls], axis=0) if len(props) else walls
                sub_goal = goal_xy

            # desired_vel = RAW gate direction (unchanged; inference has no planner) so the model
            # learns to follow the gate goal and weave from its own ToF/vision.
            dvec = goal_xy - xy[0]
            cy, sy = np.cos(-yaw[0]), np.sin(-yaw[0])                # world->body rot
            bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
            nrm = max(float(np.hypot(bx, by)), 1e-6)
            desired_vel = torch.tensor(
                [[bx / nrm * args_cli.base_speed, by / nrm * args_cli.base_speed, 0.0]],
                device=dev, dtype=torch.float32).repeat(N, 1)
            yr, sp = expert.command(xy, yaw, local[:, 2].cpu().numpy(), goal_xy=sub_goal[None, :])

            inp = assemble(dtof, flow, flow_valid, baro, grey, tof_norm, gyro, filt["quat"],
                           dtof_norm, dtof_valid, desired_vel)
            grey_small = Fn.interpolate(inp["front_grey"], size=(60, 90), mode="bilinear", align_corners=False)
            if cam_drop_ep:
                grey_small = torch.zeros_like(grey_small)   # blind the camera → net must use ToF
            for k in buf:
                if k == "front_grey":
                    buf[k].append(grey_small[0].detach().to(torch.float16).cpu())
                else:
                    buf[k].append(inp[k][0].detach().cpu())
            labels.append(torch.tensor([yr[0], sp[0]], dtype=torch.float32))

            # --- drive with expert cmd + DART noise (label stays clean) ---
            drive_noise = 0.85 * drive_noise + np.random.randn() * args_cli.noise_std
            if nav_model is not None:
                # DAgger: STEER with the nav policy (its own drift states), label stays the expert.
                model_inp = {**inp, "front_grey": grey_small}
                with torch.no_grad():
                    nav_cmd, nav_hidden = nav_model(model_inp, nav_hidden)
                yr_d = float(nav_cmd[0, 0]) + drive_noise
                sp_d = float(np.clip(nav_cmd[0, 1].item(), 0.3, args_cli.base_speed + 0.4))
                action = cmd_to_action(yr_d, sp_d, float(local[0, 2]), dev, N)
            elif rl_actor is not None:
                # actuate via the RL velocity controller executing the expert's yaw + a cruise forward
                base_lin_vel = robot.data.root_lin_vel_b[:, :3]
                base_ang_vel = robot.data.root_ang_vel_b[:, :3]
                proj_grav = robot.data.projected_gravity_b
                base_h = (robot.data.root_pos_w - origin)[:, 2:3]
                steer = torch.tensor([[yr[0] + drive_noise, args_cli.rl_cruise]], device=dev, dtype=torch.float32).repeat(N, 1)
                rl_obs = torch.cat([base_lin_vel, base_ang_vel, proj_grav, base_h, steer, rl_last_action], dim=1)
                with torch.no_grad():
                    action = rl_actor(rl_obs).clamp(-1.0, 1.0)
                rl_last_action = action.detach()
                if os.environ.get("DBG_RL") and ep == 0 and t % 15 == 0:
                    log(f"  [dbg t={t}] xy=({xy[0,0]:.2f},{xy[0,1]:.2f}) h={float(base_h[0,0]):.2f} "
                        f"expert_yr={yr[0]:+.2f} ach_yaw={float(base_ang_vel[0,2]):+.2f} "
                        f"vx_b={float(base_lin_vel[0,0]):+.2f} vy_b={float(base_lin_vel[0,1]):+.2f} "
                        f"act=[{','.join(f'{float(a):+.2f}' for a in action[0])}]")
            else:
                action = cmd_to_action(yr[0] + drive_noise, sp[0], float(local[0, 2]), dev, N)
            obs, _r, dones, _i = env.step(action)
            if bool(dones[0].item()):
                break
        T = len(labels)
        if T < 10:
            log(f"[ep{ep:02d}] too short ({T}) — skipped"); continue
        epdict = {k: torch.stack(v) for k, v in buf.items()}
        epdict["label"] = torch.stack(labels)              # (T,2) = (yaw_rate, forward_speed)
        if args_cli.collect_depth and len(depth_buf) == T:
            epdict["front_depth"] = torch.stack(depth_buf)  # (T,1,16,24) mono-depth aux GT
        # episode quality tags (for post-hoc clean-demo filtering): gates reached + completion
        epdict["gates_reached"] = int(goal_idx + 1)
        epdict["completed"] = bool(goal_idx + 1 >= len(GATE_CENTERS_2D))
        episodes.append(epdict)
        total_frames += T
        log(f"[ep{ep:02d}] T={T} frames gates={goal_idx + 1}/{len(GATE_CENTERS_2D)} (total={total_frames})")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    torch.save({"episodes": episodes, "meta": {"env": "warehouse_gate", "control_dt": control_dt,
                "label_keys": ["yaw_rate", "forward_speed"], "n_episodes": len(episodes),
                "n_frames": total_frames}}, args_cli.out)
    log(f"[done] wrote {len(episodes)} episodes / {total_frames} frames -> {args_cli.out}")


if __name__ == "__main__":
    main()
    os._exit(0)  # Isaac close() hangs; data already saved
