"""Synthetic YOLO detection-dataset generator from the warehouse drone scene (Agent B, Track 1d).

Flies the v12 nav model (so viewpoints are the REAL onboard distribution) across a seed/density
sweep, adds `instance_segmentation_fast` to the front camera, derives tight 2D boxes from the
per-instance mask (bbox annotators are unsupported by IsaacLab's Camera), and writes YOLO-format
labels. Classes collapsed to {0 gate, 1 person, 2 obstacle}; raw class kept in metadata.

Split by SCENE/SEED (not adjacent frames) to avoid video-frame leakage.

    <env_isaaclab py> sims/scripts/gen_yolo_dataset.py --headless \
        --weights <v12>/best.pt --episodes 24 --out_root out/yolo_warehouse
"""
from __future__ import annotations
import argparse, json, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--episodes", type=int, default=24)
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--stride", type=int, default=4, help="capture every Nth control step")
parser.add_argument("--base_speed", type=float, default=1.2)
parser.add_argument("--out_root", type=str, default="out/yolo_warehouse")
parser.add_argument("--min_box_px", type=int, default=6, help="drop boxes smaller than this (either side)")
parser.add_argument("--max_frames", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math as _math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (  # noqa: E402
    WarehouseNavEnvCfg_PLAY_WithSensors_Coll)
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg  # noqa: E402
from fused_model import FusedSensorNet  # noqa: E402

_VCFG = VelocityCommandActionCfg()
MAX_SPEED, MAX_YAWRATE, MAX_INCL = _VCFG.max_speed, _VCFG.max_yawrate, _VCFG.max_inclination
TASK_ID = "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0"
GATE_CENTERS_2D = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)
PASS_RADIUS = GW.FixedGateCourseCommandCfg().success_radius

# class collapse. nc=2 {gate, person}: static props (pallet/crate/box/cone/klt/forklift) are handled by
# nav's camera+ToF (ToF-redundancy finding), so they're dropped from detection; only gates (navigation)
# and people (moving hazards) are learned. Set DIMA_YOLO_NC=3 to restore the old 3-class map.
if os.environ.get("DIMA_YOLO_NC", "2") == "3":
    RAW2ID = {"gate": 0, "person": 1, "pallet": 2, "crate": 2, "box": 2, "cone": 2, "klt": 2, "forklift": 2}
    CLASS_NAMES = ["gate", "person", "obstacle"]
else:
    RAW2ID = {"gate": 0, "person": 1}   # static props dropped (no label emitted for them)
    CLASS_NAMES = ["gate", "person"]
TARGET_H, _K_ALT, _VZ_MAX = 2.0, 1.2, 0.8


def log(m): print(m, flush=True)


def cmd_to_action(yr, fwd, h, dev, N):
    a0 = float(fwd) / (MAX_SPEED / 2.0) - 1.0
    a2 = float(yr) / MAX_YAWRATE
    speed = max(0.05, a0 + 1.0)
    vz_des = max(-_VZ_MAX, min(_VZ_MAX, _K_ALT * (TARGET_H - float(h))))
    s = max(-1.0, min(1.0, vz_des / speed))
    a1 = _math.asin(s) / MAX_INCL
    return torch.tensor([[a0, a1, a2, 0.0]], device=dev, dtype=torch.float32).clamp(-1, 1).repeat(N, 1)


def boxes_from_seg(seg_hw, id2sem, W, H, min_px):
    """seg_hw: (H,W) int32; id2sem: {idstr:{'class':name}}. -> list of (cls_id, cx,cy,w,h) normalized."""
    out = []
    ids = np.unique(seg_hw)
    for iid in ids.tolist():
        sem = id2sem.get(iid) or id2sem.get(str(iid))   # keys may be int OR str
        if not sem:
            continue
        raw = (sem.get("class") or "").lower()
        if raw not in RAW2ID:
            continue
        ys, xs = np.where(seg_hw == iid)
        if xs.size == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < min_px or bh < min_px:
            continue
        cx, cy = (x0 + x1 + 1) / 2.0 / W, (y0 + y1 + 1) / 2.0 / H
        out.append((RAW2ID[raw], cx, cy, bw / W, bh / H, raw))
    return out


def split_of(seed):
    r = seed % 5
    return "val" if r == 0 else ("test" if r == 1 else "train")


def main():
    env_cfg = WarehouseNavEnvCfg_PLAY_WithSensors_Coll()
    env_cfg.scene.num_envs = 1
    # add the instance-seg annotator to the existing front camera (no sensors.py edit)
    cam_cfg = env_cfg.scene.front_camera
    cam_cfg.data_types = list(cam_cfg.data_types) + ["instance_segmentation_fast"]
    cam_cfg.colorize_instance_segmentation = False
    cam_cfg.semantic_filter = ["class"]
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(args_cli.out_root, sub, sp), exist_ok=True)

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
    _venc = "cnn" if any(k.startswith("vision_cnn.") for k in _sd) else "vit"
    model = FusedSensorNet(out_dim=2, vision_encoder=_venc).to(dev).eval()
    model.load_state_dict(_sd, strict=True)
    robot = uenv.scene["robot"]; origin = uenv.scene.env_origins
    cam = uenv.scene["front_camera"]
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

    counts = {"train": 0, "val": 0, "test": 0}
    per_class = {c: 0 for c in CLASS_NAMES}
    total_frames = 0
    for ep in range(args_cli.episodes):
        seed = args_cli.seed + ep
        os.environ["FOREST_GATE_SEED"] = str(seed)
        torch.manual_seed(seed)
        # sweep density across episodes for diversity
        dens = [0.15, 0.3, 0.45][ep % 3]
        env_cfg.events.reset_obstacles.params["prop_density"] = dens
        if hasattr(est, "reset"):
            est.reset()
        env.reset()
        hidden = None
        goal_idx = 0
        sp = split_of(seed)
        for t in range(args_cli.max_steps):
            xy_now = (robot.data.root_pos_w[0] - origin[0])[:2].cpu().numpy().astype(np.float64)
            h_now = float(robot.data.root_pos_w[0, 2].item())
            yaw_now = _yaw_of(robot.data.root_quat_w)
            goal_xy = GATE_CENTERS_2D[min(goal_idx, K - 1)]
            if np.linalg.norm(xy_now - goal_xy) < PASS_RADIUS and goal_idx < K - 1:
                goal_idx += 1; goal_xy = GATE_CENTERS_2D[goal_idx]
            dvec = goal_xy - xy_now
            cy, sy = _math.cos(-yaw_now), _math.sin(-yaw_now)
            bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
            nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
            desired_vel = torch.tensor([[bx / nrm * args_cli.base_speed, by / nrm * args_cli.base_speed, 0.0]],
                                       device=dev, dtype=torch.float32).repeat(N, 1)
            inp = sense(desired_vel)
            with torch.no_grad():
                cmd, hidden = model(inp, hidden, mask=None)
            yaw_rate = float(cmd[0, 0].item()); fwd = float(max(0.1, min(MAX_SPEED, cmd[0, 1].item())))

            # capture BEFORE stepping (seg aligned with current render)
            if t % args_cli.stride == 0 and total_frames < args_cli.max_frames:
                seg = cam.data.output["instance_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
                _info_all = cam.data.info[0] if isinstance(cam.data.info, (list, tuple)) else cam.data.info
                info = _info_all["instance_segmentation_fast"] if isinstance(_info_all, dict) and "instance_segmentation_fast" in _info_all else _info_all
                id2sem = (info or {}).get("idToSemantics", (info or {}).get("idToLabels", {})) if isinstance(info, dict) else {}
                if os.environ.get("DEBUG_SEG") and t < 24:
                    vis = {int(i): (id2sem.get(int(i)) or id2sem.get(str(i)) or {}).get("class")
                           for i in np.unique(seg).tolist()}
                    log(f"  [dbg t={t}] visible id->class: {vis}")
                Hh, Ww = seg.shape
                boxes = boxes_from_seg(seg, id2sem, Ww, Hh, args_cli.min_box_px)
                if boxes:  # only keep frames with >=1 labelled object
                    grey = inp["front_grey"][0, 0].cpu().numpy()
                    img = (np.clip(grey, 0, 1) * 255).astype(np.uint8)
                    img3 = np.repeat(img[:, :, None], 3, axis=2)  # greyscale->3ch (HM01B0 style)
                    stem = f"{sp}_s{seed}_t{t:04d}"
                    imageio.imwrite(os.path.join(args_cli.out_root, "images", sp, stem + ".png"), img3)
                    with open(os.path.join(args_cli.out_root, "labels", sp, stem + ".txt"), "w") as f:
                        for cid, cx, cyb, bw, bh, raw in boxes:
                            f.write(f"{cid} {cx:.6f} {cyb:.6f} {bw:.6f} {bh:.6f}\n")
                            per_class[CLASS_NAMES[cid]] += 1
                    counts[sp] += 1; total_frames += 1

            obs, _r, dones, _i = env.step(cmd_to_action(yaw_rate, fwd, h_now, dev, N))
            if bool(dones[0].item()):
                break
        log(f"[ep{ep:02d}] seed={seed} dens={dens} split={sp} frames={counts}")
        if total_frames >= args_cli.max_frames:
            break

    # dataset.yaml
    yaml_path = os.path.join(args_cli.out_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args_cli.out_root)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    manifest = {"counts": counts, "per_class_boxes": per_class, "total_frames": total_frames,
                "classes": CLASS_NAMES, "raw2id": RAW2ID, "args": vars(args_cli),
                "sim": {"task": TASK_ID, "cam_res": list(seg.shape) if 'seg' in dir() else None}}
    json.dump(manifest, open(os.path.join(args_cli.out_root, "generation_manifest.json"), "w"), indent=2)
    log(f"\n[done] frames={counts} per_class={per_class} -> {args_cli.out_root}")
    log(f"[out] {yaml_path}")


if __name__ == "__main__":
    main()
    os._exit(0)
