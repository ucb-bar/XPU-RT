"""Closed-loop forest-trail nav eval (Workstream B.6 / task #57).

The real "genuinely good?" acceptance test for a nav head: fly it in the forest
env over N seeded episodes and measure trail-following, not offline accuracy.
The nav model (greyscale DroNet — regression or classifier — or the ViT trail
head) sees the onboard FPV, emits a yaw-rate, and the FROZEN steering inner-loop
policy (model_6998) tracks it. Same DroNet->yaw seam as
``pilot_forest_with_dronet_scheduled.py`` (softmax-expected yaw for the
classifier), stripped of the scheduler/YOLO/video machinery.

Per episode we log, in the env-local frame:
  * progress   = along-trail distance / trail length   (straight: x/L; curved: arc/total)
  * offset     = lateral distance from centreline       (straight: |y|; curved: perp dist)
  * jerk proxy = mean |Δ yaw-rate| per control step
Episode outcome: SUCCESS if progress >= --success_frac, else OFF_TRAIL / CRASH
(read from the termination manager when available, else inferred from geometry).

    <env_isaaclab py> sims/scripts/eval_forest_nav.py --headless \
        --nav_arch dronet --nav_head regression \
        --nav_weights <run>/best.pt --trail straight --episodes 10

--headless is REQUIRED (offscreen kit → the FPV camera actually renders; see
dima-isaac-headless-render). Head-to-head: run once per nav head on the SAME
--seed base and compare the tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# isaaclab is a SOURCE checkout here (not pip-installed) — add it to path before import.
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
from isaaclab.app import AppLauncher  # noqa: E402

# ---- CLI (parse BEFORE launching the app) ----
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nav_arch", choices=["dronet", "vit"], default="dronet")
parser.add_argument("--nav_head", choices=["regression", "classifier"], default="regression")
parser.add_argument("--nav_size", choices=["small", "large"], default="small", help="DroNet only.")
parser.add_argument("--nav_rgb", action="store_true", help="Nav model uses RGB (default greyscale).")
parser.add_argument("--nav_weights", type=str, required=True)
parser.add_argument("--inner_ckpt", type=str,
                    default="/scratch2/dima/misc_sw/FreshScheduler/logs/rsl_rl/"
                            "crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt",
                    help="Frozen steering inner-loop policy (model_6998).")
parser.add_argument("--trail", choices=["straight", "curved"], default="straight")
parser.add_argument("--with_humans", action="store_true")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1000, help="Base seed; episode i uses seed+i.")
parser.add_argument("--max_steps", type=int, default=1500, help="Control steps per episode cap.")
parser.add_argument("--forward_velocity", type=float, default=1.0)
parser.add_argument("--omega_clamp", type=float, default=1.5)
parser.add_argument("--camera_update_period", type=float, default=0.1, help="Nav inference period (s), ZOH between.")
parser.add_argument("--success_frac", type=float, default=0.9, help="Progress fraction counted as reaching the end.")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)  # adds --headless, --enable_cameras, --device, etc.
args_cli = parser.parse_args()
args_cli.headless = True          # offscreen kit → FPV camera renders (see dima-isaac-headless-render)
args_cli.enable_cameras = True    # forest env FPV camera
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# freshscheduler_root == XPU-RT; the vitfly zoo lives at DIMA/vitfly/models (sibling of XPU-RT).
VITFLY_MODELS = os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models"))
if VITFLY_MODELS not in sys.path:
    sys.path.insert(0, VITFLY_MODELS)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from qnn_models.dronet import DronetTorch  # noqa: E402
from sims.isaaclab_tasks.forest_trail.config import crazyflie as _forest_register  # noqa: E402,F401
from sims.isaaclab_tasks.track_steering_vision.config import crazyflie as _track_register  # noqa: E402,F401
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    SteeringTrackingPPORunnerCfg,
)
from sims.isaaclab_tasks.forest_trail.config.crazyflie.forest_env_cfg import (  # noqa: E402
    ForestTrailEnvCfg_PLAY,
    ForestTrailEnvCfg_PLAY_WithHumans,
    ForestTrailEnvCfg_Curved_PLAY,
    ForestTrailEnvCfg_Curved_PLAY_WithHumans,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def preprocess(rgb_uint8: np.ndarray, img_size: int, device: str, greyscale: bool) -> torch.Tensor:
    """(H,W,3) uint8 RGB -> (1,C,S,S) float in [0,1]. Matches the pilot seam."""
    img = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device).float().div_(255.0)
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
    if greyscale:
        w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
        img = (img * w).sum(dim=1, keepdim=True)
    return F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)


def build_nav_model(device):
    img_size = 112 if args_cli.nav_size == "small" else 224
    grey = not args_cli.nav_rgb
    if args_cli.nav_arch == "vit":
        from trail_vit import TrailViT
        model = TrailViT(head=args_cli.nav_head)
        img_size = 112  # ViT interpolates internally to 60x90; feed 112
    else:
        model = DronetTorch(
            img_dims=(img_size, img_size), img_channels=3 if args_cli.nav_rgb else 1,
            output_dim=3 if args_cli.nav_head == "classifier" else 1,
            small=(args_cli.nav_size == "small"), head=args_cli.nav_head,
        )
    state = torch.load(args_cli.nav_weights, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    log(f"[nav] {args_cli.nav_arch} head={args_cli.nav_head} "
        f"{'grey' if grey else 'rgb'} input={img_size} weights={args_cli.nav_weights}")
    return model, img_size, grey


def nav_yaw(model, x) -> float:
    """Run the nav model -> clamped yaw-rate command (rad/s)."""
    with torch.no_grad():
        out, _coll = model(x)
    if args_cli.nav_head == "classifier":
        p = torch.softmax(out, dim=1)[0]           # [lc, sc, rc]
        steer = float(args_cli.omega_clamp * (p[2] - p[0]).item())
    else:
        steer = float(out.item())
    return max(-args_cli.omega_clamp, min(args_cli.omega_clamp, steer))


# ---- trail geometry (mirrors mdp_terminations) ----
class Geometry:
    def __init__(self, env_cfg, curved: bool):
        self.curved = curved
        p = env_cfg.terminations.off_trail.params
        if curved:
            self.pts = np.asarray(p["waypoints"], dtype=np.float64)  # (M,2)
            seg = self.pts[1:] - self.pts[:-1]
            self.seg_len = np.linalg.norm(seg, axis=1)
            self.total_arc = float(self.seg_len.sum())
            self.arc_starts = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
            self.lateral_margin = float(p.get("lateral_margin", 3.0))
        else:
            self.trail_length = float(p.get("trail_length", 30.0))
            self.lateral_margin = float(p.get("lateral_margin", 3.0))

    def progress_offset(self, xy: np.ndarray):
        """Return (progress_frac in [0,1+], lateral_offset_m)."""
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


def main() -> int:
    device = args_cli.device if torch.cuda.is_available() else "cpu"
    curved = args_cli.trail == "curved"
    if curved:
        cfg_cls = ForestTrailEnvCfg_Curved_PLAY_WithHumans if args_cli.with_humans else ForestTrailEnvCfg_Curved_PLAY
        task_id = ("Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-v0" if args_cli.with_humans
                   else "Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-v0")
    else:
        cfg_cls = ForestTrailEnvCfg_PLAY_WithHumans if args_cli.with_humans else ForestTrailEnvCfg_PLAY
        task_id = ("Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0" if args_cli.with_humans
                   else "Isaac-Forest-Trail-Vision-Crazyflie-Play-v0")

    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.fpv_camera.update_period = min(args_cli.camera_update_period, float(env_cfg.sim.dt * env_cfg.decimation))
    # Keep episodes long enough to traverse; we cap by --max_steps ourselves.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    geom = Geometry(env_cfg, curved)

    log(f"[env] gym.make {task_id}")
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True

    # frozen steering inner loop (model_6998)
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
    # rsl_rl version drift: model_6998 was trained under the old (now-deleted) xpurt
    # env whose Gaussian stored `distribution.std_param`; the current rsl_rl expects
    # `distribution.log_std_param`. Inference uses the actor MEAN (std is irrelevant),
    # so remap std->log(std) into a patched checkpoint and load that (keeps strict=True,
    # obs-normalizer state, etc. intact).
    inner_ckpt = args_cli.inner_ckpt
    _loaded = torch.load(inner_ckpt, map_location=device, weights_only=False)
    _asd = _loaded.get("actor_state_dict", {})
    if "distribution.std_param" in _asd and "distribution.log_std_param" not in _asd:
        _asd["distribution.log_std_param"] = _asd.pop("distribution.std_param").clamp_min(1e-6).log()
        inner_ckpt = os.path.join(
            "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad",
            "model_6998_logstd.pt")
        torch.save(_loaded, inner_ckpt)
        log(f"[inner] remapped std_param->log_std_param (rsl_rl drift) -> {inner_ckpt}")
    runner.load(inner_ckpt)
    inner_policy = runner.get_inference_policy(device=device)
    log(f"[inner] steering policy loaded: {args_cli.inner_ckpt}")

    nav_model, img_size, grey = build_nav_model(device)
    steering_term = uenv.command_manager.get_term("steering_command")
    camera = uenv.scene["fpv_camera"]
    robot = uenv.scene["robot"]
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    cam_every = max(1, int(round(args_cli.camera_update_period / control_dt)))

    def read_outcome(progress, offset, height):
        """Classify a finished episode. Prefer the termination manager, fall back to geometry."""
        try:
            tm = uenv.termination_manager
            terms = {n: bool(tm.get_term(n)[0].item()) for n in tm.active_terms}
        except Exception:
            terms = {}
        if progress >= args_cli.success_frac:
            return "success"
        if terms.get("off_trail") or offset > geom.lateral_margin:
            return "off_trail"
        if any("crash" in n and v for n, v in terms.items()) or height < 0.2:
            return "crash"
        if terms.get("time_out"):
            return "timeout"
        return "off_trail" if offset > geom.lateral_margin * 0.6 else "crash"

    results = []
    for ep in range(args_cli.episodes):
        torch.manual_seed(args_cli.seed + ep)
        reset_out = env.reset()  # RslRlVecEnvWrapper.reset() -> (obs, extras) or obs
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        cached_w = 0.0
        max_prog, off_sum, off_n, off_max = 0.0, 0.0, 0, 0.0
        prev_w, jerk_sum = None, 0.0
        outcome, last_h = "timeout", 1.0
        for t in range(args_cli.max_steps):
            if t % cam_every == 0:
                rgb = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                x = preprocess(np.ascontiguousarray(rgb), img_size, device, grey)
                cached_w = nav_yaw(nav_model, x)
            steering_term.target_velocity.fill_(args_cli.forward_velocity)
            steering_term.target_yaw_rate.fill_(cached_w)
            with torch.no_grad():
                actions = inner_policy(obs)
            obs, _rew, dones, _info = env.step(actions)

            xy = (robot.data.root_pos_w[0] - uenv.scene.env_origins[0])[:2].cpu().numpy().astype(np.float64)
            last_h = float(robot.data.root_pos_w[0, 2].item())
            yaw_rate = float(robot.data.root_ang_vel_b[0, 2].item())
            prog, off = geom.progress_offset(xy)
            max_prog = max(max_prog, prog)
            off_sum += off; off_n += 1; off_max = max(off_max, off)
            if prev_w is not None:
                jerk_sum += abs(yaw_rate - prev_w)
            prev_w = yaw_rate
            if bool(dones[0].item()):
                outcome = read_outcome(max_prog, off, last_h)
                break
        else:
            # loop hit --max_steps without the env terminating: reached the end,
            # or simply ran out of budget mid-trail (NOT a crash).
            outcome = "success" if max_prog >= args_cli.success_frac else "timeout"
        rec = {"episode": ep, "seed": args_cli.seed + ep, "outcome": outcome,
               "progress": round(max_prog, 4), "mean_offset": round(off_sum / max(1, off_n), 4),
               "max_offset": round(off_max, 4), "mean_jerk": round(jerk_sum / max(1, off_n), 5),
               "steps": off_n}
        results.append(rec)
        log(f"[ep{ep:02d}] outcome={outcome:9s} progress={max_prog:.2f} "
            f"mean_off={rec['mean_offset']:.2f} max_off={off_max:.2f} steps={off_n}")

    n = len(results)
    succ = sum(r["outcome"] == "success" for r in results)
    agg = {
        "trail": args_cli.trail, "nav_arch": args_cli.nav_arch, "nav_head": args_cli.nav_head,
        "episodes": n, "success_rate": round(succ / max(1, n), 3),
        "mean_progress": round(sum(r["progress"] for r in results) / max(1, n), 3),
        "mean_offset": round(sum(r["mean_offset"] for r in results) / max(1, n), 3),
        "mean_jerk": round(sum(r["mean_jerk"] for r in results) / max(1, n), 5),
        "outcomes": {k: sum(r["outcome"] == k for r in results) for k in ("success", "off_trail", "crash", "timeout")},
    }
    log("\n=== FOREST NAV EVAL ===")
    log(json.dumps(agg, indent=2))
    out = args_cli.out or os.path.join(
        "/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad",
        f"forestnav_{args_cli.nav_arch}_{args_cli.nav_head}_{args_cli.trail}.json")
    with open(out, "w") as f:
        json.dump({"agg": agg, "episodes": results, "args": vars(args_cli)}, f, indent=2)
    log(f"[out] wrote {out}")
    return 0


if __name__ == "__main__":
    main()
    # simulation_app.close() hangs indefinitely in this build; outputs already
    # flushed -> hard-exit. See dima-isaac-training-env.
    os._exit(0)
