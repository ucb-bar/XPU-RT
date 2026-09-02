"""Quantitative navigation EVAL harness (W4) — the instrument that answers "do we have enough
/ good-enough training data?" by measuring generalization on HELD-OUT seeds.

Rolls out a policy on a registered warehouse-nav task for many episodes, records per-step state
(pos/vel/action/nearest-obstacle/goal), reduces each episode with warehouse_nav.eval_metrics,
and writes: per-episode CSV, an aggregate summary, and plots (top-down trajectories coloured by
outcome + closest-obstacle-distance vs time). Optionally sweeps a commanded forward-speed cap
(vitfly-style) and plots success/jerk/clearance vs speed.

Policies:
  --policy scripted   naive goal-seeker (no obstacle avoidance) — validates the instrument today,
                      before any nav policy is trained (gives a real mix of success/collision).
  --policy random     action-space noise (floor/sanity baseline).
  --policy <path.pt>  a trained rsl_rl checkpoint (exported policy).  [loads when we have one]

    <xpurt python> sims/scripts/eval_suite.py --headless --task Isaac-Drone-Warehouse-Nav-Crazyflie-Play-v0 \
        --policy scripted --episodes-per-env 4 --seed 12345
"""
import argparse, math, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Nav-Crazyflie-Play-v0")
parser.add_argument("--policy", type=str, default="scripted", help="scripted | random | <ckpt.pt>")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes-per-env", type=int, default=4)
parser.add_argument("--seed", type=int, default=12345, help="HELD-OUT eval seed (distinct from training)")
parser.add_argument("--sweep", type=str, default="", help="comma speed caps for a vitfly-style sweep, e.g. 0.5,1.0,1.5")
parser.add_argument("--max-steps", type=int, default=6000, help="global rollout step cap (safety)")
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab_tasks.utils import parse_env_cfg
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401  (registers the gym ids)
from sims.isaaclab_tasks.warehouse_nav import mdp_nav
from sims.isaaclab_tasks.warehouse_nav import eval_metrics as EM


def scripted_policy(uenv, speed_cap=1.0):
    """Naive goal-seeker: yaw toward the goal, climb/descend toward its height, fly forward.
    NO obstacle avoidance — it will hit clutter it can't turn around, which is exactly what we
    want to measure a floor for. speed_cap in [0,1] scales the forward channel (sweep knob)."""
    g = mdp_nav.goal_vector_b(uenv)                    # (N,4): unit dir (yaw frame) + distance
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    a = torch.zeros(uenv.num_envs, 4, device=uenv.device)
    yaw_err = torch.atan2(gy, gx)                      # +y is left in the yaw frame
    a[:, 2] = (yaw_err / (math.pi / 2)).clamp(-1, 1)   # yawrate channel
    a[:, 1] = (gz * 2.0).clamp(-1, 1)                  # inclination (climb) channel
    facing = gx.clamp(min=0.0)                         # ~cos(yaw error): slow when goal is behind
    a[:, 0] = (speed_cap * (0.35 + 0.65 * facing)).clamp(0, 1)  # forward speed channel
    return a


def load_ckpt_policy(path, uenv):
    obj = torch.load(path, map_location=uenv.device)
    if hasattr(obj, "eval"):          # TorchScript / exported nn.Module
        obj.eval()
        return lambda o: obj(o)
    raise SystemExit(f"[eval] unsupported checkpoint format at {path}; expected an exported policy module")


def local_pos(uenv):
    return (uenv.scene["robot"].data.root_pos_w - uenv.scene.env_origins).cpu().numpy()


def main():
    N = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    dt = uenv.step_dt  # control-rate seconds
    print(f"[eval] task={args_cli.task} envs={N} dt={dt:.4f}s policy={args_cli.policy} seed={args_cli.seed}", flush=True)

    speeds = [float(s) for s in args_cli.sweep.split(",")] if args_cli.sweep else [1.0]
    use_ckpt = os.path.isfile(args_cli.policy)
    ckpt_fn = load_ckpt_policy(args_cli.policy, uenv) if use_ckpt else None

    outdir = os.path.join(_ROOT, "out", "eval")
    os.makedirs(outdir, exist_ok=True)
    tag = f"__{args_cli.tag}" if args_cli.tag else ""
    polname = "ckpt" if use_ckpt else args_cli.policy

    episodes_by_speed = {}           # speed_cap -> [episode metric dicts]
    trajectories_by_speed = {}       # speed_cap -> [(pos array, outcome str)]  (first speed only, for plots)

    for speed_cap in speeds:
        obs, _ = env.reset()
        buf = [dict(pos=[], vel=[], action=[], min_obs_d=[]) for _ in range(N)]
        goal_local = [None] * N
        done_counts = [0] * N
        eps, trajs = [], []
        step = 0
        target = args_cli.episodes_per_env
        while min(done_counts) < target and step < args_cli.max_steps:
            step += 1
            # --- record pre-step state (the state the action is based on) ---
            p = local_pos(uenv)
            v = uenv.scene["robot"].data.root_lin_vel_w.cpu().numpy()
            d = mdp_nav.active_obstacle_min_dist(uenv).cpu().numpy()
            gpos = (uenv.command_manager.get_term("goal").goal_pos_w - uenv.scene.env_origins).cpu().numpy()
            # --- action ---
            if ckpt_fn is not None:
                with torch.no_grad():
                    action = ckpt_fn(obs["policy"] if isinstance(obs, dict) else obs)
            elif args_cli.policy == "random":
                action = torch.rand(N, 4, device=uenv.device) * 2 - 1
            else:
                action = scripted_policy(uenv, speed_cap)
            for e in range(N):
                buf[e]["pos"].append(p[e]); buf[e]["vel"].append(v[e])
                buf[e]["action"].append(action[e].detach().cpu().numpy()); buf[e]["min_obs_d"].append(d[e])
                if goal_local[e] is None:
                    goal_local[e] = gpos[e]
            obs, _, terminated, truncated, _ = env.step(action)
            term = terminated.cpu().numpy(); trunc = truncated.cpu().numpy()
            succ = uenv._ep_success.cpu().numpy() if hasattr(uenv, "_ep_success") else np.zeros(N, bool)
            crash = uenv._ep_crash.cpu().numpy() if hasattr(uenv, "_ep_crash") else np.zeros(N, bool)
            for e in range(N):
                if not (term[e] or trunc[e]):
                    continue
                traj = {k: np.asarray(buf[e][k]) for k in buf[e]}
                traj["goal"] = goal_local[e]
                is_to = bool(trunc[e]) and not bool(term[e] and not trunc[e])
                traj["success"] = bool(succ[e]); traj["collision"] = bool(crash[e])
                traj["timeout"] = bool(is_to) and not bool(succ[e])
                if done_counts[e] < target:
                    m = EM.episode_metrics(traj, dt)
                    m["env"] = e; m["speed_cap"] = speed_cap
                    eps.append(m)
                    outc = "success" if m["success"] else ("collision" if m["collision"] else "timeout")
                    trajs.append((traj["pos"], outc))
                done_counts[e] += 1
                buf[e] = dict(pos=[], vel=[], action=[], min_obs_d=[]); goal_local[e] = None
            if step % 200 == 0:
                print(f"[eval] speed={speed_cap} step={step} episodes={[c for c in done_counts]}", flush=True)
        episodes_by_speed[speed_cap] = eps
        trajectories_by_speed[speed_cap] = trajs
        summ = EM.aggregate(eps)
        print(f"[eval] speed_cap={speed_cap}: {summ}", flush=True)

    # ---------- write CSV + summary ----------
    import csv
    all_eps = [e for eps in episodes_by_speed.values() for e in eps]
    csvp = os.path.join(outdir, f"eval__{args_cli.task}__{polname}{tag}.csv")
    keys = ["speed_cap", "env", "success", "collision", "timeout", "spl", "path_len_m",
            "straight_len_m", "min_clearance_m", "mean_speed_ms", "max_speed_ms",
            "mean_accel_ms2", "mean_jerk_ms3", "mean_action_rate", "time_to_goal_s", "duration_s", "steps"]
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for e in all_eps:
            w.writerow(e)
    print(f"[eval] wrote {csvp} ({len(all_eps)} episodes)", flush=True)

    # ---------- plots ----------
    # (a) top-down trajectories (first speed), coloured by outcome
    trajs = trajectories_by_speed[speeds[0]]
    col = {"success": "#1a9850", "collision": "#d73027", "timeout": "#4575b4"}
    fig, ax = plt.subplots(figsize=(6, 7))
    for pos, outc in trajs:
        ax.plot(pos[:, 0], pos[:, 1], color=col[outc], alpha=0.6, lw=1.3)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=col[outc], ms=3)
    for k, c in col.items():
        ax.plot([], [], color=c, label=k)
    ax.set_aspect("equal"); ax.set_xlabel("x (m, env-local)"); ax.set_ylabel("y (m, env-local)")
    ax.set_title(f"trajectories — {polname} (seed {args_cli.seed})"); ax.legend(loc="best", fontsize=8)
    trajp = os.path.join(outdir, f"traj__{args_cli.task}__{polname}{tag}.png")
    fig.tight_layout(); fig.savefig(trajp, dpi=110); plt.close(fig)
    print(f"[eval] wrote {trajp}", flush=True)

    # (b) per-episode min-clearance (closest-obstacle) distribution — CRL closest-obstacle plot
    fig, ax = plt.subplots(figsize=(7, 4))
    mins = [e["min_clearance_m"] for e in episodes_by_speed[speeds[0]] if np.isfinite(e["min_clearance_m"])]
    ax.hist(mins, bins=20, color="#666"); ax.axvline(0.5, color="r", ls="--", label="0.5 m")
    ax.set_xlabel("per-episode min clearance (m)"); ax.set_ylabel("episodes"); ax.legend()
    ax.set_title(f"closest-obstacle distribution — {polname}")
    clrp = os.path.join(outdir, f"clearance__{args_cli.task}__{polname}{tag}.png")
    fig.tight_layout(); fig.savefig(clrp, dpi=110); plt.close(fig)
    print(f"[eval] wrote {clrp}", flush=True)

    # (c) speed sweep, if requested
    if len(speeds) > 1:
        rows = EM.speed_sweep(episodes_by_speed)
        sp = [r["speed_setpoint_ms"] for r in rows]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        axs[0].plot(sp, [r["success_rate"] for r in rows], "-o"); axs[0].set_title("success rate"); axs[0].set_ylim(0, 1)
        axs[1].plot(sp, [r["mean_jerk_ms3_mean"] for r in rows], "-o", color="#d95f02"); axs[1].set_title("mean jerk (m/s³)")
        axs[2].plot(sp, [r["min_clearance_m_mean"] for r in rows], "-o", color="#7570b3"); axs[2].set_title("mean min-clearance (m)")
        for a in axs:
            a.set_xlabel("forward-speed cap")
        swp = os.path.join(outdir, f"sweep__{args_cli.task}__{polname}{tag}.png")
        fig.tight_layout(); fig.savefig(swp, dpi=110); plt.close(fig)
        print(f"[eval] wrote {swp}", flush=True)

    print("[eval] SUMMARY:", EM.aggregate(all_eps), flush=True)


if __name__ == "__main__":
    main()
    print("[eval] done; hard-exiting to skip the hanging close()", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
