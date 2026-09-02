"""Debug / verification harness for the forest-trail onboard SENSOR RIG (Workstream S).

Launches a ``*_WithSensors`` forest-trail env headless (with --enable_cameras), and:

  1. Steps the env a few times at the spawn and dumps the front greyscale (N)
     frame + the four 8x8 ToF patches so we can eyeball the raw sensor output.
  2. Runs an explicit "obstacle-ahead" test: teleports the drone ~2.5 m in front
     of a real tree, facing it (yaw toward the tree), and confirms the FORWARD
     (N) ToF patch reports SMALLER ranges than the side (E/S/W) patches.

Per-step per-ToF min/mean/max ranges are printed, and PNGs (greyscale frame +
ToF heatmaps + cross-composite) are written to the scratchpad.

    <xpurt python> sims/scripts/debug_forest_sensors.py --headless --enable_cameras

Isaac's close() hangs for minutes, so we os._exit(0) after writing outputs.
"""
import argparse, math, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

_SCRATCH = ("/tmp/claude-2621/-scratch-agustin-projects-DIMA/"
            "057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-WithSensors-v0")
parser.add_argument("--free_steps", type=int, default=4, help="spawn frames to capture")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--outdir", type=str, default=_SCRATCH)
parser.add_argument("--noise", action="store_true", help="also print a noised ToF stack")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch
import imageio.v2 as imageio
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from isaaclab_tasks.utils import parse_env_cfg
import sims.isaaclab_tasks.forest_trail.config.crazyflie  # noqa: F401  (registers gym ids)
from sims.isaaclab_tasks.forest_trail import sensors
from sims.isaaclab_tasks.forest_trail.forest_scene import DEFAULT_STRAIGHT_POSITIONS


def _stats_line(tag, stack):
    """One-line per-ToF min/mean/max for a (4,8,8) stack ordered [N,E,S,W]."""
    parts = []
    for i, d in enumerate(sensors.TOF_ORDER):
        p = stack[i]
        parts.append(f"{d}:[{p.min():.2f}/{p.mean():.2f}/{p.max():.2f}]")
    print(f"[dbg] {tag} ToF min/mean/max (m)  " + "  ".join(parts), flush=True)


def _save_greyscale(env, path):
    g = sensors.front_greyscale(env)[0, 0]  # (H, W) float [0,1]
    img = (g.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, img)
    print(f"[dbg] wrote greyscale {path}  shape={tuple(img.shape)} (1-channel)", flush=True)


def _save_tof_heatmap(stack, composite, path, title):
    """Heatmap PNG: 4 patches [N,E,S,W] on top row + the cross-composite."""
    fig = Figure(figsize=(12, 4))
    FigureCanvasAgg(fig)
    for i, d in enumerate(sensors.TOF_ORDER):
        ax = fig.add_subplot(1, 5, i + 1)
        im = ax.imshow(stack[i], cmap="viridis_r", vmin=sensors.TOF_RANGE_MIN,
                       vmax=sensors.TOF_RANGE_MAX)
        ax.set_title(f"ToF {d}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(1, 5, 5)
    im = ax.imshow(composite, cmap="viridis_r", vmin=sensors.TOF_RANGE_MIN,
                   vmax=sensors.TOF_RANGE_MAX)
    ax.set_title("cross-composite")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[dbg] wrote ToF heatmap {path}", flush=True)


def _capture(uenv, tag, outdir, idx):
    """Read sensors, print stats, dump PNGs. Returns the (4,8,8) numpy stack.

    Reads are taken off the *unwrapped* env (``uenv``); the gym wrapper does not
    expose ``.scene`` / ``.num_envs``.
    """
    stack_t = sensors.tof_stack(uenv)          # (1, 4, 8, 8)
    comp_t = sensors.tof_cross_composite(uenv)  # (1, 24, 24)
    assert stack_t.shape == (uenv.num_envs, 4, sensors.TOF_ZONES, sensors.TOF_ZONES), stack_t.shape
    assert comp_t.shape == (uenv.num_envs, 3 * sensors.TOF_ZONES, 3 * sensors.TOF_ZONES), comp_t.shape
    stack = stack_t[0].cpu().numpy()
    comp = comp_t[0].cpu().numpy()
    _stats_line(tag, stack)
    _save_greyscale(uenv, os.path.join(outdir, f"grey_{tag}.png"))
    _save_tof_heatmap(stack, comp, os.path.join(outdir, f"tof_{tag}.png"),
                      f"{tag}  (stack {tuple(stack.shape)}, composite {tuple(comp.shape)})")
    return stack


def _settle(gym_env, uenv, n):
    """Step with zero action (via the gym wrapper) so physics + cameras refresh."""
    for _ in range(n):
        gym_env.step(torch.zeros(uenv.num_envs, 4, device=uenv.device))


def main():
    os.makedirs(args_cli.outdir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    print(f"[dbg] task={args_cli.task}  dt={uenv.step_dt:.4f}s  outdir={args_cli.outdir}", flush=True)
    print(f"[dbg] sensors on scene: {[k for k in ('front_camera',) + tuple(sensors.TOF_KEYS.values())]}",
          flush=True)

    env.reset()
    _settle(env, uenv, 6)  # let physics + cameras warm up

    # ── 1. Spawn captures ─────────────────────────────────────────────────────
    for i in range(args_cli.free_steps):
        _settle(env, uenv, 2)
        _capture(uenv, f"spawn{i}", args_cli.outdir, i)

    # ── 2. Obstacle-ahead test: face the drone at a real tree ─────────────────
    # Pick the tree nearest the spawn (smallest x) and fly the drone toward it,
    # facing it. Pine trunks are thin and the foliage sits high, so we sweep a
    # few standoffs + heights and keep the closest-facing sample: the point is
    # to show the FORWARD (N) ToF drops below the (open) side patches once the
    # tree fills its cone.
    trees = sorted(DEFAULT_STRAIGHT_POSITIONS, key=lambda p: p[0])
    tx, ty = trees[0]
    origin = uenv.scene.env_origins[0]  # world origin of env 0
    yaw = math.atan2(ty, tx)  # heading from the env origin toward the tree
    dx, dy = math.cos(yaw), math.sin(yaw)
    qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    print(f"[dbg] obstacle test: tree@({tx:.2f},{ty:.2f}) yaw={math.degrees(yaw):.1f}deg; "
          f"sweeping standoff x height, facing the tree", flush=True)

    best = None  # (n_mean, standoff, z, stack)
    for stand in (2.0, 1.2, 0.7):
        for z in (1.0, 1.8, 2.6):
            px, py = tx - stand * dx, ty - stand * dy
            pose = torch.tensor([[origin[0].item() + px, origin[1].item() + py,
                                  origin[2].item() + z, qw, 0.0, 0.0, qz]], device=dev)
            for _ in range(3):  # hold pose so depth annotators re-render there
                robot.write_root_pose_to_sim(pose)
                robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
                env.step(torch.zeros(1, 4, device=dev))
            stack = sensors.tof_stack(uenv)[0].cpu().numpy()
            n_mean = float(stack[0].mean())
            _stats_line(f"stand={stand:.1f}m z={z:.1f}m", stack)
            if best is None or n_mean < best[0]:
                best = (n_mean, stand, z, stack)

    _, bstand, bz, ahead = best
    comp = sensors.tof_cross_composite(uenv)  # shape-check only (latest pose)
    assert ahead.shape == (4, sensors.TOF_ZONES, sensors.TOF_ZONES), ahead.shape
    assert comp.shape[1:] == (3 * sensors.TOF_ZONES, 3 * sensors.TOF_ZONES), comp.shape
    # Re-pose to the winning sample and dump its PNGs.
    px, py = tx - bstand * dx, ty - bstand * dy
    pose = torch.tensor([[origin[0].item() + px, origin[1].item() + py,
                          origin[2].item() + bz, qw, 0.0, 0.0, qz]], device=dev)
    for _ in range(3):
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        env.step(torch.zeros(1, 4, device=dev))
    ahead = _capture(uenv, "tree_ahead", args_cli.outdir, 0)

    # Verdict: forward (N, index 0) mean range < each side (E,S,W).
    n_mean = float(ahead[0].mean())
    side_means = {d: float(ahead[i].mean()) for i, d in enumerate(sensors.TOF_ORDER) if d != "N"}
    ok = all(n_mean < sm for sm in side_means.values())
    print(f"[dbg] best sample: standoff={bstand:.1f}m height={bz:.1f}m", flush=True)
    print(f"[dbg] VERDICT obstacle-ahead: N_mean={n_mean:.2f} m  sides={ {k: round(v,2) for k,v in side_means.items()} }",
          flush=True)
    print(f"[dbg] N closer than all sides? {'PASS' if ok else 'FAIL'}", flush=True)

    if args_cli.noise:
        rng = torch.Generator(device=dev); rng.manual_seed(args_cli.seed)
        noised = sensors.add_tof_noise(sensors.tof_stack(uenv), rng=rng)[0].cpu().numpy()
        _stats_line("tree_ahead+noise", noised)

    print("[dbg] done; hard-exiting to skip the hanging close()", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
