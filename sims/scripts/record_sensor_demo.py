"""Single multi-panel demo video: RED drone + real trained model + ALL sensor inputs + overhead path.

One composite .mp4 (for slides) that shows, all synchronized frame-for-frame:

  * TOP-LEFT  — 3rd-person CHASE cam following the bright-red drone (smooth, every frame).
  * TOP-RIGHT — the full onboard SENSOR SUITE exactly as the model receives each input, properly
    rendered per modality and refreshed at each sensor's TRUE rate:
      - FPV      : HM01B0 greyscale, interpolated to the model's 60x90 input   @ 10 Hz
      - Cross ToF: 4x VL53L5CX 8x8 zones, + layout depth heatmap               @ 10 Hz
      - Flow     : PMW3901 optical-flow vector (dx,dy)                          @ 50 Hz
      - Altitude : downward VL53L1X ToF + barometer traces                     @ 50 / 25 Hz
      - IMU      : gyro body-rates traces + Madgwick roll/pitch/yaw            @ 50 Hz
      - Goal     : desired-velocity command to the next gate (model input)     @ 50 Hz
  * BOTTOM (elongated) — a FIXED overhead camera framed tightly on the aisle the drone flies, with
    the drone's flight path drawn as a red trace (projected via the camera's real intrinsics/pose),
    the live drone position, and the 4 gate markers.

The flight is the REAL trained model: it loads our v12 CNN checkpoint and its forward pass drives the
steering/avoidance/gate-following every step (constant 0.9 m/s cruise, matching the validated eval —
the speed head underfits, the net's job is steering). Records the FIRST successful 4/4 weave-through
on the honest crowded collidable course (deepest run as fallback).

    <env_isaaclab py> sims/scripts/record_sensor_demo.py --headless \
        --weights train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
        --save_video out/v12_crowded_sensor_demo.mp4
"""

from __future__ import annotations

import argparse
import math as _math
import os
import sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
sys.path.insert(0, os.path.abspath(os.path.join(freshscheduler_root, "..", "vitfly", "models")))
sys.path.insert(0, os.path.join(freshscheduler_root, "hil"))  # for safety_layer
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--weights", type=str,
                    default=os.path.join(freshscheduler_root, "sims/models/warehouse/nav_fused_v12_cnn.pt"))
parser.add_argument("--save_video", type=str, default="out/v12_crowded_sensor_demo.mp4")
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--base_speed", type=float, default=1.4)
parser.add_argument("--obstacle_level", type=int, default=8)
parser.add_argument("--prop_density", type=float, default=0.30)
parser.add_argument("--fixed_speed", type=float, default=0.9)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--decimation", type=int, default=None,
                    help="Override env decimation. Base cfg is 2 (sim.dt 0.01 * 2 = 20 ms control = 50 Hz, "
                         "which equals the control/nav net period -> 1:1 alias). Pass 1 for 10 ms/100 Hz "
                         "control (Nyquist headroom over the 20 ms net period). NOTE: the trained MLP "
                         "controller was learned at 50 Hz; at 100 Hz halve --moment_scale or expect drift.")
parser.add_argument("--sim_dt", type=float, default=None,
                    help="Override sim.dt (base 0.01 = 100 Hz physics). e.g. 0.005 = 200 Hz physics for finer "
                         "dynamics while keeping the control rate via decimation.")
parser.add_argument("--dump_figure_data", type=str, default=None,
                    help="Capture stroboscopic-figure data (clean overhead bg + all drone poses + "
                         "cam intrinsics/pose + gates + a few sensor snapshots) to a dir, for "
                         "compose_paper_figure.py. Emits an .npz + snapshot PNGs at run end.")
parser.add_argument("--controller", choices=["classical", "rl"], default="classical",
                    help="classical = Lee velocity-command law (default); rl = the PPO-trained MLP "
                         "controller (16->256/128/64->4 ELU) via DirectThrustMoment — makes the demo "
                         "use ALL THREE learned nets (nav LSTM-conv + MLP control + YOLO).")
parser.add_argument("--rl_ckpt", type=str,
                    default=os.environ.get("MODELBLASTER_WAREHOUSE_RL_CKPT",
                    os.path.join(freshscheduler_root, "sims/models/warehouse/rl_controller_velctrl_dr4.pt")))
parser.add_argument("--cruise_speed", type=float, default=1.3)
parser.add_argument("--yaw_scale", type=float, default=1.0)
parser.add_argument("--moment_scale", type=float, default=0.01)
parser.add_argument("--gantt_schedule", type=str,
                    default="/scratch2/agustin/XPU-RT/schedules/scheduled_networks_k1_live_stack_cpsat_profiled.json",
                    help="XPU-RT scheduled_*.json to embed as the Gantt strip (red playhead synced to sim "
                         "time + sensor-input arrows). Empty string disables the strip.")
parser.add_argument("--safety", action="store_true",
                    help="USE the YOLO detections: route them through hil/safety_layer.apply_safety so "
                         "person/obstacle-ahead slows + steers the command (gates never braked). Closes "
                         "the perception->control loop instead of YOLO being observer-only.")
parser.add_argument("--yolo", type=str,
                    default=os.path.join(freshscheduler_root, "sims/models/warehouse/yolov8n_gate_person_128x192.pt"),
                    help="warehouse YOLOv8n best.pt: overlay gate/person boxes on FPV + a "
                         "K1 scheduling strip. Default = the in-repo 128x192 nc=2 detector.")
parser.add_argument("--clean_overview", action="store_true",
                    help="CLEAN-BACKGROUND mode (paper figure): place obstacles/gates/racks (reset + "
                         "hide_roof), HIDE the drone, sweep a set of fixed 3/4 axonometric iso poses "
                         "down the aisle (preview PNG each), and save clean_bg.npz (drone-free iso + "
                         "top-down backgrounds + calibrations). No flight, no video, no model needed.")
parser.add_argument("--clean_out", type=str,
                    default=os.path.join(freshscheduler_root, "sims/out/figdata_mega"),
                    help="output dir for --clean_overview (clean_bg.npz + preview PNGs).")
parser.add_argument("--iso_choice", type=str, default="",
                    help="--clean_overview: name of the swept candidate to write into clean_bg.npz "
                         "(default = the first candidate). Re-run with this once you have picked from "
                         "the preview PNGs, or rebuild offline from clean_sweep.npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})
from collections import deque  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: E402,F401
from sims.isaaclab_tasks.forest_trail import sensors as S  # noqa: E402
from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW  # noqa: E402
from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (  # noqa: E402
    WarehouseNavEnvCfg_PLAY_WithSensors_Coll,
)
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg  # noqa: E402
from sims.isaaclab_tasks.track_steering_vision.mdp_actions import DirectThrustMomentActionCfg  # noqa: E402
from safety_layer import apply_safety, SafetyConfig  # noqa: E402
from fused_model import FusedSensorNet  # noqa: E402

_VCFG = VelocityCommandActionCfg()
MAX_SPEED, MAX_YAWRATE = _VCFG.max_speed, _VCFG.max_yawrate
_MAX_INCL = _VCFG.max_inclination
TASK_ID = "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0"
GATE_CENTERS_2D = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)
GATE_Z = float(GW.FUSED_GATES[0][0][2])
PASS_RADIUS = GW.FixedGateCourseCommandCfg().success_radius

TARGET_H = 2.0
_K_ALT, _VZ_MAX = 1.2, 0.8

# Overhead camera: fixed, framed on the navigated aisle (NOT the whole warehouse).
OV_W, OV_H, OV_HFOV = 1280, 384, 74.0          # elongated strip (≈3.3:1)
OV_CENTER_LOCAL = (-8.0, 13.5)                 # aisle centre (x, y) in env-local metres
OV_HEIGHT = 13.0                               # metres above ground (roof hidden above 6.3 m)
# ROS optical-frame quaternion for a straight-down camera with image +x→world +y, image +y→world +x.
OV_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# Isometric OVERVIEW camera (paper figure): a FIXED axonometric view (NOT following the drone),
# framing the WHOLE aisle flight region — gates + full path + drone — like the top-down but 3/4.
# eye = aisle centre + ISO_EYE_OFFSET (env-local metres); centre sits at cruise altitude. The eye
# is offset to the south (open aisle mouth) + slightly west, elevated ~35°, so the racks frame the
# corridor rather than occlude it. Tune ISO_EYE_OFFSET until the region fills the 50° frame.
ISO_CENTER_LOCAL = (OV_CENTER_LOCAL[0], OV_CENTER_LOCAL[1], TARGET_H)
ISO_EYE_OFFSET = (-5.0, -14.0, 11.0)

FIG_DENSE = 10          # dump dense per-moment frames every FIG_DENSE control steps (figure data)


def log(m):
    print(m, flush=True)


def hide_roof(stage):
    """Make every mesh whose world-bound sits above 6.3 m invisible (roof/high beams), so the
    overhead camera sees the aisle floor instead of the ceiling. Copied from fpv_rich_flight.py."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    nh = 0
    for pr in stage.Traverse():
        if pr.GetTypeName() != "Mesh":
            continue
        try:
            b = cache.ComputeWorldBound(pr).ComputeAlignedRange()
            if not b.IsEmpty() and b.GetMin()[2] > 6.3:
                UsdGeom.Imageable(pr).MakeInvisible(); nh += 1
        except Exception:
            pass
    return nh


def cmd_to_action(yr, fwd, h, dev, N):
    a0 = float(fwd) / (MAX_SPEED / 2.0) - 1.0
    a2 = float(yr) / MAX_YAWRATE
    speed = max(0.05, a0 + 1.0)
    vz_des = max(-_VZ_MAX, min(_VZ_MAX, _K_ALT * (TARGET_H - float(h))))
    s = max(-1.0, min(1.0, vz_des / speed))
    a1 = _math.asin(s) / _MAX_INCL
    return torch.tensor([[a0, a1, a2, 0.0]], device=dev, dtype=torch.float32).clamp(-1.0, 1.0).repeat(N, 1)


def build_rl_actor(ckpt_path, dev):
    """RL/MLP controller actor (ELU MLP, obs_normalization=False). Architecture is INFERRED from the
    checkpoint's Linear weight shapes, so it works for any hidden_dims (e.g. [256,128,64] or the bigger
    [512,512,512,256] mlp_control_big) without editing this function."""
    import re as _re
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    asd = ck["actor_state_dict"]
    lin_idx = sorted({int(m.group(1)) for k in asd for m in [_re.match(r"mlp\.(\d+)\.weight", k)] if m})
    layers = []
    for j, i in enumerate(lin_idx):
        out_f, in_f = asd[f"mlp.{i}.weight"].shape
        layers.append(nn.Linear(in_f, out_f))
        if j < len(lin_idx) - 1:
            layers.append(nn.ELU(alpha=1.0))
    mlp = nn.Sequential(*layers).to(dev)
    stripped = {k[len("mlp."):]: v for k, v in asd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(stripped, strict=True)
    mlp.eval()
    _dims = [layers[0].in_features] + [l.out_features for l in layers if isinstance(l, nn.Linear)]
    log(f"[ctrl] actor arch inferred from ckpt: {_dims}")
    return mlp


def _quat_to_rpy(q):
    w, x, y, z = [float(v) for v in q]
    roll = _math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = _math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = _math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def project(K, pos_w, quat_wxyz, pts_world):
    """Project world points (M,3) → pixel (u,v) using the camera's real intrinsics K and pose
    (pos_w, ROS-optical cam→world quaternion). Returns (u, v, valid)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
                  [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]])
    d = np.asarray(pts_world, dtype=np.float64) - np.asarray(pos_w, dtype=np.float64)[None, :]
    Xc = d @ R                              # R^T · d  (world→cam)
    zc = np.clip(Xc[:, 2], 1e-6, None)
    u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    return u, v, (Xc[:, 2] > 0.05)


# ── Per-sensor refresh gates at the 50 Hz control loop (round(50 / true_Hz)) ──────────────────
CTRL_HZ = 50.0
G_FPV = max(1, round(CTRL_HZ / 10.0))                            # front cam 10 Hz -> 5
G_TOF = max(1, round(CTRL_HZ / 10.0))                            # ToF 10 Hz       -> 5
G_FLOW = max(1, round(CTRL_HZ / min(S.FLOW_HZ, CTRL_HZ)))        # 100->cap 50     -> 1
G_DTOF = max(1, round(CTRL_HZ / min(S.DOWN_TOF_HZ, CTRL_HZ)))    # 50              -> 1
G_BARO = max(1, round(CTRL_HZ / min(S.BARO_HZ, CTRL_HZ)))        # 25              -> 2
G_IMU = 1                                                        # ~100->cap 50    -> 1
G_YOLO = max(1, round(CTRL_HZ / 4.0))                            # YOLO detector 4 Hz (K1 227ms svc, util 0.91)
G_CTRL = 1                                                       # controller every control step
YOLO_CLASSES = ["gate", "person"]                                # nc=2 (static obstacles dropped; nav handles them)
YOLO_COLORS = {0: "#ffd400", 1: "#ff4b4b"}                       # gate=yellow, person=red
_TOF_VMAX = S.TOF_RANGE_MAX

# --- K1 XPU-RT schedule (for the embedded Gantt strip) ---
GANTT_KIND_COLOR = {"mlp_control": "#5aa469", "fused_full": "#9b6bd6",
                    "yolov8_nano": "#ff8a3d", "yolov8_nano_64x": "#ff8a3d", "yolov8_nano_128x": "#ff8a3d",
                    "yolov8_nano_64x96": "#ff8a3d",
                    "dronet": "#4b9fe0", "ffn_block": "#2ca6a4", "attn_block": "#1f7a78"}
GANTT_KIND_LABEL = {"mlp_control": "MLP ctrl", "fused_full": "nav (LSTM-conv)",
                    "yolov8_nano": "YOLOv8n", "yolov8_nano_64x": "YOLOv8n 64x96", "yolov8_nano_128x": "YOLOv8n 128x192",
                    "yolov8_nano_64x96": "YOLOv8n 64x96",
                    "dronet": "dronet", "ffn_block": "nav-transformer FFN", "attn_block": "nav-transformer attn"}
# IME MAC-unit dispatches (cluster-0 only) are drawn with this accent edge + hatch so the offload pops.
GANTT_IME_EDGE = "#e8402c"


def _kind_of_job(job_name):
    return job_name.rstrip("0123456789")


def load_schedule_gantt(path):
    """Parse an XPU-RT scheduled_*.json -> (lanes, bars, makespan).
    lanes: ordered list of core names; bars: list of (lane_idx, start_ms, dur_ms, kind, impl).
    `impl` is "ime" when the solver placed the dispatch on the cluster-0 MAC unit, else "rvv"/None."""
    import json as _json
    d = _json.load(open(path))
    md = d["metadata"]
    lanes = list(md["machines"])
    lane_idx = {m: i for i, m in enumerate(lanes)}
    makespan = float(md.get("makespan") or 0.0)
    bars = []
    for disp in d["dispatches"].values():
        # A sharded dispatch spans several harts: hardware_target is "+"-joined
        # (e.g. "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3"). Draw the bar on EVERY hart it
        # occupies so a 4-hart split shows as a wide block across 4 lanes.
        target = disp.get("hardware_target", "")
        st = float(disp.get("start_time", 0.0)); du = float(disp.get("duration", 0.0))
        kind = _kind_of_job(disp.get("job_name", "")); impl = disp.get("impl")
        for core in target.split("+"):
            if core in lane_idx:
                bars.append((lane_idx[core], st, max(du, 0.05), kind, impl))
        makespan = max(makespan, st + du)
    return lanes, bars, makespan


class Compositor:
    """Cached matplotlib-Agg figure: chase + full sensor bank + elongated overhead-with-path.
    Update handles per frame (gated by each sensor's Hz) → one RGB frame for imageio."""

    TRACE = 200

    def __init__(self, gantt=None):
        self._gantt = gantt              # (lanes, bars, makespan) or None
        _nrows = 4 if gantt else 3
        _h = 13.4 if gantt else 12
        self.fig = Figure(figsize=(18, _h), dpi=100)
        FigureCanvasAgg(self.fig)
        self.fig.patch.set_facecolor("white")
        _hr = [1.0, 1.0, 1.05, 0.5] if gantt else [1.0, 1.0, 1.05]
        gs = self.fig.add_gridspec(_nrows, 5, height_ratios=_hr,
                                   width_ratios=[1.25, 1.25, 1.2, 1.0, 1.0],
                                   hspace=0.55, wspace=0.38,
                                   left=0.015, right=0.982, top=0.94, bottom=0.04)

        # chase (top-left, 2x2 — a touch smaller than before; every frame). Status shown in-corner
        # (no big figure title).
        ax_chase = self.fig.add_subplot(gs[0:2, 0:2])
        ax_chase.axis("off")
        ax_chase.set_title("Chase cam — following the drone (red)", fontsize=17, weight="bold")
        self.im_chase = ax_chase.imshow(np.zeros((540, 960, 3), dtype=np.uint8))
        # K1 multi-rate scheduling indicator (bottom-left of chase): 3 lanes lit when each net fires.
        # Latencies are REAL, measured on the SpacemiT K1 (rvv_x60 int8; WALL_CYCLES @ 24 MHz rdtime).
        self._sched_lanes = [("CTRL", "100 Hz · 0.08 ms"), ("NAV", "50 Hz · 4.0 ms"),
                             ("YOLO", "5.4 Hz · 184 ms")]
        self.sched_txt = []
        ax_chase.text(0.015, 0.16, "K1 schedule (measured):", transform=ax_chase.transAxes, va="center",
                      fontsize=11, weight="bold", color="white",
                      bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))
        for i, (nm, hz) in enumerate(self._sched_lanes):
            t = ax_chase.text(0.015, 0.115 - i * 0.045, f"● {nm} · {hz}", transform=ax_chase.transAxes,
                              va="center", fontsize=12, weight="bold", color="#555",
                              bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))
            self.sched_txt.append(t)
        self.status = ax_chase.text(0.015, 0.985, "", transform=ax_chase.transAxes, va="top",
                                    ha="left", fontsize=15, weight="bold", color="white",
                                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.55, pad=0.4))

        # MLP-controller HUD (inset, bottom-right of chase): the 4-D [thrust, Mx, My, Mz] the MLP
        # emits each step and that IS applied to the motors (DirectThrustMoment). Shown only in
        # --controller rl, so the viewer sees the MLP genuinely closing the control loop.
        self.ax_mlp = ax_chase.inset_axes([0.70, 0.04, 0.28, 0.26])
        self.ax_mlp.set_facecolor((0, 0, 0, 0.5))
        self.ax_mlp.set_title("MLP ctrl → motors", fontsize=10, color="white", pad=3, weight="bold")
        self.ax_mlp.set_ylim(-1.08, 1.08)
        self.ax_mlp.set_xticks(range(4))
        self.ax_mlp.set_xticklabels(["T", "Mx", "My", "Mz"], fontsize=9, color="white")
        self.ax_mlp.tick_params(colors="white", labelsize=8)
        for s in self.ax_mlp.spines.values():
            s.set_color("white"); s.set_alpha(0.4)
        self.ax_mlp.axhline(0, color="white", lw=0.7, alpha=0.5)
        self.mlp_bars = self.ax_mlp.bar(range(4), [0, 0, 0, 0],
                                        color=["#5aa469", "tab:red", "tab:green", "tab:blue"])
        self.ax_mlp.set_visible(False)   # enabled per-frame only when snap carries an MLP action

        # FPV greyscale + Cross ToF STACKED in the middle column, both enlarged.
        ax_fpv = self.fig.add_subplot(gs[0, 2])
        ax_fpv.axis("off")
        ax_fpv.set_title("FPV · HM01B0 grey 60×90 · 10 Hz\n(model input + YOLOv8n boxes)", fontsize=14)
        self.ax_fpv = ax_fpv
        self.fpv_boxes = []          # dynamic YOLO box artists (Rectangle + label), cleared each update
        self.im_fpv = ax_fpv.imshow(np.zeros((60, 90)), cmap="gray", vmin=0.0, vmax=1.0,
                                    interpolation="nearest", aspect="auto")

        ax_tof = self.fig.add_subplot(gs[1, 2])
        ax_tof.set_title("Cross ToF · 4×VL53L5CX · 10 Hz\n(near = red, far = blue)", fontsize=14)
        ax_tof.set_xticks([]); ax_tof.set_yticks([]); ax_tof.set_facecolor("white")
        try:
            cmap = matplotlib.colormaps["turbo_r"].copy()
        except Exception:
            cmap = matplotlib.cm.get_cmap("turbo_r").copy()
        cmap.set_bad(color="white")                     # white background so the + cross is clear
        self.im_tof = ax_tof.imshow(np.full((24, 24), np.nan), cmap=cmap, vmin=0.0, vmax=_TOF_VMAX,
                                    interpolation="nearest", aspect="equal")
        for (r, c, lbl) in [(4, 12, "N"), (12, 4, "W"), (12, 20, "E"), (20, 12, "S")]:
            ax_tof.text(c, r, lbl, ha="center", va="center", color="black", fontsize=13, weight="bold")

        # right block: vector panels stacked in one column (flow / goal), time-series in the
        # other (altitude / IMU).
        _leg = dict(fontsize=11, columnspacing=0.7, handlelength=0.9, handletextpad=0.3,
                    borderaxespad=0.2, framealpha=0.7)
        ax_flow = self.fig.add_subplot(gs[0, 3])
        ax_flow.set_title("Optical flow · PMW3901 · 50 Hz", fontsize=14)
        ax_flow.set_xlim(-1.1, 1.1); ax_flow.set_ylim(-1.1, 1.1)
        ax_flow.set_xticks([]); ax_flow.set_yticks([]); ax_flow.set_aspect("equal")
        ax_flow.axhline(0, color="0.8", lw=0.8); ax_flow.axvline(0, color="0.8", lw=0.8)
        self.q_flow = ax_flow.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                     color="tab:red", width=0.03)
        self.flow_txt = ax_flow.text(0.03, 0.03, "", transform=ax_flow.transAxes, fontsize=12,
                                     va="bottom", ha="left", color="0.3")

        ax_goal = self.fig.add_subplot(gs[1, 3])            # goal UNDER optical flow
        ax_goal.set_title("Goal cmd → next gate\n(body frame · fwd = up)", fontsize=14)
        ax_goal.set_xlim(-1.1, 1.1); ax_goal.set_ylim(-1.1, 1.1)
        ax_goal.set_xticks([]); ax_goal.set_yticks([]); ax_goal.set_aspect("equal")
        ax_goal.axhline(0, color="0.8", lw=0.8); ax_goal.axvline(0, color="0.8", lw=0.8)
        self.q_goal = ax_goal.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                     color="tab:green", width=0.03)

        ax_alt = self.fig.add_subplot(gs[0, 4])
        ax_alt.set_title("Altitude (m) · 50 / 25 Hz", fontsize=14)
        ax_alt.grid(True, alpha=0.3); ax_alt.axhline(TARGET_H, color="0.6", lw=1.0, ls="--")
        self.l_dtof = Line2D([], [], color="tab:purple", lw=2.2, label="down-ToF")
        self.l_baro = Line2D([], [], color="tab:orange", lw=1.8, label="baro")
        for ln in (self.l_dtof, self.l_baro):
            ax_alt.add_line(ln)
        ax_alt.legend(loc="upper left", ncol=2, **_leg)
        ax_alt.set_xlabel("time (s)")
        self.ax_alt = ax_alt

        ax_imu = self.fig.add_subplot(gs[1, 4])
        ax_imu.set_title("IMU · ω (rad/s) · 50 Hz", fontsize=14)
        ax_imu.grid(True, alpha=0.3); ax_imu.axhline(0, color="k", lw=0.6, alpha=0.4)
        self.l_wx = Line2D([], [], color="tab:red", lw=1.8, label="ωx")
        self.l_wy = Line2D([], [], color="tab:green", lw=1.8, label="ωy")
        self.l_wz = Line2D([], [], color="tab:blue", lw=1.8, label="ωz")
        for ln in (self.l_wx, self.l_wy, self.l_wz):
            ax_imu.add_line(ln)
        ax_imu.legend(loc="upper left", ncol=3, **_leg)
        ax_imu.set_xlabel("time (s)")
        self.att_txt = ax_imu.text(0.98, 0.03, "", transform=ax_imu.transAxes, fontsize=12,
                                   va="bottom", ha="right", color="0.25")
        self.ax_imu = ax_imu

        # elongated overhead camera + flight path (bottom, full width)
        ax_top = self.fig.add_subplot(gs[2, 0:5])
        ax_top.set_title("Fixed overhead camera (aisle section) — flight path (red) · gates ○",
                         fontsize=16, weight="bold")
        ax_top.axis("off")
        self.im_top = ax_top.imshow(np.zeros((OV_H, OV_W, 3), dtype=np.uint8), aspect="auto")
        (self.l_path,) = ax_top.plot([], [], color="red", lw=3.0, alpha=0.95)
        (self.m_drone,) = ax_top.plot([], [], marker="o", color="red", ms=13, mec="white", mew=1.6)
        (self.m_gates,) = ax_top.plot([], [], linestyle="none", marker="o", ms=17, mfc="none",
                                      mec="yellow", mew=2.6)
        self.gate_labels = []
        self.ax_top = ax_top
        ax_top.set_xlim(0, OV_W); ax_top.set_ylim(OV_H, 0)

        # --- K1 XPU-RT schedule strip (full-width bottom): static schedule + red playhead + sensor arrows
        self.ax_gantt = None
        if gantt is not None:
            lanes, bars, makespan = gantt
            self._gantt_makespan = max(makespan, 1.0)
            axg = self.fig.add_subplot(gs[3, 0:5])
            self.ax_gantt = axg
            nlane = len(lanes)
            seen = set()
            ime_seen = False
            for (li, st, du, kind, impl) in bars:
                col = GANTT_KIND_COLOR.get(kind, "#888888")
                lbl = GANTT_KIND_LABEL.get(kind, kind) if kind not in seen else None
                seen.add(kind)
                if impl == "ime":       # cluster-0 MAC unit: accent edge + hatch so the offload pops
                    axg.add_patch(Rectangle((st, li + 0.12), du, 0.76, facecolor=col,
                                            edgecolor=GANTT_IME_EDGE, linewidth=1.3, hatch="///",
                                            label=lbl))
                    if not ime_seen:    # one legend proxy for the IME treatment
                        axg.add_patch(Rectangle((st, li + 0.12), 0, 0, facecolor="#dddddd",
                                                edgecolor=GANTT_IME_EDGE, linewidth=1.3, hatch="///",
                                                label="IME MAC (cluster-0)"))
                        ime_seen = True
                else:
                    axg.add_patch(Rectangle((st, li + 0.12), du, 0.76, facecolor=col,
                                            edgecolor="none", label=lbl))
            axg.set_xlim(0, self._gantt_makespan); axg.set_ylim(nlane, -0.6)
            axg.set_yticks([i + 0.5 for i in range(nlane)])
            axg.set_yticklabels(lanes, fontsize=9)
            _has_ime = any(len(b) > 4 and b[4] == "ime" for b in bars)
            _title = ("K1 XPU-RT schedule (real board profile · 8 harts"
                      + (" + IME MAC" if _has_ime else "") + " · CP-SAT deadline-aware) — "
                      "red playhead = HW position synced to sim · ▼ = sensor input")
            axg.set_title(_title, fontsize=13, weight="bold")
            axg.set_xlabel("schedule time (ms)", fontsize=11)
            axg.tick_params(axis="x", labelsize=9)
            axg.legend(loc="upper right", ncol=5, fontsize=9, framealpha=0.9, handlelength=1.0,
                       bbox_to_anchor=(1.0, 1.34))
            self.gantt_playhead = axg.axvline(0.0, color="red", lw=2.4, zorder=8)
            self.gantt_arrows = []          # dynamic sensor-input arrows for the current sweep
            self._gantt_wrap = -1

        self.reset_traces()

    def reset_traces(self):
        self.t_imu = deque(maxlen=self.TRACE); self.wx = deque(maxlen=self.TRACE)
        self.wy = deque(maxlen=self.TRACE); self.wz = deque(maxlen=self.TRACE)
        self.t_dtof = deque(maxlen=self.TRACE); self.dtof = deque(maxlen=self.TRACE)
        self.t_baro = deque(maxlen=self.TRACE); self.baro = deque(maxlen=self.TRACE)
        self.path_w = []                      # accumulated drone world positions (whole episode)
        for gl in self.gate_labels:
            gl.remove()
        self.gate_labels = []

    @staticmethod
    def _tof_plus(stack):
        comp = np.full((24, 24), np.nan)
        comp[0:8, 8:16] = stack[0]     # N
        comp[8:16, 0:8] = stack[3]     # W
        comp[8:16, 16:24] = stack[1]   # E
        comp[16:24, 8:16] = stack[2]   # S
        return comp

    def frame(self, fi, tsim, chase_rgb, top_rgb, ovK, ovpos, ovquat, gates_world, snap, banner,
              det=None, fires=None):
        self.im_chase.set_data(chase_rgb)
        # MLP-controller HUD: live [thrust, Mx, My, Mz] the MLP applied to the motors this step.
        _ma = snap.get("mlp_action")
        if _ma is not None:
            self.ax_mlp.set_visible(True)
            for _b, _v in zip(self.mlp_bars, _ma):
                _b.set_height(float(_v))
        if fi % G_FPV == 0:
            self.im_fpv.set_data(snap["fpv"])
        # YOLO boxes on the FPV (redraw on each YOLO tick; det = list of (cls, x0,y0,x1,y1, conf) in 60x90)
        if det is not None:
            for a in self.fpv_boxes:
                a.remove()
            self.fpv_boxes = []
            for cls, x0, y0, x1, y1, conf in det:
                col = YOLO_COLORS.get(int(cls), "#ffffff")
                r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.8)
                self.ax_fpv.add_patch(r); self.fpv_boxes.append(r)
                lab = self.ax_fpv.text(x0, y0 - 1.5, f"{YOLO_CLASSES[int(cls)]} {conf:.2f}",
                                       color=col, fontsize=8, weight="bold", va="bottom")
                self.fpv_boxes.append(lab)
        # scheduling indicator: light the lanes that fired this tick
        if fires is not None:
            lit = {"CTRL": "#39ff5a", "NAV": "#39ff5a", "YOLO": "#39ff5a"}
            for i, (nm, hz) in enumerate(self._sched_lanes):
                on = fires.get(nm, False)
                self.sched_txt[i].set_color(lit[nm] if on else "#556")
        # K1 schedule strip: advance the red playhead (synced to sim time) + drop a ▼ at each sensor input
        if self.ax_gantt is not None:
            sim_ms = tsim * 1000.0
            x = sim_ms % self._gantt_makespan
            wrap = int(sim_ms // self._gantt_makespan)
            self.gantt_playhead.set_xdata([x, x])
            if wrap != self._gantt_wrap:                 # new sweep through the schedule -> clear arrows
                for a in self.gantt_arrows:
                    a.remove()
                self.gantt_arrows = []
                self._gantt_wrap = wrap
            if fires is not None and fires.get("CAM", False):   # camera frame in (10 Hz) -> mark it
                ar = self.ax_gantt.annotate("", xy=(x, -0.02), xytext=(x, -0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#d81e5b", lw=1.8), zorder=9)
                self.gantt_arrows.append(ar)
        if fi % G_TOF == 0:
            self.im_tof.set_data(self._tof_plus(snap["tof"]))
        if fi % G_FLOW == 0:
            dx, dy = snap["flow"]
            n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            self.q_flow.set_UVC(np.array([dx / n * min(1.0, n / 300.0)]),
                                np.array([dy / n * min(1.0, n / 300.0)]))
            self.flow_txt.set_text(f"dx={dx:+.0f} dy={dy:+.0f}")
        # goal command (every frame)
        gx, gy = snap["goal"]
        gn = max(1e-6, (gx * gx + gy * gy) ** 0.5)
        self.q_goal.set_UVC(np.array([gy / gn]), np.array([gx / gn]))   # fwd(gx)→up, left(gy)→right-neg
        if fi % G_IMU == 0:
            self.t_imu.append(tsim)
            self.wx.append(snap["w"][0]); self.wy.append(snap["w"][1]); self.wz.append(snap["w"][2])
            ta = np.fromiter(self.t_imu, float)
            self.l_wx.set_data(ta, np.fromiter(self.wx, float))
            self.l_wy.set_data(ta, np.fromiter(self.wy, float))
            self.l_wz.set_data(ta, np.fromiter(self.wz, float))
            if len(ta) >= 2:
                self.ax_imu.set_xlim(ta[0], ta[-1]); self.ax_imu.relim()
                self.ax_imu.autoscale_view(scalex=False, scaley=True)
            r, p, y = snap["rpy"]
            self.att_txt.set_text(f"r{_math.degrees(r):+.0f}° p{_math.degrees(p):+.0f}° "
                                  f"y{_math.degrees(y):+.0f}°")
        if fi % G_DTOF == 0:
            self.t_dtof.append(tsim); self.dtof.append(snap["dtof"])
        if fi % G_BARO == 0:
            self.t_baro.append(tsim); self.baro.append(snap["baro"])
        if self.t_dtof:
            self.l_dtof.set_data(np.fromiter(self.t_dtof, float), np.fromiter(self.dtof, float))
        if self.t_baro:
            self.l_baro.set_data(np.fromiter(self.t_baro, float), np.fromiter(self.baro, float))
        if self.t_dtof:
            t0 = self.t_dtof[0]
            self.ax_alt.set_xlim(t0, max(t0 + 1e-3, tsim))
            allv = list(self.dtof) + list(self.baro)
            self.ax_alt.set_ylim(min(min(allv) - 0.2, TARGET_H - 0.3),
                                 max(max(allv) + 0.2, TARGET_H + 0.3))

        # overhead camera + projected path/gates/drone
        self.im_top.set_data(top_rgb)
        self.path_w.append(snap["pos_w"])
        pw = np.asarray(self.path_w, dtype=np.float64)
        pu, pv, pvalid = project(ovK, ovpos, ovquat, pw)
        self.l_path.set_data(pu, pv)
        self.m_drone.set_data([pu[-1]], [pv[-1]])
        gu, gv, gvalid = project(ovK, ovpos, ovquat, gates_world)
        self.m_gates.set_data(gu[gvalid], gv[gvalid])
        if not self.gate_labels:
            for gi in range(len(gates_world)):
                if gvalid[gi]:
                    self.gate_labels.append(self.ax_top.text(
                        gu[gi], gv[gi] - 20, f"G{gi + 1}", ha="center", va="bottom",
                        color="yellow", fontsize=15, weight="bold"))

        self.status.set_text(banner)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def main():
    env_cfg = WarehouseNavEnvCfg_PLAY_WithSensors_Coll()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.obstacle_count.params["min_level"] = args_cli.obstacle_level
    env_cfg.events.reset_obstacles.params["prop_density"] = args_cli.prop_density
    if args_cli.controller == "rl":
        # swap the classical velocity-command action for the RL/MLP controller's thrust/moment interface
        env_cfg.actions.velocity = DirectThrustMomentActionCfg(
            asset_name="robot", body_name="body", thrust_to_weight=1.9, moment_scale=args_cli.moment_scale)
    # Sim-rate overrides (Nyquist / step-size realism). Base: sim.dt=0.01 (100 Hz phys), decimation=2
    # -> 20 ms control (50 Hz) = the control/nav net period. decimation=1 -> 10 ms (100 Hz).
    if args_cli.sim_dt is not None:
        env_cfg.sim.dt = args_cli.sim_dt
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    env_cfg.sim.render_interval = env_cfg.decimation
    log(f"[sim] dt={env_cfg.sim.dt} decimation={env_cfg.decimation} -> control_dt="
        f"{env_cfg.sim.dt * env_cfg.decimation * 1000:.1f} ms ({1.0/(env_cfg.sim.dt*env_cfg.decimation):.0f} Hz)")
    env_cfg.episode_length_s = max(env_cfg.episode_length_s,
                                   args_cli.max_steps * float(env_cfg.sim.dt * env_cfg.decimation) + 1.0)
    env_cfg.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.7, 0.0, 0.0))
    # inject the fixed overhead camera (registered via the scene cfg instance __dict__)
    env_cfg.scene.overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCam", update_period=0.0,
        height=OV_H, width=OV_W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=S._focal_from_hfov(OV_HFOV),
                                         focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 200.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"))
    # inject an ISOMETRIC / axonometric 3D camera on the aisle (posed via set_world_poses_from_view
    # after reset) for the paper figure's 3D drone views.
    env_cfg.scene.iso_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/IsoCam", update_period=0.0,
        height=768, width=1024, data_types=["rgb"],
        # refresh the pose tensor on every buffer update — otherwise data.pos_w/quat stay stale
        # (0) after set_world_poses_from_view, which breaks the figure projection calibration.
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=S._focal_from_hfov(50.0),
                                         focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 250.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"))

    log(f"[env] gym.make {TASK_ID}")
    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    uenv.sim._disable_app_control_on_stop_handle = True
    dev = uenv.device
    N = uenv.num_envs
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    est = model = actor = yolo = None
    # CLEAN-BACKGROUND mode skips the nav/controller/detector nets entirely (no flight).
    if not args_cli.clean_overview:
        est = StateEstimator(N, dev, control_dt=control_dt)
        _sd = torch.load(args_cli.weights, map_location=dev, weights_only=True)
        _venc = "cnn" if any(k.startswith("vision_cnn.") for k in _sd) else "vit"
        # auto-detect LSTM capacity from the checkpoint (train_fused --lstm_hidden/--lstm_layers)
        _lh = int(_sd["lstm.weight_hh_l0"].shape[1]) if "lstm.weight_hh_l0" in _sd else 128
        _ll = sum(1 for k in _sd if k.startswith("lstm.weight_ih_l") and "_reverse" not in k) or 3
        model = FusedSensorNet(out_dim=2, vision_encoder=_venc, lstm_hidden=_lh, lstm_layers=_ll).to(dev).eval()
        model.load_state_dict(_sd, strict=True)
        log(f"[nav] FusedSensorNet out_dim=2 vision={_venc} loaded: {args_cli.weights}")

    if args_cli.controller == "rl" and not args_cli.clean_overview:
        actor = build_rl_actor(args_cli.rl_ckpt, dev)
        log(f"[ctrl] RL/MLP controller loaded: {args_cli.rl_ckpt} (moment_scale={args_cli.moment_scale})")

    if args_cli.yolo and not args_cli.clean_overview:
        from ultralytics import YOLO
        yolo = YOLO(args_cli.yolo)
        log(f"[yolo] warehouse detector loaded: {args_cli.yolo}")

    def run_yolo(grey):
        """grey: (H,W) float [0,1] at the DETECTION tap res (128x192 hi-res, from the 240x320 native).
        Predict at rect 128x192 (matches the K1 build); return boxes in 60x90 FPV-panel pixel coords."""
        H, W = grey.shape
        img = (np.clip(grey, 0, 1) * 255).astype(np.uint8)
        img3 = np.repeat(img[:, :, None], 3, axis=2)
        r = yolo.predict(img3, imgsz=[128, 192], conf=0.30, verbose=False)[0]   # rect (h,w) == K1 build; 0 grey-bar pad
        out = []
        for b in r.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            # scale from the detection-tap pixels to the 60x90 FPV panel so the compositor is unchanged
            sx, sy = 90.0 / W, 60.0 / H
            out.append((int(b.cls[0].item()), x0 * sx, y0 * sy, x1 * sx, y1 * sy, float(b.conf[0].item())))
        return out

    robot = uenv.scene["robot"]
    coll = uenv.scene["obstacles"]        # RigidObjectCollection: people + props + forklift
    origin = uenv.scene.env_origins
    chase = uenv.scene["chase_camera"]
    ov = uenv.scene["overview_camera"]
    iso = uenv.scene["iso_camera"]
    K = len(GATE_CENTERS_2D)
    o0 = origin[0].cpu().numpy()
    gates_world = np.concatenate([GATE_CENTERS_2D + o0[:2],
                                  np.full((K, 1), GATE_Z + o0[2])], axis=1)  # (K,3) world

    # place + freeze the overhead camera above the aisle section, looking straight down
    ov_eye = torch.tensor([[OV_CENTER_LOCAL[0] + o0[0], OV_CENTER_LOCAL[1] + o0[1],
                            o0[2] + OV_HEIGHT]], device=dev, dtype=torch.float32)
    ov_quat = torch.tensor([OV_QUAT], device=dev, dtype=torch.float32)
    ov.set_world_poses(ov_eye, ov_quat, convention="ros")

    # initial reset + hide the roof so the overhead camera sees the floor
    env.reset()
    ov.set_world_poses(ov_eye, ov_quat, convention="ros")
    nh = hide_roof(uenv.sim.stage)
    log(f"[demo] hid {nh} roof prims for the overhead view")
    ovK = ov.data.intrinsic_matrices[0].cpu().numpy()
    ovpos = ov.data.pos_w[0].cpu().numpy()
    ovquat = ov.data.quat_w_ros[0].cpu().numpy()  # (w,x,y,z) cam→world, ROS optical
    log(f"[demo] overhead K fx={ovK[0,0]:.1f} cx={ovK[0,2]:.1f}  pos={ovpos.round(2)}  quat={ovquat.round(3)}")

    # FIXED isometric overview camera: posed ONCE over the aisle centre, looking at it — it does
    # NOT follow the drone (the whole flight region stays framed for the paper figure).
    iso_center = torch.tensor([[ISO_CENTER_LOCAL[0] + o0[0], ISO_CENTER_LOCAL[1] + o0[1],
                                o0[2] + ISO_CENTER_LOCAL[2]]], device=dev, dtype=torch.float32)
    iso_eye = iso_center + torch.tensor([list(ISO_EYE_OFFSET)], device=dev, dtype=torch.float32)

    def _place_iso():
        iso.set_world_poses_from_view(iso_eye, iso_center)

    _place_iso()
    # NB: iso.data.pos_w/quat are stale until the first render tick, so the iso calibration is
    # captured lazily on the first figure frame (below), not here.
    iso_calib = {}
    log(f"[demo] FIXED iso cam eye={iso_eye[0].cpu().numpy().round(2)} -> "
        f"center={iso_center[0].cpu().numpy().round(2)} (offset {ISO_EYE_OFFSET})")

    # ── CLEAN-BACKGROUND mode (paper figure) ─────────────────────────────────────────────────
    # Obstacles/gates/racks are placed (env.reset above) and the roof is hidden. Now: hide the
    # drone, sweep a set of FIXED 3/4 axonometric iso poses down the aisle (preview PNG each), and
    # save drone-free iso + top-down backgrounds with their calibrations. No flight, no video.
    if args_cli.clean_overview:
        from pxr import Usd, UsdGeom
        import imageio.v2 as _iio

        def _grab(cam):
            rgb = cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
            return rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        def _render_settle(cam, n_step=4, n_render=30):
            # A few real sim steps flush fabric so the camera's pose TENSOR (data.pos_w/quat) syncs
            # to the freshly-set prim transform (render-only leaves it stale at 0); then render-only
            # passes let the RTX denoiser converge. Physics stepping only makes the (hidden) drone
            # fall — the static gates/racks and the placed props are unaffected visually.
            for _ in range(n_step):
                uenv.sim.step(render=True)
                cam.update(dt=control_dt, force_recompute=True)
            for _ in range(n_render):
                uenv.sim.render()
                cam.update(dt=control_dt, force_recompute=True)
            # force one explicit pose read now that the XformPrimView world transform is synced,
            # so data.pos_w / quat_w_ros reflect the pose we just set (not the stale 0 default).
            try:
                cam._update_poses(cam._ALL_INDICES)
            except Exception as _e:
                log(f"[clean] _update_poses fallback failed: {_e}")

        def _red_count(img):
            r = img[:, :, 0].astype(np.int32); g = img[:, :, 1].astype(np.int32); b = img[:, :, 2].astype(np.int32)
            return int(np.count_nonzero((r > 120) & (r - g > 50) & (r - b > 50)))

        # HIDE THE DRONE: make the whole robot subtree invisible so no red drone bakes into the bg.
        stage = uenv.sim.stage
        hid_robot = []
        for cand in (f"/World/envs/env_0/Robot", f"/World/envs/env_0/robot"):
            pr = stage.GetPrimAtPath(cand)
            if pr and pr.IsValid():
                UsdGeom.Imageable(pr).MakeInvisible(); hid_robot.append(cand)
        if not hid_robot:                          # fallback: any prim named Robot under the env
            for pr in stage.Traverse():
                if pr.GetName() in ("Robot", "robot") and "/envs/env_" in str(pr.GetPath()):
                    UsdGeom.Imageable(pr).MakeInvisible(); hid_robot.append(str(pr.GetPath()))
        log(f"[clean] hid drone prim(s): {hid_robot}")

        outd = args_cli.clean_out
        os.makedirs(outd, exist_ok=True)

        # 1) CLEAN top-down overhead (already posed straight down). Capture calibration LAZILY.
        _render_settle(ov)
        ov_bg = _grab(ov)
        ovK = ov.data.intrinsic_matrices[0].cpu().numpy()
        ovpos = ov.data.pos_w[0].cpu().numpy()
        ovquat = ov.data.quat_w_ros[0].cpu().numpy()
        _iio.imwrite(os.path.join(outd, "ov_clean.png"), ov_bg)
        log(f"[clean] overhead bg {ov_bg.shape} red_px={_red_count(ov_bg)} "
            f"pos={ovpos.round(2)} -> {outd}/ov_clean.png")

        # 2) Sweep FIXED 3/4 axonometric iso poses DOWN THE AISLE (env-local eye,target metres).
        # Aisle centre x=-8, corridor x in [-9.96,-6.0]; rack rows flank at x=-10.5 (W) & x=-5.5 (E);
        # mouth y~6, gates y=9,13,17,21, far end y~24. Views look NORTH (+y) and down, framing all
        # gates + both rack rows + the aisle floor where the flight path will be drawn.
        CANDIDATES = [
            # name              eye (x, y, z)            target (x, y, z)
            ("A_eastmouth",   (-6.3,  2.0, 5.8),   (-8.0, 16.0, 2.0)),   # inside aisle, east-offset, shallow
            ("B_westmouth",   (-9.6,  2.0, 5.8),   (-8.0, 16.0, 2.0)),   # inside aisle, west-offset, shallow
            ("C_centrehigh",  (-8.0,  0.5, 9.0),   (-8.0, 15.0, 1.5)),   # aisle centre, high, steeper
            ("D_eastout",     (-3.5,  2.0, 9.5),   (-8.0, 15.0, 2.0)),   # E of E-rack, high to clear it
            ("E_eastlow",     (-6.2,  3.5, 4.8),   (-8.2, 17.5, 2.2)),   # inside, low & shallow, close
            ("F_westout",     (-12.8, 2.0, 9.5),   (-8.0, 15.0, 2.0)),   # W of W-rack, high to clear it
            ("G_eastfarsouth",(-6.7, -1.5, 7.0),   (-8.0, 17.0, 2.0)),   # far south for full recession
            ("H_task",        (-3.5,  4.0, 7.0),   (-8.0, 14.0, 2.5)),   # the task's suggested start
        ]
        sweep = []  # (name, eye, tgt, iso_bg, K, pos, quat, red)
        for name, eye_l, tgt_l in CANDIDATES:
            eye_w = torch.tensor([[eye_l[0] + o0[0], eye_l[1] + o0[1], eye_l[2] + o0[2]]],
                                 device=dev, dtype=torch.float32)
            tgt_w = torch.tensor([[tgt_l[0] + o0[0], tgt_l[1] + o0[1], tgt_l[2] + o0[2]]],
                                 device=dev, dtype=torch.float32)
            iso.set_world_poses_from_view(eye_w, tgt_w)
            _render_settle(iso)
            img = _grab(iso)
            K3 = iso.data.intrinsic_matrices[0].cpu().numpy()
            pos = iso.data.pos_w[0].cpu().numpy()
            quat = iso.data.quat_w_ros[0].cpu().numpy()
            red = _red_count(img)
            _iio.imwrite(os.path.join(outd, f"iso_cand_{name}.png"), img)
            sweep.append((name, np.asarray(eye_l, np.float32), np.asarray(tgt_l, np.float32),
                          img, K3, pos, quat, red))
            log(f"[clean] cand {name:14s} eye_local={eye_l} tgt_local={tgt_l} "
                f"red_px={red} pos={pos.round(2)} -> {outd}/iso_cand_{name}.png")

        # 3) save the whole sweep so the final pose can be rebuilt offline (no relaunch).
        np.savez_compressed(
            os.path.join(outd, "clean_sweep.npz"),
            names=np.asarray([s[0] for s in sweep]),
            eyes_local=np.stack([s[1] for s in sweep]),
            tgts_local=np.stack([s[2] for s in sweep]),
            iso_bgs=np.stack([s[3] for s in sweep]),
            isoKs=np.stack([s[4] for s in sweep]),
            isopos=np.stack([s[5] for s in sweep]),
            isoquat=np.stack([s[6] for s in sweep]),
            red=np.asarray([s[7] for s in sweep], np.int64),
            ov_bg=ov_bg, ovK=ovK, ovpos=ovpos, ovquat=ovquat,
            gates_world=gates_world, o0=o0,
        )
        log(f"[clean] wrote {outd}/clean_sweep.npz ({len(sweep)} candidates)")

        # 4) write clean_bg.npz for the CHOSEN candidate (default = first).
        idx = 0
        if args_cli.iso_choice:
            for i, s in enumerate(sweep):
                if s[0] == args_cli.iso_choice:
                    idx = i; break
            else:
                log(f"[clean] WARN: iso_choice '{args_cli.iso_choice}' not found; using {sweep[0][0]}")
        ch = sweep[idx]
        np.savez_compressed(
            os.path.join(outd, "clean_bg.npz"),
            iso_bg=ch[3], isoK=ch[4], isopos=ch[5], isoquat=ch[6],
            ov_bg=ov_bg, ovK=ovK, ovpos=ovpos, ovquat=ovquat,
            iso_eye_local=ch[1], iso_target_local=ch[2], iso_name=np.asarray(ch[0]),
            gates_world=gates_world,
        )
        _iio.imwrite(os.path.join(outd, "iso_clean.png"), ch[3])
        log(f"[clean] CHOSEN '{ch[0]}' eye_local={tuple(ch[1])} tgt_local={tuple(ch[2])} "
            f"iso_red_px={ch[7]} ov_red_px={_red_count(ov_bg)}")
        log(f"[clean] chosen isopos={ch[5].round(3)} isoquat={ch[6].round(4)}")
        # sanity-check the calibration: project each gate world point through isoK/isopos/isoquat
        # and confirm the pixel lands inside the frame (should sit on the yellow gate frame).
        H_iso, W_iso = ch[3].shape[:2]
        gu, gv, gvalid = project(ch[4], ch[5], ch[6], gates_world)
        for gi in range(len(gates_world)):
            inframe = bool(gvalid[gi] and 0 <= gu[gi] < W_iso and 0 <= gv[gi] < H_iso)
            log(f"[clean] gate{gi+1} world={gates_world[gi].round(2)} -> iso px=({gu[gi]:.0f},{gv[gi]:.0f}) "
                f"{'IN-FRAME' if inframe else 'off-frame'} (img {W_iso}x{H_iso})")
        log(f"[clean] wrote {outd}/clean_bg.npz + iso_clean.png + ov_clean.png")
        env.close()
        return

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
        inp = {"front_grey": grey.float(), "tof_cross": tof_norm, "optical_flow": flow,
               "down_tof": dtof_norm, "baro": baro / 10.0, "quat": filt["quat"], "body_rates": gyro,
               "desired_vel": desired_vel,
               "flags": torch.cat([flow_valid, dtof_valid, torch.ones(N, 4, device=dev)], dim=1)}
        return inp, filt

    def _drive_chase():
        p = robot.data.root_pos_w[0]; q = robot.data.root_quat_w[0]
        w, x, y, z = q[0], q[1], q[2], q[3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fx, fy = torch.cos(yaw), torch.sin(yaw)
        eye = torch.stack([p[0] - fx * 2.3, p[1] - fy * 2.3, p[2] + 1.25])
        tgt = torch.stack([p[0] + fx * 3.0, p[1] + fy * 3.0, p[2] - 0.05])
        chase.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))
        # NB: the iso overview camera is FIXED (placed once via _place_iso) — it is intentionally
        # NOT re-posed here, so it keeps the whole flight region framed for the paper figure.

    def _rgb(cam):
        rgb = cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
        return rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    gantt = None
    if args_cli.gantt_schedule and os.path.exists(args_cli.gantt_schedule):
        try:
            gantt = load_schedule_gantt(args_cli.gantt_schedule)
            log(f"[gantt] embedded XPU-RT schedule: {args_cli.gantt_schedule} "
                f"({len(gantt[1])} dispatches, makespan {gantt[2]:.1f} ms)")
        except Exception as e:
            log(f"[gantt] failed to load schedule ({e}); strip disabled")
    comp = Compositor(gantt=gantt)
    final = args_cli.save_video
    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
    best_tmp, best_prog, captured = None, -1.0, False
    _figdata = {"saved": False}   # stroboscopic-figure capture (Workstream G)

    for ep in range(args_cli.episodes):
        torch.manual_seed(args_cli.seed + ep)
        if hasattr(est, "reset"):
            est.reset()
        env.reset()
        ov.set_world_poses(ov_eye, ov_quat, convention="ros")
        _place_iso()                       # re-freeze the fixed iso overview after the reset
        comp.reset_traces()
        hidden = None
        last_action = torch.zeros((N, 4), device=dev, dtype=torch.float32)
        last_safety_dets = []          # most-recent YOLO detections (held between 4 Hz ticks)
        safety_tele = None
        tmp = f"{final}.ep{ep:02d}.tmp.mp4"
        writer = imageio.get_writer(tmp, fps=args_cli.fps, codec="libx264", quality=8,
                                    macro_block_size=None)
        goal_idx, gates_passed, outcome = 0, 0, "timeout"
        t = 0
        # ENRICHED figure capture: FULL-run per-step arrays + dense per-moment frames.
        _figep = ({"poses": [], "t_s": [], "obst_pos": [], "goal_cmd": [], "imu_w": [],
                   "alt_dtof": [], "alt_baro": [],
                   "dense_chase": [], "dense_fpv": [], "dense_tof": [], "dense_det": [],
                   "frame_steps": [], "iso_frames": [],
                   "ov_bg": None, "iso_bg": None} if args_cli.dump_figure_data else None)
        last_det = []   # freshest YOLO detections, held between the sparse figure snapshots
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

            _drive_chase()
            inp, filt = sense(desired_vel)
            with torch.no_grad():
                cmd, hidden = model(inp, hidden)
            if args_cli.controller == "rl":
                yaw_rate = float(cmd[0, 0].item()) * args_cli.yaw_scale
                fwd = args_cli.cruise_speed
            else:
                yaw_rate = float(cmd[0, 0].item())
                fwd = args_cli.fixed_speed if args_cli.fixed_speed > 0 else float(max(0.1, min(MAX_SPEED, cmd[0, 1].item())))

            grey60 = F.interpolate(inp["front_grey"], size=(60, 90), mode="bilinear",
                                   align_corners=False)[0, 0].cpu().numpy()
            # hi-res detection tap (128x192) from the 240x320 native — real detail, matches the K1 yolo build
            grey128 = F.interpolate(inp["front_grey"], size=(128, 192), mode="bilinear",
                                    align_corners=False)[0, 0].cpu().numpy()
            snap = {
                "fpv": np.clip(grey60, 0.0, 1.0),
                "tof": S.tof_stack(uenv)[0].cpu().numpy(),
                "flow": inp["optical_flow"][0].cpu().numpy(),
                "goal": (float(bx / nrm), float(by / nrm)),
                "w": robot.data.root_ang_vel_b[0, :3].cpu().numpy(),
                "rpy": _quat_to_rpy(filt["quat"][0].cpu().numpy()),
                "dtof": float(S.down_tof(uenv)[0, 0].item()),
                "baro": float(S.barometer(uenv)[0, 1].item()),
                "pos_w": robot.data.root_pos_w[0].cpu().numpy(),
                # MLP controller's applied 4-D output [thrust, Mx, My, Mz] (prev step; one-frame lag).
                # Non-None only in --controller rl, where this vector IS the motor command.
                "mlp_action": (last_action[0].detach().cpu().numpy()
                               if args_cli.controller == "rl" else None),
            }
            det = None
            yolo_fire = False
            if yolo is not None and t % G_YOLO == 0:
                det = run_yolo(grey128); yolo_fire = True
                last_det = det
                # hold the freshest detections for the safety layer (px 90x60 -> normalized xywh)
                last_safety_dets = [{"cls": c, "xywh": ((x0 + x1) / 2 / 90.0, (y0 + y1) / 2 / 60.0,
                                                        (x1 - x0) / 90.0, (y1 - y0) / 60.0), "conf": cf}
                                    for (c, x0, y0, x1, y1, cf) in det]
            # USE YOLO: route detections through the safety layer to modulate the command
            if args_cli.safety:
                (yaw_rate, fwd), safety_tele = apply_safety((yaw_rate, fwd), last_safety_dets)
            _cn = "nav LSTM-conv + MLP ctrl + YOLOv8n" if args_cli.controller == "rl" else "v12 CNN nav + YOLOv8n"
            _sfx = ""
            if args_cli.safety and safety_tele and safety_tele.get("trigger"):
                _sfx = f"  ·  SAFETY: {safety_tele['trigger']} → x{safety_tele['speed_scale']:.2f}"
            banner = f"{_cn} · K1 multi-rate\ngate {min(gates_passed, K)}/{K} · t={t * control_dt:4.1f}s{_sfx}"
            fires = {"CTRL": (t % G_CTRL == 0), "NAV": True, "YOLO": yolo_fire,
                     "CAM": (t % G_FPV == 0)}
            writer.append_data(comp.frame(t, t * control_dt, _rgb(chase), _rgb(ov),
                                          ovK, ovpos, ovquat, gates_world, snap, banner,
                                          det=det, fires=fires))
            if _figep is not None:
                if _figep["ov_bg"] is None:            # clean overhead + iso background (frame 0)
                    _figep["ov_bg"] = _rgb(ov)
                    _figep["iso_bg"] = _rgb(iso)
                    # iso calibration is now valid (camera has rendered) — capture it once
                    iso_calib["K"] = iso.data.intrinsic_matrices[0].cpu().numpy()
                    iso_calib["pos"] = iso.data.pos_w[0].cpu().numpy()
                    iso_calib["quat"] = iso.data.quat_w_ros[0].cpu().numpy()
                # --- FULL-run per-step arrays (every control step) ---
                _figep["poses"].append(np.concatenate([snap["pos_w"],
                                       robot.data.root_quat_w[0].cpu().numpy()]))   # x,y,z,qw,qx,qy,qz
                _figep["t_s"].append(t * control_dt)
                _figep["obst_pos"].append(coll.data.object_pos_w[0].cpu().numpy())  # (M,3) world
                _figep["goal_cmd"].append(desired_vel[0].cpu().numpy())            # (3,) goal→next-gate
                _figep["imu_w"].append(np.asarray(snap["w"], dtype=np.float32))    # (3,) body ang-vel
                _figep["alt_dtof"].append(np.float32(snap["dtof"]))
                _figep["alt_baro"].append(np.float32(snap["baro"]))
                # --- dense per-moment frames (every FIG_DENSE steps) for post-hoc moment selection ---
                if t % FIG_DENSE == 0:
                    # run YOLO fresh so the boxes match THIS fpv frame (cls,x0,y0,x1,y1,conf in 90×60)
                    dgrey = run_yolo(grey128) if yolo is not None else []
                    _figep["dense_chase"].append(_rgb(chase))
                    _figep["dense_fpv"].append(snap["fpv"].copy())               # (60,90) float
                    _figep["dense_tof"].append(snap["tof"].copy())               # (4,8,8) float m
                    _figep["dense_det"].append(np.asarray(dgrey, dtype=np.float32).reshape(-1, 6))
                    _figep["frame_steps"].append(t)
                    _figep["iso_frames"].append(_rgb(iso))                       # fixed iso, drone in it

            if args_cli.controller == "rl":
                steer_cmd = torch.tensor([[yaw_rate, fwd]], device=dev, dtype=torch.float32)
                rl_obs = torch.cat([robot.data.root_lin_vel_b[:, :3], robot.data.root_ang_vel_b[:, :3],
                                    robot.data.projected_gravity_b, (robot.data.root_pos_w - origin)[:, 2:3],
                                    steer_cmd, last_action], dim=1)
                with torch.no_grad():
                    action = actor(rl_obs).clamp(-1.0, 1.0)
                last_action = action.detach()
                obs, _r, dones, _i = env.step(action)
            else:
                obs, _r, dones, _i = env.step(cmd_to_action(yaw_rate, fwd, h_now, dev, N))
            last_h = float(robot.data.root_pos_w[0, 2].item())
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
                elif any("collision" in nm and v for nm, v in terms.items()) or last_h < 0.2:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                break
        else:
            outcome = "success" if gates_passed >= K else "timeout"
        writer.close()
        prog = gates_passed / K
        log(f"[ep{ep:02d}] outcome={outcome:9s} gates={gates_passed}/{K} steps={t + 1}")

        if outcome == "success":
            os.replace(tmp, final); captured = True
            if _figep is not None:
                _figdata["ep"] = _figep
            log(f"[demo] captured successful ep{ep:02d} ({gates_passed}/{K}) → {final}")
            break
        if prog > best_prog:
            if best_tmp and os.path.exists(best_tmp):
                os.remove(best_tmp)
            best_tmp, best_prog = tmp, prog
            if _figep is not None:
                _figdata["ep"] = _figep
        else:
            os.remove(tmp)

    if not captured:
        if best_tmp and os.path.exists(best_tmp):
            os.replace(best_tmp, final)
            log(f"[demo] no full success in {args_cli.episodes} eps; kept deepest run "
                f"(progress={best_prog:.2f}) → {final}")
        else:
            log("[demo] nothing recorded")
    log(f"[out] wrote {final}")

    # ENRICHED paper-figure data: a SINGLE figure_data.npz (full-run per-step arrays + fixed
    # overhead & iso calibrations/backgrounds + gates + dense per-moment frames) plus one .npz
    # per dense moment. Captured for the KEPT (best 4/4, else deepest) episode.
    if args_cli.dump_figure_data and _figdata.get("ep"):
        fe = _figdata["ep"]
        d = args_cli.dump_figure_data
        os.makedirs(d, exist_ok=True)

        # obstacle kind + person mask (fixed for the run), derived from the collection object names
        onames = list(coll.object_names)

        def _kind_of(nm):
            if nm.startswith("person_"):
                return "person"
            if nm.startswith("forklift"):
                return "forklift"
            return nm.rsplit("_", 1)[0]        # "crate_7" -> "crate"

        obst_kind = np.asarray([_kind_of(nm) for nm in onames])            # (M,) str
        person_mask = np.asarray([nm.startswith("person_") for nm in onames], dtype=bool)

        # iso calibration fallback (should already be captured on the kept episode's frame 0)
        if not iso_calib:
            iso_calib["K"] = iso.data.intrinsic_matrices[0].cpu().numpy()
            iso_calib["pos"] = iso.data.pos_w[0].cpu().numpy()
            iso_calib["quat"] = iso.data.quat_w_ros[0].cpu().numpy()

        # iso_over: one mid-flight iso frame with the drone in it (middle of the dense captures)
        iso_frames = fe["iso_frames"]
        iso_over = iso_frames[len(iso_frames) // 2] if iso_frames else fe["iso_bg"]

        # per-moment YOLO detections as a ragged object array (each (k,6): cls,x0,y0,x1,y1,conf)
        det_obj = np.empty(len(fe["dense_det"]), dtype=object)
        for i, dd in enumerate(fe["dense_det"]):
            det_obj[i] = dd

        np.savez_compressed(
            os.path.join(d, "figure_data.npz"),
            # --- full-run per-step (T,...) ---
            poses=np.asarray(fe["poses"], dtype=np.float64),               # (T,7) x,y,z,qw,qx,qy,qz
            t_s=np.asarray(fe["t_s"], dtype=np.float64),                   # (T,) seconds
            obst_pos=np.asarray(fe["obst_pos"], dtype=np.float32),         # (T,M,3) world
            obst_kind=obst_kind,                                          # (M,) str
            person_mask=person_mask,                                      # (M,) bool
            goal_cmd=np.asarray(fe["goal_cmd"], dtype=np.float32),         # (T,3) goal→next-gate cmd
            imu_w=np.asarray(fe["imu_w"], dtype=np.float32),               # (T,3) body ang-vel
            alt_dtof=np.asarray(fe["alt_dtof"], dtype=np.float32),         # (T,)
            alt_baro=np.asarray(fe["alt_baro"], dtype=np.float32),         # (T,)
            gates_world=gates_world,                                       # (G,3)
            # --- fixed overhead (top-down) camera ---
            ov_bg=fe["ov_bg"], ovK=ovK, ovpos=ovpos, ovquat=ovquat,
            # --- fixed isometric overview camera ---
            iso_bg=fe["iso_bg"], iso_over=iso_over,
            isoK=iso_calib["K"], isopos=iso_calib["pos"], isoquat=iso_calib["quat"],
            # --- dense per-moment frames (n,...) for post-hoc moment selection ---
            chase=np.asarray(fe["dense_chase"], dtype=np.uint8),           # (n,Hc,Wc,3)
            fpv=np.asarray(fe["dense_fpv"], dtype=np.float32),             # (n,60,90)
            tof=np.asarray(fe["dense_tof"], dtype=np.float32),             # (n,4,8,8)
            det=det_obj,                                                  # (n,) object -> (k,6)
            frame_steps=np.asarray(fe["frame_steps"], dtype=np.int64),     # (n,)
        )

        # per-moment frame files (convenience: one .npz per dense moment)
        fdir = os.path.join(d, "frames")
        os.makedirs(fdir, exist_ok=True)
        for i, st in enumerate(fe["frame_steps"]):
            np.savez(os.path.join(fdir, f"frame_{i:03d}.npz"),
                     step=np.int64(st), t_s=np.float64(st * control_dt),
                     chase=fe["dense_chase"][i], fpv=fe["dense_fpv"][i],
                     tof=fe["dense_tof"][i], det=fe["dense_det"][i])
        log(f"[figure] wrote {d}/figure_data.npz + {len(fe['frame_steps'])} frame files  "
            f"(T={len(fe['poses'])} steps, M={len(onames)} obstacles, "
            f"{len(fe['frame_steps'])} dense moments)")


if __name__ == "__main__":
    main()
    os._exit(0)
