# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Onboard sensor rig for the forest-trail Crazyflie (Workstream S).

Simulates the *real* sensors carried by the physical trail-following drone so
downstream perception models consume sensor-accurate inputs:

1. **Himax HM01B0 front camera** — monochrome QVGA (320x240 native), ~87°
   horizontal FoV, mounted facing forward (+x, "N"), co-aligned with the drone
   body. Rendered as RGB in sim; consumers convert to 1-channel luma via
   :func:`front_greyscale` (luma = 0.299R + 0.587G + 0.114B).

2. **4x ST VL53L5CX multizone ToF lidars** in a cross formation N / E / S / W
   (one per cardinal direction, 90° apart, rigidly parented to the body). Each
   is an 8x8-zone sensor with a 63° *diagonal* FoV and a 0.02-4.0 m range. In
   sim each is a tiny 8x8 DEPTH camera (``distance_to_camera``) yaw-rotated to
   its cardinal heading. The forward (N) ToF is co-registered with the front
   camera (same body offset, both looking down +x).

The camera cfgs reuse the ``CameraCfg`` + ``CameraCfg.OffsetCfg(..., convention
="ros")`` idiom from ``track_steering_vision`` / ``warehouse_nav`` so they parent
to ``{ENV_REGEX_NS}/Robot/body`` exactly like the existing ``fpv_camera``.

Runtime helpers read the attached sensors into consumable tensors:

- :func:`tof_stack`            -> (num_envs, 4, 8, 8)  ordered [N, E, S, W]
- :func:`tof_cross_composite`  -> (num_envs, 24, 24)   4 patches in a + layout
- :func:`front_greyscale`      -> (num_envs, 1, H, W)  luma from the front cam
- :func:`add_tof_noise`        -> VL53L5CX-style multiplicative range noise +
                                  optional far-zone dropout (opt-in)
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg


# ── Sensor geometry constants (from the datasheets) ───────────────────────────

# Himax HM01B0: monochrome, QVGA native, ~87° horizontal FoV. Downstream models
# resize to 112x112, so QVGA is plenty of resolution and keeps rendering cheap.
FRONT_CAM_WIDTH = 320
FRONT_CAM_HEIGHT = 240
FRONT_CAM_HFOV_DEG = 87.0
FRONT_CAM_CLIP = (0.05, 50.0)

# ST VL53L5CX: 8x8 zones, 63° *diagonal* FoV, 2 cm - 4 m range.
TOF_ZONES = 8                       # 8x8 grid
TOF_DIAG_FOV_DEG = 63.0
TOF_RANGE_MIN = 0.02
TOF_RANGE_MAX = 4.0
TOF_CLIP = (TOF_RANGE_MIN, TOF_RANGE_MAX)

# Standard 35 mm-equivalent aperture used across the repo's PinholeCameraCfgs.
_H_APERTURE = 20.955

# Body offset shared by the front camera and the N ToF (co-registered), mirrors
# the existing fpv_camera mount just ahead of the body origin.
_FRONT_OFFSET_POS = (0.06, 0.0, 0.01)

# ROS-convention quaternion (w, x, y, z) for a camera looking straight down the
# body +x axis. This is the exact rot used by every fpv_camera in the repo.
_FORWARD_ROS_QUAT = (0.5, -0.5, 0.5, -0.5)

# Cardinal ToF headings as a yaw (rad) about the body +z axis, applied on top of
# the forward-looking base quaternion. +x = forward = N, +z = up, so +y = left.
#   N =   0°  (+x, forward)      E = -90°  (-y, right)
#   S = 180°  (-x, behind)       W = +90°  (+y, left)
_TOF_YAW_DEG = {"N": 0.0, "E": -90.0, "S": 180.0, "W": 90.0}

# Fixed order used by tof_stack / tof_cross_composite. Do not reorder — several
# consumers index into the stack positionally.
TOF_ORDER = ("N", "E", "S", "W")

# Scene attribute names of the five sensor prims (used by the runtime helpers).
FRONT_CAM_KEY = "front_camera"
TOF_KEYS = {d: f"tof_{d.lower()}" for d in TOF_ORDER}  # {"N": "tof_n", ...}


# ── FoV / aperture math ───────────────────────────────────────────────────────

def _focal_from_hfov(hfov_deg: float, h_aperture: float = _H_APERTURE) -> float:
    """Focal length (mm) giving a horizontal FoV of ``hfov_deg`` for an aperture.

    HFOV = 2 * atan(aperture / (2 * focal))  ->  focal = aperture / (2 tan(HFOV/2)).
    """
    return h_aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def _focal_from_diag_fov(
    diag_fov_deg: float, h_aperture: float = _H_APERTURE, aspect: float = 1.0
) -> float:
    """Focal length (mm) giving a *diagonal* FoV of ``diag_fov_deg``.

    For a sensor with square pixels the vertical aperture is ``h_aperture *
    aspect`` (aspect = height/width = 1 for the square 8x8 ToF), so the diagonal
    aperture is ``sqrt(h^2 + v^2)`` and
        diag_FOV = 2 * atan(diag_aperture / (2 * focal)).
    """
    v_aperture = h_aperture * aspect
    diag_aperture = math.hypot(h_aperture, v_aperture)
    return diag_aperture / (2.0 * math.tan(math.radians(diag_fov_deg) / 2.0))


def _quat_mul(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _yaw_rotate_forward_quat(yaw_deg: float):
    """ROS-convention quat for the forward camera yawed ``yaw_deg`` about body +z.

    Pre-multiplying the forward-looking base quaternion by a yaw rotation about
    the body's +z rotates the whole optical frame in the horizontal plane, so
    the ToF's boresight points along the requested cardinal direction while
    keeping "up" up.
    """
    half = math.radians(yaw_deg) / 2.0
    q_yaw = (math.cos(half), 0.0, 0.0, math.sin(half))
    return _quat_mul(q_yaw, _FORWARD_ROS_QUAT)


# ── Camera-cfg factories ──────────────────────────────────────────────────────

def front_greyscale_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/body/front_cam",
    update_period: float = 0.1,
) -> CameraCfg:
    """Front Himax HM01B0 camera cfg (mono QVGA, ~87° HFOV, facing N/+x).

    Rendered as RGB; use :func:`front_greyscale` at the consumer to get the
    1-channel luma the physical mono sensor produces. Also renders an aligned
    ``distance_to_camera`` depth annotator — this is a TRAINING-ONLY supervision
    signal for the monocular-depth auxiliary task (:func:`front_depth`); the
    physical drone has no depth cam, and the deployed net never reads it.
    """
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=FRONT_CAM_HEIGHT,
        width=FRONT_CAM_WIDTH,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal_from_hfov(FRONT_CAM_HFOV_DEG),
            focus_distance=400.0,
            horizontal_aperture=_H_APERTURE,
            clipping_range=FRONT_CAM_CLIP,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_FRONT_OFFSET_POS,
            rot=_FORWARD_ROS_QUAT,
            convention="ros",
        ),
    )


def chase_camera_cfg(prim_path: str = "{ENV_REGEX_NS}/ChaseCam",
                     width: int = 640, height: int = 400) -> CameraCfg:
    """Third-person 'chase' camera for demo videos — world-anchored (NOT parented to
    the drone), repositioned each step by the eval loop to sit behind+above the drone
    and look forward down the course, so the drone + gates + trees are all in frame.
    """
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,          # render every step while recording
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal_from_hfov(70.0),
            focus_distance=400.0,
            horizontal_aperture=_H_APERTURE,
            clipping_range=(0.05, 200.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


def tof_camera_cfg(
    direction: str,
    prim_path: str | None = None,
    update_period: float = 0.1,
) -> CameraCfg:
    """One VL53L5CX ToF as an 8x8 depth camera facing ``direction`` (N/E/S/W).

    Depth is exposed via ``distance_to_camera`` and clipped to the sensor's
    0.02-4.0 m range. The N ToF shares the front camera's body offset so the two
    forward sensors are co-registered.
    """
    direction = direction.upper()
    if direction not in _TOF_YAW_DEG:
        raise ValueError(f"tof_camera_cfg: direction must be one of {TOF_ORDER}, got {direction!r}")
    if prim_path is None:
        prim_path = "{ENV_REGEX_NS}/Robot/body/tof_" + direction.lower()

    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=TOF_ZONES,
        width=TOF_ZONES,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            # Square sensor -> aspect 1.0; focal chosen for a 63° diagonal FoV.
            focal_length=_focal_from_diag_fov(TOF_DIAG_FOV_DEG, aspect=1.0),
            focus_distance=400.0,
            horizontal_aperture=_H_APERTURE,
            clipping_range=TOF_CLIP,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_FRONT_OFFSET_POS,  # cross rigidly parented to the body origin
            rot=_yaw_rotate_forward_quat(_TOF_YAW_DEG[direction]),
            convention="ros",
        ),
    )


def build_sensor_cfgs() -> dict[str, CameraCfg]:
    """All five sensor cfgs keyed by their scene attribute name.

    Keys: ``front_camera``, ``tof_n``, ``tof_e``, ``tof_s``, ``tof_w``.
    """
    cfgs: dict[str, CameraCfg] = {FRONT_CAM_KEY: front_greyscale_camera_cfg()}
    for d in TOF_ORDER:
        cfgs[TOF_KEYS[d]] = tof_camera_cfg(d)
    return cfgs


# ── Runtime read helpers ──────────────────────────────────────────────────────

def _depth_patch(env, key: str) -> torch.Tensor:
    """Raw (num_envs, 8, 8) depth for one ToF, inf/NaN clamped to max range."""
    cam = env.scene[key]
    depth = cam.data.output["distance_to_camera"]  # (N, H, W, 1)
    depth = depth[..., 0]  # (N, H, W)
    # "No return" shows up as inf (beyond clip) or NaN; the physical sensor
    # reports max-range for those zones.
    depth = torch.nan_to_num(depth, nan=TOF_RANGE_MAX, posinf=TOF_RANGE_MAX, neginf=TOF_RANGE_MAX)
    return depth.clamp(TOF_RANGE_MIN, TOF_RANGE_MAX)


def tof_stack(env) -> torch.Tensor:
    """Depth from the 4 ToFs as (num_envs, 4, 8, 8) ordered [N, E, S, W].

    inf/NaN "no-return" zones are clamped to the 4.0 m max range.
    """
    patches = [_depth_patch(env, TOF_KEYS[d]) for d in TOF_ORDER]
    return torch.stack(patches, dim=1)  # (N, 4, 8, 8)


def tof_cross_composite(env) -> torch.Tensor:
    """Arrange the 4 ToF patches into a single 24x24 depth image (cross layout).

    The 8x8 patches are placed in a plus/cross so the spatial layout mirrors the
    physical cross rig (looking down on the drone, +x/N up)::

        rows\\cols   0:8        8:16       16:24
          0:8       zeros    [ N ]        zeros
          8:16      [ W ]    zeros(ctr)   [ E ]
         16:24      zeros    [ S ]        zeros

    Empty cells (including the 8x8 centre) are zero. Returns (num_envs, 24, 24).
    """
    stack = tof_stack(env)  # (N, 4, 8, 8) -> [N, E, S, W]
    n, e, s, w = stack[:, 0], stack[:, 1], stack[:, 2], stack[:, 3]
    num_envs = stack.shape[0]
    z = TOF_ZONES
    comp = torch.zeros((num_envs, 3 * z, 3 * z), dtype=stack.dtype, device=stack.device)
    comp[:, 0:z,        z:2 * z]    = n   # top-centre    (forward)
    comp[:, z:2 * z,    0:z]        = w   # middle-left   (left)
    comp[:, z:2 * z,    2 * z:3 * z] = e  # middle-right  (right)
    comp[:, 2 * z:3 * z, z:2 * z]   = s   # bottom-centre (behind)
    return comp


def front_greyscale(env, key: str = FRONT_CAM_KEY) -> torch.Tensor:
    """Front-camera luma as (num_envs, 1, H, W), float in [0, 1].

    luma = 0.299 R + 0.587 G + 0.114 B (Rec.601), matching the mono output of
    the physical Himax HM01B0.
    """
    cam = env.scene[key]
    rgb = cam.data.output["rgb"]  # (N, H, W, C) uint8 or float
    rgb = rgb[..., :3].to(torch.float32)
    if rgb.max() > 1.5:  # uint8-valued render -> normalise to [0, 1]
        rgb = rgb / 255.0
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    luma = (rgb * weights).sum(dim=-1)  # (N, H, W)
    return luma.unsqueeze(1)  # (N, 1, H, W)


# Monocular-depth auxiliary supervision range (avoidance-relevant band). Near
# obstacles matter most, so the target is NORMALISED INVERSE depth: near -> 1,
# far -> 0. Training-only (see front_greyscale_camera_cfg); never read at deploy.
FRONT_DEPTH_MIN = 0.15
FRONT_DEPTH_MAX = 10.0


def front_depth(env, key: str = FRONT_CAM_KEY, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Front-camera ground-truth depth as normalised INVERSE depth, (N, 1, h, w) in [0, 1].

    TRAINING-ONLY supervision for the mono-depth auxiliary task. Reads the
    ``distance_to_camera`` annotator co-registered with the greyscale render,
    clamps to the avoidance band, and maps to inverse depth (near=1, far=0) so
    the loss weights nearby obstacles — exactly the signal the nav policy needs.
    ``out_hw`` optionally area-resizes to the aux-head output resolution.
    """
    cam = env.scene[key]
    d = cam.data.output["distance_to_camera"]  # (N, H, W) or (N, H, W, 1)
    if d.dim() == 4:
        d = d[..., 0]
    d = torch.nan_to_num(d, nan=FRONT_DEPTH_MAX, posinf=FRONT_DEPTH_MAX, neginf=FRONT_DEPTH_MAX)
    d = d.clamp(FRONT_DEPTH_MIN, FRONT_DEPTH_MAX)
    inv = (1.0 / d - 1.0 / FRONT_DEPTH_MAX) / (1.0 / FRONT_DEPTH_MIN - 1.0 / FRONT_DEPTH_MAX)
    inv = inv.unsqueeze(1)  # (N, 1, H, W)
    if out_hw is not None:
        inv = torch.nn.functional.interpolate(inv, size=out_hw, mode="area")
    return inv


# ── Downward stack (Bitcraze Flow-deck-v2 + barometer) ────────────────────────
#
# These three are DERIVED from drone state rather than new camera prims: the real
# PMW3901 optical-flow chip only outputs aggregate (dx, dy) ~ ground-relative
# horizontal velocity / height; the downward VL53L1X reads height-above-ground
# (~= altitude over the ~flat forest floor); the barometer reads altitude. This
# keeps the rig cheap and matches what the sensors physically provide. (A true
# downward depth-cam can replace the z-based height later if terrain matters.)

# VL53L1X downward single-zone ToF: 27° FoV, 4 cm - 4 m range, up to 50 Hz.
DOWN_TOF_RANGE_MIN = 0.04
DOWN_TOF_RANGE_MAX = 4.0
DOWN_TOF_HZ = 50.0

# PMW3901 optical flow: 42° FoV, 80 mm min working distance, 121 FPS. Outputs
# aggregate motion (dx, dy) in "counts" ~ angular ground flow. We model
#   flow = (v_body_xy / height) * FLOW_SCALE   (counts per (m/s)/m of flow)
# FLOW_SCALE folds the pixel resolution over the FoV; absolute value is
# arbitrary as long as it's consistent (the model / estimator learns the scale).
FLOW_MIN_HEIGHT = 0.08   # below this the sensor can't focus -> invalid
FLOW_SCALE = 500.0       # counts per unit angular flow (order-of-magnitude PMW3901)
FLOW_HZ = 100.0

# Barometer: relative + global altitude. Slow drift dominates its error, so the
# stateful drift is maintained by the estimator; here we add white noise.
BARO_NOISE_M = 0.15
BARO_HZ = 25.0


def height_agl(env) -> torch.Tensor:
    """Height above ground level, (num_envs,), from the body z over the env origin.

    Flat-floor assumption for the forest scene; the cross ToFs handle lateral
    obstacle depth, this is the downward altitude channel.
    """
    robot = env.scene["robot"]
    return (robot.data.root_pos_w - env.scene.env_origins)[:, 2]


def down_tof(env, noise: bool = False, rng: torch.Generator | None = None) -> torch.Tensor:
    """Downward VL53L1X range as (num_envs, 1), clamped to 0.04-4.0 m.

    Values outside range read as the saturated bound (use :func:`normalize_range`
    to also get a validity flag).
    """
    h = height_agl(env).unsqueeze(1)
    if noise:
        n = torch.randn(h.shape, generator=rng, device=h.device, dtype=h.dtype)
        h = h * (1.0 + 0.02 * n)  # ~2% ranging noise
    return h.clamp(DOWN_TOF_RANGE_MIN, DOWN_TOF_RANGE_MAX)


def optical_flow(env, noise: bool = False, rng: torch.Generator | None = None) -> torch.Tensor:
    """PMW3901 aggregate optical flow (num_envs, 2) = (dx, dy).

    Modeled as body-frame horizontal velocity divided by height-above-ground and
    scaled to sensor counts (``FLOW_SCALE``). Below ``FLOW_MIN_HEIGHT`` the sensor
    can't range and flow reads ~0 (flagged invalid via :func:`optical_flow_valid`).
    """
    robot = env.scene["robot"]
    v_xy = robot.data.root_lin_vel_b[:, :2]              # (N, 2) body-frame horiz vel
    h = height_agl(env).clamp(min=FLOW_MIN_HEIGHT).unsqueeze(1)
    flow = (v_xy / h) * FLOW_SCALE
    below = (height_agl(env) < FLOW_MIN_HEIGHT).unsqueeze(1)
    flow = torch.where(below, torch.zeros_like(flow), flow)
    if noise:
        n = torch.randn(flow.shape, generator=rng, device=flow.device, dtype=flow.dtype)
        flow = flow + 2.0 * n  # few-count sensor noise
    return flow


def optical_flow_valid(env) -> torch.Tensor:
    """(num_envs, 1) validity flag: 1 where the drone is above the flow min height."""
    return (height_agl(env) >= FLOW_MIN_HEIGHT).unsqueeze(1).to(torch.float32)


def barometer(env, drift: torch.Tensor | None = None, ref: torch.Tensor | None = None,
              noise: bool = False, rng: torch.Generator | None = None) -> torch.Tensor:
    """Barometric altitude as (num_envs, 2) = (relative, global).

    ``global`` = altitude above the world origin; ``relative`` = altitude above a
    per-env reference ``ref`` (defaults to 0 -> equals global). ``drift`` (an
    optional per-env additive bias the estimator maintains as a random walk)
    captures the barometer's dominant slow-drift error; here we add white noise.
    """
    robot = env.scene["robot"]
    glob = robot.data.root_pos_w[:, 2].clone()
    if drift is not None:
        glob = glob + drift
    if noise:
        n = torch.randn(glob.shape, generator=rng, device=glob.device, dtype=glob.dtype)
        glob = glob + BARO_NOISE_M * n
    rel = glob - (ref if ref is not None else torch.zeros_like(glob))
    return torch.stack([rel, glob], dim=1)


def normalize_range(x: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ``x`` to [0,1] over [lo,hi] and return (normalized, validity_flag).

    validity = 1 where ``lo <= x <= hi`` (in-range / not saturated), else 0. The
    normalized value is clamped to [0,1]. Use to feed the model spec-faithful,
    range-aware inputs (task F1b).
    """
    valid = ((x >= lo) & (x <= hi)).to(x.dtype)
    norm = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return norm, valid


# ── ToF noise model ───────────────────────────────────────────────────────────

def add_tof_noise(
    depth: torch.Tensor,
    rng: torch.Generator | None = None,
    range_noise_pct: float = 0.03,
    dropout_prob: float = 0.05,
    dropout_range_frac: float = 0.6,
) -> torch.Tensor:
    """Apply VL53L5CX-style noise to a depth tensor (opt-in / toggleable).

    Models two dominant error sources of the real multizone ToF:

    * **Multiplicative range noise** — ranging error grows with distance; the
      datasheet quotes a few-% std, so each zone is scaled by
      ``1 + N(0, range_noise_pct)``.
    * **Far-zone dropout** — zones staring at distant / low-reflectance targets
      occasionally fail to get a return. Zones whose (noised) range exceeds
      ``dropout_range_frac * TOF_RANGE_MAX`` are dropped with probability
      ``dropout_prob`` and reported as the max range (the sensor's "no target"
      code, consistent with :func:`tof_stack`'s clamping).

    Works on any shape ((N,4,8,8), (N,8,8), ...); operates elementwise. Set
    ``dropout_prob=0`` to disable dropout, ``range_noise_pct=0`` for noise-free.

    Args:
        depth: Clamped depth in metres.
        rng: Optional ``torch.Generator`` for reproducibility.
        range_noise_pct: Std of the multiplicative range error (fraction).
        dropout_prob: Per-zone dropout probability for far zones.
        dropout_range_frac: Fraction of max range above which dropout applies.

    Returns:
        A new tensor (same shape) with noise applied, re-clamped to sensor range.
    """
    noisy = depth.clone()
    if range_noise_pct > 0.0:
        noise = torch.randn(depth.shape, generator=rng, device=depth.device, dtype=depth.dtype)
        noisy = noisy * (1.0 + range_noise_pct * noise)
    if dropout_prob > 0.0:
        far = noisy > (dropout_range_frac * TOF_RANGE_MAX)
        roll = torch.rand(depth.shape, generator=rng, device=depth.device, dtype=depth.dtype)
        drop = far & (roll < dropout_prob)
        noisy = torch.where(drop, torch.full_like(noisy, TOF_RANGE_MAX), noisy)
    return noisy.clamp(TOF_RANGE_MIN, TOF_RANGE_MAX)
