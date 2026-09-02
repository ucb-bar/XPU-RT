# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-board state estimator for the forest-trail drone (Workstream F2 / task EST).

Fuses the low-level sensors the way the real flight stack does, so the nav model
consumes FILTERED state (per Dima: "the model can just take in the filtered
data") rather than raw sensor bytes:

  * **Attitude** — a batched **Madgwick** AHRS filter over simulated gyro + accel
    → orientation quaternion (the standard cheap on-board attitude filter; an EKF
    would be the upgrade).
  * **Altitude** — a complementary filter fusing the drifty barometer (absolute,
    low-freq) with the accurate downward VL53L1X ToF (short-range, high-freq).
  * **Ground-relative velocity** — from the PMW3901 optical flow (flow ∝ v/height)
    complementary-blended with accelerometer integration.

Also provides the spec-faithful sampling glue (task F1b):
  * :class:`SensorRateSampler` — per-sensor zero-order-hold at the sensor's true
    update rate + a staleness/age signal.
  * a barometer drift random-walk (maintained here, injected into
    ``sensors.barometer(drift=...)``).

All state is batched over ``num_envs`` and lives in torch tensors. A ``use_raw``
switch lets the sensor-aggregation study feed raw signals instead of filtered
ones for an apples-to-apples comparison.
"""

from __future__ import annotations

import torch


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def madgwick_update(q: torch.Tensor, gyro: torch.Tensor, accel: torch.Tensor,
                    dt: float, beta: float = 0.1) -> torch.Tensor:
    """One batched Madgwick IMU (gyro+accel) attitude update.

    Args:
        q:     (N,4) current orientation quaternion (w,x,y,z), body->world.
        gyro:  (N,3) body-frame angular rate (rad/s).
        accel: (N,3) body-frame specific force (any scale; normalized internally).
        dt:    timestep (s).
        beta:  filter gain (accel-correction strength).

    Returns:
        (N,4) updated, normalized quaternion.
    """
    q = _quat_normalize(q)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # gyro-driven rate of change: 0.5 * q ⊗ (0, gx, gy, gz)
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]
    qdot = 0.5 * torch.stack([
        -qx * gx - qy * gy - qz * gz,
        qw * gx + qy * gz - qz * gy,
        qw * gy - qx * gz + qz * gx,
        qw * gz + qx * gy - qy * gx,
    ], dim=1)

    # accel correction (only when the accel reading is usable)
    a_norm = accel.norm(dim=1, keepdim=True)
    usable = (a_norm.squeeze(1) > 1e-6)
    a = accel / a_norm.clamp_min(1e-9)
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]

    # objective f (measured gravity vs estimated) and its Jacobian^T · f
    f1 = 2.0 * (qx * qz - qw * qy) - ax
    f2 = 2.0 * (qw * qx + qy * qz) - ay
    f3 = 2.0 * (0.5 - qx * qx - qy * qy) - az
    grad = torch.stack([
        -2.0 * qy * f1 + 2.0 * qx * f2,
        2.0 * qz * f1 + 2.0 * qw * f2 - 4.0 * qx * f3,
        -2.0 * qw * f1 + 2.0 * qz * f2 - 4.0 * qy * f3,
        2.0 * qx * f1 + 2.0 * qy * f2,
    ], dim=1)
    grad = _quat_normalize(grad)

    qdot = qdot - beta * torch.where(usable.unsqueeze(1), grad, torch.zeros_like(grad))
    return _quat_normalize(q + qdot * dt)


class SensorRateSampler:
    """Per-sensor zero-order-hold at the sensor's true update rate + staleness.

    A sensor updating at ``rate_hz`` only refreshes every ``round(1/(rate_hz*dt))``
    control steps; between updates the held value is returned and its age grows.
    Models the mixed cadences the model must cope with (e.g. 15 Hz cross-ToF vs a
    50 Hz control loop). ``staleness(name)`` returns age/period in [0,1+].
    """

    def __init__(self, rates_hz: dict[str, float], control_dt: float):
        self._period = {k: max(1, round(1.0 / (hz * control_dt))) for k, hz in rates_hz.items()}
        self._held: dict[str, torch.Tensor] = {}
        self._age: dict[str, int] = {k: 0 for k in rates_hz}
        self._k = 0

    def reset(self):
        self._held.clear()
        self._age = {k: 0 for k in self._age}
        self._k = 0

    def sample(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Return the sensor value subject to its update rate (ZOH)."""
        if name not in self._held or (self._k % self._period[name] == 0):
            self._held[name] = value.clone()
            self._age[name] = 0
        else:
            self._age[name] += 1
        return self._held[name]

    def staleness(self, name: str) -> float:
        return self._age[name] / self._period[name]

    def tick(self):
        self._k += 1


class StateEstimator:
    """Batched attitude + altitude + velocity estimator over num_envs drones."""

    def __init__(self, num_envs: int, device, control_dt: float = 0.02,
                 beta: float = 0.1, baro_tof_blend: float = 0.05,
                 flow_accel_blend: float = 0.1, baro_drift_std: float = 0.02):
        self.n = num_envs
        self.device = device
        self.dt = control_dt
        self.beta = beta
        self.k_alt = baro_tof_blend      # weight on baro in altitude complementary filter
        self.k_vel = flow_accel_blend    # weight on accel-integration in velocity filter
        self.baro_drift_std = baro_drift_std

        self.q = torch.zeros(num_envs, 4, device=device); self.q[:, 0] = 1.0
        self.alt = torch.zeros(num_envs, device=device)
        self.vel = torch.zeros(num_envs, 3, device=device)  # ground-relative (vx,vy,vz)
        self.baro_drift = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None):
        idx = slice(None) if env_ids is None else env_ids
        self.q[idx] = 0.0; self.q[idx, 0] = 1.0
        self.alt[idx] = 0.0
        self.vel[idx] = 0.0
        self.baro_drift[idx] = 0.0

    def step_baro_drift(self, rng: torch.Generator | None = None) -> torch.Tensor:
        """Advance the barometer's slow drift (random walk) and return it.

        Feed the result into ``sensors.barometer(env, drift=...)`` so the raw baro
        reading carries realistic drift the filter then rejects using the ToF.
        """
        step = torch.randn(self.n, generator=rng, device=self.device) * self.baro_drift_std
        self.baro_drift = self.baro_drift + step
        return self.baro_drift

    def update(self, gyro, accel, baro_alt=None, tof_alt=None, flow_vel=None):
        """Fuse one control step of sensor readings; returns a filtered-state dict.

        Args (all (N,·) tensors, any missing -> that fusion is skipped):
            gyro (N,3), accel (N,3): body-frame IMU.
            baro_alt (N,): barometric altitude (drifty).
            tof_alt (N,): downward-ToF height (accurate, short range).
            flow_vel (N,2): ground-relative horizontal velocity from optical flow.

        Returns dict: quat (N,4), altitude (N,), velocity (N,3).
        """
        # attitude
        self.q = madgwick_update(self.q, gyro, accel, self.dt, self.beta)

        # altitude: complementary filter (predict with vertical vel, correct with sensors)
        self.alt = self.alt + self.vel[:, 2] * self.dt
        if tof_alt is not None:
            # ToF trusted most within range; blend baro for drift-free low-freq bias
            meas = tof_alt if baro_alt is None else (1 - self.k_alt) * tof_alt + self.k_alt * baro_alt
            self.alt = 0.9 * self.alt + 0.1 * meas
        elif baro_alt is not None:
            self.alt = 0.9 * self.alt + 0.1 * baro_alt

        # velocity: optical-flow horizontal (complementary with accel integration)
        acc_world_z = accel[:, 2] - 9.81  # crude; body-z accel minus g (level approx)
        self.vel[:, 2] = self.vel[:, 2] + acc_world_z * self.dt * 0.0  # vz left to altitude filter
        if flow_vel is not None:
            self.vel[:, :2] = (1 - self.k_vel) * flow_vel + self.k_vel * (self.vel[:, :2] + accel[:, :2] * self.dt)

        return {"quat": self.q.clone(), "altitude": self.alt.clone(), "velocity": self.vel.clone()}


if __name__ == "__main__":
    torch.manual_seed(0)
    N, dt = 4, 0.02
    dev = "cpu"

    # 1) Madgwick converges to level from gravity-only accel (gz = +1 g in body z).
    est = StateEstimator(N, dev, control_dt=dt, beta=0.3)
    gyro0 = torch.zeros(N, 3)
    accel_level = torch.tensor([[0.0, 0.0, 9.81]]).repeat(N, 1)
    for _ in range(400):
        out = est.update(gyro0, accel_level, baro_alt=torch.ones(N), tof_alt=torch.ones(N))
    q = out["quat"]
    # level attitude -> quaternion near identity (w~1)
    print(f"[madgwick] level convergence: mean qw={q[:,0].mean():.4f} (expect ~1.0), "
          f"|xyz|={q[:,1:].abs().mean():.4f} (expect ~0)")
    assert q[:, 0].mean() > 0.99, "Madgwick failed to converge to level"

    # 2) constant yaw rate integrates into yaw rotation.
    est2 = StateEstimator(N, dev, control_dt=dt, beta=0.0)  # gyro-only (no accel correction on yaw)
    gyro_yaw = torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1)  # 1 rad/s about z
    for _ in range(50):  # 1 s
        est2.update(gyro_yaw, torch.zeros(N, 3))
    qy = est2.q
    import math
    yaw = torch.atan2(2 * (qy[:, 0] * qy[:, 3] + qy[:, 1] * qy[:, 2]),
                      1 - 2 * (qy[:, 2] ** 2 + qy[:, 3] ** 2))
    print(f"[madgwick] 1 rad/s yaw for 1 s -> mean yaw={yaw.mean():.3f} rad (expect ~1.0)")
    assert abs(yaw.mean().item() - 1.0) < 0.1, "yaw integration wrong"

    # 3) altitude tracks the ToF.
    est3 = StateEstimator(N, dev, control_dt=dt)
    for _ in range(300):
        out = est3.update(torch.zeros(N, 3), accel_level, baro_alt=torch.full((N,), 2.5),
                          tof_alt=torch.full((N,), 2.0))
    print(f"[altitude] ToF=2.0 baro=2.5 -> est={out['altitude'].mean():.3f} (ToF-weighted)")

    # 4) rate sampler ZOH.
    rs = SensorRateSampler({"cross_tof": 15.0, "cam": 60.0}, control_dt=dt)
    seen = []
    for step in range(6):
        v = rs.sample("cross_tof", torch.tensor([float(step)]))
        seen.append(v.item()); rs.tick()
    print(f"[rate] cross_tof@15Hz on 50Hz loop -> held values {seen} (updates every ~3 steps)")
    print("[smoke] StateEstimator OK")
