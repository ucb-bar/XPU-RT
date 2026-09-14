"""Numerically verify the fine-tick rescaling against the ACTUAL controller math.

arm_pd_ee_target_delta_pose_align2 (use_target=True) integrates the commanded
target pose as, from PDEEPoseController.compute_target_pose:

    target <- ( Pose(p=c) * delta * Pose(p=c)^-1 ) * target

with c = compute_fk(current qpos) = the actual ee position.  Claim: replacing one
200 ms delta by N sub-deltas with dp/N and rotvec/N reproduces the same target
after N sub-steps (exactly when c is frozen, to O(|dp|*angle) when c drifts).
"""
import numpy as np, sapien.core as sapien
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2axangle

def step(target, c, dp, rotvec):
    dq = Rotation.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]]
    d = sapien.Pose(dp, dq)
    C = sapien.Pose(p=c)
    return (C * d * C.inv()) * target

rng = np.random.default_rng(0)
N = 5
worst_p = worst_q = 0.0
worst_p_drift = worst_q_drift = 0.0
for trial in range(2000):
    # magnitudes typical of octo bridge actions
    dp = rng.normal(0, 0.02, 3)
    euler = rng.normal(0, 0.15, 3)
    ax, ang = euler2axangle(*euler)
    rotvec = ax * ang
    t0 = sapien.Pose(rng.normal(0, 0.3, 3),
                     Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_quat()[[3, 0, 1, 2]])
    c = rng.normal(0, 0.3, 3)

    coarse = step(t0, c, dp, rotvec)

    fine = t0
    for _ in range(N):
        fine = step(fine, c, dp / N, rotvec / N)
    worst_p = max(worst_p, np.abs(coarse.p - fine.p).max())
    worst_q = max(worst_q, min(np.abs(coarse.q - fine.q).max(),
                               np.abs(coarse.q + fine.q).max()))

    # with the pivot c drifting 2 mm per sub-tick (the arm is servoing)
    fine2, cc = t0, c.copy()
    for _ in range(N):
        fine2 = step(fine2, cc, dp / N, rotvec / N)
        cc = cc + rng.normal(0, 0.002, 3)
    worst_p_drift = max(worst_p_drift, np.abs(coarse.p - fine2.p).max())
    worst_q_drift = max(worst_q_drift, min(np.abs(coarse.q - fine2.q).max(),
                                           np.abs(coarse.q + fine2.q).max()))

    # counter-example: scaling the EULER angles instead of the rotvec
    if trial == 0:
        ax2, ang2 = euler2axangle(*(euler / N))
        fine3 = t0
        for _ in range(N):
            fine3 = step(fine3, c, dp / N, ax2 * ang2)
        print(f"[euler-scaled variant] quat err vs coarse = "
              f"{min(np.abs(coarse.q-fine3.q).max(), np.abs(coarse.q+fine3.q).max()):.3e} "
              f"(rotvec-scaled: {worst_q:.3e})")

print(f"N={N} sub-steps, 2000 random trials, |dp|~2cm, |euler|~0.15 rad")
print(f"  frozen pivot : max |dp| err = {worst_p:.3e} m,  max |quat| err = {worst_q:.3e}")
print(f"  drifting pivot (2 mm/sub-tick): max |dp| err = {worst_p_drift:.3e} m, "
      f"max |quat| err = {worst_q_drift:.3e}")

# ---- same check on the REAL octo action distribution ------------------------
import glob
_A = np.concatenate([np.load(f) for f in sorted(glob.glob(
    "/scratch2/dima/misc_sw/octo_work/sim_eval/runs/"
    "widowx_put_eggplant_in_basket_octo-small_rng*/ep*_raw_actions.npy"))])
_tr = np.linalg.norm(_A[:, :3], axis=1)
_ang = np.array([euler2axangle(*a)[1] for a in _A[:, 3:6]])
print(f"\nreal octo bridge actions (n={len(_A)}): |world_vector| mean {_tr.mean():.4f} m "
      f"p95 {np.percentile(_tr,95):.4f}; rot angle mean {_ang.mean():.4f} rad "
      f"p95 {np.percentile(_ang,95):.4f}")
_errs = []
_idx = rng.choice(len(_A), 4000, replace=False)
for i in _idx:
    dp = _A[i, :3].astype(float)
    ax, a = euler2axangle(*_A[i, 3:6].astype(float))
    rv = ax * a
    t0 = sapien.Pose(rng.normal(0, 0.3, 3),
                     Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_quat()[[3, 0, 1, 2]])
    c = rng.normal(0, 0.3, 3)
    coarse = step(t0, c, dp, rv)
    fine = t0
    for _ in range(N):
        fine = step(fine, c, dp / N, rv / N)
    _errs.append(np.linalg.norm(coarse.p - fine.p))
_errs = np.array(_errs)
print(f"target-position divergence, one 200 ms delta vs 5 x (delta/5): "
      f"mean {_errs.mean()*1000:.3f} mm, p95 {np.percentile(_errs,95)*1000:.3f} mm, "
      f"max {_errs.max()*1000:.3f} mm  (per-action commanded motion is ~10 mm)")
