#!/usr/bin/env python3
"""AUDIT PASS 1 -- reduce traces_torque2/ to a per-episode, PER-JOINT scalar cache.

Everything the audit needs that the shipped energy_cache.json throws away:
  * per-joint  INT tau_i^2 dt   (the shipped cache sums over joints first)
  * per-joint  INT omega_i^2 dt
  * per-joint  max|tau|, mean|tau|, and the count of ticks at the force limit
  * the single largest per-tick contribution to sum_i tau_i^2   (spike dominance)
  * episode duration, success, and the number of DISTINCT sim states
    (google_robot replicates each state ACT_EVERY=9 times -- see pass 2)

Writes energy_audit_cache.json next to this file. Read-only w.r.t. the traces.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
OUT = HERE / "energy_audit_cache.json"

# URDF active-joint order (verified against the qf columns: the vertical-axis
# joints -- widowx `waist`, google `joint_torso`/`joint_head_pan` -- are the
# columns that read ~0).
NAMES = {
    8: ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
        "wrist_rotate", "left_finger", "right_finger"],
    11: ["joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow",
         "joint_forearm", "joint_wrist", "joint_gripper",
         "joint_finger_right", "joint_finger_left",
         "joint_head_pan", "joint_head_tilt"],
}
# force_limit as configured in the agent config (NOT the URDF effort tag).
FLIM = {
    8:  [200., 200., 100., 100., 100., 100., 60., 60.],
    11: [300., 300., 100., 100., 100., 100., 100., 60., 60., 100., 100.],
}
# URDF <limit effort=...> -- the description's own idea of the joint's rating.
ULIM = {
    8:  [10., 20., 15., 2., 5., 1., 5., 5.],
    11: [10.] * 11,
}
JG = {8: slice(0, 6), 11: slice(0, 7)}      # as in fig_energy.py

cache = {}
for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.load(open(f))
    dt = s["tick_ms"] / 1000.0
    act_every = int(s.get("act_every_ticks", 1))
    eps = []
    for e in s["episodes"]:
        ep = e["episode_id"]
        try:
            qf = np.load(d / f"ep{ep:02d}_qf.npy")
            w = np.load(d / f"ep{ep:02d}_qvel.npy")
        except Exception:
            continue
        dof = qf.shape[1]
        fl = np.asarray(FLIM[dof]); ul = np.asarray(ULIM[dof])
        per_tick = (qf ** 2).sum(1)                  # sum_i tau_i^2 at each tick
        # distinct sim states: google replicates each state act_every times
        chg = 1 + int((np.abs(np.diff(qf, axis=0)).max(1) > 1e-12).sum())
        eps.append(dict(
            ep=ep, T=int(qf.shape[0]), dur_s=float(qf.shape[0] * dt),
            success=bool(e["success"]), dof=dof, distinct=chg,
            t2=float(per_tick.sum() * dt),                       # SHIPPED metric
            t2_j=[float(x) for x in (qf ** 2).sum(0) * dt],      # per joint
            t2_arm=float(((qf ** 2).sum(0) * dt)[JG[dof]].sum()),
            t2_norm=float((((qf / fl) ** 2).sum(0) * dt).sum()), # 1/flim^2 weights
            t2_unorm=float((((qf / ul) ** 2).sum(0) * dt).sum()),# 1/urdf_eff^2
            eff=float(((w ** 2).sum(0) * dt)[JG[dof]].sum()),    # SHIPPED metric
            eff_all=float(((w ** 2).sum(0) * dt).sum()),
            eff_j=[float(x) for x in (w ** 2).sum(0) * dt],
            tmax=[float(x) for x in np.abs(qf).max(0)],
            tmean=[float(x) for x in np.abs(qf).mean(0)],
            n_sat_flim=int((np.abs(qf) >= fl * 0.999).sum()),
            n_sat_urdf=int((np.abs(qf) >= ul * 0.999).sum()),
            peak_tick_share=float(per_tick.max() * dt / (per_tick.sum() * dt))
                            if per_tick.sum() > 0 else 0.0,
            top1pct_share=float(
                np.sort(per_tick)[-max(1, len(per_tick)//100):].sum()
                / per_tick.sum()) if per_tick.sum() > 0 else 0.0,
            n_inf=e["n_inferences"], n_act=e.get("n_actuations", e["ticks"]),
        ))
    if not eps:
        continue
    cache[d.name] = dict(dt=dt, act_every=act_every, dof=eps[0]["dof"],
                         tick_ms=s["tick_ms"], act_ms=s.get("act_ms", s["tick_ms"]),
                         sim_freq=s["sim_freq"],
                         names=NAMES[eps[0]["dof"]], flim=FLIM[eps[0]["dof"]],
                         sr=100.0 * s["n_success"] / s["n_episodes"], eps=eps)
    print(f"[ok] {d.name:22s} {len(eps)} eps dof={eps[0]['dof']} "
          f"act_every={act_every}", flush=True)
json.dump(cache, open(OUT, "w"))
print(f"[done] {OUT}  {OUT.stat().st_size/1024:.0f} KB, {len(cache)} cells")
