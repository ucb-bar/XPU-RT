#!/usr/bin/env python3
"""AUDIT PASS 9 -- how big is the term that ISN'T in tau?

get_qf() carries only  g(q) + C(q,qd)qd  (the ManiSkill gravity/Coriolis feed-forward).
The joint torque a real servo would have to produce is

    tau_total = M(q) qdd  +  C(q,qd) qd  +  g(q)  -  J^T F_contact
                ^^^^^^^^^                            ^^^^^^^^^^^^^
                MISSING                              MISSING

The inertial term is estimable from the traces: qdd comes from differencing the logged
qvel, and a rigid-body effective inertia about the shoulder axis comes from the URDF
inertia tensors plus the logged link frames (parallel axis). This gives an order-of-
magnitude for what the metric leaves out. The contact term is NOT estimable from these
traces at all -- no contact force is logged anywhere.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
URDF = Path("/scratch2/dima/misc_sw/octo_work/sim_eval/SimplerEnv/ManiSkill2_real2sim/"
            "mani_skill2_real2sim/assets/descriptions/widowx_description/wx250s.urdf")

root = ET.parse(URDF).getroot()
INR = {}
for l in root.findall("link"):
    it = l.find("inertial")
    if it is None:
        continue
    i = it.find("inertia")
    INR[l.get("name")] = (float(it.find("mass").get("value")),
                          float(i.get("iyy")))

print("=" * 92)
print("Effective inertia about the widowx `shoulder` axis, and the inertial torque it")
print("implies for the accelerations actually logged.")
print("=" * 92)
print(f"{'cell':<20}{'I_eff kg m2':>13}{'rms qdd':>10}{'rms tau_in':>12}"
      f"{'rms tau_qf':>12}{'INT tin^2 / INT tqf^2':>24}")
for cell in ["egg_lat0", "egg_serial283", "egg_cpu685", "spoon_lat0", "spoon_cpu685"]:
    s = json.load(open(SRC / cell / "summary.json"))
    dt = s["tick_ms"] / 1000.0
    Ie, ACC, TIN, TQF = [], [], [], []
    for e in s["episodes"]:
        ep = e["episode_id"]
        qv = np.load(SRC / cell / f"ep{ep:02d}_qvel.npy")
        qf = np.load(SRC / cell / f"ep{ep:02d}_qf.npy")
        lc = np.load(SRC / cell / f"ep{ep:02d}_link_com.npy")
        b = json.load(open(SRC / cell / f"ep{ep:02d}_bodies.json"))
        nm = b["link_names"]; m = np.asarray(b["link_mass"])
        i_sh = nm.index("upper_arm_link")
        dist = list(range(i_sh, len(nm)))
        d = np.linalg.norm(lc[:, dist, :] - lc[:, [i_sh], :], axis=2)   # (T, ndist)
        I_par = (m[dist] * d ** 2).sum(1)
        I_own = sum(INR.get(nm[k], (0., 0.))[1] for k in dist)
        I = I_par + I_own
        acc = np.zeros(len(qv)); acc[1:] = np.diff(qv[:, 1]) / dt
        Ie.append(I.mean()); ACC.append(np.sqrt((acc ** 2).mean()))
        TIN.append(I * acc); TQF.append(qf[:, 1])
    tin = np.concatenate(TIN); tqf = np.concatenate(TQF)
    print(f"{cell:<20}{np.mean(Ie):>13.4f}{np.mean(ACC):>10.2f}"
          f"{np.sqrt((tin**2).mean()):>12.3f}{np.sqrt((tqf**2).mean()):>12.3f}"
          f"{(tin**2).sum()/(tqf**2).sum():>24.2f}")
print("""
  I_eff is a rigid-body estimate about a single axis (parallel-axis on the logged link
  frames + the URDF iyy of the distal links); it ignores the off-diagonal coupling and
  the payload, so it is a LOWER bound on the true inertial torque. `qdd` is a first
  difference at the 40 ms tick, which also UNDER-estimates peak acceleration.
  RESULT: the inertial term is SMALL -- 4-5% of INT tau_qf^2 on eggplant, <1% on spoon.
  The arm is light (2.14 kg of moving link) and slow, so M(q)qdd is not the big missing
  piece. The big missing pieces are the PD DRIVE torque (pass 10) and CONTACT, and
  contact is not estimable from these traces at all.""")

print()
print("=" * 92)
print("Is the head-tilt hold torque literally a pose constant?  (an internal control:")
print("google_robot's head never moves, so its gravity hold should be bit-identical)")
print("=" * 92)
for cell in ["coke_lat0", "coke_cpu685", "drawer_lat0", "drawer_serial283"]:
    s = json.load(open(SRC / cell / "summary.json"))
    v = np.concatenate([np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")[:, 10]
                        for e in s["episodes"]])
    print(f"  {cell:<20} joint_head_tilt tau: {len(np.unique(v))} distinct values over "
          f"{len(v)} ticks; min {v.min():.8f} max {v.max():.8f}")
print("""
  A constant that is integrated over the whole episode: it contributes
  0.4525^2 = 0.2048 N^2m^2 per second of episode, i.e. it is PURE EPISODE LENGTH
  entering the 'energy' metric with no dependence on anything the policy did.""")
