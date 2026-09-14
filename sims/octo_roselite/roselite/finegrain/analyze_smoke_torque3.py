"""The four ENERGY_FIX.md checks, run on real trace_eval2.py rollouts.

Usage: python analyze_smoke_torque3.py smoke_torque3
Reads the UNREDUCED cells (per-actuation .npy still present).
"""
import json, os, sys, glob
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "smoke_torque3"
JG = {8: slice(0, 6), 11: slice(0, 7)}
FING = {8: [6, 7], 11: [7, 8]}
CONTACT_N = 20.0          # N, mean external contact force over an actuation
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def load_cell(d):
    s = json.load(open(f"{d}/summary.json"))
    dof = len(s["joint_names"]); arm = JG.get(dof, slice(None)); fing = FING.get(dof, [])
    adt = s["act_ms"] / 1000.0; tdt = s["tick_ms"] / 1000.0
    E = []
    for e in s["episodes"]:
        ep = e["episode_id"]
        g = lambda n: np.load(f"{d}/ep{ep:02d}_{n}.npy")
        con = g("contact"); tk = g("act_tick"); ee = g("ee_xyz")
        E.append(dict(ep=ep, success=e["success"], scal=e,
                      tau2=g("tau_drive_sq"), tau2s=g("tau_drive_sq_sus"),
                      tot2=g("tau_total_sq"),
                      qf2s=g("qf_sub_sq"), qv2s=g("qvel_sub_sq"),
                      taumax=g("tau_drive_max"), sat=g("sat_frac"),
                      con=con, tk=tk, ee=ee[tk], qfT=g("qf"), qvT=g("qvel"), eeT=ee))
    return dict(name=os.path.basename(d), s=s, arm=arm, fing=fing, adt=adt, tdt=tdt, E=E)


def pooled(c, key, cols=None):
    return np.concatenate([(e[key][:, cols] if cols is not None else e[key]) for e in c["E"]], 0)


cells = [load_cell(d) for d in sorted(glob.glob(f"{ROOT}/*")) if os.path.isdir(d)]
print(f"cells: {[c['name'] for c in cells]}\n")

# ---------------------------------------------------------------- CHECK (i)
print("=" * 100)
print("CHECK (i)  the metric is NOT zero at tick 0")
print("=" * 100)
print(f"{'cell':18s} {'n_ep':>4} {'OLD qf(tick0)^2':>17} {'qf_substep^2':>14} {'tau_drive^2':>13} {'tau_total^2':>13}")
for c in cells:
    v = [e["scal"] for e in c["E"]]
    f = lambda k: np.array([x[k] for x in v])
    print(f"{c['name']:18s} {len(v):4d} {str(set(np.round(f('qf_tick0'), 12))):>17} "
          f"{f('qf_sub_tick0').min():7.3f}-{f('qf_sub_tick0').max():<6.3f} "
          f"{f('tau_drive_tick0').min():6.4g}-{f('tau_drive_tick0').max():<6.4g} "
          f"{f('tau_total_tick0').min():6.3f}-{f('tau_total_tick0').max():<6.3f}")
print("  OLD channel is EXACTLY 0 on tick 0 of every episode (trace_eval.py logs get_qf()")
print("  before the first step_action, and BaseAgent.reset() has just set_qf(zeros)).")
print("  tau_total = qf + tau_drive, read inside the substep loop, is not.\n")

# ------------------------------------------------------- CHECK (ii) and (iii)
def pair_check(c, label, mask_fn, match="tcp"):
    arm = c["arm"]
    T = pooled(c, "tau2", arm).sum(1); TS = pooled(c, "tau2s", arm).sum(1)
    Q = pooled(c, "qf2s", arm).sum(1);  V = pooled(c, "qv2s", arm).sum(1)
    F = pooled(c, "con")[:, 2]; EE = np.concatenate([e["ee"] for e in c["E"]], 0)
    hi, lo = mask_fn(F, V)
    if hi.sum() == 0 or lo.sum() == 0:
        print(f"  {c['name']:18s} {label:34s} n_contact={hi.sum():4d} n_free={lo.sum():4d}  -- no pairs")
        return
    ih, il = np.where(hi)[0], np.where(lo)[0]
    if match == "tcp":
        d = np.linalg.norm(EE[ih][:, None, :] - EE[il][None, :, :], axis=2)
        j = d.argmin(1); ok = d[np.arange(len(ih)), j] <= 0.03
        a, b = ih[ok], il[j[ok]]
    else:
        a, b = ih, il
    if len(a) == 0:
        print(f"  {c['name']:18s} {label:34s} no pose-matched pairs within 3 cm")
        return
    r = lambda X: float(np.median(X[a]) / max(np.median(X[b]), 1e-12))
    print(f"  {c['name']:18s} {label:31s} n={len(a):4d} | "
          f"tau_drive^2 {np.median(T[a]):9.3f}/{np.median(T[b]):7.4f}={r(T):9.1f}x | "
          f"sustained {np.median(TS[a]):9.3f}/{np.median(TS[b]):8.5f}={r(TS):9.1f}x | "
          f"qf^2 {np.median(Q[a]):7.2f}/{np.median(Q[b]):6.2f}={r(Q):5.2f}x | "
          f"w^2 {r(V):5.2f}x | F {np.median(F[a]):6.1f}/{np.median(F[b]):.1f} N")

print("=" * 100)
print("CHECK (ii) contact vs free at MATCHED end-effector position (<= 3 cm), real rollouts")
print("           this is ENERGY_AUDIT.md 3.1's own pose control, applied to the new signal")
print("=" * 100)
for c in cells:
    pair_check(c, "CONTACT vs FREE, pose-matched",
               lambda F, V: (F >= CONTACT_N, F <= 0.0))
print("""  ENERGY_AUDIT.md 3.1 got 1.00 / 1.09 / 1.16 / 1.00 for the qf metric under exactly
  this pose control -- reproduced in the qf^2 column here. The drive torque does not
  collapse: contact is a real, separate axis of the signal.\n""")

print("=" * 100)
print("CHECK (iii) sustained torque at ~ZERO joint velocity (the stall case)")
print("            both bins restricted to the lowest quartile of Sum omega^2")
print("=" * 100)
for c in cells:
    V = pooled(c, "qv2s", c["arm"]).sum(1); thr = np.percentile(V, 25)
    pair_check(c, "in contact vs free, both |w| low",
               lambda F, V, thr=thr: ((F >= CONTACT_N) & (V <= thr), (F <= 0.0) & (V <= thr)),
               match="none")
print()

# ---------------------------------------------------------------- CHECK (iv)
print("=" * 100)
print("CHECK (iv) gripper: finger drive torque rises on contact")
print("=" * 100)
print(f"{'cell':18s} {'n_grip_contact':>14} {'|tau_fing| contact':>19} {'free':>10} {'ratio':>9} "
      f"{'qf_fing contact':>16} {'free':>9} {'ratio':>7} {'grip F':>9}")
for c in cells:
    fi = c["fing"]
    if not fi:
        continue
    tf = np.sqrt(pooled(c, "tau2", fi).sum(1))       # rms-ish finger drive force
    qf = np.sqrt(pooled(c, "qf2s", fi).sum(1))
    G = pooled(c, "con")[:, 4]
    hi, lo = G >= CONTACT_N, G <= 0.0
    if hi.sum() == 0 or lo.sum() == 0:
        print(f"{c['name']:18s} {hi.sum():14d}   -- no contrast"); continue
    print(f"{c['name']:18s} {hi.sum():14d} {np.median(tf[hi]):19.4f} {np.median(tf[lo]):10.5f} "
          f"{np.median(tf[hi])/max(np.median(tf[lo]),1e-12):9.1f} {np.median(qf[hi]):16.5f} "
          f"{np.median(qf[lo]):9.5f} {np.median(qf[hi])/max(np.median(qf[lo]),1e-12):7.2f} "
          f"{np.median(G[hi]):9.1f}")
print("  (ENERGY_AUDIT.md 1.1: with the fingers closed on the object the OLD channel reads")
print("   0.015 N on egg / 0.018 N on coke -- 'a real gripper clamping a coke can holds tens")
print("   of newtons'. The contact column now says how many.)\n")

# ------------------------------------------------- confounders: duration, reach
print("=" * 100)
print("CONFOUNDERS: is the new metric still duration x arm-extension?")
print("=" * 100)
rows = []
for c in cells:
    for e in c["E"]:
        s = e["scal"]
        rows.append((c["name"], s["duration_s"], s["tcp_reach_mean"], s["t2_qf_tick_arm"],
                     s["t2_drive_arm"], s["t2_drive_arm_sus"], s["contact_impulse_Ns"],
                     s["work_abs_arm"], bool(s["success"])))
import itertools
def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
names = ["OLD INT.qf^2dt", "NEW INT.tau_drive^2dt", "NEW sustained tau^2",
         "NEW contact impulse", "NEW INT|tau.w|dt"]
cols = [3, 4, 5, 6, 7]
for rate in (False, True):
    print(f"\n-- {'PER SECOND' if rate else 'INTEGRAL'}")
    print(f"{'scope':20s} {'n':>3} " + " ".join(f"{n:>23s}" for n in names))
    for scope in ["ALL"] + [c["name"] for c in cells]:
        R = [r for r in rows if scope == "ALL" or r[0] == scope]
        if len(R) < 3: continue
        d = np.array([r[1] for r in R], float); k = [r[2] for r in R]
        out = []
        for ci in cols:
            y = np.array([r[ci] for r in R], float)
            if rate: y = y / d
            out.append(f"dur{corr(d,y):+.2f} reach{corr(k,y):+.2f}")
        print(f"{scope:20s} {len(R):3d} " + " ".join(f"{o:>23s}" for o in out))
print("\nper-cell medians (per-second, arm joints):")
print(f"{'cell':18s} {'succ':>5} {'dur s':>7} {'reach m':>8} {'OLD/s':>10} {'DRIVE/s':>12} "
      f"{'SUSTAIN/s':>12} {'Wabs J':>9} {'|tau|max':>9} {'sat%':>6} {'INTFdt Ns':>10} {'Fmax N':>9} {'gripF N':>9}")
for c in cells:
    v = [e["scal"] for e in c["E"]]
    m = lambda k: float(np.median([x[k] for x in v]))
    mm = lambda k: float(np.max([x[k] for x in v]))
    print(f"{c['name']:18s} {sum(x['success'] for x in v):2d}/{len(v):<2d} {m('duration_s'):7.2f} "
          f"{m('tcp_reach_mean'):8.4f} {m('t2_qf_tick_arm')/m('duration_s'):10.4f} "
          f"{m('t2_drive_arm')/m('duration_s'):12.2f} {m('t2_drive_arm_sus')/m('duration_s'):12.4f} "
          f"{m('work_abs_arm'):9.3f} {mm('tau_drive_absmax_arm'):9.2f} {100*m('sat_frac_arm'):6.2f} "
          f"{m('contact_impulse_Ns'):10.2f} {mm('contact_Fsum_max'):9.1f} {mm('contact_grip_Fmax'):9.1f}")
