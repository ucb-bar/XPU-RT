"""traces_torque3 sweep analysis: the new actuator-torque and contact columns
against the old gravity-hold column, on the same 36 cells x 24 episodes.

Usage: python analyze_torque3.py [traces_torque3]
Reads the REDUCED cells (energy2.json + series.npz).
"""
import json, os, sys, glob
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "traces_torque3"
TASKS = ["egg", "spoon", "coke", "drawer"]
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300",
        "pipe200fix", "serial283", "fp32_555", "cpu685"]
CAD = {"lat0": ("--", 0), "pipe110fix": (111, 385), "p105w300": (125, 259),
       "p130w275": (130, 272), "p150w300": (150, 282), "pipe200fix": (219, 260),
       "serial283": (283, 283), "fp32_555": (555, 555), "cpu685": (685, 685)}
rng = np.random.default_rng(0)

C = {}
for d in sorted(glob.glob(f"{ROOT}/*")):
    if not os.path.isfile(f"{d}/energy2.json"):
        continue
    k = os.path.basename(d)
    t, a = k.split("_", 1)
    if t in TASKS and a in ARMS:
        C[(t, a)] = json.load(open(f"{d}/energy2.json"))
print(f"loaded {len(C)}/36 cells from {ROOT}\n")
missing = [f"{t}_{a}" for t in TASKS for a in ARMS if (t, a) not in C]
if missing:
    print("MISSING:", missing, "\n")


def col(t, a, k):
    return np.array([e[k] for e in sorted(C[(t, a)]["episodes"], key=lambda x: x["ep"])], float)


def paired_ratio(t, a, k, per_s=False):
    """median over episodes of  metric(a,e)/metric(lat0,e), with a paired
    bootstrap 95% CI. Every cell of a task shares init_rng=100 and episode ids
    0..23, so episode e is the same scene in every arm."""
    x, y = col(t, a, k), col(t, "lat0", k)
    if per_s:
        x = x / col(t, a, "duration_s"); y = y / col(t, "lat0", "duration_s")
    ok = (y > 0) & np.isfinite(x) & np.isfinite(y)
    r = x[ok] / y[ok]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan")
    b = np.median(r[rng.integers(0, r.size, (4000, r.size))], axis=1)
    return float(np.median(r)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ------------------------------------------------------------------- table
print("=" * 132)
print("MEDIAN PER CELL. old = INT Sum qf^2 dt over ALL dof -- EXACTLY fig_energy.py's t2,")
print("so it should reproduce ENERGY_RESULTS.md's 82.1 / 35.8 / 7854.3 / 6754.5 for lat0")
print("to within run-to-run nondeterminism (NONDETERMINISM.md: ~20% of runs diverge).")
print("drive = INT Sum tau_drive^2 dt.  sus = the same over substeps >= 5 (drops widowx's")
print("5 Hz setpoint-step transient).  Fimp = INT Sum|contact F| dt.  Wabs = INT Sum|tau.w| dt.")
print("=" * 132)
for t in TASKS:
    print(f"\n--- {t}")
    print(f"{'arm':12s} {'cad/age':>10} {'succ':>6} {'dur s':>6} {'reach':>6} | "
          f"{'old':>9} {'x':>6} | {'drive':>11} {'x':>6} | {'sus':>10} {'x':>6} | "
          f"{'Fimp Ns':>9} {'x':>6} | {'Wabs J':>8} {'x':>6} | {'|t|max':>7} {'sat%':>5}")
    for a in ARMS:
        if (t, a) not in C:
            continue
        s = C[(t, a)]
        m = lambda k: float(np.median(col(t, a, k)))
        r = lambda k: m(k) / max(float(np.median(col(t, "lat0", k))), 1e-12)
        print(f"{a:12s} {str(CAD[a][0]):>4}/{CAD[a][1]:<5} "
              f"{100*s['n_success']/s['n_episodes']:5.1f}% {m('duration_s'):6.2f} "
              f"{m('tcp_reach_mean'):6.3f} | {m('t2_qf_tick'):9.2f} {r('t2_qf_tick'):6.2f} | "
              f"{m('t2_drive_arm'):11.1f} {r('t2_drive_arm'):6.2f} | "
              f"{m('t2_drive_arm_sus'):10.3f} {r('t2_drive_arm_sus'):6.2f} | "
              f"{m('contact_impulse_Ns'):9.2f} {r('contact_impulse_Ns'):6.2f} | "
              f"{m('work_abs_arm'):8.3f} {r('work_abs_arm'):6.2f} | "
              f"{float(np.max(col(t,a,'tau_drive_absmax_arm'))):7.1f} {100*m('sat_frac_arm'):5.2f}")

# ------------------------------------------------------ confounder analysis
print("\n" + "=" * 132)
print("IS THE NEW METRIC STILL duration x arm-extension?  Pearson r, pooled over the")
print("216 episodes of each task (the audit's 3.4 / 1.x test, rerun on every column).")
print("=" * 132)
KEYS = [("OLD INT.qf^2dt", "t2_qf_tick_arm"), ("drive INT.tau^2dt", "t2_drive_arm"),
        ("drive sustained", "t2_drive_arm_sus"), ("contact impulse", "contact_impulse_Ns"),
        ("INT|tau.w|dt", "work_abs_arm"), ("peak contact F", "contact_Fsum_max")]
for rate in (False, True):
    print(f"\n-- {'PER SECOND (metric / duration)' if rate else 'INTEGRAL over the episode'}")
    print(f"{'task':8s} {'n':>4} " + " ".join(f"{n:>19s}" for n, _ in KEYS))
    for t in TASKS:
        A = [a for a in ARMS if (t, a) in C]
        if not A:
            continue
        dur = np.concatenate([col(t, a, "duration_s") for a in A])
        rea = np.concatenate([col(t, a, "tcp_reach_mean") for a in A])
        out = []
        for _, k in KEYS:
            y = np.concatenate([col(t, a, k) for a in A])
            if rate:
                y = y / dur
            out.append(f"dur{pear(dur,y):+.2f} rch{pear(rea,y):+.2f}")
        print(f"{t:8s} {dur.size:4d} " + " ".join(f"{o:>19s}" for o in out))
print("\nAn INTEGRAL over the episode correlates with duration by construction, so the")
print("PER SECOND block is the fair test of whether the column is still 'duration x reach'.")
print("\nENERGY_AUDIT.md 3.4 measured r(old, duration) = +0.85 egg / +0.49 spoon /")
print("+0.92 coke / +0.71 drawer, and 1.x measured r(old, reach) up to +0.97.")
print("A new column is only an improvement if BOTH of its numbers are materially lower.")

# --------------------------------------------------- paired per-second CIs
print("\n" + "=" * 132)
print("PAIRED ON THE SHARED SCENE, PER SECOND. ratio vs lat0, median [95% bootstrap CI].")
print("A CI containing 1.00 means the cell is indistinguishable from the ideal arm.")
print("=" * 132)
for t in TASKS:
    print(f"\n--- {t}")
    print(f"{'arm':12s} {'succ':>6} | {'OLD/s':>24} | {'drive/s':>24} | "
          f"{'sustained/s':>24} | {'contact impulse/s':>24}")
    for a in ARMS:
        if (t, a) not in C or a == "lat0":
            continue
        s = C[(t, a)]
        cells = []
        for k in ("t2_qf_tick", "t2_drive_arm", "t2_drive_arm_sus", "contact_impulse_Ns"):
            v, lo, hi = paired_ratio(t, a, k, per_s=True)
            star = " " if (lo <= 1.0 <= hi) else "*"
            cells.append(f"{v:6.2f} [{lo:5.2f},{hi:6.2f}]{star}")
        print(f"{a:12s} {100*s['n_success']/s['n_episodes']:5.1f}% | " + " | ".join(f"{c:>24}" for c in cells))
print("\n* = the 95% CI excludes 1.00.")
