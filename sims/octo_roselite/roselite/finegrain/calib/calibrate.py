#!/usr/bin/env python3
r"""Calibrate the simulated actuator-energy proxy into electrical joules and watts.

READ-ONLY on everything outside calib/.  Writes only
  calib/calibrated_curated.tsv
  calib/calibrated_plane.tsv
  calib/calibration.json
and prints the tables in ENERGY_CALIBRATION.md.

    python3 calib/calibrate.py            # curated set + plane
    python3 calib/calibrate.py --curated  # curated only (fast)

WHAT IS BEING CALIBRATED
------------------------
The sweep reports  t2_drive_arm_sus = INT Sum_i tau_i^2 dt  [N^2 m^2 s], the PD
drive torque PhysX applies to the arm joints (ENERGY_AUDIT.md / trace_eval2.py).
The decomposition asked for is

    E_electrical  =  W_mech/eta  +  (R/Kt^2) * INT Sum tau^2 dt  +  P_idle * T

and the middle term is exactly what the N^2 m^2 s channel measures -- ONCE a
per-joint R/Kt^2 is supplied.  This script supplies it, from ROBOTIS' published
stall-torque/stall-current pairs, and then applies three models:

  M1 "sim-literal"   -- convert t2_drive_arm_sus at face value.  Reported so the
                        reader can see it, and REJECTED: it returns 10-1000 kW,
                        because the simulated PD torque is 13-107x the torque a
                        WidowX servo can produce (see the tau_drv/tau_qf column).
  M2 "supply-bounded"-- the same, clipped at the arm's 12 V x 5 A rail.  A real
                        Dynamixel handed an impossible torque does not dissipate
                        impossible power; it saturates, draws its limit, and
                        fails to track.  This is the UPPER bracket.
  M3 "quasi-static"  -- the copper loss of the gravity+Coriolis torque the same
                        executed trajectory actually requires, from the
                        t2_qf_sub_arm channel (proved to be exactly g(q)+C(q,qd)qd
                        in ENERGY_AUDIT.md 1.1).  Blind to contact and to the
                        tracking torque, so this is the LOWER bracket.

M3 <= truth <= M2.  Both are reported per cell.

Every constant is in CONST below with its source.  Nothing here is fitted.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CURATED = BASE / "traces_torque3"
PLANE = BASE / "g5grid" / "plane3_runs"

# =====================================================================
# CONSTANTS.  Source tag: [ROBOTIS]=official eManual, [TROSSEN]=official
# Interbotix docs, [RETAIL]=retailer mirror of a Trossen spec sheet (not on any
# official page), [DERIVED]=computed here from cited numbers, [ASSUMED]=not
# sourced anywhere, stated as an assumption.
# =====================================================================

V_BUS = 12.0            # [ROBOTIS] recommended operating voltage of both parts
                        # https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/
                        # https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/

# (tau_stall N.m, I_stall A, I_standby A) at 12.0 V.
SERVO = {
    # [ROBOTIS] XM430-W350 @12.0V: 4.1 N.m / 2.3 A ; standby 40 mA ; 353.5:1 ; coreless
    "XM430-W350": dict(tau_stall=4.1, i_stall=2.3, i_standby=0.040, gear=353.5),
    # [ROBOTIS] XL430-W250 @12.0V: 1.5 N.m / 1.4 A ; standby 52 mA ; 258.5:1 ; cored
    "XL430-W250": dict(tau_stall=1.5, i_stall=1.4, i_standby=0.052, gear=258.5),
}

# [TROSSEN] wx250s servo complement, per joint, with the shadowed (dual) joints.
# https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html
# Joint order matches drive_config.json's joint_names for the 8-dof widowx model;
# ARM = the first 6 (trace_eval2.py _JG = {8: slice(0,6)}).
WX250S = [
    ("waist",         "XM430-W350", 1),
    ("shoulder",      "XM430-W350", 2),   # IDs 2,3 shadowed
    ("elbow",         "XM430-W350", 2),   # IDs 4,5 shadowed
    ("forearm_roll",  "XM430-W350", 1),
    ("wrist_angle",   "XM430-W350", 1),
    ("wrist_rotate",  "XL430-W250", 1),
    # not in ARM, but powered, so it counts toward standby:
    ("gripper",       "XL430-W250", 1),
]

# [RETAIL] 12 V / 5 A shipped PSU.  Corroborated by three retailer mirrors of the
# Trossen spec sheet (tribotix.com, mgsuperlabs.com, wevolver.com); NOT stated on
# any docs.trossenrobotics.com page.  Treated as the hard electrical ceiling.
P_SUPPLY_WX = 12.0 * 5.0        # 60 W

# [ASSUMED] gear-train + motor efficiency for the mechanical term.  ROBOTIS
# publishes no efficiency figure and no no-load current.  0.70 central, [0.5,0.85].
ETA_GEAR = 0.70
ETA_GEAR_LO, ETA_GEAR_HI = 0.85, 0.50     # hi eta -> lo energy

# [ASSUMED, from ENERGY_AUDIT.md 4.3's reading of the eManual performance-graph
# JPEG] no-load (friction) current ~0.20 A per XM430 while moving.  ROBOTIS
# publishes no no-load-current row for any X-series part; this is the single
# largest unsourced term in the budget and is carried only in the HIGH bound.
I_NOLOAD_HI = 0.20

# [THIRD PARTY] Oaki et al., arXiv:2605.15949, joint-level viscous friction for
# XM430-W350-driven joints: FV in [0.0287, 0.317] N.m.s/rad.  At the qvel seen
# here (rms 0.06-0.6 rad/s) FV*w^2 <= 0.12 W per joint -> negligible, and it is
# folded into ETA_GEAR rather than carried separately.
FV_MAX = 0.317

# [THIRD PARTY, disagreeing] the same paper states 2.3179 N.m/A as the
# current-to-torque constant for the XM430-W350, 30% above the stall-pair ratio
# derived below, with no stated definition.  Carried as the LOW bound on copper.
KT_OAKI_XM430 = 2.3179

# ---------------------------------------------------------------------
# google_robot.  Everyday Robots published NO actuator specification: RT-1
# (arXiv:2212.06817) describes the hardware in one clause, the SimplerEnv URDF
# sets effort="10.0" uniformly on every joint including the wheels, and there are
# no <transmission>/<actuator> tags.  A PROXY is therefore used and every google
# number below inherits the proxy's uncertainty.  See ENERGY_CALIBRATION.md 6.
# ---------------------------------------------------------------------
GOOGLE_PROXY = "Harmonic Drive FHA-14C-100 joint module (Kt, R) + Franka Panda / " \
               "Kinova Gen3 whole-arm power envelope (P_idle, P_cap)"
# [HARMONIC DRIVE] FHA-C mini 24 V spec sheet, FHA-14C at 100:1 --
# https://www.harmonicdrive.net/_hd/content/catalogs/pdf/fha-c-miniature-spec-sheet-24v.pdf
#   Kt = 2.9 N.m/A, and the datasheet's Note 3 says it is OUTPUT-referenced
#        ("considering the efficiency of the gear") -- the same convention as the
#        WidowX numbers derived above, so the two are directly comparable.
#   R  = 0.07 Ohm phase resistance at 20 C.  The datasheet does NOT state whether
#        this is phase-to-phase or phase-to-neutral, which is a factor-of-2 on the
#        copper term; the central value below takes the 3-phase sinusoidal form
#        P = 1.5 * I^2 * R_phase, and the [lo, hi] band spans both conventions.
#   rated / max output torque 6.8 / 28 N.m, 24 VDC, 1.2 kg.
# This is the ONLY commercially published robot joint module found that gives Kt
# AND R AND gear ratio in one official document; Kinova, Franka and UR all publish
# system power only and no motor constants.
KT_FHA14C = 2.9
R_FHA14C = 0.07
C_GOOGLE = 1.5 * R_FHA14C / KT_FHA14C ** 2      # 0.0125 W per (N.m)^2
# [FRANKA / KINOVA] whole-arm electrical envelope for a 7-dof collaborative arm.
#   Franka Panda datasheet: arm typical ~60 W, arm max ~350 W (+80 W control box)
#     https://download.franka.de/Datasheet-EN.pdf
#   Kinova Gen3 User Guide R07: 24 VDC, external supply rated 300 W (a ceiling,
#     not a measured draw)  https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
#   A third-party reseller mirror quotes Gen3 idle 25 W / average 36 W / peak
#     155 W; NOT found in Kinova's own docs -> used only to set the low bound.
P_IDLE_GOOGLE = 40.0            # [PROXY] W, in [25, 60]
P_SUPPLY_GOOGLE = 350.0         # [PROXY] W ceiling (Franka Panda arm max)

# =====================================================================


def servo_table(complement):
    """Per-joint derived electrical constants.

    K   = tau_stall / I_stall            output-referenced torque constant  [N.m/A]
    R   = V_BUS / I_stall                total series resistance at stall   [Ohm]
    c   = R / (n * K^2)                  copper coefficient on JOINT torque [W/(N.m)^2]
    Pmx = n * V_BUS * I_stall            that joint's stall dissipation     [W]

    The stall condition closes the loop: at stall omega=0, so the whole supply
    goes into the winding, V*I = I^2*R with R = V/I identically.  Substituting
    I = tau/K gives P_copper = (tau/K)^2 * R exactly, and at tau = tau_stall it
    returns V*I_stall.  For a joint with n shadowed servos each carrying tau/n:
    P = n * ((tau/n)/K)^2 * R = (R/(n K^2)) tau^2.
    """
    rows = []
    for name, part, n in complement:
        s = SERVO[part]
        K = s["tau_stall"] / s["i_stall"]
        R = V_BUS / s["i_stall"]
        rows.append(dict(joint=name, part=part, n=n, tau_stall=s["tau_stall"],
                         i_stall=s["i_stall"], K=K, R=R, c=R / (n * K * K),
                         p_max=n * V_BUS * s["i_stall"],
                         tau_joint_max=n * s["tau_stall"],
                         i_standby=s["i_standby"]))
    return rows


WX_ROWS = servo_table(WX250S)
WX_ARM = WX_ROWS[:6]                                   # ARM = slice(0,6)
P_STANDBY_WX = V_BUS * sum(r["n"] * r["i_standby"] for r in WX_ROWS)   # all 9 servos

# Per-joint share of the arm's INT tau^2 dt, measured on the per-joint arrays in
# smoke_torque3/ (only place they survive; the sweep keeps arm-summed scalars).
# Used to collapse the 6 per-joint c_i into one scalar c-bar for each channel.
#   drive channel : waist .21-.28  shoulder .39-.43  elbow .06-.09
#                   forearm .10-.11  wrist_angle .10  wrist_rotate .06-.08
#   qf channel    : shoulder .67-.75  elbow .23-.33  everything else <1%
SHARE_DRIVE_WX = np.array([0.243, 0.410, 0.074, 0.105, 0.100, 0.068])
SHARE_QF_WX = np.array([0.005, 0.711, 0.282, 0.001, 0.001, 0.000])


def cbar(rows, share):
    c = np.array([r["c"] for r in rows])
    return float((c * share).sum() / share.sum())


CBAR_DRIVE_WX = cbar(WX_ARM, SHARE_DRIVE_WX)
CBAR_QF_WX = cbar(WX_ARM, SHARE_QF_WX)

# Uncertainty on the copper coefficient.
#   LOW : R is an upper bound (V/I_stall lumps in the H-bridge drop and wiring);
#         if half the stall drop is in the driver the copper term halves.  AND
#         Oaki's larger Kt cuts it by (1.7826/2.3179)^2 = 0.591.
#   HIGH: the derived value itself.
CU_LO = 0.5 * (SERVO["XM430-W350"]["tau_stall"] / SERVO["XM430-W350"]["i_stall"]
               / KT_OAKI_XM430) ** 2
CU_HI = 1.0


def p_idle_wx(hi=False):
    """Quiescent electrical floor of the arm, W."""
    if not hi:
        return P_STANDBY_WX
    n_moving = sum(r["n"] for r in WX_ARM)          # 8 servos on the 6 arm joints
    return P_STANDBY_WX + V_BUS * I_NOLOAD_HI * n_moving


# =====================================================================
# loaders -- read only
# =====================================================================
CUR_ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300",
            "pipe200fix", "serial283", "fp32_555", "cpu685"]
CUR_LABEL = {"lat0": "ideal", "pipe110fix": "pipe110", "p105w300": "p105w300",
             "p130w275": "p130w275", "p150w300": "p150w300",
             "pipe200fix": "pipe200fix", "serial283": "serial283",
             "fp32_555": "fp32_555", "cpu685": "cpu685"}
TASKS = [("egg", "widowx"), ("spoon", "widowx"), ("coke", "google"),
         ("drawer", "google")]


def load_curated():
    D, DIR = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in glob.glob(str(CURATED / "*" / "energy2.json")):
        m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", Path(f).parent.name)
        if m:
            k, sd = (m.group(1), m.group(2)), int(m.group(3) or 100)
            D[k][sd] = json.load(open(f))
            DIR[k][sd] = Path(f).parent
    return D, DIR


def load_plane():
    D = collections.defaultdict(dict)
    for f in glob.glob(str(PLANE / "*" / "*" / "energy2.json")):
        m = re.match(r"(egg|spoon|coke|drawer)_(g\d+_\d+)_rng(\d+)$", Path(f).parent.name)
        if m:
            D[(m.group(1), m.group(2))][int(m.group(3))] = json.load(open(f))
    return D


# =====================================================================
# the calibration itself
# =====================================================================
# Arm torque-capacity vector, for the feasibility test.  ||tau_cap|| is the
# largest arm-vector torque magnitude the six joints can produce simultaneously,
# so Sum tau_i^2 > ||tau_cap||^2 is a SUFFICIENT condition for "no WidowX 250 S
# can produce this", independent of how the torque is distributed.
TAU_CAP_WX = float(np.sqrt(sum(r["tau_joint_max"] ** 2 for r in WX_ARM)))   # 13.68 N.m
# google: the SimplerEnv URDF's uniform effort="10.0" is a placeholder, so the
# only honest capacity number is the PROXY's.  See ENERGY_CALIBRATION.md 6.
# [PROXY] arm-vector capacity: 7 joints, Kinova Gen3's published output torques
# (large joints 39 N.m, small joints 13 N.m nominal) -> sqrt(4*39^2 + 3*13^2).
TAU_CAP_GOOGLE = float(np.sqrt(4 * 39.0 ** 2 + 3 * 13.0 ** 2))     # 84.5 N.m


def load_series(cell_dir):
    """Per-actuation Sum_arm tau_drive^2 (the _sus variant), if series.npz exists.

    Verified against the scalars: sum_k tau2s[k] * act_dt reproduces
    energy2.json's t2_drive_arm_sus to 1e-6 relative on every episode checked.
    Only the curated set carries series.npz; the 44-arm plane does not.
    """
    f = Path(cell_dir) / "series.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {int(k.split("_")[1]): z[k].astype(np.float64)
            for k in z.files if k.startswith("tau2s_")}


def _params(emb, bound):
    if emb == "widowx":
        cu = CU_LO if bound == "lo" else CU_HI
        return dict(c_drv=CBAR_DRIVE_WX * cu, c_qf=CBAR_QF_WX * cu,
                    p_idle=p_idle_wx(hi=(bound == "hi")), p_cap=P_SUPPLY_WX,
                    tau_cap=TAU_CAP_WX)
    # google band: c spans the two phase-resistance conventions (x0.67 .. x1.33)
    # and P_idle spans Kinova's contested 25 W idle .. Franka's 60 W typical.
    kc = 0.67 if bound == "lo" else (1.33 if bound == "hi" else 1.0)
    ki = 25.0 / 40.0 if bound == "lo" else (60.0 / 40.0 if bound == "hi" else 1.0)
    return dict(c_drv=C_GOOGLE * kc, c_qf=C_GOOGLE * kc, p_idle=P_IDLE_GOOGLE * ki,
                p_cap=P_SUPPLY_GOOGLE, tau_cap=TAU_CAP_GOOGLE)


def calibrate_episode(e, emb, bound="mid", ser=None, act_dt=None):
    """One episode -> the three models, in joules.

    `ser` is the per-actuation Sum tau^2 array for this episode (series.npz).
    When it is present M2 clips the demand ACTUATION BY ACTUATION, which is the
    finest resolution available; when it is absent M2 falls back to clipping the
    episode mean, which is a strict UPPER bound on the per-actuation result
    (Jensen: min() is concave, so clipping the mean >= mean of the clipped).
    """
    P = _params(emb, bound)
    T = e["duration_s"]
    eta = ETA_GEAR_LO if bound == "lo" else (ETA_GEAR_HI if bound == "hi" else ETA_GEAR)

    e_cu_drv = P["c_drv"] * e["t2_drive_arm_sus"]           # J
    e_mech = e["work_abs_arm"] / eta                        # J
    e_idle = P["p_idle"] * T                                # J
    m1 = e_cu_drv + e_mech + e_idle
    p_mech = e_mech / max(T, 1e-9)

    if ser is not None and ser.size:
        adt = act_dt if act_dt else T / ser.size
        p_k = P["c_drv"] * ser + p_mech + P["p_idle"]
        m2 = float(np.minimum(p_k, P["p_cap"]).sum() * adt)
        duty = float((p_k > P["p_cap"]).mean())
        infeas = float((ser > P["tau_cap"] ** 2).mean())
        m2_res = "per-actuation"
    else:
        m2 = min(m1 / max(T, 1e-9), P["p_cap"]) * T
        duty = float(m1 / max(T, 1e-9) > P["p_cap"])
        infeas = float(np.sqrt(e["t2_drive_arm_sus"] / max(T, 1e-9)) > P["tau_cap"])
        m2_res = "episode-mean (upper bound)"

    m3 = P["c_qf"] * e["t2_qf_sub_arm"] + e_idle

    return dict(T=T, m1=m1, m2=m2, m3=m3, duty=duty, infeas=infeas, _m2res=m2_res,
                p1=m1 / max(T, 1e-9), p2=m2 / max(T, 1e-9), p3=m3 / max(T, 1e-9),
                e_cu_drv=e_cu_drv, e_mech=e_mech, e_idle=e_idle,
                t2=e["t2_drive_arm_sus"], t2qf=e["t2_qf_sub_arm"],
                tau_rms=np.sqrt(e["t2_drive_arm_sus"] / max(T, 1e-9)),
                tau_qf_rms=np.sqrt(e["t2_qf_sub_arm"] / max(T, 1e-9)))


KEYS = ("t2", "m1", "m2", "m3", "T", "p1", "p2", "p3", "duty", "infeas",
        "e_cu_drv", "e_mech", "e_idle", "tau_rms", "tau_qf_rms")
# Channels that need series.npz.  It exists for seed 100 and 110-129 but NOT for
# 101-109, so these are reduced over the series-bearing seeds ONLY -- mixing the
# per-actuation clip with the episode-mean fallback inside one cell would bias
# the 10-seed arms (1 series seed) against the 30-seed arms (21).
SERIES_KEYS = ("m2", "p2", "duty", "infeas")


def reduce_cell(seeds, emb, bound="mid", dirs=None):
    """fig_metrics3.py's reduction: per-seed MEDIAN episode, then mean over seeds.

    Applied identically to the raw N^2 m^2 s channel and to every calibrated
    model, so the ratios are comparable line for line.  `dirs` maps seed ->
    cell directory so series.npz is picked up where it exists.
    """
    out = {k: [] for k in KEYS}
    out_ser = {k: [] for k in SERIES_KEYS}
    n_ser = 0
    for seed, d in seeds.items():
        S = load_series(dirs[seed]) if dirs else None
        adt = d.get("act_ms", 0.0) / 1000.0 or None
        rows = [calibrate_episode(e, emb, bound, None, adt) for e in d["episodes"]]
        for k in KEYS:
            out[k].append(float(np.median([r[k] for r in rows])))
        if S:
            n_ser += 1
            rs = [calibrate_episode(e, emb, bound, S.get(e["ep"]), adt)
                  for e in d["episodes"]]
            for k in SERIES_KEYS:
                out_ser[k].append(float(np.median([r[k] for r in rs])))
    res = {}
    for k in KEYS:
        res[k] = float(np.mean(out[k]))
        res[k + "_seeds"] = out[k]
    # M2 without series is the episode-mean clip: a strict upper bound on the
    # per-actuation clip (min() is concave -> min(mean) >= mean(min)).  Keep it
    # under its own name and let M2 be the resolved one where it exists.
    res["m2_epmean"] = res["m2"]
    res["p2_epmean"] = res["p2"]
    if n_ser:
        for k in SERIES_KEYS:
            res[k] = float(np.mean(out_ser[k]))
            res[k + "_seeds"] = out_ser[k]
        res["m2_resolution"] = f"per-actuation, {n_ser} of {len(seeds)} seeds"
    else:
        res["m2_resolution"] = f"episode-mean UPPER BOUND, 0 of {len(seeds)} seeds"
    res["n_m2_seeds"] = n_ser
    res["n_seeds"] = len(seeds)
    res["n_ep"] = sum(d["n_episodes"] for d in seeds.values())
    res["success"] = float(np.mean([100 * d["n_success"] / d["n_episodes"]
                                    for d in seeds.values()]))
    res["sat"] = float(np.mean([np.median([e["sat_frac_arm"] for e in d["episodes"]])
                                for d in seeds.values()]))
    return res


def boot_ci(v, B=4000, seed=0):
    if len(v) < 2:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    a = np.asarray(v, float)
    m = [r.choice(a, a.size, replace=True).mean() for _ in range(B)]
    return tuple(float(x) for x in np.percentile(m, [2.5, 97.5]))


# =====================================================================
# reporting
# =====================================================================
def print_constants():
    print("=" * 100)
    print("ACTUATOR MODEL -- WidowX 250 S (wx250s), 12.0 V")
    print("=" * 100)
    print(f"{'joint':14s} {'servo':13s} {'n':>2s} {'tau_st':>7s} {'I_st':>6s} "
          f"{'K=t/I':>7s} {'R=V/I':>7s} {'c=R/(nK^2)':>11s} {'P_stall':>8s} {'tau_max':>8s}")
    print(f"{'':14s} {'':13s} {'':>2s} {'[N.m]':>7s} {'[A]':>6s} "
          f"{'[N.m/A]':>7s} {'[Ohm]':>7s} {'[W/(N.m)^2]':>11s} {'[W]':>8s} {'[N.m]':>8s}")
    for r in WX_ROWS:
        tag = "" if r in WX_ARM else "   (not in ARM; counts for standby only)"
        print(f"{r['joint']:14s} {r['part']:13s} {r['n']:2d} {r['tau_stall']:7.2f} "
              f"{r['i_stall']:6.2f} {r['K']:7.4f} {r['R']:7.4f} {r['c']:11.4f} "
              f"{r['p_max']:8.1f} {r['tau_joint_max']:8.2f}{tag}")
    print()
    print(f"  standby floor, all 9 servos torque-disabled : {P_STANDBY_WX:.3f} W  [ROBOTIS]")
    print(f"  + no-load current 0.20 A x 8 arm servos     : {p_idle_wx(hi=True):.3f} W  [ASSUMED]")
    print(f"  shipped PSU ceiling                         : {P_SUPPLY_WX:.1f} W  [RETAIL]")
    print()
    print(f"  c-bar on the DRIVE channel  (t2_drive_arm_sus) : {CBAR_DRIVE_WX:.4f} W/(N.m)^2")
    print(f"  c-bar on the QF    channel  (t2_qf_sub_arm)    : {CBAR_QF_WX:.4f} W/(N.m)^2")
    print(f"  copper uncertainty multiplier                  : [{CU_LO:.3f}, {CU_HI:.3f}]")
    print(f"  gear/motor efficiency (ASSUMED)                : {ETA_GEAR:.2f} "
          f"in [{ETA_GEAR_HI:.2f}, {ETA_GEAR_LO:.2f}]")
    print()
    print("  google_robot (Everyday Robots): NO published actuator specification of any")
    print("  kind (papers, archived site, press, patents all checked -- see")
    print("  ENERGY_CALIBRATION.md 6).  PROXY:")
    print(f"    {GOOGLE_PROXY}")
    print(f"    Kt = {KT_FHA14C} N.m/A (output-referenced), R = {R_FHA14C} Ohm/phase")
    print(f"    -> c = 1.5 R/Kt^2 = {C_GOOGLE:.5f} W/(N.m)^2   "
          f"(2000x smaller than the WidowX's, because a 24 V harmonic-drive joint")
    print( "       is far more efficient than a 12 V hobby servo)")
    print(f"    P_idle = {P_IDLE_GOOGLE:.0f} W in [25, 60]; cap = {P_SUPPLY_GOOGLE:.0f} W; "
          f"||tau_cap|| = {TAU_CAP_GOOGLE:.1f} N.m")
    print()


def curated_report(D, DIR, out_tsv):
    lines = ["task\tarm\tn_seeds\tn_ep\tsuccess%\tdur_s\ttau_drv_rms_Nm\ttau_qf_rms_Nm\t"
             "infeas_frac\tsupply_duty\tt2_N2m2s\tt2_ratio\tM1_J\tM1_W\tM2_J\tM2_W\tM2_ratio\t"
             "M3_J\tM3_W\tM3_ratio\tM3_lo_J\tM3_hi_J\tM2_lo_J\tM2_hi_J\tsat_frac"]
    summary = {}
    for task, emb in TASKS:
        arms = [a for a in CUR_ARMS if (task, a) in D]
        red = {a: reduce_cell(D[(task, a)], emb, "mid", DIR[(task, a)]) for a in arms}
        rlo = {a: reduce_cell(D[(task, a)], emb, "lo", DIR[(task, a)]) for a in arms}
        rhi = {a: reduce_cell(D[(task, a)], emb, "hi", DIR[(task, a)]) for a in arms}
        b = red["lat0"]
        print("=" * 150)
        print(f"{task}  ({emb})   baseline = lat0 (\"ideal\"), the reference "
              f"fig_metrics3.py uses.")
        print(f"  M2 / infeas / duty use series.npz and are reduced over the "
              f"series-bearing seeds only: " +
              ", ".join(f"{CUR_LABEL[a]}={red[a]['n_m2_seeds']}/{red[a]['n_seeds']}"
                        for a in arms))
        print("=" * 150)
        print(f"{'arm':11s} {'sd':>3s} {'succ%':>6s} {'dur':>6s} {'tauD':>7s} {'tauQF':>6s} "
              f"{'infeas':>6s} {'duty':>5s} | {'t2 [N2m2s]':>11s} {'t2 x':>7s} | "
              f"{'M1 [J]':>10s} {'M1 [W]':>9s} | {'M2 [J]':>8s} {'M2 [W]':>7s} {'M2 x':>6s} | "
              f"{'M3 [J]':>8s} {'M3 [W]':>7s} {'M3 x':>6s} {'M3 range [J]':>15s}")
        for a in arms:
            r, rl, rh = red[a], rlo[a], rhi[a]
            print(f"{CUR_LABEL[a]:11s} {r['n_seeds']:3d} {r['success']:6.1f} {r['T']:6.2f} "
                  f"{r['tau_rms']:7.1f} {r['tau_qf_rms']:6.2f} {r['infeas']:6.3f} "
                  f"{r['duty']:5.3f} | {r['t2']:11.4g} {r['t2']/b['t2']:7.3f} | "
                  f"{r['m1']:10.4g} {r['p1']:9.4g} | "
                  f"{r['m2']:8.1f} {r['p2']:7.2f} {r['m2']/b['m2']:6.3f} | "
                  f"{r['m3']:8.1f} {r['p3']:7.2f} {r['m3']/b['m3']:6.3f} "
                  f"{'[%.0f, %.0f]' % (rl['m3'], rh['m3']):>15s}")
            lines.append("\t".join([
                task, CUR_LABEL[a], str(r["n_seeds"]), str(r["n_ep"]),
                f"{r['success']:.2f}", f"{r['T']:.3f}", f"{r['tau_rms']:.3f}",
                f"{r['tau_qf_rms']:.4f}", f"{r['infeas']:.4f}", f"{r['duty']:.4f}",
                f"{r['t2']:.6g}", f"{r['t2']/b['t2']:.4f}",
                f"{r['m1']:.6g}", f"{r['p1']:.6g}", f"{r['m2']:.4f}", f"{r['p2']:.4f}",
                f"{r['m2']/b['m2']:.4f}", f"{r['m3']:.4f}", f"{r['p3']:.4f}",
                f"{r['m3']/b['m3']:.4f}", f"{rl['m3']:.4f}", f"{rh['m3']:.4f}",
                f"{rl['m2']:.4f}", f"{rh['m2']:.4f}", f"{r['sat']:.4f}"]))
        # --- ordering, with the CI-overlap test that decides whether a swap is real
        def order(key):
            return sorted(arms, key=lambda a: red[a][key])
        o = {k: order(k) for k in ("t2", "m2", "m3")}
        for k in ("m2", "m3"):
            same = o[k] == o["t2"]
            # a swap only counts if the two cells' bootstrap CIs are disjoint
            hard = []
            for x, y in zip(o["t2"], o[k]):
                if x != y:
                    cx, cy = boot_ci(red[x][k + "_seeds"]), boot_ci(red[y][k + "_seeds"])
                    if cx[1] < cy[0] or cy[1] < cx[0]:
                        hard.append((CUR_LABEL[x], CUR_LABEL[y]))
            print(f"  ordering {k.upper()} vs t2: "
                  f"{'IDENTICAL' if same else 'reordered'}"
                  f"{'' if same else ('  -- but every swap is inside the seed CIs'
                                      if not hard else '  -- CI-SEPARATED SWAPS: '
                                      + str(hard))}")
        print(f"  spread (max/min)   t2 = {red[o['t2'][-1]]['t2']/red[o['t2'][0]]['t2']:7.2f}x"
              f"   M2 = {red[o['m2'][-1]]['m2']/red[o['m2'][0]]['m2']:5.2f}x"
              f"   M3 = {red[o['m3'][-1]]['m3']/red[o['m3'][0]]['m3']:5.2f}x")
        worst = o["t2"][-1]
        for k, lab in (("m2", "M2"), ("m3", "M3")):
            lo, hi = boot_ci(red[worst][k + "_seeds"])
            print(f"  {lab} ratio, {CUR_LABEL[worst]} / ideal: "
                  f"{red[worst][k]/b[k]:.3f}  [{lo/b[k]:.3f}, {hi/b[k]:.3f}]   "
                  f"(uncalibrated t2 ratio {red[worst]['t2']/b['t2']:.2f}x)")
        print()
        summary[task] = {}
        for a in arms:
            d = {k: red[a][k] for k in KEYS}
            d.update(success=red[a]["success"], n_seeds=red[a]["n_seeds"],
                     n_ep=red[a]["n_ep"], sat=red[a]["sat"],
                     m3_lo=rlo[a]["m3"], m3_hi=rhi[a]["m3"],
                     m2_lo=rlo[a]["m2"], m2_hi=rhi[a]["m2"],
                     m2_epmean=red[a]["m2_epmean"], n_m2_seeds=red[a]["n_m2_seeds"],
                     m2_resolution=red[a]["m2_resolution"],
                     t2_ratio=red[a]["t2"] / b["t2"], m2_ratio=red[a]["m2"] / b["m2"],
                     m3_ratio=red[a]["m3"] / b["m3"])
            summary[task][CUR_LABEL[a]] = d
    out_tsv.write_text("\n".join(lines) + "\n")
    print(f"[ok] {out_tsv}")
    return summary


def plane_report(D, out_tsv):
    if not D:
        print("[skip] no plane cells found under", PLANE)
        return {}
    print()
    print("=" * 100)
    print("THE 44-ARM PLANE.  No series.npz here, so M2 is the EPISODE-MEAN clip --")
    print("a strict UPPER BOUND on the per-actuation M2 (min() is concave).  On the two")
    print("widowx tasks the episode mean is 400x over the 60 W rail on every arm, so that")
    print("bound saturates at P_supply * T and carries no information beyond duration;")
    print("M3 is the usable column there.  On google both are informative.")
    print("=" * 100)
    lines = ["task\tarm\tn_seeds\tn_ep\tsuccess%\tdur_s\tt2_N2m2s\tM2ub_J\tM2ub_W\tM3_J\tM3_W"]
    summary = {}
    for task, emb in TASKS:
        arms = sorted(a for (t, a) in D if t == task)
        if not arms:
            continue
        red = {a: reduce_cell(D[(task, a)], emb) for a in arms}   # no series.npz on the plane
        m3 = np.array([red[a]["m3"] for a in arms])
        t2 = np.array([red[a]["t2"] for a in arms])
        m2 = np.array([red[a]["m2"] for a in arms])
        rho_t2_m3 = float(np.corrcoef(np.argsort(np.argsort(t2)),
                                      np.argsort(np.argsort(m3)))[0, 1])
        rho_t2_m2 = float(np.corrcoef(np.argsort(np.argsort(t2)),
                                      np.argsort(np.argsort(m2)))[0, 1])
        print(f"plane {task:7s}  {len(arms):2d} arms  "
              f"t2 spread {t2.max()/t2.min():7.2f}x   "
              f"M2ub spread {m2.max()/m2.min():5.2f}x  M3 spread {m3.max()/m3.min():5.2f}x   "
              f"Spearman(t2,M2)={rho_t2_m2:+.3f}  Spearman(t2,M3)={rho_t2_m3:+.3f}")
        print(f"        M3 watts across the plane: "
              f"{min(red[a]['p3'] for a in arms):.2f} - {max(red[a]['p3'] for a in arms):.2f} W ; "
              f"M2ub watts: {min(red[a]['p2'] for a in arms):.2f} - "
              f"{max(red[a]['p2'] for a in arms):.2f} W")
        for a in arms:
            r = red[a]
            lines.append("\t".join([task, a, str(r["n_seeds"]), str(r["n_ep"]),
                                    f"{r['success']:.2f}", f"{r['T']:.3f}",
                                    f"{r['t2']:.6g}", f"{r['m2']:.4f}", f"{r['p2']:.4f}",
                                    f"{r['m3']:.4f}", f"{r['p3']:.4f}"]))
        summary[task] = dict(n_arms=len(arms),
                             spearman_t2_m2=rho_t2_m2, spearman_t2_m3=rho_t2_m3,
                             t2_spread=float(t2.max() / t2.min()),
                             m2_spread=float(m2.max() / m2.min()),
                             m3_spread=float(m3.max() / m3.min()),
                             p3_min=min(red[a]["p3"] for a in arms),
                             p3_max=max(red[a]["p3"] for a in arms),
                             p2_min=min(red[a]["p2"] for a in arms),
                             p2_max=max(red[a]["p2"] for a in arms))
    out_tsv.write_text("\n".join(lines) + "\n")
    print(f"[ok] {out_tsv}")
    return summary


def sanity(D, DIR):
    """Bracket the WidowX 250 S's real power envelope and show where we land."""
    print("=" * 100)
    print("SANITY CHECK -- does the calibration land inside a WidowX 250 S's envelope?")
    print("=" * 100)
    # gravity-hold anchor, straight off the qf channel (proved = g(q)+C(q,qd)qd)
    eps = [e for a in CUR_ARMS if ("egg", a) in D
           for d in D[("egg", a)].values() for e in d["episodes"]]
    tq = np.array([np.sqrt(e["t2_qf_sub_arm"] / max(e["duration_s"], 1e-9)) for e in eps])
    td = np.array([np.sqrt(e["t2_drive_arm_sus"] / max(e["duration_s"], 1e-9)) for e in eps])
    print(f"  gravity+Coriolis torque, arm-vector rms, {len(eps)} eggplant episodes: "
          f"{np.percentile(tq,5):.2f} - {np.percentile(tq,95):.2f} N.m "
          f"(median {np.median(tq):.2f})")
    print(f"  PD drive torque,         arm-vector rms, same episodes            : "
          f"{np.percentile(td,5):.1f} - {np.percentile(td,95):.1f} N.m "
          f"(median {np.median(td):.1f})")
    print(f"  ratio drive/quasi-static: median {np.median(td/tq):.0f}x   "
          f"-> the drive torque is {np.median(td/tq):.0f}x what a real WidowX "
          f"servo would have to make")
    print(f"  shoulder joint capacity (2 x XM430-W350 stall): "
          f"{WX_ARM[1]['tau_joint_max']:.1f} N.m")
    print()
    floors = [
        ("all 9 servos torque-disabled (standby)", P_STANDBY_WX),
        ("holding the arm's own weight (shoulder 1.36 + elbow 1.25 N.m rms,\n"
         "    measured on ep00_qf.npy; copper only)",
         CBAR_QF_WX * (1.363 ** 2 + 1.251 ** 2)),
        ("standby + gravity hold", P_STANDBY_WX + CBAR_QF_WX * (1.363**2 + 1.251**2)),
        ("+ 0.20 A no-load on 8 arm servos [ASSUMED]",
         p_idle_wx(hi=True) + CBAR_QF_WX * (1.363**2 + 1.251**2)),
        ("every arm servo at stall simultaneously (thermally impossible)",
         sum(r["p_max"] for r in WX_ARM)),
        ("shipped 12 V / 5 A PSU [RETAIL]", P_SUPPLY_WX),
    ]
    print("  plausible envelope:")
    for lab, w in floors:
        print(f"    {w:8.2f} W   {lab}")
    print()
    for task in ("egg", "spoon"):
        arms = [a for a in CUR_ARMS if (task, a) in D]
        red = {a: reduce_cell(D[(task, a)], "widowx", "mid", DIR[(task, a)])
               for a in arms}
        p3 = [red[a]["p3"] for a in arms]
        p2 = [red[a]["p2"] for a in arms]
        p1 = [red[a]["p1"] for a in arms]
        print(f"  {task:6s}  M2 lands at {min(p2):6.2f} - {max(p2):6.2f} W "
              f"(cap {P_SUPPLY_WX:.0f} W)")
        print(f"  {'':6s}  M3 lands at {min(p3):6.2f} - {max(p3):6.2f} W "
              f"-> INSIDE the envelope ({P_STANDBY_WX:.1f} .. {P_SUPPLY_WX:.0f} W): "
              f"{'YES' if max(p3) <= P_SUPPLY_WX and min(p3) >= P_STANDBY_WX else 'NO'}")
        print(f"          M1 lands at {min(p1):.3g} - {max(p1):.3g} W "
              f"-> {max(p1)/P_SUPPLY_WX:.0f}x the whole power supply. REJECTED.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", action="store_true", help="skip the 44-arm plane")
    a = ap.parse_args()
    print_constants()
    D, DIR = load_curated()
    print(f"[load] curated: {len(D)} (task,arm) cells, "
          f"{sum(len(v) for v in D.values())} seed-cells\n")
    cur = curated_report(D, DIR, HERE / "calibrated_curated.tsv")
    sanity(D, DIR)
    pl = {}
    if not a.curated:
        P = load_plane()
        print(f"[load] plane: {len(P)} (task,arm) cells, "
              f"{sum(len(v) for v in P.values())} seed-cells")
        pl = plane_report(P, HERE / "calibrated_plane.tsv")
    (HERE / "calibration.json").write_text(json.dumps(dict(
        constants=dict(
            V_BUS=V_BUS, servo=SERVO, wx250s=[dict(r) for r in WX_ROWS],
            P_STANDBY_WX=P_STANDBY_WX, P_SUPPLY_WX=P_SUPPLY_WX,
            CBAR_DRIVE_WX=CBAR_DRIVE_WX, CBAR_QF_WX=CBAR_QF_WX,
            CU_LO=CU_LO, ETA_GEAR=ETA_GEAR,
            google_proxy=GOOGLE_PROXY, C_GOOGLE=C_GOOGLE,
            P_IDLE_GOOGLE=P_IDLE_GOOGLE, P_SUPPLY_GOOGLE=P_SUPPLY_GOOGLE),
        curated=cur, plane=pl), indent=1))
    print(f"[ok] {HERE/'calibration.json'}")


if __name__ == "__main__":
    main()
