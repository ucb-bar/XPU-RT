#!/usr/bin/env python3
"""AUDIT PASS 12 -- how far can this be pushed toward watts?

Answer: exactly one physical quantity can be calibrated, and it is NOT the metric.
It is "the copper loss a real WidowX-250 S would dissipate holding the SIMULATED arm's
gravity/Coriolis torque". Everything else in the real power budget is either not
published or not in the traces.

SERVO DATA (all from the ROBOTIS e-Manual / Trossen docs; see ENERGY_AUDIT.md §4 for
URLs). wx250s = 7x XM430-W350 + 2x XL430-W250, with shoulder and elbow each DUAL
(a shadow servo mirrors the master via Secondary_ID in
interbotix_xsarm_control/config/wx250s.yaml), on a 12 V / 5 A supply.

  XM430-W350 @12.0V : stall 4.1 N.m at 2.3 A -> ROBOTIS prints 1.783 N.m/A
                      standby 40 mA, gear 353.5:1, coreless
  XL430-W250 @12.0V : stall 1.5 N.m at 1.4 A -> 1.071 N.m/A
                      standby 52 mA, gear 258.5:1, cored

ROBOTIS publishes NO winding resistance and NO motor Kt for any X-series part, and no
third party has measured them. R is therefore INFERRED as V_stall/I_stall, which lumps
in the H-bridge drop and wiring and is an UPPER bound on the winding:
  XM430: 12.0/2.3 = 5.22 ohm   (5.29 / 5.48 at 11.1 / 14.8 V -- consistent)
  XL430: 12.0/1.4 = 8.57 ohm

With K = tau_stall/I_stall (output-referenced, gear-inclusive) and R = V/I_stall, the
stall condition V*I = I^2*R closes, so
    P_copper(tau) = (tau/K)^2 * R
and for an n-servo joint sharing tau evenly,
    P_copper = n * ((tau/n)/K)^2 * R = (R / (n K^2)) * tau^2
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
C = json.load(open(HERE / "energy_audit_cache.json"))
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]

V = 12.0
XM = dict(K=1.783, I_stall=2.3, standby=0.040)
XL = dict(K=1.071, I_stall=1.4, standby=0.052)
XM["R"] = V / XM["I_stall"]; XL["R"] = V / XL["I_stall"]

# wx250s joints, in the qf column order. n = servos on the joint.
JOINTS = [("waist",        XM, 1), ("shoulder",     XM, 2), ("elbow",        XM, 2),
          ("forearm_roll", XM, 1), ("wrist_angle",  XM, 1), ("wrist_rotate", XL, 1)]
COEF = np.array([s["R"] / (n * s["K"] ** 2) for _, s, n in JOINTS])   # W per (N.m)^2
# quiescent floor: 7 XM standby + 2 XL standby, all servos powered, torque disabled
P_QUIESCENT = V * (7 * XM["standby"] + 2 * XL["standby"])

print("=" * 92)
print("Per-joint copper coefficient  c_i = R_i / (n_i Kt_i^2)   [W per (N.m)^2]")
print("=" * 92)
for (nm, s, n), c in zip(JOINTS, COEF):
    print(f"  {nm:<14} {'XM430-W350' if s is XM else 'XL430-W250':<12} n={n}  "
          f"K={s['K']:.3f} N.m/A  R={s['R']:.2f} ohm  ->  c = {c:.3f}")
print(f"\n  spread across the arm joints: {COEF.max()/COEF.min():.1f}x  "
      f"(the shipped metric sets every c_i to 1)")
print(f"  BUT shoulder and elbow carry 99.7% of the metric and share c = {COEF[1]:.3f},")
print(f"  which is why the reweighting of section 1.3 barely moves the widowx ranking.")
print(f"\n  quiescent floor, 9 servos at published standby, torque DISABLED: "
      f"{P_QUIESCENT:.2f} W")
print("  The 2 gripper-finger dof are EXCLUDED: the sim models them as prismatic joints")
print("  in newtons, and the rack pitch radius that would convert N to servo N.m is not")
print("  published. Their tau is 0.01-0.09 N anyway (< 0.1% of the metric).")

print()
print("=" * 92)
print("CALIBRATED GRAVITY-HOLD COPPER LOSS -- widowx only.  NOT total actuator energy.")
print("=" * 92)
print(f"{'cell':<18}{'P_copper W':>12}{'+quiescent':>12}{'E per ep J':>12}"
      f"{'published t2 x':>16}{'calibrated E x':>16}")
for t in ["egg", "spoon"]:
    b = None
    for a in ARMS:
        cell = f"{t}_{a}"
        e = C[cell]["eps"]
        dur = np.array([x["dur_s"] for x in e])
        ms = np.array([x["t2_j"] for x in e]) / dur[:, None]      # <tau_i^2> per joint
        pc = (ms[:, :6] * COEF).sum(1)                            # W, arm joints
        tot = pc + P_QUIESCENT
        E = np.median(tot * dur)
        t2x = np.median([x["t2"] for x in e])
        if b is None:
            b = (E, t2x)
        print(f"{cell:<18}{np.median(pc):>12.2f}{np.median(tot):>12.2f}{E:>12.1f}"
              f"{t2x/b[1]:>16.2f}{E/b[0]:>16.2f}")
    print()
print("""RANGE. The copper term is the only line with a real uncertainty budget:
  * R = V_stall/I_stall includes the driver drop, so it is an UPPER bound. If half the
    stall drop is in the H-bridge, the copper term halves. -> [0.5x, 1.0x] of the above.
  * K is ROBOTIS' printed output-referenced stall constant. Oaki et al. (arXiv:2605.15949)
    use 2.3179 N.m/A for the same part -- 30% higher, uncited, definition unstated. Using
    theirs would cut the copper term by 41%.  -> a further [0.59x, 1.0x].
  * the dual-joint even split is an assumption; the shadow mirrors in hardware, so it is
    a good one.
  Combining: P_copper is 3.0 W [1.2, 3.0] on egg_lat0 and 4.0 W [1.6, 4.0] on egg_cpu685.

WHAT IS NOT IN THE RANGE, because it is not determinable:
  * the tau being converted is the SIMULATED gravity/Coriolis feed-forward, not the real
    arm's joint torque. The real servo must additionally produce the tracking torque, the
    contact torque, and the grasp force -- none of which is in the array (sections 1-2).
  * no-load / friction current: NOT PUBLISHED for any X-series part. Read off the
    XM430 performance-graph JPEG it is ~0.20 A ~ 2.4 W per servo while moving, i.e.
    plausibly LARGER than the entire gravity-hold copper loss. Not usable as a number.
  * "standby current" is not defined by ROBOTIS as torque-enabled or torque-disabled.
    4.61 W is therefore a floor, not an estimate.
  * U2D2 and the power-hub board publish no quiescent draw at all.
  * google_robot: motor type, Kt, gear ratio, torque limits and power draw are ALL
    unpublished (RT-1 describes the hardware in one clause; the SimplerEnv URDF sets
    effort="10.0" uniformly on every joint including the wheels, a placeholder). A
    calibrated number for coke/drawer is NOT DETERMINABLE FROM ANY AVAILABLE SOURCE.""")
