# Calibrating `∫Στ²dt` into joules

Scope: the `t2_drive_arm_sus` channel of `traces_torque3/` (36 cells, 30 seeds on the six
headline widowx arms) and `g5grid/plane3_runs/` (44 arms × 4 tasks × 10 seeds).
Nothing outside `calib/` was modified. Everything below is reproduced by
`calib/calibrate.py`; the numeric tables are `calib/calibrated_curated.tsv`,
`calib/calibrated_plane.tsv` and `calib/calibration.json`.

Source tags used throughout: **[ROBOTIS] / [TROSSEN] / [HARMONIC DRIVE] / [FRANKA] /
[KINOVA]** = the vendor's own published document; **[RETAIL]** = a retailer's mirror of a
vendor spec sheet with no official page behind it; **[3RD PARTY]** = a paper or community
package; **[DERIVED]** = computed here from tagged numbers, derivation shown;
**[ASSUMED]** = not sourced anywhere, stated as an assumption.

---

## 0. Verdict, up front

**`ENERGY_AUDIT.md` §4's conclusion — "ROBOTIS publishes no winding resistance or Kt,
therefore no calibration" — is wrong in its premise and right in its conclusion, for a
different reason.**

* The premise fails. ROBOTIS publishes a **stall torque / stall current pair** for every
  X-series part. At stall ω = 0, so the back-EMF vanishes and the entire supply goes into
  the winding: `V·I_stall = I_stall²·R` closes for `R = V/I_stall`, and
  `K = τ_stall/I_stall` is the output-referenced torque constant that makes
  `P_copper = (τ/K)²R` reproduce `V·I_stall` exactly at stall. That is a **complete and
  self-consistent motor model** from published numbers. §1.
* The conclusion survives on different grounds. Converting the **drive-torque** channel at
  face value returns **1.6 kW – 84 kW** for a 2.1 kg arm on a 60 W supply. Not because the
  motor constants are wrong, but because the simulated PD torque is **a median 94×** the
  torque the same executed trajectory physically requires (§4). The blocker is the
  simulated controller, not the datasheet.

**Conversion factor, the headline number:**

| channel | coefficient | meaning |
|---|---|---|
| `t2_drive_arm_sus` → J | **c̄ = 1.641 W per (N·m)²** | copper loss of the PD drive torque |
| `t2_qf_sub_arm` → J | **c̄ = 0.827 W per (N·m)²** | copper loss of the gravity+Coriolis torque |
| idle floor | **P_idle = 4.61 W** | 9 servos, torque-disabled |
| hard ceiling | **P_supply = 60 W** | shipped 12 V / 5 A PSU |

**Calibrated headline (eggplant, 30 seeds × 24 episodes per arm):**
the ideal arm burns **117 – 763 J** per episode (**7.6 – 52.7 W**) and `cpu685` burns
**200 – 1440 J** (**8.4 – 60.0 W**). The published `3.36×` ratio becomes **1.71× – 1.89×**.

**On spoon the headline claim does not survive**: the published **34.3×** becomes
**1.43× – 3.65×**. §5.

---

## 1. The actuator model, and why the stall pair is enough

### 1.1 The arm

The WidowX 250 S (`wx250s`) carries **7 × XM430-W350-T + 2 × XL430-W250-T**, with
`shoulder` and `elbow` each **doubled** — a shadow servo mirrored in hardware on the same
axis, sharing the load. [TROSSEN]
<https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/wx250s.html>

| joint (sim name) | servo | n | ROBOTIS ID |
|---|---|---|---|
| `waist` | XM430-W350 | 1 | 1 |
| `shoulder` | XM430-W350 | **2** | 2, 3 |
| `elbow` | XM430-W350 | **2** | 4, 5 |
| `forearm_roll` | XM430-W350 | 1 | 6 |
| `wrist_angle` | XM430-W350 | 1 | 7 |
| `wrist_rotate` | XL430-W250 | 1 | 8 |
| gripper (*not* in the `_arm` channels; counts for standby) | XL430-W250 | 1 | 9 |

Reach 650 mm, payload 250 g, 6 DOF. [TROSSEN, same page]
The ALOHA leader variant has a different BOM (XC430 gripper, XM430 wrist rotate) and is
**not** what SIMPLER-env models.
<https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/awx250s.html>

### 1.2 What ROBOTIS does and does not publish

| | XM430-W350 @ 12.0 V | XL430-W250 @ 12.0 V |
|---|---|---|
| stall torque / stall current | **4.1 N·m / 2.3 A** [ROBOTIS] | **1.5 N·m / 1.4 A** [ROBOTIS] |
| also published at | 3.8/2.1 @ 11.1 V, 4.8/2.7 @ 14.8 V | 1.4/1.3 @ 11.1 V, 1.0/1.0 @ 9.0 V |
| no-load speed | 46 rpm | 61 rpm |
| standby current | **40 mA** | **52 mA** |
| gear ratio | 353.5 : 1 | 258.5 : 1 |
| motor | coreless | cored |
| **winding resistance R** | **not published** | **not published** |
| **motor torque constant Kt** | **not published** | **not published** |
| **no-load current** | **not published** (no such row exists) | **not published** |

<https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/> ·
<https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/>

Confirmed absent from the eManual, `docs.robotis.com`, the robotis.us store page,
servodatabase.com, and the DYNAMIXEL Workbench documentation. The XM430's control table
does give a **current unit of 2.69 mA/LSB** for Goal Current / Present Current but states
no mA→N·m relation. The XL430 has **no current register at all** — Velocity / Position /
Extended-Position / PWM only — so a current-referenced model is inapplicable to
`wrist_rotate` other than through the stall pair.

### 1.3 The derivation ROBOTIS' silence does not block

At stall the rotor is locked, ω = 0, back-EMF = 0, so the whole terminal voltage appears
across the series resistance:

```
      V = I_stall · R                 =>   R  = V / I_stall           [DERIVED]
      τ_stall = K · I_stall           =>   K  = τ_stall / I_stall     [DERIVED]
      P_copper(τ) = I²R = (τ/K)² R
```

and the model closes on itself: at τ = τ_stall it returns `(τ_stall/K)²R = I_stall²R =
V·I_stall`, the arm's actual stall power draw. Nothing is fitted.

Two honest caveats, both stated as such:

* **R is an upper bound on the winding.** `V/I_stall` lumps in the H-bridge R_DS(on), the
  bus wiring and the connector. If half the stall drop is in the driver, the *winding*
  resistance — and the copper term — halves. Carried as the low bound (×0.5).
* **K is output-referenced** (post-gearbox, gear efficiency already folded in), because it
  is τ at the output shaft over current at the terminals. That is the right convention
  here, since the sim's `τ` is a joint torque. Do **not** mix it with a motor-referenced
  Kt.

Internal consistency check: R computed at three voltages from the three published stall
pairs is **5.29 / 5.22 / 5.48 Ω** (11.1 / 12.0 / 14.8 V) for the XM430 — a 5 % spread
across a 33 % voltage range, which is what a fixed resistance should look like and would
not be true if the stall pairs were arbitrary.

Third-party corroboration that this is the standard practice: the community
`dynamixel_hardware` ros2_control package requires the user to supply "motor torque
constant in Nm/A" and its own example config uses `torque_constant = 1.79` — i.e. 4.1/2.3.
[3RD PARTY] <https://github.com/dynamixel-community/dynamixel_hardware>

**Dissent, recorded:** Oaki et al., arXiv:2605.15949, state **2.3179 N·m/A** for the
XM430-W350, 30 % above the stall-pair ratio, with no stated definition and no statement of
whether it was measured. Using it instead cuts the copper term by
`(1.7826/2.3179)² = 0.591`. Carried as part of the low bound.
<https://arxiv.org/abs/2605.15949>

### 1.4 The per-joint coefficients

For a joint whose `n` shadowed servos split τ evenly,
`P = n·((τ/n)/K)²·R = (R/(nK²))·τ²`:

| joint | servo | n | K = τ/I [N·m/A] | R = V/I [Ω] | **c = R/(nK²) [W/(N·m)²]** | joint stall τ [N·m] | joint stall P [W] |
|---|---|---|---|---|---|---|---|
| `waist` | XM430 | 1 | 1.7826 | 5.2174 | **1.6419** | 4.10 | 27.6 |
| `shoulder` | XM430 | 2 | 1.7826 | 5.2174 | **0.8209** | 8.20 | 55.2 |
| `elbow` | XM430 | 2 | 1.7826 | 5.2174 | **0.8209** | 8.20 | 55.2 |
| `forearm_roll` | XM430 | 1 | 1.7826 | 5.2174 | **1.6419** | 4.10 | 27.6 |
| `wrist_angle` | XM430 | 1 | 1.7826 | 5.2174 | **1.6419** | 4.10 | 27.6 |
| `wrist_rotate` | XL430 | 1 | 1.0714 | 8.5714 | **7.4667** | 1.50 | 16.8 |

The shipped metric's implicit `c_i = 1` is therefore wrong by up to **9.1×** per joint.
Whether that matters depends on which joints carry the integral, measured on the
per-joint arrays in `smoke_torque3/` (the only place they survive; the sweep keeps
arm-summed scalars):

| channel | `waist` | `shoulder` | `elbow` | `forearm_roll` | `wrist_angle` | `wrist_rotate` | **c̄** |
|---|---|---|---|---|---|---|---|
| drive `t2_drive_arm_sus` | 21–28 % | 39–43 % | 6–9 % | 10–11 % | 10 % | 6–8 % | **1.641** |
| gravity `t2_qf_sub_arm` | <1 % | 67–75 % | 23–33 % | <1 % | <1 % | <1 % | **0.827** |

The gravity channel lands almost entirely on the two **doubled** joints, which share the
same `c`, so `c̄_qf = 0.827` is insensitive to the weighting to within 1 %. The drive
channel is spread and `c̄_drv = 1.641` carries a ±5 % weighting sensitivity across arms
(1.56 on `egg_lat0`, 1.71 on `egg_cpu685`).

### 1.5 Idle, no-load and friction

* **Standby**, all 9 servos, torque disabled: `12 V × (7×40 + 2×52) mA` = **4.608 W**.
  [ROBOTIS] ROBOTIS does not state whether "standby" means torque-enabled or disabled, so
  this is a **floor**, not an estimate.
* **No-load / friction current: not published for any X-series part.**
  `ENERGY_AUDIT.md` §4.3 read ≈ 0.20 A off the eManual performance-graph JPEG; that JPEG
  cannot be parsed by any text tool and I could not re-verify it. If true it is **19.2 W**
  across the 8 arm servos — larger than the entire copper term. **[ASSUMED]**, and carried
  only in the high bound. *This is the single largest unsourced term in the budget.*
* **Viscous friction**: Oaki et al. identify joint-level `FV ∈ [0.0287, 0.317] N·m·s/rad`
  on XM430-driven joints. [3RD PARTY] At the joint speeds in these traces
  (rms 0.06–0.6 rad/s) `FV·ω²` ≤ 0.12 W per joint — negligible, and folded into η.
* **Gear/motor efficiency η for the mechanical term: not published.** **[ASSUMED]**
  η = 0.70 in [0.50, 0.85].
* **Supply**: 12 V / 5 A = **60 W**. [RETAIL] — corroborated across three retailer mirrors
  of the Trossen spec sheet (tribotix.com, mgsuperlabs.com, wevolver.com) but **absent from
  every `docs.trossenrobotics.com` page**. Treated as the hard electrical ceiling.
* **Arm mass: not published by Trossen.** Third-party figures conflict (2.8 kg vs 5.2 lb);
  the SimplerEnv URDF sums to **2.138 kg**. Only the URDF value is used, and only for the
  gravity anchor in §6.

---

## 2. The decomposition

```
   E_electrical  =  W_mech / η          mechanical output, referred to the terminals
                 +  Σ_i c_i ∫τ_i² dt    copper loss                    <- the N²m²s channel
                 +  P_idle · T          quiescent
```

The middle term is exactly `t2_drive_arm_sus` once `c_i` is supplied — which was the point
of the exercise. Two notes on the other two:

**`work_abs_arm` and `work_drive_arm` really are joules.** Verified by construction in
`trace_eval2.py:695-698`: `work += τ·qvel·SUBSTEP_DT` accumulated inside the substep loop
and summed over the episode, i.e. `∫Στ·ω dt` at 500 Hz. `work_abs` is the same with `|·|`.
`|·|` is the right choice electrically: a Dynamixel cannot regenerate into the bus, so a
braking joint still dissipates.

**But `W_mech` is not the real arm's mechanical work.** The simulator hands gravity to a
feed-forward (`balance_passive_force: true`, so `qf = g(q)+C(q,q̇)q̇` is written separately)
and the drive only supplies `M(q)q̈`, so `work_drive_arm` has a **median of −8 J** on
eggplant — the stiff damper absorbs more than the spring delivers. The real arm's
mechanical work is instead bounded by the potential-energy swing of its own links, which
from `ep*_link_com.npy` × `link_mass` is **0.5–1.1 J per episode** (total variation 1.0–1.5 J).
**Mechanical work is under 1 % of the calibrated total and is not what this metric is
about.** The energy of a WidowX doing tabletop manipulation is a *holding* cost, not a
*moving* cost.

---

## 3. Three models

| | what it converts | status |
|---|---|---|
| **M1 sim-literal** | `c̄_drv·t2_drive_arm_sus + W_abs/η + P_idle·T` | **rejected**, §4 |
| **M2 supply-bounded** | the same, clipped actuation-by-actuation at the 60 W rail | **upper bracket** |
| **M3 quasi-static** | `c̄_qf·t2_qf_sub_arm + P_idle·T` | **lower bracket** |

**M2** is what a real Dynamixel does when handed an impossible command: it does not
dissipate impossible power, it saturates at its current limit, draws the rail, and fails to
track. Clipping is done per actuation (40 ms on widowx, 333 ms on google) using the
per-actuation `Στ²` series in `series.npz` — verified to reproduce `t2_drive_arm_sus` to
1 × 10⁻⁶ relative. `series.npz` exists for seeds 100 and 110–129 but **not** 101–109, so
M2, `infeas` and `duty` are reduced over the series-bearing seeds only (21 of 30 on the six
headline widowx arms; **1 of 10** on `pipe110`, `p130w275`, `fp32_555` and on every google
cell — treat those four rows as indicative). Where no series exists at all (the 44-arm
plane) M2 degrades to clipping the episode mean, which is a strict **upper bound** on the
per-actuation result because `min()` is concave.

**M3** converts the gravity+Coriolis channel, proved in `ENERGY_AUDIT.md` §1.1 to be
exactly `g(q) + C(q,q̇)q̇` and nothing else. This is the copper loss of holding the arm along
the trajectory it actually flew. It is a **lower** bracket because it is blind to contact
and to the tracking torque — the very complaint the audit made about the old metric. Here
that blindness is used deliberately, as one side of a bracket.

`M3 ≤ truth ≤ M2`.

---

## 4. Why M1 is rejected — the actual finding

The simulated PD torque is not a torque any WidowX joint can produce.

| | eggplant, 5 040 episodes |
|---|---|
| gravity+Coriolis torque, arm-vector rms | **1.73 – 2.53 N·m** (median 1.96) |
| PD drive torque, arm-vector rms, same episodes | **15.4 – 270.8 N·m** (median 199.0) |
| ratio | **median 94×** |
| shoulder joint capacity (2 × XM430-W350 at stall) | **8.2 N·m** |
| arm-vector capacity ‖τ_cap‖ = √Σ(n·τ_stall)² | **13.7 N·m** |

Independent anchor from the sim's own mass model: the gravity hold at the shoulder,
straight off `ep00_qf.npy`, is **1.36 N·m rms / 1.53 N·m peak**, and reconstructing it from
`ep00_link_com.npy` × `link_mass` gives 1.05 / 1.16 N·m about a shoulder-COM pivot proxy.
The two agree to ~30 %, which is the accuracy of the pivot proxy. Either way the arm's own
weight needs **~1–1.5 N·m of the shoulder's 8.2 N·m** — the sim's arm is a perfectly
reasonable WidowX load. It is the controller that is not: `drive_config.json` has stiffness
730–1270 N·m/rad and damping 152–330 N·m·s/rad against `force_limit` of 100–200 N·m, i.e.
**12–67× the real joint's stall torque**, and `sat_frac_arm` shows the arm pinned at that
limit 12–35 % of substeps on eggplant (0.1–18 % on spoon). A 0.6° tracking error against 1000 N·m/rad is 10 N·m.

Two more realizability failures in the same direction:

* **Contact.** On eggplant the arm presses with `contact_Fnet_mean` = **198–484 N**. At the
  logged mean TCP reach of 0.28 m that is **55–136 N·m** at the shoulder. A WidowX 250 S
  can press with at most `8.2 / 0.28 ≈ 29 N` at that reach, and is *rated* for a 250 g
  payload (2.45 N). The sim's eggplant episodes are 80–200× outside the rated envelope.
* **Duty.** Per actuation, the demand exceeds the 60 W rail **21 %** (spoon/ideal) to
  **100 %** (egg/cpu685) of the time, and exceeds ‖τ_cap‖ **19 %** (spoon/ideal) to **80 %** (egg/fp32_555) of the time.

So M1 returns **1.6 kW – 84 kW** (723× – 1400× the whole power supply) and is rejected. It
is reported in the tables anyway, because "how far outside the hardware envelope is this
trajectory" is a real quantity and M1 measures it.

**The asymmetry worth noting: this is a widowx problem, not a robot problem.** On
`google_robot` the drive torque is only **1.4 – 3.1×** the quasi-static torque, because its
controllers are `interpolate_by_planner` (the drive target moves every substep) while
widowx's `arm_pd_ee_target_delta_pose_align2` has `interpolate=False` and steps the setpoint
once per 200 ms control period. M1 on google lands at **51 – 91 W**, inside a 7-DOF arm's
published envelope, and needs no rejection. The metric is broken by the *simulated
controller's* setpoint discipline, not by the actuator data.

---

## 5. Calibrated results — the curated set

`ideal` = `lat0`, the same baseline `paper/fig_metrics3.py` normalises to. Reduction is
identical to that figure: per-seed **median** episode, then mean over seeds.
Full table incl. all nine arms, both bounds and the raw channels:
`calib/calibrated_curated.tsv`.

### 5.1 eggplant in basket — widowx, 30 seeds × 24 episodes per arm

| arm | succ % | dur s | τ_drv rms | **t2 ×** (published) | **M3 J / W / ×** | **M2 J / W / ×** | M1 W |
|---|---|---|---|---|---|---|---|
| ideal | 54.9 | 15.85 | 134.7 | **1.000** | 117.0 / 7.64 / **1.000** | 762.7 / 52.70 / **1.000** | 3.3e4 |
| p105w300 | 52.4 | 17.86 | 154.6 | **1.293** | 132.5 / 7.63 / **1.132** | 1039.5 / 57.64 / **1.363** | 4.2e4 |
| p150w300 | 49.4 | 20.18 | 158.6 | **1.552** | 147.5 / 7.63 / **1.261** | 1142.7 / 57.29 / **1.498** | 4.5e4 |
| pipe200fix | 32.5 | 23.97 | 199.1 | **2.611** | 184.8 / 7.89 / **1.579** | 1423.6 / 59.92 / **1.866** | 6.6e4 |
| serial283 | 19.3 | 24.00 | 212.2 | **2.932** | 188.5 / 7.94 / **1.611** | 1440.0 / 60.00 / **1.888** | 7.5e4 |
| cpu685 | 1.7 | 24.00 | 225.6 | **3.361** | 200.1 / 8.36 / **1.710** | 1440.0 / 60.00 / **1.888** | 8.4e4 |

Ordering: **M3 is identical to the published t2 ordering** across all nine arms. M2
reorders once (`p130w275` above `pipe110`) and the swap is inside the seed CIs.
Spread collapses from **3.36× to 1.71–1.89×**.

### 5.2 spoon on towel — widowx, 30 seeds × 24 episodes per arm

| arm | succ % | dur s | τ_drv rms | **t2 ×** (published) | **M3 J / W / ×** | **M2 J / W / ×** | M1 W |
|---|---|---|---|---|---|---|---|
| ideal | 47.8 | 11.32 | 28.6 | **1.000** | 86.0 / 8.48 / **1.000** | 197.3 / 19.44 / **1.000** | 1 592 |
| p105w300 | 43.8 | 11.72 | 45.4 | **2.481** | 88.6 / 8.51 / **1.030** | 271.9 / 25.31 / **1.379** | 3 645 |
| p150w300 | 37.2 | 11.97 | 58.9 | **4.513** | 91.1 / 8.55 / **1.059** | 336.8 / 30.03 / **1.707** | 6 129 |
| pipe200fix | 24.9 | 12.00 | 78.9 | **7.798** | 95.1 / 8.51 / **1.106** | 401.4 / 35.51 / **2.035** | 1.1e4 |
| serial283 | 13.2 | 12.00 | 97.3 | **12.482** | 101.7 / 8.69 / **1.182** | 492.0 / 42.59 / **2.494** | 1.6e4 |
| cpu685 | 1.4 | 12.00 | 161.9 | **34.347** | 122.6 / 10.28 / **1.425** | 720.0 / 60.00 / **3.650** | 4.3e4 |

**This is the headline.** Every episode here runs to the same 12 s horizon, so the whole
effect is torque, not duration — and calibration cuts it from **34.3×** to **1.43× (M3) –
3.65× (M2)**. Both calibrated columns stay **monotone in cadence** and both are still
statistically separated from 1.0, so the *direction* of the paper's claim holds; the
*magnitude* does not. The reason is arithmetic and not subtle: a **4.6 W constant sits under
a 3.9–5.7 W signal** in M3, and M2 runs into the 60 W rail before the torque ratio can
express itself.

### 5.3 google_robot — coke and drawer, 10 seeds, PROXY actuator (§6)

| task | arm | **t2 ×** | **M3 J / W / ×** | **M2 J / W / ×** |
|---|---|---|---|---|
| coke | ideal | 1.000 | 1133 / 43.2 / 1.000 | 1781 / 77.6 / 1.000 |
| coke | p105w300 | 0.657 | 1117 / 43.0 / 0.985 | 1232 / 58.1 / 0.692 |
| coke | serial283 | 0.898 | 1149 / 43.2 / 1.014 | 1737 / 70.6 / 0.975 |
| coke | cpu685 | 2.510 | 1165 / 43.7 / 1.028 | 2192 / 89.9 / 1.231 |
| drawer | ideal | 1.000 | 1565 / 42.9 / 1.000 | 1573 / 50.6 / 1.000 |
| drawer | cpu685 | 1.098 | 1577 / 42.9 / 1.007 | 1509 / 51.3 / 0.959 |

Drawer is **flat to within ±5 % on every calibrated model**, corroborating the audit's §3.5
paired test. Coke keeps its ordering under M2 but the spread falls from 4.05× to 1.95×.
Every google M2 row rests on **one** series-bearing seed — indicative only.

### 5.4 The 44-arm plane, 42 240 episodes

| task | t2 spread | **M3 spread** | Spearman(t2, M3) | M3 watts |
|---|---|---|---|---|
| egg | 2.42× | **1.38×** | +0.918 | 7.62 – 8.03 W |
| spoon | 5.00× | **1.14×** | +0.774 | 8.34 – 8.74 W |
| coke | 1.64× | **1.25×** | +0.258 | 42.87 – 43.17 W |
| drawer | 1.53× | **1.04×** | −0.043 | 42.84 – 42.96 W |

The plane has no `series.npz`, so its M2 is the episode-mean upper bound and, on the two
widowx tasks, saturates at 60 W on all 44 arms — it carries nothing beyond duration there
and is reported as `M2ub` for that reason. M3 is the usable column. The two widowx tasks
keep a strong rank correlation with the raw channel (+0.77, +0.92); **the two google tasks
lose it entirely** (+0.26, −0.04), i.e. on coke and drawer the published τ² ordering is not
an ordering in energy at all.

---

## 6. google_robot: not determinable, and the proxy used instead

**Everyday Robots published no actuator specification of any kind.** Checked and empty:
RT-1 (arXiv:2212.06817 §4 — "a 7 degree-of-freedom arm, a two-fingered gripper, and a
mobile base", the entire hardware description), RT-2 (arXiv:2307.15818), SayCan
(arXiv:2204.01691), AutoRT (arXiv:2401.12963), the archived everydayrobots.com technology
page (web.archive.org, 2022-01-12, fetched through its own Next.js JSON endpoints), the
x.company project page, IEEE Spectrum's coverage (whose thesis is precisely that Alphabet
would not disclose specs), and Google Patents — assignee `"Everyday Robots"` returns **zero
results**, and the X Development actuator patents (US11325246B2, WO2022203933A1) use only
illustrative numbers tied to no product. Even the payload appears only as the AutoRT robot
constitution's *"shall not attempt to lift objects that are heavier than a book."* No motor
type, torque constant, gear ratio, joint torque limit, winding resistance, arm mass,
battery capacity or power figure exists anywhere. The SimplerEnv URDF sets
`effort="10.0"` **identically on every joint, wheels and fingers included**, with no
`<transmission>` or `<actuator>` tags — a solver-stability placeholder, not a rating.

**Proxy, clearly labelled:**

* Copper coefficient from the **Harmonic Drive FHA-14C at 100:1** — the only commercially
  published robot **joint module** found that states Kt *and* R *and* gear ratio in one
  official document (Kinova, Franka and UR all publish system power and no motor
  constants). `Kt = 2.9 N·m/A`, and the datasheet's Note 3 says it is **output-referenced**
  ("considering the efficiency of the gear") — the same convention as the WidowX numbers,
  so the two are directly comparable. `R = 0.07 Ω` per phase at 20 °C; the datasheet does
  **not** state phase-to-phase vs phase-to-neutral, a factor of 2 on copper.
  `c = 1.5·R/Kt² = 0.0125 W/(N·m)²`, taking the three-phase sinusoidal form; band [0.0083,
  0.0166]. Rated/max output torque 6.8 / 28 N·m, 24 VDC. [HARMONIC DRIVE]
  <https://www.harmonicdrive.net/_hd/content/catalogs/pdf/fha-c-miniature-spec-sheet-24v.pdf>
  *That c is 130× smaller than the WidowX's, which is the real content of the comparison: a
  24 V harmonic-drive joint module is far more electrically efficient than a 12 V hobby
  servo, and the same trajectory costs far less on it.*
* Idle and ceiling from published **whole-arm** envelopes for 7-DOF collaborative arms:
  Franka Panda arm typical **~60 W** / max **~350 W** (+80 W control box)
  [FRANKA] <https://download.franka.de/Datasheet-EN.pdf>; Kinova Gen3 24 VDC with a **300 W**
  supply rating [KINOVA] <https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf>;
  UR5e typical **200 W** / max **570 W** [UNIVERSAL ROBOTS]
  <https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf>.
  Used: `P_idle = 40 W` in [25, 60], `P_cap = 350 W`. **[ASSUMED/PROXY]**
  None of these vendors publishes a true idle figure distinct from typical operation; the
  Gen3 "idle 25 W / average 36 W" breakdown circulating online is from a **reseller mirror**
  and is not in Kinova's own documentation.
* `‖τ_cap‖ = 81.2 N·m` from Kinova Gen3's published output torques (4 large joints at
  39 N·m, 3 small at 13 N·m). **[ASSUMED/PROXY]**

**How much depends on the proxy.** `c` and `P_idle` are single scalars applied to every
google cell, so they **cancel in the copper ratio** but not once `P_idle` is added — and
`P_idle` is ~92 % of M3 on coke. So: **google's absolute watts are proxy-determined to
within a factor of ~2, and its calibrated *ratios* are proxy-determined too**, because the
ratio is dominated by the constant. The one thing that is *not* proxy-dependent is the
qualitative result — drawer flat, coke's spread compressed — which follows from the shape
of the τ² distribution, not from the constants. Take the google rows as an
order-of-magnitude statement about a class of arm, not a measurement of the Everyday Robots
manipulator.

---

## 7. Sanity check against reality

**The envelope.** Every rung derived above, in watts:

| W | what |
|---|---|
| 4.61 | all 9 servos powered, torque disabled (standby) [ROBOTIS] |
| 2.83 | copper to hold the arm's own weight (shoulder 1.36 + elbow 1.25 N·m rms) [DERIVED] |
| **7.44** | **standby + gravity hold — the floor for "arm on, holding a pose"** |
| 26.6 | + 0.20 A no-load on 8 arm servos [ASSUMED, unverified] |
| 60.0 | shipped 12 V / 5 A PSU — the hard ceiling [RETAIL] |
| 210 | every arm servo at stall simultaneously — thermally impossible, and 3.5× the PSU |

**Where the calibration lands.**

| | eggplant | spoon | verdict |
|---|---|---|---|
| M3 | **7.62 – 8.36 W** | **8.38 – 10.28 W** | inside, and within 12 % of the 7.44 W gravity-hold floor |
| M2 | 52.7 – 60.0 W | 19.4 – 60.0 W | inside, pinned at the rail for the worst arms |
| M1 | 3.3e4 – 8.4e4 W | 1.6e3 – 4.3e4 W | **1400× the supply. rejected.** |

M3 sitting 0.2–2.8 W above a floor computed independently from `ep00_qf.npy` and from the
link masses is the check that matters: the model reproduces a number derived two other ways.
The remaining question is not whether M3 is plausible but whether it is *complete* — it is
not, by construction, and M2 is the other end of the bracket.

Corroborating external figures, both weak but not contradicted: no watt-level measurement
of any WidowX/ViperX arm exists in the literature (checked ALOHA arXiv:2304.13705, ALOHA 2
arXiv:2405.02292, Mobile ALOHA arXiv:2401.02117, Trossen/Interbotix docs, retailer
listings — all silent). Mobile ALOHA's 1.26 kWh battery over a reported ~12 h implies
~105 W for a **whole system** of two arms plus mobile base, onboard compute and three
cameras — consistent with a per-arm figure in the tens of watts, but it isolates nothing.

---

## 8. Uncertainty, stated plainly

| term | value | source | multiplier on the low/high bound |
|---|---|---|---|
| stall pairs (τ, I) at 12 V | 4.1/2.3, 1.5/1.4 | **[ROBOTIS]** | — |
| standby current | 40 / 52 mA | **[ROBOTIS]** | — |
| servo complement + doubling | 7×XM430 + 2×XL430 | **[TROSSEN]** | — |
| `R = V/I_stall` | 5.22 / 8.57 Ω | **[DERIVED]**, upper bound | ×0.5 low |
| `K = τ_stall/I_stall` | 1.783 / 1.071 N·m/A | **[DERIVED]** | ×0.591 low (Oaki's 2.3179) |
| joint-share weighting of c̄ | measured on `smoke_torque3` | **[DERIVED]** | ±5 % |
| supply ceiling 60 W | 12 V × 5 A | **[RETAIL]**, no official page | — |
| **no-load current 0.20 A** | 19.2 W across 8 servos | **[ASSUMED]**, unverifiable JPEG | high bound only |
| gear/motor efficiency η | 0.70 | **[ASSUMED]** | [0.50, 0.85] |
| arm mass 2.138 kg | SimplerEnv URDF | **[SIM]**, Trossen publishes none | anchor only |
| everything google | proxy | **[ASSUMED/PROXY]** | ×0.67 – ×1.33 on c, [25,60] W idle |

Combined, on the eggplant `ideal` cell: **M3 = 117 J [86, 421]**, i.e. **7.64 W [5.6, 27.5]**.
The width is almost entirely the unpublished no-load current; drop that term and the band
is [86, 117] J, i.e. the high bound collapses onto the central value (the copper
coefficient has no upper multiplier — `R = V/I_stall` is already an upper bound). On `cpu685`: **M3 = 200 J [137, 661]**.

**What is outside the range because it is not determinable:**

* The contact and tracking torque a real servo would additionally have to produce. M3 omits
  it, M2 bounds it at the rail, and nothing in the traces resolves it — because, per §4, the
  simulated version of it is 94× too large.
* Whether ROBOTIS' "standby" is torque-enabled. If it is not, the 4.61 W floor is low.
* The controller board (U2D2) and the power hub — neither publishes a quiescent draw.
* Reflected rotor inertia. A 353.5:1 gearbox reflects `N²·J_rotor` to the joint; for a
  coreless motor with `J_rotor ~ 10⁻⁷ kg·m²` that is ~0.0125 kg·m², **comparable to the link
  inertias themselves**. The sim models every joint as an ideal direct-drive revolute with
  no reflected inertia at all. This is a first-order structural gap between the simulated
  and the real arm and it is not repairable from the recorded traces.

---

## 9. What this means for the paper

1. **The absolute numbers are now sayable, with a bracket.** A WidowX 250 S executing these
   episodes draws **7.6 – 10.3 W (M3)** to **19 – 60 W (M2)**, i.e. **86 – 1440 J per
   episode**. Not "uncalibrated, ratios only".
2. **The ordering survives; the magnitudes do not.** Eggplant's ordering is *identical*
   under M3 and its published **3.36×** becomes **1.71–1.89×**. Spoon reorders only inside
   the seed CIs but its published **34.3×** becomes **1.43–3.65×**. Any sentence quoting a
   τ² ratio as an energy ratio needs the factor restated.
3. **On the plane, the google tasks lose the ordering entirely** — Spearman(t2, M3) is
   +0.26 on coke and −0.04 on drawer across 44 arms. The published τ² ordering there is not
   an energy ordering.
4. **`sat_frac_arm`, `infeas_frac` and `supply_duty` are the honest schedule-discriminating
   metrics on widowx**, and they are dimensionless and defensible in a way the joules are
   not: `infeas_frac` is *the fraction of the episode during which the commanded joint
   torque exceeds what a WidowX 250 S can produce*, and it runs 19 % → 80 % across the arms.
   That is a real, hardware-referenced statement about what latency does, and it does not
   need a motor model to be believed.
5. **Before the next trace run**: log per-joint `τ` (not just the arm sum) and write
   `series.npz` on every seed. Both would remove the two largest methodological compromises
   here — the fixed joint-share weighting of c̄, and M2's 21-of-30 seed coverage.
