# The real actuator torque, and the contact force — what `trace_eval2.py` logs and why

Companion to `ENERGY_AUDIT.md`, which established that the shipped "actuator energy"
column is not one. This document is the repair: what the real signal is in SAPIEN 2.2.2,
the falsification tests it had to pass before anything was re-run, what changed
numerically against `traces_torque2/`, and which conclusions in `ENERGY_RESULTS.md`
survive.

Nothing in `trace_eval.py`, `paper/fig_energy.py`, `paper/make_energy_cache.py` or
`ENERGY_RESULTS.md` was modified. The new harness is `trace_eval2.py`; the new sweep is
`traces_torque3/`, cell-for-cell comparable with `traces_torque2/` (same four tasks, same
nine measured arms, same `--init-rng 100`, same `--n 24`).

---

## 0. Verdict up front

**Yes, the real drive torque and the real contact force are both available, and both are
now logged.** The old column was a gravity feed-forward; the new one is the force the PD
drive actually applies, validated against the articulation's own equation of motion, plus
the contact impulse PhysX reports, validated against a known weight. On the four
falsification tests they behave the way a torque and a contact force must and the old
column provably does not.

**But "energy" is still the wrong word, for a different reason than before**, and §7 says
so plainly. What is now defensible is *actuator effort* and *contact*: dimensionless,
per-second, paired. Watts still require motor constants that are not published for either
arm (`ENERGY_AUDIT.md` §4 is unchanged by any of this).

---

## 1. What SAPIEN 2.2.2 actually exposes

Checked against the installed build (`sapien 2.2.2`,
`/scratch2/dima/miniforge3/envs/octo_sim`), not against the documentation.

| candidate | verdict |
|---|---|
| `Articulation.get_qf()` after stepping vs before | **no difference in kind.** `get_qf()` reads back whatever `set_qf()` last wrote. `base_controller.py:229-238` writes `compute_passive_force(external=False)` every substep and no controller adds a `qf` term, so it is g(q)+C(q,q̇)q̇ whenever you read it. Reading it *after* the step (rather than before, as `trace_eval.py` does) fixes only the tick-0 zero, not the content. |
| `Articulation.get_qacc()` | **useless here.** Returns exactly `0` on every dof at every substep of every task tried. It is never populated. |
| `Articulation.compute_inverse_dynamics(qacc)` | usable, but pointless without `qacc` (see above). With `qacc = 0` it returns 0, so it cannot be used to back out the applied force. |
| `compute_passive_force(external=True)` vs `external=False` | **bit-identical**, measured: max abs difference **0.000e+00 N·m** over 1000 substeps of a widowx press, 656 of which carry up to **856 N** of external contact. The `external` flag covers forces added through `Link.add_force_at_point` / `add_force_torque`, not contact reactions. It is not a contact channel, and `ENERGY_AUDIT.md` §2.5's suggestion to log `compute_passive_force(external=True)` would have recovered nothing. |
| `get_net_contact_forces` / `compute_generalized_external_force` on `Articulation` / `Link` | `get_net_contact_forces` **does not exist** in this version. `compute_generalized_external_force(forces, torques)` exists but takes forces you supply — it maps a wrench you already have into generalized coordinates; it does not report contact. |
| **`get_drive_target()` / `get_drive_velocity_target()` + `Joint.stiffness/damping/force_limit`** | **this is the signal.** All present, all live-readable, and the targets are read off the articulation rather than out of the config, so `google_robot`'s planner-interpolated targets (which change every substep) are captured correctly. |
| **`scene.get_contacts()` → `Contact.points[k].impulse`** | **this is the contact.** Present, and its units and sign convention are pinned down empirically in §1.2. |

### 1.1 The drive torque

```
tau_drive = clip( K*(q_target - q_post) + D*(v_target - v_post),  +-force_limit )
```

evaluated **once per PhysX substep** (500 Hz on widowx, 513 Hz on google_robot), with
`q_post`/`v_post` read *after* `scene.step()`. Post-step, not pre-step, because PhysX
solves articulation drives as an **implicit** spring: the drive force is the one
consistent with the end-of-substep state.

**Validation.** `balance_passive_force=True` means `qf` cancels g+C exactly, so in free
space the articulation obeys `M(q) q̈ = tau_drive` with nothing else in it. Comparing the
reconstruction against `compute_manipulator_inertia_matrix() @ (Δq̇/dt)`:

| | widowx, 1400 contact-free substeps | google_robot, 1710 contact-free substeps |
|---|---|---|
| pre-step form, max abs residual | **288 N·m** | 1.94 N·m |
| **post-step form, max abs residual** | **0.024 N·m** | 0.854 N·m |
| post-step, median rel. residual, per substep | 2.0 % | 54 % |
| post-step, median rel. residual, per 50-substep block | 1.0 % | 10.9 % |
| post-step, median rel. residual, per actuation block | 1.4 % | 9.8 % |

The post-step form is what PhysX applies: on widowx it beats the pre-step form by
**12 000×** in worst-case absolute residual. The google_robot per-substep number looks
bad and is not: that arm's planner-interpolated controller produces `|M q̈|` of only
0.14 N·m, the median *signed* residual is ±0.0007 N·m per joint (i.e. **unbiased**), and
the scatter falls as 1/√B under block averaging (0.121 → 0.048 → 0.025 → 0.014 N·m for
B = 1, 10, 50, 171) — the signature of zero-mean noise in the *validation's* finite
difference of a float32 `qvel` over 1.95 ms, not of an error in the reconstruction.
`tau_drive` itself involves no differencing: every term in it is a direct read.

A second, independent confirmation falls out of the same test. On both embodiments the
residual is ~0 on every arm joint and ~`|tau|` on the **finger** joints — because the
fingers sit against their own joint limit, where a hard constraint absorbs the drive
force and the joint does not accelerate. That is a stall, correctly reported as
"the actuator is pushing and nothing is moving", which is exactly the case the old column
could not see.

### 1.2 The contact force

`scene.get_contacts()`, summing `ContactPoint.impulse` per contact pair, per substep.
Three things had to be established and all three were measured, not assumed:

1. **Units.** `impulse` is an **impulse in N·s over the substep**, so force = `impulse /
   substep_dt`. Measured: an eggplant of 0.0209 kg resting on the sink gives
   `|Σ impulse| / dt = 0.2056 N` against `m g = 0.2049 N` — **0.3 %**.
2. **Sign.** `impulse` acts on `actor0`; the reaction acts on `actor1`. Measured: with
   that convention, the net external contact force on the robot while it presses *down*
   into the sink is `[+5.0, +251, +433] N` — upward, as a surface pushing back must be.
3. **Robot-internal contacts must be excluded.** Adjacent gripper links of the widowx
   self-collide continuously with ~16 kN of solver impulse. Counting them puts the
   "contact force on the robot" at 67 kN at rest. Only pairs with **exactly one** side on
   the robot are counted.

Three channels are kept because they answer different questions: `Fnet` (magnitude of the
vector sum — the net push), `Fsum` (sum of magnitudes — a two-fingered grasp has
`Fnet ≈ 0` and `Fsum ≈ 2 ×` the grip force), and `Fgrip` (`Fsum` restricted to finger
links).

### 1.3 The honest actuator column is `qf + tau_drive`

The simulator splits the generalized force its actuators inject into a feed-forward
(`qf`, the gravity hold, written by the controller) and a PD feedback (the drive). A real
servo has to produce both. `tau_total = qf + tau_drive` is therefore the full actuator
torque; `tau_drive` alone is the part that carries tracking, stall and contact. Both are
logged.

---

## 2. Sampling: what cadence, and why

Everything above is accumulated **inside the substep loop** and reduced **once per
ACTUATION**, not once per tick.

* On google_robot `ACT_EVERY = 9`: the simulator does not advance on 8 of every 9 logged
  ticks (`ENERGY_AUDIT.md` §2.7a), so a per-tick sample would write the same state nine
  times. Per-actuation is the cadence at which the signal changes.
* Reducing *from the substeps* rather than sampling at the boundary makes
  `∫Στ²dt = Σ_actuations mean_substeps(τ²)·act_dt` an **exact 500/513 Hz integral**, which
  is what `ENERGY_AUDIT.md` §5.2 asked for. The "brief impacts are under-counted" caveat
  is now genuinely false rather than vacuously false.
* Storage stays small: one row per actuation, not per substep. A widowx episode is 600
  rows × 8 joints; a drawer episode 113 × 11.

### 2.1 The setpoint-step transient — disclosed, and separated

widowx's arm controller (`arm_pd_ee_target_delta_pose_align2`, `interpolate=False`) steps
the PhysX drive target **once per 200 ms control period** and holds it for all 100
substeps. A step of Δq against K ≈ 1200 N·m/rad is an instantaneous K·Δq that the
330 N·m·s/rad damper kills in one or two substeps. Measured in free space over 12
actuations:

| | widowx | google_robot |
|---|---|---|
| share of `∫Στ²dt` carried by **substep 0** of each actuation | **96.9 %** | 0.27 % |
| uniform share would be | 1.0 % | 0.58 % |
| mean(substep 0) / mean(substeps ≥ 10) | **23 139×** | 0.44× |

google_robot does not have it (`interpolate_by_planner` moves the target every substep).
This spike is a torque PhysX really applies, and its size is the commanded step size —
which latency does change — so it is not noise. But a column that is 97 % one substep in
100 measures commanded step size, not contact. So **both** are logged:
`tau_drive_sq` over all substeps and `tau_drive_sq_sus` over substeps ≥ 5.

Contact does not live in the transient. In the press counterfactual of §3.2, over the
control steps where the gripper is jammed into the sink, substeps 0–4 carry **4.8 %** of
`Στ²` and `mean(substeps ≥ 5) = 62 241` against `mean(all) = 62 104` — the sustained
channel keeps the whole contact signal and drops the whole step spike. Free-space
`_sus` is 0.01 against contact `_sus` of 62 241: a **6-million-fold** contrast, sharper
than the 52 000× the all-substep column gives.

---

## 3. The four falsification tests

Every number below is reproducible. The controlled experiments are
`step2_press.py` / `step2_grip.py` (kept in the scratch dir and reproduced verbatim in
§3.2/§3.4); the real-rollout ones come out of `analyze_smoke_torque3.py` over
`smoke_torque3/`.

### 3.1 CHECK (i) — it is not zero at tick 0

`trace_eval.py` logs `get_qf()` *before* the first `step_action`, and `BaseAgent.reset()`
(`base_agent.py:159`) has just done `set_qf(zeros)`. The old channel is therefore
**exactly 0.000000e+00 on tick 0 of all 864 episodes** of `traces_torque2/`, while the
arm is visibly holding itself against gravity.

Reading inside the substep loop fixes it:

| channel | tick 0, widowx eggplant | tick 0, google drawer |
|---|---|---|
| OLD per-tick `Σ qf²` | **0.000000** | **0.000000** |
| new substep-resolved `Σ qf²` | 3.51 | 101.6 |
| new `Σ τ_drive²` | 0.038 | 9.0e-07 |
| new `Σ τ_total²` = `Σ(qf+τ_drive)²` | **3.45** | **101.6** |

The honest reading: `tau_drive` at tick 0 is genuinely near zero and *should* be — the
robot starts at rest with its drive target at its own qpos, so the PD has nothing to do.
The actuator column that must be non-zero is `tau_total`, and it is. **PASS.**

### 3.2 CHECK (ii) — it is not a pure function of (pose, velocity)

Two constructions, both counterfactual: the same reset, the same deterministic command
sequence, run twice, differing *only* in whether an obstacle is there. The two branches
are bit-identical until first contact, so at contact onset they sit at the same state by
construction.

**(a) widowx pressing into the sink** (`step2_press.py`: 10 control steps of
`world_vector = [0,0,-0.03]`; branch B has the sink's collision shapes disabled).
At the first contact substep (332 of 1000):

| | branch A (sink present) | branch B (sink removed) |
|---|---|---|
| `‖Δq‖∞` between branches | **5.5e-04 rad** | |
| `‖Δq̇‖∞` between branches | 3.6e-01 rad/s | |
| **`‖Δqf‖∞` between branches** | **2.4e-07 N·m** | |
| `Σ τ_drive²` (arm) | **14 122** | **0.0204** |
| ratio | | **694 000×** |
| external contact force on robot | 132 N | 0 N |

The old channel is **identical to seven decimal places** across a 694 000× difference in
the applied torque. Over the whole press window the old channel actually reads *lower*
in contact than free (`Σqf²` 3.30 vs 3.61, ratio **0.91×**), while `Σ τ_drive²` is
66 185 vs 0.020 (median).

**(b) google_robot closing the gripper on the coke can** (`step2_grip.py`: the can is
teleported between the fingers and pinned; branch B has it 10 m away). This one matches
velocity as well as position. At substep 240:

| | contact | free |
|---|---|---|
| `‖Δq‖∞` | **2.5e-04 rad** | |
| `‖Δq̇‖∞` | **4.9e-03 rad/s** | |
| `‖Δqf‖∞` | **3.4e-05 N·m** | |
| `Σ|τ_finger|` | **0.0912 N·m** | **0.00013 N·m** |
| ratio | | **715×** |
| finger contact force | 0.86 N | 0 N |

and it grows to 913× at substep 250, 2 031× at 260, 118 448× at 295. **PASS, twice.**

**(c) the audit's own pose control, on real rollouts.** `ENERGY_AUDIT.md` §3.1 killed the
old metric by pairing each stalled tick with a freely-moving tick within 3 cm in TCP
space; the apparent 1.23–2.48× stall signal collapsed to **1.00 / 1.09 / 1.16 / 1.00**.
The same pairing is rerun on `smoke_torque3/` in §5, this time labelling stall by
*measured contact* rather than by inference, and the new column does not collapse.

### 3.3 CHECK (iii) — sustained torque at ~zero joint velocity

Real rollout, `google_robot_close_drawer`, `p105w300`, the failing episode in which the
policy pushes the drawer from 0.20 m to 0.062 m and jams. Pooled over its 113
actuations, comparing **in contact** against **free** with *both* bins restricted to the
lowest quartile of `Σω²` — i.e. holding joint velocity matched:

| bin | n | median `Σ τ_drive²` | median `Σ qf²` | median `Σ ω²` | median contact F |
|---|---|---|---|---|---|
| in contact, `‖ω‖` low | 10 | **3 525.06** | 340.3 | 0.00829 | 82.3 N |
| free, `‖ω‖` low | 19 | **0.07** | 233.6 | 0.00813 | 0.0 N |
| **ratio** | | **≈ 50 000×** | **1.46×** | 1.02× | — |

At matched, essentially-zero joint velocity the drive torque is five orders of magnitude
larger when the arm is pushing. The old column moves by 1.46×, and `ENERGY_AUDIT.md` §3.1
already showed that 1.46× is arm extension, not contact. **PASS.**

### 3.4 CHECK (iv) — finger torque rises on contact

Same controlled grasp as §3.2(b), second half of the closing motion (513 substeps):

| | fingers on the can | fingers on nothing |
|---|---|---|
| mean `Σ|τ_finger|` (drive) | **56.21 N·m** | 25.68 N·m (jammed at the joint limit) |
| mean `Σ|qf_finger|` (old channel) | **0.0284 N·m** | 0.0126 N·m |
| mean gripper contact force | **206.2 N** (peak 362.7 N) | **0.0 N** |
| final finger qpos | 0.415 rad (stopped by the can) | 1.073 rad (joint limit) |

During the closing transit, where the free branch's fingers are still moving, the ratio
is 3 000–300 000× (substeps 250–600). `ENERGY_AUDIT.md` §1.1 measured the old channel at
**0.015 N (egg) / 0.018 N (coke)** with the fingers clamped on the object and observed
that "a real gripper clamping a coke can holds tens of newtons". The contact channel now
says how many: **206 N mean, 363 N peak**. **PASS.**

### 3.5 Summary of the four checks

| check | old channel | new channel | verdict |
|---|---|---|---|
| (i) non-zero at tick 0 | exactly 0 in 864/864 episodes | `τ_total²` = 3.45 (widowx) / 101.6 (google) | **PASS** |
| (ii) not a function of (q, q̇) | identical to 2.4e-07 N·m across the pair | 694 000× (press) / 715× (grasp) | **PASS** |
| (iii) stall at zero velocity | 1.46× | ≈ 50 000× | **PASS** |
| (iv) grasp force | 0.015–0.018 N | 206 N mean / 363 N peak contact | **PASS** |

---

## 4. What `trace_eval2.py` logs

Every array and every `summary.json` field `trace_eval.py` wrote is still written,
unchanged — including the per-tick `ep*_qf.npy` with its tick-0 zero — so old and new are
directly comparable **on the same episodes**. That matters: the harness is not
run-to-run deterministic (`NONDETERMINISM.md`, ~20 % of byte-identical invocations
diverge), so the only trustworthy old-vs-new comparison is the one inside a single
`traces_torque3` run, which is exactly what the `t2_qf_*` columns provide.

New arrays, one row per **actuation**, each row reduced from that actuation's substeps:

| file | shape | contents |
|---|---|---|
| `ep??_act_tick.npy` | (n_act,) | tick index of each actuation |
| `ep??_tau_drive_mean.npy` | (n_act, dof) | mean τ_drive |
| `ep??_tau_drive_sq.npy` | (n_act, dof) | mean τ_drive² — `Σ_a row·act_dt` is the exact 500/513 Hz integral |
| `ep??_tau_drive_sq_sus.npy` | (n_act, dof) | the same over substeps ≥ 5 (see §2.1) |
| `ep??_tau_drive_max.npy` | (n_act, dof) | max \|τ_drive\| |
| `ep??_tau_total_sq.npy` | (n_act, dof) | mean (qf + τ_drive)² |
| `ep??_qf_sub_sq.npy` | (n_act, dof) | mean qf² — the OLD quantity at substep resolution |
| `ep??_qvel_sub_sq.npy` | (n_act, dof) | mean q̇² |
| `ep??_work_drive.npy` | (n_act, dof) | ∫ τ_drive·q̇ dt, signed [J] |
| `ep??_work_abs.npy` | (n_act, dof) | ∫ \|τ_drive·q̇\| dt [J] |
| `ep??_sat_frac.npy` | (n_act, dof) | fraction of substeps at ≥ 99 % of `force_limit` |
| `ep??_contact.npy` | (n_act, 8) | `Fnet_mean, Fnet_max, Fsum_mean, Fsum_max, Fgrip_mean, Fgrip_max, n_ext_mean, impulse_sum` |
| `drive_config.json` | | K, D, force_limit, joint names, substep timing, the reconstruction formula |

and ~45 per-episode scalars in `summary.json` (`t2_drive_arm`, `t2_drive_arm_sus`,
`t2_total_arm`, `t2_qf_tick_arm`, `t2_qf_sub_arm`, `work_abs_arm`,
`contact_impulse_Ns`, `contact_Fsum_max`, `contact_grip_Fmax`, `sat_frac_arm`,
`tau_drive_absmax_arm`, `tcp_reach_mean`, `duration_s`, the four tick-0 diagnostics, …),
so a reduced cell needs no arrays at all.

Two conventions worth stating because they differ from `fig_energy.py`:

* **Arm joints only, applied to *both* the τ and the ω terms.** `fig_energy.py:75`
  applied `JG` to ω² only and summed τ² over all dof (`ENERGY_AUDIT.md` §2.1). Here
  `ARM = slice(0,6)` (widowx) / `slice(0,7)` (google) is applied consistently, and the
  all-dof variants are kept alongside under `*_all` for anyone who wants them.
* **Contact is measured, not inferred.** `ENERGY_AUDIT.md` §3.1 had to *infer* stalls
  from "commanded a translation, TCP did not move". Stall is now a measurement.

### 4.1 Sanity of the contact channel, on a real rollout

`smoke_torque3/egg_lat0`, episode 0 (a success in 6.4 s). Per-actuation `Fnet`:

```
actuation   0..79   free space                    0.00 N
actuation      80   gripper closes on eggplant  155.95 N   (Sum tau_drive^2 jumps 0.04 -> 4491)
actuation 100..159  carrying the eggplant         1.68 N   (Fsum 18-20 N over 2 finger contacts)
```

Zero in free space, an impact spike at the grasp, a small steady hold while carrying.
Over the whole cell `Fnet/Fsum = 0.972` with a mean of 1.83 external contact pairs, so
the vector sum and the sum of magnitudes agree and there is no double counting.

The **large** numbers (the 579 N median while in contact, pooled over the cell) come from
the two *failing* episodes, where the arm jams into the sink for the full 24 s horizon.
That is a real reading of the *simulated* robot — whose `force_limit` is `[200, 200, 100,
100, 100, 100]` N·m, i.e. **12–67× the real WidowX-250 S servo rating**
(`ENERGY_AUDIT.md` §4.1). The sim arm can and does push with several hundred newtons.
**Do not read these contact forces as predictions for the real hardware.**

---

## 5. Smoke test — the four checks on real rollouts

`smoke_torque3/`, four cells x four episodes, covering both embodiments and both a grasp
and a push: `egg_lat0`, `egg_cpu685` (widowx, grasp, ideal and worst arm),
`coke_p105w300` (google, grasp), `drawer_p105w300` (google, push).
Reproduce with `python analyze_smoke_torque3.py smoke_torque3`.

### 5.1 (i) tick 0

| cell | OLD per-tick `Σqf²` | new `Σqf²` (substep) | new `Στ_drive²` | new `Στ_total²` |
|---|---|---|---|---|
| egg_lat0 | **0.0** | 3.510–3.514 | 2.1e-04 – 0.145 | **3.32–3.52** |
| egg_cpu685 | **0.0** | 3.512 | 5.0e-08 | **3.51** |
| coke_p105w300 | **0.0** | 101.59 | 9.0e-07 | **101.57** |
| drawer_p105w300 | **0.0** | 101.59 | 9.0e-07 | **101.57** |

`{0.0}` is the *set* of distinct values over all episodes of the cell. **PASS.**

### 5.2 (ii) contact vs free at matched TCP position (≤ 3 cm), the audit's own control

| cell | n pairs | `Στ_drive²` contact/free | ratio | sustained ratio | **`Σqf²` ratio** | contact F |
|---|---|---|---|---|---|---|
| egg_lat0 | 814 | 65 647 / 0.0040 | **1.7e7×** | 6.9e7× | **0.84×** | 605 / 0 N |
| egg_cpu685 | 460 | 76 645 / 0.036 | **2.1e6×** | 4.6e7× | **0.94×** | 833 / 0 N |
| coke_p105w300 | 60 | 15 352 / 0.239 | **6.4e4×** | 6.3e4× | **1.05×** | 85 / 0 N |
| drawer_p105w300 | 6 | 4 218 / 0.020 | **2.1e5×** | 2.1e5× | **1.00×** | 46 / 0 N |

The `Σqf²` column reproduces `ENERGY_AUDIT.md` §3.1's 1.00 / 1.09 / 1.16 / 1.00 — under
the identical pose control, the old channel is flat. The drive torque is four to seven
orders of magnitude apart. **PASS.**

### 5.3 (iii) stall — matched, essentially-zero joint velocity

| cell | n | `Στ_drive²` contact/free | ratio | `Σqf²` ratio | `Σω²` ratio | contact F |
|---|---|---|---|---|---|---|
| egg_lat0 | 162 | 8 176 / 0.026 | 3.1e5× | 0.97× | 0.63× | 147 / 0 N |
| egg_cpu685 | 60 | 16 280 / 5.6e-06 | 2.9e9× | 0.26× | — | 188 / 0 N |
| coke_p105w300 | 14 | 2 083 / 0.045 | 4.6e4× | 1.28× | 1.24× | 68 / 0 N |
| drawer_p105w300 | 8 | 23 514 / 0.040 | 5.8e5× | 1.42× | 1.45× | 296 / 0 N |

**PASS.** Note the drawer row is the case the whole metric was introduced for: the
gripper pushing the drawer shut, joint velocity matched to the free bin, and the drive
torque 580 000× larger. The old column moves 1.42×, which §3.1 of the audit already
showed is arm extension.

### 5.4 (iv) grasp

| cell | n grip-contact actuations | `Σ\|τ_finger\|` contact/free | ratio | **old `Σ\|qf_finger\|` ratio** | grip F |
|---|---|---|---|---|---|
| egg_lat0 | 920 | 27.68 / 1.414 | 19.6× | 7.03× | **584 N** |
| egg_cpu685 | 1297 | 60.28 / 1.414 | 42.6× | 2.64× | **532 N** |
| coke_p105w300 | 43 | 15.52 / 0.00107 | **14 475×** | **0.60×** | **74 N** |
| drawer_p105w300 | 46 | 18.78 / 2.375 | 7.9× | 0.91× | 118 N |

On coke — the case named in the brief — the finger drive force rises by four orders of
magnitude on contact while the old channel moves the **wrong way** (0.60×). **PASS.**

### 5.5 The confounders, on the smoke cells

Pearson r pooled over the 16 smoke episodes (small n; §6 has the same test at n = 216
per task):

| | r(·, duration) | r(·, mean TCP reach) |
|---|---|---|
| OLD `∫Σqf²dt` | +0.71 | **+0.88** |
| OLD, per second | +0.53 | **+0.96** |
| NEW `∫Στ_drive²dt` | **+0.10** | **−0.55** |
| NEW, per second | **+0.08** | **−0.56** |
| NEW contact impulse, per second | +0.07 | −0.61 |
| NEW `∫Σ|τ·ω|dt`, per second | +0.05 | −0.61 |

The old column is arm extension (+0.96 as a rate). The new one is not; if anything it is
*anti*-correlated with reach, which is mechanically sensible — an arm jammed against
something is a **retracted** arm pushing, not an extended one holding.

### 5.6 Per-cell medians, per second

```
cell              succ   dur s  reach m    OLD/s     DRIVE/s   SUSTAIN/s   Wabs J  |tau|max  sat%  INTFdt Ns  Fmax N  gripF N
coke_p105w300      1/4   26.67   0.6689   308.55     2269.48    2275.50    184.24    300.00  1.10     347.36  2665.7    317.1
drawer_p105w300    1/4   37.67   0.6028   277.39      528.30     538.15    109.96    240.63  0.38      78.29  2346.0   1125.4
egg_cpu685         0/4   24.00   0.3463     6.07    49579.38   49592.79   4250.58    200.00 33.73   12526.23  7535.6   3659.9
egg_lat0           2/4   15.80   0.2896     4.08    34132.36   34130.50   1904.92    200.00 15.54    7100.84  4174.2   2340.8
```

Three things to read off this:

1. **`|tau|max` sits exactly on `force_limit`** (200 N·m widowx shoulder, 300 N·m google
   torso). `ENERGY_AUDIT.md` §3.3 found the old channel reached the limit on **6 ticks in
   476 194**, all of them solver blow-ups; the drive saturates on **15–34 % of substeps**
   on widowx. That is what a drive does and what a feed-forward cannot.
2. **`SUSTAIN/s ≈ DRIVE/s` on real rollouts.** The 97 % setpoint-step transient of §2.1 is
   a *free-space* phenomenon; once the arm is actually touching things, contact torque
   dominates it completely. So on this data the choice of column does not matter — which
   is itself worth knowing, and is why both are kept.
3. **`Wabs` is in real joules and needs no motor model.** 1.9 kJ (egg_lat0) to 4.3 kJ
   (egg_cpu685) of mechanical work at the joints over an episode. This, not `∫Στ²dt`, is
   the quantity that can honestly be called energy — see §7.

---

## 6. The re-run sweep: `traces_torque3/`

36 cells (4 tasks x 9 arms) x 24 episodes, `--init-rng 100`, the same nine MEASURED
QRB5165 latency/cadence pairs as `job_torque2.sh`, run through `trace_eval2.py` on the 25
g5 workers (`g5fine/job_torque3.sh`, `g5fine/launch_torque3.sh`, joblists balanced by the
measured `traces_torque2` wall times). Reduced on the worker by
`g5fine/reduce_torque3.py` — per-episode scalars in `energy2.json`, a compact
per-actuation `series.npz` (~100 kB/cell instead of 13 MB), every `.npy` deleted.
Analysis: `python analyze_torque3.py traces_torque3`.

**On comparability.** The harness is not run-to-run deterministic (`NONDETERMINISM.md`),
so `traces_torque3` episode *e* is not the same rollout as `traces_torque2` episode *e*.
The old-vs-new comparison that matters is therefore the one **inside** `traces_torque3`,
where `t2_qf_tick` is `fig_energy.py`'s exact column computed on the very same rollouts
as the new ones. Cell-level medians are still comparable against `ENERGY_RESULTS.md`,
which is the level it reports at.

