# Adversarial audit of the actuator-energy metric

Scope: `∫Στᵢ²dt` and `∫Σω_arm²dt` as computed by `paper/fig_energy.py`,
`paper/make_energy_cache.py` and `paper/fig_pareto_energy.py` over
`traces_torque2/` (36 cells × 24 episodes, `init_rng=100`).
Nothing in that list was modified. Everything below is reproducible from
`energy_audit_0*.py` in this directory.

---

## VERDICT

**`∫Στᵢ²dt` is not an actuator-energy metric and it cannot be calibrated to watts as
it stands.** The array it integrates, `robot.get_qf()`, does not contain the actuator
torque. It contains the **gravity + Coriolis feed-forward that ManiSkill *writes into*
the simulator** each substep, and nothing else. It is therefore blind — by construction,
not by approximation — to contact, to stall, to grasp force, and to the PD drive torque
that actually moves the arm. The single claim the metric was introduced to support,

> "the only term here that sees a stall or a collision, because a servo pushing into a
> rigid constraint has large τ and ~zero ω"

is exactly backwards: it is the one quantity in the simulator that provably *cannot*
see a stall or a collision. Measured directly (§3.1): pair every stalled tick with a
freely-moving tick at the same end-effector position and the per-tick `Στ²` ratio is
**1.00 / 1.09 / 1.16 / 1.00** on egg / spoon / coke / drawer.

What the number does measure is real, but it is something else: **93–98 %** of it (§2.2) is
the static gravity-hold torque of the arm's own weight, integrated over the episode. It
is `duration × mean-squared arm extension`. On spoon that is literally what it tracks —
`corr(mean horizontal TCP reach, rms τ) = +0.97` (coke +0.88, drawer +0.70, egg +0.66).

Three of the four numeric claims in `ENERGY_RESULTS.md` do not survive the audit:

| claim | audit |
|---|---|
| "the only term that sees a stall or collision" | **false.** at ticks where the policy commands a translation and the TCP does not move, `Στ²` is 1.23–2.48× its freely-moving value — but hold the TCP position fixed and that collapses to **1.00–1.16×**. The gap was arm extension, not contact. §3.1 |
| spoon "monotone in cadence 1.00 → … → 2.59×" | **artifact.** the ideal arm finishes in 9.3 s and every other arm runs to the 12 s horizon. Paired per-episode and per-second: 1.00, 1.02, 1.04, 1.00, 1.02, 1.05, 1.02, 1.24, 1.35 — flat until the two 0 %-success arms. §3.4, §3.5 |
| "google_robot is flat" | **holds, but was never tested.** every published google ratio has a bootstrap CI containing 1.00; the cells cannot distinguish a 40 % effect from zero. The paired test (§3.5) does support flatness. |
| fingers "unweighted dominate by ~100×" | **wrong on widowx** (0.2× on egg, <0.01× on spoon), right in direction on google (21×, 13×). And the fingers are *excluded from ω² but included in τ²* — a real inconsistency, worth 0.01–0.10 % of τ². §2.1 |

Recommendation (§5): **do not ship this as energy.** Either drop the τ² column, or
re-run the traces logging the quantity that actually is the actuator torque. The three-
line change needed is in §5.2.

---

## 1. Is τ² the right proxy?

### 1.1 What `get_qf()` returns

`mani_skill2_real2sim/agents/base_controller.py:229-238`:

```python
def before_simulation_step(self):
    if self.balance_passive_force:
        qf = self.articulation.compute_passive_force(external=False)   # g(q) + C(q,q̇)q̇
    else:
        qf = np.zeros(self.articulation.dof)
    for controller in self.controllers.values():
        ret = controller.before_simulation_step()
        if ret is not None and "qf" in ret:                # NO controller in this repo
            qf = qf + ret["qf"]                            # returns a "qf" key
    self.articulation.set_qf(qf)
```

`set_qf` is called once per **sim substep** with the gravity/Coriolis feed-forward and
nothing else — grepping the tree, no controller returns a `qf` term. The torque that
moves the arm is a PhysX **articulation drive**
(`pd_joint_pos.py:35`, `set_drive_property(stiffness, damping, force_limit, "force")`);
PhysX applies it internally and it never passes through `qf`. `get_qf()` reads back what
`set_qf` last wrote. So

> **τ as logged = g(q) + C(q,q̇)q̇.  Not the actuator torque. Not post-gearbox. Not
> pre-gearbox. Not a torque any motor produces.**

Five independent checks on the traces confirm this reading — `energy_audit_02_provenance.py`,
`energy_audit_03_contact.py`:

| test | prediction if τ is the feed-forward | measured |
|---|---|---|
| **T1** tick 0 is logged before the first `step_action`; `BaseAgent.reset()` (`base_agent.py:159`) does `set_qf(zeros)` | τ ≡ 0 at tick 0 even though the arm is held against gravity | **0.000e+00** on every joint of all 864 episodes |
| **T2** `joint_head_tilt` — google's head never moves, so its hold torque is an analytic constant | one distinct non-zero value, bit-identical across tasks and arms | **−0.45250091 N·m**, 1 non-zero value in 13 634 / 19 784 / 21 113 ticks, identical in `coke_*` and `drawer_*` |
| **T3** vertical-axis joints (`waist`, `joint_torso`, `joint_head_pan`, all `axis="0 0 1"` off a world-aligned parent) carry **zero** gravity moment, so only Coriolis remains | τ → 0 as ‖ω‖ → 0 | `joint_torso`, slowest velocity decile: mean \|τ\| **0.00025 N·m** (drawer, i.e. while pushing a drawer shut); fastest decile 0.0091. `joint_head_pan`: **0.00000000 always** |
| **T4** the peak of \|τ_torso\| should land where the arm is *moving*, not where it is jammed | argmax coincides with ordinary-or-high ‖ω‖ | at argmax\|τ_torso\| (drawer): ‖ω‖ = 0.59 / 0.83 / 0.50 rad/s for lat0 / serial283 / cpu685 against episode means of 0.65 / 0.66 / 0.42 — at speed, never at a stall |
| **T5** magnitude relative to the configured `force_limit` | tiny — a gravity hold, not a drive | excluding the one-tick solver transients of §3.2, max \|τ\| over all 432 widowx episodes = **6.59 N·m**, and max \|τ\|/force_limit = **5.9 %**; google 68.9 N·m, **29.3 %** |

And the gripper: with the fingers stationary and closed on the object, mean \|τ_finger\|
is **0.015 N** (egg) / **0.018 N** (coke). A real gripper clamping a coke can holds tens
of newtons. The grasp force is not in the signal.

### 1.2 The unit question

Because τ is not a motor torque, the "copper loss ∝ τ²" chain has no first link.
For completeness the rest of the chain is also broken:

* **No gear ratio anywhere in the model.** For a real servo, motor current tracks
  *motor* torque = joint torque / (N · η). The sim has no N and no η. Every joint here
  is a direct-drive ideal revolute.
* **Summing τᵢ² across joints with R_i = 1 is dimensionally arbitrary.** Copper loss is
  `Σ (τᵢ/(Ktᵢ Nᵢ ηᵢ))² Rᵢ`; the shipped sum sets that whole per-joint coefficient to 1.
* Two joints carry the metric almost entirely: `shoulder` **62–90 %** and `elbow`
  **9–37 %**; every other joint is < 1 % (§2.1 table). So the implicit R_i=1 weighting
  is really just a shoulder:elbow mixing ratio.

### 1.3 How much does reweighting change the ranking?

`energy_audit_04_decompose.py §4`. Three alternatives, all on the median episode:

| variant | egg | spoon | coke | drawer |
|---|---|---|---|---|
| Spearman(τ², arm-joints-only) | +1.000 | +1.000 | +1.000 | +1.000 |
| Spearman(τ², Σ(τᵢ/force_limitᵢ)²) | +1.000 | +1.000 | +0.933 | **+0.650** |
| Spearman(τ², Σ(τᵢ/URDF_effortᵢ)²) | +0.983 | +1.000 | +1.000 | +1.000 |
| max rank shift, `1/force_limit²` | 0 | 0 | 1 | **4 of 9** |

On the widowx tasks the ranking is insensitive to the weights — because two joints carry
everything and the weight change only rebalances shoulder against elbow. On **drawer**
it is not: normalising by `force_limit` moves an arm four places and drops Spearman to
0.65, i.e. the published drawer ordering is a property of the weighting choice.

Note also that `force_limit` is itself not a physical rating. The widowx config uses
`[200, 200, 100, 100, 100, 100]` N·m; the URDF `<limit effort>` for the same joints is
`[10, 20, 15, 2, 5, 1]`, and the real arm's largest Dynamixel stalls at single-digit
N·m (§4). The config numbers are system-identification handles, **12–67×** the physical
rating. Normalising by them is not "normalising by the joint's capability".

---

## 2. What is missing

### 2.1 The finger/head inconsistency — real, and small

`fig_energy.py:75` `JG = {8: slice(0,6), 11: slice(0,7)}` is applied to the **ω²** term
only (`EF.append(((w**2).sum(0)*dt)[a].sum())`), while the τ² term is
`T2.append((qf**2).sum(1).sum()*dt)` — summed over **all** dof. So yes: the 2 finger
joints, and on google the 2 head joints, are excluded from `eff` and included in `t2`.
The stated reason ("unweighted they dominate by ~100×") is measured wrong for widowx:

| task | ∫Σω²dt, arm joints | excluded joints | ratio |
|---|---|---|---|
| egg | 16.86 | 4.02 | **0.24×** |
| spoon | 1.98 | 0.01 | **0.005×** |
| coke | 0.99 | 21.09 | 21.3× |
| drawer | 1.82 | 22.79 | 12.6× |

The justification given (`gripper_damping = 8.0`) is google's value; widowx's is 200.
For **τ²** the inconsistency is numerically harmless: the excluded joints contribute
**0.100 % (egg), 0.012 % (spoon), 0.069 % (coke), 0.093 % (drawer)**. Fix it for
correctness, not for the numbers.

Per-joint share of ∫Στ²dt, ideal arm:

```
egg      shoulder 62.26%  elbow 37.48%  wrist_angle 0.10%  fingers* 0.10%  rest <0.04%
spoon    shoulder 76.23%  elbow 23.73%  rest <0.02%
coke     joint_shoulder 88.18%  joint_elbow 11.20%  joint_bicep 0.38%  head_tilt* 0.07%
drawer   joint_shoulder 90.09%  joint_elbow  9.20%  joint_bicep 0.54%  head_tilt* 0.09%
                                                                 (* excluded from ω²)
```

### 2.2 Gravity holding torque — it is not "missing", it is *everything*

The question was whether τ² captures the static hold. It captures essentially nothing
else. Decompose each episode into a DC part (what a hypothetical arm frozen at its own
time-average pose would be charged) and the rest:

`t2_dc = duration · Σᵢ (mean_t τᵢ)²`

| task | DC share of ∫Στ²dt, across all 9 arms |
|---|---|
| egg | 93.9 – 98.1 % |
| spoon | 95.3 – 97.9 % |
| coke | 96.3 – 97.4 % |
| drawer | 93.2 – 94.8 % |

**93–98 % of the metric is a static pose-hold term.** Should it be baselined out? For a
"which schedule wastes energy" question, yes — as it stands the metric charges every arm
for the weight of the robot, per second, and the schedule only modulates the remaining
2–7 %. `joint_head_tilt`'s constant −0.4525 N·m is the reductio: it adds
0.2048 N²m²·s⁻¹ × episode length to google's "energy" while being, by construction,
independent of everything the policy did.

### 2.3 The PD drive torque — 1 to 4 orders of magnitude larger

`energy_audit_10_drive.py`. The stiffness half of the drive needs `q_target` and `q`,
neither logged. The **damping** half needs only `q̇`, which is logged, so
`dᵢ·q̇ᵢ` is a strictly recoverable *component* of the real drive torque:

| cell | ∫Στ_qf² dt | ∫Σ(d·q̇)² dt | ratio | ticks where \|d·q̇\| ≥ force_limit |
|---|---|---|---|---|
| egg_lat0 | 82.1 | 1 765 996 | **21 508×** | 48.2 % |
| egg_cpu685 | 118.2 | 6 884 883 | 58 264× | 62.7 % |
| spoon_lat0 | 35.8 | 131 299 | 3 670× | 30.6 % |
| coke_lat0 | 7 854 | 321 583 | 41× | 15.7 % |
| drawer_lat0 | 6 755 | 759 139 | 112× | 20.8 % |

Caveat, stated plainly: PhysX solves articulation drives implicitly, so `d·q̇` is *not*
literally the applied torque — the applied torque is clipped at `force_limit`, and the
config damping (330 N·m·s/rad on a joint limited to 200 N·m) is a system-ID handle, not
a physical damper. The defensible conclusion is the weaker one, and it is still decisive:
**the sim's own actuator torque is O(force_limit) = 100–300 N·m and saturating on a
large fraction of ticks, while the logged τ never exceeds 3.7 N·m on widowx.** Two
orders of magnitude, in the term that would carry the collision.

### 2.4 Inertial term — small, and honestly so

`energy_audit_09_missing.py`. Effective inertia about the widowx shoulder from URDF
`iyy` + parallel axis on the logged link frames, times `q̈` from differencing `q̇`:

| cell | I_eff (kg m²) | rms q̈ (rad/s²) | rms τ_inertia | rms τ_qf | ∫τ_in²/∫τ_qf² |
|---|---|---|---|---|---|
| egg_lat0 | 0.062 | 2.74 | 0.315 | 1.557 | 0.04 |
| egg_cpu685 | 0.068 | 5.75 | 0.424 | 1.880 | 0.05 |
| spoon_cpu685 | 0.075 | 1.82 | 0.209 | 2.395 | 0.01 |

`M(q)q̈` is 1–5 % of what is measured. The arm is light (2.14 kg of moving link) and
slow. This is not where the missing energy is.

### 2.5 Contact — not determinable from this data

No contact force, no constraint impulse, and no net-external-wrench is logged anywhere
in `trace_eval.py`. `compute_passive_force(external=False)` explicitly *excludes* the
external term. The magnitude of the missing contact work is **not determinable from
these traces**; recovering it requires re-running with `get_net_contact_forces()` or
`compute_passive_force(external=True)` logged.

### 2.6 The mobile base, drivetrain, friction, quiescent

* **Base.** Both google tasks instantiate `google_robot_static`
  (`grasp_single_in_scene.py:68`, `open_drawer_in_scene.py:43`) → `mobile_base=False` →
  `google_robot_meta_sim_fix_wheel_fix_fingertip.urdf`, in which the wheels are fixed
  joints. The 52.91 kg base is therefore not an actuated dof at all and contributes
  exactly zero to both metrics. That is correct for a static-base task, but it means any
  real-robot wattage would have to add the base's standby draw as a separate constant.
* **Drivetrain / friction.** `arm_friction = 0.0` in both configs, and the joints are
  direct-drive. A Dynamixel's no-load current at 12 V (0.15 A ⇒ ~1.8 W per servo) is a
  larger power term than the whole gravity hold on widowx (§4). Not modelled at all.
* **Electrical / quiescent.** Not modelled. On widowx this is the *dominant* real term.

### 2.7 Sampling — the tick grid is fine; the signal is not

Two separate things, and the docs conflate them.

**(a) google_robot's traces are not sampled at 37 ms.** `trace_eval.py` logs `get_qf()`
every *tick*, but `base.step_action()` only runs when `t % ACT_EVERY == 0`, and
`ACT_EVERY = 9` in `native` actuation. So the simulator does not advance on 8 of every 9
logged ticks and the same state is written 9 times. Measured
(`energy_audit_04_decompose.py §6`): coke has **81 distinct states behind 720 logged
ticks** → an effective sample period of **329 ms**, not 37 ms. Drawer: 114 / 1017 →
330 ms. The `dt = tick_ms/1000` bookkeeping still gives the right total time, so the
integral is not *wrong*; but the resolution claim in the docs is off by 9×.

**(b) The integral is nevertheless converged, because the integrand is smooth.**
Decimating the tick grid (`energy_audit_06_stall.py §3`):

| cell | stride 1 | ×2 | ×4 | ×8 | Δ(1→2) | Δ(1→4) |
|---|---|---|---|---|---|---|
| egg_lat0 | 82.1 | 82.0 | 81.6 | 81.0 | −0.1 % | −0.6 % |
| spoon_cpu685 | 92.8 | 92.5 | 92.1 | 92.3 | −0.3 % | −0.8 % |
| coke_lat0 | 7854.3 | 7845.6 | 7829.3 | 7796.0 | −0.1 % | −0.3 % |
| drawer_lat0 | 6754.5 | 6751.7 | 6760.0 | 6746.4 | −0.0 % | +0.1 % |

Sub-1 % out to 320 ms on widowx. So the documented worry — "`get_qf()` is sampled at
tick boundaries so brief impacts are under-counted" — is moot: the integrand contains no
impacts to under-count. It is a smooth function of pose. **The tick-sampled integral is
unbiased for the quantity it integrates; that quantity is the wrong one.**

---

## 3. Edge cases

### 3.1 Smashing into surfaces — the case the metric was chosen for, and it does not fire

A stall is observable without a contact log: the policy is *commanding* a translation and
the TCP is *not moving*. Using `applied_actions.npy[:, :3]` (the commanded
`world_vector`) and `ee_xyz.npy`, over all 216 episodes of each task
(`energy_audit_06_stall.py §1`):

* **STALL** = commanded ‖Δx‖ ≥ 80th pct **and** ‖d(ee)/dt‖ ≤ 20th pct of that subset
* **FREE**  = commanded ‖Δx‖ ≥ 80th pct **and** ‖d(ee)/dt‖ ≥ 80th pct

| task | n stall | n free | median Στ² STALL | FREE | **raw ratio** | median Σω² STALL / FREE |
|---|---|---|---|---|---|---|
| egg | 4 323 | 4 323 | 4.16 | 3.38 | 1.23 | 3.80 |
| spoon | 2 377 | 2 377 | 4.67 | 3.80 | 1.23 | 0.28 |
| coke | 554 | 554 | 247.35 | 154.66 | 1.60 | 0.15 |
| drawer | 825 | 825 | 317.25 | 127.79 | **2.48** | **0.00** |

(google's traces are decimated to their `ACT_EVERY = 9` distinct states first — 8 of
every 9 logged ticks have `d(ee) = 0` by construction, §2.7a, and would otherwise flood
the stall bin with non-events.)

Taken alone, the drawer row looks like the metric *does* fire: 2.48× at ticks where the
gripper is being commanded forward and the TCP is not moving at all (Σω² is literally
0.00). **It does not.** τ = `g(q) + C(q,q̇)q̇` is a function of pose and speed and of
nothing else (§1.1), so a stall/free gap can only be one of those two. And the arm stalls
*where it is extended* — pushing a drawer shut, reaching into the sink — which is exactly
where the gravity hold is largest. Control for it: pair each STALL tick with its nearest
FREE tick in TCP space, keep the pair only if the two end-effector positions are within
3 cm, and take the median ratio.

| task | raw ratio | matched pairs | **pose-matched ratio** |
|---|---|---|---|
| egg | 1.23 | 4 269 | **1.00** |
| spoon | 1.23 | 2 081 | **1.09** |
| coke | 1.60 | 100 | **1.16** |
| drawer | **2.48** | 113 | **1.00** |

Hold the end-effector position fixed and the entire effect disappears. What looked like
a collision signal was arm extension — which is what τ measures — and not contact, which
τ cannot see. **The metric does not fire on the case it was introduced for.**

### 3.2 Spike domination — yes, and it is solver noise, not contact

`energy_audit_05_spikes.py`. There exist single ticks where τ jumps three orders of
magnitude and returns on the very next tick:

```
egg_fp32_555 ep07, ticks 422-427 (shoulder column):
  t=422  τ = [-0.008  -1.647  -1.223 ...]   ‖ω‖ =  3.03
  t=424  τ = [-0.006  -1.638  -1.223 ...]   ‖ω‖ =  2.80
  t=425  τ = [-279.7  +464.7  +608.4 ...]   ‖ω‖ = 22.66     <-- one tick
  t=426  τ = [-0.017  -1.657  -1.217 ...]   ‖ω‖ =  6.92
```

`‖ω‖ = 22.7 rad/s` against a URDF velocity limit of 3.14 rad/s; in `ep09` a single tick
reaches **ω_wrist_rotate = −81 rad/s**, 26× the limit. These are PhysX/inverse-dynamics
transients, not contacts — a contact reaction is sustained while the arm pushes, a
solver transient lasts exactly one tick and vanishes.

Flag a tick when its `Στ²` exceeds 25× the episode median **and both neighbouring
ticks are below 5×** it — an excursion exactly one tick long. A contact reaction is
sustained for as long as the arm pushes, so this shape cannot flag one.

| task | flagged ticks | episodes affected | share of all ∫Στ²dt they carry | share of ∫Σω²dt |
|---|---|---|---|---|
| **egg** | **16** | 13 / 216 | **67.1 %** | 2.5 % |
| spoon | 0 | 0 | 0 % | 0 % |
| coke | 0 | 0 | 0 % | 0 % |
| drawer | 0 | 0 | 0 % | 0 % |

**Sixteen ticks out of 476 194 carry two-thirds of the eggplant task's total τ².**
What they do to the cell maxima:

| cell | max episode t2 | de-spiked | worst single episode |
|---|---|---|---|
| egg_fp32_555 | **27 381** | 136 | ep07, one tick = **99.6 %** of the episode |
| egg_p130w275 | 8 536 | 144 | ep16, one tick = 98.5 % |
| egg_pipe110fix | 565 | 141 | ep08, one tick = 81.9 % |

The "eggplant outlier near 27 000" that `fig_pareto_energy.py` calls a heavy tail is a
single-tick simulator blow-up. The median is largely protected — de-spiking moves the
published medians by −5.1 % (`egg_serial283`, 1.19× → 1.13×), −2.0 % (`egg_fp32_555`,
1.24× → 1.22×) and ≤ 0.2 % elsewhere — but **the box plots, the whiskers and the
strip-plot scatter in `fig_energy.png` are not protected**, and any mean-based statistic
over the eggplant cells is meaningless.

### 3.3 Force saturation — never, which is itself the finding

Across all 864 episodes and 476 194 logged ticks, the number of ticks at or above the
configured `force_limit` is **6** — and all 6 are inside the solver blow-ups of §3.2.
τ never saturates because τ is not the drive. Excluding those transients, the largest
\|τ\|/`force_limit` anywhere is **5.9 %** on widowx (6.59 N·m) and **29.3 %** on google
(68.94 N·m). Meanwhile the recoverable *damping* component of the
actual drive exceeds the limit on 30–65 % of widowx ticks (§2.3). So the answer to
"is τ² measuring the limit rather than the demand" is: it is measuring neither.

### 3.4 Episode length — this is most of the published effect

`energy_audit_04_decompose.py §1-2`. Pooled over the 216 episodes of a task:

| task | r(t2, duration) | R² of the one-parameter model `t2 = k · duration` |
|---|---|---|
| egg | **+0.846** | **0.705** |
| spoon | +0.488 | 0.223 |
| coke | **+0.924** | **0.826** |
| drawer | +0.713 | 0.503 |

(de-spiked; `energy_audit_07_stats.py`). On coke, 83 % of the variance in the "energy"
metric is explained by episode length alone, with a single scalar. The mechanism is that
a success terminates the episode:

```
egg_lat0    successes:  t2 median  23.6  (7.0 s)    failures:  92.8  (24.0 s)
spoon_lat0  successes:  t2 median  32.4  (7.2 s)    failures:  82.1  (12.0 s)
coke_lat0   successes:  t2 median 1465.8 (8.5 s)    failures: 8412.0 (26.7 s)
```

The distributions are strictly **bimodal**, and the two modes differ by ~4–6×. The
published statistic is a *median over that bimodal mixture*, so it is close to a step
function of the success rate: whichever mode the 12th-of-24 episode lands in is what
gets printed. `egg_p150w300` (50 % success) prints **0.81×** — *cheaper* than the ideal
arm at 37.5 % — purely because its median episode is nearly a success. This is not
"unconditioned on success"; it is conditioned on success in the most brittle way
available.

Integral vs. rate, on spoon, where it matters most:

| arm | median duration | published `t2 ×` | **per-second `×`** |
|---|---|---|---|
| lat0 | 9.30 s | 1.00 | 1.00 |
| pipe110fix | 12.00 s | 1.38 | **1.02** |
| p105w300 | 12.00 s | 1.37 | **0.96** |
| p130w275 | 12.00 s | 1.28 | **0.93** |
| p150w300 | 12.00 s | 1.22 | **1.06** |
| pipe200fix | 12.00 s | 1.42 | **1.03** |
| serial283 | 12.00 s | 1.62 | **1.04** |
| fp32_555 | 12.00 s | 2.03 | 1.30 |
| cpu685 | 12.00 s | 2.59 | 1.67 |

The published "monotone 1.00 → 2.59×" is, to within 6 %, the statement that the ideal
arm finishes in 9.3 s and everyone else hits the 12 s horizon. That is **mission time** —
which the same document says is plotted separately. Spread across the 9 arms:

| task | published | per-second |
|---|---|---|
| egg | 1.77× | 1.40× |
| spoon | **2.59×** | **1.79×** |
| coke | 1.44× | 1.31× |
| drawer | 1.30× | 1.19× |

### 3.5 The statistics under the plot

`energy_audit_07_stats.py`, `energy_audit_08_paired.py`. 20 000-resample bootstrap CI on
each published median ratio. The self-ratio of the ideal arm against itself is
**1.00 [0.45, 2.17]** on egg — a factor of 4.8 — because of the bimodality in §3.4. With
that noise floor, on **drawer every one of the nine published ratios has a CI containing
1.00**, and on coke only `cpu685` comes close to excluding it. The "google is flat"
finding is currently an absence of statistical power, not a measurement of flatness.

There is a much better test available and it is not being used: **every cell of a task
uses `init_rng=100` and episode ids 0…23, so episode *i* starts from the same scene in
every arm.** Pairing on episode id:

| task | arm | unpaired (published) | **paired** [95 % CI] | **paired, per-second** [95 % CI] |
|---|---|---|---|---|
| egg | p150w300 | 0.81 | 1.06 [0.73, 1.57] | **0.99 [0.92, 1.15]** |
| egg | serial283 | 1.13 | 1.28 [0.94, 3.32] | **1.10 [0.94, 1.22]** |
| egg | fp32_555 | 1.22 | 1.32 [1.17, 1.77] | **1.07 [0.92, 1.24]** |
| egg | cpu685 | 1.44 | 1.59 [1.24, 3.17] | 1.37 [1.19, 1.55] |
| spoon | pipe110fix | 1.38 | 1.19 [1.08, 1.43] | **1.02 [0.95, 1.10]** |
| spoon | serial283 | 1.62 | 1.40 [1.14, 1.67] | **1.02 [0.95, 1.22]** |
| spoon | fp32_555 | 2.03 | 1.63 [1.26, 2.01] | 1.24 [1.08, 1.41] |
| spoon | cpu685 | 2.59 | 1.66 [1.38, 2.23] | 1.35 [1.19, 1.57] |
| coke | p150w300 | 0.72 | 0.97 [0.44, 1.06] | 0.96 [0.79, 1.06] |
| coke | cpu685 | 1.04 | 1.08 [0.98, 1.64] | 1.08 [0.98, 1.20] |
| drawer | pipe200fix | 1.20 | 0.98 [0.88, 1.19] | 0.98 [0.94, 1.08] |
| drawer | cpu685 | 0.98 | 0.98 [0.82, 1.27] | 0.95 [0.82, 1.09] |

(full 36-row table: `python3 energy_audit_08_paired.py`)

Pairing removes every "cheaper than the ideal arm" result — those were scene-assignment
artifacts — and shrinks the effect to this:

> **After pairing on the shared scene, removing solver transients, and dividing out
> mission length, the only arms on any of the four tasks whose actuator-effort *rate*
> differs significantly from the ideal arm are `cpu685` (egg 1.37 [1.19, 1.55], spoon
> 1.35 [1.19, 1.57]) and `fp32_555` on spoon (1.24 [1.08, 1.41]) — and both of those
> succeed 0 % of the time. The other 28 arm×task cells sit at 0.91–1.10 with a CI
> containing 1.00.**

That residual effect is real and paired-significant, but its content is "an arm that
never completes the task holds its arm extended over the sink for the full 24 s horizon
instead of reaching in and stopping". It is a statement about the failure mode, not
about actuator energy.

---

## 4. Calibration to real units

Short answer: **no.** One narrow physical quantity can be calibrated for widowx, and it
is not the metric; for google_robot nothing at all is determinable.

### 4.1 The hardware numbers, and what is missing from them

The WidowX-250 S (`wx250s`) carries **7 × XM430-W350 + 2 × XL430-W250** on a
**12 V / 5 A** supply. `shoulder` and `elbow` are each **dual**: a shadow servo mirrors
the master in hardware (`Secondary_ID: 2` / `4`, `Drive_Mode: 1` in
`interbotix_xsarm_control/config/wx250s.yaml`), so each of those joints has two motors
sharing the load. (The XM540-W270 named in the audit brief is **not** on this arm — it
appears on the larger ViperX/WidowX-350 class.)

| | XM430-W350 @ 12.0 V | XL430-W250 @ 12.0 V |
|---|---|---|
| stall torque / current | 4.1 N·m at 2.3 A | 1.5 N·m at 1.4 A |
| ROBOTIS' printed output constant | **1.783 N·m/A** | **1.071 N·m/A** |
| gear ratio | 353.5 : 1 | 258.5 : 1 |
| motor | coreless | cored |
| standby current | 40 mA | 52 mA |
| **winding resistance R** | **not published** | **not published** |
| **motor torque constant Kt** | **not published** | **not published** |
| **no-load current** | **not published** (no such row) | **not published** |
| peak efficiency (read off the JPEG performance graph) | ≈ 43 % at ≈ 0.85 N·m | ≈ 24 % at ≈ 0.24 N·m |

ROBOTIS publishes no winding resistance and no motor-level Kt for **any** X-series part,
and no third party has measured them. (Oaki et al., arXiv:2605.15949, state
2.3179 N·m/A for the XM430-W350 with no citation and no stated definition — 30 % above
ROBOTIS' own printed figure for the same part at the same voltage. Do not mix the two.)

R must therefore be **inferred** as `V_stall / I_stall`, which lumps in the H-bridge
drop and wiring and is an **upper bound** on the winding: XM430 **5.22 Ω**
(5.29 / 5.48 Ω at 11.1 / 14.8 V — reassuringly consistent), XL430 **8.57 Ω**.

Note also what this makes of the sim's `force_limit`: the config uses `[200, 200, 100,
100, 100, 100]` N·m for joints whose real servos stall at `[4.1, 8.2, 8.2, 4.1, 4.1,
1.5]`. The configured limits are **12–67× the physical rating**. They are
system-identification handles. So is the URDF `<limit effort>` (`[10, 20, 15, 2, 5, 1]`,
2–2.5× the real servo on the big joints).

### 4.2 The conversion, and the one number it produces

With `K` the output-referenced stall constant and `R = V/I_stall`, the stall condition
`V·I = I²R` closes, so for a joint whose `n` servos split τ evenly:

```
    P_copper,i(τ) = n · ((τ/n)/K_i)² · R_i  =  c_i · τ²,      c_i = R_i / (n_i K_i²)
```

| joint | servo | n | c_i [W per (N·m)²] |
|---|---|---|---|
| waist | XM430 | 1 | 1.641 |
| **shoulder** | XM430 | **2** | **0.821** |
| **elbow** | XM430 | **2** | **0.821** |
| forearm_roll | XM430 | 1 | 1.641 |
| wrist_angle | XM430 | 1 | 1.641 |
| wrist_rotate | XL430 | 1 | **7.473** |

**The per-joint coefficient spans 9.1×**, against the shipped metric's implicit
`c_i = 1`. That is the direct answer to "how wrong is R_i = 1": nine-fold, per joint.
It happens not to matter much here only because `shoulder` and `elbow` carry 99.7 % of
the metric and share the same coefficient — which is also why the §1.3 reweighting
barely moved the widowx ranking. Fingers are excluded: the sim models them as prismatic
joints in newtons and the rack pitch radius needed to convert N to servo N·m is not
published (their τ is 0.01–0.09 N regardless).

Adding the quiescent floor — 7 × 40 mA + 2 × 52 mA = 384 mA at 12 V = **4.61 W**, all
servos powered, torque disabled — `energy_audit_12_calibration.py` gives:

| cell | P_copper (W) | + quiescent (W) | E per episode (J) | published `t2 ×` | **calibrated E ×** |
|---|---|---|---|---|---|
| egg_lat0 | 3.01 | 7.61 | 178.0 | 1.00 | 1.00 |
| egg_serial283 | 3.48 | 8.09 | 190.6 | 1.19 | **1.07** |
| egg_cpu685 | 4.04 | 8.65 | 207.6 | 1.44 | **1.17** |
| spoon_lat0 | 3.80 | 8.41 | 74.5 | 1.00 | 1.00 |
| spoon_serial283 | 3.96 | 8.57 | 102.8 | 1.62 | **1.38** |
| spoon_cpu685 | 6.35 | 10.95 | 131.5 | 2.59 | **1.76** |

So even taking the metric at face value, the moment a real power budget is attached the
headline ratios shrink by a third to a half, because a 4.6 W constant sits under a 3–6 W
signal.

### 4.3 Why the number above is not the answer, stated plainly

**Assumptions, all of them:** τ is a joint torque a real servo would have to produce
(false, §1.1); the sim's link masses are the real arm's (unverifiable — Trossen does not
publish a mass for the stock `wx250s`; the ALOHA variant's datasheet says 3.8 kg, a
third-party aggregator says 2.81 kg, and the SimplerEnv URDF sums to 2.14 kg of moving
link); the dual joints split torque evenly (good — the shadow mirrors in hardware);
K is ROBOTIS' printed stall figure; R = V/I_stall; the gripper is excluded.

**Uncertainty on the copper term itself:** R is an upper bound — if half the stall drop
is in the driver, the copper term halves (×0.5). Using Oaki et al.'s K instead cuts it
41 % (×0.59). Combining: `P_copper` is **3.0 W in [1.2, 3.0]** on `egg_lat0` and
**4.0 W in [1.6, 4.0]** on `egg_cpu685`.

**What is outside that range because it is not determinable:**

* The τ being converted is the **simulated gravity/Coriolis feed-forward**, not the real
  arm's joint torque. The real servo additionally produces the tracking torque, the
  contact torque and the grasp force — none of which is in the array.
* **No-load / friction current: not published** for any X-series part. Read off the
  XM430's performance-graph JPEG it is ≈ 0.20 A ≈ 2.4 W per servo while moving —
  plausibly **larger than the entire gravity-hold copper loss**. Not usable as a number,
  and not ignorable either.
* ROBOTIS does not define whether "standby" is torque-enabled or torque-disabled, so
  4.61 W is a floor, not an estimate. Neither the U2D2 nor the power-hub board publishes
  a quiescent draw.
* Gear-train efficiency is not published; the ~72 % implied by
  (Kt_motor × N) / K_published is itself derived, not measured.

**google_robot: not determinable from any available source.** RT-1 (arXiv:2212.06817)
describes the hardware in a single clause — "a 7 degree-of-freedom arm, a two-fingered
gripper, and a mobile base" — and its action space is Cartesian, so there are no joint
torque limits to specify. No motor type, torque constant, gear ratio, joint torque limit
or arm power figure is published anywhere. The SimplerEnv URDF sets `effort="10.0"`
**uniformly on every joint, wheels included** — a placeholder, not a rating — and has no
`<transmission>` or `<actuator>` tags at all. The only physically meaningful numbers in
the asset are the link masses. **Any watt figure for coke or drawer would be invented.**

### 4.4 Verdict on calibration

Calibration is **not defensible**, for one decisive reason and several supporting ones.
The decisive one: the quantity being converted is not a motor torque, so the conversion
has nothing to convert. Even if it were, half the widowx power budget (friction, no-load,
controller) is unpublished, and google_robot is a complete blank. The honest
normalisation available today is dimensionless and is given in §5.1.

---

## 5. Recommendation

**ONE recommendation: none of (a), (b) or (c). Do (d) — rename and re-scope the
existing column to the quantity it actually is, report it per-second and paired, and
re-run the traces to get the real actuator torque before the paper claims "energy"
again.**

All three offered options presuppose that `get_qf()` is an actuator torque. It is not
(§1.1). (a) keeping it "as a dimensionless proxy with stated caveats" would require the
caveat "this quantity is blind to contact, stall, grasp force and the drive torque",
which is not a caveat, it is a retraction. (b) reweighting cannot add information that
was never in the array; on three of four tasks it does not even change the ranking
(§1.3). (c) is ruled out in §4.

### 5.1 What to report from the existing traces, now

Rename the column **"gravity-hold effort (dimensionless)"** — never "energy", never
"joules", never "copper loss" — and compute

```
                    1     ⌠T_ae      ⎛ τ_i(t) ⎞²
    GHE(a,e)  =  ───────  ⎮      Σ   ⎜ ────── ⎟   dt
                  T_ae    ⌡0    i∈arm ⎝  τ̂_i  ⎠

    report:   median_e [ GHE(a,e) / GHE(ideal,e) ]   with a paired-bootstrap 95 % CI
```

with, in order of how much each one matters:

1. **τ named honestly.** τ = `get_qf()` = `g(q) + C(q,q̇)q̇`. Say so in the caption.
2. **Paired on `e`.** Every cell of a task shares `init_rng=100` and episode ids 0…23,
   so episode *e* starts from the same scene in every arm. Not pairing is what produced
   the "cheaper than the ideal arm" entries (§3.5).
3. **Divided by `T_ae`.** Otherwise the column is mostly mission time (§3.4), which the
   same document says is plotted separately.
4. **Arm joints only** — `JG` applied to *both* terms, fixing the §2.1 inconsistency.
5. **Normalised by the real servo stall torque** — for widowx,
   `τ̂ = [4.1, 8.2, 8.2, 4.1, 4.1, 1.5]` N·m (the dual shoulder and elbow get 2 × 4.1;
   §4.1). Not the agent-config `force_limit`, which is 12–67× the physical rating, and
   not the URDF `<limit effort>`, which is 2–2.5× it on the big joints and 0.67× on
   `wrist_rotate`. **On google_robot no rating is published by anyone** (§4.3), so
   `τ̂ = 1` there and the
   google column is an unweighted arm-joint sum — say so, and do not compare the two
   embodiments' absolute values. This choice moves an arm 4 places on drawer (§1.3), so
   it must be stated rather than defaulted.
6. **Isolated single-tick solver transients removed** (§3.2), and the removal disclosed.
7. **A bootstrap CI on every number.** Without one, 28 of 32 cells read as effects.

Run: `python3 energy_audit_11_proposed.py`. The result:

| task | arms whose CI excludes 1.00 | value | success rate of those arms |
|---|---|---|---|
| egg | `p130w275`, `cpu685` | 1.10 [1.06, 1.24], **1.37 [1.19, 1.56]** | 37.5 %, **0 %** |
| spoon | `fp32_555`, `cpu685` | 1.24 [1.08, 1.41], **1.35 [1.23, 1.57]** | **0 %**, **0 %** |
| coke | *none* | — (0.91–1.08, all CIs contain 1.00) | — |
| drawer | *none* | — (0.94–1.06, all CIs contain 1.00) | — |

That is the whole defensible effect: **28 of 32 scheduled arm×task cells are
indistinguishable from the ideal arm, and the four that are not are the arms that never
finish the task.** It is a much weaker claim than the current figure makes, and it is
the one the data supports.

(The `p130w275` egg cell at 1.10 [1.06, 1.24] is a lone significant mid-band point with
no neighbours; at 32 cells and α = 0.05 it is roughly what one expects by chance. The
two `cpu685` cells and `spoon fp32_555` are the effect.)

### 5.2 The re-run that would let the paper say "energy"

Three additions to the episode loop in `trace_eval.py`. This is for the next sweep —
the shipped file was deliberately not touched by this audit.

```python
# ---- sketch, not a patch. All three APIs exist in SAPIEN 2.2.2 and are already
# ---- used elsewhere in this tree.

# 1. THE ACTUAL DRIVE TORQUE -- the term that carries a stall.
#    K / D / FLIM are the arrays already in the agent config; get_drive_target() is
#    used at mani_skill2_real2sim/utils/sapien_utils.py:374.
q, qd = robot.get_qpos(), robot.get_qvel()
qt = robot.get_drive_target()
tau_drive_log.append(np.clip(K * (qt - q) - D * qd, -FLIM, FLIM))

# 2. CONTACT -- the thing the metric was supposed to see.
#    scene.get_contacts() is used at grasp_single_in_scene.py:411 and
#    put_on_in_scene.py:96. Contact.points[k].impulse is a 3-vector impulse over the
#    substep, so |sum of impulses| / substep_dt is the net contact force on the link.
links = set(robot.get_links())
f = 0.0
for c in env._scene.get_contacts():
    if c.actor0 in links or c.actor1 in links:
        f += np.linalg.norm(np.sum([pt.impulse for pt in c.points], axis=0)) / substep_dt
contact_force_log.append(f)

# 3. MECHANICAL WORK -- unambiguous, and it needs no motor model at all.
#    W_grav from the link COMs already logged; W_drive = INT tau_drive . qd dt.
```

Better still, accumulate (1) and (3) inside `before_simulation_step` so they integrate
at the 500/513 Hz sim rate rather than the tick rate — at which point the "brief impacts
are under-counted" caveat becomes true and worth stating, which today it is not (§2.7b).

With `tau_drive` logged, `Σ(τ_drive,i / (Kt_i N_i η_i))² R_i` is a defensible copper-loss
model for widowx and the calibration in §4 becomes worth doing. Without it, no
reweighting, no normalisation and no motor constant can rescue the column.

### 5.3 What to change in the prose regardless

* Delete "the only term here that sees a stall or a collision" from `ENERGY_RESULTS.md`,
  `fig_energy.py` (docstring + figure footer) and `fig_pareto_energy.py`. It is false
  (§3.1) and it is the sentence the whole metric rests on.
* Delete "copper loss (I²R) proxy" for the same reason.
* Fix "Fingers excluded … unweighted they dominate by ~100×": measured, they are 0.24×
  on egg and 0.005× on spoon, and 21×/13× on google (§2.1). The `gripper_damping = 8.0`
  cited as the reason is google's value; widowx's is 200.
* Fix "sampled at tick boundaries so brief impacts are under-counted": the integral is
  converged to <1 % out to a 320 ms stride (§2.7b), and google's traces are already at
  330 ms effective resolution because the sim only advances every 9th logged tick.
* The `fig_pareto_energy.py` "eggplant outlier near 27,000" is a single-tick PhysX
  transient, not a heavy tail (§3.2).

---

## 6. Files

| file | what it establishes |
|---|---|
| `energy_audit_01_scan.py` | per-episode, per-joint reduction of `traces_torque2/` → `energy_audit_cache.json` |
| `energy_audit_02_provenance.py` | T1–T5: `get_qf()` is the gravity/Coriolis feed-forward |
| `energy_audit_03_contact.py` | vertical-axis joints vanish as ‖ω‖→0; grasp force absent |
| `energy_audit_04_decompose.py` | duration coupling, per-joint shares, reweighting, saturation, aliasing |
| `energy_audit_05_spikes.py` | solver transients and what they carry |
| `energy_audit_06_stall.py` | **the stall test and its pose control**, DC/AC split, decimation |
| `energy_audit_07_stats.py` | bootstrap CIs, de-spiked clock test, spread |
| `energy_audit_08_paired.py` | bimodality, and the paired-on-scene test |
| `energy_audit_09_missing.py` | inertial term, head-tilt constant |
| `energy_audit_10_drive.py` | the PD drive torque that is not in the metric |
| `energy_audit_11_proposed.py` | the replacement statistic |
| `energy_audit_12_calibration.py` | the Dynamixel conversion, and its uncertainty budget |

Reproduction check: the audit pipeline reproduces the published medians exactly —
egg 82.1, spoon 35.8, coke 7854.3, drawer 6754.5 for the ideal arm, matching the table
in `ENERGY_RESULTS.md`. Everything above is a different reading of the same numbers,
not a different number.


---

*Audit performed 2026-09-08 against `traces_torque2/` as committed. `fig_energy.py`,
`make_energy_cache.py`, `ENERGY_RESULTS.md` and `trace_eval.py` were not modified.
`energy_audit_01_scan.py` writes `energy_audit_cache.json` (1.0 MB); nothing else here
writes outside this directory.*
