# Fine-grained stepping and actuation on the google_robot envs

Why `google_robot` cannot simply be run at the ~40 ms control grid that `widowx` runs at,
what each obstacle actually is, and which of them are fixable.

Target: **27 Hz = 37.04 ms**. `sim_freq = 513 = 3^3 x 19`, and the harness requires a tick
that divides `sim_freq` and is a multiple of the native `control_freq = 3`, so the legal
ticks are 3 / 9 / 27 / 57 / 171 / 513 Hz. 27 Hz is the only one near widowx's 25 Hz
(40 ms) — an 8% mismatch, close enough to call parity.

Reference baselines, `octo-small`, zero latency, seed 80, 24 episodes:
`close_drawer` **10/24 = 41.7%**, `pick_coke_can` **10/24 = 41.7%**. Both reproduced
exactly by this harness through `--control-mode` with the registered mode, which is what
makes every comparison below interpretable.

## Registered configurations

| | widowx | google_robot |
|---|---|---|
| control_mode | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | `arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner` |
| arm delta anchoring | `use_target=True` — accumulating | `use_target=False` — **pose-relative** |
| arm trajectory | direct PD to target | **planner**, vel/acc/jerk limited (1.5 / 2.0 / 50) |
| gripper | absolute normalised position, stateless | **delta**, driven by a sticky counter |
| control_freq / sim_freq | 5 / 500 | 3 / 513 |
| sim steps per control period | 100 | 171 |

## The three mechanisms

### 1. Pose-relative anchoring discards tracking lag

`use_target=False` means every control step re-anchors the commanded delta to where the
arm **currently is**. Any lag between commanded and achieved pose is silently dropped.

widowx's `use_target=True` accumulates instead, so five steps of 0.2*delta leave the
target exactly where one step of delta would. That is what makes its fine-tick rescaling
sound, and it is the property google's registered controller lacks.

**CORRECTION (this superseded an earlier claim in this document).** I first wrote that
pose-relative anchoring is what makes *refinement* fail. The data says otherwise, and so
does the algebra. Under a first-order tracking model, a pose-relative controller commands
`pose + d/N` each of N substeps and achieves `f*d/N`, so total travel is `f*d` —
**independent of N**. It undershoots by the tracking factor `f` at every rate equally.

What `use_target=False` actually explains is why the controller cannot be SWAPPED: the
undershoot is baked into the real2sim calibration, so restoring it (B, C, G) moves the
zero-latency baseline at the NATIVE rate, where refinement is not yet involved.

The N-dependence — the thing that breaks refinement — is mechanism 2.

### 2. Planner path truncation

The planner produces a time-optimal path whose length is set by the vel/acc/jerk limits,
not by the control period, and `before_simulation_step` walks only
`sim_freq / control_freq` steps of it before the next control step re-plans
(`pd_joint_pos.py:100-160`). Natively that is 171 steps; at 27 Hz it is **19 (11%)**.

Scaling the target down does not compensate: under acceleration limits a 1/9-distance
move takes roughly 1/3 the time, not 1/9, so a proportionally larger fraction of the plan
is left unexecuted. Target accumulation does **not** fix this — the target keeps
advancing while the arm falls progressively further behind.

### 3. The sticky gripper hold is a COUNT, not a duration

SimplerEnv's octo wrapper closes the google gripper by repeating the open->close
transition for `sticky_gripper_num_repeat = 15` control steps
(`simpler_env/policies/octo/octo_model.py:40`; widowx is `= 1`, i.e. no hold at all,
because its gripper is an absolute position). The harness replays this verbatim,
advancing once per actuation.

15 is a count, but closing a gripper needs a fixed wall clock:

| | sticky hold |
|---|---|
| native 3 Hz | 15 x 333 ms = **5.00 s** |
| fine 27 Hz, unscaled | 15 x 37 ms = **0.56 s** |

Refining actuation 9x shortens the grasp window 9x. `close_drawer` only pushes and is
unaffected; `pick_coke_can` needs the grasp. This is the same split as the original
Bug 2, and it would have produced a misleading "parity fails on coke".

**Fixed**: `--sticky-repeat auto` = `15 x ACT_HZ / NATIVE_CF`, holding the window at
5.00 s. Every native-rate run has `ACT_HZ == NATIVE_CF`, so the factor is exactly 1 and
nothing already measured changes. widowx has no sticky machine, so it is unaffected.

## What was tried, and what the data says

All at zero latency, seed 80, n=24 (F/F0 n=12). Gripper held at the registered planner
variant throughout, so the ARM controller is the only variable. The sticky fix was active
in every fine-27 run (135 actuations x 37.04 ms = 5.00 s, printed in each log), so
mechanism 3 is excluded as a cause of any collapse below.

| | actuation | arm controller | sim_freq | drawer | coke |
|---|---|---|---|---|---|
| A | native 3 Hz | registered (pose-rel + planner) | 513 | **10/24 = 41.7%** | **10/24 = 41.7%** |
| B | native 3 Hz | target + planner | 513 | 4/24 = 16.7% | 8/24 = 33.3% |
| C | native 3 Hz | target, no planner | 513 | 6/24 = 25.0% | **1/24 = 4.2%** |
| D | fine 27 Hz | target + planner | 513 | 0/24 | 0/24 |
| E | fine 27 Hz | target, no planner | 513 | 7/24 = 29.2% | **0/24** |
| F0 | native 3 Hz | registered | 4617 | — | 3/12 = 25.0% |
| F | fine 27 Hz | registered | 4617 | — | 0/12 |

Both A references reproduce the stock baselines exactly, so the harness and the
`--control-mode` path are sound and every other row is interpretable.

### Every route to 40 ms fails, and each fails at its own control

**Controller swap (B, C).** Neither alternative reproduces the reference *at the native
rate*, where actuation is not yet a variable: C collapses coke to 1/24 and B moves both
tasks far outside the measured +/-2.50 pt noise floor. SIMPLER's google configuration is
tuned as a unit — controller, PD gains and planner limits together — so substituting the
arm controller changes the simulation regardless of the control rate. D and E are
therefore uninterpretable as parity tests: they are built on a controller that already
fails at 3 Hz.

Note E vs C on drawer (7/24 vs 6/24): for a target-accumulating, planner-free controller
the refinement itself is roughly neutral. **Refinement is not the problem — the
controller that permits refinement is.** On coke even that controller gives 0/24, because
grasping needs the fidelity the swap destroys.

**sim_freq scaling (F0, F).** Raising `sim_freq` 9x is NOT neutral at the native rate:
F0 gives 3/12 = 25.0% against A's 41.7%. The physics rate is itself part of the validated
configuration (contact resolution and the PD's effective stiffness per step both move
with it), so F cannot be attributed to truncation. Route abandoned.

**Registered controller at the fine tick (the original attempt).** 0/24 — mechanism 2,
compounded by the mechanism-1 undershoot it already carries.

**Non-planner gripper (G, H).** The last untested lever: E kept the registered planner
gripper, which truncates like the arm. Swapping it for `gripper_pd_joint_target_delta_pos`
gives coke **2/12 = 16.7% at the NATIVE rate** against A's 41.7%, and 0/12 at 27 Hz — it
fails its own control too. Four independent routes, four failures at the native rate.

### ROOT CAUSE, in one line

**The planner's path length is set by physics limits, not by the control period.** A
time-optimal move of distance `d` under acceleration limit `a` takes `t = 2*sqrt(d/a)`, so
subdividing the control period by N shrinks the distance by N but the plan duration only
by `sqrt(N)`. The executed fraction therefore falls by `sqrt(N)`: at the native 3 Hz the
arm completes ~75% of each planned move, at 27 Hz only ~25%. It re-plans from wherever it
reached, every period, and never traverses a full command.

widowx has no planner — `arm_pd_ee_target_delta_pose_align2` is a direct PD onto an
accumulating target — so there is no path to truncate and subdividing is exact. That, and
not `use_target`, is the asymmetry.

The empirical proof is E vs C, which differ ONLY in control rate and share the
planner-free controller: **7/24 at 27 Hz vs 6/24 at 3 Hz — refinement is neutral.** With
the planner present, the same comparison is D vs B: **0/24 vs 4/24 — refinement is fatal.**

### Conclusion

`google_robot` **cannot be run at a ~40 ms control grid while remaining comparable to
SIMPLER's validated real2sim configuration.** Every route that reaches 27 Hz also changes
the zero-latency native baseline, which is the thing all published numbers are anchored
to. This is a property of the SIMPLER google configuration, not a harness limitation.

The mechanisms are now understood well enough to say *why*, and to say it is not a bug to
be fixed: mechanism 1 is intrinsic to `use_target=False`, mechanism 2 is intrinsic to a
planner whose path length is set by physics limits rather than the control period, and
both are load-bearing for the real2sim tuning. Mechanism 3 was a genuine harness bug and
is fixed.

### The valid alternative

Parity of RATE is unobtainable; parity of METHOD is not. Both embodiments can be run at
their OWN native control rate — widowx 5 Hz / 200 ms, google 3 Hz / 333 ms — which is
the faithful model for each and needs no new validation, since `native` is what the
registered configuration already does. That tests the same hypothesis (does the
cadence effect survive when the policy can no longer outrun the actuator?) without
moving either simulation off its validated configuration.

## Harness flags added

| flag | purpose |
|---|---|
| `--control-mode` | override the registered control_mode (validated: reproduces both references exactly) |
| `--sim-freq` | override the physics rate, to isolate path truncation |
| `--sticky-repeat` | `auto` scales the gripper hold to keep its wall-clock duration fixed |
| `--allow-lag-discard` | permit a pose-relative controller under fine rescaling, for measurement only |

The pre-existing guard was also corrected. It refused any `interpolate_by_planner` mode
under fine rescaling; the real hazard is a **pose-relative** arm controller, so it now
keys on that, and splits the mode string on `_gripper_` first because the gripper half
legitimately contains `target_delta` and would otherwise mask a pose-relative arm.
