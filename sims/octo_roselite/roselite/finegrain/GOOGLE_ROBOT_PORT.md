# The fine-grain harness on google_robot tasks

Status: **FIXED and validated** (2026-09-06). `finegrain_eval.py` now reproduces
the stock harness at zero latency on both google_robot tasks. Two independent
bugs had to be fixed; the second one is not mentioned in the original diagnosis
below and was found while validating the first.

Harness at the fix: md5 `b9be655f74a6`. Widowx behaviour is **unchanged** --
verified, see "Widowx regression" below.

---

## The two bugs

### Bug 1 -- fine-tick actuation under a planner-interpolated controller

The fine harness modelled "sim ticks faster than results arrive" by raising the
env's `control_freq` to the tick rate and scaling the position deltas by
`tick_ms / action_dt_ms`. That is sound for widowx, whose control mode is

    arm_pd_ee_target_delta_pose_align2

a pure INTEGRATOR of the deltas, so nine steps of one ninth the delta is a good
approximation of one full step. The google_robot tasks use

    arm_pd_ee_delta_pose_align_interpolate_by_planner
    gripper_pd_joint_target_delta_pos_interpolate_by_planner

(`open_drawer_in_scene.py:47`, `grasp_single_in_scene.py`, `move_near_in_scene.py`).
Here the controller PLANS a trajectory to the target over the control period,
subject to velocity and acceleration limits. Shrinking the control period to
37 ms while shrinking the target proportionally does not reproduce the motion --
the planner is time-limited and the arm barely moves.

MEASURED, `google_robot_close_drawer`, octo-small (1.0), zero latency, 24 eps:

| harness | tick | actuation | delta scale | seed | success |
|---|---|---|---|---|---|
| stock `run_eval.py` | 3 Hz | 3 Hz | n/a | 80 | **10/24 = 41.7%** |
| `finegrain_eval.py --tick-hz 3` | 3 Hz | 3 Hz | 1.0 | 80 | 9/24 = 37.5% |
| `finegrain_eval.py` (old default) | 27 Hz | **27 Hz** | **1/9** | 80 | **0/24 = 0.0%** |

**The fix.** A new `--actuation {auto,fine,native}` flag, defaulting to `auto`
= `native` for google_robot and `fine` for widowx:

* `fine` -- the pre-existing model. The env runs at the fine tick, deltas are
  rescaled. Unchanged, and still the default for widowx.
* `native` -- the env runs at its OWN `control_freq` and the deltas are applied
  unscaled. The fine grid still carries the whole dispatch/arrival schedule; it
  only decides WHICH result is current at each native step (zero-order hold).

`native` is arguably the more faithful model in both cases -- the real robot also
only actuates at its control rate. The fine tick was never needed to actuate
faster than the robot can; it was needed so LATENCY is not quantised to the
control period, and that survives, because snapshot and arrival times stay on
the 37.037 ms tick grid.

A guard now refuses to run at all if the control mode contains
`interpolate_by_planner` while `--actuation fine` is in force with a delta scale
!= 1, so the silent 40-point floor cannot come back.

### Bug 2 -- the gripper command is a DELTA on google_robot, with a state machine

Not in the original diagnosis. `to_env_action()` binarised the gripper as
`2*(g>0.5)-1`, which is the **widowx** convention: `gripper_pd_joint_pos` there
is an ABSOLUTE normalised joint position. google_robot uses
`gripper_pd_joint_target_delta_pos_interpolate_by_planner`, a **delta** command,
and SimplerEnv's octo wrapper drives it through a sticky state machine: it sends
the open<->close TRANSITION (`previous - current`), and once a transition is
detected it repeats it for `sticky_gripper_num_repeat = 15` control steps so the
gripper has time to actually close (`octo15_inference.py`, the
`policy_setup == "google_robot"` branch).

Sending +1 to a delta controller commands "open by one unit" on every step
forever, so the gripper never closes. `close_drawer` only pushes, so it was
barely affected and Bug 1 alone appeared to fix it; `pick_coke_can` needs a
grasp, so it stayed on the floor. MEASURED with Bug 1 fixed and Bug 2 present,
zero latency:

| task | seed | stock | fix 1 only | fix 1 + fix 2 |
|---|---|---|---|---|
| close_drawer | 80 | 10/24 | 10/24 | 10/24 |
| close_drawer | 81 | 10/24 | 12/24 | 9/24 |
| pick_coke_can | 80 | 10/24 | **0/24** | 9/24 |
| pick_coke_can | 81 | 7/24 | **2/24** | 9/24 |

**The fix.** The harness runs the same state machine itself (it calls
`model.sample_actions()` directly rather than `policy.step()`), advancing it once
per **ACTUATION**. Transcription verified against `octo15_inference.py` over 200
random sequences x 120 steps: 0 mismatches.

Per-ACTUATION, not per-RESULT, deliberately. The counter is written in control
steps. Advancing it per result would make the sticky window span
`15 x cadence` seconds -- 1.8 s at the pipe110 cadence, 10.3 s at cpu685 --
penalising fast-cadence arms and rewarding slow ones on any grasping task, which
is a wrapper artifact sitting directly on top of the contrast the sweep measures.
Per-actuation holds the gripper for a fixed 15 control steps in every arm, so
only the CONTENT of the gripper signal varies with latency. At zero latency
cadence == the native control period, results and actuations are 1:1, and the two
choices coincide -- which is why the gate passes either way.

---

## Validation gate (the requirement before any sweep)

Fixed harness, `--latency-ms 0`, 24 episodes, octo-small (1.0), local TITAN RTX.
Reference is `run_eval.py --ckpt hf://rail-berkeley/octo-small` on the same box.

| task | seed | stock | fixed fine harness | delta |
|---|---|---|---|---|
| google_robot_close_drawer | 80 | 10/24 = 41.7% | **10/24 = 41.7%** | 0 eps |
| google_robot_close_drawer | 81 | 10/24 = 41.7% | **9/24 = 37.5%** | -1 ep |
| google_robot_pick_coke_can | 80 | 10/24 = 41.7% | **9/24 = 37.5%** | -1 ep |
| google_robot_pick_coke_can | 81 | 7/24 = 29.2% | **9/24 = 37.5%** | +2 eps |

Every arm within two episodes. PASSED.

## Widowx regression

The fix must not move widowx. It does not, and the equality is exact, not
statistical:

| run | reference | re-run on the fixed harness |
|---|---|---|
| spoon lat0 seed 90 (w0) | 11/24 = 45.8% | **11/24**, successes `{0,4,8,9,10,12,15,18,20,21,22}` -- the IDENTICAL set |
| egg lat0 seed 100 (w4) | 9/24 = 37.5% | **9/24 = 37.5%** |

Each re-run was done on the same worker that produced the reference. By code
path, widowx is provably untouched: `--actuation auto` selects `fine`, so
`ACT_EVERY == 1` (actuate every tick), `SCALE == TICK_MS/BASE_MS` as before, the
gripper map is the same pure function, `HOLD_CMD == to_env_action(HOLD)`, and the
success-detector stride is unchanged.

---

## RESOLUTION LIMIT of `--actuation native` -- state this in any result

Arrival and snapshot times stay on the 37.037 ms tick grid, but the command is
sampled at the native control boundary, so:

* Changing the modelled latency alters the ACTUATED command sequence only when an
  arrival crosses a native control boundary. For an arm whose cadence equals the
  native period the arrival phase is fixed, so latency is resolved in steps of
  **333.3 ms** on google_robot (200 ms on widowx). For a cadence incommensurate
  with the control period the crossing points move through the episode, so the
  effective resolution is finer than 333.3 ms but never as fine as the tick.
* None of the six sweep arms collapse onto each other: they differ in cadence as
  well as latency (0/333.3, 117.7/117.6, 231.8/203.0, 283.4/283.4, 555.0/555.0,
  684.8/684.8), so each produces a distinct command sequence.
* Second, smaller artifact: the env state only exists on the native grid, so a
  snapshot taken at a fine tick inside a native period returns that period's
  starting state. The true observation age is therefore up to one
  tick-to-boundary interval larger than the modelled one.

This is a real loss of resolution relative to the widowx `fine` mode and must be
carried into any cross-embodiment comparison -- see `g5fine/RESULTS_GOOGLE.txt`
section on what the pipe110 arm can and cannot test.

## Reproducibility note found while validating

Same box, same seed: **bit-reproducible** (spoon rng90 reproduced its exact
success set). Different GPU: **not**. `coke lat0 rng80` scores 9/24 on the local
TITAN RTX and 14/24 on an A10G (measured twice on two different A10G boxes, both
14/24). The google_robot scenes are contact-rich and small numeric differences
compound over an episode. Consequences for the sweep design:

* every arm of a seed runs on the same worker, so the PAIRED per-seed penalty is
  unaffected;
* the seed -> worker map is identical for every arm, so marginal per-arm rates
  are not confounded either;
* but a google_robot marginal rate is only comparable to another measured on the
  same GPU. The gate above is TITAN RTX vs TITAN RTX; the sweep is all A10G.

## Alternatives considered and rejected

* Switch google_robot to a non-interpolated control mode. Makes the rescaling
  valid but changes the embodiment's dynamics and breaks comparability with the
  published SIMPLER numbers.
* Sub-step the physics under the ManiSkill control loop. Correct but invasive.
* Restrict the fine-grain study to widowx. Gives up the second embodiment, which
  was the whole point of the follow-up.
