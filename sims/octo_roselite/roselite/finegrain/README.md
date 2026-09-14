# RoSE-lite FINE-GRAIN: sim ticks faster than policy results arrive

Rebuild of the latency replay so the simulator steps at **40 ms (25 Hz)** while
the policy updates only when the MODELLED QRB5165 latency says a result landed,
holding the last commanded action (zero-order hold) in between.

Replaces `../latency_eval.py`, which stepped at 200 ms and quantised latency to
`D = ceil(latency / 200 ms)` control steps -- so every latency in (200, 400] ms
collapsed to the same `D = 2`, and the robot almost never ran on stale data.

## Files

| file | what |
|---|---|
| `finegrain_eval.py` | the harness. Fine tick, ZOH, separate cadence/latency, per-tick action-age logging. |
| `check_scaling.py` | numerical proof that the 1/5 delta rescaling is exact against the real controller math. |
| `analyze_fine.py` | pooled success rate + staleness + Fisher tests -> `RESULTS_FINEGRAIN.txt`, `curve_fine.json`. |
| `metrics_fine.py` | funnel, time-to-success, hold fraction, excursion -> `METRICS_FINEGRAIN.txt`. |
| `plot_fine.py` | `success_vs_latency_finegrain.png` (success curve + sensor->actuation age). |
| `animate_fine_replay.py` | per-tick replay video: fresh vs held ticks, age sawtooth, action staircase. |
| `sweep.sh` / `run_arm.sh` / `record_videos.sh` / `make_fine_videos.sh` | drivers. |

Runs land in `runs/` and `runs_video/` **under this directory** -- never in
`../runs/`, which `../analyze.py` globs.

## Reproduce

```bash
source env.sh                 # ../env.sh: conda octo_sim + VK_ICD + XLA flags
./sweep.sh                    # 5 latency arms x 3 seeds
./run_arm.sh ctrl_lat0 0 200 0    # the validation arm
python analyze_fine.py && python metrics_fine.py && python plot_fine.py
./record_videos.sh vid_serial283 283.4 283.4 4 && ./make_fine_videos.sh
```

## The correctness issue this file exists to get right

Octo emits ~5 Hz end-effector pose deltas. The controller is
`arm_pd_ee_target_delta_pose_align2` (`widowx/defaults.py:131`) with
`use_target=True`, `normalize_action=False`: the commanded target pose is a pure
**integrator** of the deltas. Stepping 5x more often without compensation moves
the arm 5x as far.

So `world_vector` and the rotation are scaled by `tick_ms / 200 ms = 1/5`.
The **rotvec** is scaled, not the euler angles: `R(axis, ang/5)^5 == R(axis, ang)`
exactly, whereas `R(euler/5)^5 != R(euler)`. `check_scaling.py` measures both --
rotvec-scaled quaternion error 1.2e-7 (float noise) vs euler-scaled 8.5e-4.

The **gripper is NOT scaled**. `gripper_pd_joint_pos` is a
`PDJointPosMimicControllerConfig` with `use_delta` unset (`defaults.py:161`) --
an absolute joint position in [-1,1] mapped onto [0.014, 0.038] m. Octo's raw
gripper output is an open/close probability binarised at 0.5. Scaling it would
command a half-open gripper, not a slower one.

Two further choices that keep the arms comparable:

* **Observation history** is always `[frame(t - 200 ms), frame(t)]` regardless of
  dispatch cadence, so only actuation latency varies between arms.
* **`evaluate()` is sampled on the 200 ms grid**, not every tick. Sampling the
  success detector 5x more often would catch transients the 5 Hz baseline misses
  and inflate the rate.
* **The stock `ActionEnsembler` is kept in every arm**, one push per dispatch.
  The coarse study ran its serial arms with the ensembler OFF, which alone cost
  ~39 points (52.8% -> 14.2%) and confounded every serial number it reported.

## Caveat: ZOH is the pessimistic end of the design space

The task under study is pure zero-order hold: the landed action is repeated for
the whole gap. That is what `animate_fine_replay.py` verifies tick by tick
("applied-action changes on a HELD tick: 0").

A deployed chunked policy has a cheaper option. Octo predicts a 4-entry chunk =
4 x 200 ms = 800 ms of motion, so during a 283 ms gap it could walk forward
through the chunk instead of repeating entry 0, and only fall back to ZOH once
the chunk runs dry. That would recover some of the `serial283` loss and is the
obvious mitigation to test next. It is NOT modelled here, so the numbers in
`RESULTS_FINEGRAIN.txt` are the pessimistic bound for a chunk-capable policy and
the correct bound for a non-chunked one.

## Videos

`videos/fine_*.mp4` -- per-tick replays. Green tick = a fresh policy result
landed; red = zero-order hold on stale data; grey = nothing has landed yet. The
middle panel is the observation age sawtooth (rising during a hold, dropping on
each landing) and the bottom panel is the applied action, visibly a staircase.

| clip | held ticks | mean / max obs age |
|---|---|---|
| `fine_0ms_control_VALIDATED.mp4` | 80.0% | 80 / 160 ms |
| `fine_117.7ms_pipelined-110.mp4` | 64.5% | 157 / 200 ms |
| `fine_283.4ms_serial.mp4` | 84.7% | 405 / 560 ms |
| `fine_684.8ms_cpu-monolith.mp4` | 91.3% | 1008 / 1360 ms |
