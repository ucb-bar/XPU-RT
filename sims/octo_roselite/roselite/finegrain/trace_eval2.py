"""RoSE-lite FINE-GRAIN v2: identical rollouts to trace_eval.py, PLUS the actuator
torque and the contact force that trace_eval.py cannot see.

WHY THIS FILE EXISTS
--------------------
`trace_eval.py` logs `robot.get_qf()` once per tick and the paper called the
integral of its square "actuator energy". It is not. `base_controller.py:229-238`
sets `qf = articulation.compute_passive_force(external=False)` every substep and
NO controller in the ManiSkill2_real2sim tree returns a `qf` term, so `get_qf()`
is exactly the gravity + Coriolis FEED-FORWARD the controller writes into the
simulator. The torque that actually moves the arm is a PhysX articulation DRIVE
(`pd_joint_pos.py:35`, `set_drive_property(stiffness, damping, force_limit,
"force")`) and it never passes through qf. See ENERGY_AUDIT.md.

WHAT IS ADDED HERE
------------------
1. tau_drive -- the PD joint-drive force PhysX applies, RECONSTRUCTED per SUBSTEP
   as

       tau_drive = clip( K*(q_target - q_post) + D*(v_target - v_post), +-force_limit )

   with K/D/force_limit read off the live `sapien.Joint` objects (so the agent
   config, whatever it is, is respected), q_target/v_target read off the live
   articulation (so the google_robot planner-interpolated targets, which change
   every substep, are respected), and q_post/v_post read AFTER `scene.step()`.
   POST-step, not pre-step: PhysX solves articulation drives IMPLICITLY, so the
   spring is evaluated at the end of the substep. Measured (see ENERGY_FIX.md):
   against the equation of motion M(q)qddot = tau_drive (which holds exactly in
   free space because qf already cancels g + C), the post-step form has a
   residual of <0.3% of |tau| while the pre-step form is 36x worse on the first
   substeps after a target update.

2. contact -- the external contact force on the robot, from
   `scene.get_contacts()` and `ContactPoint.impulse`. `impulse` is an IMPULSE in
   N.s over the substep and acts on `actor0` (verified: an eggplant of 0.0209 kg
   resting on the sink gives |sum impulse|/dt = 0.2056 N against m*g = 0.2049 N,
   0.3%). Robot-INTERNAL contacts are excluded -- adjacent gripper links of the
   widowx self-collide with ~16 kN of solver impulse and would swamp everything.

3. tau_total = qf + tau_drive -- the full generalized force the simulated
   actuators inject. A real servo has to produce the gravity hold AND the
   tracking/contact torque; the simulator merely splits them into a feed-forward
   and a PD feedback. This is the honest "actuator torque" column.

SAMPLING (this is the part the old file got wrong)
--------------------------------------------------
Everything above is accumulated INSIDE the substep loop, at sim_freq (500 Hz on
widowx, 513 Hz on google_robot), and reduced once per ACTUATION -- not once per
tick. On google_robot ACT_EVERY = 9, i.e. the simulator does not advance on 8 of
every 9 logged ticks, so a per-tick sample of these channels would repeat the
same state 9 times (ENERGY_AUDIT.md 2.7a). Per-actuation with substep-resolution
reduction is the cadence at which the signal actually changes, and it makes
    INT Sum tau^2 dt = Sum_actuations mean_substeps(tau^2) * act_dt
an EXACT substep-rate integral rather than a tick-rate approximation.

THE SETPOINT-STEP TRANSIENT (disclose this, it is large)
--------------------------------------------------------
widowx's arm controller is `arm_pd_ee_target_delta_pose_align2` with
`interpolate=False`: the PhysX drive target STEPS once per control period and is
then held for all 100 substeps. A step in the setpoint of dq against a stiffness
of ~1200 N.m/rad is an instantaneous K*dq that the 330 N.m.s/rad damper kills in
one or two substeps. Measured in free space over 12 actuations: substep 0 alone
carries 96.9% of INT Sum tau^2 dt (uniform would be 1.0%) and its mean is 23000x
the mean of substeps >= 10. It is a torque PhysX really applies -- a real servo
handed a 5 Hz step setpoint really would draw that spike, and its size is the
commanded step size, which the latency does change -- but a column that is 97%
one substep in 100 is a measure of commanded step size, not of contact.
google_robot does NOT have this: its controllers are `interpolate_by_planner`, so
the target moves every substep and substep 0 carries 0.27% (uniform 0.58%).
So BOTH are logged: `tau_drive_sq` over all substeps, and `tau_drive_sq_sus`
over substeps >= SUS_SKIP = 5 (10.0 ms of widowx's 200 ms actuation, 9.7 ms of
google's 333 ms). Contact does not live in the transient: in the press
counterfactual of ENERGY_FIX.md the contact torque is sustained across every
substep of the press, so the _sus column keeps it and drops the step spike.

Every array and every summary field trace_eval.py wrote is still written,
unchanged, so old and new traces are directly comparable on the same episodes.
The per-tick `ep*_qf.npy` in particular is byte-for-byte the old channel,
including its 0 at tick 0.

NEW FILES PER EPISODE
---------------------
  ep??_act_tick.npy       (n_act,)      tick index of each actuation
  ep??_tau_drive_mean.npy (n_act, dof)  mean_substeps tau_drive
  ep??_tau_drive_sq.npy   (n_act, dof)  mean_substeps tau_drive^2
  ep??_tau_drive_sq_sus.npy (n_act,dof)  the same over substeps >= SUS_SKIP only
  ep??_tau_drive_max.npy  (n_act, dof)  max_substeps |tau_drive|
  ep??_tau_total_sq.npy   (n_act, dof)  mean_substeps (qf + tau_drive)^2
  ep??_qf_sub_sq.npy      (n_act, dof)  mean_substeps qf^2      (substep-rate old metric)
  ep??_qvel_sub_sq.npy    (n_act, dof)  mean_substeps qvel^2
  ep??_work_drive.npy     (n_act, dof)  INT tau_drive*qvel dt   (signed)
  ep??_work_abs.npy       (n_act, dof)  INT |tau_drive*qvel| dt
  ep??_sat_frac.npy       (n_act, dof)  fraction of substeps at >=99% force_limit
  ep??_contact.npy        (n_act, 8)    see CONTACT_COLS below
  drive_config.json                     K, D, force_limit, joint names, timing

--- the original trace_eval.py docstring follows, unchanged ---

RoSE-lite FINE-GRAIN: sim ticks faster than policy results arrive, ZOH in between.

Why this exists
---------------
The coarse harness (../latency_eval.py) stepped the env at control_freq=5 (200 ms)
and quantised latency to D=ceil(latency/200ms) *control steps*.  Every latency in
(200,400] ms therefore collapsed to the same D=2, the policy produced a result
roughly every control step, and the robot almost never ran on stale data -- which
is the effect the study is trying to expose.

Here the simulator runs at a FINE CONTROL TICK (default 40 ms / 25 Hz; the tick
must divide sim_freq=500 Hz) while the policy only updates when the MODELLED
hardware latency says a result has landed.  Between updates the last commanded
action is held (zero-order hold).  Semantics follow
XPU-RT/sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py:

    * camera snapshot is taken when a job STARTS,
    * the job's output is applied when the job ENDS,
    * outside that, ZOH on the last commanded action.

Timing model (MODELLED, driven by MEASURED QRB5165 numbers)
-----------------------------------------------------------
Two independent numbers, never collapsed into one:

    --issue-period-ms  CADENCE: how often a fresh result is produced.
    --latency-ms       AGE: how old the observation behind a result is when it
                       is applied.

Serial (one inference in flight) => cadence == latency.
Pipelined (k in flight)          => cadence < latency; both are modelled.

Job j is dispatched at continuous time  j*issue_period.  On the tick grid
(T(t) = t*tick_ms):

    snapshot tick    s_j = ceil(j*issue_period / tick_ms)
    application tick a_j = ceil((j*issue_period + latency) / tick_ms)

latency 0 => a_j == s_j: observe and act on the same tick, i.e. the stock
free-running configuration, just sampled on a finer grid.

THE ACTION RESCALING (the correctness issue this file exists to get right)
-------------------------------------------------------------------------
Octo emits ~5 Hz end-effector pose DELTAS.  The controller is
arm_pd_ee_target_delta_pose_align2 with use_target=True and normalize_action=False
(mani_skill2_real2sim/agents/configs/widowx/defaults.py:131), i.e. the commanded
target pose is a pure INTEGRATOR of the deltas:

    target_{k+1} = ( P(c) * delta * P(c)^-1 ) * target_k        (frame ee_align2)

with c the *current actual* ee position.  Translation is exactly additive
(t <- t + dp), and rotation composes multiplicatively about a fixed axis.  So
applying a delta every 40 ms instead of every 200 ms moves the arm 5x as far
unless the delta is scaled by (tick_ms / 200 ms).  We therefore scale

    world_vector   *= tick_ms/action_dt_ms
    rot_axangle    *= tick_ms/action_dt_ms      <- the ROTVEC, not the euler
                                                  angles: R(axis, ang/5)^5 ==
                                                  R(axis, ang) exactly, whereas
                                                  R(euler/5)^5 != R(euler).

The GRIPPER IS NOT SCALED -- and it is not even the same KIND of command on the
two embodiments, which is a trap this harness fell into once already.

  widowx        gripper_pd_joint_pos is a PDJointPosMimicController with
                use_delta unset (=False) and normalize_action=True
                (defaults.py:161), i.e. an ABSOLUTE joint position in [-1,1]
                mapped onto [0.014, 0.038] m.  Octo's raw gripper output is an
                open/close probability in [0,1], binarised at 0.5 by
                2*(g>0.5)-1.  Scaling it would command a half-open gripper, not
                a slower one.

  google_robot  gripper_pd_joint_target_delta_pos_interpolate_by_planner is a
                DELTA command, and SimplerEnv's octo wrapper drives it through a
                sticky state machine: the open<->close TRANSITION (previous
                minus current) is what is sent, and once a transition is
                detected it is repeated for sticky_gripper_num_repeat = 15
                policy steps so the gripper has time to actually close
                (octo15_inference.py, the policy_setup == "google_robot" branch).

Sending the widowx binariser to the google_robot delta controller commands
"open by 1 unit" on every step forever, so the gripper never closes and no grasp
task can succeed.  MEASURED: with that bug and the actuation fix already in
place, google_robot_pick_coke_can scored 0/24 and 2/24 at ZERO latency where
stock gets 10/24 and 7/24; close_drawer, which only pushes, was barely affected.
This harness therefore runs the SAME state machine itself, advancing it once per
ACTUATION -- i.e. once per control step, which is the unit its 15-step counter is
written in. At zero latency, cadence == the native control period, so results and
actuations are 1:1 and this is EXACTLY stock.

Advancing it once per RESULT instead was considered and rejected. The counter
would then span 15 x cadence seconds, so the sticky window would shrink to 1.8 s
at the pipe110 cadence and stretch to 10.3 s at the cpu685 cadence -- a
mechanical artifact of a wrapper heuristic that would penalise fast-cadence arms
and reward slow ones on any grasping task, confounding exactly the contrast the
sweep is trying to measure. Per-ACTUATION keeps the gripper hold at a fixed 15
control steps for every arm, so only the CONTENT of the gripper signal varies
with latency. The pre-first-result HOLD is NOT fed through the machine, so
`previous_gripper_action` is still initialised by the first real result, as in
stock.

Observation history
-------------------
Octo consumes a 2-frame history.  To keep the *observation* identical across
arms and isolate pure actuation latency, the camera is buffered and the history
handed to the policy is always [frame(t - 200 ms), frame(t)] regardless of the
dispatch cadence (--history-spacing fixed, the default).  --history-spacing
cadence instead uses the previous dispatch's frame.

Success sampling
----------------
evaluate() is called on the 200 ms grid only (every base_ms/tick_ms ticks), not
every tick, so the success detector is sampled at exactly the baseline's rate.
Sampling it 5x more often would catch transients the 5 Hz baseline misses and
inflate the success rate.

TWO ACTUATION MODES (--actuation, default auto)
----------------------------------------------
The rescaling above is only legitimate when the controller is a pure integrator
of the deltas.  That holds for widowx

    arm_pd_ee_target_delta_pose_align2         (widowx, use_target=True)

but NOT for google_robot

    arm_pd_ee_delta_pose_align_interpolate_by_planner
    gripper_pd_joint_target_delta_pos_interpolate_by_planner

whose controller PLANS a trajectory to the target over the control period under
velocity/acceleration limits.  Shrinking the control period 9x while shrinking
the target 9x does not reproduce the motion -- the planner is time-limited and
the arm barely moves.  Measured on google_robot_close_drawer, octo-small, zero
latency: stock 10/24, this harness at the 27 Hz tick 0/24.  See
GOOGLE_ROBOT_PORT.md.

So there are two modes:

  --actuation fine    env.control_freq = the fine tick, deltas scaled by
                      tick_ms/action_dt_ms.  The original model.  Default for
                      widowx, where it is exact.

  --actuation native  env.control_freq = the env's OWN control_freq, deltas
                      applied unscaled.  The fine grid still carries the
                      dispatch/arrival schedule -- it only decides WHICH result
                      is current at each native step (zero-order hold).  Default
                      for google_robot.

'native' is arguably the more faithful model in both cases: the real robot also
only actuates at its control rate.  The fine tick was never needed to actuate
faster than the robot can -- it was needed so LATENCY is not quantised to the
control period, and that property survives, because snapshot and arrival times
stay on the fine grid.

RESOLUTION LIMIT OF 'native'.  Two latencies that land inside the same native
control period produce the SAME actuation, because the command is sampled at the
native boundary.  So the actuation effect of latency is resolved to the native
control period -- 333.3 ms on google_robot, 200 ms on widowx -- while arrival
times themselves remain resolved to the tick (37.0 / 40 ms).  Concretely, on
google_robot every modelled latency in [0, 333.3) ms that shares a cadence gives
the same command sequence.  A second, smaller artifact: the env state only
exists on the native grid, so a snapshot taken at a fine tick inside a native
period returns the state from that period's start, making the true observation
age up to one tick-to-boundary interval larger than the modelled one.
"""
import argparse, json, math, os, sys, time
from collections import deque

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISPLAY"] = ""

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
ap.add_argument("--ckpt", default="hf://rail-berkeley/octo-small")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--ep-start", type=int, default=0)
ap.add_argument("--init-rng", type=int, default=0)
ap.add_argument("--tick-hz", type=int, default=None,
                help="fine control tick. Default: auto -- the divisor of the env's "
                     "sim_freq that is also a multiple of its native control_freq "
                     "and lands nearest --tick-ms-target. (widowx 500/5 -> 25 Hz, "
                     "google_robot 513/3 -> 27 Hz.)")
ap.add_argument("--tick-ms-target", type=float, default=40.0,
                help="target tick period the auto tick-hz search aims at.")
ap.add_argument("--latency-ms", type=float, default=0.0,
                help="MEASURED board latency: age of the observation behind an applied result.")
ap.add_argument("--issue-period-ms", type=float, default=None,
                help="cadence at which fresh results arrive. Default: max(latency, action-dt) "
                     "(= serial, one in flight).")
ap.add_argument("--action-dt-ms", type=float, default=None,
                help="the dt Octo's deltas were authored for. Default: the env's "
                     "NATIVE control period (widowx 5 Hz -> 200 ms, google_robot "
                     "3 Hz -> 333.33 ms). Deltas are scaled by tick_ms/action_dt_ms.")
ap.add_argument("--horizon-ms", type=float, default=None,
                help="episode wall-clock horizon. Default: the env's registered "
                     "max_episode_steps x its native control period, i.e. exactly the "
                     "wall-clock the stock baseline gets. Do NOT leave this at a "
                     "hardcoded value across tasks -- eggplant is 120x200=24000 ms but "
                     "spoon/carrot/cube are 60x200=12000 ms and the google_robot "
                     "drawers are 113x333.33=37667 ms; over-granting it silently "
                     "inflates success.")
ap.add_argument("--no-scale", action="store_true",
                help="ABLATION: do not rescale the deltas (arm moves tick_hz/5 times too far).")
ap.add_argument("--ensemble", default="stock", choices=["stock", "none"],
                help="stock = the SimplerEnv ActionEnsembler, one push per DISPATCH "
                     "(identical to the 5 Hz baseline when cadence==200 ms).")
ap.add_argument("--history-spacing", default="fixed", choices=["fixed", "cadence"])
ap.add_argument("--actuation", default="auto", choices=["auto", "fine", "native"],
                help="WHERE the commanded action is applied. 'fine': the env runs at the "
                     "fine tick and deltas are rescaled by tick/action_dt -- valid only "
                     "for a controller that integrates the deltas (widowx "
                     "arm_pd_ee_target_delta_pose_align2). 'native': the env runs at its "
                     "OWN control_freq, deltas unscaled, and the fine grid only decides "
                     "which result is current at each native step (ZOH) -- required for "
                     "the google_robot *_interpolate_by_planner controllers. 'auto' "
                     "(default) = native for google_robot, fine for widowx, which leaves "
                     "every widowx run bit-identical to the pre-fix harness.")
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--save-video-every", type=int, default=0,
                help="render EVERY tick for these episodes and write an mp4 (slow).")
args = ap.parse_args()

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1536)])

import jax
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from octo.model.octo_model import OctoModel
from transforms3d.euler import euler2axangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from octo15_inference import Octo15Inference
import mediapy as media

CHUNK = 4

# The two SIMPLER embodiments do NOT share a timebase, and getting this wrong is
# silent rather than loud, so derive everything from the task:
#
#   widowx        control_freq 5, sim_freq 500   (put_on_in_scene.py:167-168)
#   google_robot  control_freq 3, sim_freq 513   (grasp_single_in_scene.py:69,
#                                                 move_near_in_scene.py:63-64,
#                                                 open_drawer_in_scene.py:44-45)
#
# 513 = 3^3 * 19, so the 25 Hz tick this harness used for widowx does NOT divide
# it -- google_robot needs 27 Hz. Both are asserted against the constructed env
# below, so a wrong entry here fails loudly instead of quietly mis-timing a sweep.
NATIVE = {"widowx_bridge": (5, 500), "google_robot": (3, 513)}
policy_setup = "widowx_bridge" if "widowx" in args.task else "google_robot"
NATIVE_CF, SIM_FREQ = NATIVE[policy_setup]

# Actuation mode. google_robot's controllers are planner-interpolated, so the
# fine-tick rescaling is invalid there (0/24 at zero latency where stock gets
# 10/24) -- it must actuate on its native grid. widowx keeps 'fine' so every
# existing widowx run stays reproducible.
ACTUATE = args.actuation
if ACTUATE == "auto":
    ACTUATE = "native" if policy_setup == "google_robot" else "fine"

def _legal_ticks(sim_freq, native_cf):
    """Ticks that divide sim_freq AND are a multiple of the native control rate.

    The second condition is what keeps BASE_MS/TICK_MS an exact integer, so the
    success detector and the 2-frame history land on the native grid rather than
    drifting off it by a fraction of a tick."""
    return sorted(d for d in range(1, sim_freq + 1)
                  if sim_freq % d == 0 and d % native_cf == 0)

LEGAL = _legal_ticks(SIM_FREQ, NATIVE_CF)
if args.tick_hz is None:
    TICK_HZ = min(LEGAL, key=lambda d: abs(1000.0 / d - args.tick_ms_target))
else:
    TICK_HZ = int(args.tick_hz)
    if TICK_HZ not in LEGAL:
        raise SystemExit(
            f"tick_hz {TICK_HZ} is illegal for {args.task}: it must divide "
            f"sim_freq {SIM_FREQ} and be a multiple of the native control_freq "
            f"{NATIVE_CF}. Legal ticks: {LEGAL}")
TICK_MS = 1000.0 / TICK_HZ

# The ACTUATION grid. In 'fine' mode it is the tick grid itself. In 'native' mode
# it is the env's own control grid; TICK_HZ is a multiple of NATIVE_CF by
# construction (see _legal_ticks), so ACT_EVERY is an exact integer number of
# ticks and actuation always lands on a tick boundary.
ACT_HZ = TICK_HZ if ACTUATE == "fine" else NATIVE_CF
ACT_EVERY = TICK_HZ // ACT_HZ
ACT_MS = 1000.0 / ACT_HZ
assert TICK_HZ % ACT_HZ == 0, (TICK_HZ, ACT_HZ)

BASE_MS = float(args.action_dt_ms) if args.action_dt_ms is not None else 1000.0 / NATIVE_CF

# Horizon: the stock baseline gets max_episode_steps native steps, so the fine
# harness must get exactly the same wall-clock, no more.
if args.horizon_ms is not None:
    HORIZON_MS = float(args.horizon_ms)
else:
    import gymnasium as _gym_reg
    _env_id, _ = simpler_env.ENVIRONMENT_MAP[args.task]
    _steps = _gym_reg.spec(_env_id).max_episode_steps
    HORIZON_MS = _steps * (1000.0 / NATIVE_CF)
    print(f"[horizon] {args.task}: {_steps} native steps x {1000.0/NATIVE_CF:.2f} ms "
          f"= {HORIZON_MS:.0f} ms", flush=True)

# Scale the deltas from the dt they were authored for to the dt they are applied
# at. 'fine' -> tick_ms/action_dt (the original); 'native' -> 1.0, since the
# actuation period IS the native action period.
SCALE = 1.0 if args.no_scale else ACT_MS / BASE_MS
MAX_TICKS = int(round(HORIZON_MS / TICK_MS))
EVAL_STRIDE = max(int(round(BASE_MS / TICK_MS)), 1)   # sample success on the native grid
HIST_GAP = max(int(round(BASE_MS / TICK_MS)), 1)      # 2-frame history spacing, in ticks

LAT = float(args.latency_ms)
PERIOD = float(args.issue_period_ms) if args.issue_period_ms is not None else max(LAT, BASE_MS)
assert PERIOD > 0

# ---- precompute the dispatch schedule on the tick grid (deterministic) ----
def _ceil_tick(t_ms):
    return int(math.ceil(t_ms / TICK_MS - 1e-9))

snap_tick, apply_tick = [], []
j = 0
while True:
    s = _ceil_tick(j * PERIOD)
    if s >= MAX_TICKS:
        break
    snap_tick.append(s)
    apply_tick.append(_ceil_tick(j * PERIOD + LAT))
    j += 1
N_JOBS = len(snap_tick)
# jobs whose result lands after the horizon are still dispatched (they cost the
# board time) but never applied.
apply_at = {}          # tick -> [job ids landing on this tick], in dispatch order
for jj, a in enumerate(apply_tick):
    if a < MAX_TICKS:
        apply_at.setdefault(a, []).append(jj)
snap_at = {}                   # tick -> job id (cadence >= tick so 1:1)
for jj, s_ in enumerate(snap_tick):
    snap_at[s_] = jj
render_ticks = set(snap_tick) | {0}
if args.history_spacing == "fixed":
    render_ticks |= {s - HIST_GAP for s in snap_tick if s - HIST_GAP >= 0}

stale_ticks = [apply_tick[i] - snap_tick[i] for i in range(N_JOBS)]
max_inflight = 0
_ev = sorted([(snap_tick[i], 1) for i in range(N_JOBS)] +
             [(apply_tick[i], -1) for i in range(N_JOBS)])
_c = 0
for _t, _d in _ev:
    _c += _d
    max_inflight = max(max_inflight, _c)

run = args.out or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs",
    f"tick{TICK_HZ}hz_lat{LAT:g}_per{PERIOD:g}_ens-{args.ensemble}"
    f"{'_noscale' if args.no_scale else ''}"
    f"{'_actnative' if ACTUATE == 'native' else ''}_rng{args.init_rng}{args.tag}")
os.makedirs(run, exist_ok=True)
print(f"[run dir] {run}", flush=True)
print(f"[MODELLED timing] tick={TICK_MS:g} ms ({TICK_HZ} Hz), horizon={MAX_TICKS} ticks "
      f"({HORIZON_MS:g} ms)", flush=True)
print(f"[MODELLED timing] latency={LAT:g} ms  cadence={PERIOD:g} ms  -> staleness "
      f"{min(stale_ticks) if stale_ticks else 0}-{max(stale_ticks) if stale_ticks else 0} ticks "
      f"(mean {np.mean(stale_ticks) if stale_ticks else 0:.2f} = "
      f"{np.mean(stale_ticks)*TICK_MS if stale_ticks else 0:.1f} ms); "
      f"max in flight {max_inflight}; {N_JOBS} dispatches/episode", flush=True)
print(f"[actuation] mode={ACTUATE}: commands applied every {ACT_EVERY} tick(s) "
      f"= {ACT_MS:g} ms ({ACT_HZ} Hz); dispatch/arrival stay on the {TICK_MS:g} ms tick grid. "
      f"Latency resolution: arrivals {TICK_MS:g} ms, ACTUATION {ACT_MS:g} ms.", flush=True)
print(f"[rescaling] delta scale = {SCALE:.6g} (actuation dt {ACT_MS:g} ms / action dt {BASE_MS:g} ms); "
      f"gripper NOT scaled; ensemble={args.ensemble}; history={args.history_spacing}", flush=True)

# ---- build the env at the fine tick ------------------------------------------
# prepackaged_config forcibly overwrites control_freq (put_on_in_scene.py:167),
# so patch the prepackaged config rather than passing control_freq through.
# Each prepackaged-config base class hardcodes its own control_freq/sim_freq
# (PutOnBridgeInSceneEnv for widowx; GraspSingleInSceneEnv, MoveNearInSceneEnv and
# OpenDrawerInSceneEnv for google_robot), so patch every class that defines the
# method rather than naming them -- a new task class is then covered for free.
# The observed native config is captured on the way through and cross-checked
# against the NATIVE table above.
import pkgutil, importlib, inspect
import mani_skill2_real2sim.envs.custom_scenes as _cs

_OBSERVED = {}
def _make_prepack_patch(orig):
    def _patched(self):
        ret = orig(self)
        # Record the NATIVE rate, not a patched one. Some of these classes chain
        # through super(), so an outer patch can see a value an inner patch has
        # already rewritten -- ignore anything that already equals the fine tick.
        cf = ret.get("control_freq")
        if cf is not None and cf != TICK_HZ:
            _OBSERVED["control_freq"] = cf
        if ret.get("sim_freq") is not None:
            _OBSERVED["sim_freq"] = ret["sim_freq"]
        # In 'native' mode ACT_HZ == the native control_freq, so this write is a
        # no-op and the env is left exactly as the task registered it.
        ret["control_freq"] = ACT_HZ
        return ret
    _patched._fine_tick_patch = True
    return _patched

_patched_classes = []
_seen_cls = set()
for _mi in pkgutil.iter_modules(_cs.__path__):
    try:
        _mod = importlib.import_module(f"{_cs.__name__}.{_mi.name}")
    except Exception:
        continue
    for _nm, _cls in inspect.getmembers(_mod, inspect.isclass):
        # These classes are re-exported across modules, so getmembers finds the
        # same object more than once; wrapping twice would nest the patches.
        if id(_cls) in _seen_cls:
            continue
        _seen_cls.add(id(_cls))
        _fn = _cls.__dict__.get("_setup_prepackaged_env_init_config")
        if _fn is not None and not getattr(_fn, "_fine_tick_patch", False):
            _cls._setup_prepackaged_env_init_config = _make_prepack_patch(_fn)
            _patched_classes.append(_cls.__name__)
print(f"[patch] control_freq {ACT_HZ} Hz forced on: {', '.join(sorted(set(_patched_classes)))}",
      flush=True)

env = simpler_env.make(args.task)
base = env.unwrapped
assert base.control_freq == ACT_HZ, (base.control_freq, ACT_HZ)
if base.sim_freq != SIM_FREQ:
    raise SystemExit(
        f"NATIVE table says sim_freq {SIM_FREQ} for {policy_setup} but the env "
        f"reports {base.sim_freq}. The tick and horizon were derived from the wrong "
        f"timebase -- fix the NATIVE table before trusting any run.")
if _OBSERVED.get("control_freq") not in (None, NATIVE_CF):
    raise SystemExit(
        f"NATIVE table says control_freq {NATIVE_CF} for {policy_setup} but the env's "
        f"prepackaged config says {_OBSERVED['control_freq']}. The delta rescaling and "
        f"the horizon are both derived from it -- fix the NATIVE table.")
print(f"[env] control_freq={base.control_freq} sim_freq={base.sim_freq} "
      f"sim_steps_per_control={base._sim_steps_per_control}", flush=True)

# The rescaling is only legitimate for a controller that INTEGRATES the deltas.
# The google_robot controllers plan a trajectory over the control period, so
# N steps of 1/N the delta is not the same motion -- refuse that combination
# loudly rather than reproducing the silent 40-point floor of GOOGLE_ROBOT_PORT.md.
_cm = str(getattr(base, "control_mode", None) or getattr(base.agent, "control_mode", "?"))
print(f"[env] control_mode={_cm}", flush=True)
if "interpolate_by_planner" in _cm and ACTUATE == "fine" and abs(SCALE - 1.0) > 1e-9:
    raise SystemExit(
        f"REFUSING to run: control_mode {_cm} plans a trajectory over the control "
        f"period, so the fine-tick delta rescaling (scale {SCALE:.4g}) does not "
        f"reproduce the motion -- it scores 0/24 at ZERO latency where the stock "
        f"harness gets 10/24. Use --actuation native (the default for google_robot).")

# base.get_obs() returns the RAW camera dict; the rgb/depth conversion lives in
# RGBDObservationWrapper.  Replay the ObservationWrapper chain innermost-first so
# a hand-rolled render matches exactly what env.step() would have handed back.
import gymnasium as _gym
_obs_wrappers = []
_w = env
while isinstance(_w, _gym.Wrapper):
    if isinstance(_w, _gym.ObservationWrapper):
        _obs_wrappers.append(_w)
    _w = _w.env
_obs_wrappers.reverse()
print(f"[env] observation wrappers: {[type(x).__name__ for x in _obs_wrappers]}", flush=True)


def render_now():
    o = base.get_obs()
    for _ow in _obs_wrappers:
        o = _ow.observation(o)
    return get_image_from_maniskill2_obs_dict(env, o)


# ============================================================================
# THE NEW CHANNELS: real drive torque + real contact force, at the SUBSTEP rate
# ============================================================================
# Everything here is additive. The hooks wrap `agent.before_simulation_step`
# (which is what writes qf and, on google_robot, advances the planner-
# interpolated drive targets) and `env._after_simulation_step` (a no-op in
# sapien_env.py, called immediately after `scene.step()`), so they bracket
# exactly one PhysX substep and cannot change the rollout: they only read.
SUBSTEP_DT = 1.0 / SIM_FREQ
N_SUB = None            # substeps per actuation; filled in from the env below
SUS_SKIP = 5            # substeps skipped by the _sus channel (see the docstring)

CONTACT_COLS = ["Fnet_mean", "Fnet_max", "Fsum_mean", "Fsum_max",
                "Fgrip_mean", "Fgrip_max", "n_ext_mean", "impulse_sum"]
#  Fnet  = |sum of external contact impulses on the robot| / substep_dt   [N]
#  Fsum  = sum of |impulse| over external contacts / substep_dt           [N]
#          (Fsum >= |Fnet|: a grasp with two opposed fingers has Fnet ~ 0 and
#           Fsum ~ 2x the grip force, so BOTH are needed.)
#  Fgrip = Fsum restricted to contacts involving a finger link             [N]
#  impulse_sum = INT Fsum dt over the actuation                          [N.s]

_DRV = {}               # live per-episode handles: robot, K, D, FL, link ids
_PRE = {}               # per-substep pre-step reads
_ACC = {}               # per-actuation accumulators


def _drive_refresh():
    """Re-read the articulation handles. MUST be called after every env.reset():
    ManiSkill2 reconfigures the scene, which rebuilds the articulation, and the
    old Python handle then points at freed PhysX memory (get_qpos() returns a
    garbage-length array)."""
    global N_SUB
    robot = base.agent.robot
    aj = robot.get_active_joints()
    links = robot.get_links()
    _DRV.update(
        robot=robot,
        K=np.array([j.stiffness for j in aj], dtype=np.float64),
        D=np.array([j.damping for j in aj], dtype=np.float64),
        FL=np.array([j.force_limit for j in aj], dtype=np.float64),
        names=[j.name for j in aj],
        rids={l.get_id() for l in links},
        fids={l.get_id() for l in links if "finger" in l.get_name()},
        dof=robot.dof)
    N_SUB = base._sim_steps_per_control


def _acc_reset():
    d = _DRV["dof"]
    _ACC.clear()
    _ACC.update(n=0, nsus=0, tau2sus=np.zeros(d),
                tau=np.zeros(d), tau2=np.zeros(d), taumax=np.zeros(d),
                tot2=np.zeros(d), qf=np.zeros(d), qf2=np.zeros(d),
                v2=np.zeros(d), work=np.zeros(d), workabs=np.zeros(d),
                sat=np.zeros(d),
                Fnet=0.0, Fnetmax=0.0, Fsum=0.0, Fsummax=0.0,
                Fgrip=0.0, Fgripmax=0.0, nc=0.0, imp=0.0)


def _acc_flush():
    """Reduce one actuation's substeps to the per-actuation record."""
    n = max(_ACC["n"], 1)
    ns = max(_ACC["nsus"], 1)
    return (_ACC["tau"] / n, _ACC["tau2"] / n, _ACC["tau2sus"] / ns, _ACC["taumax"].copy(),
            _ACC["tot2"] / n, _ACC["qf2"] / n, _ACC["v2"] / n,
            _ACC["work"].copy(), _ACC["workabs"].copy(), _ACC["sat"] / n,
            np.array([_ACC["Fnet"] / n, _ACC["Fnetmax"],
                      _ACC["Fsum"] / n, _ACC["Fsummax"],
                      _ACC["Fgrip"] / n, _ACC["Fgripmax"],
                      _ACC["nc"] / n, _ACC["imp"]]))


def _hook_pre_orig():
    raise RuntimeError("unreachable")


def _install_hooks():
    """Idempotent. Re-run after every reset in case the agent object was rebuilt."""
    ag = base.agent
    if not getattr(ag, "_te2_hooked", False):
        _orig = ag.before_simulation_step

        def _pre():
            _orig()                       # writes qf, advances interpolated targets
            if not _ACC:
                # outside an actuation (env.reset settling): _DRV may still point
                # at the articulation the reconfigure just freed. Read nothing.
                return
            r = _DRV["robot"]
            _PRE["qt"] = r.get_drive_target()
            _PRE["vt"] = r.get_drive_velocity_target()
            _PRE["qf"] = r.get_qf()
        ag.before_simulation_step = _pre
        ag._te2_hooked = True

    if not getattr(base, "_te2_hooked", False):
        _orig_a = base._after_simulation_step

        def _post():
            _orig_a()
            if not _ACC:
                return
            r = _DRV["robot"]
            K, D, FL = _DRV["K"], _DRV["D"], _DRV["FL"]
            q = r.get_qpos().astype(np.float64)
            v = r.get_qvel().astype(np.float64)
            qf = _PRE["qf"].astype(np.float64)
            tau = np.clip(K * (_PRE["qt"] - q) + D * (_PRE["vt"] - v), -FL, FL)
            tot = qf + tau
            a = np.abs(tau)
            _ACC["n"] += 1
            _ACC["tau"] += tau
            _ACC["tau2"] += tau * tau
            if _ACC["n"] > SUS_SKIP:
                _ACC["nsus"] += 1
                _ACC["tau2sus"] += tau * tau
            np.maximum(_ACC["taumax"], a, out=_ACC["taumax"])
            _ACC["tot2"] += tot * tot
            _ACC["qf"] += qf
            _ACC["qf2"] += qf * qf
            _ACC["v2"] += v * v
            p = tau * v
            _ACC["work"] += p * SUBSTEP_DT
            _ACC["workabs"] += np.abs(p) * SUBSTEP_DT
            _ACC["sat"] += (a >= 0.99 * FL)
            # ---- contact -------------------------------------------------
            rids, fids = _DRV["rids"], _DRV["fids"]
            net = np.zeros(3)
            fsum = 0.0
            fgrip = 0.0
            ncon = 0
            for c in base._scene.get_contacts():
                i0 = c.actor0.get_id()
                i1 = c.actor1.get_id()
                in0 = i0 in rids
                in1 = i1 in rids
                if in0 == in1:
                    continue          # both robot (self-collision) or neither
                imp = np.zeros(3)
                for pt in c.points:
                    imp += pt.impulse
                m = float(np.linalg.norm(imp))
                if m == 0.0:
                    continue
                ncon += 1
                fsum += m
                # ContactPoint.impulse acts on actor0; the reaction is on actor1
                net += imp if in0 else -imp
                if (i0 in fids) or (i1 in fids):
                    fgrip += m
            Fn = float(np.linalg.norm(net)) / SUBSTEP_DT
            Fs = fsum / SUBSTEP_DT
            Fg = fgrip / SUBSTEP_DT
            _ACC["Fnet"] += Fn
            _ACC["Fsum"] += Fs
            _ACC["Fgrip"] += Fg
            _ACC["imp"] += fsum
            _ACC["nc"] += ncon
            if Fn > _ACC["Fnetmax"]:
                _ACC["Fnetmax"] = Fn
            if Fs > _ACC["Fsummax"]:
                _ACC["Fsummax"] = Fs
            if Fg > _ACC["Fgripmax"]:
                _ACC["Fgripmax"] = Fg
        base._after_simulation_step = _post
        base._te2_hooked = True


_drive_refresh()
_install_hooks()
print(f"[drive] joints        {_DRV['names']}", flush=True)
print(f"[drive] stiffness     {np.array2string(_DRV['K'], precision=1)}", flush=True)
print(f"[drive] damping       {np.array2string(_DRV['D'], precision=1)}", flush=True)
print(f"[drive] force_limit   {np.array2string(_DRV['FL'], precision=1)}", flush=True)
print(f"[drive] substep {SUBSTEP_DT*1000:.4f} ms x {N_SUB} per actuation "
      f"= {SUBSTEP_DT*N_SUB*1000:.2f} ms (= ACT_MS {ACT_MS:g}); "
      f"tau/contact are reduced ONCE PER ACTUATION from {N_SUB} substeps.", flush=True)
json.dump(dict(joint_names=_DRV["names"], stiffness=_DRV["K"].tolist(),
               damping=_DRV["D"].tolist(), force_limit=_DRV["FL"].tolist(),
               dof=int(_DRV["dof"]), sim_freq=SIM_FREQ, substep_dt=SUBSTEP_DT,
               substeps_per_actuation=int(N_SUB), act_ms=ACT_MS, tick_ms=TICK_MS,
               act_every_ticks=ACT_EVERY, contact_cols=CONTACT_COLS,
               drive_mode=sorted({j.drive_mode for j in base.agent.robot.get_active_joints()}),
               balance_passive_force=bool(base.agent.controller.balance_passive_force),
               reconstruction="clip(K*(qtarget-q_post)+D*(vtarget-v_post),+-force_limit) per substep"),
          open(f"{run}/drive_config.json", "w"), indent=1)
_DRIVE_MODES = sorted({j.drive_mode for j in base.agent.robot.get_active_joints()})
if _DRIVE_MODES != ["force"]:
    raise SystemExit(
        f"drive_mode {_DRIVE_MODES} is not 'force'; the tau = K*dq + D*dv "
        f"reconstruction is only valid for a force-mode drive (an acceleration-"
        f"mode drive multiplies by the effective inertia). Refusing to log a "
        f"quantity that would silently be wrong.")

model = OctoModel.load_pretrained(args.ckpt)
policy = Octo15Inference(model, policy_setup=policy_setup, init_rng=args.init_rng,
                         legacy_unnorm=False, action_ensemble=False)

class StickyGripper:
    """SimplerEnv's google_robot gripper post-processing, replayed verbatim.

    Mirrors octo15_inference.py's google_robot branch (itself a copy of
    SimplerEnv's octo wrapper). The fine harness calls model.sample_actions()
    directly rather than policy.step(), so it has to run this itself. State is
    per EPISODE and advances once per policy RESULT.
    """
    N_REPEAT = 15

    def __init__(self):
        self.reset()

    def reset(self):
        self.on = False
        self.repeat = 0
        self.action = 0.0
        self.prev = None

    def __call__(self, g):
        cur = np.array([float(g)])
        rel = np.array([0.0]) if self.prev is None else self.prev - cur
        self.prev = cur
        if np.abs(rel) > 0.5 and self.on is False:
            self.on = True
            self.action = rel
        if self.on:
            self.repeat += 1
            rel = self.action
        if self.repeat == self.N_REPEAT:
            self.on = False
            self.repeat = 0
            self.action = 0.0
        return float(rel)


sticky = StickyGripper() if policy_setup == "google_robot" else None

HOLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # zero delta, gripper open
# The command held before the FIRST result lands. widowx's gripper channel is an
# absolute position (+1 = open); google_robot's is a delta, where "hold" is 0.
HOLD_CMD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     1.0 if policy_setup == "widowx_bridge" else 0.0])


def predict_chunk(frames):
    """frames: list of resized uint8 frames, oldest first -> (4,7) raw chunk."""
    policy.rng, key = jax.random.split(policy.rng)
    images = np.stack(frames, axis=0)
    pad_mask = np.ones(len(frames), dtype=bool)
    obs = {"image_primary": images[None], "timestep_pad_mask": pad_mask[None]}
    chunk = np.asarray(policy.model.sample_actions(
        obs, policy.task, unnormalization_statistics=policy.stats, rng=key))[0]
    assert chunk.shape == (CHUNK, 7), chunk.shape
    return chunk


def to_env_action(raw7):
    """raw octo 7-vector -> env action, with the rescaling and the gripper map.

    STATEFUL on google_robot (the sticky machine), so this must be called exactly
    once per ACTUATION -- never once per fine tick and never once per result.
    On widowx it is a pure function of raw7, so nothing about its call site
    matters there.
    """
    wv = np.asarray(raw7[:3], dtype=np.float64) * SCALE
    roll, pitch, yaw = np.asarray(raw7[3:6], dtype=np.float64)
    ax, ang = euler2axangle(roll, pitch, yaw)
    rot = ax * ang * SCALE                     # scale the ROTVEC (exact composition)
    if sticky is not None:
        grip = sticky(raw7[6])                 # google_robot: sticky DELTA command
    else:
        grip = 2.0 * (float(raw7[6]) > 0.5) - 1.0   # widowx: absolute, NOT scaled
    return np.concatenate([wv, rot, np.array([grip])])


results, t_start = [], time.time()
for k in range(args.n):
    ep_id = args.ep_start + k
    want_video = bool(args.save_video_every) and (k % args.save_video_every == 0)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    # reset() reconfigures the scene and rebuilds the articulation: the old
    # handle is dangling and get_qpos() on it returns garbage. Re-read first.
    _drive_refresh()
    _install_hooks()
    instruction = base.get_language_instruction()
    policy.reset(instruction)
    ensembler = ActionEnsembler(CHUNK, 0.0) if args.ensemble == "stock" else None
    if sticky is not None:
        sticky.reset()

    frame_buf = {}                 # tick -> resized uint8 frame
    img = get_image_from_maniskill2_obs_dict(env, obs)
    frame_buf[0] = policy._resize_image(img)
    prev_snap_frame = None
    inflight = {}                  # job id -> chunk
    cur_raw = None                 # last commanded raw action (ZOH memory)
    cur_src_tick = None            # snapshot tick behind cur_raw
    applied, ages, fresh_flags, act_ages = [], [], [], []
    ee_xyz, ee_quat, evt = [], [], []          # TRACE: end-effector path + events
    obj_xyz = []                                # TRACE: the manipulated object's path
    qvel_log = []                               # TRACE: joint velocities
    qf_log = []                                 # TRACE: joint forces (stall/I^2R proxy)
    # ---- v2 channels, one row per ACTUATION, each row reduced from N_SUB substeps
    act_tick, L_tau, L_tau2, L_taumax, L_tot2, L_qf2, L_v2 = [], [], [], [], [], [], []
    L_tau2s = []
    L_work, L_workabs, L_sat, L_con = [], [], [], []
    link_com, link_m = [], None                 # TRACE: per-link COM positions + masses.
    # Needed because the LIFT term must be the potential energy of the actual
    # moving bodies (Sum m_i g z_i over link COMs), NOT the end-effector's height
    # times some lumped mass -- the arm's COM can descend while the TCP rises.
    _camname = '3rd_view_camera' if policy_setup == 'widowx_bridge' else 'overhead_camera'
    _cp = obs.get('camera_param', {})
    if _camname not in _cp and _cp:
        _camname = sorted(_cp)[-1]
    _cam = {k: np.asarray(v).tolist() for k, v in _cp.get(_camname, {}).items()}
    _bg = render_now()                          # background plate, tick 0
    n_act = 0
    video = [img] if want_video else []
    n_inf = 0
    info = {"success": False, "episode_stats": {}}
    success = False
    t = 0
    t0 = time.time()
    while t < MAX_TICKS:
        need_render = want_video or (t in render_ticks)
        if need_render and t not in frame_buf:
            frame_buf[t] = policy._resize_image(render_now())

        # --- job START: snapshot the camera and run the inference -------------
        if t in snap_at:
            jid = snap_at[t]
            if args.history_spacing == "fixed":
                hist = ([frame_buf[t - HIST_GAP]] if (t - HIST_GAP) in frame_buf else []) + [frame_buf[t]]
            else:
                hist = ([prev_snap_frame] if prev_snap_frame is not None else []) + [frame_buf[t]]
            inflight[jid] = predict_chunk(hist)
            prev_snap_frame = frame_buf[t]
            n_inf += 1

        # --- job END: the result lands, becomes the new commanded action ------
        for jj in apply_at.get(t, []):
            chunk = inflight.pop(jj, None)
            if chunk is None:
                continue
            raw7 = ensembler.ensemble_action(chunk) if ensembler is not None else chunk[0]
            cur_raw = np.asarray(raw7, dtype=np.float64)
            cur_src_tick = snap_tick[jj]

        # --- ZOH otherwise ----------------------------------------------------
        if cur_raw is None:
            raw7, age, fresh = HOLD, float("nan"), False
        else:
            raw7 = cur_raw
            age = (t - cur_src_tick) * TICK_MS
            fresh = (t in apply_at)
        applied.append(np.asarray(raw7, dtype=np.float64))
        _p = base.tcp.pose
        ee_xyz.append(np.asarray(_p.p, dtype=np.float64))
        ee_quat.append(np.asarray(_p.q, dtype=np.float64))
        # The background plate is the tick-0 frame, so without this the object's
        # own motion is invisible and an event marker drawn at the END-EFFECTOR
        # looks detached from the object it refers to.
        qvel_log.append(np.asarray(base.agent.robot.get_qvel(), dtype=np.float64))
        # Joint FORCE as well as velocity. tau.omega (mechanical power) is ~0 when the
        # arm is blocked -- pushing into the sink wall is high torque at zero velocity --
        # so every motion-based energy metric scores a collision as free. A real servo
        # burns I^2 R there, which goes as tau^2. This is the term that separates an arm
        # that reaches cleanly from one that slams into the scenery.
        qf_log.append(np.asarray(base.agent.robot.get_qf(), dtype=np.float64))
        _lk = [l for l in base.agent.robot.get_links() if float(l.get_mass()) > 0]
        if link_m is None:
            link_m = [float(l.get_mass()) for l in _lk]
            link_names = [l.name for l in _lk]
        link_com.append(np.asarray([l.get_pose().p for l in _lk], dtype=np.float64))
        _o = getattr(base, 'episode_source_obj', None) or getattr(base, 'obj', None)
        obj_xyz.append(np.asarray(_o.pose.p, dtype=np.float64) if _o is not None
                       else np.full(3, np.nan))
        ages.append(age)
        fresh_flags.append(bool(fresh))

        # --- ACTUATE ----------------------------------------------------------
        # 'fine': ACT_EVERY == 1, every tick, exactly as before.
        # 'native': only on the env's own control grid, so the controller gets
        # its full control period and the delta is applied unscaled. The command
        # is whatever the fine-grid ZOH says is current AT the native boundary.
        act_now = (t % ACT_EVERY == 0)
        if act_now:
            # HOLD_CMD rather than to_env_action(HOLD): the pre-first-result hold
            # is not a policy output and must not advance the sticky machine.
            _acc_reset()
            base.step_action(HOLD_CMD if cur_raw is None else to_env_action(raw7))
            (_tau, _tau2, _tau2s, _taumax, _tot2, _qf2, _v2,
             _wk, _wka, _st, _cn) = _acc_flush()
            act_tick.append(t)
            L_tau.append(_tau); L_tau2.append(_tau2); L_tau2s.append(_tau2s)
            L_taumax.append(_taumax)
            L_tot2.append(_tot2); L_qf2.append(_qf2); L_v2.append(_v2)
            L_work.append(_wk); L_workabs.append(_wka); L_sat.append(_st); L_con.append(_cn)
            _ACC.clear()
            base._elapsed_steps += 1
            n_act += 1
            if not (isinstance(age, float) and math.isnan(age)):
                act_ages.append(age)
        t += 1

        if want_video and act_now:
            video.append(render_now())

        # --- success detector sampled on the native grid only -----------------
        if ((act_now if ACTUATE == "native" else (t % EVAL_STRIDE == 0))
                or (t >= MAX_TICKS)):
            info = base.get_info()
            if bool(info["success"]):
                success = True
            # TRACE: log the evaluator sample and the gripper command at this tick,
            # so grasp onset / release / on-target can be located on the path.
            evt.append({'tick': int(t),
                        'stats': {k: bool(v) for k, v in
                                  (info.get('episode_stats') or {}).items()},
                        'grip': float(raw7[6]) if cur_raw is not None else 1.0})
            if success:
                break
        # drop stale frames we will never need again
        if len(frame_buf) > 8:
            for key_t in [x for x in frame_buf if x < t - HIST_GAP - 1]:
                frame_buf.pop(key_t, None)

    stats = info.get("episode_stats", {})
    stats = {k2: (bool(v) if isinstance(v, (bool, np.bool_)) else
                  (v.tolist() if isinstance(v, np.ndarray) else v))
             for k2, v in stats.items()}
    A = np.stack(applied)
    # ---- v2 reduction: per-episode scalars ---------------------------------
    # ARM = the arm joints only. fig_energy.py's JG, applied to BOTH the tau and
    # the omega terms this time (ENERGY_AUDIT.md 2.1 flagged that the shipped
    # code applied it to omega^2 only).
    _JG = {8: slice(0, 6), 11: slice(0, 7)}
    ARM = _JG.get(int(_DRV["dof"]), slice(None))
    _ac = np.stack(L_tau2); _at = np.stack(L_tot2); _aq = np.stack(L_qf2)
    _acs = np.stack(L_tau2s)
    _av = np.stack(L_v2);   _aw = np.stack(L_work); _awa = np.stack(L_workabs)
    _as = np.stack(L_sat);  _am = np.stack(L_taumax); _acon = np.stack(L_con)
    _adt = ACT_MS / 1000.0
    _tdt = TICK_MS / 1000.0
    QF = np.stack(qf_log); QV = np.stack(qvel_log)
    _root = np.asarray(base.agent.robot.get_root_pose().p, dtype=np.float64)
    _ee = np.stack(ee_xyz)
    _reach = float(np.mean(np.linalg.norm(_ee[:, :2] - _root[:2], axis=1)))
    _dur = len(A) * _tdt
    v2 = dict(
        duration_s=_dur,
        n_actuations_logged=int(_acon.shape[0]),
        # --- the OLD metric, recomputed here so the two are on the same episodes
        t2_qf_tick=float((QF ** 2).sum() * _tdt),
        t2_qf_tick_arm=float((QF[:, ARM] ** 2).sum() * _tdt),
        eff_tick_arm=float((QV[:, ARM] ** 2).sum() * _tdt),
        # --- the same thing integrated at the SUBSTEP rate
        t2_qf_sub=float(_aq.sum() * _adt),
        t2_qf_sub_arm=float(_aq[:, ARM].sum() * _adt),
        eff_sub_arm=float(_av[:, ARM].sum() * _adt),
        # --- THE NEW SIGNAL
        t2_drive_arm=float(_ac[:, ARM].sum() * _adt),
        # substeps >= SUS_SKIP only: drops widowx's 5 Hz setpoint-step transient,
        # which carries 97% of the free-space integral (see the module docstring).
        t2_drive_arm_sus=float(_acs[:, ARM].sum() * _adt),
        t2_drive_all_sus=float(_acs.sum() * _adt),
        t2_drive_all=float(_ac.sum() * _adt),
        t2_total_arm=float(_at[:, ARM].sum() * _adt),
        t2_total_all=float(_at.sum() * _adt),
        tau_drive_rms_arm=float(np.sqrt(max(_ac[:, ARM].sum() * _adt, 0.0) / max(_dur, 1e-9))),
        tau_drive_absmax_arm=float(_am[:, ARM].max()),
        sat_frac_arm=float(_as[:, ARM].mean()),
        work_drive_arm=float(_aw[:, ARM].sum()),
        work_abs_arm=float(_awa[:, ARM].sum()),
        work_abs_all=float(_awa.sum()),
        # --- contact
        contact_impulse_Ns=float(_acon[:, 7].sum()),
        contact_Fsum_mean=float(_acon[:, 2].mean()),
        contact_Fsum_p95=float(np.percentile(_acon[:, 2], 95)),
        contact_Fsum_max=float(_acon[:, 3].max()),
        contact_Fnet_mean=float(_acon[:, 0].mean()),
        contact_Fnet_max=float(_acon[:, 1].max()),
        contact_grip_impulse_Ns=float((_acon[:, 4] * _adt).sum()),
        contact_grip_Fmax=float(_acon[:, 5].max()),
        contact_frac_actuations=float(np.mean(_acon[:, 6] > 0)),
        # --- the confounders the audit showed the old metric was made of
        tcp_reach_mean=_reach,
        # CHECK (i) of ENERGY_FIX.md: the old per-tick qf is logged BEFORE the
        # first step_action and BaseAgent.reset() has just done set_qf(zeros), so
        # it is identically 0 on tick 0 of every episode ever run. Both new
        # channels are read from inside the substep loop and are not.
        tau_drive_tick0=float(_ac[0, ARM].sum()),
        tau_total_tick0=float(_at[0, ARM].sum()),
        qf_sub_tick0=float(_aq[0, ARM].sum()),
        qf_tick0=float((QF[0] ** 2).sum()))
    ag = np.asarray(ages, dtype=np.float64)
    ag_ok = ag[~np.isnan(ag)]
    n_hold = int(np.isnan(ag).sum())
    act_ag = np.asarray(act_ages, dtype=np.float64)
    results.append(dict(
        episode_id=ep_id, success=success, ticks=len(A),
        n_actuations=n_act, sim_ms=len(A) * TICK_MS, n_inferences=n_inf,
        act_age_mean_ms=float(act_ag.mean()) if act_ag.size else None,
        act_age_max_ms=float(act_ag.max()) if act_ag.size else None,
        frac_fresh=float(np.mean(fresh_flags)),
        n_initial_hold_ticks=n_hold,
        age_mean_ms=float(ag_ok.mean()) if ag_ok.size else None,
        age_max_ms=float(ag_ok.max()) if ag_ok.size else None,
        age_p50_ms=float(np.median(ag_ok)) if ag_ok.size else None,
        instruction=instruction, episode_stats=stats,
        wall_s=round(time.time() - t0, 1),
        gripper_frac_closed=float((A[:, 6] <= 0.5).mean()),
        mean_abs_rot=[float(x) for x in np.abs(A[:, 3:6]).mean(0)],
        mean_abs_trans=[float(x) for x in np.abs(A[:, :3]).mean(0)], **v2))
    n_ok = sum(r["success"] for r in results)
    r = results[-1]
    print(f"[ep {ep_id:2d}] success={success} ticks={len(A)} ({len(A)*TICK_MS/1000:.1f}s) "
          f"inf={n_inf} age_mean={r['age_mean_ms']} age_max={r['age_max_ms']} "
          f"running={n_ok}/{len(results)} ({time.time()-t0:.0f}s) stats={stats}", flush=True)
    print(f"          [v2] INT.tau_drive^2 dt={v2['t2_drive_arm']:.4g}  INT.tau_qf^2 dt={v2['t2_qf_tick_arm']:.4g}  "
          f"(sus {v2['t2_drive_arm_sus']:.4g})  |tau|max={v2['tau_drive_absmax_arm']:.3g} N.m  sat={100*v2['sat_frac_arm']:.2f}%  "
          f"contact INT.F dt={v2['contact_impulse_Ns']:.4g} N.s  Fmax={v2['contact_Fsum_max']:.4g} N  "
          f"grip Fmax={v2['contact_grip_Fmax']:.4g} N  tick0: tau_drive^2={v2['tau_drive_tick0']:.4g} "
          f"tau_total^2={v2['tau_total_tick0']:.4g} qf_sub^2={v2['qf_sub_tick0']:.4g} qf_tick^2={v2['qf_tick0']:.3g}",
          flush=True)
    if want_video:
        media.write_video(f"{run}/ep{ep_id:02d}_success_{success}.mp4", video, fps=ACT_HZ)
    np.save(f"{run}/ep{ep_id:02d}_applied_actions.npy", A)
    np.save(f"{run}/ep{ep_id:02d}_action_age_ms.npy", ag)
    np.save(f"{run}/ep{ep_id:02d}_ee_xyz.npy", np.stack(ee_xyz))
    np.save(f"{run}/ep{ep_id:02d}_ee_quat.npy", np.stack(ee_quat))
    np.save(f"{run}/ep{ep_id:02d}_obj_xyz.npy", np.stack(obj_xyz))
    np.save(f"{run}/ep{ep_id:02d}_qvel.npy", np.stack(qvel_log))
    np.save(f"{run}/ep{ep_id:02d}_qf.npy", np.stack(qf_log))
    # ---- v2 arrays (per ACTUATION, reduced from the substeps of that actuation)
    np.save(f"{run}/ep{ep_id:02d}_act_tick.npy", np.asarray(act_tick, dtype=np.int32))
    np.save(f"{run}/ep{ep_id:02d}_tau_drive_mean.npy", np.stack(L_tau))
    np.save(f"{run}/ep{ep_id:02d}_tau_drive_sq.npy", _ac)
    np.save(f"{run}/ep{ep_id:02d}_tau_drive_sq_sus.npy", _acs)
    np.save(f"{run}/ep{ep_id:02d}_tau_drive_max.npy", _am)
    np.save(f"{run}/ep{ep_id:02d}_tau_total_sq.npy", _at)
    np.save(f"{run}/ep{ep_id:02d}_qf_sub_sq.npy", _aq)
    np.save(f"{run}/ep{ep_id:02d}_qvel_sub_sq.npy", _av)
    np.save(f"{run}/ep{ep_id:02d}_work_drive.npy", _aw)
    np.save(f"{run}/ep{ep_id:02d}_work_abs.npy", _awa)
    np.save(f"{run}/ep{ep_id:02d}_sat_frac.npy", _as)
    np.save(f"{run}/ep{ep_id:02d}_contact.npy", _acon)
    np.save(f"{run}/ep{ep_id:02d}_link_com.npy", np.stack(link_com))
    json.dump({"link_mass": link_m, "link_names": link_names,
               "payload_mass": float(_o.get_mass()) if _o is not None and hasattr(_o, "get_mass") else None},
              open(f"{run}/ep{ep_id:02d}_bodies.json", "w"))
    media.write_image(f"{run}/ep{ep_id:02d}_bg.png", _bg)
    json.dump({'camera': _camname, 'camera_param': _cam, 'events': evt,
               'tick_ms': TICK_MS, 'act_every': ACT_EVERY, 'success': bool(success)},
              open(f"{run}/ep{ep_id:02d}_trace.json", 'w'), indent=1)

n_ok = sum(r["success"] for r in results)
all_age = np.concatenate([np.load(f"{run}/ep{r['episode_id']:02d}_action_age_ms.npy") for r in results])
all_age = all_age[~np.isnan(all_age)]
summary = dict(
    task=args.task, ckpt=args.ckpt, init_rng=args.init_rng,
    tick_hz=TICK_HZ, tick_ms=TICK_MS, sim_freq=SIM_FREQ,
    actuation=ACTUATE, act_hz=ACT_HZ, act_ms=ACT_MS, act_every_ticks=ACT_EVERY,
    latency_resolution_ms=TICK_MS, actuation_resolution_ms=ACT_MS,
    latency_ms=LAT, issue_period_ms=PERIOD, action_dt_ms=BASE_MS,
    delta_scale=SCALE, no_scale=args.no_scale, ensemble=args.ensemble,
    history_spacing=args.history_spacing,
    max_ticks=MAX_TICKS, horizon_ms=HORIZON_MS,
    native_control_freq=NATIVE_CF, policy_setup=policy_setup,
    staleness_ticks_mean=float(np.mean(stale_ticks)) if stale_ticks else 0.0,
    staleness_ticks_min=int(min(stale_ticks)) if stale_ticks else 0,
    staleness_ticks_max=int(max(stale_ticks)) if stale_ticks else 0,
    max_inflight=max_inflight, dispatches_per_full_episode=N_JOBS,
    age_mean_ms=float(all_age.mean()), age_max_ms=float(all_age.max()),
    age_p50_ms=float(np.median(all_age)),
    substep_dt=SUBSTEP_DT, substeps_per_actuation=int(N_SUB), sus_skip=SUS_SKIP,
    drive_stiffness=_DRV["K"].tolist(), drive_damping=_DRV["D"].tolist(),
    drive_force_limit=_DRV["FL"].tolist(), joint_names=_DRV["names"],
    contact_cols=CONTACT_COLS, trace_eval_version=2,
    n_episodes=len(results), n_success=n_ok,
    success_rate=n_ok / len(results), wall_s=round(time.time() - t_start, 1),
    episodes=results)
json.dump(summary, open(f"{run}/summary.json", "w"), indent=2)
print(f"\n=== {args.task} | tick {TICK_MS:g}ms | latency {LAT:g}ms | cadence {PERIOD:g}ms "
      f"| ens={args.ensemble} | rng={args.init_rng} ===")
print(f"SUCCESS RATE (MEASURED under MODELLED latency): "
      f"{n_ok}/{len(results)} = {100*n_ok/len(results):.1f}%")
print(f"action age at application: mean {all_age.mean():.1f} ms, max {all_age.max():.1f} ms")
print(f"total wall {time.time()-t_start:.0f}s -> {run}/summary.json")
