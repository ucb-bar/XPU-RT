"""RoSE-lite FINE-GRAIN: sim ticks faster than policy results arrive, ZOH in between.

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
ap.add_argument("--arm-gain-scale", type=float, default=None,
                help="Scale google's arm PD stiffness and damping. A PLANNER-FREE "
                     "controller reaches its target by PD tracking alone, so it "
                     "undershoots by a factor set by the gains -- which is why the "
                     "planner-free configs score well on the PUSH task and fail the "
                     "GRASP. Raising the gains trades real2sim fidelity (already "
                     "abandoned under sim-only rules) for tracking accuracy. Applies to "
                     "google_robot only.")
ap.add_argument("--planner-limit-scale", type=float, default=None,
                help="Scale the google planner's velocity/acceleration/jerk limits "
                     "(GoogleRobotDefaultConfig.arm_*_limit). THE lever for running "
                     "google at a fine control grid. The planner's path duration is "
                     "t = 2*sqrt(d/a), so subdividing the period by N shrinks the period "
                     "by N but the plan only by sqrt(N): the executed fraction falls as "
                     "P/(T*sqrt(N)) and the arm never completes a move. Note this is "
                     "INDEPENDENT of sim_freq -- more sim steps inside a period do not "
                     "buy the plan more wall-clock time. Raising the acceleration limit "
                     "does: 27 Hz needs a ~ 32 rad/s^2 against the configured 2.0. This "
                     "ABANDONS the real2sim calibration and is valid only for sim-internal "
                     "comparisons; it must be applied to the native-rate control too.")
ap.add_argument("--sim-freq", type=int, default=None,
                help="Override the env's sim_freq (physics rate). The planner-interpolated "
                     "google controllers execute only sim_freq/control_freq steps of their "
                     "planned path per control period, so refining control 9x cuts that "
                     "from 171 steps to 19 (11%%) and the arm never traverses its plan. "
                     "Scaling sim_freq by the same factor (513 -> 4617) restores 171 steps "
                     "per period, isolating path TRUNCATION from the other mechanisms. "
                     "Costs 9x the physics compute. The tick must still divide sim_freq.")
ap.add_argument("--allow-lag-discard", action="store_true",
                help="Permit a POSE-RELATIVE arm controller under fine-tick rescaling. "
                     "Normally refused: each control step re-anchors the delta to the arm's "
                     "current pose, so tracking lag is discarded and compounds (0/24 at zero "
                     "latency). Use ONLY to measure that mechanism deliberately.")
ap.add_argument("--sticky-repeat", default="auto",
                help="google_robot gripper sticky hold, in ACTUATIONS. Upstream uses 15, "
                     "which at the native 3 Hz is 15 x 333 ms = 5.0 s of wall clock -- "
                     "the time the gripper physically needs to close. It is a COUNT, not "
                     "a duration, so refining actuation shortens the hold proportionally: "
                     "at 27 Hz the same 15 gives 0.56 s and no grasp task can succeed. "
                     "'auto' (default) scales it as 15 x ACT_HZ/NATIVE_CF, holding the "
                     "wall-clock window fixed; an integer overrides. In every native-rate "
                     "run ACT_HZ == NATIVE_CF, so 'auto' is exactly 15 and nothing already "
                     "measured changes. widowx has no sticky machine at all (its gripper "
                     "is an absolute position), so this is ignored there.")
ap.add_argument("--control-mode", default=None,
                help="Override the env's registered control_mode. The google_robot tasks "
                     "register arm_pd_ee_delta_pose_align_interpolate_by_planner, which is "
                     "POSE-RELATIVE (use_target=False): every control step re-anchors the "
                     "delta to where the arm actually IS, so any tracking lag is DISCARDED "
                     "rather than carried. That is what makes fine-tick rescaling collapse "
                     "on google_robot -- not the planner per se. The target-accumulating "
                     "variants (arm_pd_ee_target_delta_pose_align[_interpolate_by_planner]) "
                     "retain the commanded displacement the way widowx's "
                     "arm_pd_ee_target_delta_pose_align2 does. Changing this moves the sim "
                     "off SIMPLER's validated real2sim configuration, so ANY use must be "
                     "validated at the NATIVE control rate against the registered mode "
                     "first.")
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
            if args.sim_freq is not None:
                ret["sim_freq"] = args.sim_freq
        # In 'native' mode ACT_HZ == the native control_freq, so this write is a
        # no-op and the env is left exactly as the task registered it.
        ret["control_freq"] = ACT_HZ
        if args.control_mode is not None and ret.get("control_mode") is not None:
            _OBSERVED.setdefault("control_mode_registered", ret["control_mode"])
            ret["control_mode"] = args.control_mode
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

if args.planner_limit_scale is not None and policy_setup == "google_robot":
    from mani_skill2_real2sim.agents.configs.google_robot import defaults as _gd
    _k = args.planner_limit_scale
    _orig_init = _gd.GoogleRobotDefaultConfig.__init__
    def _scaled_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        # The GRIPPER is planner-interpolated too
        # (gripper_pd_joint_target_delta_pos_interpolate_by_planner), so it truncates
        # exactly like the arm. Scaling only the arm left the gripper unable to close at
        # 27 Hz -- measured grip_frac_closed 0.013 vs 0.257 at the native rate, which is
        # why coke scored 0/12 while the arm itself was tracking fine.
        for _n in ("arm_vel_limit", "arm_acc_limit", "arm_jerk_limit",
                   "gripper_vel_limit", "gripper_acc_limit", "gripper_jerk_limit"):
            setattr(self, _n, getattr(self, _n) * _k)
        if args.arm_gain_scale is not None:
            import numpy as _np
            for _n in ("arm_stiffness", "arm_damping"):
                _v = getattr(self, _n)
                setattr(self, _n, (_np.asarray(_v) * args.arm_gain_scale).tolist()
                        if isinstance(_v, (list, tuple)) else _v * args.arm_gain_scale)
    _gd.GoogleRobotDefaultConfig.__init__ = _scaled_init
    print(f"[planner] arm AND gripper vel/acc/jerk limits scaled x{_k:g} -- real2sim "
          f"calibration ABANDONED (sim-internal comparisons only)", flush=True)

env = simpler_env.make(args.task)
base = env.unwrapped
assert base.control_freq == ACT_HZ, (base.control_freq, ACT_HZ)
if args.sim_freq is not None and base.sim_freq == args.sim_freq:
    print(f"[env] sim_freq OVERRIDDEN {_OBSERVED.get('sim_freq')} -> {base.sim_freq} "
          f"({base._sim_steps_per_control} sim steps per control period; the native rate "
          f"gives {_OBSERVED.get('sim_freq', 0) // NATIVE_CF})", flush=True)
elif base.sim_freq != SIM_FREQ:
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
# Split off the gripper half: the gripper mode legitimately contains "target_delta"
# and would otherwise mask a pose-relative ARM controller.
_arm_cm = _cm.split("_gripper_")[0]
_accumulates = "target" in _arm_cm
if not _accumulates and ACTUATE == "fine" and abs(SCALE - 1.0) > 1e-9 and not args.allow_lag_discard:
    raise SystemExit(
        f"REFUSING to run: arm control_mode {_arm_cm} is POSE-RELATIVE "
        f"(use_target=False), so each control step re-anchors the delta to the arm's "
        f"CURRENT pose and any tracking lag is discarded. Under fine-tick rescaling "
        f"(scale {SCALE:.4g}) the lag compounds and the arm crawls -- it scores 0/24 at "
        f"ZERO latency where the stock harness gets 10/24 (GOOGLE_ROBOT_PORT.md). Use "
        f"--actuation native, or --control-mode with a target-accumulating arm "
        f"controller (validate it at the native rate first).")
if "interpolate_by_planner" in _arm_cm and ACTUATE == "fine" and abs(SCALE - 1.0) > 1e-9:
    _fine_steps = SIM_FREQ // ACT_HZ
    _nat_steps = SIM_FREQ // NATIVE_CF
    print(f"[warn] arm controller {_arm_cm} plans a trajectory over the control period, "
          f"but only {_fine_steps} of the {_nat_steps} sim steps a native period would "
          f"give are executed before it re-plans ({100.0*_fine_steps/_nat_steps:.0f}%). "
          f"Target accumulation carries the commanded displacement, but it does NOT fix "
          f"path truncation -- the arm can fall progressively behind a target that keeps "
          f"advancing. NOT the validated SIMPLER configuration; validate against the "
          f"native rate before trusting any result.", flush=True)

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
    def __init__(self, n_repeat=15):
        # Upstream's 15 is in control steps at the NATIVE rate. Scaled by the caller
        # so the hold stays a fixed wall-clock duration when actuation is refined.
        self.N_REPEAT = int(n_repeat)
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


if args.sticky_repeat == "auto":
    _STICKY_N = int(round(15 * ACT_HZ / NATIVE_CF))
else:
    _STICKY_N = int(args.sticky_repeat)
sticky = StickyGripper(_STICKY_N) if policy_setup == "google_robot" else None
if sticky is not None:
    print(f"[gripper] sticky hold = {_STICKY_N} actuations x {ACT_MS:.2f} ms = "
          f"{_STICKY_N * ACT_MS / 1000:.2f} s wall clock "
          f"(upstream 15 at the native {NATIVE_CF} Hz = {15 * 1000.0 / NATIVE_CF / 1000:.2f} s)",
          flush=True)

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
            base.step_action(HOLD_CMD if cur_raw is None else to_env_action(raw7))
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
        mean_abs_trans=[float(x) for x in np.abs(A[:, :3]).mean(0)]))
    n_ok = sum(r["success"] for r in results)
    r = results[-1]
    print(f"[ep {ep_id:2d}] success={success} ticks={len(A)} ({len(A)*TICK_MS/1000:.1f}s) "
          f"inf={n_inf} age_mean={r['age_mean_ms']} age_max={r['age_max_ms']} "
          f"running={n_ok}/{len(results)} ({time.time()-t0:.0f}s) stats={stats}", flush=True)
    if want_video:
        media.write_video(f"{run}/ep{ep_id:02d}_success_{success}.mp4", video, fps=ACT_HZ)
    np.save(f"{run}/ep{ep_id:02d}_applied_actions.npy", A)
    np.save(f"{run}/ep{ep_id:02d}_action_age_ms.npy", ag)

n_ok = sum(r["success"] for r in results)
all_age = np.concatenate([np.load(f"{run}/ep{r['episode_id']:02d}_action_age_ms.npy") for r in results])
all_age = all_age[~np.isnan(all_age)]
summary = dict(
    task=args.task, ckpt=args.ckpt, init_rng=args.init_rng,
    tick_hz=TICK_HZ, tick_ms=TICK_MS, sim_freq=int(base.sim_freq),
    sim_freq_native=SIM_FREQ, sim_steps_per_control=int(base._sim_steps_per_control),
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
