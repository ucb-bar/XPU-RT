"""RoSE-lite: timing-only replay of Octo in SIMPLER with MEASURED QRB5165 latency.

This is NOT full RoSE: no hardware in the loop, no board execution, no int8
weights. The policy runs at FULL PRECISION (fp32) on the host GPU exactly as in
the free-running baseline; the only thing the board measurement contributes is
*when* a policy output becomes available to the environment.

Timing model (MODELLED, from MEASURED board latency)
----------------------------------------------------
PutEggplantInBasketScene-v0 sets control_freq=5 (put_on_in_scene.py:167), so one
env step advances 200 ms. An inference issued at env step k with observation o_k
becomes available at step

    k + D,   D = ceil(latency_ms / 200 ms)

latency 0 -> D=0 -> the stock configuration, bit-for-bit.

Scheduling: SERIAL by default. A single QRB5165 chain runs one inference at a
time, so at most one request is in flight and the next is issued the instant the
previous lands -> inferences at steps 0, D, 2D, ...  --pipeline models the
optimistic alternative (a fresh request every 200 ms, each D steps late), which
one accelerator chain cannot actually sustain; it is an upper bound, not a
deployable configuration.

Chunk consumption (--chunk-mode):
  from_arrival  (default) on arrival the 4-action chunk is executed open-loop
                from entry 0, one entry per step, until the next chunk lands and
                supersedes it. This is what deployed chunked policies do. For
                D<=4 it never runs dry, so no zero-order hold is needed.
  timestamp     entry j of a chunk issued at step k is the action *for* step
                k+j; entries already in the past on arrival are discarded, and
                the last entry is held (ZOH) once the chunk is exhausted.

Ensembling (--ensemble):
  Stock SimplerEnv averages the last 4 predictions that target the current
  timestep (ActionEnsembler, temp=0.0 -> uniform mean), which assumes a fresh
  inference EVERY step. Under serial latency at most one prediction ever targets
  a given timestep, so ensembling is vacuous by construction and the latency runs
  use --ensemble none. Because that differs from the stock baseline, a
  latency-0 --ensemble none control is run separately so the ensembler's
  contribution is separated from latency's rather than silently absorbed.
  --ensemble stock reproduces stock behaviour and is only meaningful at D=0.
  --pipeline uses timestep-aligned averaging over however many predictions
  actually target the current step (4 at D=0, 2 at D=2, 1 at D=3, 0 at D>=4).

At t < D nothing has landed, so the robot HOLDS: zero end-effector delta (the
controller is arm_pd_ee_target_delta_pose_align2, so a zero delta means "no new
command" and the arm keeps servoing to its last target) with the gripper open,
which is its reset state.
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
ap.add_argument("--latency-ms", type=float, default=0.0)
ap.add_argument("--control-period-ms", type=float, default=200.0)
ap.add_argument("--chunk-mode", default="from_arrival",
                choices=["from_arrival", "timestamp"])
ap.add_argument("--ensemble", default="none", choices=["none", "stock"])
ap.add_argument("--pipeline", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--save-video-every", type=int, default=0)
ap.add_argument("--tag", default="")
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from octo15_inference import Octo15Inference
import mediapy as media

D = max(int(math.ceil(args.latency_ms / args.control_period_ms - 1e-9)), 0)
CHUNK = 4
HOLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # zero ee delta, gripper open

if args.ensemble == "stock" and D > 0 and not args.pipeline:
    print("[WARN] --ensemble stock with serial latency D>0: ActionEnsembler's "
          "history indexing assumes one push per timestep. Results not trustworthy.",
          flush=True)

run = args.out or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs",
    f"lat{args.latency_ms:g}ms_D{D}_{args.chunk_mode}_ens-{args.ensemble}"
    f"{'_pipe' if args.pipeline else ''}_rng{args.init_rng}{args.tag}")
os.makedirs(run, exist_ok=True)
print(f"[run dir] {run}", flush=True)
print(f"[timing MODELLED] latency={args.latency_ms} ms / period "
      f"{args.control_period_ms} ms -> D={D} steps; "
      f"schedule={'PIPELINED (upper bound)' if args.pipeline else 'serial'}; "
      f"chunk_mode={args.chunk_mode}; ensemble={args.ensemble}", flush=True)

policy_setup = "widowx_bridge" if "widowx" in args.task else "google_robot"
env = simpler_env.make(args.task)
model = OctoModel.load_pretrained(args.ckpt)
policy = Octo15Inference(model, policy_setup=policy_setup, init_rng=args.init_rng,
                         legacy_unnorm=False, action_ensemble=False)


def predict_chunk():
    """One inference on the policy's current 2-frame history -> (4,7) chunk.
    Mirrors Octo15Inference.step() up to (but not including) ensembling."""
    policy.rng, key = jax.random.split(policy.rng)
    images, pad_mask = policy._obtain_image_history_and_mask()
    obs = {"image_primary": images[None], "timestep_pad_mask": pad_mask[None]}
    chunk = np.asarray(policy.model.sample_actions(
        obs, policy.task, unnormalization_statistics=policy.stats, rng=key))[0]
    assert chunk.shape == (CHUNK, 7), chunk.shape
    return chunk


def postprocess(raw7):
    """raw 7-vector -> env action, identical to Octo15Inference (widowx branch)."""
    wv = np.asarray(raw7[:3], dtype=np.float64)
    roll, pitch, yaw = np.asarray(raw7[3:6], dtype=np.float64)
    ax, ang = euler2axangle(roll, pitch, yaw)
    grip = 2.0 * (float(raw7[6]) > 0.5) - 1.0
    return np.concatenate([wv, ax * ang, np.array([grip])])


results, t_start = [], time.time()
for k in range(args.n):
    ep_id = args.ep_start + k
    obs, reset_info = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    instruction = env.unwrapped.get_language_instruction()
    policy.reset(instruction)
    ensembler = ActionEnsembler(CHUNK, 0.0) if args.ensemble == "stock" else None
    image = get_image_from_maniskill2_obs_dict(env, obs)
    images_log, applied, fresh_flags = [image], [], []

    inflight = []            # (arrival_step, issue_step, chunk) not yet delivered
    delivered = deque(maxlen=CHUNK + 2)   # (issue_step, chunk) already landed
    plan = deque()           # (target_step | None, raw7) queued for execution
    last_raw = None          # ZOH memory
    n_inf, busy, t = 0, False, 0
    done = truncated = False
    info = {}
    t0 = time.time()
    while not (done or truncated):
        # Camera runs free at 5 Hz regardless of policy rate: every env frame is
        # buffered, so an inference issued at step t sees [frame_{t-1}, frame_t],
        # exactly as in the stock config. This isolates pure actuation latency
        # from any degradation of the observation history.
        policy._add_image_to_history(policy._resize_image(image))

        # --- run the scheduler to a fixpoint for this control period ---
        issued_this_step = False
        new_arrivals = []
        while True:
            landed = [r for r in inflight if r[0] <= t]
            if landed:
                inflight = [r for r in inflight if r[0] > t]
                busy = False
                new_arrivals.extend(landed)
                for _a, iss, ch in landed:
                    delivered.append((iss, ch))
                continue
            if (args.pipeline or not busy) and not issued_this_step:
                inflight.append((t + D, t, predict_chunk()))
                n_inf += 1
                busy = True
                issued_this_step = True
                continue
            break

        # --- turn what has landed into this step's plan ---
        if args.pipeline:
            # timestep-aligned averaging over every prediction targeting step t
            cands = [ch[t - iss] for iss, ch in delivered if 0 <= t - iss < CHUNK]
            plan = deque([(None, np.mean(np.stack(cands), axis=0))]) if cands else deque()
        elif new_arrivals:
            _a, iss, ch = new_arrivals[-1]
            if ensembler is not None:
                plan = deque([(None, ensembler.ensemble_action(ch))])
            elif args.chunk_mode == "from_arrival":
                plan = deque((None, a) for a in ch)
            else:  # timestamp-aligned
                plan = deque((iss + j, ch[j]) for j in range(CHUNK) if iss + j >= t)

        # --- select this step's action ---
        while plan and plan[0][0] is not None and plan[0][0] < t:
            plan.popleft()                       # drop entries whose slot passed
        if plan and (plan[0][0] is None or plan[0][0] == t):
            _tgt, raw7 = plan.popleft()
            last_raw, fresh = raw7, True
        elif last_raw is not None:
            raw7, fresh = last_raw, False        # ZOH: chunk exhausted
        else:
            raw7, fresh = HOLD, False            # t < D: nothing has landed yet

        applied.append(np.asarray(raw7, dtype=np.float64))
        fresh_flags.append(bool(fresh))
        obs, reward, done, truncated, info = env.step(postprocess(raw7))
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images_log.append(image)
        t += 1

    success = bool(info.get("success", done))
    stats = info.get("episode_stats", {})
    stats = {k2: (bool(v) if isinstance(v, (bool, np.bool_)) else
                  (v.tolist() if isinstance(v, np.ndarray) else v))
             for k2, v in stats.items()}
    A = np.stack(applied)
    results.append(dict(episode_id=ep_id, success=success, steps=len(A),
                        n_inferences=n_inf, frac_fresh=float(np.mean(fresh_flags)),
                        instruction=instruction, episode_stats=stats,
                        wall_s=round(time.time() - t0, 1),
                        gripper_frac_closed=float((A[:, 6] <= 0.5).mean()),
                        mean_abs_rot=[float(x) for x in np.abs(A[:, 3:6]).mean(0)],
                        mean_abs_trans=[float(x) for x in np.abs(A[:, :3]).mean(0)]))
    n_ok = sum(r["success"] for r in results)
    print(f"[ep {ep_id:2d}] success={success} steps={len(A)} inf={n_inf} "
          f"fresh={results[-1]['frac_fresh']:.2f} running={n_ok}/{len(results)} "
          f"({time.time()-t0:.0f}s) stats={stats}", flush=True)
    if args.save_video_every and (k % args.save_video_every == 0):
        media.write_video(f"{run}/ep{ep_id:02d}_success_{success}.mp4", images_log, fps=5)
    np.save(f"{run}/ep{ep_id:02d}_applied_actions.npy", A)

n_ok = sum(r["success"] for r in results)
summary = dict(task=args.task, ckpt=args.ckpt, init_rng=args.init_rng,
               latency_ms=args.latency_ms, control_period_ms=args.control_period_ms,
               D_steps=D, chunk_mode=args.chunk_mode, ensemble=args.ensemble,
               pipeline=args.pipeline,
               n_episodes=len(results), n_success=n_ok,
               success_rate=n_ok / len(results), wall_s=round(time.time() - t_start, 1),
               episodes=results)
json.dump(summary, open(f"{run}/summary.json", "w"), indent=2)
print(f"\n=== {args.task} | latency={args.latency_ms}ms (D={D}) | rng={args.init_rng} "
      f"| {args.chunk_mode} | ens={args.ensemble}{' | pipelined' if args.pipeline else ''} ===")
print(f"SUCCESS RATE (MEASURED under MODELLED latency): "
      f"{n_ok}/{len(results)} = {100*n_ok/len(results):.1f}%")
print(f"total wall {time.time()-t_start:.0f}s -> {run}/summary.json")
