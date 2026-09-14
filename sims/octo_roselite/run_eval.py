"""Closed-loop evaluation of Octo in SIMPLER-env (ManiSkill2_real2sim).

Runs the official SIMPLER 'visual matching' protocol: the prepackaged env
config + obj_init_options episode_id sweep (0..23 for the bridge tasks).
"""
import argparse, json, os, sys, time
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISPLAY"] = ""

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
ap.add_argument("--ckpt", default="hf://rail-berkeley/octo-small-1.5")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--ep-start", type=int, default=0)
ap.add_argument("--init-rng", type=int, default=0)
ap.add_argument("--legacy-unnorm", action="store_true")
ap.add_argument("--no-ensemble", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--save-video-every", type=int, default=1)
ap.add_argument("--tag", default="")
args = ap.parse_args()

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from octo.model.octo_model import OctoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octo15_inference import Octo15Inference
import mediapy as media

run = args.out or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs",
    f"{args.task}_{os.path.basename(args.ckpt)}_rng{args.init_rng}"
    f"{'_legacy' if args.legacy_unnorm else ''}{'_noens' if args.no_ensemble else ''}{args.tag}")
os.makedirs(run, exist_ok=True)
print(f"[run dir] {run}", flush=True)

policy_setup = "widowx_bridge" if "widowx" in args.task else "google_robot"
env = simpler_env.make(args.task)
model = OctoModel.load_pretrained(args.ckpt)
policy = Octo15Inference(model, policy_setup=policy_setup, init_rng=args.init_rng,
                         legacy_unnorm=args.legacy_unnorm,
                         action_ensemble=not args.no_ensemble)

results, t_start = [], time.time()
for k in range(args.n):
    ep_id = args.ep_start + k
    obs, reset_info = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    instruction = env.unwrapped.get_language_instruction()
    policy.reset(instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    images, raws = [image], []
    done = truncated = False
    info = {}
    t0 = time.time()
    while not (done or truncated):
        raw_action, action = policy.step(image, instruction)
        raws.append(np.concatenate([raw_action["world_vector"],
                                    raw_action["rotation_delta"],
                                    raw_action["open_gripper"]]))
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
    success = bool(info.get("success", done))
    stats = info.get("episode_stats", {})
    stats = {k2: (bool(v) if isinstance(v, (bool, np.bool_)) else
                  (v.tolist() if isinstance(v, np.ndarray) else v))
             for k2, v in stats.items()}
    raws = np.stack(raws)
    results.append(dict(episode_id=ep_id, success=success, steps=len(raws),
                        instruction=instruction, episode_stats=stats,
                        wall_s=round(time.time() - t0, 1),
                        gripper_frac_closed=float((raws[:, 6] <= 0.5).mean()),
                        gripper_raw_min=float(raws[:, 6].min()),
                        gripper_raw_max=float(raws[:, 6].max()),
                        mean_abs_rot=[float(x) for x in np.abs(raws[:, 3:6]).mean(0)],
                        mean_abs_trans=[float(x) for x in np.abs(raws[:, :3]).mean(0)]))
    n_ok = sum(r["success"] for r in results)
    print(f"[ep {ep_id:2d}] success={success}  steps={len(raws)}  "
          f"grip_closed_frac={results[-1]['gripper_frac_closed']:.2f}  "
          f"running={n_ok}/{len(results)}  ({time.time()-t0:.0f}s)  stats={stats}", flush=True)
    if args.save_video_every and (k % args.save_video_every == 0):
        media.write_video(f"{run}/ep{ep_id:02d}_success_{success}.mp4", images, fps=5)
    np.save(f"{run}/ep{ep_id:02d}_raw_actions.npy", raws)

n_ok = sum(r["success"] for r in results)
summary = dict(task=args.task, ckpt=args.ckpt, init_rng=args.init_rng,
               legacy_unnorm=args.legacy_unnorm, action_ensemble=not args.no_ensemble,
               n_episodes=len(results), n_success=n_ok,
               success_rate=n_ok / len(results), wall_s=round(time.time() - t_start, 1),
               episodes=results)
json.dump(summary, open(f"{run}/summary.json", "w"), indent=2)
print(f"\n=== {args.task} | {args.ckpt} | init_rng={args.init_rng} "
      f"| legacy_unnorm={args.legacy_unnorm} ===")
print(f"SUCCESS RATE (MEASURED): {n_ok}/{len(results)} = {100*n_ok/len(results):.1f}%")
print(f"total wall {time.time()-t_start:.0f}s -> {run}/summary.json")
