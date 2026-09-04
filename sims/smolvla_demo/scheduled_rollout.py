#!/usr/bin/env python3
"""SmolVLA in IsaacLab Arena, driven at the latency measured on the QRB5165.

The demo in `run_demo.sh` runs the policy as if inference were free: every env
step calls `select_action`, and a fresh action chunk appears the instant the
queue empties.  On the board it does not.  One SmolVLA forward -- vision,
text, state projection, prefill, then ten flow-matching denoise steps -- takes
`metadata.makespan` milliseconds in the profiled schedule, and the robot has to
keep moving while that runs.

This driver plays that schedule back in sim time, the same way
`sims/scripts/play/play_dronet_mlp_scheduled.py` plays a schedule back for the
DroNet+MLP stack:

  * an observation is captured at the sim time inference *starts*;
  * the chunk it produces becomes available `makespan` ms of SIM time later;
  * whatever the robot does in between is done on stale actions.

Local wall-clock is irrelevant here -- the policy runs on whatever GPU is
present, and its speed is deliberately not what drives the schedule.  The board
timing does.

The arithmetic that makes this interesting, for the shipped config:

    makespan            3351.7 ms   (schedules/...smolvla_v3_unrolled10...json)
    chunk_size          50 actions  (= n_action_steps, so the whole chunk runs)
    env control rate    30 Hz       -> 33.3 ms per env step
    chunk covers        50 * 33.3   = 1666.7 ms of motion
    latency in steps    3351.7/33.3 = 100.6 env steps

The chunk covers half the time it takes to produce.  The robot runs dry for
~51 steps out of every ~101 waiting for the next one, and that is a property of
the deployment, not of the policy.  `--latency-mode none` reproduces the
original free-inference behaviour for comparison.

Usage:
    ./run_demo_scheduled.sh                      # board timing
    LATENCY_MODE=none ./run_demo_scheduled.sh    # free-inference baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

CKPT = "nvidia/smolvla-arena-gr1-microwave"
DEFAULT_SCHEDULE = (
    REPO / "schedules"
    / "scheduled_networks_smolvla_v3_unrolled10_qrb5165_greedy_profiled.json"
)


# ----------------------------------------------------------------- schedule --
def _model_from_dispatch_key(key: str) -> str:
    """'vision_dispatch_3' -> 'vision', 'decode_7_dispatch_0' -> 'decode_7'.

    Same convention as the DroNet scheduled driver, so a schedule produced by
    the same scheduler reads identically on both sides.
    """
    if "_dispatch" not in key:
        raise ValueError(f"dispatch key has no '_dispatch' segment: {key!r}")
    return key.split("_dispatch", 1)[0]


def load_schedule(path: Path) -> dict:
    """Read a profiled schedule and return its makespan plus per-model windows.

    Durations in these files are milliseconds; `metadata.makespan` is the
    end-to-end span of one inference, which is exactly the latency we owe the
    environment.
    """
    data = json.loads(Path(path).read_text())
    dispatches = data.get("dispatches")
    if not isinstance(dispatches, dict):
        raise ValueError(f"{path}: no 'dispatches' object")
    meta = data.get("metadata", {})

    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    max_end = 0.0
    for key, d in dispatches.items():
        start = float(d["start_time"])
        end = start + float(d["duration"])
        by_model[_model_from_dispatch_key(str(key))].append((start, end))
        max_end = max(max_end, end)

    makespan = float(meta.get("makespan") or 0.0)
    if makespan <= 0.0:
        makespan = max_end
    # Group the per-instance decode/action stages back into logical stages so
    # the timeline is readable: decode_0..decode_9 are one stage, ten passes.
    stages: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for model, spans in by_model.items():
        base = model.rsplit("_", 1)[0] if model.rsplit("_", 1)[-1].isdigit() else model
        stages[base].extend(spans)

    return {
        "path": str(path),
        "makespan_ms": makespan,
        "n_dispatches": len(dispatches),
        "hw_map": meta.get("profile_hw", {}),
        "stages": {k: sorted(v) for k, v in stages.items()},
    }



def _extract_success(info: dict, default: bool = False) -> bool:
    """Pull the env's own success flag out of the step info.

    The Arena wrapper reports it in two places: `final_info.is_success` is the
    per-env boolean and `log['Episode_Termination/success']` the aggregate.
    Falling back to `terminated` would be wrong here -- `time_out` is also a
    termination term, so a timed-out episode raises `terminated` as well.
    """
    try:
        return bool(np.asarray(info["final_info"]["is_success"]).any())
    except Exception:
        pass
    try:
        return float(info["log"]["Episode_Termination/success"]) > 0.0
    except Exception:
        pass
    return default


# -------------------------------------------------------------------- main --
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE,
                    help="profiled schedule whose makespan is the inference latency")
    ap.add_argument("--latency-mode", choices=["schedule", "none", "manual"],
                    default=os.environ.get("LATENCY_MODE", "schedule"),
                    help="'schedule' uses the board makespan; 'none' is the "
                         "free-inference baseline; 'manual' uses --latency-ms")
    ap.add_argument("--latency-ms", type=float, default=None,
                    help="explicit latency for --latency-mode manual")
    ap.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", 1)))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", 300)))
    ap.add_argument("--control-hz", type=float, default=30.0,
                    help="env control rate; the Arena wrapper advertises 30 fps")
    ap.add_argument("--stall-policy", choices=["hold", "zero"], default="hold",
                    help="what the robot does while a chunk is in flight")
    ap.add_argument("--replan", choices=["on_empty", "pipelined"], default="on_empty",
                    help="'on_empty' starts the next inference only once the "
                         "current chunk runs out; 'pipelined' starts it as soon "
                         "as the previous one returns, so chunks arrive every "
                         "`latency` ms instead of every latency+coverage ms")
    ap.add_argument("--no-prime", action="store_true",
                    help="do NOT prime the first chunk. Off by default: with "
                         "latency applied to the very first chunk the robot has "
                         "no pose to hold and gets commanded zeros, which just "
                         "collapses the GR1 and measures the harness, not the "
                         "deployment.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "scheduled")
    ap.add_argument("--headless", default=os.environ.get("HEADLESS", "true"))
    ap.add_argument("--video", action="store_true", default=False)
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    sched = load_schedule(args.schedule)
    if args.latency_mode == "schedule":
        latency_ms = sched["makespan_ms"]
    elif args.latency_mode == "manual":
        if args.latency_ms is None:
            raise SystemExit("--latency-mode manual needs --latency-ms")
        latency_ms = float(args.latency_ms)
    else:
        latency_ms = 0.0

    dt = 1.0 / args.control_hz
    latency_steps = latency_ms / 1000.0 / dt

    print("=" * 66)
    print("SmolVLA @ QRB5165 timing, played back in Isaac sim time")
    print("=" * 66)
    print(f"  schedule       : {Path(sched['path']).name}")
    print(f"  dispatches     : {sched['n_dispatches']}  hw map {sched['hw_map']}")
    print(f"  makespan       : {sched['makespan_ms']:.1f} ms")
    print(f"  latency mode   : {args.latency_mode}  -> {latency_ms:.1f} ms "
          f"({latency_steps:.1f} env steps @ {args.control_hz:.0f} Hz)")
    for stage, spans in sorted(sched["stages"].items(),
                               key=lambda kv: sum(e - s for s, e in kv[1]),
                               reverse=True)[:6]:
        busy = sum(e - s for s, e in spans)
        print(f"    {stage:<12} {len(spans):>3} dispatch(es)  {busy:8.1f} ms busy")

    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    # ---- policy -----------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition, transition_to_policy_action,
    )

    print(f"\nLoading {CKPT} on {args.device} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(args.device).to(torch.float32).eval()
    policy.reset()
    chunk_size = int(policy.config.chunk_size)
    n_action_steps = int(policy.config.n_action_steps)
    print(f"  chunk_size={chunk_size} n_action_steps={n_action_steps} "
          f"denoise steps={getattr(policy.config, 'num_steps', '?')}")
    print(f"  one chunk covers {n_action_steps * dt * 1000:.1f} ms of motion; "
          f"producing it costs {latency_ms:.1f} ms")

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json",
        overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {
                "rename_map": {
                    "observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam",
                },
            },
        },
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # ---- env (same call shape as run_demo.sh / onnx_rollout.py) -----------
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import IsaaclabArenaEnv
    from lerobot.envs.utils import preprocess_observation, add_envs_task

    print("\nBuilding IsaacLab Arena env (gr1_microwave / gr1_pink / mustard_bottle) ...",
          flush=True)
    env_cfg = IsaaclabArenaEnv(
        environment="gr1_microwave",
        embodiment="gr1_pink",
        object="mustard_bottle",
        episode_length=args.max_steps,
        headless=(str(args.headless).lower() != "false"),
        device=args.device,
        seed=args.seed,
        state_keys="robot_joint_pos",
        camera_keys="robot_pov_cam_rgb",
        enable_cameras=True,
        video=bool(args.video),
        video_length=args.max_steps,
        video_interval=15,
    )
    env_dict = make_env(env_cfg, n_envs=1, trust_remote_code=True)
    env = next(iter(env_dict.values()))[0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config,
    )

    def to_env_action(a: torch.Tensor) -> np.ndarray:
        a = postprocessor(a)
        a = env_postprocessor({"action": a})
        return a["action"].detach().cpu().numpy()

    # ---- rollout ----------------------------------------------------------
    episodes = []
    t_wall0 = time.perf_counter()
    for ep in range(args.episodes):
        policy.reset()
        observation, _ = env.reset()

        chunk: torch.Tensor | None = None   # chunk currently being executed
        chunk_pos = 0
        pending: tuple[float, torch.Tensor] | None = None  # (ready_t, chunk)
        last_action: np.ndarray | None = None

        n_infer = 0
        n_stall = 0
        timeline = []
        success = False
        steps_used = args.max_steps

        for step in range(args.max_steps):
            t = step * dt

            # A chunk that was in flight lands once sim time reaches its
            # readiness. This is the only place latency enters.
            if pending is not None and t + 1e-9 >= pending[0]:
                # A landing chunk always supersedes whatever is executing: it
                # was computed from a newer observation.
                chunk, chunk_pos, pending = pending[1], 0, None

            exhausted = chunk is None or chunk_pos >= min(n_action_steps, chunk.shape[1])
            # `on_empty` waits for the chunk to run out before thinking again, so
            # the cadence is coverage + latency. `pipelined` keeps exactly one
            # inference in flight at all times, so the cadence is just latency.
            want = exhausted if args.replan == "on_empty" else True
            first = chunk is None and pending is None
            if pending is None and want:
                obs = preprocess_observation(observation)
                obs = add_envs_task(env, obs)
                obs = env_preprocessor(obs)
                obs = preprocessor(obs)
                with torch.inference_mode():
                    new_chunk = policy.predict_action_chunk(obs)
                n_infer += 1
                # The first chunk lands immediately unless --no-prime: before it
                # exists there is no commanded pose to hold, and commanding zero
                # joint targets folds the humanoid up. Priming measures the
                # steady-state cadence rather than that artefact.
                if latency_ms <= 0.0 or (first and not args.no_prime):
                    chunk, chunk_pos = new_chunk, 0
                else:
                    pending = (t + latency_ms / 1000.0, new_chunk)
                exhausted = chunk is None or chunk_pos >= min(n_action_steps, chunk.shape[1])

            if not exhausted:
                action_np = to_env_action(chunk[:, chunk_pos])
                chunk_pos += 1
                stalled = False
            else:
                # Nothing fresh to execute: the chunk ran out before its
                # successor arrived. Hold the last command (a held joint
                # target) or command zeros.
                stalled = True
                n_stall += 1
                if args.stall_policy == "hold" and last_action is not None:
                    action_np = last_action
                else:
                    action_np = np.zeros_like(last_action) if last_action is not None \
                        else to_env_action(torch.zeros(1, policy.config.max_action_dim,
                                                       device=args.device))

            last_action = action_np
            timeline.append({"step": step, "t_ms": t * 1000.0, "stalled": bool(stalled),
                             "in_flight": pending is not None})

            observation, reward, terminated, truncated, info = env.step(action_np)

            done = bool(np.any(terminated)) or bool(np.any(truncated))
            if done:
                # This env has TWO termination terms, `success` and `time_out`,
                # and both raise `terminated` -- so `terminated.any()` counts a
                # timeout as a win. Read the explicit flag instead.
                success = _extract_success(info, default=False)
                steps_used = step + 1
                break

        stall_frac = n_stall / max(steps_used, 1)
        episodes.append({
            "episode": ep, "success": success, "steps": steps_used,
            "inferences": n_infer, "stalled_steps": n_stall,
            "stall_fraction": stall_frac,
            "sim_seconds": steps_used * dt,
            "timeline": timeline,
        })
        print(f"[ep {ep}] success={success} steps={steps_used} "
              f"inferences={n_infer} stalled={n_stall} ({stall_frac*100:.1f}% of steps)",
              flush=True)

    wall = time.perf_counter() - t_wall0
    n_ok = sum(1 for e in episodes if e["success"])
    tot_steps = sum(e["steps"] for e in episodes)
    tot_stall = sum(e["stalled_steps"] for e in episodes)

    print("\n" + "=" * 66)
    print(f"  latency mode     : {args.latency_mode} ({latency_ms:.1f} ms)")
    print(f"  success          : {n_ok}/{len(episodes)}")
    print(f"  env steps        : {tot_steps}  ({tot_steps * dt:.2f} s sim)")
    print(f"  stalled steps    : {tot_stall} ({tot_stall / max(tot_steps,1) * 100:.1f}%)")
    print(f"  inferences       : {sum(e['inferences'] for e in episodes)}")
    print(f"  wall clock       : {wall:.1f}s")
    print("=" * 66)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.latency_mode if args.latency_mode != "manual" else f"manual{latency_ms:.0f}"
    if latency_ms > 0.0:
        tag = f"{tag}_{args.replan}"
    out = args.out / f"scheduled_rollout_{tag}.json"
    out.write_text(json.dumps({
        "schedule": sched,
        "latency_mode": args.latency_mode,
        "latency_ms": latency_ms,
        "control_hz": args.control_hz,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "chunk_covers_ms": n_action_steps * dt * 1000.0,
        "stall_policy": args.stall_policy,
        "replan": args.replan,
        "primed": not args.no_prime,
        "episodes": episodes,
        "wall_seconds": wall,
    }, indent=1))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
