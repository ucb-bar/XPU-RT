"""
M13 — RL policy scheduler (PPO via stable_baselines3).

Simplified MDP that fits training in a CPU-only session:
  * Op order is fixed by HEFT upward-rank priority (the policy does NOT learn
    ordering, only placement).
  * At each step the policy chooses a machine combination for the next ready
    op from HEFT's priority queue.
  * State (Box): per-machine free-time (3 floats) + current op's normalized
    per-machine cost (3 floats) + infeasibility mask (3 bits) + topo depth.
  * Action: Discrete(MAX_COMBOS).
  * Reward: -delta(machine_busy_until) per step (dense shaping) +
            -(makespan / lower_bound) terminal reward.

Inference: load policy, step through the env greedily (argmax) on a single
workload, recover (t, alpha).

Lazy imports for gymnasium and stable_baselines3 so the registry survives
without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


MAX_COMBOS = 3
OBS_DIM = MAX_COMBOS + MAX_COMBOS + MAX_COMBOS + 1  # free + costs + feasibility + depth


# ---------------------------------------------------------------------------
# Lazy library helpers
# ---------------------------------------------------------------------------


def _lazy_gym():
    try:
        import gymnasium as gym  # noqa
        return gym
    except ImportError as exc:
        raise RuntimeError("gymnasium is required for the RL scheduler.") from exc


def _lazy_sb3():
    try:
        from stable_baselines3 import PPO  # noqa
        return PPO
    except ImportError as exc:
        raise RuntimeError("stable_baselines3 is required for the RL scheduler.") from exc


# ---------------------------------------------------------------------------
# Lower bound (shared with the cost model)
# ---------------------------------------------------------------------------


def _lower_bound(workload: Workload) -> float:
    from scheduler_ml import _lower_bound_makespan
    return _lower_bound_makespan(workload)


def _heft_priority_order(workload: Workload) -> List[int]:
    from scheduler_heft import _upward_rank
    rank = _upward_rank(workload)
    # Sort by descending rank; tie-break by original index for determinism.
    return [i for _, i in sorted([(-r, idx) for idx, r in enumerate(rank)])]


# ---------------------------------------------------------------------------
# Gym Environment
# ---------------------------------------------------------------------------


def make_env_class():
    gym = _lazy_gym()
    from gymnasium import spaces

    class XPURTSchedulingEnv(gym.Env):
        """One workload per episode. HEFT-priority order is fixed; the
        policy picks the combo for each op in turn."""

        metadata = {"render_modes": []}

        def __init__(self, workload_provider, max_ops: int = 50,
                     reward_normalize: bool = True):
            """``workload_provider()`` returns a fresh ``Workload`` per
            episode (so we get curriculum / random sampling)."""
            super().__init__()
            self.workload_provider = workload_provider
            self.max_ops = max_ops
            self.reward_normalize = reward_normalize
            self.action_space = spaces.Discrete(MAX_COMBOS)
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32,
            )
            self._reset_state()

        def _reset_state(self):
            self.workload = None
            self.order: List[int] = []
            self.step_idx = 0
            self.combo_choice: Dict[int, int] = {}
            self.finish: Dict[int, float] = {}
            self.machine_busy: List[float] = [0.0, 0.0, 0.0]
            self.lower_bound = 1.0

        def _current_obs(self) -> np.ndarray:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            if self.workload is None or self.step_idx >= len(self.order):
                return obs
            op = self.workload.operations[self.order[self.step_idx]]
            combos = self.workload.get_machine_combinations()
            machines = list(self.workload.machines)
            # Per-machine free time normalized by lower_bound.
            for k in range(MAX_COMBOS):
                obs[k] = self.machine_busy[k] / max(1.0, self.lower_bound)
            # Per-combo cost normalized.
            for k in range(MAX_COMBOS):
                if k < len(combos):
                    cost = float(op.get_duration_for_combination(k, combos, machines))
                    obs[MAX_COMBOS + k] = min(1e3, cost / max(1.0, self.lower_bound))
                else:
                    obs[MAX_COMBOS + k] = 10.0  # pad
            # Feasibility mask.
            for k in range(MAX_COMBOS):
                if k < len(combos):
                    obs[2 * MAX_COMBOS + k] = 0.0 if k in op.infeasible_combinations else 1.0
                else:
                    obs[2 * MAX_COMBOS + k] = 0.0
            obs[3 * MAX_COMBOS] = self.step_idx / max(1, len(self.order))
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._reset_state()
            self.workload = self.workload_provider()
            self.order = _heft_priority_order(self.workload)
            self.lower_bound = _lower_bound(self.workload)
            return self._current_obs(), {}

        def step(self, action):
            from scheduler_heft import _earliest_start_on_combo
            assert self.workload is not None
            op_idx = self.order[self.step_idx]
            op = self.workload.operations[op_idx]
            combos = self.workload.get_machine_combinations()
            machines = list(self.workload.machines)
            n_combos = len(combos)
            # Force feasibility: if action picks an infeasible combo or out
            # of range, fall back to a feasible one.
            k = int(action)
            if k >= n_combos or k in op.infeasible_combinations:
                feasible = [c for c in range(n_combos) if c not in op.infeasible_combinations]
                k = feasible[0] if feasible else 0

            # Compute earliest start using the running schedule state.
            pred_finish = self.finish
            pred_combo = self.combo_choice
            machine_busy_dict = {machines[i]: self.machine_busy[i] if i < len(self.machine_busy) else 0.0
                                 for i in range(len(machines))}
            est = _earliest_start_on_combo(
                self.workload, op, k, pred_finish, pred_combo, machine_busy_dict
            )
            dur = float(op.get_duration_for_combination(k, combos, machines))
            finish_t = est + dur

            # Dense reward shaping: penalize busy-time growth.
            before = sum(self.machine_busy[:n_combos])
            for m in combos[k]:
                mi = machines.index(m) if m in machines else 0
                if mi < len(self.machine_busy):
                    self.machine_busy[mi] = max(self.machine_busy[mi], finish_t)
            after = sum(self.machine_busy[:n_combos])
            dense_r = -(after - before) / max(1.0, self.lower_bound)

            self.combo_choice[op_idx] = k
            self.finish[op_idx] = finish_t
            self.step_idx += 1

            terminated = self.step_idx >= len(self.order)
            truncated = False
            terminal_r = 0.0
            if terminated:
                makespan = max(self.finish.values()) if self.finish else 0.0
                terminal_r = -float(makespan) / max(1.0, self.lower_bound)
            return self._current_obs(), float(dense_r * 0.1 + terminal_r), terminated, truncated, {}

    return XPURTSchedulingEnv


# ---------------------------------------------------------------------------
# Inference / scheduler entry
# ---------------------------------------------------------------------------


_RL_MODEL_CACHE: Dict[str, Any] = {}


def load_rl_policy(path: Optional[str] = None):
    PPO = _lazy_sb3()
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "models" / "rl_policy_v1.zip")
    if path in _RL_MODEL_CACHE:
        return _RL_MODEL_CACHE[path]
    if not Path(path).exists():
        raise FileNotFoundError(f"RL policy checkpoint not found: {path}")
    model = PPO.load(path)
    _RL_MODEL_CACHE[path] = model
    return model


def rl_policy_scheduler(workload: Workload, *,
                        model_path: Optional[str] = None,
                        fallback_ratio: float = 1.15,
                        **_) -> Tuple[np.ndarray, np.ndarray, None, None]:
    """Run the trained PPO policy to schedule the workload greedily."""
    from scheduler_heft import (
        heft, _build_topo_order, _feasible_combinations,
        _earliest_start_on_combo,
    )

    # HEFT baseline for fallback comparison.
    heft_t, heft_alpha, _, _ = heft(workload)
    combos = workload.get_machine_combinations()
    machines = list(workload.machines)
    n_combos = len(combos)
    n = len(workload.operations)
    if n == 0:
        return heft_t, heft_alpha, None, None

    try:
        EnvCls = make_env_class()
        model = load_rl_policy(model_path)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[rl_policy] {exc} — falling back to HEFT")
        return heft_t, heft_alpha, None, None

    # Single-episode rollout.
    env = EnvCls(workload_provider=lambda: workload)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    # Recover (t, alpha) from env state.
    t_new = np.zeros(n)
    alpha_new = np.zeros((n, n_combos))
    for i in range(n):
        if i in env.combo_choice:
            k = env.combo_choice[i]
            alpha_new[i, k] = 1.0
            t_new[i] = float(env.finish[i] - float(workload.operations[i].get_duration_for_combination(
                k, combos, machines)))
    rl_ms = max(env.finish.values()) if env.finish else 0.0

    # HEFT fallback if RL is too bad.
    heft_ms = 0.0
    for i in range(n):
        k = int(np.argmax(heft_alpha[i]))
        d = float(workload.operations[i].get_duration_for_combination(k, combos, machines))
        f = float(heft_t[i]) + d
        if f > heft_ms:
            heft_ms = f
    if rl_ms > fallback_ratio * heft_ms:
        return heft_t, heft_alpha, None, None

    return t_new, alpha_new, None, None
