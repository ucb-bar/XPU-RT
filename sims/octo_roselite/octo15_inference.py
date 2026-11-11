"""Octo-1.5-compatible inference wrapper for SimplerEnv.

SimplerEnv's stock simpler_env/policies/octo/octo_model.py targets octo-1.0
(commit 653c54ac). Two incompatibilities with the octo-1.5 code/checkpoint:

  1. observation key renamed  `pad_mask` -> `timestep_pad_mask`.
  2. unnormalization. octo-1.5's bridge_dataset action statistics carry
     mask = [T,T,T,T,T,T,False]: the gripper dim is *not* normalized, so the
     model emits it raw in [0,1]. The stock wrapper applies `a*std + mean` to
     all 7 dims. With bridge std[6]=0.4877, mean[6]=0.5879 that maps a
     commanded-close 0.0 -> 0.588 and a commanded-open 1.0 -> 1.076; both are
     > 0.5, so the binarizer `2*(g>0.5)-1` returns +1 (OPEN) forever and the
     gripper can never close.  We instead let octo do the masked
     unnormalization via `unnormalization_statistics=`.

Everything else (image history/pad mask, action ensembling, euler2axangle,
sticky gripper) follows the stock wrapper.
"""
from collections import deque
from typing import Optional
import os

import jax
import numpy as np
import tensorflow as tf
from transforms3d.euler import euler2axangle

from simpler_env.utils.action.action_ensemble import ActionEnsembler


class Octo15Inference:
    def __init__(
        self,
        model,
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        exec_horizon: int = 1,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        legacy_unnorm: bool = False,   # True == reproduce the stock octo-1.0 wrapper bug
        action_ensemble: bool = True,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            self.dataset_id = "bridge_dataset"
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            self.dataset_id = "fractal20220817_data"
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(policy_setup)
        self.policy_setup = policy_setup
        self.model = model
        self.legacy_unnorm = legacy_unnorm
        self.stats = self.model.dataset_statistics[self.dataset_id]["action"]
        self.action_mean = np.asarray(self.stats["mean"])
        self.action_std = np.asarray(self.stats["std"])

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            self.rng, _key = jax.random.split(self.rng)

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.action_ensembler = (
            ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
            if self.action_ensemble else None
        )
        self.num_image_history = 0

    def _resize_image(self, image):
        image = tf.image.resize(image, size=(self.image_size, self.image_size),
                                method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()

    def _add_image_to_history(self, image):
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self):
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=bool)
        pad_mask[: horizon - min(horizon, self.num_image_history)] = False
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, image, task_description: Optional[str] = None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        self.rng, key = jax.random.split(self.rng)
        obs = {"image_primary": images, "timestep_pad_mask": pad_mask}

        if self.legacy_unnorm:
            norm = np.asarray(self.model.sample_actions(obs, self.task, rng=key))
            raw_actions = norm * self.action_std[None] + self.action_mean[None]
        else:
            raw_actions = np.asarray(self.model.sample_actions(
                obs, self.task, unnormalization_statistics=self.stats, rng=key))
        raw_actions = raw_actions[0]

        assert raw_actions.shape == (self.pred_action_horizon, 7), raw_actions.shape
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        ax, ang = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = ax * ang * self.action_scale

        if self.policy_setup == "google_robot":
            cur = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                rel = np.array([0])
            else:
                rel = self.previous_gripper_action - cur
            self.previous_gripper_action = cur
            if np.abs(rel) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        else:  # widowx_bridge
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
