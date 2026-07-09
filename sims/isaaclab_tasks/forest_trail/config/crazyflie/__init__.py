# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crazyflie forest-trail task registrations."""

import gymnasium as gym

# Reuse the agent config from track_steering_vision so the same trained
# velocity-tracker checkpoint can drive this env unchanged.
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie import agents


# ── Straight trail ────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Forest-Trail-Vision-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Forest-Trail-Vision-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Forest-Trail-Vision-Crazyflie-Play-WithHumans-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg_PLAY_WithHumans",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# ── Curved trail ──────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Forest-Trail-Curved-Vision-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg_Curved",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg_Curved_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Forest-Trail-Curved-Vision-Crazyflie-Play-WithHumans-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forest_env_cfg:ForestTrailEnvCfg_Curved_PLAY_WithHumans",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)
