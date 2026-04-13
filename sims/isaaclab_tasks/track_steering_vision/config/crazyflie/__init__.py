# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crazyflie steering tracking task configurations."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg:TrackSteeringEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg:TrackSteeringEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# Harsh reward variants
gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Harsh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg_harsh:TrackSteeringEnvCfg_Harsh",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Harsh-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg_harsh:TrackSteeringEnvCfg_Harsh_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)
