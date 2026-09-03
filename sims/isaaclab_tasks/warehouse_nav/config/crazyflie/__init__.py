# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warehouse navigation task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Drone-Warehouse-Nav-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Warehouse-Nav-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg_PLAY_WithSensors",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg_PLAY_WithSensors_Coll",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

# THE complex collidable course (gates + racks + tall-thin stacked props + people, all colliders).
gym.register(
    id="Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-Crowded-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg_PLAY_WithSensors_Coll_Crowded",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Warehouse-Nav-Crazyflie-FPV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseNavEnvCfg_FPV",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Warehouse-Grand-Locomotion-FPV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:WarehouseGrandLocoEnvCfg_FPV",
        "rsl_rl_cfg_entry_point": "sims.isaaclab_tasks.track_steering_vision.config.crazyflie.agents.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

for _cid, _cls in [
    ("Warehouse-Grand", "WarehouseGrandEnvCfg_FPV"),
    ("Aisle-Slalom", "AisleSlalomEnvCfg_FPV"),
    ("Cross-Shelf", "CrossShelfEnvCfg_FPV"),
    ("Vertical-Gates", "VerticalGatesEnvCfg_FPV"),
    ("Dynamic-Dodge", "DynamicDodgeEnvCfg_FPV"),
    ("Mixed-Clutter", "MixedClutterEnvCfg_FPV"),
    ("Warehouse-Grand-Rich", "WarehouseGrandRichEnvCfg"),
    ("Aisle-Slalom-Rich", "AisleSlalomRichEnvCfg"),
    ("Cross-Shelf-Rich", "CrossShelfRichEnvCfg"),
    ("Vertical-Gates-Rich", "VerticalGatesRichEnvCfg"),
    ("Dynamic-Dodge-Rich", "DynamicDodgeRichEnvCfg"),
    ("Mixed-Clutter-Rich", "MixedClutterRichEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Drone-{_cid}-Crazyflie-FPV-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.warehouse_nav_env_cfg:{_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Drone-Dense-Crossing-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dense_crossing_env_cfg:DenseCrossingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Dense-Crossing-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dense_crossing_env_cfg:DenseCrossingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

# Crowd navigation: cross an open loading hall full of mutually-avoiding walking people at a
# fixed altitude. Reuses the warehouse-nav PPO runner cfg (same obs/action dims family).
gym.register(
    id="Isaac-Drone-Crowd-Nav-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crowd_nav_env_cfg:CrowdNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Drone-Crowd-Nav-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crowd_nav_env_cfg:CrowdNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)
