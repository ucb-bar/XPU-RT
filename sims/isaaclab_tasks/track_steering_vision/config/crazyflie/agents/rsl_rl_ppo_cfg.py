# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for steering angle tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg,
)


@configclass
class SteeringTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for steering tracking task."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "crazyflie_steering_tracking"

    # Map observation groups to actor/critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # Actor network (policy)
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic network (value function)
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    # PPO Algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
