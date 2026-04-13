# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for steering angle tracking."""

from __future__ import annotations

from dataclasses import MISSING

import torch
from collections.abc import Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class SteeringCommand(CommandTerm):
    """Command term for steering angle (yaw rate) and forward velocity.

    The command is a 2D vector: [target_yaw_rate, target_forward_velocity]
    """

    cfg: SteeringCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: SteeringCommandCfg, env):
        """Initialize the command term.

        Args:
            cfg: Configuration for the command term.
            env: The environment.
        """
        super().__init__(cfg, env)

        # Create buffers
        self.target_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self.target_velocity = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """String representation."""
        msg = "SteeringCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape)}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, 2)."""
        return torch.stack([self.target_yaw_rate, self.target_velocity], dim=1)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command for specified environments.

        Args:
            env_ids: List of environment IDs to resample.
        """
        # Sample random yaw rates
        self.target_yaw_rate[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.ranges.yaw_rate[0], self.cfg.ranges.yaw_rate[1]
        )

        # Sample random forward velocities
        self.target_velocity[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.ranges.velocity[0], self.cfg.ranges.velocity[1]
        )

    def _update_command(self):
        """Update the command (no-op for this simple command)."""
        pass

    def _update_metrics(self):
        """Update metrics for logging (no-op for this simple command)."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization.

        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # No visualization for now
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        # No visualization for now
        pass


@configclass
class SteeringCommandCfg(CommandTermCfg):
    """Configuration for steering command term."""

    class_type: type = SteeringCommand

    @configclass
    class Ranges:
        """Ranges for the steering command."""

        yaw_rate: tuple[float, float] = MISSING
        """Range for target yaw rate (rad/s). Default is MISSING."""

        velocity: tuple[float, float] = MISSING
        """Range for target forward velocity (m/s). Default is MISSING."""

    ranges: Ranges = Ranges()
    """Ranges for the command."""

    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    """Time range for resampling commands (s). Default is (5.0, 5.0)."""
