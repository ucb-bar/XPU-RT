# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forest-trail testbed for evaluating vision-based steering policies.

The task is intentionally a thin extension of ``track_steering_vision``: the
robot, action space, observation space, inner-loop velocity-tracker policy,
and most rewards are inherited unchanged. Only the scene differs (a straight
trail through procedurally placed cylinder "trees") and one extra termination
fires when the drone leaves the trail corridor.

Future iterations will add curved trails (set of trail-shape configs) and
swap the cylinder placeholders for proper USD tree assets.
"""
