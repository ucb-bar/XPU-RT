"""Policies for :class:`compgen.agent.loop.PhasedDriveLoop`.

A *policy* is a callable matching the signature

    (phase: int, registry: Registry, context: dict) ->
        list[tuple[str, dict]]      # (name, kwargs) per step

Given a phase and the registry, it returns the ordered steps the drive
loop should execute. The drive loop resolves each step against tools
and invent-slots.

Policies live here so that both real LLM-backed policies and
deterministic defaults are swappable via
``drive_loop.context["policy"] = my_policy``.
"""

from __future__ import annotations

from compgen.agent.policies.default import DeterministicDefaultPolicy

__all__ = ["DeterministicDefaultPolicy"]
