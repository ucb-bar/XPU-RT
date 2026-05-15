"""Agent IR -- deliberation, evidence, and learning around Recipe IR.

This dialect family represents agent state as compiler IR without granting
the agent authority over correctness. Recipe IR remains the bounded
deployment-decision contract; Agent IR captures intent, admissible evidence,
generation requests, claims, frontier state, critique, memory, and roles.
"""

from __future__ import annotations

from compgen.ir.agent.dialect import Agent

__all__ = ["Agent"]
