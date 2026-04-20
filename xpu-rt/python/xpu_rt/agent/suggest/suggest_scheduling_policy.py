"""Suggester for ``propose_scheduling_policy``.

Static for the common case (no data-dependent edges); dynamic when the
dossier reports any branching / control-flow heavy regions.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester


@register_suggester("propose_scheduling_policy")
def suggest_scheduling_policy(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    has_dynamic = bool(getattr(dossier, "graph_break_count", 0))
    if has_dynamic:
        return [ProposalCandidate(
            chosen={
                "megakernel_ref": "agent_megakernel",
                "policy": "dynamic", "early_push": True,
            },
            rationale="data-dependent edges detected — dynamic on-chip scheduler",
            expected_impact=0.7,
            target_feature_justification="dossier reports graph breaks",
            metadata={"reason": "graph_breaks"},
        )]
    return [ProposalCandidate(
        chosen={
            "megakernel_ref": "agent_megakernel",
            "policy": "static",
            "sm_count": 32,
        },
        rationale="no data-dependent edges — static per-SM queue",
        expected_impact=0.6,
        target_feature_justification="dossier shows static control flow",
        metadata={"reason": "static"},
    )]


__all__ = ["suggest_scheduling_policy"]
