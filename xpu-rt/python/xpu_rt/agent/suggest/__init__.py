"""Model-aware proposal suggesters for every registered invent slot.

When the agent doesn't know what to propose, it calls ``suggest_proposals``
which dispatches into one suggester per slot. Each suggester walks the
session's recipe + dossier + target profile and returns a small ranked
list of pre-built proposal candidates the agent can submit verbatim
via ``propose_invent_slot``.

The point: an agent should NEVER need to invent a proposal payload
from scratch — it gets concrete candidates with rationales, then
picks one (or batches several via ``batch_propose``).

Design contract per suggester:

    fn(recipe: ModuleOp, dossier, target, *, k: int) -> list[ProposalCandidate]

Returns at most ``k`` candidates, sorted descending by score. Each
candidate carries the raw ``chosen`` payload + a human-readable
``rationale`` + a coarse ``expected_impact`` score in [0, 1].
"""

from __future__ import annotations

# Auto-import per-slot modules so they register on package import.
from compgen.agent.suggest import (  # noqa: F401, E402
    suggest_buffer_lifetime,
    suggest_dequant_fusion,
    suggest_fusion,
    suggest_layout_plan,
    suggest_megakernel,
    suggest_numerics_plan,
    suggest_peephole_pattern,
    suggest_rematerialization,
    suggest_scheduling_policy,
)
from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import (
    SUGGESTERS,
    register_suggester,
    suggest,
    supported_slot_names,
)

__all__ = [
    "ProposalCandidate",
    "SUGGESTERS",
    "register_suggester",
    "suggest",
    "supported_slot_names",
]
