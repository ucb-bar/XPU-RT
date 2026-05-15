"""Shared candidate dataclass used by every suggester."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProposalCandidate:
    """One pre-built proposal an agent can submit verbatim.

    Attributes:
        chosen: The ``chosen`` block ready to drop into the
            ``propose_invent_slot.proposal['chosen']`` payload. Schema
            depends on the slot.
        rationale: One-sentence explanation the agent can read.
        expected_impact: Coarse benefit estimate in [0, 1] used to
            sort candidates. Higher = more confident this candidate
            improves the bundle.
        target_feature_justification: Free-form string the suggester
            fills with the hardware feature it's leaning on (e.g.
            "Hexagon HVX 32x32 tile alignment"). Becomes the
            ``target_feature_justification`` field of the proposal.
        members: For dedup'd candidates that represent N structurally-
            equivalent matches (e.g. all 3 RMSNorm rsqrt→mul pairs),
            this carries the per-instance ``chosen`` blocks without
            eating the agent's ``k`` budget. ``chosen`` represents
            the FIRST instance; the rest live here.
        slot_name: Set by the dispatcher so ``next_call`` is self-
            contained.
        metadata: Suggester-specific extra detail. Not used by gates.
    """

    chosen: dict[str, Any]
    rationale: str = ""
    expected_impact: float = 0.5
    target_feature_justification: str = ""
    members: list[dict[str, Any]] = field(default_factory=list)
    slot_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_proposal(self) -> dict[str, Any]:
        """Wrap this candidate in the propose_invent_slot payload shape."""
        return {
            "chosen": dict(self.chosen),
            "candidates": [{"chosen": dict(self.chosen), "rationale": self.rationale, "score": self.expected_impact}],
            "target_feature_justification": self.target_feature_justification,
            "select_vs_invent": "invent",
        }

    def next_call(self) -> dict[str, Any]:
        """Self-describing hint: which MCP tool the agent should call
        next to apply this candidate, with the args already filled in.

        Single instance → ``propose_invent_slot``. When ``members``
        carries multiple equivalent instances → ``batch_propose`` so
        the agent applies all of them in one tool call.
        """
        if not self.members or len(self.members) <= 1:
            return {
                "tool": "propose_invent_slot",
                "args": {
                    "slot_name": self.slot_name,
                    "proposal": self.to_proposal(),
                },
            }
        # Multi-member: build a batch with one proposal per member.
        proposals = []
        for m in self.members:
            chosen = dict(m.get("chosen") or self.chosen)
            proposals.append(
                {
                    "slot_name": self.slot_name,
                    "proposal": {
                        "chosen": chosen,
                        "candidates": [],
                        "target_feature_justification": self.target_feature_justification,
                        "select_vs_invent": "invent",
                    },
                }
            )
        return {
            "tool": "batch_propose",
            "args": {"proposals": proposals, "atomic": False},
        }


__all__ = ["ProposalCandidate"]
