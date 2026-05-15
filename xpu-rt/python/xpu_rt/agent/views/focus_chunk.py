"""focus_chunk — lazy per-region expansion (P2.1).

When :func:`compgen.agent.views.canonical_view.canonical_view` shows a
region the LLM wants to drill into, the Tactician calls
``focus_chunk`` with the region id to get its full dossier (op list,
candidate set, contract envelope, recent decisions).

The chunk is *not* size-bounded — it is the unit of expansion. The
caller is responsible for budgeting (the Tactician focuses one region
at a time, so one chunk fits the prompt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FocusChunk:
    """All the per-region detail the Tactician needs.

    The shape mirrors :class:`compgen.graph_compilation.graph_dossier`
    just enough to make the chunk consumable without coupling to the
    full dossier class.
    """

    region_id: str
    ops: tuple[str, ...]
    candidate_set: tuple[dict[str, Any], ...]
    contract_envelope: dict[str, Any]
    recent_decisions: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "ops": list(self.ops),
            "candidate_set": [dict(c) for c in self.candidate_set],
            "contract_envelope": dict(self.contract_envelope),
            "recent_decisions": [dict(d) for d in self.recent_decisions],
        }


class FocusChunkNotFoundError(ValueError):
    """The requested region id is not present in the session state."""


def focus_chunk(session_state: dict[str, Any], *, region_id: str) -> FocusChunk:
    """Extract one region's full dossier from a session-state dict.

    Raises :class:`FocusChunkNotFoundError` if no region with the
    given id exists — silently returning an empty chunk would let
    a downstream LLM call see "no candidates" when in fact the
    caller asked for a region that does not exist.
    """

    for region in session_state.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        if str(region.get("region_id")) != region_id:
            continue
        return FocusChunk(
            region_id=region_id,
            ops=tuple(str(o) for o in region.get("ops", []) or []),
            candidate_set=tuple(
                dict(c) if isinstance(c, dict) else {"value": c}
                for c in region.get("candidate_set", []) or []
            ),
            contract_envelope=dict(region.get("contract_envelope") or {}),
            recent_decisions=tuple(
                dict(d) if isinstance(d, dict) else {}
                for d in region.get("recent_decisions", []) or []
            ),
        )
    raise FocusChunkNotFoundError(
        f"region_id={region_id!r} not found in session state"
    )


__all__ = ["FocusChunk", "FocusChunkNotFoundError", "focus_chunk"]
