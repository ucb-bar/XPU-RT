"""P3.3 — rank_candidates primitive.

Reorders a precomputed list of :class:`compgen.agent.suggest._candidate.ProposalCandidate`
by likely benefit. The contract:

* The output is always a *permutation* of the input — the LLM cannot
  invent a candidate not in the list. The decorator's
  ``output_schema`` enforces shape; the test
  ``test_output_is_a_permutation_of_input`` enforces the set-equality
  invariant on every input.
* The deterministic fallback sorts by ``expected_impact`` descending,
  then by ``rationale`` (alphabetical) as a stable tiebreak. This is
  the no-LLM baseline P4 ablations measure against.
* The primary path (LLM-driven) is currently a *stub* that delegates
  to the fallback — wire-in for a live LLM provider lands when the
  P2 Tactician/Strategist loop is plumbed (a fresh-agent task will
  prove the LLM path produces a better ranking on a real workload).
  The stub is honest: it does not pretend to be an LLM-ranked output.

The primitive is exposed to MCP / fresh-agent harness through the
P3.0 registry (``compgen.llm.call_site.list_call_sites``) and via the
shipped ToolCard ``compgen_rank_candidates`` (lands when the
bridge picks up call-site cards in a follow-up — for now the
primitive is callable directly through its Python API).
"""

from __future__ import annotations

from typing import Any

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.llm.call_site import llm_call_site, register_fallback

# JSON-schema for the wrapper's output. Each rank entry carries
# ``index`` (offset into the input list, 0-based) and ``score``
# (re-stated confidence in [0, 1]) so the consumer can both follow
# the permutation and surface a confidence summary in the agent's
# decision log.
RANK_CANDIDATES_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ranking", "fallback_used"],
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "score"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "fallback_used": {"type": "boolean"},
        "n_candidates": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}


@register_fallback("rank_candidates_deterministic")
def _rank_candidates_fallback(candidates: list[ProposalCandidate]) -> dict[str, Any]:
    """Deterministic fallback: descending ``expected_impact``, then
    alphabetical ``rationale`` as a stable tiebreak."""

    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-pair[1].expected_impact, pair[1].rationale))
    return {
        "ranking": [
            {
                "index": idx,
                "score": float(cand.expected_impact),
                "rationale": cand.rationale,
            }
            for idx, cand in indexed
        ],
        "fallback_used": True,
        "n_candidates": len(candidates),
    }


@llm_call_site(
    site_id="rank_candidates",
    leverage="Rank a precomputed candidate set by likely benefit. "
    "The LLM never invents a new candidate; it only re-orders the input.",
    inputs=["list[ProposalCandidate]"],
    output_schema=RANK_CANDIDATES_OUTPUT_SCHEMA,
    forbidden=[
        "invent_candidate_not_in_input_list",
        "invent_numerical_threshold",
        "be_sole_correctness_decider",
    ],
    fallback="rank_candidates_deterministic",
    description=(
        "P3.3 — highest-leverage primitive per the agent-loop plan. "
        "Collapses combinatorial explorations from O(N!) to O(k) by "
        "asking the LLM to rank the precomputed candidate set the "
        "Tactician's candidate-generator produced."
    ),
)
def rank_candidates(candidates: list[ProposalCandidate]) -> dict[str, Any]:
    """LLM-ranked candidate ordering.

    The primary path currently delegates to the deterministic
    fallback; a live-LLM implementation lands when the Tactician
    loop is wired (P2.5). The primitive is callable today and its
    output satisfies the contract — but the ``fallback_used`` flag
    surfaces honestly that no LLM ranked the list.
    """

    # Honest delegation while the LLM wiring lands in P2.5; the
    # decorator still re-validates the output_schema so the contract
    # holds whether the path is "primary" or "fallback".
    return _rank_candidates_fallback(candidates)


__all__ = ["RANK_CANDIDATES_OUTPUT_SCHEMA", "rank_candidates"]
