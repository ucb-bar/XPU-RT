"""Two-tier ``contract_feedback`` re-entry into Recipe IR.

The Phase D auction collects ``ProviderResult.contract_feedback``
from every fulfilled bid. Each entry is the provider proposing a
contract refinement: "row-major B is 1.4× faster on sm_90 for K ≥ 64",
"f16 inputs with f32 accumulator hits Tensor Cores", etc. routes
these entries into the compiler in two tiers:

1. **Typed allowlist** auto-applies. The kinds are a small bounded set:

   * ``layout_swap`` — change an IO tensor's layout
   * ``dtype_widen`` — widen an IO tensor's dtype
   * ``accumulator_widen`` — widen the accumulator (e.g. f32 → f64)
   * ``alignment_request`` — request stricter IO alignment
   * ``fast_math_opt_in`` — opt the contract into fast-math

   For each allowlisted entry the generator emits a structured
   Recipe-IR proposal dict at
   ``04_kernel_codegen/contract_feedback_proposals.json`` which a
   subsequent iteration of the agent_decision request emit can
   surface as a new candidate (the action-space machinery's Family 7
   in the milestone plan; threading the proposal through to action
   generation is the next milestone's plumbing job).

2. **Non-allowlisted** entries (anything else) are persisted as
   advisory data at
   ``04_kernel_codegen/auction/<task_id>/contract_feedback.json::non_allowlisted``
   and can later be surfaced in ``agent_decision_request.advisory`` so
   the outer agent decides what to do. lands the data layer; the
   agent_decision_request advisory wiring is conservative — emitted
   when the file is present, ignored otherwise.

The classifier never silently drops entries. Every feedback entry the
auction collects ends up in one of the two buckets with a typed
``kind`` field set (inferred when the provider didn't supply one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from compgen.kernels.provider import ContractFeedback

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Typed allowlist
# --------------------------------------------------------------------------- #


_TYPED_ALLOWLIST: frozenset[str] = frozenset(
    {
        "layout_swap",
        "dtype_widen",
        "accumulator_widen",
        "alignment_request",
        "fast_math_opt_in",
    }
)


def _infer_kind(feedback: ContractFeedback) -> str:
    """Heuristic kind derivation from ``field`` when the provider didn't
    supply a typed ``kind``.

    Only used when ``feedback.kind`` is empty. Maps:

    - ``field`` containing ``"layout"`` → ``layout_swap``
    - ``field`` containing ``"dtype"`` (and not ``accumulator``) → ``dtype_widen``
    - ``field`` containing ``"accumulator"`` → ``accumulator_widen``
    - ``field`` containing ``"align"`` → ``alignment_request``
    - ``field`` containing ``"fast_math"`` → ``fast_math_opt_in``
    - anything else → empty string (non-allowlisted)
    """
    if feedback.kind:
        return feedback.kind
    f = (feedback.field or "").lower()
    if not f:
        return ""
    if "accumulator" in f:
        return "accumulator_widen"
    if "layout" in f:
        return "layout_swap"
    if "dtype" in f:
        return "dtype_widen"
    if "align" in f:
        return "alignment_request"
    if "fast_math" in f or "fast-math" in f:
        return "fast_math_opt_in"
    return ""


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassifiedFeedback:
    """A ``ContractFeedback`` entry tagged with its inferred + classified ``kind``.

    ``provider_name`` records which auction bidder emitted it so the
    advisory + retry trail stays auditable.
    """

    provider_name: str
    kind: str  # always non-empty in the allowlisted bucket
    is_allowlisted: bool
    feedback: ContractFeedback

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "kind": self.kind,
            "is_allowlisted": self.is_allowlisted,
            "feedback": self.feedback.to_dict(),
        }


def classify_feedback(
    *,
    provider_name: str,
    feedbacks: list[ContractFeedback],
) -> tuple[list[ClassifiedFeedback], list[ClassifiedFeedback]]:
    """Split a provider's feedback list into (allowlisted, non_allowlisted).

    For every entry with empty ``kind``, infer it via :func:`_infer_kind`.
    Allowlisted entries have a ``kind`` in :data:`_TYPED_ALLOWLIST`;
    everything else (including entries whose kind couldn't be inferred)
    lands in the non-allowlisted bucket with ``kind`` either as the
    provider-supplied value or the empty string.
    """
    allowlisted: list[ClassifiedFeedback] = []
    non_allowlisted: list[ClassifiedFeedback] = []
    for fb in feedbacks:
        kind = _infer_kind(fb)
        is_allow = kind in _TYPED_ALLOWLIST
        rec = ClassifiedFeedback(
            provider_name=provider_name,
            kind=kind,
            is_allowlisted=is_allow,
            feedback=fb,
        )
        (allowlisted if is_allow else non_allowlisted).append(rec)
    return allowlisted, non_allowlisted


# --------------------------------------------------------------------------- #
# Recipe-IR proposal generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeedbackProposal:
    """A structured Recipe-IR proposal generated from one allowlisted entry.

    The ``op`` field names the Recipe-IR op the next iteration would
    instantiate (e.g. ``SetLayout``, ``WidenDtype``,
    ``WidenAccumulator``, ``SetAlignment``, ``EnableFastMath``). The
    ``args`` field carries the parameters. The proposal is structured
    JSON; full materialisation as a Recipe-IR ``Op`` happens in the
    next iteration of the agent-decision loop, when the action space
    re-emits with feedback-driven candidates.
    """

    op: str
    args: dict[str, Any]
    rationale: str
    applies_when: str
    source_provider: str
    source_kind: str
    measured_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": dict(self.args),
            "rationale": self.rationale,
            "applies_when": self.applies_when,
            "source_provider": self.source_provider,
            "source_kind": self.source_kind,
            "measured_gain": self.measured_gain,
        }


_KIND_TO_RECIPE_OP: dict[str, str] = {
    "layout_swap": "SetLayout",
    "dtype_widen": "WidenDtype",
    "accumulator_widen": "WidenAccumulator",
    "alignment_request": "SetAlignment",
    "fast_math_opt_in": "EnableFastMath",
}


def to_recipe_ir_proposal(entry: ClassifiedFeedback) -> FeedbackProposal:
    """Translate one allowlisted ``ClassifiedFeedback`` into a typed
    Recipe-IR proposal dict.

    The ``args`` shape depends on the ``kind``:

    * ``layout_swap``: ``{"target_field": <field>, "new_layout": <suggested>}``
    * ``dtype_widen``: ``{"target_field": <field>, "new_dtype": <suggested>}``
    * ``accumulator_widen``: ``{"new_accumulator_dtype": <suggested>}``
    * ``alignment_request``: ``{"target_field": <field>, "new_alignment_bytes": <int>}``
    * ``fast_math_opt_in``: ``{"enable": True}``
    """
    if not entry.is_allowlisted:
        raise ValueError(
            f"to_recipe_ir_proposal called on non-allowlisted entry; "
            f"kind={entry.kind!r}"
        )
    op = _KIND_TO_RECIPE_OP[entry.kind]
    fb = entry.feedback
    args: dict[str, Any]
    if entry.kind == "layout_swap":
        args = {"target_field": fb.field, "new_layout": fb.suggested_value}
    elif entry.kind == "dtype_widen":
        args = {"target_field": fb.field, "new_dtype": fb.suggested_value}
    elif entry.kind == "accumulator_widen":
        args = {"new_accumulator_dtype": fb.suggested_value}
    elif entry.kind == "alignment_request":
        try:
            new_align = int(fb.suggested_value)
        except (TypeError, ValueError):
            new_align = 0
        args = {"target_field": fb.field, "new_alignment_bytes": new_align}
    elif entry.kind == "fast_math_opt_in":
        args = {"enable": True}
    else:
        # Defensive — _KIND_TO_RECIPE_OP and _TYPED_ALLOWLIST should match.
        args = {}
    return FeedbackProposal(
        op=op,
        args=args,
        rationale=fb.reason,
        applies_when=fb.applies_when,
        source_provider=entry.provider_name,
        source_kind=entry.kind,
        measured_gain=fb.measured_gain,
    )


# --------------------------------------------------------------------------- #
# Persistence — auction-side
# --------------------------------------------------------------------------- #


_FEEDBACK_SCHEMA = "auction_contract_feedback_v1"
_PROPOSALS_SCHEMA = "contract_feedback_proposals_v1"


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_auction_feedback_artifacts(
    *,
    run_dir: Path,
    task_id: str,
    contract_hash: str,
    per_provider_feedback: list[tuple[str, list[ContractFeedback]]],
) -> dict[str, Path]:
    """Write the auction's two feedback artifacts:

    1. ``04_kernel_codegen/auction/<task_id>/contract_feedback.json``
       with both buckets (allowlisted, non_allowlisted) plus the
       structured Recipe-IR proposals.
    2. ``04_kernel_codegen/contract_feedback_proposals.json`` —
       run-wide aggregate the next iteration's action-space generator
       (Family 7) will read.

    Returns a dict of the written paths so the caller can record them
    in the auction report.
    """
    auction_dir = run_dir / "04_kernel_codegen" / "auction" / task_id
    auction_dir.mkdir(parents=True, exist_ok=True)

    all_allowlisted: list[ClassifiedFeedback] = []
    all_non_allowlisted: list[ClassifiedFeedback] = []
    for provider_name, feedbacks in per_provider_feedback:
        a, na = classify_feedback(provider_name=provider_name, feedbacks=feedbacks)
        all_allowlisted.extend(a)
        all_non_allowlisted.extend(na)

    proposals = [to_recipe_ir_proposal(e).to_dict() for e in all_allowlisted]

    auction_body = {
        "schema_version": _FEEDBACK_SCHEMA,
        "generated_at_utc": _utcnow(),
        "task_id": task_id,
        "contract_hash": contract_hash,
        "allowlisted": [e.to_dict() for e in all_allowlisted],
        "non_allowlisted": [e.to_dict() for e in all_non_allowlisted],
        "proposals": proposals,
        "counts": {
            "total": len(all_allowlisted) + len(all_non_allowlisted),
            "allowlisted": len(all_allowlisted),
            "non_allowlisted": len(all_non_allowlisted),
            "proposals": len(proposals),
        },
    }
    auction_path = auction_dir / "contract_feedback.json"
    auction_path.write_text(
        json.dumps(auction_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Run-wide aggregate. Subsequent auction tasks may also write
    # feedback; we maintain an append-only proposals list keyed by
    # contract_hash so the next iteration sees them all in one shot.
    aggregate_path = run_dir / "04_kernel_codegen" / "contract_feedback_proposals.json"
    if aggregate_path.exists():
        try:
            agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            agg = {"schema_version": _PROPOSALS_SCHEMA, "entries": []}
    else:
        agg = {"schema_version": _PROPOSALS_SCHEMA, "entries": []}

    # Replace any prior entry for the same task (idempotent re-run).
    agg["entries"] = [
        e for e in agg.get("entries", []) if e.get("task_id") != task_id
    ]
    agg["entries"].append(
        {
            "task_id": task_id,
            "contract_hash": contract_hash,
            "generated_at_utc": _utcnow(),
            "proposals": proposals,
            "non_allowlisted_advisory": [
                e.to_dict() for e in all_non_allowlisted
            ],
        }
    )
    agg["schema_version"] = _PROPOSALS_SCHEMA
    aggregate_path.write_text(
        json.dumps(agg, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log.info(
        "m59.feedback_written",
        task_id=task_id,
        allowlisted=len(all_allowlisted),
        non_allowlisted=len(all_non_allowlisted),
        proposals=len(proposals),
    )
    return {
        "auction_path": auction_path,
        "aggregate_path": aggregate_path,
    }


__all__ = [
    "ClassifiedFeedback",
    "FeedbackProposal",
    "classify_feedback",
    "to_recipe_ir_proposal",
    "write_auction_feedback_artifacts",
]
