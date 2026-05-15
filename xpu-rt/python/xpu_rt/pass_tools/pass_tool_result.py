"""pass_tool_result_v1.

A pass tool emits a typed result carrying *only* a ``recipe_delta``
— a list of Recipe-IR operation dicts. Hard rule 4: pass tools
never mutate Payload IR directly. The verifier downstream
adjudicates the delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

RESULT_SCHEMA_VERSION: Final[str] = "pass_tool_result_v1"

RESULT_STATUSES: Final[tuple[str, ...]] = (
    "proposal",
    "no_op",
    "blocked",
)


class PassToolResultError(ValueError):
    """A PassToolResult body violated the schema."""


@dataclass(frozen=True)
class PassToolResult:
    """Typed result of invoking a pass tool.

    Attributes:
        schema_version: Always ``pass_tool_result_v1``.
        tool_id: Mirrors :attr:`PassToolCard.tool_id`.
        status: One of :data:`RESULT_STATUSES`.
        recipe_delta: List of Recipe-IR op dicts. Must be empty
            when ``status != "proposal"``.
        refinement_claim: Mirrors the tool's declared refinement
            kind (``tolerance_eps`` etc.).
        evidence: Free-form provenance from the pass-tool body
            (matched pattern, single-consumer flag, …).
        detail: Human-readable detail for ``status=blocked``.
    """

    schema_version: str
    tool_id: str
    status: str
    recipe_delta: tuple[dict[str, Any], ...] = ()
    refinement_claim: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise PassToolResultError(
                f"pass tool {self.tool_id!r} status={self.status!r} "
                f"must be one of {RESULT_STATUSES}"
            )
        if self.status != "proposal" and self.recipe_delta:
            raise PassToolResultError(
                f"pass tool {self.tool_id!r} status={self.status!r} "
                f"emits a recipe_delta — only status=proposal may carry "
                f"recipe_delta"
            )
        if self.status == "proposal" and not self.recipe_delta:
            raise PassToolResultError(
                f"pass tool {self.tool_id!r} status=proposal must carry a "
                f"non-empty recipe_delta"
            )
        for op in self.recipe_delta:
            if "op" not in op:
                raise PassToolResultError(
                    f"pass tool {self.tool_id!r} recipe_delta entry missing "
                    f"'op' field: {op!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "status": self.status,
            "recipe_delta": [dict(op) for op in self.recipe_delta],
            "refinement_claim": self.refinement_claim,
            "evidence": dict(self.evidence),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "PassToolResult":
        return cls(
            schema_version=str(body.get("schema_version", RESULT_SCHEMA_VERSION)),
            tool_id=str(body["tool_id"]),
            status=str(body["status"]),
            recipe_delta=tuple(body.get("recipe_delta", ()) or ()),
            refinement_claim=str(body.get("refinement_claim", "")),
            evidence=dict(body.get("evidence", {}) or {}),
            detail=str(body.get("detail", "")),
        )


def make_proposal(
    *,
    tool_id: str,
    recipe_delta: list[dict[str, Any]],
    refinement_claim: str,
    evidence: dict[str, Any] | None = None,
) -> PassToolResult:
    """Build a status=proposal result with the schema baked in."""

    return PassToolResult(
        schema_version=RESULT_SCHEMA_VERSION,
        tool_id=tool_id,
        status="proposal",
        recipe_delta=tuple(recipe_delta),
        refinement_claim=refinement_claim,
        evidence=evidence or {},
    )


def make_no_op(
    *,
    tool_id: str,
    refinement_claim: str = "",
    detail: str = "",
) -> PassToolResult:
    return PassToolResult(
        schema_version=RESULT_SCHEMA_VERSION,
        tool_id=tool_id,
        status="no_op",
        refinement_claim=refinement_claim,
        detail=detail,
    )


def make_blocked(
    *,
    tool_id: str,
    detail: str,
) -> PassToolResult:
    return PassToolResult(
        schema_version=RESULT_SCHEMA_VERSION,
        tool_id=tool_id,
        status="blocked",
        detail=detail,
    )
