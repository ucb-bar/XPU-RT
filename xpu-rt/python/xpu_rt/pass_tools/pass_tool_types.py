"""PassToolCard schema.

Carries the agent-facing pass-tool contract: which IR levels it
reads, what Recipe-IR ops it may emit, the refinement kind, and the
verifier that adjudicates its output. wraps every payload-IR
pass behind one of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PASS_PHASES: Final[tuple[str, ...]] = (
    "recipe_authoring",
    "recipe_lowering",
    "payload_canonicalization",
    "kernel_codegen",
    "execution_plan",
    "runtime_emission",
)

REFINEMENT_KINDS: Final[tuple[str, ...]] = (
    "tolerance_eps",
    "structural_equivalence",
    "differential_then_z3_if_promoted",
    "z3_proof",
    "none",
)


class PassToolCardError(ValueError):
    """A PassToolCard YAML body violated the schema."""


@dataclass(frozen=True)
class PassToolCard:
    schema_version: str
    tool_id: str
    phase: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    allowed_recipe_ops: tuple[str, ...]
    refinement_kind: str
    verifier: str
    entrypoint: str
    description: str = ""

    @classmethod
    def from_dict(
        cls, body: dict[str, Any], *, source: Path | None = None
    ) -> "PassToolCard":
        try:
            schema_version = str(body["schema_version"])
            tool_id = str(body["tool_id"])
            phase = str(body["phase"])
            entrypoint = str(body["entrypoint"])
        except KeyError as exc:
            raise PassToolCardError(
                f"pass-tool card missing required field {exc.args[0]!r} "
                f"(source={source})"
            ) from exc
        if phase not in PASS_PHASES:
            raise PassToolCardError(
                f"pass-tool {tool_id!r} phase={phase!r} must be one of "
                f"{PASS_PHASES} (source={source})"
            )
        refinement = body.get("refinement", {}) or {}
        refinement_kind = str(refinement.get("kind", "none"))
        if refinement_kind not in REFINEMENT_KINDS:
            raise PassToolCardError(
                f"pass-tool {tool_id!r} refinement.kind={refinement_kind!r} must be "
                f"one of {REFINEMENT_KINDS} (source={source})"
            )
        verifier = str(refinement.get("verifier", ""))
        writes = tuple(body.get("writes", ()))
        if "payload_ir" in writes:
            raise PassToolCardError(
                f"pass-tool {tool_id!r} declares writes=[payload_ir] — pass tools "
                f"must only emit recipe_delta, never mutate payload IR directly "
                f"(source={source})"
            )
        return cls(
            schema_version=schema_version,
            tool_id=tool_id,
            phase=phase,
            reads=tuple(body.get("reads", ())),
            writes=writes,
            allowed_recipe_ops=tuple(body.get("allowed_recipe_ops", ())),
            refinement_kind=refinement_kind,
            verifier=verifier,
            entrypoint=entrypoint,
            description=str(body.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "phase": self.phase,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "allowed_recipe_ops": list(self.allowed_recipe_ops),
            "refinement": {
                "kind": self.refinement_kind,
                "verifier": self.verifier,
            },
            "entrypoint": self.entrypoint,
            "description": self.description,
        }
