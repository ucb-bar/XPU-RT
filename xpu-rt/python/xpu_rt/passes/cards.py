"""Pass-card schema + registry.

A pass card is the canonical typed description of one compiler pass
the agent can request. The registry loads every card under a root
directory, validates them, and answers ``does this pass_id resolve``
queries from the agent-decision validator.

Schema (``pass_card_v1`` YAML)::

    schema_version: pass_card_v1
    pass_id: set_tile_params
    display_name: Set tile parameters
    level: payload
    family: tiling
    reads: [payload.mlir, candidate_actions.json]
    writes: [transformed_payload.real.mlir, real_set_tile_manifest.json]
    preconditions:
      - "region.kind == matmul"
      - "candidate.kind == set_tile_params"
      - "candidate.legality.ok == true"
    invalidates:
      - payload_summary
      - cost_preview
      - kernel_contracts
    preserves_refinement: bit_equality
    verification:
      - structural
      - differential
    cost: medium
    failure_modes:
      - "non_divisible_tile"
      - "boundary_handling_required"
      - "unsupported_op_in_region"
    mcp_tool: mcp__compgen__compgen_emit_agent_decision_request
    example_invocation:
      kind: set_tile_params
      candidate_id: "tile_M16_N16_K16"

Validation rules:

- ``pass_id`` matches ``^[a-z][a-z0-9_]*$``.
- ``level`` ∈ :data:`PASS_LEVELS`.
- ``family`` ∈ :data:`PASS_FAMILIES`.
- ``preserves_refinement`` ∈ :data:`REFINEMENT_KINDS`.
- ``cost`` ∈ ``{cheap, medium, expensive}``.
- ``preconditions``, ``invalidates``, and ``failure_modes`` are
  non-empty lists of non-empty strings.
- ``reads`` and ``writes`` may be empty (a profile pass might only
  read and emit a report).
- The card hashes deterministically — two cards with the same fields
  produce the same :func:`PassCard.content_hash`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

PASS_LEVELS: tuple[str, ...] = (
    "fx",
    "payload",
    "recipe",
    "semantic",
    "tile",
    "kernel",
    "plan",
    "runtime",
)

PASS_FAMILIES: tuple[str, ...] = (
    "canonicalize",
    "fusion",
    "tiling",
    "layout",
    "quant",
    "dispatch",
    "codegen",
    "verify",
    "profile",
    "promote",
    "scheduling",
    "memory",
)

REFINEMENT_KINDS: tuple[str, ...] = (
    "bit_equality",
    "tolerance_eps",
    "none",
    "unknown",
)

COST_KINDS: tuple[str, ...] = ("cheap", "medium", "expensive")

_PASS_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PassCardError(RuntimeError):
    """Raised when a pass card is malformed or fails validation."""


@dataclass(frozen=True)
class PassCard:
    """One typed pass card.

    Frozen so the registry can hand instances out without worrying
    about mutation. Round-tripping through :meth:`to_dict` /
    :meth:`from_dict` is byte-stable.
    """

    schema_version: str
    pass_id: str
    display_name: str
    level: str
    family: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    preconditions: tuple[str, ...]
    invalidates: tuple[str, ...]
    preserves_refinement: str
    verification: tuple[str, ...]
    cost: str
    failure_modes: tuple[str, ...]
    mcp_tool: str = ""
    example_invocation: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pass_id": self.pass_id,
            "display_name": self.display_name,
            "level": self.level,
            "family": self.family,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "preconditions": list(self.preconditions),
            "invalidates": list(self.invalidates),
            "preserves_refinement": self.preserves_refinement,
            "verification": list(self.verification),
            "cost": self.cost,
            "failure_modes": list(self.failure_modes),
            "mcp_tool": self.mcp_tool,
            "example_invocation": dict(self.example_invocation),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> PassCard:
        try:
            return cls(
                schema_version=str(data["schema_version"]),
                pass_id=str(data["pass_id"]),
                display_name=str(data["display_name"]),
                level=str(data["level"]),
                family=str(data["family"]),
                reads=tuple(data.get("reads") or ()),
                writes=tuple(data.get("writes") or ()),
                preconditions=tuple(data.get("preconditions") or ()),
                invalidates=tuple(data.get("invalidates") or ()),
                preserves_refinement=str(data["preserves_refinement"]),
                verification=tuple(data.get("verification") or ()),
                cost=str(data["cost"]),
                failure_modes=tuple(data.get("failure_modes") or ()),
                mcp_tool=str(data.get("mcp_tool", "")),
                example_invocation=dict(data.get("example_invocation") or {}),
                notes=str(data.get("notes", "")),
                source_path=source_path,
            )
        except KeyError as exc:
            raise PassCardError(
                f"pass card missing required field {exc.args[0]!r}"
                + (f" in {source_path}" if source_path else "")
            ) from exc

    def content_hash(self) -> str:
        """Stable SHA256[:16] over the canonical JSON projection."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _check_non_empty_list(name: str, value: tuple[str, ...], *, source: Path | None) -> None:
    if not value:
        raise PassCardError(
            f"pass card field {name!r} must be a non-empty list"
            + (f" in {source}" if source else "")
        )
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise PassCardError(
                f"pass card field {name!r} entries must be non-empty strings"
                + (f" in {source}" if source else "")
            )


def validate_card(card: PassCard) -> None:
    """Raise :class:`PassCardError` if ``card`` violates the schema."""
    if card.schema_version != "pass_card_v1":
        raise PassCardError(
            f"pass card schema_version {card.schema_version!r} must be 'pass_card_v1'"
        )
    if not _PASS_ID_RE.match(card.pass_id):
        raise PassCardError(
            f"pass_id {card.pass_id!r} must match {_PASS_ID_RE.pattern}"
        )
    if not card.display_name.strip():
        raise PassCardError(f"pass card {card.pass_id}: display_name is empty")
    if card.level not in PASS_LEVELS:
        raise PassCardError(
            f"pass card {card.pass_id}: level {card.level!r} must be in {PASS_LEVELS}"
        )
    if card.family not in PASS_FAMILIES:
        raise PassCardError(
            f"pass card {card.pass_id}: family {card.family!r} must be in {PASS_FAMILIES}"
        )
    if card.preserves_refinement not in REFINEMENT_KINDS:
        raise PassCardError(
            f"pass card {card.pass_id}: preserves_refinement "
            f"{card.preserves_refinement!r} must be in {REFINEMENT_KINDS}"
        )
    if card.cost not in COST_KINDS:
        raise PassCardError(
            f"pass card {card.pass_id}: cost {card.cost!r} must be in {COST_KINDS}"
        )
    _check_non_empty_list("preconditions", card.preconditions, source=card.source_path)
    _check_non_empty_list("invalidates", card.invalidates, source=card.source_path)
    _check_non_empty_list("failure_modes", card.failure_modes, source=card.source_path)
    _check_non_empty_list("verification", card.verification, source=card.source_path)


def load_card(path: Path) -> PassCard:
    """Load + validate a single pass card YAML."""
    if not path.exists():
        raise PassCardError(f"pass card not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PassCardError(f"pass card {path}: must be a YAML mapping")
    card = PassCard.from_dict(raw, source_path=path)
    validate_card(card)
    return card


def iter_cards(root: Path) -> Iterator[PassCard]:
    """Yield every pass card under ``root`` (sorted by pass_id)."""
    if not root.exists():
        return
    cards: list[PassCard] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        cards.append(load_card(path))
    cards.sort(key=lambda c: c.pass_id)
    yield from cards


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class PassCardRegistry:
    """In-memory pass-card registry.

    Loaded once per request build / response validate; cheap to
    reconstruct.
    """

    cards: dict[str, PassCard] = field(default_factory=dict)
    root: Path | None = None

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        validate_summary_invalidates: bool = True,
    ) -> PassCardRegistry:
        """Load + validate every pass card under ``root``.

        ``validate_summary_invalidates`` (default True) cross-links each
        card's ``invalidates`` field with the analysis-summary registry
        (M-32). Unknown summary ids raise :class:`PassCardError`. The
        flag exists so the M-32 schema tests can build a fresh registry
        without circular-import drama; production callers should leave
        it at the default.
        """
        registry = cls(root=Path(root))
        for card in iter_cards(Path(root)):
            if card.pass_id in registry.cards:
                raise PassCardError(
                    f"duplicate pass_id {card.pass_id!r} "
                    f"({registry.cards[card.pass_id].source_path} vs {card.source_path})"
                )
            registry.cards[card.pass_id] = card
        if validate_summary_invalidates:
            registry._cross_link_invalidates_to_summaries()
        return registry

    def _cross_link_invalidates_to_summaries(self) -> None:
        """Assert every card's ``invalidates`` references a known summary id.

        M-32 cross-link: pass cards declare which summaries they
        invalidate; M-33 enforces that downstream consumers do not
        read a stale summary. Without this cross-link, a typo in an
        ``invalidates`` field would be silently invisible to the
        invalidation tracker.
        """
        from compgen.analysis.checkpoints import (
            AnalysisSummaryError,
            assert_resolvable,
        )

        for card in self.cards.values():
            try:
                assert_resolvable(list(card.invalidates))
            except AnalysisSummaryError as exc:
                raise PassCardError(
                    f"pass card {card.pass_id} declares invalidates {list(card.invalidates)} "
                    f"but at least one id is not a known analysis summary: {exc}"
                ) from exc

    def __contains__(self, pass_id: str) -> bool:
        return pass_id in self.cards

    def __iter__(self) -> Iterator[PassCard]:
        for pass_id in sorted(self.cards):
            yield self.cards[pass_id]

    def __len__(self) -> int:
        return len(self.cards)

    def get(self, pass_id: str) -> PassCard | None:
        return self.cards.get(pass_id)

    def require(self, pass_id: str) -> PassCard:
        card = self.cards.get(pass_id)
        if card is None:
            from compgen.audit.errors import MissingPassCard
            raise MissingPassCard(
                f"pass_id {pass_id!r} has no pass card under {self.root}"
            )
        return card

    def passes_allowed(self) -> tuple[str, ...]:
        return tuple(sorted(self.cards))

    def assert_resolvable(self, pass_ids: list[str] | tuple[str, ...]) -> None:
        """Raise :class:`MissingPassCard` if any id is unknown."""
        missing = [p for p in pass_ids if p not in self.cards]
        if missing:
            from compgen.audit.errors import MissingPassCard
            raise MissingPassCard(
                f"the following pass_ids are referenced but have no card: "
                f"{missing} (registry root: {self.root})"
            )

    def cards_for(self, pass_ids: list[str] | tuple[str, ...]) -> list[PassCard]:
        """Return cards in the order requested (must already be resolvable)."""
        self.assert_resolvable(list(pass_ids))
        return [self.cards[p] for p in pass_ids]


def default_registry_root() -> Path:
    """Repo-rooted default registry location."""
    return Path(__file__).resolve().parents[3] / "docs" / "generated" / "pass_cards"
