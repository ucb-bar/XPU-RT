"""Pass-plan scheduler invariants (M-34.2 + M-34.3).

The agent's response can carry a ``pass_plan`` — an ordered list of
:class:`PassPlanStep` records — instead of a single
``selected_candidate_id``. The validator checks the plan against four
invariants:

1. **Phase ordering** (M-34.1): a phase-N pass cannot appear before
   any phase-(<N) pass in the plan.
2. **Pair: requires_after** (M-34.2): if a card declares
   ``requires_after: [other_pass]``, the plan must contain ``other_pass``
   strictly later than self.
3. **Pair: excludes** (M-34.2): if a card declares
   ``excludes: [other_pass]``, the plan must not contain ``other_pass``
   anywhere.
4. **Structural** (M-34.3): every ``pass_id`` resolves to a real card,
   ``candidate_id`` resolves against the request's
   ``candidate_ids_allowed``, and no duplicate steps.

The validator is non-raising for diagnostic use
(:func:`inspect_pass_plan`) and raising for the validator path
(:func:`assert_pass_plan_valid`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from compgen.audit.errors import (
    PairContractViolation,
    PassPlanInvalid,
    PhaseTransitionViolation,
)
from compgen.passes.cards import (
    PASS_PHASES,
    PassCard,
    PassCardRegistry,
    phase_index,
)


@dataclass(frozen=True)
class PassPlanStep:
    """One step in the agent's pass plan.

    Mirrors the JSON shape carried by ``agent_decision_response.pass_plan``:

      {
        "pass_id":       "set_tile_params",
        "region_id":     "matmul_0",
        "candidate_id":  "tile_M16_N16_K16",
        "rationale":     {...}    # optional free-form
      }
    """

    pass_id: str
    region_id: str = ""
    candidate_id: str = ""
    rationale: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "rationale": dict(self.rationale),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PassPlanStep:
        if "pass_id" not in raw:
            raise PassPlanInvalid(
                f"pass plan step missing 'pass_id': {raw}"
            )
        return cls(
            pass_id=str(raw["pass_id"]),
            region_id=str(raw.get("region_id", "")),
            candidate_id=str(raw.get("candidate_id", "")),
            rationale=dict(raw.get("rationale") or {}),
        )


@dataclass(frozen=True)
class PassPlanReport:
    """Diagnostic record for a plan validation."""

    plan: tuple[PassPlanStep, ...]
    holds: bool
    structural_ok: bool
    phase_ok: bool
    requires_after_ok: bool
    excludes_ok: bool
    structural_detail: str = ""
    phase_detail: str = ""
    requires_after_detail: str = ""
    excludes_detail: str = ""

    @property
    def detail(self) -> str:
        bits: list[str] = []
        for d in (
            self.structural_detail,
            self.phase_detail,
            self.requires_after_detail,
            self.excludes_detail,
        ):
            if d:
                bits.append(d)
        return "; ".join(bits)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _check_structural(
    plan: Sequence[PassPlanStep],
    *,
    registry: PassCardRegistry,
    candidate_ids_allowed: Iterable[str] | None,
) -> tuple[bool, str]:
    seen: set[tuple[str, str, str]] = set()
    for i, step in enumerate(plan):
        if not step.pass_id:
            return False, f"plan step {i}: empty pass_id"
        if step.pass_id not in registry:
            return False, (
                f"plan step {i} pass_id={step.pass_id!r} has no pass card"
            )
        key = (step.pass_id, step.region_id, step.candidate_id)
        if key in seen:
            return False, (
                f"plan step {i} duplicates earlier step: {key}"
            )
        seen.add(key)
        if (
            candidate_ids_allowed is not None
            and step.candidate_id
            and step.candidate_id not in set(candidate_ids_allowed)
        ):
            return False, (
                f"plan step {i} candidate_id={step.candidate_id!r} not in "
                f"candidate_ids_allowed"
            )
    return True, ""


def _check_phase_ordering(
    plan: Sequence[PassPlanStep],
    *,
    registry: PassCardRegistry,
) -> tuple[bool, str]:
    last_phase_idx = -1
    last_phase_name = ""
    for i, step in enumerate(plan):
        card = registry.get(step.pass_id)
        if card is None:
            # Structural check should already have caught this.
            return False, f"plan step {i}: pass_id has no card"
        idx = phase_index(card.effective_phase())
        if idx < last_phase_idx:
            return False, (
                f"plan step {i} ({step.pass_id}) is in phase "
                f"{card.effective_phase()!r} (index {idx}) but a previous "
                f"step was in phase {last_phase_name!r} (index "
                f"{last_phase_idx}); phase order is strict: {PASS_PHASES}"
            )
        last_phase_idx = idx
        last_phase_name = card.effective_phase()
    return True, ""


def _check_requires_after(
    plan: Sequence[PassPlanStep],
    *,
    registry: PassCardRegistry,
) -> tuple[bool, str]:
    # Index pass_id positions in the plan
    positions: dict[str, list[int]] = {}
    for i, step in enumerate(plan):
        positions.setdefault(step.pass_id, []).append(i)
    for i, step in enumerate(plan):
        card = registry.get(step.pass_id)
        if card is None:
            continue
        for required in card.requires_after:
            if required not in positions:
                return False, (
                    f"plan step {i} ({step.pass_id}) requires "
                    f"{required!r} to appear after it; "
                    f"{required!r} is not in the plan"
                )
            if not any(j > i for j in positions[required]):
                return False, (
                    f"plan step {i} ({step.pass_id}) requires "
                    f"{required!r} to appear strictly later, but "
                    f"{required!r} only appears at or before position {i}"
                )
    return True, ""


def _check_excludes(
    plan: Sequence[PassPlanStep],
    *,
    registry: PassCardRegistry,
) -> tuple[bool, str]:
    pass_ids_in_plan = {s.pass_id for s in plan}
    for i, step in enumerate(plan):
        card = registry.get(step.pass_id)
        if card is None:
            continue
        for excluded in card.excludes:
            if excluded in pass_ids_in_plan and excluded != step.pass_id:
                return False, (
                    f"plan step {i} ({step.pass_id}) excludes "
                    f"{excluded!r}, but {excluded!r} is also in the plan"
                )
    return True, ""


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def inspect_pass_plan(
    plan: Sequence[PassPlanStep] | Sequence[dict[str, Any]],
    *,
    registry: PassCardRegistry,
    candidate_ids_allowed: Iterable[str] | None = None,
) -> PassPlanReport:
    """Non-raising diagnostic. Returns a :class:`PassPlanReport`."""
    coerced: tuple[PassPlanStep, ...] = tuple(
        s if isinstance(s, PassPlanStep) else PassPlanStep.from_dict(s)
        for s in plan
    )

    structural_ok, structural_detail = _check_structural(
        coerced,
        registry=registry,
        candidate_ids_allowed=candidate_ids_allowed,
    )
    phase_ok, phase_detail = (True, "")
    requires_after_ok, requires_after_detail = (True, "")
    excludes_ok, excludes_detail = (True, "")
    # Skip dependent checks if structural failed — their messages would
    # be confusing.
    if structural_ok:
        phase_ok, phase_detail = _check_phase_ordering(
            coerced, registry=registry,
        )
        requires_after_ok, requires_after_detail = _check_requires_after(
            coerced, registry=registry,
        )
        excludes_ok, excludes_detail = _check_excludes(
            coerced, registry=registry,
        )

    holds = all((structural_ok, phase_ok, requires_after_ok, excludes_ok))
    return PassPlanReport(
        plan=coerced,
        holds=holds,
        structural_ok=structural_ok,
        phase_ok=phase_ok,
        requires_after_ok=requires_after_ok,
        excludes_ok=excludes_ok,
        structural_detail=structural_detail,
        phase_detail=phase_detail,
        requires_after_detail=requires_after_detail,
        excludes_detail=excludes_detail,
    )


def assert_pass_plan_valid(
    plan: Sequence[PassPlanStep] | Sequence[dict[str, Any]],
    *,
    registry: PassCardRegistry,
    candidate_ids_allowed: Iterable[str] | None = None,
) -> None:
    """Raise the most specific typed error for the first violation found."""
    report = inspect_pass_plan(
        plan, registry=registry,
        candidate_ids_allowed=candidate_ids_allowed,
    )
    if not report.structural_ok:
        raise PassPlanInvalid(report.structural_detail)
    if not report.phase_ok:
        raise PhaseTransitionViolation(report.phase_detail)
    if not report.requires_after_ok:
        raise PairContractViolation(report.requires_after_detail)
    if not report.excludes_ok:
        raise PairContractViolation(report.excludes_detail)
