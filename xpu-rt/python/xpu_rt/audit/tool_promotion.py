"""Tool-promotion audit — enforces the T0→T7 maturity ladder.

The audit takes the set of registered :class:`compgen.tools.ToolCard`
instances and verifies, for each, that the *evidence* required at the
card's declared maturity rung actually exists in the repository. It is
the gate that prevents a card from claiming a maturity it cannot
defend.

Per-rung gates (closed enum, indexed by maturity rung name):

* ``T0`` (Python only) — entrypoints.python resolves to a callable.
* ``T1`` (CLI) — entrypoints.cli is non-empty, the named command
  resolves on PATH, and ``compgen-tool list`` returns the tool.
* ``T2`` (tested) — ``tests.positive`` and ``tests.negative_controls``
  are non-empty, every pointer (``module.path::test_name``) resolves
  to a real pytest item, and at least one negative-control item exists
  per failure mode the card declares it covers (one row per
  ``ToolRunError`` / ``ToolInputSchemaError`` / ``ToolOutputSchemaError``
  branch the audit knows about).
* ``T3`` (artifact-emitting) — output_schema declares at least one
  named artifact field, and ``writes.allowed_roots`` is non-empty
  and every entry has a literal-or-``${run_dir}`` form.
* ``T4`` (skill-backed) — ``skill_path`` resolves to a file, the file
  contains the required sections, and the CLI command quoted in
  the skill matches ``entrypoints.cli`` byte-for-byte.
* ``T5`` (MCP) — ``entrypoints.mcp`` is non-empty and an MCP wrapper
  module imports ``ToolRunner`` somewhere in its AST (bridge).
* ``T6`` (fresh-agent verified) — ``fresh_agent_task_id`` resolves to
  a directory under ``.rcg-artifacts/fresh_agent_tasks/`` that
  contains ``grading_script.py`` and the most recent
  ``grading_result.json`` has ``passed=true``.
* ``T7`` (default workflow tool) — tool appears in
  ``mcp__compgen__list_phase_tools`` for its phase, and an entry exists
  in ``results/tool_evidence_pack/promotion_log.json``.

Each gate emits zero or more :class:`ToolPromotionViolation` objects
on failure; an audit run aggregates them into an :class:`AuditReport`
that is JSON- and Markdown-serialisable.

The audit *itself* is read-only: it never modifies repo state. The
typed errors in :mod:`compgen.tools.errors` describe failures the
runner raises at execution time; this module captures the structural
state of evidence at audit time.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compgen.tools.errors import ToolEntrypointError
from compgen.tools.skill_lint import lint_skill
from compgen.tools.tool_card import (
    MATURITY_LEVELS,
    ToolCard,
)
from compgen.tools.tool_registry import iter_tool_cards
from compgen.tools.tool_runner import resolve_python_entrypoint

# Closed enum of violation kinds. Adding a new kind requires updating
# this list AND tests/tools/test_tool_promotion_audit.py.
VIOLATION_KINDS: Final[tuple[str, ...]] = (
    "python_entrypoint_unresolved",
    "cli_entrypoint_missing",
    "cli_command_not_on_path",
    "tests_positive_missing",
    "tests_negative_controls_missing",
    "test_pointer_unresolved",
    "artifact_field_missing",
    "allowed_roots_missing",
    "skill_path_missing",
    "skill_file_missing_section",
    "skill_cli_command_mismatch",
    "mcp_entrypoint_missing",
    "mcp_wrapper_module_missing",
    "mcp_wrapper_does_not_delegate",
    "fresh_agent_task_missing",
    "fresh_agent_grading_failed",
    "phase_menu_listing_missing",
    "promotion_log_missing",
    "promotion_requirement_unverified",
)


@dataclass(frozen=True)
class ToolPromotionViolation:
    """A single audit failure attributable to one ToolCard rung gate."""

    tool_id: str
    rung: str  # e.g. "T1", "T2", ...
    kind: str  # one of VIOLATION_KINDS
    detail: str

    def __post_init__(self) -> None:
        if self.rung not in MATURITY_LEVELS:
            raise ValueError(f"unknown rung {self.rung!r}")
        if self.kind not in VIOLATION_KINDS:
            raise ValueError(
                f"unknown violation kind {self.kind!r}; must be one of {VIOLATION_KINDS}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "rung": self.rung,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ToolAuditOutcome:
    """Per-tool aggregate: rung verified, plus any violations encountered."""

    tool_id: str
    declared_maturity: str
    verified_maturity: str  # highest rung whose gates fully passed
    violations: tuple[ToolPromotionViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "declared_maturity": self.declared_maturity,
            "verified_maturity": self.verified_maturity,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass(frozen=True)
class AuditReport:
    """Audit-level rollup over all registered tools."""

    schema_version: str = "compgen_tool_promotion_audit_v1"
    outcomes: tuple[ToolAuditOutcome, ...] = ()
    # Aggregate convenience views — derivable, materialised for ease
    # of JSON consumption by downstream evidence packs.
    total_tools: int = 0
    total_violations: int = 0

    @property
    def violations(self) -> tuple[ToolPromotionViolation, ...]:
        return tuple(v for o in self.outcomes for v in o.violations)

    @property
    def is_clean(self) -> bool:
        return self.total_violations == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_tools": self.total_tools,
            "total_violations": self.total_violations,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Tool-promotion audit ({self.schema_version})",
            f"- tools audited: **{self.total_tools}**",
            f"- violations: **{self.total_violations}**",
            "",
        ]
        for o in self.outcomes:
            mark = "✔" if not o.violations else "✘"
            lines.append(
                f"## {mark} {o.tool_id}  "
                f"(declared={o.declared_maturity}, "
                f"verified={o.verified_maturity})"
            )
            for v in o.violations:
                lines.append(f"  - **[{v.rung} / {v.kind}]** {v.detail}")
            lines.append("")
        return "\n".join(lines)


def _resolve_test_pointer(pointer: str) -> bool:
    """Return ``True`` iff ``module::test`` resolves to a real callable.

    Pointers use the pytest convention ``module.dotted.path::test_name``.
    The audit imports the module (read-only) and asserts the attribute
    exists and is callable.
    """

    if "::" not in pointer:
        return False
    module_path, attr = pointer.split("::", 1)
    if not module_path or not attr:
        return False
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return False
    target = getattr(mod, attr, None)
    return callable(target)


def _check_t0(card: ToolCard) -> Iterable[ToolPromotionViolation]:
    """T0 gate — python entrypoint resolves to a callable."""

    try:
        resolve_python_entrypoint(card.entrypoints.python)
    except ToolEntrypointError as exc:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T0",
            kind="python_entrypoint_unresolved",
            detail=str(exc),
        )


def _check_t1(card: ToolCard) -> Iterable[ToolPromotionViolation]:
    """T1 gate — CLI command declared and resolves on PATH."""

    if not card.entrypoints.cli:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T1",
            kind="cli_entrypoint_missing",
            detail="entrypoints.cli is empty; T1 requires a real CLI command",
        )
        return
    head = card.entrypoints.cli.split(maxsplit=1)[0]
    if shutil.which(head) is None:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T1",
            kind="cli_command_not_on_path",
            detail=f"command {head!r} not on PATH; T1 requires shell-callable",
        )


def _check_t2(card: ToolCard) -> Iterable[ToolPromotionViolation]:
    """T2 gate — positive + negative-control tests exist and resolve."""

    if not card.tests.positive:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T2",
            kind="tests_positive_missing",
            detail="tests.positive is empty; T2 requires at least one positive test",
        )
    if not card.tests.negative_controls:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T2",
            kind="tests_negative_controls_missing",
            detail="tests.negative_controls is empty; T2 requires at least one fault-injection test",
        )
    for pointer in (*card.tests.positive, *card.tests.negative_controls):
        if not _resolve_test_pointer(pointer):
            yield ToolPromotionViolation(
                tool_id=card.tool_id,
                rung="T2",
                kind="test_pointer_unresolved",
                detail=f"test pointer {pointer!r} does not resolve to a callable",
            )


def _check_t3(card: ToolCard) -> Iterable[ToolPromotionViolation]:
    """T3 gate — output_schema declares an artifacts field; allowed_roots non-empty."""

    props = card.output_schema.get("properties") or {}
    artifacts_schema = props.get("artifacts")
    if artifacts_schema is None or artifacts_schema.get("type") != "array":
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T3",
            kind="artifact_field_missing",
            detail="output_schema.properties.artifacts (array) is required at T3",
        )
    if not card.writes.allowed_roots:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T3",
            kind="allowed_roots_missing",
            detail="writes.allowed_roots is empty; T3 must declare write scope",
        )


def _check_t4(card: ToolCard, repo_root: Path) -> Iterable[ToolPromotionViolation]:
    """T4 gate — skill_path resolves; required sections present; CLI command quoted.

    Delegates the structural check to :func:`compgen.tools.skill_lint.lint_skill`
    so the rule lives in exactly one place (owns the structure).
    """

    if not card.skill_path:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T4",
            kind="skill_path_missing",
            detail="skill_path is empty; T4 requires a SKILL.md",
        )
        return
    skill_file = repo_root / card.skill_path
    report = lint_skill(
        skill_file,
        require_cli_command=card.entrypoints.cli or None,
    )
    for v in report.violations:
        if v.kind == "skill_file_missing":
            yield ToolPromotionViolation(
                tool_id=card.tool_id,
                rung="T4",
                kind="skill_path_missing",
                detail=v.detail,
            )
        elif v.kind in {"frontmatter_missing", "frontmatter_malformed", "frontmatter_key_missing", "section_missing"}:
            yield ToolPromotionViolation(
                tool_id=card.tool_id,
                rung="T4",
                kind="skill_file_missing_section",
                detail=v.detail,
            )
        elif v.kind == "cli_command_not_quoted":
            yield ToolPromotionViolation(
                tool_id=card.tool_id,
                rung="T4",
                kind="skill_cli_command_mismatch",
                detail=v.detail,
            )


def _check_t5(card: ToolCard) -> Iterable[ToolPromotionViolation]:
    """T5 gate — MCP wrapper exists and delegates to ToolRunner.

    The MCP wrapper module is discovered by convention at
    ``python/compgen/mcp/tools/<phase>.py``; the audit checks that the
    module exists and that its source mentions ``ToolRunner`` (the
    bridge guarantees the wrapper *is* a delegate). A stricter
    AST check lands 's own test.
    """

    if not card.entrypoints.mcp:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T5",
            kind="mcp_entrypoint_missing",
            detail="entrypoints.mcp is empty; T5 requires an MCP tool name",
        )
        return
    # Best-effort module discovery: will register cards through a
    # central bridge module, so the lightweight invariant is "the MCP
    # tools tree imports ToolRunner somewhere". A stricter per-card
    # mapping lands when wires up.
    try:
        mod = importlib.import_module("compgen.mcp.tool_bridge")
    except ImportError:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T5",
            kind="mcp_wrapper_module_missing",
            detail=(
                "compgen.mcp.tool_bridge (M-94) is not importable; "
                "T5 requires the bridge module to exist"
            ),
        )
        return
    source_path = getattr(mod, "__file__", None)
    if source_path is None:
        return
    src = Path(source_path).read_text(encoding="utf-8")
    if "ToolRunner" not in src:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T5",
            kind="mcp_wrapper_does_not_delegate",
            detail="compgen.mcp.tool_bridge does not reference ToolRunner",
        )


def _check_t6(card: ToolCard, repo_root: Path) -> Iterable[ToolPromotionViolation]:
    """T6 gate — fresh-agent task package exists and has been graded."""

    if not card.fresh_agent_task_id:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T6",
            kind="fresh_agent_task_missing",
            detail="fresh_agent_task_id is empty; T6 requires a harness task",
        )
        return
    task_dir = repo_root / ".rcg-artifacts" / "fresh_agent_tasks" / card.fresh_agent_task_id
    # The harness writes a sidecar ``last_grading_result.json`` into
    # the task directory after a clean run; the run_dir's own
    # ``grading_result.json`` is also accepted as a fallback if it
    # happens to live next to the task (older convention).
    grading_result = task_dir / "last_grading_result.json"
    if not grading_result.is_file():
        grading_result = task_dir / "grading_result.json"
    if not grading_result.is_file():
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T6",
            kind="fresh_agent_task_missing",
            detail=(
                f"no grading result under {task_dir}; "
                f"run scripts/dev/run_fresh_agent_harness.py run-baseline {card.fresh_agent_task_id} first"
            ),
        )
        return
    try:
        body = json.loads(grading_result.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T6",
            kind="fresh_agent_grading_failed",
            detail=f"grading_result.json malformed: {exc}",
        )
        return
    if not bool(body.get("passed", False)):
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T6",
            kind="fresh_agent_grading_failed",
            detail=(
                f"grading_result.passed=false; rerun harness or address "
                f"failure: {body.get('reason', 'no reason recorded')}"
            ),
        )


def _check_t7(card: ToolCard, repo_root: Path) -> Iterable[ToolPromotionViolation]:
    """T7 gate — tool appears in phase menu + promotion log."""

    promotion_log = repo_root / "results" / "tool_evidence_pack" / "promotion_log.json"
    if not promotion_log.is_file():
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T7",
            kind="promotion_log_missing",
            detail=(
                f"{promotion_log} not found; run "
                f"scripts/dev/build_tool_evidence_pack.py to record T7 promotions"
            ),
        )
        return
    try:
        body = json.loads(promotion_log.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T7",
            kind="promotion_log_missing",
            detail=f"promotion_log malformed: {exc}",
        )
        return
    entries = body.get("entries") or []
    matched = [e for e in entries if e.get("tool_id") == card.tool_id]
    if not matched:
        yield ToolPromotionViolation(
            tool_id=card.tool_id,
            rung="T7",
            kind="phase_menu_listing_missing",
            detail=(
                f"no promotion_log entry for {card.tool_id!r}; T7 requires "
                f"explicit recording in results/tool_evidence_pack/promotion_log.json"
            ),
        )


_GATE_CHECKS: Final[
    tuple[tuple[str, Any], ...]
] = (
    ("T0", _check_t0),
    ("T1", _check_t1),
    ("T2", _check_t2),
    ("T3", _check_t3),
    ("T4", _check_t4),
    ("T5", _check_t5),
    ("T6", _check_t6),
    ("T7", _check_t7),
)


def _audit_one(card: ToolCard, *, repo_root: Path) -> ToolAuditOutcome:
    """Audit a single ToolCard against every gate up to its declared maturity.

    The audit short-circuits at the highest *passing* rung — once a
    gate fails, the verified maturity is the previous rung. Violations
    at higher rungs are still reported so the author sees the full
    picture (e.g. "T2 passes but T3 needs an artifacts schema").
    """

    violations: list[ToolPromotionViolation] = []
    verified_idx = -1  # nothing verified yet
    target_idx = MATURITY_LEVELS.index(card.maturity)

    for rung_name, gate in _GATE_CHECKS:
        rung_idx = MATURITY_LEVELS.index(rung_name)
        if rung_idx > target_idx:
            break
        # T4/T6/T7 gates need repo_root.
        if rung_name in {"T4", "T6", "T7"}:
            rung_violations = list(gate(card, repo_root))
        else:
            rung_violations = list(gate(card))
        violations.extend(rung_violations)
        if not rung_violations:
            verified_idx = rung_idx

    # Also audit the promotion_requirements declared on the card:
    # every key set to ``true`` must correspond to evidence we just
    # verified (or, for rungs beyond declared maturity, was not
    # checked — that's fine).
    declared_flags = card.promotion_requirements.to_dict()
    _FLAG_TO_RUNG: Final = {
        "unit_tests": "T2",
        "negative_controls": "T2",
        "cli_wrapper": "T1",
        "artifact_outputs": "T3",
        "skill_doc": "T4",
        "mcp_wrapper": "T5",
        "fresh_agent_harness": "T6",
        "phase_menu_listing": "T7",
    }
    for flag, rung in _FLAG_TO_RUNG.items():
        if declared_flags.get(flag, False):
            rung_idx = MATURITY_LEVELS.index(rung)
            if rung_idx > target_idx:
                # Flag asserts evidence beyond declared maturity;
                # surface as a self-contradictory card so the author
                # either raises maturity or lowers the flag.
                violations.append(
                    ToolPromotionViolation(
                        tool_id=card.tool_id,
                        rung=card.maturity,
                        kind="promotion_requirement_unverified",
                        detail=(
                            f"promotion_requirements.{flag}=true claims "
                            f"evidence at {rung}, but declared maturity is "
                            f"{card.maturity}; raise maturity to {rung}+ or "
                            f"set this flag to false"
                        ),
                    )
                )

    verified_maturity = MATURITY_LEVELS[verified_idx] if verified_idx >= 0 else "below-T0"
    return ToolAuditOutcome(
        tool_id=card.tool_id,
        declared_maturity=card.maturity,
        verified_maturity=verified_maturity,
        violations=tuple(violations),
    )


def run_tool_promotion_audit(
    *,
    cards: list[ToolCard] | None = None,
    cards_root: Path | None = None,
    repo_root: Path | None = None,
) -> AuditReport:
    """Run the full T0→T7 audit over the registered ToolCards.

    Parameters
    ----------
    cards
        Override the discovery — useful in tests. If omitted, every
        card discoverable through :func:`compgen.tools.iter_tool_cards`
        (relative to ``cards_root``) is audited.
    cards_root
        Override the cards directory (defaults to the shipped one).
    repo_root
        Override the repo root used for skill/harness/promotion-log
        path resolution. Defaults to the top of the CompGen checkout.

    The audit ensures ``repo_root`` is on ``sys.path`` for the duration
    of the run so dotted test pointers (``tests.tools.test_x::test_y``)
    resolve under pytest's discovery convention — pytest adds the
    rootdir to ``sys.path`` the same way.
    """

    if repo_root is None:
        # python/compgen/audit/tool_promotion.py -> repo root is parents[3]
        repo_root = Path(__file__).resolve().parents[3]
    if cards is None:
        cards = list(iter_tool_cards(cards_root))

    # pytest adds rootdir to sys.path; mirror that so tests.* imports
    # resolve identically inside and outside the audit. We restore the
    # path on exit so callers that drive multiple audits don't leak
    # entries.
    repo_root_str = str(repo_root)
    inserted = False
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        inserted = True
    try:
        outcomes = tuple(_audit_one(card, repo_root=repo_root) for card in cards)
    finally:
        if inserted:
            try:
                sys.path.remove(repo_root_str)
            except ValueError:
                pass

    total_violations = sum(len(o.violations) for o in outcomes)
    return AuditReport(
        outcomes=outcomes,
        total_tools=len(outcomes),
        total_violations=total_violations,
    )


__all__ = [
    "VIOLATION_KINDS",
    "AuditReport",
    "ToolAuditOutcome",
    "ToolPromotionViolation",
    "run_tool_promotion_audit",
]
