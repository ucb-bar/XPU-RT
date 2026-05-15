"""Skill linter.

A skill is the operational manual that lets a *fresh* Claude session
discover, call, and trust a tool. The linter enforces the structural
contract every shipped SKILL.md must honour:

* a frontmatter block (``---`` YAML), containing ``name`` and
  ``description`` at minimum;
* the six required headings (``## When to use``, ``## First command``,
  ``## Required artifacts``, ``## How to interpret``, ``## Forbidden``,
  ``## Caveats``);
* if a CLI command is associated with the skill, the skill body must
  quote it byte-for-byte (so the fresh-agent grader can grep for it
  before invoking it).

The linter returns a typed report. The T4 gate consumes the same
report rather than re-implementing section checks; that keeps the rule
in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## When to use",
    "## First command",
    "## Required artifacts",
    "## How to interpret",
    "## Forbidden",
    "## Caveats",
)

REQUIRED_FRONTMATTER_KEYS: Final[tuple[str, ...]] = ("name", "description")

SKILL_VIOLATION_KINDS: Final[tuple[str, ...]] = (
    "skill_file_missing",
    "frontmatter_missing",
    "frontmatter_malformed",
    "frontmatter_key_missing",
    "section_missing",
    "cli_command_not_quoted",
)


@dataclass(frozen=True)
class SkillViolation:
    """Single skill-lint failure."""

    path: str
    kind: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in SKILL_VIOLATION_KINDS:
            raise ValueError(
                f"unknown skill violation kind {self.kind!r}; "
                f"must be one of {SKILL_VIOLATION_KINDS}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class SkillLintReport:
    """Lint result for a single skill file."""

    path: str
    name: str
    description: str
    violations: tuple[SkillViolation, ...]

    @property
    def is_clean(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "violations": [v.to_dict() for v in self.violations],
        }


def _extract_frontmatter(body: str) -> tuple[dict[str, str] | None, str | None]:
    """Extract YAML frontmatter from ``body``.

    Returns ``(mapping, error)``: on success ``mapping`` is the parsed
    dict and ``error`` is ``None``; on missing/malformed frontmatter
    ``mapping`` is ``None`` and ``error`` describes the failure.
    """

    if not body.startswith("---\n"):
        return None, "no leading '---' frontmatter delimiter"
    end = body.find("\n---\n", 4)
    if end == -1:
        return None, "frontmatter is not terminated with a '---' line"
    raw = body[4:end]
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, f"frontmatter YAML did not parse: {exc}"
    if not isinstance(loaded, dict):
        return None, f"frontmatter must be a mapping; got {type(loaded).__name__}"
    return {str(k): str(v) for k, v in loaded.items()}, None


def lint_skill(
    skill_path: Path,
    *,
    require_cli_command: str | None = None,
) -> SkillLintReport:
    """Lint ``skill_path`` and return the report.

    Parameters
    ----------
    skill_path
        Path to ``SKILL.md``. The file is read once; no other I/O.
    require_cli_command
        If non-None, the skill body must quote this command verbatim.
        Used by the T4 gate to bind a skill to a ToolCard's
        ``entrypoints.cli``.
    """

    violations: list[SkillViolation] = []
    if not skill_path.is_file():
        violations.append(
            SkillViolation(
                path=str(skill_path),
                kind="skill_file_missing",
                detail=f"{skill_path} does not exist",
            )
        )
        return SkillLintReport(
            path=str(skill_path), name="", description="", violations=tuple(violations)
        )

    body = skill_path.read_text(encoding="utf-8")
    frontmatter, fm_error = _extract_frontmatter(body)
    name = ""
    description = ""
    if frontmatter is None:
        violations.append(
            SkillViolation(
                path=str(skill_path),
                kind=(
                    "frontmatter_missing"
                    if fm_error and "no leading" in fm_error
                    else "frontmatter_malformed"
                ),
                detail=fm_error or "frontmatter could not be parsed",
            )
        )
    else:
        for key in REQUIRED_FRONTMATTER_KEYS:
            if not frontmatter.get(key):
                violations.append(
                    SkillViolation(
                        path=str(skill_path),
                        kind="frontmatter_key_missing",
                        detail=f"frontmatter missing required key {key!r}",
                    )
                )
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

    for section in REQUIRED_SECTIONS:
        if section not in body:
            violations.append(
                SkillViolation(
                    path=str(skill_path),
                    kind="section_missing",
                    detail=f"required section {section!r} not found",
                )
            )

    if require_cli_command and require_cli_command not in body:
        violations.append(
            SkillViolation(
                path=str(skill_path),
                kind="cli_command_not_quoted",
                detail=(
                    f"skill must quote the exact CLI command "
                    f"{require_cli_command!r} (the M-92 T4 gate "
                    f"enforces this binding)"
                ),
            )
        )

    return SkillLintReport(
        path=str(skill_path),
        name=name,
        description=description,
        violations=tuple(violations),
    )


__all__ = [
    "REQUIRED_FRONTMATTER_KEYS",
    "REQUIRED_SECTIONS",
    "SKILL_VIOLATION_KINDS",
    "SkillLintReport",
    "SkillViolation",
    "lint_skill",
]
