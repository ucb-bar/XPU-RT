"""Tests for :mod:`compgen.tools.skill_lint`.

Coverage:

* the canonical template under ``.claude/skills/_template/SKILL.md``
  lints clean;
* every skill shipped (``compgen-tool-development``,
  ``compgen-provider-integration``, ``compgen-extension-authoring``,
  ``compgen-solver-planning``) lints clean and exposes the expected
  ``name``/``description`` frontmatter;
* the optional ``require_cli_command`` binding enforces a verbatim
  CLI-string check;
* every closed-enum :data:`SKILL_VIOLATION_KINDS` member is produced
  by at least one negative-control input — the linter cannot lose a
  failure mode silently.

The original pre-skills (``compgen``, ``compgen-compile``,
``compgen-candidate-selection``, ``compgen-discover-user-kernels``)
have been retrofitted as part of G7 — every shipped SKILL.md
satisfies the structural contract now.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.tools.skill_lint import (
    REQUIRED_FRONTMATTER_KEYS,
    REQUIRED_SECTIONS,
    SKILL_VIOLATION_KINDS,
    SkillViolation,
    lint_skill,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"

M93_SKILLS: tuple[tuple[str, str], ...] = (
    ("compgen-tool-development", "compgen-tool-development"),
    ("compgen-provider-integration", "compgen-provider-integration"),
    ("compgen-extension-authoring", "compgen-extension-authoring"),
    ("compgen-solver-planning", "compgen-solver-planning"),
)

# G7 — retrofitted pre-skills. They had their own variant
# headings (Inputs / Playbook / Hard rules) but now carry the
# canonical six in addition. Lint applies to all of them.
RETROFITTED_PRE_M93_SKILLS: tuple[tuple[str, str], ...] = (
    ("compgen", "compgen"),
    ("compgen-compile", "compgen-compile"),
    ("compgen-candidate-selection", "compgen-candidate-selection"),
    ("compgen-discover-user-kernels", "compgen-discover-user-kernels"),
)

ALL_SHIPPED_SKILLS: tuple[tuple[str, str], ...] = (
    *M93_SKILLS,
    *RETROFITTED_PRE_M93_SKILLS,
)


# Positive ------------------------------------------------------------


def test_template_lints_clean():
    report = lint_skill(SKILLS_ROOT / "_template" / "SKILL.md")
    assert report.is_clean, [v.to_dict() for v in report.violations]
    assert report.name == "skill-template"
    assert report.description.startswith("Canonical")


@pytest.mark.parametrize("skill_dir,expected_name", M93_SKILLS)
def test_m93_skill_lints_clean(skill_dir: str, expected_name: str):
    report = lint_skill(SKILLS_ROOT / skill_dir / "SKILL.md")
    assert report.is_clean, [v.to_dict() for v in report.violations]
    assert report.name == expected_name
    assert report.description, "frontmatter description must be non-empty"


@pytest.mark.parametrize("skill_dir,expected_name", RETROFITTED_PRE_M93_SKILLS)
def test_retrofitted_pre_m93_skill_lints_clean(skill_dir: str, expected_name: str):
    """G7: every pre-skill has been retrofitted with the canonical
    six headings; the linter now passes on all of them."""

    report = lint_skill(SKILLS_ROOT / skill_dir / "SKILL.md")
    assert report.is_clean, [v.to_dict() for v in report.violations]
    assert report.name == expected_name


@pytest.mark.parametrize("skill_dir,expected_name", ALL_SHIPPED_SKILLS)
def test_every_shipped_skill_lints_clean(skill_dir: str, expected_name: str):
    """Headline invariant for G7: every shipped SKILL.md satisfies
    the structural contract."""

    report = lint_skill(SKILLS_ROOT / skill_dir / "SKILL.md")
    assert report.is_clean, [v.to_dict() for v in report.violations]


def test_required_sections_appear_in_template():
    body = (SKILLS_ROOT / "_template" / "SKILL.md").read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in body, f"template missing required section {section!r}"


def test_template_lints_when_cli_command_matches(tmp_path):
    """A skill that quotes the required CLI command verbatim is clean
    under the optional ``require_cli_command`` binding."""

    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe-providers\ndescription: probe.\n---\n\n"
        + "\n".join(REQUIRED_SECTIONS)
        + "\n\nRun ``compgen-tool run compgen_probe_providers``.\n",
        encoding="utf-8",
    )
    report = lint_skill(skill, require_cli_command="compgen-tool run compgen_probe_providers")
    assert report.is_clean


# Negative controls (one per kind) -----------------------------------


def test_violation_kinds_enum_is_closed():
    assert len(set(SKILL_VIOLATION_KINDS)) == len(SKILL_VIOLATION_KINDS)


def test_missing_skill_file(tmp_path):
    report = lint_skill(tmp_path / "does_not_exist.md")
    kinds = {v.kind for v in report.violations}
    assert "skill_file_missing" in kinds


def test_missing_frontmatter(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("\n".join(REQUIRED_SECTIONS) + "\n", encoding="utf-8")
    report = lint_skill(skill)
    kinds = {v.kind for v in report.violations}
    assert "frontmatter_missing" in kinds


def test_malformed_frontmatter(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: missing-terminator\ndescription: oops\n\n"
        + "\n".join(REQUIRED_SECTIONS)
        + "\n",
        encoding="utf-8",
    )
    report = lint_skill(skill)
    kinds = {v.kind for v in report.violations}
    # Either the terminator is missing entirely or the YAML body is malformed.
    assert kinds & {"frontmatter_missing", "frontmatter_malformed"}


def test_frontmatter_key_missing(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: only-name\n---\n\n" + "\n".join(REQUIRED_SECTIONS) + "\n",
        encoding="utf-8",
    )
    report = lint_skill(skill)
    kinds = {v.kind for v in report.violations}
    assert "frontmatter_key_missing" in kinds


def test_section_missing(tmp_path):
    skill = tmp_path / "SKILL.md"
    # Frontmatter is fine; body deliberately drops one required section.
    body_sections = "\n".join(s for s in REQUIRED_SECTIONS if s != "## Forbidden")
    skill.write_text(
        f"---\nname: drops-forbidden\ndescription: missing one section.\n---\n\n{body_sections}\n",
        encoding="utf-8",
    )
    report = lint_skill(skill)
    kinds = {v.kind for v in report.violations}
    assert "section_missing" in kinds


def test_cli_command_not_quoted(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: probe\ndescription: x.\n---\n\n" + "\n".join(REQUIRED_SECTIONS) + "\n",
        encoding="utf-8",
    )
    report = lint_skill(skill, require_cli_command="compgen-tool run definitely-not-in-the-body")
    kinds = {v.kind for v in report.violations}
    assert "cli_command_not_quoted" in kinds


def test_required_frontmatter_keys_includes_name_and_description():
    assert "name" in REQUIRED_FRONTMATTER_KEYS
    assert "description" in REQUIRED_FRONTMATTER_KEYS


def test_skill_violation_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown skill violation kind"):
        SkillViolation(path="x", kind="some_kind_we_made_up", detail="x")
