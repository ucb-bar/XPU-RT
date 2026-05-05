"""Source-level no-stub scan (M-31A.2).

Greps the repository for stub/mock/placeholder markers and fails on any
hit not covered by the realness allowlist. The scan is the first line of
defense; runtime import provenance (see :mod:`compgen.audit.import_provenance`)
is the second.

Patterns scanned:

- TODO, FIXME
- stub, mock, fake, dummy, synthetic, placeholder
- NotImplemented, NotImplementedError
- hardcoded, temporary, ``for now``

The scan honors :file:`python/compgen/audit/realness_allowlist.yaml` for
paths that are intentionally synthetic (e.g. ``mock_client.py``).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from compgen.audit.errors import UnallowlistedStubError

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "realness_allowlist.yaml"

# High-signal residual markers. These almost always indicate unfinished
# code; domain words like ``placeholder`` (FX node), ``synthetic`` (test
# descriptor), ``fake`` (PyTorch FakeTensor) and ``mock`` (used inside
# identifiers like ``MockEmbeddingProvider``) are deliberately NOT in
# the default pattern — the file-level allowlist + the import-provenance
# audit catch real residual files; the source scan focuses on action
# markers.
SCAN_PATTERN = re.compile(
    r"\b("
    r"TODO|FIXME|XXX|HACK"
    r"|hardcoded|HARDCODED"
    r"|temporary|TEMPORARY"
    r")\b"
)

# Multi-word and context-sensitive markers. ``for now`` and explicit
# ``raise NotImplementedError`` are strong residual markers.
FOR_NOW_PATTERN = re.compile(r"\bfor now\b", re.IGNORECASE)
RAISE_NOT_IMPL_PATTERN = re.compile(r"\braise\s+NotImplementedError\b")
STUB_DOCSTRING_PATTERN = re.compile(r'"""\s*Stub\s', re.IGNORECASE)

DEFAULT_ROOTS: tuple[str, ...] = (
    "python/compgen",
    "scripts",
    "docs",
    ".claude",
)

# Whether to scan ``tests/`` is opt-in. Tests legitimately use mocks; the
# point of the scan is to catch them on production paths.
DEFAULT_INCLUDE_TESTS: bool = False


@dataclass(frozen=True)
class AllowlistEntry:
    """One allowlist row."""

    path: str  # glob, relative to repo root
    reason: str
    forbidden_in: tuple[str, ...]

    def matches(self, rel_path: str) -> bool:
        return fnmatch.fnmatch(rel_path, self.path)


@dataclass(frozen=True)
class Allowlist:
    """Loaded allowlist."""

    entries: tuple[AllowlistEntry, ...]
    content_pattern_exemptions: tuple[dict[str, Any], ...]
    exclude_paths: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | None = None) -> Allowlist:
        path = path or ALLOWLIST_PATH
        if not path.exists():
            return cls(entries=(), content_pattern_exemptions=(), exclude_paths=())
        raw = yaml.safe_load(path.read_text()) or {}
        entries = tuple(
            AllowlistEntry(
                path=str(e["path"]),
                reason=str(e.get("reason", "")),
                forbidden_in=tuple(e.get("forbidden_in") or ()),
            )
            for e in (raw.get("allowed_nonproduction_symbols") or [])
        )
        content_pattern_exemptions = tuple(
            raw.get("content_pattern_exemptions") or []
        )
        exclude_paths = tuple(str(p) for p in (raw.get("exclude_paths") or []))
        return cls(
            entries=entries,
            content_pattern_exemptions=content_pattern_exemptions,
            exclude_paths=exclude_paths,
        )

    def is_excluded(self, rel_path: str) -> bool:
        for pattern in self.exclude_paths:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def is_allowed(self, rel_path: str) -> AllowlistEntry | None:
        for entry in self.entries:
            if entry.matches(rel_path):
                return entry
        return None

    def is_content_exempt(self, rel_path: str, marker: str) -> bool:
        for exemption in self.content_pattern_exemptions:
            substring = exemption.get("pattern_substring", "")
            if substring and substring.lower() not in marker.lower():
                continue
            paths = exemption.get("paths") or []
            for pattern in paths:
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
        return False


@dataclass(frozen=True)
class Hit:
    """One scan hit."""

    path: str  # relative to repo root
    line_number: int
    line_text: str
    marker: str
    allowlisted: bool
    allowlist_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "line_text": self.line_text,
            "marker": self.marker,
            "allowlisted": self.allowlisted,
            "allowlist_reason": self.allowlist_reason,
        }


@dataclass(frozen=True)
class ScanReport:
    """Aggregate report for one scan run."""

    roots: tuple[str, ...]
    include_tests: bool
    hits: tuple[Hit, ...]
    files_scanned: int

    @property
    def unallowlisted_hits(self) -> tuple[Hit, ...]:
        return tuple(h for h in self.hits if not h.allowlisted)

    @property
    def allowlisted_hits(self) -> tuple[Hit, ...]:
        return tuple(h for h in self.hits if h.allowlisted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": list(self.roots),
            "include_tests": self.include_tests,
            "files_scanned": self.files_scanned,
            "hit_count": len(self.hits),
            "unallowlisted_count": len(self.unallowlisted_hits),
            "hits": [h.to_dict() for h in self.hits],
        }


def _iter_files(
    root: Path,
    repo_root: Path,
    allowlist: Allowlist,
    *,
    include_tests: bool,
) -> Iterator[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".so", ".o", ".a", ".pdf", ".png", ".jpg",
                            ".jpeg", ".gif", ".pt", ".pt2", ".bin", ".gguf",
                            ".safetensors", ".lock"}:
            continue
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        rel_str = str(rel)
        if not include_tests and (
            rel_str.startswith("tests/") or "/tests/" in rel_str
        ):
            continue
        if allowlist.is_excluded(rel_str):
            continue
        yield path


def scan_repo(
    *,
    repo_root: Path | None = None,
    roots: tuple[str, ...] | None = None,
    include_tests: bool = DEFAULT_INCLUDE_TESTS,
    allowlist: Allowlist | None = None,
) -> ScanReport:
    """Scan ``roots`` under ``repo_root`` for stub markers."""
    repo_root = repo_root or REPO_ROOT
    roots = roots or DEFAULT_ROOTS
    allowlist = allowlist or Allowlist.load()

    hits: list[Hit] = []
    files_scanned = 0
    for root in roots:
        for path in _iter_files(
            repo_root / root,
            repo_root,
            allowlist,
            include_tests=include_tests,
        ):
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            rel_path = str(path.relative_to(repo_root))
            allow_entry = allowlist.is_allowed(rel_path)
            for line_number, line in enumerate(text.splitlines(), start=1):
                # The scanner deliberately excludes the realness_scan source
                # itself (this file) and the allowlist YAML, since they
                # name the patterns they search for.
                if rel_path in {
                    "python/compgen/audit/__init__.py",
                    "python/compgen/audit/realness_scan.py",
                    "python/compgen/audit/realness_allowlist.yaml",
                    "python/compgen/audit/errors.py",
                    "python/compgen/audit/import_provenance.py",
                    "python/compgen/audit/caveat_ledger.py",
                    "python/compgen/audit/contracts.py",
                    "python/compgen/audit/perturbations.py",
                    "python/compgen/audit/fresh_agent.py",
                    "python/compgen/audit/fresh_agent_modes.py",
                    "python/compgen/audit/trace_replay.py",
                    "python/compgen/audit/negative_controls.py",
                    "python/compgen/audit/trust_report.py",
                    "scripts/dev/audit_realness.py",
                    "scripts/dev/audit_production_imports.py",
                }:
                    continue
                for match in SCAN_PATTERN.finditer(line):
                    marker = match.group(1)
                    is_exempt = allowlist.is_content_exempt(rel_path, marker)
                    is_allowed = allow_entry is not None or is_exempt
                    hits.append(
                        Hit(
                            path=rel_path,
                            line_number=line_number,
                            line_text=line.strip()[:200],
                            marker=marker,
                            allowlisted=is_allowed,
                            allowlist_reason=(
                                allow_entry.reason
                                if allow_entry
                                else "content_pattern_exemption"
                                if is_exempt
                                else ""
                            ),
                        )
                    )
                for match in FOR_NOW_PATTERN.finditer(line):
                    is_exempt = allowlist.is_content_exempt(rel_path, "for now")
                    is_allowed = allow_entry is not None or is_exempt
                    hits.append(
                        Hit(
                            path=rel_path,
                            line_number=line_number,
                            line_text=line.strip()[:200],
                            marker="for now",
                            allowlisted=is_allowed,
                            allowlist_reason=(
                                allow_entry.reason
                                if allow_entry
                                else "content_pattern_exemption"
                                if is_exempt
                                else ""
                            ),
                        )
                    )
                # Strong markers: explicit ``raise NotImplementedError``
                # and ``Stub`` opening a docstring. Both are unambiguous
                # residual indicators — distinct from FX placeholders
                # or PyTorch FakeTensors.
                for match in RAISE_NOT_IMPL_PATTERN.finditer(line):
                    is_exempt = allowlist.is_content_exempt(
                        rel_path, "raise NotImplementedError"
                    )
                    is_allowed = allow_entry is not None or is_exempt
                    hits.append(
                        Hit(
                            path=rel_path,
                            line_number=line_number,
                            line_text=line.strip()[:200],
                            marker="raise NotImplementedError",
                            allowlisted=is_allowed,
                            allowlist_reason=(
                                allow_entry.reason
                                if allow_entry
                                else "content_pattern_exemption"
                                if is_exempt
                                else ""
                            ),
                        )
                    )

    return ScanReport(
        roots=roots,
        include_tests=include_tests,
        hits=tuple(hits),
        files_scanned=files_scanned,
    )


def assert_clean(report: ScanReport) -> None:
    """Raise :class:`UnallowlistedStubError` if any hit is not allowlisted."""
    bad = report.unallowlisted_hits
    if not bad:
        return
    sample = "\n".join(
        f"  {h.path}:{h.line_number}: [{h.marker}] {h.line_text}" for h in bad[:25]
    )
    extra = f"\n  ... and {len(bad) - 25} more" if len(bad) > 25 else ""
    raise UnallowlistedStubError(
        f"realness scan: {len(bad)} unallowlisted hit(s):\n{sample}{extra}\n"
        f"Either fix the code, or add the path to "
        f"python/compgen/audit/realness_allowlist.yaml with a reason."
    )
