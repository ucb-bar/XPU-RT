"""Source manifest data types.

A :class:`SourceRef` names one ingestible item — a local file, a local
directory (recursively walked), or an HTTP(S) URL — together with a
*role hint* that helps the router downstream pick the right bucket
when the content is ambiguous. The role is a hint only; the LLM is
free to overrule it.

A :class:`SourceManifest` bundles refs for one target together with the
target id, target-profile cross-reference, and the ISA family label.
Per-target seed modules (e.g. ``xpu_rt.memory.seeds.gemmini``) export
one of these constants and let the pipeline do the actual work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

SourceKind = Literal["path", "url", "directory"]
RoleHint = Literal[
    "isa",
    "architecture",
    "intrinsics",
    "examples",
    "constraints",
    "auto",
]


@dataclass(frozen=True)
class SourceRef:
    """One ingestible item.

    Attributes:
        locator: Local path string or URL.
        kind: How to interpret ``locator`` — ``"path"`` for a single file,
            ``"directory"`` for a directory tree (recursively walked), or
            ``"url"`` for an HTTP(S) URL fetched via Crawl4AI.
        role: A bucket hint, or ``"auto"`` (the default) to let the
            router decide solely from content.
        line_range: Optional ``(start, end)`` 1-indexed inclusive line
            slice; useful for pulling just the ISA section out of a
            long README without paying to ingest the rest.
        glob: Restrict directory walks to files matching this glob
            (e.g. ``"*.adoc"``). Ignored for non-directory kinds.
        max_depth: Directory recursion depth (0 = top-level only).
        tags: Free-form labels passed through to extracted records so a
            user can correlate "this exemplar came from the bareMetalC
            directory".
    """

    locator: str
    kind: SourceKind = "path"
    role: RoleHint = "auto"
    line_range: tuple[int, int] | None = None
    glob: str = ""
    max_depth: int = 4
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind in ("path", "directory") and not self.locator:
            raise ValueError("local SourceRef requires a non-empty locator")
        if self.kind == "url" and not (
            self.locator.startswith("http://") or self.locator.startswith("https://")
        ):
            raise ValueError(f"url SourceRef must use http(s); got {self.locator!r}")

    def as_path(self) -> Path:
        if self.kind == "url":
            raise ValueError(f"SourceRef {self.locator!r} is a URL, not a local path")
        return Path(self.locator)


@dataclass(frozen=True)
class SourceManifest:
    """Per-target ingestion plan.

    The pipeline reads this, walks each :class:`SourceRef`, and folds
    every routed result into the target's knowledge card.
    """

    target_id: str
    target_profile_ref: str
    isa_family: str
    sources: tuple[SourceRef, ...] = ()
    # Static, no-LLM facts the seed wants on the card before any
    # ingestion runs (e.g. memory tier sizes pulled straight from a
    # vendor config). Stored as a plain dict-of-anything so manifests
    # can stay declarative.
    static_hardware_facts: dict[str, object] = field(default_factory=dict)
    description: str = ""

    def with_sources(self, *more: SourceRef) -> SourceManifest:
        return SourceManifest(
            target_id=self.target_id,
            target_profile_ref=self.target_profile_ref,
            isa_family=self.isa_family,
            sources=self.sources + tuple(more),
            static_hardware_facts=dict(self.static_hardware_facts),
            description=self.description,
        )


def expand_sources(refs: Iterable[SourceRef]) -> list[SourceRef]:
    """Expand directory refs into one ``kind="path"`` ref per matching file.

    URL refs and explicit path refs pass through untouched. Directory
    refs are walked respecting ``glob`` and ``max_depth``; each resulting
    file inherits the parent's ``role`` and ``tags``.
    """
    out: list[SourceRef] = []
    for ref in refs:
        if ref.kind != "directory":
            out.append(ref)
            continue
        base = ref.as_path()
        if not base.is_dir():
            continue
        pattern = ref.glob or "*"
        candidates = sorted(_walk(base, ref.max_depth, pattern))
        for candidate in candidates:
            out.append(
                SourceRef(
                    locator=str(candidate),
                    kind="path",
                    role=ref.role,
                    line_range=None,
                    glob="",
                    max_depth=0,
                    tags=ref.tags,
                )
            )
    return out


def _walk(base: Path, max_depth: int, pattern: str) -> Iterable[Path]:
    base_depth = len(base.parts)
    stack = [base]
    while stack:
        cur = stack.pop()
        try:
            entries = sorted(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if (len(entry.parts) - base_depth) < max_depth:
                    stack.append(entry)
                continue
            if entry.match(pattern):
                yield entry


def manifest_to_json(manifest: SourceManifest) -> str:
    """Serialize a manifest to JSON for logs/debug; not on a hot path."""
    return json.dumps(
        {
            "target_id": manifest.target_id,
            "target_profile_ref": manifest.target_profile_ref,
            "isa_family": manifest.isa_family,
            "description": manifest.description,
            "static_hardware_facts": manifest.static_hardware_facts,
            "sources": [
                {
                    "locator": s.locator,
                    "kind": s.kind,
                    "role": s.role,
                    "line_range": list(s.line_range) if s.line_range else None,
                    "glob": s.glob,
                    "max_depth": s.max_depth,
                    "tags": list(s.tags),
                }
                for s in manifest.sources
            ],
        },
        indent=2,
    )
