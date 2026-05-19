"""Per-format loaders that normalize ingested content into UTF-8 text.

Each loader returns a ``(text, content_kind)`` pair. ``content_kind`` is
a short tag (``"markdown"``, ``"asciidoc"``, ``"c-header"``, ``"scala"``,
``"c"``, ``"pdf"``, ``"plain"``, ``"html"``) that downstream callers
pass to the router so it can tune extraction (e.g. treat C headers
as a likely source of intrinsic signatures rather than narrative ISA
text).

Loaders are deliberately minimal — no LLM, no heuristics on *content*.
They strip nothing, fold no structure. Their only job is "give me a
UTF-8 string and a hint about what it is".
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


ContentKind = str  # one of CONTENT_KINDS below — kept as str to allow extensions
CONTENT_KINDS: tuple[str, ...] = (
    "markdown",
    "asciidoc",
    "c-header",
    "scala",
    "c",
    "cpp",
    "python",
    "rust",
    "html",
    "plain",
    "pdf",
)


_EXT_TO_KIND: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "plain",
    ".txt": "plain",
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".h": "c-header",
    ".hpp": "c-header",
    ".hh": "c-header",
    ".scala": "scala",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".py": "python",
    ".rs": "rust",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


def infer_kind(path: Path) -> ContentKind:
    return _EXT_TO_KIND.get(path.suffix.lower(), "plain")


def load_path(path: Path, *, line_range: tuple[int, int] | None = None) -> tuple[str, ContentKind]:
    """Read a single file and return ``(text, content_kind)``.

    ``line_range`` is 1-indexed inclusive. PDF files use ``pypdf``
    (installed via the ``ingest`` optional extra) and silently raise
    :class:`ImportError` if the extra is missing — the pipeline turns
    that into a typed ``IngestExtraNotInstalled`` so callers don't have
    to catch ImportError directly.
    """
    if not path.is_file():
        raise FileNotFoundError(f"ingest loader: not a file: {path}")
    kind = infer_kind(path)
    if kind == "pdf":
        text = _load_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    if line_range is not None:
        start, end = line_range
        lines = text.splitlines()
        text = "\n".join(lines[max(0, start - 1) : end])
    return text, kind


def _load_pdf(path: Path) -> str:
    """PDF reader behind the ``ingest`` optional extra."""
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover — exercised when extra absent
        raise ImportError(
            "pypdf is required for PDF ingestion. Install with `uv sync --extra ingest`."
        ) from exc
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — page-level failure shouldn't fail the whole doc
            logger.warning("ingest loader: pdf page extraction failed for %s", path)
    return "\n\n".join(parts)
