"""Chunk long content into router-sized slices.

The router is a single LLM (or single Claude Code agent-file request)
per chunk, and modern LLMs handle ~100KB of text comfortably. The
strategy is paragraph-aware: prefer to split on blank lines, falling
back to single-newline boundaries, then to mid-line if neither
appears in time. The goal is to keep ``mnemonic <-> description``
pairs and ``#define`` macros together rather than to optimize chunk
balance.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_CHARS = 60_000
DEFAULT_OVERLAP_CHARS = 800


@dataclass(frozen=True)
class TextChunk:
    """One unit of text handed to the router."""

    text: str
    start_offset: int
    end_offset: int
    chunk_index: int
    chunk_total: int


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Returns a single chunk when the input fits; otherwise splits on the
    most-paragraph-like boundary within the last 10% of each window,
    falling back to a hard cut if no such boundary exists. Adjacent
    chunks share ``overlap_chars`` of suffix→prefix overlap so an
    instruction definition straddling a boundary still survives.
    """
    n = len(text)
    if n <= max_chars:
        return [TextChunk(text=text, start_offset=0, end_offset=n, chunk_index=0, chunk_total=1)]

    chunks: list[TextChunk] = []
    start = 0
    while start < n:
        ideal_end = min(start + max_chars, n)
        if ideal_end >= n:
            cut = n
        else:
            window_lo = ideal_end - max(max_chars // 10, 256)
            cut = _best_split(text, lo=window_lo, hi=ideal_end)
        chunks.append(
            TextChunk(
                text=text[start:cut],
                start_offset=start,
                end_offset=cut,
                chunk_index=len(chunks),
                chunk_total=0,  # filled in below
            )
        )
        if cut >= n:
            break
        start = max(cut - overlap_chars, cut - (max_chars // 2))

    total = len(chunks)
    return [
        TextChunk(
            text=c.text,
            start_offset=c.start_offset,
            end_offset=c.end_offset,
            chunk_index=c.chunk_index,
            chunk_total=total,
        )
        for c in chunks
    ]


def _best_split(text: str, *, lo: int, hi: int) -> int:
    """Return the best cut offset in ``[lo, hi]``, preferring paragraph breaks."""
    candidates = (
        text.rfind("\n\n", lo, hi),
        text.rfind("\n#", lo, hi),  # markdown heading start
        text.rfind("\n```", lo, hi),  # code fence boundary
        text.rfind("\n", lo, hi),
    )
    for c in candidates:
        if c != -1:
            return c + 1
    return hi
