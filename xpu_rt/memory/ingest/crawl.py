"""URL fetching for the ingestion pipeline (optional ``ingest`` extra).

Wraps Crawl4AI when the optional dependency is installed; otherwise
raises :class:`IngestExtraNotInstalled` so the pipeline can report a
clean error without forcing every install to drag in Playwright.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchedDoc:
    """One URL crawl result."""

    url: str
    markdown: str
    sha256: str
    fetched_at: str
    title: str = ""
    bytes_count: int = 0


class IngestExtraNotInstalled(RuntimeError):
    """Raised when the ``ingest`` optional extra is required but absent."""


def fetch_url(url: str, *, follow_links: bool = False) -> FetchedDoc:
    """Fetch ``url`` via Crawl4AI and return the extracted markdown.

    Raises :class:`IngestExtraNotInstalled` when Crawl4AI is not
    importable — install with ``uv sync --extra ingest``.
    """
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IngestExtraNotInstalled(
            "URL ingestion requires Crawl4AI. Install with "
            "`uv sync --extra ingest` and re-run."
        ) from exc

    import asyncio

    async def _run() -> FetchedDoc:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        markdown = getattr(result, "markdown", "") or ""
        title = getattr(result, "metadata", {}).get("title", "") if hasattr(result, "metadata") else ""
        return FetchedDoc(
            url=url,
            markdown=markdown,
            sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            title=str(title),
            bytes_count=len(markdown),
        )

    return asyncio.run(_run())


__all__ = ["FetchedDoc", "IngestExtraNotInstalled", "fetch_url"]
