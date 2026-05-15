"""Content-addressed blob store (Layer A).

Stores large immutable artifacts: source code, generated kernels,
pass code, PDL rewrites, MLIR modules, traces, profiler outputs,
verifier logs, benchmark outputs, prompt/response records.

Artifacts are stored by their SHA-256 content hash, making them
immutable and deduplicated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

log = structlog.get_logger()


def content_hash(content: str | bytes) -> str:
    """Compute SHA-256 hash of content, return hex[:16]."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


class BlobStore:
    """Content-addressed filesystem store for large artifacts.

    Blobs are stored as ``{root}/{hash[:2]}/{hash[2:]}``.

    Attributes:
        root: Root directory for the blob store.
    """

    def __init__(self, root: Path = Path(".compgen_cache/blobs")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, hash_: str) -> Path:
        """Get the filesystem path for a content hash."""
        return self.root / hash_[:2] / hash_[2:]

    def store(self, content: str | bytes) -> str:
        """Store content and return its content hash.

        If the content already exists (same hash), this is a no-op.

        Args:
            content: The artifact content (text or binary).

        Returns:
            The content hash (hex[:16]).
        """
        hash_ = content_hash(content)
        path = self._blob_path(hash_)

        if path.exists():
            return hash_

        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)

        log.debug("blob.stored", hash=hash_, size=len(content))
        return hash_

    def load(self, hash_: str) -> str | None:
        """Load content by its hash.

        Args:
            hash_: The content hash.

        Returns:
            The content as a string, or None if not found.
        """
        path = self._blob_path(hash_)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def load_bytes(self, hash_: str) -> bytes | None:
        """Load binary content by its hash."""
        path = self._blob_path(hash_)
        if not path.exists():
            return None
        return path.read_bytes()

    def exists(self, hash_: str) -> bool:
        """Check if a blob exists."""
        return self._blob_path(hash_).exists()

    def count(self) -> int:
        """Count total blobs in the store."""
        total = 0
        if self.root.exists():
            for prefix_dir in self.root.iterdir():
                if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                    total += sum(1 for _ in prefix_dir.iterdir())
        return total


__all__ = ["BlobStore", "content_hash"]
