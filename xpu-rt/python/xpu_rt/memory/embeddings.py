"""Embedding-based semantic similarity for knowledge retrieval.

Provides an embedding provider protocol and cosine similarity search
to enhance CompilerMemory's retrieve_similar() with semantic matching
instead of exact scope_key matching.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from compgen.memory.schema import KnowledgeItem
    from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Embed text into a fixed-dimension vector."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class MockEmbeddingProvider:
    """Deterministic mock embedding provider for tests.

    Uses hash-based vectors to produce stable, reproducible embeddings
    that maintain basic similarity properties.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding from text hash."""
        h = hashlib.sha256(text.encode()).hexdigest()
        # Use pairs of hex chars as seed values
        raw = []
        for i in range(0, min(len(h), self._dim * 2), 2):
            val = int(h[i:i+2], 16) / 255.0 - 0.5
            raw.append(val)
        # Pad if needed
        while len(raw) < self._dim:
            raw.append(0.0)
        raw = raw[:self._dim]
        # Normalize
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def embed_and_store(
    memory: CompilerMemory,
    knowledge_id: str,
    text: str,
    provider: EmbeddingProvider,
) -> str:
    """Compute embedding for text and store as blob in memory.

    Args:
        memory: CompilerMemory instance.
        knowledge_id: The knowledge item to associate with.
        text: Text to embed.
        provider: Embedding provider.

    Returns:
        Blob hash of stored embedding.
    """
    vector = provider.embed(text)
    blob_content = json.dumps({"vector": vector, "dimension": provider.dimension})
    blob_hash = memory.blobs.store(blob_content)

    # Update the knowledge item's embedding_hash
    memory.db.execute(
        "UPDATE knowledge_items SET embedding_hash = ? WHERE knowledge_id = ?",
        (blob_hash, knowledge_id),
    )
    memory.db.commit()
    return blob_hash


def retrieve_by_similarity(
    memory: CompilerMemory,
    query_text: str,
    provider: EmbeddingProvider,
    top_k: int = 5,
) -> list[KnowledgeItem]:
    """Retrieve knowledge items by embedding similarity.

    Args:
        memory: CompilerMemory instance.
        query_text: Text to find similar items for.
        provider: Embedding provider.
        top_k: Number of results to return.

    Returns:
        List of KnowledgeItem sorted by similarity (highest first).
    """
    from compgen.memory.schema import KnowledgeItem as KI

    query_vec = provider.embed(query_text)

    # Fetch all items with embeddings
    rows = memory.db.fetchall(
        "SELECT * FROM knowledge_items WHERE embedding_hash != '' AND embedding_hash IS NOT NULL",
        (),
    )

    scored: list[tuple[float, Any]] = []
    for row in rows:
        item = memory._row_to_knowledge(row)
        try:
            blob = memory.blobs.load(item.embedding_hash)
            data = json.loads(blob)
            item_vec = data.get("vector", [])
            sim = cosine_similarity(query_vec, item_vec)
            scored.append((sim, item))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "cosine_similarity",
    "embed_and_store",
    "retrieve_by_similarity",
]
