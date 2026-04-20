"""Tests for embedding-based semantic similarity (Unit 13)."""

from __future__ import annotations

import pytest
from compgen.memory.embeddings import MockEmbeddingProvider, cosine_similarity, embed_and_store, retrieve_by_similarity


@pytest.fixture
def memory(tmp_path):
    from compgen.memory.store import CompilerMemory

    return CompilerMemory(
        db_path=tmp_path / "test.db",
        blob_root=tmp_path / "blobs",
    )


@pytest.fixture
def provider():
    return MockEmbeddingProvider(dim=64)


class TestMockEmbeddingProvider:
    def test_dimension(self, provider):
        assert provider.dimension == 64

    def test_embed_returns_correct_dim(self, provider):
        vec = provider.embed("hello world")
        assert len(vec) == 64

    def test_embed_deterministic(self, provider):
        v1 = provider.embed("hello")
        v2 = provider.embed("hello")
        assert v1 == v2

    def test_different_texts_different_vectors(self, provider):
        v1 = provider.embed("matmul optimization")
        v2 = provider.embed("completely unrelated text")
        assert v1 != v2

    def test_embed_normalized(self, provider):
        import math

        vec = provider.embed("test")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.01


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_different_lengths(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestEmbedAndRetrieve:
    def test_store_and_retrieve(self, memory, provider):
        from compgen.memory.schema import KnowledgeKind, ScopeKind

        # Store 3 items with embeddings
        items = []
        for text in ["matmul tiling optimization", "convolution layout", "softmax kernel"]:
            item = memory.store_knowledge(
                kind=KnowledgeKind.OPTIMIZATION_TACTIC,
                summary=text,
                scope_kind=ScopeKind.GLOBAL,
            )
            embed_and_store(memory, item.knowledge_id, text, provider)
            items.append(item)

        # Query for similar
        results = retrieve_by_similarity(memory, "matmul tiling", provider, top_k=3)
        assert len(results) > 0
        # All 3 items should be returned (mock embeddings may not match semantically)
        summaries = {r.summary for r in results}
        assert "matmul tiling optimization" in summaries

    def test_empty_store_returns_empty(self, memory, provider):
        results = retrieve_by_similarity(memory, "anything", provider, top_k=5)
        assert results == []
