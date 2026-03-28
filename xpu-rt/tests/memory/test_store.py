"""Tests for the unified Compiler Memory System."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from compgen.memory.blobs import BlobStore, content_hash
from compgen.memory.schema import (
    CandidateStatus,
    GeneratorKind,
    KnowledgeKind,
    ObjectKind,
    ScopeKind,
)
from compgen.memory.store import CompilerMemory


@pytest.fixture
def memory(tmp_path: Path) -> CompilerMemory:
    """Create a CompilerMemory in a temp directory."""
    return CompilerMemory(
        db_path=tmp_path / "test_memory.db",
        blob_root=tmp_path / "blobs",
    )


class TestBlobStore:
    """Test content-addressed blob store."""

    def test_store_and_load(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        h = store.store("hello world")
        assert store.exists(h)
        assert store.load(h) == "hello world"

    def test_dedup(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        h1 = store.store("same content")
        h2 = store.store("same content")
        assert h1 == h2
        assert store.count() == 1

    def test_different_content(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        h1 = store.store("content A")
        h2 = store.store("content B")
        assert h1 != h2
        assert store.count() == 2

    def test_nonexistent(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        assert store.load("nonexistent") is None
        assert not store.exists("nonexistent")


class TestCompilerMemoryTasks:
    """Test task lifecycle."""

    def test_create_and_get(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL, workload_key="matmul", target_key="h100")
        assert task.task_id
        assert task.task_kind == ObjectKind.KERNEL

        retrieved = memory.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.workload_key == "matmul"

    def test_nonexistent_task(self, memory: CompilerMemory) -> None:
        assert memory.get_task("nonexistent") is None


class TestCompilerMemoryCandidates:
    """Test candidate lifecycle."""

    def test_full_lifecycle(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        candidate = memory.record_candidate(task.task_id, artifact="def kernel(): pass")
        assert candidate.status == CandidateStatus.NEW
        assert candidate.artifact_hash

        # Verify blob was stored
        code = memory.blobs.load(candidate.artifact_hash)
        assert code == "def kernel(): pass"

        # Evaluate
        eval_ = memory.record_evaluation(
            candidate.candidate_id,
            compile_ok=True,
            correctness_ok=True,
            latency_us=100.0,
            score=0.8,
        )
        assert eval_.compile_ok

        # Promote
        promo = memory.promote_candidate(
            candidate.candidate_id,
            promotion_key="matmul_h100_latency",
            measured_gain=1.5,
        )
        assert promo.version == 1

        # Check status updated
        candidates = memory.get_candidates(task.task_id, CandidateStatus.PROMOTED)
        assert len(candidates) == 1

    def test_version_incrementing(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        c1 = memory.record_candidate(task.task_id, artifact="v1")
        c2 = memory.record_candidate(task.task_id, artifact="v2")

        p1 = memory.promote_candidate(c1.candidate_id, promotion_key="test_key")
        p2 = memory.promote_candidate(c2.candidate_id, promotion_key="test_key")
        assert p1.version == 1
        assert p2.version == 2


class TestCompilerMemoryKnowledge:
    """Test knowledge management (L2)."""

    def test_store_and_retrieve(self, memory: CompilerMemory) -> None:
        item = memory.store_knowledge(
            kind=KnowledgeKind.HARDWARE_RULE,
            summary="On H100, use WGMMA for matmul",
            scope_kind=ScopeKind.HARDWARE_FAMILY,
            scope_key="hopper",
            source="autocomp",
        )
        assert item.knowledge_id

        results = memory.retrieve_knowledge(
            kind=KnowledgeKind.HARDWARE_RULE,
            scope_kind=ScopeKind.HARDWARE_FAMILY,
            scope_key="hopper",
        )
        assert len(results) == 1
        assert results[0].summary == "On H100, use WGMMA for matmul"

    def test_retrieve_similar(self, memory: CompilerMemory) -> None:
        memory.store_knowledge(
            kind=KnowledgeKind.OPTIMIZATION_TACTIC,
            summary="Tile matmul by 128",
            scope_kind=ScopeKind.OPERATOR_FAMILY,
            scope_key="matmul",
        )
        memory.store_knowledge(
            kind=KnowledgeKind.HARDWARE_RULE,
            summary="Use shared memory for reductions",
            scope_kind=ScopeKind.HARDWARE_FAMILY,
            scope_key="cuda",
        )

        results = memory.retrieve_similar(op_family="matmul", hardware_signature="cuda")
        assert len(results) >= 2

    def test_knowledge_use_tracking(self, memory: CompilerMemory) -> None:
        item = memory.store_knowledge(
            kind=KnowledgeKind.SCHEDULE_TEMPLATE,
            summary="Tiled matmul schedule",
        )
        memory.record_knowledge_use(item.knowledge_id, won=True)
        memory.record_knowledge_use(item.knowledge_id, won=False)

        results = memory.retrieve_knowledge(kind=KnowledgeKind.SCHEDULE_TEMPLATE)
        assert results[0].uses == 2
        assert results[0].wins == 1
        assert results[0].failures == 1


class TestCompilerMemoryReplay:
    """Test replay buffer (L0/L1)."""

    def test_record_and_replay(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)

        memory.record_episode_step(task.task_id, action="tile", reward=0.5, step_number=0)
        memory.record_episode_step(task.task_id, action="vectorize", reward=0.3, step_number=1)
        memory.record_episode_step(task.task_id, action="promote", reward=1.0, step_number=2)

        steps = memory.replay_task(task.task_id)
        assert len(steps) == 3
        assert steps[0].action == "tile"
        assert steps[2].reward == 1.0


class TestCompilerMemoryProviderIngestion:
    """Test provider knowledge ingestion."""

    def test_ingest_exports(self, memory: CompilerMemory) -> None:
        count = memory.ingest_provider_knowledge(
            provider_name="autocomp",
            exports=[
                {"kind": "schedule_template", "scope": "operator_family", "scope_key": "matmul",
                 "content": "def schedule(): ...", "summary": "Tiled matmul for H100"},
                {"kind": "hardware_rule", "scope": "hardware_family", "scope_key": "hopper",
                 "content": "", "summary": "WGMMA requires specific tile alignment"},
            ],
        )
        assert count == 2

        results = memory.retrieve_knowledge(kind=KnowledgeKind.SCHEDULE_TEMPLATE)
        assert len(results) == 1
        assert "autocomp" in results[0].source


class TestCompilerMemoryStats:
    """Test stats."""

    def test_empty_stats(self, memory: CompilerMemory) -> None:
        stats = memory.stats()
        assert stats["tasks"] == 0
        assert stats["candidates"] == 0

    def test_stats_after_operations(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        memory.record_candidate(task.task_id, artifact="code")
        memory.store_knowledge(KnowledgeKind.HARDWARE_RULE, "rule")

        stats = memory.stats()
        assert stats["tasks"] == 1
        assert stats["candidates"] == 1
        assert stats["knowledge_items"] == 1
        assert stats["blobs"] >= 1
