"""Tests for candidate lineage graph querying."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.memory.schema import CandidateStatus, GeneratorKind, ObjectKind
from compgen.memory.store import CompilerMemory
from compgen.promotion.lineage import (
    LineageGraph,
    LineageNode,
    build_lineage_graph,
    find_lineage_siblings,
    get_promotion_history,
)


@pytest.fixture
def memory(tmp_path: Path) -> CompilerMemory:
    """Create a CompilerMemory in a temp directory."""
    return CompilerMemory(
        db_path=tmp_path / "test.db",
        blob_root=tmp_path / "blobs",
    )


class TestBuildLineageGraph:
    """Tests for build_lineage_graph."""

    def test_three_node_chain(self, memory: CompilerMemory) -> None:
        """Build a grandparent -> parent -> child chain and verify order."""
        task = memory.create_task(ObjectKind.KERNEL, workload_key="matmul")

        grandparent = memory.record_candidate(
            task.task_id,
            artifact="gp_code",
            generator_kind=GeneratorKind.TEMPLATE,
            generation_round=0,
        )
        parent = memory.record_candidate(
            task.task_id,
            artifact="parent_code",
            generator_kind=GeneratorKind.MUTATION,
            generation_round=1,
            parent_candidate_id=grandparent.candidate_id,
        )
        child = memory.record_candidate(
            task.task_id,
            artifact="child_code",
            generator_kind=GeneratorKind.LLM,
            generation_round=2,
            parent_candidate_id=parent.candidate_id,
        )

        graph = build_lineage_graph(memory, child.candidate_id)

        assert len(graph.nodes) == 3
        assert graph.root_id == grandparent.candidate_id

        # Root-to-leaf order
        assert graph.nodes[0].candidate_id == grandparent.candidate_id
        assert graph.nodes[1].candidate_id == parent.candidate_id
        assert graph.nodes[2].candidate_id == child.candidate_id

        # Parent links
        assert graph.nodes[0].parent_id is None
        assert graph.nodes[1].parent_id == grandparent.candidate_id
        assert graph.nodes[2].parent_id == parent.candidate_id

        # Generator kinds
        assert graph.nodes[0].generator_kind == "template"
        assert graph.nodes[1].generator_kind == "mutation"
        assert graph.nodes[2].generator_kind == "llm"

        # No promotions yet
        for node in graph.nodes:
            assert node.promotion_key is None

    def test_single_node(self, memory: CompilerMemory) -> None:
        """A candidate with no parent returns a single-node graph."""
        task = memory.create_task(ObjectKind.KERNEL)
        candidate = memory.record_candidate(task.task_id, artifact="solo")

        graph = build_lineage_graph(memory, candidate.candidate_id)

        assert len(graph.nodes) == 1
        assert graph.root_id == candidate.candidate_id
        assert graph.nodes[0].parent_id is None

    def test_nonexistent_candidate(self, memory: CompilerMemory) -> None:
        """Querying a nonexistent candidate returns an empty graph."""
        graph = build_lineage_graph(memory, "nonexistent_id")

        assert len(graph.nodes) == 0
        assert graph.root_id == ""

    def test_promoted_child_appears_in_lineage(self, memory: CompilerMemory) -> None:
        """Promote the child and verify promotion_key appears in its node."""
        task = memory.create_task(ObjectKind.KERNEL, workload_key="conv2d")

        parent = memory.record_candidate(
            task.task_id,
            artifact="parent_kernel",
            generation_round=0,
        )
        child = memory.record_candidate(
            task.task_id,
            artifact="child_kernel",
            generation_round=1,
            parent_candidate_id=parent.candidate_id,
        )
        memory.promote_candidate(
            child.candidate_id,
            promotion_key="conv2d_h100_latency",
            reason="best candidate",
        )

        graph = build_lineage_graph(memory, child.candidate_id)

        assert len(graph.nodes) == 2
        # Parent has no promotion
        assert graph.nodes[0].promotion_key is None
        # Child has promotion
        assert graph.nodes[1].promotion_key == "conv2d_h100_latency"
        assert graph.nodes[1].status == CandidateStatus.PROMOTED.value


class TestGetPromotionHistory:
    """Tests for get_promotion_history."""

    def test_multiple_versions(self, memory: CompilerMemory) -> None:
        """Promote two candidates under the same key and verify history."""
        task = memory.create_task(ObjectKind.KERNEL)
        c1 = memory.record_candidate(task.task_id, artifact="v1_code")
        c2 = memory.record_candidate(task.task_id, artifact="v2_code")

        memory.promote_candidate(c1.candidate_id, promotion_key="matmul_key", reason="initial")
        memory.promote_candidate(c2.candidate_id, promotion_key="matmul_key", reason="improved")

        history = get_promotion_history(memory, "matmul_key")

        assert len(history) == 2
        assert history[0].version == 1
        assert history[0].reason == "initial"
        assert history[1].version == 2
        assert history[1].reason == "improved"

    def test_empty_history(self, memory: CompilerMemory) -> None:
        """A key with no promotions returns an empty list."""
        history = get_promotion_history(memory, "nonexistent_key")
        assert history == []

    def test_single_promotion(self, memory: CompilerMemory) -> None:
        """A key with one promotion returns a single-element list."""
        task = memory.create_task(ObjectKind.KERNEL)
        c = memory.record_candidate(task.task_id, artifact="code")
        memory.promote_candidate(c.candidate_id, promotion_key="solo_key")

        history = get_promotion_history(memory, "solo_key")
        assert len(history) == 1
        assert history[0].version == 1


class TestFindLineageSiblings:
    """Tests for find_lineage_siblings."""

    def test_two_siblings(self, memory: CompilerMemory) -> None:
        """Two candidates sharing the same parent are siblings."""
        task = memory.create_task(ObjectKind.KERNEL)
        parent = memory.record_candidate(task.task_id, artifact="parent")
        sib_a = memory.record_candidate(
            task.task_id,
            artifact="sibling_a",
            parent_candidate_id=parent.candidate_id,
            generation_round=1,
        )
        sib_b = memory.record_candidate(
            task.task_id,
            artifact="sibling_b",
            parent_candidate_id=parent.candidate_id,
            generation_round=2,
        )

        siblings_of_a = find_lineage_siblings(memory, sib_a.candidate_id)
        assert len(siblings_of_a) == 1
        assert siblings_of_a[0].candidate_id == sib_b.candidate_id

        siblings_of_b = find_lineage_siblings(memory, sib_b.candidate_id)
        assert len(siblings_of_b) == 1
        assert siblings_of_b[0].candidate_id == sib_a.candidate_id

    def test_no_parent_returns_empty(self, memory: CompilerMemory) -> None:
        """A root candidate (no parent) has no siblings."""
        task = memory.create_task(ObjectKind.KERNEL)
        root = memory.record_candidate(task.task_id, artifact="root")

        assert find_lineage_siblings(memory, root.candidate_id) == []

    def test_nonexistent_candidate_returns_empty(self, memory: CompilerMemory) -> None:
        """A nonexistent candidate returns no siblings."""
        assert find_lineage_siblings(memory, "nonexistent") == []

    def test_only_child_returns_empty(self, memory: CompilerMemory) -> None:
        """A candidate whose parent has no other children returns empty."""
        task = memory.create_task(ObjectKind.KERNEL)
        parent = memory.record_candidate(task.task_id, artifact="parent")
        child = memory.record_candidate(
            task.task_id,
            artifact="only_child",
            parent_candidate_id=parent.candidate_id,
        )

        assert find_lineage_siblings(memory, child.candidate_id) == []
