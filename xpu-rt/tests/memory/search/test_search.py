"""Tests for the search infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.memory.schema import (
    GeneratorKind,
    KnowledgeKind,
    ObjectKind,
    ScopeKind,
)
from compgen.memory.search.frontier import SearchFrontier
from compgen.memory.search.promote import SearchPromoter
from compgen.memory.search.replay import ReplayBuffer
from compgen.memory.search.retrieve import SearchRetriever
from compgen.memory.search.scorer import score_kernel, score_pass
from compgen.memory.search.task import SearchTask
from compgen.memory.store import CompilerMemory


@pytest.fixture
def memory(tmp_path: Path) -> CompilerMemory:
    return CompilerMemory(db_path=tmp_path / "test.db", blob_root=tmp_path / "blobs")


class TestSearchTask:
    def test_for_kernel(self) -> None:
        task = SearchTask.for_kernel("t1", op_family="matmul", shapes="1024x1024", hardware="h100")
        assert task.kind == ObjectKind.KERNEL
        assert task.state.op_family == "matmul"
        assert task.state.hardware_signature == "h100"

    def test_for_pass(self) -> None:
        task = SearchTask.for_pass("t2", op_family="linalg.matmul", bottleneck="compute_bound")
        assert task.kind == ObjectKind.PASS
        assert task.state.bottleneck_signature == "compute_bound"


class TestScorer:
    def test_score_correct_kernel(self) -> None:
        from compgen.memory.schema import Evaluation

        eval_ = Evaluation(
            eval_id="e1",
            candidate_id="c1",
            compile_ok=True,
            correctness_ok=True,
            perf_ok=True,
            score=0.8,
            latency_us=100.0,
        )
        breakdown = score_kernel(eval_)
        assert breakdown.total > 0
        assert breakdown.correctness_gate == 1.0

    def test_score_incorrect_kernel(self) -> None:
        from compgen.memory.schema import Evaluation

        eval_ = Evaluation(
            eval_id="e1",
            candidate_id="c1",
            compile_ok=True,
            correctness_ok=False,
            score=0.8,
            latency_us=100.0,
        )
        breakdown = score_kernel(eval_)
        assert breakdown.total == 0.0  # gated by correctness

    def test_score_pass(self) -> None:
        from compgen.memory.schema import Evaluation

        eval_ = Evaluation(
            eval_id="e1",
            candidate_id="c1",
            compile_ok=True,
            correctness_ok=True,
            score=0.5,
            verifier_summary="valid",
        )
        breakdown = score_pass(eval_)
        assert breakdown.total > 0
        assert breakdown.proof_bonus > 0


class TestSearchRetriever:
    def test_retrieve_empty(self, memory: CompilerMemory) -> None:
        retriever = SearchRetriever(memory)
        task = SearchTask.for_kernel("t1", op_family="matmul")
        result = retriever.retrieve_for_task(task)
        assert result.is_empty

    def test_retrieve_with_knowledge(self, memory: CompilerMemory) -> None:
        memory.store_knowledge(
            kind=KnowledgeKind.SCHEDULE_TEMPLATE,
            summary="Tiled matmul",
            artifact="def schedule(): ...",
            scope_kind=ScopeKind.OPERATOR_FAMILY,
            scope_key="matmul",
        )
        memory.store_knowledge(
            kind=KnowledgeKind.HARDWARE_RULE,
            summary="Use WGMMA",
            scope_kind=ScopeKind.HARDWARE_FAMILY,
            scope_key="h100",
        )

        retriever = SearchRetriever(memory)
        task = SearchTask.for_kernel("t1", op_family="matmul", hardware="h100")
        result = retriever.retrieve_for_task(task)
        assert not result.is_empty
        assert result.total >= 2


class TestReplayBuffer:
    def test_record_and_replay(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        buffer = ReplayBuffer(memory)

        buffer.record_step(task.task_id, action="tile", reward=0.5, step_number=0)
        buffer.record_step(task.task_id, action="vectorize", reward=0.8, step_number=1)

        traj = buffer.replay(task.task_id)
        assert traj.length == 2
        assert traj.total_reward == pytest.approx(1.3)
        assert traj.best_reward == pytest.approx(0.8)

    def test_best_trajectory(self, memory: CompilerMemory) -> None:
        # Create two kernel tasks with different rewards
        t1 = memory.create_task(ObjectKind.KERNEL)
        t2 = memory.create_task(ObjectKind.KERNEL)
        buffer = ReplayBuffer(memory)

        buffer.record_step(t1.task_id, action="tile", reward=0.3)
        buffer.record_step(t2.task_id, action="tile", reward=0.9)

        best = buffer.best_trajectory_for_kind("kernel")
        assert len(best) == 2
        assert best[0].total_reward > best[1].total_reward


class TestSearchFrontier:
    def test_empty_frontier(self, memory: CompilerMemory) -> None:
        frontier = SearchFrontier(memory)
        assert frontier.next_task() is None
        assert frontier.size == 0

    def test_ucb1_selection(self, memory: CompilerMemory) -> None:
        frontier = SearchFrontier(memory)
        t1 = SearchTask.for_kernel("t1", op_family="matmul")
        t2 = SearchTask.for_kernel("t2", op_family="conv2d")

        frontier.add_task(t1)
        frontier.add_task(t2)

        # First pulls should go to unpulled tasks
        first = frontier.next_task()
        assert first is not None
        frontier.update(first.task_id, reward=0.5)

        second = frontier.next_task()
        assert second is not None
        assert second.task_id != first.task_id  # picks the other unpulled one
        frontier.update(second.task_id, reward=0.1)

        # After both pulled, UCB1 should favor the one with higher reward
        third = frontier.next_task()
        assert third is not None

    def test_remove_task(self, memory: CompilerMemory) -> None:
        frontier = SearchFrontier(memory)
        t1 = SearchTask.for_kernel("t1", op_family="matmul")
        frontier.add_task(t1)
        assert frontier.size == 1
        frontier.remove_task("t1")
        assert frontier.size == 0


class TestSearchPromoter:
    def test_promote_best(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        c1 = memory.record_candidate(task.task_id, artifact="kernel_v1", generator_kind=GeneratorKind.LLM)
        memory.record_evaluation(c1.candidate_id, correctness_ok=True, score=0.5, latency_us=100.0)

        c2 = memory.record_candidate(task.task_id, artifact="kernel_v2", generator_kind=GeneratorKind.LLM)
        memory.record_evaluation(c2.candidate_id, correctness_ok=True, score=0.9, latency_us=50.0)

        promoter = SearchPromoter(memory)
        promo = promoter.promote_best(task.task_id)
        assert promo is not None
        assert promo.measured_gain == pytest.approx(0.9)

    def test_no_candidate_qualifies(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        c1 = memory.record_candidate(task.task_id, artifact="bad_kernel")
        memory.record_evaluation(c1.candidate_id, correctness_ok=False, score=0.0)

        promoter = SearchPromoter(memory)
        promo = promoter.promote_best(task.task_id)
        assert promo is None

    def test_extract_knowledge(self, memory: CompilerMemory) -> None:
        task = memory.create_task(ObjectKind.KERNEL)
        c = memory.record_candidate(task.task_id, artifact="def kernel(): pass")
        memory.record_evaluation(c.candidate_id, correctness_ok=True, score=0.8, latency_us=50.0)
        memory.promote_candidate(c.candidate_id, promotion_key="test")

        promoter = SearchPromoter(memory)
        items = promoter.extract_knowledge(task.task_id, task_kind="kernel", op_family="matmul")
        assert len(items) >= 1
        assert "matmul" in items[0].scope_key

    def test_update_retrieval_stats(self, memory: CompilerMemory) -> None:
        item = memory.store_knowledge(KnowledgeKind.OPTIMIZATION_TACTIC, "test tactic")
        promoter = SearchPromoter(memory)
        promoter.update_retrieval_stats([item.knowledge_id], task_succeeded=True)

        updated = memory.retrieve_knowledge(kind=KnowledgeKind.OPTIMIZATION_TACTIC)
        assert updated[0].wins == 1
