"""Unified Compiler Memory interface.

Single entry point for all memory operations across the four levels:
    L0 Episode → L1 Replay → L2 Knowledge → L3 Promoted

All generated things (kernels, passes, rewrites, guards, decompositions,
translations, backend plans, schedules) use this interface.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from compgen.memory.blobs import BlobStore
from compgen.memory.schema import (
    Candidate,
    CandidateStatus,
    EpisodeStep,
    Evaluation,
    GeneratorKind,
    KnowledgeItem,
    KnowledgeKind,
    ObjectKind,
    Promotion,
    ScopeKind,
    StateSignature,
    Task,
)
from compgen.memory.sqlite_backend import SQLiteBackend

log = structlog.get_logger()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CompilerMemory:
    """Single entry point for all memory operations.

    Combines a content-addressed blob store (Layer A) with a SQLite
    metadata store (Layer B) to provide the full lifecycle for all
    generated compiler artifacts.

    Attributes:
        db: SQLite backend for relational metadata.
        blobs: Content-addressed blob store for artifacts.
    """

    def __init__(
        self,
        db_path: Path = Path(".compgen_cache/memory.db"),
        blob_root: Path = Path(".compgen_cache/blobs"),
        embedding_provider: Any = None,
    ) -> None:
        self.db = SQLiteBackend(db_path)
        self.blobs = BlobStore(blob_root)
        self.embedding_provider = embedding_provider  # Optional EmbeddingProvider for semantic retrieval

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def create_task(
        self,
        kind: ObjectKind,
        workload_key: str = "",
        region_key: str = "",
        target_key: str = "",
        hardware_key: str = "",
        objective: str = "latency",
        input_artifact: str = "",
    ) -> Task:
        """Create a new optimization task."""
        task_id = _uid()
        input_hash = self.blobs.store(input_artifact) if input_artifact else ""
        task = Task(
            task_id=task_id,
            task_kind=kind,
            workload_key=workload_key,
            region_key=region_key,
            target_key=target_key,
            hardware_key=hardware_key,
            objective=objective,
            input_artifact_hash=input_hash,
            created_at=_now(),
        )
        self.db.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                task.task_id,
                task.task_kind.value,
                task.workload_key,
                task.region_key,
                task.target_key,
                task.hardware_key,
                task.objective,
                task.input_artifact_hash,
                task.created_at,
            ),
        )
        self.db.commit()
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task by ID."""
        row = self.db.fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if row is None:
            return None
        return Task(
            task_id=row["task_id"],
            task_kind=ObjectKind(row["task_kind"]),
            workload_key=row["workload_key"],
            region_key=row["region_key"],
            target_key=row["target_key"],
            hardware_key=row["hardware_key"],
            objective=row["objective"],
            input_artifact_hash=row["input_artifact_hash"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # State signatures
    # ------------------------------------------------------------------

    def record_state(
        self,
        task_id: str,
        op_family: str = "",
        shape_signature: str = "",
        dtype_signature: str = "",
        layout_signature: str = "",
        hardware_signature: str = "",
        bottleneck_signature: str = "",
    ) -> StateSignature:
        """Record a state signature for retrieval."""
        state = StateSignature(
            state_id=_uid(),
            task_id=task_id,
            op_family=op_family,
            shape_signature=shape_signature,
            dtype_signature=dtype_signature,
            layout_signature=layout_signature,
            hardware_signature=hardware_signature,
            bottleneck_signature=bottleneck_signature,
        )
        self.db.execute(
            "INSERT INTO state_signatures VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                state.state_id,
                state.task_id,
                state.op_family,
                state.shape_signature,
                state.dtype_signature,
                state.layout_signature,
                state.hardware_signature,
                state.memory_signature,
                state.config_signature,
                state.bottleneck_signature,
                state.profile_signature,
            ),
        )
        self.db.commit()
        return state

    # ------------------------------------------------------------------
    # Candidate lifecycle
    # ------------------------------------------------------------------

    def record_candidate(
        self,
        task_id: str,
        artifact: str,
        generator_kind: GeneratorKind = GeneratorKind.LLM,
        generator_model: str = "",
        generation_round: int = 0,
        parent_candidate_id: str = "",
        state_signature_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Candidate:
        """Record a new candidate generated during search."""
        artifact_hash = self.blobs.store(artifact)
        candidate = Candidate(
            candidate_id=_uid(),
            task_id=task_id,
            artifact_hash=artifact_hash,
            parent_candidate_id=parent_candidate_id,
            generator_kind=generator_kind,
            generator_model=generator_model,
            generation_round=generation_round,
            state_signature_id=state_signature_id,
            status=CandidateStatus.NEW,
            created_at=_now(),
            metadata=metadata or {},
        )
        self.db.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                candidate.candidate_id,
                candidate.task_id,
                candidate.artifact_hash,
                candidate.parent_candidate_id,
                candidate.generator_kind.value,
                candidate.generator_model,
                candidate.generation_round,
                candidate.state_signature_id,
                candidate.status.value,
                candidate.created_at,
                json.dumps(candidate.metadata),
            ),
        )
        self.db.commit()
        return candidate

    def update_candidate_status(self, candidate_id: str, status: CandidateStatus) -> None:
        """Update a candidate's lifecycle status."""
        self.db.execute(
            "UPDATE candidates SET status = ? WHERE candidate_id = ?",
            (status.value, candidate_id),
        )
        self.db.commit()

    def get_candidates(self, task_id: str, status: CandidateStatus | None = None) -> list[Candidate]:
        """Get candidates for a task, optionally filtered by status."""
        if status is not None:
            rows = self.db.fetchall(
                "SELECT * FROM candidates WHERE task_id = ? AND status = ? ORDER BY generation_round",
                (task_id, status.value),
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM candidates WHERE task_id = ? ORDER BY generation_round",
                (task_id,),
            )
        return [self._row_to_candidate(r) for r in rows]

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def record_evaluation(
        self,
        candidate_id: str,
        compile_ok: bool = False,
        correctness_ok: bool = False,
        perf_ok: bool = False,
        score: float = 0.0,
        latency_us: float = 0.0,
        verifier_summary: str = "",
        profile_summary: str = "",
        backend: str = "",
    ) -> Evaluation:
        """Record an evaluation of a candidate."""
        evaluation = Evaluation(
            eval_id=_uid(),
            candidate_id=candidate_id,
            backend=backend,
            compile_ok=compile_ok,
            correctness_ok=correctness_ok,
            perf_ok=perf_ok,
            score=score,
            latency_us=latency_us,
            verifier_summary=verifier_summary,
            profile_summary=profile_summary,
            created_at=_now(),
        )
        self.db.execute(
            "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                evaluation.eval_id,
                evaluation.candidate_id,
                evaluation.backend,
                int(evaluation.compile_ok),
                int(evaluation.correctness_ok),
                int(evaluation.perf_ok),
                evaluation.score,
                evaluation.latency_us,
                evaluation.throughput,
                evaluation.energy,
                evaluation.verifier_summary,
                evaluation.profile_summary,
                evaluation.profile_artifact_hash,
                evaluation.created_at,
            ),
        )
        self.db.commit()
        return evaluation

    def get_evaluations(self, candidate_id: str) -> list[Evaluation]:
        """Get all evaluations for a candidate."""
        rows = self.db.fetchall(
            "SELECT * FROM evaluations WHERE candidate_id = ?",
            (candidate_id,),
        )
        return [self._row_to_evaluation(r) for r in rows]

    # ------------------------------------------------------------------
    # Knowledge management (L2)
    # ------------------------------------------------------------------

    def store_knowledge(
        self,
        kind: KnowledgeKind,
        summary: str,
        artifact: str = "",
        scope_kind: ScopeKind = ScopeKind.GLOBAL,
        scope_key: str = "",
        source: str = "",
    ) -> KnowledgeItem:
        """Store a reusable knowledge item."""
        artifact_hash = self.blobs.store(artifact) if artifact else ""
        item = KnowledgeItem(
            knowledge_id=_uid(),
            knowledge_kind=kind,
            scope_kind=scope_kind,
            scope_key=scope_key,
            summary=summary,
            artifact_hash=artifact_hash,
            source=source,
        )
        self.db.execute(
            "INSERT INTO knowledge_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                item.knowledge_id,
                item.knowledge_kind.value,
                item.scope_kind.value,
                item.scope_key,
                item.summary,
                item.artifact_hash,
                item.quality_score,
                item.uses,
                item.wins,
                item.failures,
                item.last_used_at,
                item.source,
                item.embedding_hash,
            ),
        )
        self.db.commit()
        return item

    def retrieve_knowledge(
        self,
        kind: KnowledgeKind | None = None,
        scope_kind: ScopeKind | None = None,
        scope_key: str = "",
        top_k: int = 10,
    ) -> list[KnowledgeItem]:
        """Retrieve knowledge items by kind/scope."""
        conditions: list[str] = []
        params: list[Any] = []

        if kind is not None:
            conditions.append("knowledge_kind = ?")
            params.append(kind.value)
        if scope_kind is not None:
            conditions.append("scope_kind = ?")
            params.append(scope_kind.value)
        if scope_key:
            conditions.append("scope_key = ?")
            params.append(scope_key)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM knowledge_items WHERE {where} ORDER BY quality_score DESC, wins DESC LIMIT ?"  # noqa: S608
        params.append(top_k)

        rows = self.db.fetchall(sql, tuple(params))
        return [self._row_to_knowledge(r) for r in rows]

    def retrieve_similar(
        self,
        op_family: str = "",
        hardware_signature: str = "",
        bottleneck_signature: str = "",
        top_k: int = 5,
    ) -> list[KnowledgeItem]:
        """Retrieve knowledge items similar to a state signature.

        Tries embedding-based similarity first when an embedding_provider
        is available, falling back to scope_key matching.
        """
        # Try embedding-based retrieval first (Unit 13)
        if self.embedding_provider is not None:
            try:
                from compgen.memory.embeddings import retrieve_by_similarity

                query = f"{op_family} {hardware_signature} {bottleneck_signature}".strip()
                if query:
                    results = retrieve_by_similarity(self, query, self.embedding_provider, top_k=top_k)
                    if results:
                        return results
            except Exception:
                pass  # Fall back to scope_key matching

        # Scope_key matching fallback
        results: list[KnowledgeItem] = []

        if op_family:
            results.extend(
                self.retrieve_knowledge(
                    scope_kind=ScopeKind.OPERATOR_FAMILY,
                    scope_key=op_family,
                    top_k=top_k,
                )
            )

        if hardware_signature:
            results.extend(
                self.retrieve_knowledge(
                    scope_kind=ScopeKind.HARDWARE_FAMILY,
                    scope_key=hardware_signature,
                    top_k=top_k,
                )
            )

        # Also fetch global knowledge
        results.extend(
            self.retrieve_knowledge(
                scope_kind=ScopeKind.GLOBAL,
                top_k=top_k,
            )
        )

        # Deduplicate and return top-k by quality
        seen: set[str] = set()
        unique: list[KnowledgeItem] = []
        for item in results:
            if item.knowledge_id not in seen:
                seen.add(item.knowledge_id)
                unique.append(item)

        unique.sort(key=lambda x: (x.quality_score, x.wins), reverse=True)
        return unique[:top_k]

    def record_knowledge_use(self, knowledge_id: str, won: bool) -> None:
        """Record that a knowledge item was used (and whether it helped)."""
        if won:
            self.db.execute(
                "UPDATE knowledge_items SET uses = uses + 1, wins = wins + 1, last_used_at = ? WHERE knowledge_id = ?",
                (_now(), knowledge_id),
            )
        else:
            self.db.execute(
                "UPDATE knowledge_items SET uses = uses + 1, failures = failures + 1, last_used_at = ? WHERE knowledge_id = ?",
                (_now(), knowledge_id),
            )
        self.db.commit()

    # ------------------------------------------------------------------
    # Promotion (L3)
    # ------------------------------------------------------------------

    def promote_candidate(
        self,
        candidate_id: str,
        promotion_key: str = "",
        reason: str = "",
        measured_gain: float = 0.0,
        verified_by: str = "",
        region_signature: str = "",
        contract_hash: str = "",
        gate_level: str = "",
    ) -> Promotion:
        """Promote a candidate to the immutable L3 library.

        ``region_signature`` and ``contract_hash`` (M-26) ride along as
        the two-tier cache-key index so future runs can locate the
        recipe by region pattern across models. ``gate_level`` (M-29)
        records the highest promotion gate the bundle satisfied.
        """
        # Find next version for this promotion key
        row = self.db.fetchone(
            "SELECT MAX(version) as v FROM promotions WHERE promotion_key = ?",
            (promotion_key,),
        )
        version = (row["v"] or 0) + 1

        promotion = Promotion(
            promotion_id=_uid(),
            candidate_id=candidate_id,
            promotion_key=promotion_key,
            version=version,
            reason=reason,
            measured_gain=measured_gain,
            verified_by=verified_by,
            created_at=_now(),
            region_signature=region_signature,
            contract_hash=contract_hash,
            gate_level=gate_level,
        )
        # Explicit column names: the table now has 11 columns after the
        # M-26 / M-29 migrations and a positional VALUES list would
        # silently shift values into the wrong columns on schema drift.
        self.db.execute(
            (
                "INSERT INTO promotions ("
                "promotion_id, candidate_id, promotion_key, version, reason, "
                "measured_gain, verified_by, created_at, "
                "region_signature, contract_hash, gate_level"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            ),
            (
                promotion.promotion_id,
                promotion.candidate_id,
                promotion.promotion_key,
                promotion.version,
                promotion.reason,
                promotion.measured_gain,
                promotion.verified_by,
                promotion.created_at,
                promotion.region_signature,
                promotion.contract_hash,
                promotion.gate_level,
            ),
        )
        self.update_candidate_status(candidate_id, CandidateStatus.PROMOTED)
        self.db.commit()
        return promotion

    # ------------------------------------------------------------------
    # Replay buffer (L0/L1)
    # ------------------------------------------------------------------

    def record_episode_step(
        self,
        task_id: str,
        action: str = "",
        reward: float = 0.0,
        candidate_id: str = "",
        step_number: int = 0,
        metadata: dict[str, str] | None = None,
    ) -> EpisodeStep:
        """Record one step in a search episode."""
        step = EpisodeStep(
            step_id=_uid(),
            task_id=task_id,
            candidate_id=candidate_id,
            action=action,
            reward=reward,
            step_number=step_number,
            metadata=metadata or {},
            created_at=_now(),
        )
        self.db.execute(
            "INSERT INTO episode_steps VALUES (?,?,?,?,?,?,?,?)",
            (
                step.step_id,
                step.task_id,
                step.candidate_id,
                step.action,
                step.reward,
                step.step_number,
                json.dumps(step.metadata),
                step.created_at,
            ),
        )
        self.db.commit()
        return step

    def replay_task(self, task_id: str) -> list[EpisodeStep]:
        """Replay all steps for a task."""
        rows = self.db.fetchall(
            "SELECT * FROM episode_steps WHERE task_id = ? ORDER BY step_number",
            (task_id,),
        )
        return [
            EpisodeStep(
                step_id=r["step_id"],
                task_id=r["task_id"],
                candidate_id=r["candidate_id"],
                action=r["action"],
                reward=r["reward"],
                step_number=r["step_number"],
                metadata=json.loads(r["metadata_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Provider knowledge ingestion
    # ------------------------------------------------------------------

    def ingest_provider_knowledge(
        self,
        provider_name: str,
        exports: list[dict[str, Any]],
    ) -> int:
        """Ingest knowledge exports from a kernel provider.

        Args:
            provider_name: Name of the provider (e.g., "autocomp", "kernelblaster").
            exports: List of knowledge export dicts from the provider.

        Returns:
            Number of items ingested.
        """
        count = 0
        for export in exports:
            kind_str = export.get("kind", "optimization_tactic")
            try:
                kind = KnowledgeKind(kind_str)
            except ValueError:
                kind = KnowledgeKind.OPTIMIZATION_TACTIC

            scope_str = export.get("scope", "global")
            try:
                scope = ScopeKind(scope_str)
            except ValueError:
                scope = ScopeKind.GLOBAL

            self.store_knowledge(
                kind=kind,
                summary=export.get("summary", ""),
                artifact=export.get("content", ""),
                scope_kind=scope,
                scope_key=export.get("scope_key", ""),
                source=f"provider:{provider_name}",
            )
            count += 1

        log.info("memory.ingest_provider", provider=provider_name, count=count)
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Get summary statistics."""
        return {
            "tasks": self.db.table_count("tasks"),
            "candidates": self.db.table_count("candidates"),
            "evaluations": self.db.table_count("evaluations"),
            "knowledge_items": self.db.table_count("knowledge_items"),
            "promotions": self.db.table_count("promotions"),
            "episode_steps": self.db.table_count("episode_steps"),
            "blobs": self.blobs.count(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_candidate(row: Any) -> Candidate:
        return Candidate(
            candidate_id=row["candidate_id"],
            task_id=row["task_id"],
            artifact_hash=row["artifact_hash"],
            parent_candidate_id=row["parent_candidate_id"],
            generator_kind=GeneratorKind(row["generator_kind"]),
            generator_model=row["generator_model"],
            generation_round=row["generation_round"],
            state_signature_id=row["state_signature_id"],
            status=CandidateStatus(row["status"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_evaluation(row: Any) -> Evaluation:
        return Evaluation(
            eval_id=row["eval_id"],
            candidate_id=row["candidate_id"],
            backend=row["backend"],
            compile_ok=bool(row["compile_ok"]),
            correctness_ok=bool(row["correctness_ok"]),
            perf_ok=bool(row["perf_ok"]),
            score=row["score"],
            latency_us=row["latency_us"],
            verifier_summary=row["verifier_summary"],
            profile_summary=row["profile_summary"],
            profile_artifact_hash=row["profile_artifact_hash"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_knowledge(row: Any) -> KnowledgeItem:
        return KnowledgeItem(
            knowledge_id=row["knowledge_id"],
            knowledge_kind=KnowledgeKind(row["knowledge_kind"]),
            scope_kind=ScopeKind(row["scope_kind"]),
            scope_key=row["scope_key"],
            summary=row["summary"],
            artifact_hash=row["artifact_hash"],
            quality_score=row["quality_score"],
            uses=row["uses"],
            wins=row["wins"],
            failures=row["failures"],
            last_used_at=row["last_used_at"],
            source=row["source"],
            embedding_hash=row["embedding_hash"] if "embedding_hash" in row.keys() else "",
        )


__all__ = ["CompilerMemory"]
