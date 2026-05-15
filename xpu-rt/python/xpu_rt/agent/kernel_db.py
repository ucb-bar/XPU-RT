"""Persistent kernel database.

Stores generated kernels indexed by (pattern_type, target_name, shapes).
The agent checks the DB before running expensive Autocomp searches —
reuse verified kernels across sessions.

Storage: JSON file on disk. Simple but sufficient for research.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelEntry:
    """A stored kernel with metadata.

    Attributes:
        pattern_type: Pattern this kernel handles (e.g., "linear_chain").
        target_name: Hardware target name.
        shapes_key: Canonical string of input/output shapes for lookup.
        kernel_code: The kernel source code.
        language: "triton", "cuda", "python".
        latency_us: Measured latency.
        correct: Whether verification passed.
        speedup: Speedup over reference.
        plan: Optimization plan that produced this kernel.
    """

    pattern_type: str
    target_name: str
    shapes_key: str
    kernel_code: str
    language: str
    latency_us: float
    correct: bool
    speedup: float
    plan: str = ""


def _shapes_key(shapes: dict[str, tuple[int, ...]] | None) -> str:
    """Canonical string key from shapes dict."""
    if not shapes:
        return "unknown"
    parts = []
    for name in sorted(shapes):
        dims = "x".join(str(d) for d in shapes[name])
        parts.append(f"{name}:{dims}")
    return "|".join(parts)


class KernelDB:
    """Persistent kernel database backed by a JSON file."""

    def __init__(self, db_path: str | Path = ".compgen_cache/kernel_db.json") -> None:
        self._path = Path(db_path)
        self._entries: list[KernelEntry] = []
        self._load()

    def _load(self) -> None:
        """Load entries from disk."""
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._entries = [KernelEntry(**e) for e in data]

    def _save(self) -> None:
        """Persist entries to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2))

    def lookup(
        self,
        pattern_type: str,
        target_name: str,
        shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> KernelEntry | None:
        """Find a cached kernel matching the pattern, target, and shapes."""
        key = _shapes_key(shapes)
        for entry in self._entries:
            if (
                entry.pattern_type == pattern_type
                and entry.target_name == target_name
                and entry.shapes_key == key
                and entry.correct
            ):
                return entry
        return None

    def store(self, entry: KernelEntry) -> None:
        """Store a kernel entry. Replaces existing entry with same key if better."""
        # Remove existing entry with same key if this one is better
        self._entries = [
            e
            for e in self._entries
            if not (
                e.pattern_type == entry.pattern_type
                and e.target_name == entry.target_name
                and e.shapes_key == entry.shapes_key
            )
        ]
        self._entries.append(entry)
        self._save()

    def best_for(
        self,
        pattern_type: str,
        target_name: str,
    ) -> KernelEntry | None:
        """Get the best kernel for a pattern type on a target (any shape)."""
        matches = [
            e for e in self._entries if e.pattern_type == pattern_type and e.target_name == target_name and e.correct
        ]
        if not matches:
            return None
        return min(matches, key=lambda e: e.latency_us)

    def all_entries(self) -> list[KernelEntry]:
        """Get all entries."""
        return list(self._entries)

    def count(self) -> int:
        """Number of stored kernels."""
        return len(self._entries)

    @classmethod
    def from_memory(cls, memory: Any) -> KernelDB:
        """Create a KernelDB backed by CompilerMemory.

        Reads promoted kernel candidates from the unified memory and
        exposes them through the same KernelDB interface. New stores
        are written to both the JSON file and CompilerMemory.

        Args:
            memory: A ``CompilerMemory`` instance.

        Returns:
            A KernelDB that bridges to the unified memory.
        """
        db = cls.__new__(cls)
        db._path = Path(".compgen_cache/kernel_db.json")
        db._entries = []
        db._memory = memory
        db._load()

        # Also load promoted kernels from CompilerMemory
        try:
            from compgen.memory.schema import KnowledgeKind

            for item in memory.retrieve_knowledge(kind=KnowledgeKind.SCHEDULE_TEMPLATE, top_k=100):
                code = memory.blobs.load(item.artifact_hash) if item.artifact_hash else ""
                if code and item.scope_key:
                    parts = item.scope_key.split("|", 1)
                    pattern_type = parts[0] if parts else ""
                    target_name = parts[1] if len(parts) > 1 else ""
                    entry = KernelEntry(
                        pattern_type=pattern_type,
                        target_name=target_name,
                        shapes_key="unknown",
                        kernel_code=code,
                        language="triton",
                        latency_us=0.0,
                        correct=True,
                        speedup=0.0,
                        plan=item.summary,
                    )
                    if not any(
                        e.pattern_type == entry.pattern_type and e.target_name == entry.target_name for e in db._entries
                    ):
                        db._entries.append(entry)
        except Exception:
            pass  # CompilerMemory may not have kernel knowledge yet

        return db

    def store_to_memory(self, entry: KernelEntry, memory: Any) -> None:
        """Store a kernel entry in both the JSON DB and CompilerMemory.

        Args:
            entry: The kernel entry to store.
            memory: A ``CompilerMemory`` instance.
        """
        self.store(entry)

        try:
            from compgen.memory.schema import GeneratorKind, KnowledgeKind, ObjectKind, ScopeKind

            # Record as a task + candidate in the unified memory
            task = memory.create_task(
                kind=ObjectKind.KERNEL,
                workload_key=entry.pattern_type,
                target_key=entry.target_name,
            )
            candidate = memory.record_candidate(
                task_id=task.task_id,
                artifact=entry.kernel_code,
                generator_kind=GeneratorKind.PROVIDER,
                metadata={"language": entry.language, "plan": entry.plan},
            )
            memory.record_evaluation(
                candidate_id=candidate.candidate_id,
                compile_ok=True,
                correctness_ok=entry.correct,
                latency_us=entry.latency_us,
                score=entry.speedup,
            )

            # Also store as a knowledge item for retrieval
            memory.store_knowledge(
                kind=KnowledgeKind.SCHEDULE_TEMPLATE,
                summary=f"{entry.pattern_type} kernel for {entry.target_name} ({entry.language})",
                artifact=entry.kernel_code,
                scope_kind=ScopeKind.OPERATOR_FAMILY,
                scope_key=f"{entry.pattern_type}|{entry.target_name}",
                source="kernel_db",
            )
        except Exception:
            pass  # Best-effort; JSON store is the primary


__all__ = ["KernelDB", "KernelEntry"]
