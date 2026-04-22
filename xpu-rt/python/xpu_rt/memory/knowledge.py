"""Hierarchical knowledge store under ``~/.compgen/knowledge/``.

Realisations from previous compile / bench / profile runs cluster into
the right scope so that any future agent run can query "what do I know
that applies to this target?" and get the most-specific applicable
lessons first.

Scope hierarchy (most-general at top, most-specific at bottom)::

    general/                                  applies to ANY compile
    backends/<backend>/general/               e.g. backends/gpu/general
    backends/<backend>/<vendor>/general/      e.g. backends/gpu/nvidia/general
    backends/<backend>/<vendor>/<arch>/       e.g. backends/gpu/nvidia/turing
    drivers/<driver>/general/                 e.g. drivers/cuda/general
    drivers/<driver>/<version>/               e.g. drivers/cuda/12.8

Each scope directory holds::

    lessons.jsonl       append-only learning log
    kernels/            best-known kernels at this scope (extends KernelStore)
    autotune/           autotune picks at this scope (extends autotune_cache)
    metrics.jsonl       observed (op, shape, dtype) → perf samples

Lesson categories:
  * ``perf``        — measured perf observation, often with evidence numbers
  * ``correctness`` — bug or numerical-stability observation
  * ``limit``       — a hardware/software ceiling we hit
  * ``design``      — architectural decision worth remembering
  * ``recipe``      — concrete code pattern that worked

A lesson added at a *general* scope applies to every descendant; a
lesson added at ``backends/gpu/nvidia/turing`` applies only when the
agent is targeting Turing. Querying by target walks the chain top-down
so the agent gets general lessons + arch-specific lessons together.

Pre-population: existing kernel store + autotune cache live as flat
``~/.compgen/kernels/`` and ``~/.compgen/autotune/`` directories. The
migration path is a one-time copy into ``backends/<backend>/.../<scope>``
based on the kernel's target_name. New writes go through the
hierarchical layout; lookups fall through to the legacy flat layout
for backward compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def default_knowledge_root() -> Path:
    """``~/.compgen/knowledge`` overridable via ``COMPGEN_KNOWLEDGE_ROOT``."""
    override = os.environ.get("COMPGEN_KNOWLEDGE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".compgen" / "knowledge"


# Canonical scope strings — the hierarchy.
SCOPE_GENERAL = "general"
SCOPE_BACKENDS = "backends"
SCOPE_DRIVERS = "drivers"


# ---------------------------------------------------------------------------
# Target → scope mapping
# ---------------------------------------------------------------------------


# Built-in mapping from target names / classes to scope chains. The
# chain is ORDERED from most-specific to most-general; the resolver
# walks it the other way for queries.
_TARGET_SCOPE_RULES: dict[str, list[str]] = {
    # NVIDIA GPUs
    "cuda-a100": ["backends/gpu/nvidia/ampere", "backends/gpu/nvidia/general", "backends/gpu/general", "general"],
    "cuda-h100": ["backends/gpu/nvidia/hopper", "backends/gpu/nvidia/general", "backends/gpu/general", "general"],
    "cuda-v100": ["backends/gpu/nvidia/volta", "backends/gpu/nvidia/general", "backends/gpu/general", "general"],
    "cuda-titan-rtx": ["backends/gpu/nvidia/turing", "backends/gpu/nvidia/general", "backends/gpu/general", "general"],
    "test-gpu-simt": ["backends/gpu/nvidia/turing", "backends/gpu/nvidia/general", "backends/gpu/general", "general"],
    # AMD GPUs
    "rocm-mi300": ["backends/gpu/amd/cdna3", "backends/gpu/amd/general", "backends/gpu/general", "general"],
    "rocm-mi250": ["backends/gpu/amd/cdna2", "backends/gpu/amd/general", "backends/gpu/general", "general"],
    # NPUs
    "openq_5165rb": ["backends/npu/hexagon/v69", "backends/npu/hexagon/general", "backends/npu/general", "general"],
    "trainium1": ["backends/npu/trainium/v1", "backends/npu/trainium/general", "backends/npu/general", "general"],
    "trainium2": ["backends/npu/trainium/v2", "backends/npu/trainium/general", "backends/npu/general", "general"],
    "tpu-v5": ["backends/npu/tpu/v5", "backends/npu/tpu/general", "backends/npu/general", "general"],
    # CPUs
    "cpu-host": ["backends/cpu/x86/general", "backends/cpu/general", "general"],
    "cpu-arm-neon": ["backends/cpu/arm/general", "backends/cpu/general", "general"],
    "riscv-soc": ["backends/cpu/riscv/general", "backends/cpu/general", "general"],
}


def scope_chain_for_target(target_name: str) -> list[str]:
    """Return the scope chain for ``target_name``, most-specific first.

    Falls back to ``[backends/<backend>/general, general]`` if the name
    isn't in the rule table — caller can override by setting
    ``profile.metadata['knowledge_scope']`` (see :func:`scope_chain_for_profile`).
    """
    if target_name in _TARGET_SCOPE_RULES:
        return list(_TARGET_SCOPE_RULES[target_name])
    # Fall-back guess from prefix
    if target_name.startswith("cuda"):
        return ["backends/gpu/nvidia/general", "backends/gpu/general", "general"]
    if target_name.startswith("rocm"):
        return ["backends/gpu/amd/general", "backends/gpu/general", "general"]
    if target_name.startswith("hexagon") or target_name.startswith("openq"):
        return ["backends/npu/hexagon/general", "backends/npu/general", "general"]
    if target_name.startswith("trainium"):
        return ["backends/npu/trainium/general", "backends/npu/general", "general"]
    if target_name.startswith("tpu"):
        return ["backends/npu/tpu/general", "backends/npu/general", "general"]
    if target_name.startswith("cpu"):
        return ["backends/cpu/general", "general"]
    return ["general"]


def scope_chain_for_profile(profile: Any) -> list[str]:
    """Resolve scope chain from a TargetProfile.

    Honors ``profile.metadata['knowledge_scope']`` when present (a list
    of scope strings most-specific first). Otherwise derives from
    ``profile.name`` via :func:`scope_chain_for_target`.
    """
    md = getattr(profile, "metadata", None) or {}
    explicit = md.get("knowledge_scope")
    if explicit:
        return list(explicit)
    return scope_chain_for_target(getattr(profile, "name", ""))


# ---------------------------------------------------------------------------
# Lesson dataclass
# ---------------------------------------------------------------------------


_VALID_CATEGORIES = ("perf", "correctness", "limit", "design", "recipe")
_VALID_STAGES = (
    "any",  # cross-stage principle
    "capture",  # FX / torch.export capture
    "decomp",  # FX→xDSL decomposition
    "recipe",  # Recipe IR generation / lowering
    "kernel-gen",  # generating Triton/CUDA/C kernel source
    "kernel-tune",  # autotune / refinement loop
    "fusion",  # fusion-decision logic
    "dispatch",  # runtime dispatch / sync
    "memory-plan",  # buffer / lifetime planning
    "verification",  # correctness / numeric gates
    "instrumentation",  # profiling / measurement
    "deployment",  # cold-start / persistence / packaging
)


@dataclass
class Lesson:
    """One realisation worth remembering across runs.

    Containerisation: lessons carry ``stage`` + ``op_family`` + ``topic``
    so the agent can ask narrow questions ("what do I know about
    *kernel-gen for matmul on Turing*?") instead of indiscriminately
    pulling every lesson at the scope.

    Attributes:
        scope:        Hardware/backend scope (``general`` | ``backends/...``).
        category:     ``perf | correctness | limit | design | recipe``.
        summary:      One-line, human-readable.
        stage:        Compiler stage this lesson applies to (see
                      ``_VALID_STAGES``). ``"any"`` = cross-stage.
        op_family:    Optional op family name (``matmul``, ``softmax``,
                      ``attention``, ``rmsnorm``, …). Empty = applies
                      regardless of op.
        topic:        Optional topic cluster (``tile-selection``,
                      ``fusion-decision``, ``sync-graph``, ``cost-model``,
                      …). Empty = topic-agnostic.
        evidence:     Free-form measured data backing the lesson.
        tags:         Searchable labels (open vocabulary).
        applicability: Free-form description of when this applies.
        next_action:  Optional pointer for the agent.
        id, timestamp: Auto-managed.
    """

    scope: str
    category: str
    summary: str
    id: str = ""
    stage: str = "any"
    op_family: str = ""
    topic: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    applicability: str = ""
    next_action: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(f"Lesson.category must be one of {_VALID_CATEGORIES} (got {self.category!r})")
        if self.stage not in _VALID_STAGES:
            raise ValueError(f"Lesson.stage must be one of {_VALID_STAGES} (got {self.stage!r})")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeStore:
    """File-backed hierarchical knowledge store.

    Threads / processes: writes are append-line for ``lessons.jsonl``
    (atomic if line < PIPE_BUF). Single-writer assumed; multiple
    concurrent reads are safe.
    """

    root: Path = field(default_factory=default_knowledge_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- scope helpers -----

    def scope_dir(self, scope: str) -> Path:
        path = self.root / scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def lessons_file(self, scope: str) -> Path:
        return self.scope_dir(scope) / "lessons.jsonl"

    # ----- add / query -----

    def add(self, lesson: Lesson) -> Lesson:
        """Append a lesson at ``lesson.scope``. Auto-assigns id + timestamp."""
        if not lesson.timestamp:
            lesson.timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        path = self.lessons_file(lesson.scope)
        existing = self._read_lessons(path)
        if not lesson.id:
            lesson.id = f"lesson_{len(existing) + 1:04d}"
        with path.open("a") as f:
            f.write(json.dumps(_lesson_to_dict(lesson), sort_keys=True) + "\n")
        return lesson

    def query(
        self,
        scope_chain: list[str],
        *,
        stage: str | None = None,
        op_family: str | None = None,
        topic: str | None = None,
        categories: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[Lesson]:
        """Return narrowly-scoped lessons.

        Filters intersect: a lesson must match ALL of (stage, op_family,
        topic, categories, tags) when those filters are provided. Each
        filter has "match-or-empty" semantics — a lesson with empty
        ``op_family`` matches any caller's ``op_family`` filter, since
        empty means "applies regardless of op".

        Walks the scope chain bottom-up (general → arch-specific) so
        the caller sees broad principles before narrow specifics. Use
        ``limit`` to cap context size when injecting into a prompt.
        """
        out: list[Lesson] = []
        for scope in reversed(scope_chain):
            out.extend(self._read_lessons(self.lessons_file(scope)))

        # stage filter — "any" lessons match any stage; otherwise exact match
        if stage:
            out = [l for l in out if l.stage in ("any", stage)]
        # op_family — empty lesson op_family matches any caller op_family
        if op_family:
            out = [l for l in out if not l.op_family or l.op_family == op_family]
        # topic — same "match-or-empty" rule
        if topic:
            out = [l for l in out if not l.topic or l.topic == topic]

        if categories:
            out = [l for l in out if l.category in categories]
        if tags:
            tagset = set(tags)
            out = [l for l in out if tagset.intersection(l.tags)]

        if limit is not None:
            out = out[:limit]
        return out

    def query_for_target(
        self,
        target_name: str,
        *,
        stage: str | None = None,
        op_family: str | None = None,
        topic: str | None = None,
        categories: tuple[str, ...] | None = None,
        tags: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[Lesson]:
        return self.query(
            scope_chain_for_target(target_name),
            stage=stage,
            op_family=op_family,
            topic=topic,
            categories=categories,
            tags=tags,
            limit=limit,
        )

    # --- containerised contexts ---

    def context_brief(
        self,
        target_name: str,
        *,
        stage: str,
        op_family: str | None = None,
        topic: str | None = None,
        max_lessons: int = 8,
    ) -> str:
        """Render a tight markdown brief for prompt injection.

        Use case: the codegen / refinement / fusion-oracle each call
        this with their own (stage, op_family, topic) — gets a
        lessons-list scoped to exactly what they need, capped at
        ``max_lessons`` so context doesn't bloat.
        """
        lessons = self.query_for_target(
            target_name,
            stage=stage,
            op_family=op_family,
            topic=topic,
            limit=max_lessons,
        )
        if not lessons:
            return ""
        header = (
            f"## Knowledge for target={target_name!r} stage={stage!r}"
            + (f" op={op_family!r}" if op_family else "")
            + (f" topic={topic!r}" if topic else "")
        )
        lines = [header, ""]
        for l in lessons:
            tag_blob = (" tags=[" + ", ".join(l.tags) + "]") if l.tags else ""
            lines.append(f"- **[{l.category}]** {l.summary}{tag_blob}")
            if l.next_action:
                lines.append(f"    → action: {l.next_action}")
        return "\n".join(lines)

    def list_scopes(self) -> list[str]:
        """All scopes that hold at least one lesson."""
        scopes: list[str] = []
        for path in self.root.rglob("lessons.jsonl"):
            scopes.append(str(path.parent.relative_to(self.root)).replace(os.sep, "/"))
        return sorted(scopes)

    # ----- internals -----

    @staticmethod
    def _read_lessons(path: Path) -> list[Lesson]:
        if not path.exists():
            return []
        out: list[Lesson] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(_lesson_from_dict(d))
        return out


def _lesson_to_dict(l: Lesson) -> dict[str, Any]:
    d = asdict(l)
    d["tags"] = list(d["tags"])
    return d


def _lesson_from_dict(d: dict[str, Any]) -> Lesson:
    return Lesson(
        id=d.get("id", ""),
        scope=d.get("scope", "general"),
        category=d.get("category", "design"),
        summary=d.get("summary", ""),
        stage=d.get("stage", "any"),
        op_family=d.get("op_family", ""),
        topic=d.get("topic", ""),
        evidence=d.get("evidence", {}),
        tags=tuple(d.get("tags", ())),
        applicability=d.get("applicability", ""),
        next_action=d.get("next_action", ""),
        timestamp=d.get("timestamp", ""),
    )


# ---------------------------------------------------------------------------
# Process-wide singleton (mirrors KernelStore pattern)
# ---------------------------------------------------------------------------


_singleton: KnowledgeStore | None = None


def shared_store() -> KnowledgeStore:
    global _singleton
    if _singleton is None:
        _singleton = KnowledgeStore()
    return _singleton


def set_shared_store(s: KnowledgeStore | None) -> None:
    global _singleton
    _singleton = s


__all__ = [
    "KnowledgeStore",
    "Lesson",
    "default_knowledge_root",
    "scope_chain_for_profile",
    "scope_chain_for_target",
    "set_shared_store",
    "shared_store",
]
