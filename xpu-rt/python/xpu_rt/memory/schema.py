"""Core data model for the Compiler Memory System.

All entities share one unified lifecycle:
    specification → generation → validation → profiling → selection → promotion → reuse → retirement

Every generated thing (kernel, pass, rewrite, guard, decomposition,
translation, backend plan, schedule) goes through this lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObjectKind(Enum):
    """What kind of optimization object this is."""

    KERNEL = "kernel"
    REWRITE = "rewrite"
    PASS = "pass"
    GUARD = "guard"
    DECOMPOSITION = "decomposition"
    TRANSLATION = "translation"
    BACKEND_PLAN = "backend_plan"
    SCHEDULE = "schedule"
    COST_TERM = "cost_term"


class CandidateStatus(Enum):
    """Lifecycle status of a candidate."""

    NEW = "new"
    COMPILED = "compiled"
    VERIFIED = "verified"
    PROFILED = "profiled"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    RETIRED = "retired"


class GeneratorKind(Enum):
    """How a candidate was generated."""

    LLM = "llm"
    MUTATION = "mutation"
    TEMPLATE = "template"
    RETRIEVAL_SEED = "retrieval_seed"
    PROVIDER = "provider"
    MINED = "mined"
    COOKBOOK = "cookbook"


class KnowledgeKind(Enum):
    """Types of reusable knowledge items."""

    HARDWARE_RULE = "hardware_rule"
    OPTIMIZATION_TACTIC = "optimization_tactic"
    ERROR_REPAIR = "error_repair"
    PASS_PATTERN = "pass_pattern"
    SCHEDULE_TEMPLATE = "schedule_template"
    PROOF_HINT = "proof_hint"
    LAYOUT_PREFERENCE = "layout_preference"
    FAILURE_MODE = "failure_mode"


class ScopeKind(Enum):
    """Scope of a knowledge item's applicability."""

    GLOBAL = "global"
    HARDWARE_FAMILY = "hardware_family"
    TARGET = "target"
    OPERATOR_FAMILY = "operator_family"
    WORKLOAD_BUNDLE = "workload_bundle"


# ============================================================================
# Core entities
# ============================================================================


@dataclass(frozen=True)
class Task:
    """One optimization problem to solve.

    Examples:
        - "generate a Triton kernel for region r17 on H100"
        - "generate an accfg overlap rewrite for snax_gemm loop_4"
        - "generate decomposition for unsupported TorchAO op"
    """

    task_id: str
    task_kind: ObjectKind
    workload_key: str = ""
    region_key: str = ""
    target_key: str = ""
    hardware_key: str = ""
    objective: str = "latency"
    input_artifact_hash: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class StateSignature:
    """The state a candidate was solving.

    This is the bridge between current problem and past successful
    tactics. Retrieval is keyed by state signatures, not just exact
    workload hashes.
    """

    state_id: str
    task_id: str
    op_family: str = ""
    shape_signature: str = ""
    dtype_signature: str = ""
    layout_signature: str = ""
    hardware_signature: str = ""
    memory_signature: str = ""
    config_signature: str = ""
    bottleneck_signature: str = ""
    profile_signature: str = ""


@dataclass(frozen=True)
class Candidate:
    """A candidate generated during search.

    Every kernel, pass, guard, etc. that gets generated is a Candidate.
    It may have a parent (mutation/refinement chain) and a state
    signature (what it was solving).
    """

    candidate_id: str
    task_id: str
    artifact_hash: str = ""
    parent_candidate_id: str = ""
    generator_kind: GeneratorKind = GeneratorKind.LLM
    generator_model: str = ""
    generation_round: int = 0
    state_signature_id: str = ""
    status: CandidateStatus = CandidateStatus.NEW
    created_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    """One evaluation run for one candidate."""

    eval_id: str
    candidate_id: str
    backend: str = ""
    compile_ok: bool = False
    correctness_ok: bool = False
    perf_ok: bool = False
    score: float = 0.0
    latency_us: float = 0.0
    throughput: float = 0.0
    energy: float = 0.0
    verifier_summary: str = ""
    profile_summary: str = ""
    profile_artifact_hash: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class KnowledgeItem:
    """Reusable knowledge extracted from search.

    Knowledge items are the L2 memory level — they persist across
    tasks and inform future search through retrieval.
    """

    knowledge_id: str
    knowledge_kind: KnowledgeKind = KnowledgeKind.OPTIMIZATION_TACTIC
    scope_kind: ScopeKind = ScopeKind.GLOBAL
    scope_key: str = ""
    summary: str = ""
    artifact_hash: str = ""
    quality_score: float = 0.0
    uses: int = 0
    wins: int = 0
    failures: int = 0
    last_used_at: str = ""
    source: str = ""
    embedding_hash: str = ""


@dataclass(frozen=True)
class Promotion:
    """Immutable promoted winner in the L3 library.

    M-26 attaches the two-tier cache key (``region_signature`` and
    ``contract_hash``) so the SQLite ``promotions`` table is the
    queryable index for cross-model recipe retrieval. M-29 adds
    ``gate_level`` (string form of :class:`PromotionLevel`) so audit
    consumers can rank promoted recipes by evidence strength without
    re-reading the bundle.
    """

    promotion_id: str
    candidate_id: str
    promotion_key: str = ""
    version: int = 1
    reason: str = ""
    measured_gain: float = 0.0
    verified_by: str = ""
    created_at: str = ""
    region_signature: str = ""
    contract_hash: str = ""
    gate_level: str = ""


@dataclass(frozen=True)
class EpisodeStep:
    """One step in a search episode (L0/L1 replay buffer)."""

    step_id: str
    task_id: str
    candidate_id: str = ""
    action: str = ""
    reward: float = 0.0
    step_number: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class Source:
    """Provenance of knowledge (mined, curated, online)."""

    source_id: str
    source_kind: str = ""  # "mlirAgent_mined", "cookbook_curated", "online_search"
    repo_url: str = ""
    commit_hash: str = ""
    path: str = ""
    ingested_at: str = ""


__all__ = [
    "Candidate",
    "CandidateStatus",
    "EpisodeStep",
    "Evaluation",
    "GeneratorKind",
    "KnowledgeItem",
    "KnowledgeKind",
    "ObjectKind",
    "Promotion",
    "ScopeKind",
    "Source",
    "StateSignature",
    "Task",
]
