"""Per-target hardware-spec + lessons memory.

A :class:`TargetKnowledgeCard` captures everything a kernel-generation
agent needs to know about one deployment target:

* a structured **HardwareSpec** (ISA instructions, intrinsic signatures,
  parameter ranges, memory tiers, dataflow modes, constraints),
* mined **exemplars** (paths to reference kernels worth showing the LLM),
* **doc sources** (provenance for ingested markdown / asciidoc / PDFs / URLs),
* a path to an append-only **lessons.jsonl** that the agent loop writes after
  each accepted strategy,
* a path to a **strategies.json** that tracks running confidence / usage
  counts per (archetype, dtype, layout) tuple.

The on-disk layout for each target lives under
``<repo>/.xpu_rt/knowledge/targets/<target_id>/``::

    target_card.json     # serialized TargetKnowledgeCard
    isa.md               # routed bucket from doc ingestion
    architecture.md
    intrinsics.md
    constraints.md
    exemplars/<op>.{c,scala,py,...}
    lessons.jsonl        # append-only
    strategies.json      # keyed by (archetype, dtype, layout)
    docs/<sha>.{md,html,pdf}

The card is intentionally *not* a TargetProfile — it links to one via
``target_profile_ref``. Static target descriptors stay in
``xpu_rt.targets.target_types.TargetCard``; this module is the dynamic
knowledge layer that grows over time as the agent runs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Locate the XPU-RT inner package root (matches observability convention).

    Resolution order:
      1. ``XPU_RT_REPO_ROOT`` env var (used by tests for isolation).
      2. Walk up from this file to find a parent containing both
         ``pyproject.toml`` and ``python/xpu_rt``.
      3. Fallback: three parents up (``python/xpu_rt/memory`` -> root).
    """
    env_root = os.environ.get("XPU_RT_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "python" / "xpu_rt").exists():
            return parent
    return here.parents[1]


def knowledge_root() -> Path:
    """Root directory for per-target knowledge cards on this machine."""
    override = os.environ.get("XPU_RT_KNOWLEDGE_DIR")
    if override:
        path = Path(override)
    else:
        path = _repo_root() / ".xpu_rt" / "knowledge" / "targets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def target_dir(target_id: str) -> Path:
    """Directory holding all artifacts for one target."""
    if not target_id or "/" in target_id or ".." in target_id:
        raise ValueError(f"invalid target_id: {target_id!r}")
    path = knowledge_root() / target_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


IsaFamily = Literal[
    "riscv-rvv",
    "rocc-systolic",
    "cuda-sm",
    "ukernel",
    "host-cpu",
    "triton",
    "other",
]

# Routing buckets used by the doc-ingestion subsystem. Kept here so the
# ingestion router and the card schema agree on the closed set.
BUCKETS: tuple[str, ...] = (
    "isa",
    "architecture",
    "intrinsics",
    "examples",
    "constraints",
)


@dataclass(frozen=True)
class ParameterRange:
    """A tunable parameter exposed by the target generator (e.g. vLen).

    ``values`` lists explicit named presets when ``min_value``/``max_value``
    don't capture them (e.g. ``issStructure ∈ {Unified, Shared, Split}``).
    """

    name: str
    description: str = ""
    min_value: int | float | None = None
    max_value: int | float | None = None
    default: int | float | str | None = None
    values: tuple[str, ...] = ()
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default": self.default,
            "values": list(self.values),
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> ParameterRange:
        return cls(
            name=str(body["name"]),
            description=str(body.get("description", "")),
            min_value=body.get("min_value"),
            max_value=body.get("max_value"),
            default=body.get("default"),
            values=tuple(body.get("values", ())),
            unit=str(body.get("unit", "")),
        )


@dataclass(frozen=True)
class MemoryTierSpec:
    """A memory tier exposed by the target.

    Mirrors ``xpu_rt.targets.schema.MemoryLevel`` but adds ``kind`` so
    ingestion-time data can be loaded without resolving the richer
    TargetProfile.
    """

    name: str
    kind: str  # scratchpad | l1 | l2 | l3 | hbm | dram | host | registers | accumulator
    size_bytes: int | None = None
    bandwidth_gbps: float | None = None
    latency_ns: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> MemoryTierSpec:
        return cls(
            name=str(body["name"]),
            kind=str(body.get("kind", "")),
            size_bytes=body.get("size_bytes"),
            bandwidth_gbps=body.get("bandwidth_gbps"),
            latency_ns=body.get("latency_ns"),
        )


@dataclass(frozen=True)
class ISAInstruction:
    """One ISA-level instruction the agent should know about."""

    mnemonic: str
    signature: str = ""  # textual operand layout, e.g. "rd, rs1, vs2"
    summary: str = ""
    latency_cycles: int | None = None
    funct_code: int | None = None  # for RoCC custom opcodes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> ISAInstruction:
        return cls(
            mnemonic=str(body["mnemonic"]),
            signature=str(body.get("signature", "")),
            summary=str(body.get("summary", "")),
            latency_cycles=body.get("latency_cycles"),
            funct_code=body.get("funct_code"),
            notes=str(body.get("notes", "")),
        )


@dataclass(frozen=True)
class IntrinsicSignature:
    """A C-callable intrinsic exposed by the runtime (e.g. gemmini.h macros)."""

    name: str
    c_signature: str
    summary: str = ""
    requires: tuple[str, ...] = ()  # other intrinsics that must precede this one
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "c_signature": self.c_signature,
            "summary": self.summary,
            "requires": list(self.requires),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> IntrinsicSignature:
        return cls(
            name=str(body["name"]),
            c_signature=str(body.get("c_signature", "")),
            summary=str(body.get("summary", "")),
            requires=tuple(body.get("requires", ())),
            notes=str(body.get("notes", "")),
        )


@dataclass(frozen=True)
class DerivationRule:
    """A worked-out sizing / capacity / alignment constraint.

    Constraints that depend only on the target's static configuration
    (scratchpad capacity, accumulator capacity, bank count, DIM, …)
    can be reduced to a single number at ingestion time. Capturing
    just the symbolic form ("spad_rows ≤ BANK_NUM * BANK_ROWS / 2")
    routinely gets under-applied by downstream LLMs because the
    symbols don't resolve from the rest of the card's flat fields.
    DerivationRule pins the *concrete* number alongside the symbolic
    form and tells the agent how to apply it.

    Attributes:
        name: Short slug for cross-referencing (e.g. "spad_tile_budget").
        symbolic: The original constraint as stated by the vendor.
        concrete_value: The resolved numeric bound, in the units of
            ``unit``.
        unit: e.g. "rows", "bytes", "tiles", "cycles", "—".
        derivation: One-line walkthrough of how concrete_value was
            computed from the card's static facts.
        applies_to: Free-form tag describing when this rule fires —
            e.g. "tile budgeting", "output sizing", "alignment".
        how_to_apply: Imperative sentence the prompt builder can lift
            into the user-message verbatim. E.g. "(tile_I + tile_J) *
            tile_K ≤ {concrete_value}".
    """

    name: str
    symbolic: str
    concrete_value: float
    unit: str = ""
    derivation: str = ""
    applies_to: str = ""
    how_to_apply: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> DerivationRule:
        return cls(
            name=str(body["name"]),
            symbolic=str(body.get("symbolic", "")),
            concrete_value=float(body.get("concrete_value", 0.0)),
            unit=str(body.get("unit", "")),
            derivation=str(body.get("derivation", "")),
            applies_to=str(body.get("applies_to", "")),
            how_to_apply=str(body.get("how_to_apply", "")),
        )


@dataclass(frozen=True)
class HardwareSpec:
    """Structured hardware description backing the knowledge card."""

    isa_family: str  # one of IsaFamily values (validated in TargetKnowledgeCard.from_dict)
    parameters: tuple[ParameterRange, ...] = ()
    memory_tiers: tuple[MemoryTierSpec, ...] = ()
    instructions: tuple[ISAInstruction, ...] = ()
    intrinsics: tuple[IntrinsicSignature, ...] = ()
    dataflow_modes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    # Worked-out sizing rules with concrete numeric bounds. The agent
    # loop's prompt builder surfaces these as a dedicated
    # ``## Sizing constraints (worked out)`` section. See the
    # :class:`DerivationRule` docstring + memory note
    # ``feedback_target_card_derivation_rules`` for the rationale.
    derivation_rules: tuple[DerivationRule, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "isa_family": self.isa_family,
            "parameters": [p.to_dict() for p in self.parameters],
            "memory_tiers": [m.to_dict() for m in self.memory_tiers],
            "instructions": [i.to_dict() for i in self.instructions],
            "intrinsics": [i.to_dict() for i in self.intrinsics],
            "dataflow_modes": list(self.dataflow_modes),
            "constraints": list(self.constraints),
            "derivation_rules": [r.to_dict() for r in self.derivation_rules],
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> HardwareSpec:
        return cls(
            isa_family=str(body["isa_family"]),
            parameters=tuple(ParameterRange.from_dict(p) for p in body.get("parameters", ())),
            memory_tiers=tuple(MemoryTierSpec.from_dict(m) for m in body.get("memory_tiers", ())),
            instructions=tuple(ISAInstruction.from_dict(i) for i in body.get("instructions", ())),
            intrinsics=tuple(IntrinsicSignature.from_dict(i) for i in body.get("intrinsics", ())),
            dataflow_modes=tuple(body.get("dataflow_modes", ())),
            constraints=tuple(body.get("constraints", ())),
            derivation_rules=tuple(
                DerivationRule.from_dict(r) for r in body.get("derivation_rules", ())
            ),
        )


@dataclass(frozen=True)
class KernelExemplar:
    """A reference kernel worth surfacing to the agent during prompt-build."""

    name: str
    op_family: str  # 'matmul' | 'gemm' | 'conv' | 'reduce' | 'pointwise' | ...
    path: str  # path relative to the card's exemplars/ directory
    language: str = ""  # 'c' | 'cpp' | 'scala' | 'asm' | 'cuda' | 'triton' | ...
    tags: tuple[str, ...] = ()  # free-form labels for retrieval
    source: str = ""  # provenance: original file path or URL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_family": self.op_family,
            "path": self.path,
            "language": self.language,
            "tags": list(self.tags),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> KernelExemplar:
        return cls(
            name=str(body["name"]),
            op_family=str(body.get("op_family", "")),
            path=str(body["path"]),
            language=str(body.get("language", "")),
            tags=tuple(body.get("tags", ())),
            source=str(body.get("source", "")),
        )


@dataclass(frozen=True)
class DocSource:
    """Provenance entry for one ingested document."""

    locator: str  # local path or URL
    kind: Literal["path", "url"] = "path"
    sha256: str = ""  # content hash (16 hex chars is plenty for dedup)
    fetched_at: str = ""  # ISO-8601 UTC
    bucket: str = ""  # one of BUCKETS
    bytes: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> DocSource:
        return cls(
            locator=str(body["locator"]),
            kind=body.get("kind", "path"),
            sha256=str(body.get("sha256", "")),
            fetched_at=str(body.get("fetched_at", "")),
            bucket=str(body.get("bucket", "")),
            bytes=int(body.get("bytes", 0)),
            notes=str(body.get("notes", "")),
        )


SCHEMA_VERSION = "xpu_rt_target_knowledge_v1"


@dataclass(frozen=True)
class TargetKnowledgeCard:
    """Top-level per-target knowledge record.

    Use :func:`load` to read by ``target_id`` and :func:`save` to persist
    (atomic write through a sibling ``.tmp`` file + rename).
    """

    target_id: str
    target_profile_ref: str  # configs/targets/<id>.yaml relative path
    hardware_spec: HardwareSpec
    exemplars: tuple[KernelExemplar, ...] = ()
    docs: tuple[DocSource, ...] = ()
    revision: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""  # ISO-8601 UTC; set by save() when empty
    updated_at: str = ""

    # ---- on-disk path helpers ----

    @property
    def root(self) -> Path:
        return target_dir(self.target_id)

    @property
    def card_path(self) -> Path:
        return self.root / "target_card.json"

    @property
    def lessons_path(self) -> Path:
        return self.root / "lessons.jsonl"

    @property
    def strategies_path(self) -> Path:
        return self.root / "strategies.json"

    @property
    def exemplars_dir(self) -> Path:
        path = self.root / "exemplars"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def docs_dir(self) -> Path:
        path = self.root / "docs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def bucket_path(self, bucket: str) -> Path:
        if bucket not in BUCKETS:
            raise ValueError(f"unknown bucket {bucket!r}; expected one of {BUCKETS}")
        return self.root / f"{bucket}.md"

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "target_profile_ref": self.target_profile_ref,
            "hardware_spec": self.hardware_spec.to_dict(),
            "exemplars": [e.to_dict() for e in self.exemplars],
            "docs": [d.to_dict() for d in self.docs],
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> TargetKnowledgeCard:
        schema = str(body.get("schema_version", SCHEMA_VERSION))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported target knowledge schema {schema!r}; "
                f"this xpu_rt only reads {SCHEMA_VERSION!r}"
            )
        return cls(
            target_id=str(body["target_id"]),
            target_profile_ref=str(body.get("target_profile_ref", "")),
            hardware_spec=HardwareSpec.from_dict(body["hardware_spec"]),
            exemplars=tuple(KernelExemplar.from_dict(e) for e in body.get("exemplars", ())),
            docs=tuple(DocSource.from_dict(d) for d in body.get("docs", ())),
            revision=int(body.get("revision", 1)),
            schema_version=schema,
            created_at=str(body.get("created_at", "")),
            updated_at=str(body.get("updated_at", "")),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(card: TargetKnowledgeCard) -> TargetKnowledgeCard:
    """Persist ``card`` to its directory; returns the stored copy.

    Stamps ``created_at`` on first save and ``updated_at`` on every save.
    Writes atomically via a sibling ``.tmp`` file + rename.
    """
    now = _now_iso()
    created = card.created_at or now
    persisted = dataclasses.replace(card, created_at=created, updated_at=now)
    persisted.root.mkdir(parents=True, exist_ok=True)
    tmp = persisted.card_path.with_suffix(persisted.card_path.suffix + ".tmp")
    tmp.write_text(json.dumps(persisted.to_dict(), indent=2, sort_keys=False))
    tmp.replace(persisted.card_path)
    return persisted


def load(target_id: str) -> TargetKnowledgeCard:
    """Read a knowledge card by id; raises FileNotFoundError if absent."""
    path = target_dir(target_id) / "target_card.json"
    if not path.exists():
        raise FileNotFoundError(f"no knowledge card for target_id={target_id!r} at {path}")
    body = json.loads(path.read_text())
    return TargetKnowledgeCard.from_dict(body)


def exists(target_id: str) -> bool:
    return (target_dir(target_id) / "target_card.json").exists()


def list_targets() -> list[str]:
    """All target_ids with a knowledge card on disk."""
    root = knowledge_root()
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "target_card.json").exists():
            out.append(child.name)
    return out


# ---------------------------------------------------------------------------
# Lessons + strategies — minimal append/scan helpers used by KB v2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lesson:
    """One row appended to ``lessons.jsonl`` after each accepted strategy."""

    timestamp: str
    archetype: str  # KernelContract v3 archetype
    dtype_class: str
    layout_kind: str
    op_family: str
    action: str  # short tag, e.g. "tile-K=64", "use-mvin-stride"
    measured_gain: float  # speedup factor over previous best, 1.0 if no prior
    sample_count: int = 1
    notes: str = ""

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> Lesson:
        return cls(**json.loads(line))


def append_lesson(card: TargetKnowledgeCard, lesson: Lesson) -> None:
    """Append one row to the card's ``lessons.jsonl``."""
    with card.lessons_path.open("a", encoding="utf-8") as f:
        f.write(lesson.to_json_line() + "\n")


def iter_lessons(card: TargetKnowledgeCard) -> Iterator[Lesson]:
    if not card.lessons_path.exists():
        return
    with card.lessons_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield Lesson.from_json_line(line)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping malformed lesson line: %s", exc)


__all__ = [
    "BUCKETS",
    "DerivationRule",
    "SCHEMA_VERSION",
    "DocSource",
    "HardwareSpec",
    "ISAInstruction",
    "IntrinsicSignature",
    "IsaFamily",
    "KernelExemplar",
    "Lesson",
    "MemoryTierSpec",
    "ParameterRange",
    "TargetKnowledgeCard",
    "append_lesson",
    "exists",
    "iter_lessons",
    "knowledge_root",
    "list_targets",
    "load",
    "save",
    "target_dir",
]
