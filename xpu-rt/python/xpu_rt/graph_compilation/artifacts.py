"""Typed artifact contract for graph compilation runs.

Dataclasses + jsonschema (no pydantic — stdlib only). The on-disk
encoding is JSON for the manifest and JSONL for the ledger; these
classes are the typed in-memory mirror with explicit ``from_dict`` /
``to_dict`` converters.

The schema files under ``schemas/v1/`` are the authoritative validators.
This module's converters mirror them; the round-trip test in
``tests/graph_compilation/test_artifact_contract.py`` (T18) keeps the two
in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Canonical stage order. R003 enforces that any prefix of this list is
# acceptable (so ``--stop-after stage1`` runs are valid), but no other
# ordering is.
CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "graph_capture",
    "payload_lowering",
    "graph_analysis",
    "recipe_planning",
    "gap_discovery",
    "gap_closure",
)

# On-disk directory prefix per stage_id. The M-05 insertion of
# ``03_recipe_planning`` shifts ``gap_discovery`` to ``04_`` and
# ``gap_closure`` to ``05_``. Earlier layouts (pre-M-05 and pre-M-B) are
# still readable via :func:`stage_dir` below.
STAGE_DIR_PREFIXES: dict[str, str] = {
    "graph_capture": "00_graph_capture",
    "payload_lowering": "01_payload_lowering",
    "graph_analysis": "02_graph_analysis",
    "recipe_planning": "03_recipe_planning",
    "gap_discovery": "04_gap_discovery",
    "gap_closure": "05_gap_closure",
}
# Legacy on-disk prefixes accepted by readers for backward compatibility.
# Each list is "newest first" so the resolver picks the most-recent
# layout it can find. These cover both the pre-M-B (no graph_analysis)
# and pre-M-05 (no recipe_planning) generations.
LEGACY_STAGE_DIR_PREFIXES: dict[str, list[str]] = {
    "gap_discovery": ["03_gap_discovery", "02_gap_discovery"],
    "gap_closure": ["04_gap_closure", "03_gap_closure"],
}


def stage_dir(run_dir: str | Path, stage_id: str) -> Path:
    """Return the canonical (new-layout) stage directory inside ``run_dir``,
    falling back to the legacy prefix when the canonical one is missing.

    This is read-only (does not create directories). Use
    :func:`stage_dir_canonical` when you intend to *write* into the dir.
    """
    from pathlib import Path

    rd = Path(run_dir) if not isinstance(run_dir, Path) else run_dir
    canonical = rd / STAGE_DIR_PREFIXES[stage_id]
    if canonical.exists():
        return canonical
    for legacy_prefix in LEGACY_STAGE_DIR_PREFIXES.get(stage_id, ()):
        legacy = rd / legacy_prefix
        if legacy.exists():
            return legacy
    return canonical


def stage_dir_canonical(run_dir: str | Path, stage_id: str) -> Path:
    """Return the canonical (new-layout) stage directory; never falls back."""
    from pathlib import Path

    rd = Path(run_dir) if not isinstance(run_dir, Path) else run_dir
    return rd / STAGE_DIR_PREFIXES[stage_id]


# No reserved stages today — every milestone has shipped.
RESERVED_STAGE_IDS: frozenset[str] = frozenset()

# Allowed enums.
_ARTIFACT_KINDS = frozenset({"file", "tree"})
_STAGE_STATUSES = frozenset({"pass", "fail", "skipped"})
_LEDGER_EVENTS = frozenset(
    {"start", "finish", "artifact_written", "validation_pass", "validation_fail"}
)
_RULE_STATUSES = frozenset({"pass", "fail", "skipped"})
_OVERALL_STATUSES = frozenset({"pass", "fail"})


class SchemaError(ValueError):
    """Raised when a payload does not match the typed contract."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _str(d: dict[str, Any], key: str, *, allow_none: bool = False) -> str | None:
    val = d.get(key)
    if val is None:
        _require(allow_none, f"missing required string field: {key!r}")
        return None
    _require(isinstance(val, str), f"field {key!r} must be a string")
    assert isinstance(val, str)
    return val


def _int(d: dict[str, Any], key: str) -> int:
    val = d.get(key)
    _require(isinstance(val, int) and not isinstance(val, bool), f"field {key!r} must be int")
    assert isinstance(val, int)
    return val


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to one on-disk artifact (file or directory tree).

    Paths are run-dir-relative and POSIX-style. The validator (R004)
    rejects absolute paths and paths that escape ``run_dir``.
    """

    path: str
    sha256: str
    size_bytes: int
    kind: str  # "file" | "tree"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactRef:
        path = _str(d, "path")
        sha256 = _str(d, "sha256")
        kind = _str(d, "kind")
        assert path is not None and sha256 is not None and kind is not None
        _require(kind in _ARTIFACT_KINDS, f"artifact kind must be one of {_ARTIFACT_KINDS}, got {kind!r}")
        size = _int(d, "size_bytes")
        _require(size >= 0, "size_bytes must be non-negative")
        return cls(path=path, sha256=sha256, size_bytes=size, kind=kind)


@dataclass(frozen=True)
class ModelRef:
    config_path: str
    model_id: str
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "model_id": self.model_id,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelRef:
        config_path = _str(d, "config_path")
        model_id = _str(d, "model_id")
        config_sha256 = _str(d, "config_sha256")
        assert config_path is not None and model_id is not None and config_sha256 is not None
        return cls(config_path=config_path, model_id=model_id, config_sha256=config_sha256)


@dataclass(frozen=True)
class TargetRef:
    config_path: str
    target_id: str
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "target_id": self.target_id,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetRef:
        config_path = _str(d, "config_path")
        target_id = _str(d, "target_id")
        config_sha256 = _str(d, "config_sha256")
        assert config_path is not None and target_id is not None and config_sha256 is not None
        return cls(config_path=config_path, target_id=target_id, config_sha256=config_sha256)


@dataclass(frozen=True)
class StageRecord:
    stage_id: str
    status: str
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    report_path: str
    input_hash: str
    output_hash: str
    llm_calls: int
    started_at_utc: str
    finished_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "status": self.status,
            "inputs": [a.to_dict() for a in self.inputs],
            "outputs": [a.to_dict() for a in self.outputs],
            "report_path": self.report_path,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "llm_calls": self.llm_calls,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageRecord:
        stage_id = _str(d, "stage_id")
        status = _str(d, "status")
        report_path = _str(d, "report_path")
        input_hash = _str(d, "input_hash")
        output_hash = _str(d, "output_hash")
        started = _str(d, "started_at_utc")
        finished = _str(d, "finished_at_utc")
        assert (
            stage_id is not None
            and status is not None
            and report_path is not None
            and input_hash is not None
            and output_hash is not None
            and started is not None
            and finished is not None
        )
        _require(status in _STAGE_STATUSES, f"status must be one of {_STAGE_STATUSES}, got {status!r}")
        llm_calls = _int(d, "llm_calls")
        _require(llm_calls >= 0, "llm_calls must be non-negative")

        inputs_raw = d.get("inputs", [])
        outputs_raw = d.get("outputs", [])
        _require(isinstance(inputs_raw, list), "inputs must be a list")
        _require(isinstance(outputs_raw, list), "outputs must be a list")
        inputs = tuple(ArtifactRef.from_dict(x) for x in inputs_raw)
        outputs = tuple(ArtifactRef.from_dict(x) for x in outputs_raw)

        return cls(
            stage_id=stage_id,
            status=status,
            inputs=inputs,
            outputs=outputs,
            report_path=report_path,
            input_hash=input_hash,
            output_hash=output_hash,
            llm_calls=llm_calls,
            started_at_utc=started,
            finished_at_utc=finished,
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: str  # always "run_manifest_v1"
    run_id: str
    created_at_utc: str
    git_commit: str | None
    model: ModelRef
    target: TargetRef
    seed: int
    stages: tuple[StageRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "git_commit": self.git_commit,
            "model": self.model.to_dict(),
            "target": self.target.to_dict(),
            "seed": self.seed,
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunManifest:
        schema_version = _str(d, "schema_version")
        _require(schema_version == "run_manifest_v1", f"unsupported schema_version {schema_version!r}")
        run_id = _str(d, "run_id")
        created_at_utc = _str(d, "created_at_utc")
        assert run_id is not None and created_at_utc is not None
        git_commit = _str(d, "git_commit", allow_none=True)
        seed = _int(d, "seed")

        model_raw = d.get("model")
        target_raw = d.get("target")
        _require(isinstance(model_raw, dict), "model must be an object")
        _require(isinstance(target_raw, dict), "target must be an object")
        assert isinstance(model_raw, dict)
        assert isinstance(target_raw, dict)
        model = ModelRef.from_dict(model_raw)
        target = TargetRef.from_dict(target_raw)

        stages_raw = d.get("stages", [])
        _require(isinstance(stages_raw, list), "stages must be a list")
        stages = tuple(StageRecord.from_dict(x) for x in stages_raw)

        return cls(
            schema_version=schema_version or "run_manifest_v1",
            run_id=run_id,
            created_at_utc=created_at_utc,
            git_commit=git_commit,
            model=model,
            target=target,
            seed=seed,
            stages=stages,
        )


@dataclass(frozen=True)
class StageEvent:
    schema_version: str  # always "stage_event_v1"
    stage_id: str
    event: str
    artifact_path: str | None
    sha256: str | None
    timestamp_utc: str
    note: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "event": self.event,
            "artifact_path": self.artifact_path,
            "sha256": self.sha256,
            "timestamp_utc": self.timestamp_utc,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageEvent:
        schema_version = _str(d, "schema_version")
        _require(schema_version == "stage_event_v1", f"unsupported ledger schema_version {schema_version!r}")
        stage_id = _str(d, "stage_id")
        event = _str(d, "event")
        timestamp_utc = _str(d, "timestamp_utc")
        assert stage_id is not None and event is not None and timestamp_utc is not None
        _require(event in _LEDGER_EVENTS, f"event must be one of {_LEDGER_EVENTS}, got {event!r}")
        artifact_path = _str(d, "artifact_path", allow_none=True)
        sha256 = _str(d, "sha256", allow_none=True)
        note = _str(d, "note", allow_none=True)
        return cls(
            schema_version=schema_version or "stage_event_v1",
            stage_id=stage_id,
            event=event,
            artifact_path=artifact_path,
            sha256=sha256,
            timestamp_utc=timestamp_utc,
            note=note,
        )


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str
    offending_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "detail": self.detail,
            "offending_path": self.offending_path,
        }


@dataclass(frozen=True)
class ValidationReport:
    schema_version: str  # always "validation_report_v1"
    run_dir: str
    overall: str  # "pass" | "fail"
    rules: tuple[RuleResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_dir": self.run_dir,
            "overall": self.overall,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def build(cls, run_dir: str, rules: list[RuleResult]) -> ValidationReport:
        for r in rules:
            _require(r.status in _RULE_STATUSES, f"invalid rule status {r.status!r}")
        any_fail = any(r.status == "fail" for r in rules)
        overall = "fail" if any_fail else "pass"
        _require(overall in _OVERALL_STATUSES, "internal: overall must be pass|fail")
        return cls(
            schema_version="validation_report_v1",
            run_dir=run_dir,
            overall=overall,
            rules=tuple(rules),
        )
