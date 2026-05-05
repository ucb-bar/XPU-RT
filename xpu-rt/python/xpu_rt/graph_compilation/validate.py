"""Independent validator for capture/lower run directories.

The validator is intentionally suspicious: every rule recomputes from
disk rather than trusting the manifest's own claims. The output is a
:class:`ValidationReport` with one :class:`RuleResult` per rule R001
through R012.

Rule semantics are documented in the plan
(``/home/agustin/.claude/plans/bright-seeking-wadler.md``); the
authoritative one-line summaries live as docstrings on each
``_check_*`` function below.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonschema

from compgen.graph_compilation.artifacts import (
    CANONICAL_STAGE_ORDER,
    ArtifactRef,
    RuleResult,
    RunManifest,
    SchemaError,
    StageEvent,
    StageRecord,
    ValidationReport,
)
from compgen.graph_compilation.hashing import (
    SymlinkEscapeError,
    sha256_file,
    sha256_tree,
)
from compgen.graph_compilation.schemas import load_schema

# Order matters for the report. Each entry is (rule_id, summary).
RULE_IDS: tuple[tuple[str, str], ...] = (
    ("R001_manifest_schema", "run_manifest.json exists and matches run_manifest_v1"),
    ("R002_ledger_schema", "stage_ledger.jsonl exists; every line matches stage_event_v1"),
    ("R003_stage_order", "stages are a strict prefix of the canonical order"),
    ("R004_artifact_paths", "every artifact path resolves inside run_dir"),
    ("R005_artifact_hashes", "every artifact sha256 matches recomputed hash"),
    ("R006_report_paths", "every stage's report_path exists and parses as JSON"),
    ("R007_pass_outputs", "stages with status==pass have at least one output"),
    ("R008_no_llm_calls", "stages with status==pass have llm_calls==0"),
    ("R009_hash_chain", "stage N's input_hash matches stage N-1's output_hash"),
    ("R010_ledger_completeness", "ledger has start+finish events for every manifest stage"),
    ("R011_git_commit_format", "git_commit, if non-null, is a 40-char lowercase hex sha"),
    ("R012_unique_stage_ids", "no stage_id appears more than once"),
)

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_run(run_dir: Path | str) -> ValidationReport:
    """Validate a graph compilation run directory and return a report.

    Never raises for run-content errors — encodes them as ``fail``
    rules. Internal logic errors do propagate.
    """
    run_path = Path(run_dir)
    rules: list[RuleResult] = []

    # --- R001: manifest schema ---
    manifest_path = run_path / "run_manifest.json"
    manifest_obj: dict[str, Any] | None = None
    manifest: RunManifest | None = None
    r001 = _check_manifest_schema(manifest_path)
    rules.append(r001)
    if r001.status == "pass":
        with manifest_path.open(encoding="utf-8") as f:
            manifest_obj = json.load(f)
        try:
            assert manifest_obj is not None
            manifest = RunManifest.from_dict(manifest_obj)
        except SchemaError as exc:
            # The schema check passed but the dataclass converter
            # caught a logical violation. Downgrade R001 to fail.
            rules[-1] = RuleResult(
                rule_id="R001_manifest_schema",
                status="fail",
                detail=f"manifest dataclass conversion failed: {exc}",
                offending_path=str(manifest_path),
            )

    # --- R002: ledger schema ---
    ledger_path = run_path / "stage_ledger.jsonl"
    events: list[StageEvent] = []
    r002 = _check_ledger_schema(ledger_path, events)
    rules.append(r002)

    # If the manifest could not be loaded, we still run the disk-only
    # checks but skip everything that needs the manifest object.
    if manifest is None:
        for rule_id, summary in RULE_IDS[2:]:
            rules.append(
                RuleResult(
                    rule_id=rule_id,
                    status="skipped",
                    detail=f"skipped because R001 failed: {summary}",
                )
            )
        return ValidationReport.build(str(run_path), rules)

    # --- R003: stage order ---
    rules.append(_check_stage_order(manifest))
    # --- R004: artifact paths inside run_dir ---
    rules.append(_check_artifact_paths(run_path, manifest))
    # --- R005: artifact hashes ---
    rules.append(_check_artifact_hashes(run_path, manifest))
    # --- R006: report paths ---
    rules.append(_check_report_paths(run_path, manifest))
    # --- R007: pass stages have outputs ---
    rules.append(_check_pass_outputs(manifest))
    # --- R008: no llm calls ---
    rules.append(_check_no_llm_calls(manifest))
    # --- R009: hash chain ---
    rules.append(_check_hash_chain(manifest))
    # --- R010: ledger completeness ---
    rules.append(_check_ledger_completeness(manifest, events))
    # --- R011: git commit format ---
    rules.append(_check_git_commit_format(manifest))
    # --- R012: unique stage ids ---
    rules.append(_check_unique_stage_ids(manifest))

    return ValidationReport.build(str(run_path), rules)


# --------------------------------------------------------------------------- #
# Individual rule checks
# --------------------------------------------------------------------------- #


def _check_manifest_schema(manifest_path: Path) -> RuleResult:
    """R001: manifest exists and validates against run_manifest_v1."""
    if not manifest_path.exists():
        return RuleResult(
            rule_id="R001_manifest_schema",
            status="fail",
            detail="run_manifest.json is missing",
            offending_path=str(manifest_path),
        )
    try:
        with manifest_path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as exc:
        return RuleResult(
            rule_id="R001_manifest_schema",
            status="fail",
            detail=f"run_manifest.json is not valid JSON: {exc}",
            offending_path=str(manifest_path),
        )
    schema = load_schema("run_manifest")
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as exc:
        return RuleResult(
            rule_id="R001_manifest_schema",
            status="fail",
            detail=f"manifest fails run_manifest_v1: {exc.message}",
            offending_path=str(manifest_path),
        )
    return RuleResult(
        rule_id="R001_manifest_schema",
        status="pass",
        detail="manifest matches run_manifest_v1",
    )


def _check_ledger_schema(ledger_path: Path, out_events: list[StageEvent]) -> RuleResult:
    """R002: ledger exists and every line matches stage_event_v1."""
    if not ledger_path.exists():
        return RuleResult(
            rule_id="R002_ledger_schema",
            status="fail",
            detail="stage_ledger.jsonl is missing",
            offending_path=str(ledger_path),
        )
    schema = load_schema("stage_event")
    with ledger_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                return RuleResult(
                    rule_id="R002_ledger_schema",
                    status="fail",
                    detail=f"line {line_no}: invalid JSON: {exc}",
                    offending_path=str(ledger_path),
                )
            try:
                jsonschema.validate(obj, schema)
            except jsonschema.ValidationError as exc:
                return RuleResult(
                    rule_id="R002_ledger_schema",
                    status="fail",
                    detail=f"line {line_no}: {exc.message}",
                    offending_path=str(ledger_path),
                )
            try:
                out_events.append(StageEvent.from_dict(obj))
            except SchemaError as exc:
                return RuleResult(
                    rule_id="R002_ledger_schema",
                    status="fail",
                    detail=f"line {line_no}: {exc}",
                    offending_path=str(ledger_path),
                )
    return RuleResult(
        rule_id="R002_ledger_schema",
        status="pass",
        detail=f"ledger has {len(out_events)} valid event(s)",
    )


def _check_stage_order(manifest: RunManifest) -> RuleResult:
    """R003: the manifest's stage_ids are an ordered subsequence of the
    canonical order (each stage_id appears at most once and in canonical
    order). This permits optional stages — e.g. ``graph_analysis`` —
    to be skipped while still preserving canonical ordering."""
    actual = tuple(s.stage_id for s in manifest.stages)
    if not actual:
        return RuleResult(
            rule_id="R003_stage_order",
            status="fail",
            detail="manifest has zero stages",
        )
    # Walk CANONICAL_STAGE_ORDER and consume actual in lockstep; any
    # actual stage_id not found in canonical, or out of order, is a fail.
    canon_idx = 0
    for sid in actual:
        while canon_idx < len(CANONICAL_STAGE_ORDER) and CANONICAL_STAGE_ORDER[canon_idx] != sid:
            canon_idx += 1
        if canon_idx >= len(CANONICAL_STAGE_ORDER):
            return RuleResult(
                rule_id="R003_stage_order",
                status="fail",
                detail=(
                    f"stage_id {sid!r} not found in canonical order or out of order; "
                    f"actual={list(actual)}; canonical={list(CANONICAL_STAGE_ORDER)}"
                ),
            )
        canon_idx += 1
    if len(set(actual)) != len(actual):
        dupes = [s for s in actual if actual.count(s) > 1]
        return RuleResult(
            rule_id="R003_stage_order",
            status="fail",
            detail=f"duplicate stage_ids in actual order: {sorted(set(dupes))}",
        )
    return RuleResult(
        rule_id="R003_stage_order",
        status="pass",
        detail=f"stage order is {list(actual)}",
    )


def _all_refs(manifest: RunManifest) -> Iterable[ArtifactRef]:
    for stage in manifest.stages:
        yield from stage.inputs
        yield from stage.outputs


def _check_artifact_paths(run_path: Path, manifest: RunManifest) -> RuleResult:
    """R004: every ArtifactRef.path resolves to a file/tree inside run_dir."""
    run_resolved = run_path.resolve()
    if not run_resolved.exists():
        return RuleResult(
            rule_id="R004_artifact_paths",
            status="fail",
            detail="run_dir does not exist on disk",
            offending_path=str(run_path),
        )
    for ref in _all_refs(manifest):
        if ref.path.startswith("/") or ".." in Path(ref.path).parts:
            return RuleResult(
                rule_id="R004_artifact_paths",
                status="fail",
                detail=f"path {ref.path!r} is absolute or escapes run_dir",
                offending_path=ref.path,
            )
        candidate = run_path / ref.path
        if not candidate.exists():
            return RuleResult(
                rule_id="R004_artifact_paths",
                status="fail",
                detail=f"artifact {ref.path!r} does not exist",
                offending_path=str(candidate),
            )
        # Resolve and confirm we stay inside run_dir.
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(run_resolved)
        except (FileNotFoundError, ValueError):
            return RuleResult(
                rule_id="R004_artifact_paths",
                status="fail",
                detail=f"artifact {ref.path!r} resolves outside run_dir",
                offending_path=str(candidate),
            )
        if ref.kind == "file" and not resolved.is_file():
            return RuleResult(
                rule_id="R004_artifact_paths",
                status="fail",
                detail=f"artifact {ref.path!r} declared as file but is not a file",
                offending_path=str(candidate),
            )
        if ref.kind == "tree" and not resolved.is_dir():
            return RuleResult(
                rule_id="R004_artifact_paths",
                status="fail",
                detail=f"artifact {ref.path!r} declared as tree but is not a directory",
                offending_path=str(candidate),
            )
    return RuleResult(
        rule_id="R004_artifact_paths",
        status="pass",
        detail="all artifact paths resolve inside run_dir",
    )


def _check_artifact_hashes(run_path: Path, manifest: RunManifest) -> RuleResult:
    """R005: every artifact's sha256 matches the recomputed hash."""
    for ref in _all_refs(manifest):
        candidate = run_path / ref.path
        if not candidate.exists():
            # R004 will already have flagged this; skip cleanly.
            return RuleResult(
                rule_id="R005_artifact_hashes",
                status="fail",
                detail=f"cannot hash missing artifact {ref.path!r}",
                offending_path=str(candidate),
            )
        try:
            if ref.kind == "file":
                actual = sha256_file(candidate)
            else:
                actual = sha256_tree(candidate)
        except SymlinkEscapeError as exc:
            return RuleResult(
                rule_id="R005_artifact_hashes",
                status="fail",
                detail=f"hashing {ref.path!r} failed: {exc}",
                offending_path=str(candidate),
            )
        if actual != ref.sha256:
            return RuleResult(
                rule_id="R005_artifact_hashes",
                status="fail",
                detail=f"sha256 mismatch for {ref.path!r}: declared {ref.sha256}, actual {actual}",
                offending_path=str(candidate),
            )
    return RuleResult(
        rule_id="R005_artifact_hashes",
        status="pass",
        detail="all artifact hashes match",
    )


def _check_report_paths(run_path: Path, manifest: RunManifest) -> RuleResult:
    """R006: every stage's report_path exists and parses as JSON."""
    for stage in manifest.stages:
        candidate = run_path / stage.report_path
        if not candidate.exists():
            return RuleResult(
                rule_id="R006_report_paths",
                status="fail",
                detail=f"stage {stage.stage_id!r} report missing: {stage.report_path}",
                offending_path=str(candidate),
            )
        if candidate.stat().st_size == 0:
            return RuleResult(
                rule_id="R006_report_paths",
                status="fail",
                detail=f"stage {stage.stage_id!r} report is empty",
                offending_path=str(candidate),
            )
        try:
            with candidate.open(encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as exc:
            return RuleResult(
                rule_id="R006_report_paths",
                status="fail",
                detail=f"stage {stage.stage_id!r} report is not JSON: {exc}",
                offending_path=str(candidate),
            )
    return RuleResult(
        rule_id="R006_report_paths",
        status="pass",
        detail="all stage reports exist and parse as JSON",
    )


def _check_pass_outputs(manifest: RunManifest) -> RuleResult:
    """R007: stages with status==pass declare at least one output."""
    for stage in manifest.stages:
        if stage.status == "pass" and not stage.outputs:
            return RuleResult(
                rule_id="R007_pass_outputs",
                status="fail",
                detail=f"stage {stage.stage_id!r} status==pass but has no outputs",
            )
    return RuleResult(
        rule_id="R007_pass_outputs",
        status="pass",
        detail="every passed stage has at least one output",
    )


def _check_no_llm_calls(manifest: RunManifest) -> RuleResult:
    """R008: deterministic invariant — no llm calls in graph compilation stages."""
    for stage in manifest.stages:
        if stage.status == "pass" and stage.llm_calls != 0:
            return RuleResult(
                rule_id="R008_no_llm_calls",
                status="fail",
                detail=f"stage {stage.stage_id!r} has llm_calls={stage.llm_calls} but graph compilation stages must be llm-free",
            )
    return RuleResult(
        rule_id="R008_no_llm_calls",
        status="pass",
        detail="no llm calls in any passed stage",
    )


def _check_hash_chain(manifest: RunManifest) -> RuleResult:
    """R009: stage N's input_hash matches stage N-1's output_hash."""
    prev: StageRecord | None = None
    for stage in manifest.stages:
        if prev is not None and stage.input_hash != prev.output_hash:
            return RuleResult(
                rule_id="R009_hash_chain",
                status="fail",
                detail=(
                    f"hash chain break: {stage.stage_id!r} input_hash={stage.input_hash} "
                    f"!= {prev.stage_id!r} output_hash={prev.output_hash}"
                ),
            )
        prev = stage
    return RuleResult(
        rule_id="R009_hash_chain",
        status="pass",
        detail="hash chain intact",
    )


def _check_ledger_completeness(manifest: RunManifest, events: list[StageEvent]) -> RuleResult:
    """R010: ledger has start+finish events for every stage in the manifest."""
    by_stage: dict[str, set[str]] = {}
    for ev in events:
        by_stage.setdefault(ev.stage_id, set()).add(ev.event)
    for stage in manifest.stages:
        seen = by_stage.get(stage.stage_id, set())
        missing = {"start", "finish"} - seen
        if missing:
            return RuleResult(
                rule_id="R010_ledger_completeness",
                status="fail",
                detail=f"stage {stage.stage_id!r} missing ledger events: {sorted(missing)}",
            )
    return RuleResult(
        rule_id="R010_ledger_completeness",
        status="pass",
        detail="ledger contains start+finish for every manifest stage",
    )


def _check_git_commit_format(manifest: RunManifest) -> RuleResult:
    """R011: git_commit is null or a 40-char lowercase hex sha."""
    if manifest.git_commit is None:
        return RuleResult(
            rule_id="R011_git_commit_format",
            status="pass",
            detail="git_commit is null (offline rerun allowed)",
        )
    if not _SHA40_RE.match(manifest.git_commit):
        return RuleResult(
            rule_id="R011_git_commit_format",
            status="fail",
            detail=f"git_commit {manifest.git_commit!r} is not a 40-char lowercase hex sha",
        )
    return RuleResult(
        rule_id="R011_git_commit_format",
        status="pass",
        detail="git_commit format ok",
    )


def _check_unique_stage_ids(manifest: RunManifest) -> RuleResult:
    """R012: no stage_id appears more than once in the manifest."""
    seen: set[str] = set()
    for stage in manifest.stages:
        if stage.stage_id in seen:
            return RuleResult(
                rule_id="R012_unique_stage_ids",
                status="fail",
                detail=f"duplicate stage_id {stage.stage_id!r}",
            )
        seen.add(stage.stage_id)
    return RuleResult(
        rule_id="R012_unique_stage_ids",
        status="pass",
        detail="all stage_ids are unique",
    )
