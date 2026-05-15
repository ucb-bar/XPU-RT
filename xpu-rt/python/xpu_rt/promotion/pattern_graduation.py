"""Pattern graduation — invent-slot → new TOOL (P12).

Scans ``ToolCallRecorder`` JSONL transcripts for accepted invent-slot
proposals and emits ``PatternPromotionRequest`` objects that the
human-review / automated-promotion layer consumes. This does NOT
automatically register new tools — graduation is deliberate per the
LLM-first IR principles.

Read the lifecycle:
    1. LLM emits a propose-op via an invent-slot (Phase 2/3/5).
    2. The slot's gate accepts or rejects; ToolCallRecorder records both.
    3. Over time, this module aggregates: for each pattern identity
       (kind-of-proposal + chosen-payload-signature), count the number
       of distinct workloads and targets it passed on.
    4. When ``min_workloads`` AND ``min_targets`` thresholds are met,
       a ``PatternPromotionRequest`` is emitted for review.
    5. Promotion reviewers implement a concrete pass and register it
       as a new TOOL in ``compgen.llm.registry``.

This module reads-only from transcript files; it never mutates them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternIdentity:
    """Stable key that groups "same pattern" across runs.

    Two invent-slot proposals map to the same PatternIdentity when
    they share the slot name, target_feature_justification, and a
    canonical signature of the ``chosen`` payload.
    """

    slot_name: str
    target_feature_justification: str
    chosen_signature: str  # sorted-keys hash of chosen dict


@dataclass(frozen=True)
class PatternAppearance:
    """One acceptance of a pattern — captured from a transcript entry."""

    identity: PatternIdentity
    workload: str
    target: str
    llm_turn_id: str
    transcript_path: str
    gate_status: str  # always 'accepted' for graduation-eligible appearances


@dataclass(frozen=True)
class PatternPromotionRequest:
    """Request to graduate a pattern from INVENT-SLOT to TOOL.

    Fields match the graduation contract in
    ``user_perspective/analysis/llm_first_ir_principles.md``.

    Attributes:
        identity: Stable pattern identity.
        workloads_proven: Set of workloads where this pattern passed.
        targets_proven: Set of targets where this pattern passed.
        first_seen_transcript: Path to the earliest transcript.
        latest_seen_transcript: Path to the most recent transcript.
        acceptance_count: Total number of accepted occurrences.
        chosen_exemplar: The ``chosen`` payload from the first acceptance
            (a concrete example a human reviewer can inspect).
        graduation_threshold: The thresholds that triggered graduation.
    """

    identity: PatternIdentity
    workloads_proven: frozenset[str]
    targets_proven: frozenset[str]
    first_seen_transcript: str
    latest_seen_transcript: str
    acceptance_count: int
    chosen_exemplar: dict[str, Any]
    graduation_threshold: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.identity.slot_name,
            "target_feature_justification": self.identity.target_feature_justification,
            "chosen_signature": self.identity.chosen_signature,
            "workloads_proven": sorted(self.workloads_proven),
            "targets_proven": sorted(self.targets_proven),
            "first_seen_transcript": self.first_seen_transcript,
            "latest_seen_transcript": self.latest_seen_transcript,
            "acceptance_count": self.acceptance_count,
            "chosen_exemplar": self.chosen_exemplar,
            "graduation_threshold": self.graduation_threshold,
        }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _chosen_signature(chosen: dict[str, Any]) -> str:
    """Stable signature of a ``chosen`` payload — sorted-key JSON hash."""
    import hashlib

    canonical = json.dumps(chosen, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _parse_entry(raw_line: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw_line)
    except json.JSONDecodeError:
        return None


def _is_accepted_invent(entry: dict[str, Any]) -> bool:
    if entry.get("kind") != "invent_proposal":
        return False
    gate = entry.get("gate_result") or {}
    return gate.get("status") == "accepted"


def scan_transcripts(
    transcript_paths: Iterable[Path],
    *,
    workload_field: str = "workload",
    target_field: str = "target",
) -> list[PatternAppearance]:
    """Read each JSONL transcript and extract accepted invent proposals.

    Transcripts are written by ``ToolCallRecorder``; each line is one
    record. ``workload`` and ``target`` are expected in the record's
    ``args`` (populated by the compile driver); if missing they fall
    back to the special value ``"unknown"``.
    """
    appearances: list[PatternAppearance] = []
    for tpath in transcript_paths:
        if not tpath.exists():
            continue
        for line in tpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = _parse_entry(line)
            if entry is None:
                continue
            if not _is_accepted_invent(entry):
                continue

            args = entry.get("args") or {}
            result = entry.get("result") or {}
            # Result may carry the chosen payload; fall back to args.
            chosen_raw = result.get("chosen")
            if chosen_raw is None:
                chosen_raw = args.get("chosen")
            if not isinstance(chosen_raw, dict):
                chosen_raw = {}
            target_justification = (
                args.get("target_feature_justification") or result.get("target_feature_justification") or ""
            )

            identity = PatternIdentity(
                slot_name=entry.get("name", "<unknown>"),
                target_feature_justification=target_justification,
                chosen_signature=_chosen_signature(chosen_raw),
            )
            appearances.append(
                PatternAppearance(
                    identity=identity,
                    workload=str(args.get(workload_field, "unknown")),
                    target=str(args.get(target_field, "unknown")),
                    llm_turn_id=str(entry.get("llm_turn_id", "")),
                    transcript_path=str(tpath),
                    gate_status=(entry.get("gate_result") or {}).get("status", "accepted"),
                )
            )
    return appearances


def build_promotion_requests(
    appearances: Iterable[PatternAppearance],
    *,
    min_workloads: int = 2,
    min_targets: int = 2,
    transcripts_by_identity: dict[PatternIdentity, list[dict[str, Any]]] | None = None,
) -> list[PatternPromotionRequest]:
    """Aggregate appearances and emit graduation requests.

    ``min_workloads`` / ``min_targets`` are the thresholds from
    ``llm_control_boundaries.md``. Defaults are conservative; raise
    them for higher-assurance graduation.
    """
    grouped: dict[PatternIdentity, list[PatternAppearance]] = defaultdict(list)
    for a in appearances:
        grouped[a.identity].append(a)

    requests: list[PatternPromotionRequest] = []
    for identity, apps in grouped.items():
        workloads = frozenset(a.workload for a in apps)
        targets = frozenset(a.target for a in apps)
        if len(workloads) < min_workloads or len(targets) < min_targets:
            continue
        sorted_paths = sorted(a.transcript_path for a in apps)
        exemplar: dict[str, Any] = {}
        # Find chosen_exemplar from transcripts if provided, else empty.
        if transcripts_by_identity and identity in transcripts_by_identity:
            for rec in transcripts_by_identity[identity]:
                chosen = (rec.get("result") or {}).get("chosen")
                if isinstance(chosen, dict):
                    exemplar = chosen
                    break
        requests.append(
            PatternPromotionRequest(
                identity=identity,
                workloads_proven=workloads,
                targets_proven=targets,
                first_seen_transcript=sorted_paths[0],
                latest_seen_transcript=sorted_paths[-1],
                acceptance_count=len(apps),
                chosen_exemplar=exemplar,
                graduation_threshold={
                    "min_workloads": min_workloads,
                    "min_targets": min_targets,
                },
            )
        )
    return requests


def graduate_from_transcripts(
    transcript_paths: Iterable[Path],
    *,
    min_workloads: int = 2,
    min_targets: int = 2,
) -> list[PatternPromotionRequest]:
    """Convenience: scan + build in one call."""
    appearances = scan_transcripts(transcript_paths)
    return build_promotion_requests(
        appearances,
        min_workloads=min_workloads,
        min_targets=min_targets,
    )


__all__ = [
    "PatternAppearance",
    "PatternIdentity",
    "PatternPromotionRequest",
    "build_promotion_requests",
    "graduate_from_transcripts",
    "scan_transcripts",
]
