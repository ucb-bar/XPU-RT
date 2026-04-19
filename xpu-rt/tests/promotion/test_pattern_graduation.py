"""Tests for the P12 pattern-graduation API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.promotion import (
    PatternAppearance,
    PatternIdentity,
    PatternPromotionRequest,
    build_promotion_requests,
    graduate_from_transcripts,
    scan_transcripts,
)


def _write_entry(
    f: Path,
    *,
    workload: str,
    target: str,
    chosen: dict,
    slot: str = "propose_peephole_pattern",
    gate_status: str = "accepted",
    kind: str = "invent_proposal",
    target_feature_justification: str = "supported_kernel_families[?family=='gemm']",
) -> None:
    line = json.dumps(
        {
            "phase": 3,
            "llm_turn_id": "t0",
            "kind": kind,
            "name": slot,
            "args": {
                "workload": workload,
                "target": target,
                "target_feature_justification": target_feature_justification,
            },
            "result": {"chosen": chosen},
            "select_vs_invent": "invent",
            "recipe_ir_diff": {},
            "gate_result": {"status": gate_status},
            "timestamp_iso": "2026-04-17T00:00:00Z",
            "elapsed_ms": 10,
        },
        sort_keys=True,
    )
    with f.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def test_scan_extracts_accepted_invents(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_entry(p, workload="w1", target="t1", chosen={"rewrite": "A"})
    _write_entry(p, workload="w1", target="t1", chosen={"rewrite": "A"},
                 gate_status="rejected")  # should be ignored

    apps = scan_transcripts([p])
    assert len(apps) == 1
    assert apps[0].workload == "w1"
    assert apps[0].gate_status == "accepted"


def test_scan_ignores_non_invent(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    _write_entry(p, workload="w1", target="t1", chosen={}, kind="tool_call")
    apps = scan_transcripts([p])
    assert apps == []


def test_scan_handles_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    _write_entry(p, workload="w1", target="t1", chosen={"r": 1})
    apps = scan_transcripts([p])
    assert len(apps) == 1


def test_build_requests_respects_thresholds() -> None:
    identity = PatternIdentity(
        slot_name="propose_test",
        target_feature_justification="j",
        chosen_signature="sha256:abc",
    )

    apps = [
        PatternAppearance(identity, "w1", "t1", "", "a", "accepted"),
        PatternAppearance(identity, "w2", "t2", "", "b", "accepted"),
    ]
    # 2 workloads × 2 targets meets default threshold
    reqs = build_promotion_requests(apps, min_workloads=2, min_targets=2)
    assert len(reqs) == 1
    assert reqs[0].acceptance_count == 2
    assert reqs[0].workloads_proven == frozenset({"w1", "w2"})
    assert reqs[0].targets_proven == frozenset({"t1", "t2"})

    # Raising the workload threshold to 3 filters it out
    reqs3 = build_promotion_requests(apps, min_workloads=3, min_targets=2)
    assert reqs3 == []


def test_graduate_end_to_end(tmp_path: Path) -> None:
    t1 = tmp_path / "t1.jsonl"
    t2 = tmp_path / "t2.jsonl"

    CHOSEN = {"rewrite": "fused_attention_v1"}
    _write_entry(t1, workload="smolvla", target="bpi_f3", chosen=CHOSEN)
    _write_entry(t1, workload="smolvla", target="openq_5165rb", chosen=CHOSEN)
    _write_entry(t2, workload="gemma_decode", target="openq_5165rb", chosen=CHOSEN)
    _write_entry(t2, workload="gemma_decode", target="bpi_f3", chosen=CHOSEN)

    reqs = graduate_from_transcripts([t1, t2], min_workloads=2, min_targets=2)
    assert len(reqs) == 1
    r = reqs[0]
    assert r.identity.slot_name == "propose_peephole_pattern"
    assert r.acceptance_count == 4
    assert len(r.workloads_proven) == 2
    assert len(r.targets_proven) == 2


def test_graduate_no_cross_contamination(tmp_path: Path) -> None:
    """Different chosen payloads → different identities → independent graduation."""
    p = tmp_path / "t.jsonl"
    _write_entry(p, workload="w1", target="t1", chosen={"r": "A"})
    _write_entry(p, workload="w2", target="t2", chosen={"r": "B"})  # different chosen

    reqs = graduate_from_transcripts([p], min_workloads=2, min_targets=2)
    # Neither pattern individually meets the threshold
    assert reqs == []


def test_request_to_dict() -> None:
    identity = PatternIdentity(
        slot_name="propose_test",
        target_feature_justification="j",
        chosen_signature="sha256:abc",
    )
    req = PatternPromotionRequest(
        identity=identity,
        workloads_proven=frozenset({"w1", "w2"}),
        targets_proven=frozenset({"t1", "t2"}),
        first_seen_transcript="a",
        latest_seen_transcript="b",
        acceptance_count=4,
        chosen_exemplar={"r": "A"},
        graduation_threshold={"min_workloads": 2, "min_targets": 2},
    )
    d = req.to_dict()
    assert d["slot_name"] == "propose_test"
    assert sorted(d["workloads_proven"]) == ["w1", "w2"]


def test_scan_missing_file_is_silent(tmp_path: Path) -> None:
    apps = scan_transcripts([tmp_path / "does_not_exist.jsonl"])
    assert apps == []
