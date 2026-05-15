"""M-91c — optional GPU providers carry typed blocked_proof when
their SDK isn't installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.execution_evidence import BLOCKED_PROOF_REASONS

OPTIONAL_GPU_PROVIDERS = (
    "tilelang",
    "cutlass_cute",
    "thunderkittens",
    "bitblas",
    "mirage",
    "kernelbench_caesar",
)


def test_shipped_evidence_pack_has_blocked_proof_for_each_optional_gpu_provider():
    pack = Path("results/extension_provider_evidence_pack")
    if not pack.is_dir():
        pytest.skip("no canonical evidence pack on disk")
    for pid in OPTIONAL_GPU_PROVIDERS:
        proof_path = pack / "per_provider" / pid / "blocked_proof.json"
        assert proof_path.is_file(), f"{pid}: no blocked_proof.json"
        body = json.loads(proof_path.read_text())
        assert body["provider_id"] == pid
        assert body["status"] in (
            "blocked",
            "unsupported",
            "probe_error",
            "not_installed",
        )
        assert body["blocked_reason"] in BLOCKED_PROOF_REASONS
        assert body["detail"]


def test_optional_gpu_summary_records_six_outcomes():
    """The script writes a summary file describing all 6 outcomes."""

    pack = Path("results/extension_provider_evidence_pack")
    summary = pack / "optional_gpu_blocked_proof_summary.json"
    if not summary.is_file():
        pytest.skip("optional GPU summary not yet generated on this machine")
    body = json.loads(summary.read_text())
    assert body["schema_version"] == "m91c_blocked_proof_summary_v1"
    ids = {o["provider_id"] for o in body["outcomes"]}
    assert ids == set(OPTIONAL_GPU_PROVIDERS)


def test_every_optional_gpu_blocked_proof_has_typed_reason():
    """Hard rule: every blocked_proof must use a typed blocked_reason
    enum value — no free-text reasons."""

    pack = Path("results/extension_provider_evidence_pack")
    if not pack.is_dir():
        pytest.skip("no canonical evidence pack on disk")
    for pid in OPTIONAL_GPU_PROVIDERS:
        proof_path = pack / "per_provider" / pid / "blocked_proof.json"
        if not proof_path.is_file():
            continue
        body = json.loads(proof_path.read_text())
        assert body["blocked_reason"] in BLOCKED_PROOF_REASONS, (
            f"{pid}: blocked_reason {body['blocked_reason']!r} not typed"
        )
