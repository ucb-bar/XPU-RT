"""provider_result_v1 schema + translator tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from compgen.kernels.provider import (
    ProviderResult as LegacyProviderResult,
)
from compgen.providers.result_v1 import (
    SCHEMA_VERSION,
    STATUSES,
    ProviderResultV1,
    ProviderResultV1Error,
    legacy_to_v1,
    load_result_v1,
)


# ---------------------------------------------------------------------------
# Schema discipline
# ---------------------------------------------------------------------------


def test_wrong_schema_version_rejected():
    with pytest.raises(ProviderResultV1Error, match="schema_version"):
        ProviderResultV1(
            schema_version="v999",
            task_id="x",
            provider_id="x",
            target_id="x",
            contract_hash="x",
            status="generated",
            artifacts={"source": "/tmp/x"},
        )


def test_unknown_status_rejected():
    with pytest.raises(ProviderResultV1Error, match="status"):
        ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id="x",
            provider_id="x",
            target_id="x",
            contract_hash="x",
            status="totally_made_up",
        )


def test_generated_requires_source_artifact():
    with pytest.raises(ProviderResultV1Error, match="source"):
        ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id="x",
            provider_id="x",
            target_id="x",
            contract_hash="x",
            status="generated",
            artifacts={},
        )


@pytest.mark.parametrize("status", ["blocked", "error", "contract_rejected"])
def test_non_generated_requires_detail(status: str):
    with pytest.raises(ProviderResultV1Error, match="detail"):
        ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id="x",
            provider_id="x",
            target_id="x",
            contract_hash="x",
            status=status,
        )


def test_statuses_are_typed_enum():
    assert STATUSES == ("generated", "contract_rejected", "blocked", "error")


def test_round_trip_through_json(tmp_path: Path):
    r = ProviderResultV1(
        schema_version=SCHEMA_VERSION,
        task_id="kcodegen_0007",
        provider_id="cffi_c",
        target_id="host_cpu",
        contract_hash="abc123",
        status="generated",
        artifacts={"source": "/tmp/k.c", "metadata": "/tmp/k.json"},
        claims={"estimated_latency_us": 1.5},
    )
    path = r.write(tmp_path / "result.json")
    restored = load_result_v1(path)
    assert restored == r


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


def test_legacy_found_false_translates_to_blocked():
    legacy = LegacyProviderResult(
        found=False,
        metadata={"reason": "OPENAI_API_KEY not set"},
    )
    v1 = legacy_to_v1(
        legacy,
        task_id="t",
        provider_id="kernelblaster",
        target_id="cuda_sm75",
        contract_hash="h",
    )
    assert v1.status == "blocked"
    assert "OPENAI_API_KEY" in v1.detail


def test_legacy_found_true_translates_to_generated(tmp_path: Path):
    legacy = LegacyProviderResult(
        found=True,
        correct=True,
        kernel_code="// some C code\n",
        language="c",
        latency_us=1.0,
        iterations_used=1,
        total_candidates=1,
        metadata={"provider": "cffi_c"},
    )
    v1 = legacy_to_v1(
        legacy,
        task_id="t",
        provider_id="cffi_c",
        target_id="host_cpu",
        contract_hash="h",
        artifact_dir=tmp_path,
    )
    assert v1.status == "generated"
    assert v1.artifacts["source"]
    assert Path(v1.artifacts["source"]).is_file()
    assert Path(v1.artifacts["source"]).read_text().startswith("// some C code")
    assert Path(v1.artifacts["metadata"]).is_file()
    assert v1.claims["estimated_latency_us"] == 1.0


def test_legacy_inf_latency_translates_to_none():
    legacy = LegacyProviderResult(
        found=True,
        kernel_code="x",
        language="python",
        latency_us=float("inf"),
    )
    v1 = legacy_to_v1(
        legacy,
        task_id="t",
        provider_id="x",
        target_id="x",
        contract_hash="h",
    )
    assert v1.claims["estimated_latency_us"] is None


def test_legacy_without_artifact_dir_inlines_source():
    legacy = LegacyProviderResult(
        found=True,
        kernel_code="inline source",
        language="python",
        latency_us=1.0,
    )
    v1 = legacy_to_v1(
        legacy,
        task_id="t",
        provider_id="x",
        target_id="x",
        contract_hash="h",
        artifact_dir=None,
    )
    assert v1.status == "generated"
    assert v1.artifacts["source"] == ""
    assert v1.claims["inline_source"] == "inline source"
