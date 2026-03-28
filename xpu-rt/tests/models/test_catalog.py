"""Tests for the heavyweight model catalog."""

from __future__ import annotations

from compgen.models import build_default_model_catalog


def test_default_model_catalog_contains_frontier_entries() -> None:
    catalog = build_default_model_catalog()

    assert "llama31_decoder_block" in catalog.models
    assert "smolvla_one_step" in catalog.models
    assert "groot_policy_step" in catalog.models


def test_smolvla_metadata_is_capture_ready() -> None:
    catalog = build_default_model_catalog()
    spec = catalog.get("smolvla_one_step")

    assert spec.source_model_id == "lerobot/smolvla_base"
    assert spec.capture_mode == "torch_dynamo_partitioned"
    assert spec.readiness == "analysis_only"
    assert spec.expected_status == "pass"
