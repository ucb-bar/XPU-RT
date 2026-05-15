"""Tests for compgen.llm.tools.verification."""

from __future__ import annotations

import torch
from compgen.llm import get_registry
from compgen.llm.tools import verification


def test_verification_tools_auto_registered() -> None:
    r = get_registry()
    names = {t.name for t in r.list_tools(phase=2)}
    assert "run_differential_test" in names
    assert "run_structural_check" in names


def test_differential_test_passes_on_matching_tensors() -> None:
    ref = torch.zeros(4)
    result = verification._run_differential_test_impl(
        ref_fn=lambda: ref.clone(),
        got_fn=lambda: ref.clone(),
    )
    assert result["status"] == "accepted"
    assert len(result["details"]["comparisons"]) == 1


def test_differential_test_rejects_on_mismatch() -> None:
    result = verification._run_differential_test_impl(
        ref_fn=lambda: torch.zeros(4),
        got_fn=lambda: torch.ones(4),
    )
    assert result["status"] == "rejected"
    # Max abs error should surface in comparisons
    assert result["details"]["comparisons"][0]["max_abs_error"] >= 1.0


def test_differential_test_rejects_when_ref_raises() -> None:
    def _boom() -> torch.Tensor:
        raise RuntimeError("boom")

    result = verification._run_differential_test_impl(ref_fn=_boom, got_fn=lambda: torch.zeros(4))
    assert result["status"] == "rejected"
    assert "ref_fn raised" in result["details"]["reason"]


def test_differential_test_count_mismatch() -> None:
    result = verification._run_differential_test_impl(
        ref_fn=lambda: (torch.zeros(4), torch.zeros(4)),
        got_fn=lambda: torch.zeros(4),
    )
    assert result["status"] == "rejected"
    assert "count" in result["details"]["reason"]


def test_structural_check_accepts_dict_with_schema_version() -> None:
    out = verification._run_structural_check_impl(artifact={"schema_version": "2.0", "anything": True})
    assert out["status"] == "accepted"


def test_structural_check_rejects_dict_missing_required() -> None:
    out = verification._run_structural_check_impl(artifact={"no_version": True})
    assert out["status"] == "rejected"
    assert "schema_version" in out["details"]["missing"]


def test_structural_check_rejects_unknown_type() -> None:
    out = verification._run_structural_check_impl(artifact=42)
    assert out["status"] == "rejected"
