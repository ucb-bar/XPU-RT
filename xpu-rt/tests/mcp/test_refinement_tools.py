"""Tests for ``compgen.mcp.tools.refinement`` (W8.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.mcp.session import SessionManager
from compgen.mcp.tools.refinement import (
    REFINEMENT_TOOLS,
    list_pending_refinements,
    lookup_refinement_history,
    register_refinement_attempt,
    request_refinement,
)


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


def test_refinement_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in REFINEMENT_TOOLS}
    assert names == {
        "request_refinement", "register_refinement_attempt",
        "lookup_refinement_history", "list_pending_refinements",
    }


def test_refinement_tools_in_all_tools_bundle() -> None:
    from compgen.mcp.tools import ALL_TOOLS
    names = {t["name"] for t in ALL_TOOLS}
    for n in ("request_refinement", "register_refinement_attempt",
              "lookup_refinement_history", "list_pending_refinements"):
        assert n in names


def test_request_then_register_then_lookup(sm) -> None:
    out = request_refinement(
        sm, session_id="sess1",
        kernel_fingerprint="fp1", prior_source="def kernel(x): return x",
        diagnosis_summary="memory-bound; bandwidth_efficiency=0.3",
        perf_target_us=50.0,
    )
    assert out["ok"] and not out["converged"]
    assert "PERF TARGET: ≤50.0μs" in out["prompt"]
    rid = out["request_id"]

    pending = list_pending_refinements(sm, session_id="sess1")
    assert pending["pending_count"] == 1

    reg = register_refinement_attempt(
        sm, session_id="sess1", request_id=rid,
        kernel_source="def kernel(x): return x * 2",
        perf_us=25.0, correct=True, done=False,
        rationale="vectorised the inner loop",
    )
    assert reg["ok"] and reg["attempt_count"] == 1 and not reg["converged"]

    hist = lookup_refinement_history(
        sm, session_id="sess1", kernel_fingerprint="fp1",
    )
    assert hist["attempt_count"] == 1
    assert hist["attempts"][0]["perf_us"] == 25.0
    assert "vectorised" in hist["attempts"][0]["rationale"]


def test_done_marks_kernel_converged_and_short_circuits_future_requests(sm) -> None:
    out = request_refinement(
        sm, session_id="sess1",
        kernel_fingerprint="fp2", prior_source="x", diagnosis_summary="",
    )
    register_refinement_attempt(
        sm, session_id="sess1", request_id=out["request_id"],
        kernel_source="def kernel(x): return x", perf_us=10.0,
        correct=True, done=True,
    )
    # Second request short-circuits because the kernel converged.
    out2 = request_refinement(
        sm, session_id="sess1", kernel_fingerprint="fp2",
        prior_source="anything", diagnosis_summary="",
    )
    assert out2["converged"] is True
    assert out2["last_perf_us"] == 10.0


def test_register_with_empty_source_requeues(sm) -> None:
    out = request_refinement(
        sm, session_id="sess1", kernel_fingerprint="fp3",
        prior_source="x", diagnosis_summary="",
    )
    res = register_refinement_attempt(
        sm, session_id="sess1", request_id=out["request_id"],
        kernel_source="",
    )
    assert res["ok"] is False
    assert "empty" in res["error"]
    assert list_pending_refinements(sm, session_id="sess1")["pending_count"] == 1


def test_register_unknown_request_id_errors(sm) -> None:
    res = register_refinement_attempt(
        sm, session_id="sess1", request_id="nope",
        kernel_source="def kernel(x): return x",
    )
    assert res["ok"] is False
    assert "unknown" in res["error"]


def test_multiple_attempts_accumulate_in_history(sm) -> None:
    fp = "fp_multi"
    for i in range(3):
        out = request_refinement(
            sm, session_id="sess1", kernel_fingerprint=fp,
            prior_source=f"v{i}", diagnosis_summary="",
        )
        register_refinement_attempt(
            sm, session_id="sess1", request_id=out["request_id"],
            kernel_source=f"def kernel(x): return x + {i}",
            perf_us=float(100 - i * 10),
        )
    hist = lookup_refinement_history(
        sm, session_id="sess1", kernel_fingerprint=fp,
    )
    assert hist["attempt_count"] == 3
    perfs = [a["perf_us"] for a in hist["attempts"]]
    assert perfs == [100.0, 90.0, 80.0]


def test_lookup_for_unknown_fingerprint_returns_empty_history(sm) -> None:
    hist = lookup_refinement_history(
        sm, session_id="sess1", kernel_fingerprint="never_seen",
    )
    assert hist["attempt_count"] == 0
    assert hist["converged"] is False
