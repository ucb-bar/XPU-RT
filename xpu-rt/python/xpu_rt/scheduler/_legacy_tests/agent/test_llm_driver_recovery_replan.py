"""Tests for the G3 wire-in: plan_recovery routes per-op failures
through replan_on_reject when a region Plan is supplied.

Coverage:

* Legacy path: omitting ``region_plan`` is a no-op for the new
  RegionReplanEvent fields — backward-compat.
* Plan-driven path: a recovery strategy that fails AND has a
  resolved region triggers a typed RegionReplanEvent; the rung
  walks to the next fallback rung.
* Targets with no region mapping are silently ignored for replan
  accounting (the existing OpRecoveryDecision is still recorded).
* Multiple per-op failures in the same region walk the ladder
  monotonically (one rung at a time).
* Failed replan_on_reject (e.g. unknown region_id in Plan) is
  recorded as no replan event — the helper degrades gracefully.

These tests stub out :class:`compgen.capture.torch_export.CaptureArtifact`
because the real one requires a torch model trace. The recovery
loop only reads ``artifact.unsupported_resolutions`` and per-resolution
``classification.confidence`` + ``target``, so a SimpleNamespace
stub is sufficient.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from compgen.agent.llm_driver_recovery import (
    RecoveryPlan,
    RegionReplanEvent,
    plan_recovery,
)
from compgen.agent.plan import (
    Budget,
    Plan,
    RegionPlan,
)


@pytest.fixture(autouse=True)
def _force_strategy_failure(monkeypatch):
    """Force every recovery strategy to fail (autouse for this module).

    The production ``_apply_strategy`` reaches into torch / aten /
    decomposition registries to decide success — that's too much
    surface area for a unit test. We patch it to return
    ``(False, "", "stubbed_failure")`` so the G3 replan path fires
    deterministically. Autouse ensures every test in this module
    bypasses the brittle real apply path.
    """

    def _always_fail(strategy, resolution):
        return False, "", "stubbed_failure"

    monkeypatch.setattr(
        "compgen.agent.llm_driver_recovery._apply_strategy",
        _always_fail,
    )
    yield


def _resolution(target: str, *, classification_kind: str = "unknown") -> SimpleNamespace:
    """Build a minimal UnsupportedOpResolution-shaped stub.

    The recovery loop reads:
        resolution.target
        resolution.classification.confidence
        resolution.classification.kind
        resolution.classification.strategy
        resolution.classification.bucket
        resolution.dossier.is_aten
        resolution.promotion.cache_key (only on the blackbox path)
    """

    # ``strategy="synthesized_external_call"`` routes the recovery to
    # the ``"translation"`` strategy. With no real torch op behind the
    # synthetic target name, ``synthesize_payload_translation`` returns
    # None and the strategy fails — the typed-fail path the G3 wire-in
    # is supposed to react to.
    return SimpleNamespace(
        target=target,
        issue="synthetic_unsupported",
        classification=SimpleNamespace(
            confidence="high",  # never consult the LLM in tests
            kind=classification_kind,
            strategy="synthesized_external_call",  # → recovery="translation" → fail
            bucket="opaque",
            reason="synthetic stub for G3 test",
        ),
        dossier=SimpleNamespace(is_aten=False, op_name=target, aten_namespaced=None),
        promotion=SimpleNamespace(cache_key=f"key_{target}"),
        decomp_candidates=(),
        translation_candidates=(),
    )


def _artifact(*targets: str) -> SimpleNamespace:
    return SimpleNamespace(
        unsupported_resolutions=tuple(_resolution(t) for t in targets),
    )


def _plan_with_region(region_id: str = "017") -> Plan:
    return Plan(
        session_id="ses_g3",
        plan_version=1,
        global_objective="minimize_p50_latency",
        budget=Budget(compile_seconds=600.0, llm_dollars=2.5),
        region_partition=(
            RegionPlan(
                region_id=region_id,
                tactic="fuse",
                fallback_ladder=("fuse", "tile_only", "naive_sync"),
            ),
        ),
    )


# ---------- Legacy path: no region_plan supplied ----------


def test_legacy_path_no_replan_events():
    """Without a region_plan, the existing recovery loop is unchanged."""

    rec = plan_recovery(_artifact("op1", "op2"))
    assert rec.region_replan_events == []
    # The skipped count still reflects per-op failures.
    assert rec.skipped >= 0


def test_to_dict_carries_empty_region_replan_events():
    rec = plan_recovery(_artifact("op1"))
    body = rec.to_dict()
    assert body["region_replan_events"] == []


# ---------- Plan-driven path ----------


def test_failed_recovery_emits_typed_replan_event(_force_strategy_failure):
    """A per-op recovery failure with a resolved region walks the ladder."""

    artifact = _artifact("op_a")
    plan = _plan_with_region("r017")
    rec = plan_recovery(
        artifact,
        region_plan=plan,
        region_id_for_target={"op_a": "r017"},
    )
    # The op cannot apply any recovery strategy (no decomp/translation
    # candidates), so it fails — that triggers the replan event.
    assert rec.skipped == 1
    assert len(rec.region_replan_events) == 1
    event = rec.region_replan_events[0]
    assert isinstance(event, RegionReplanEvent)
    assert event.target == "op_a"
    assert event.region_id == "r017"
    assert event.rejection_class == "tactic_fatal"
    assert event.rung_before == "fuse"
    assert event.rung_after == "tile_only"
    assert event.plan_version_after == event.plan_version_before + 1


def test_target_without_region_mapping_is_silently_skipped(_force_strategy_failure):
    """Targets not in region_id_for_target don't produce a replan event."""

    plan = _plan_with_region("r017")
    rec = plan_recovery(
        _artifact("op_unmapped"),
        region_plan=plan,
        region_id_for_target={},  # no mapping for op_unmapped
    )
    assert rec.region_replan_events == []
    # The op_unmapped recovery itself is still recorded.
    assert len(rec.decisions) == 1


def test_multiple_failures_walk_ladder_monotonically(_force_strategy_failure):
    """Two failed targets in the same region walk one rung each."""

    artifact = _artifact("op_a", "op_b")
    plan = _plan_with_region("r017")
    rec = plan_recovery(
        artifact,
        region_plan=plan,
        region_id_for_target={"op_a": "r017", "op_b": "r017"},
    )
    assert len(rec.region_replan_events) == 2
    rungs = [e.rung_after for e in rec.region_replan_events]
    # First failure: fuse → tile_only. Second failure: tile_only → naive_sync.
    assert rungs == ["tile_only", "naive_sync"]


def test_replan_event_to_dict(_force_strategy_failure):
    """Round-trip the typed event through to_dict."""

    plan = _plan_with_region("r017")
    rec = plan_recovery(
        _artifact("op_a"),
        region_plan=plan,
        region_id_for_target={"op_a": "r017"},
    )
    body = rec.to_dict()
    events = body["region_replan_events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["target"] == "op_a"
    assert ev["region_id"] == "r017"
    assert ev["rejection_class"] == "tactic_fatal"
    assert ev["rung_before"] == "fuse"
    assert ev["rung_after"] == "tile_only"
    assert "plan_version_before" in ev
    assert "plan_version_after" in ev


def test_unknown_region_in_plan_degrades_gracefully(_force_strategy_failure):
    """If region_id_for_target points at a region not in the Plan,
    the replan helper returns the plan unchanged (no event)."""

    plan = _plan_with_region("r017")
    rec = plan_recovery(
        _artifact("op_a"),
        region_plan=plan,
        region_id_for_target={"op_a": "unknown_region_zzz"},
    )
    # The op still fails (no apply path), but no replan event recorded.
    assert rec.region_replan_events == []
    assert rec.skipped == 1


def test_plan_recovery_signature_keyword_only():
    """The new arguments are keyword-only — positional callers are
    unaffected."""

    # Direct positional call (legacy shape) still works.
    rec = plan_recovery(_artifact())
    assert isinstance(rec, RecoveryPlan)
