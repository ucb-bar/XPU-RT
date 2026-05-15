"""Tests for the P2.6 online surrogate cost model."""

from __future__ import annotations

import pytest
from compgen.agent.cost_preview import CandidateInput, compute_cost_previews
from compgen.bench.surrogate import Sample, Surrogate


def _sample(fingerprint: str, cid: str, lat: float) -> Sample:
    return Sample(
        region_fingerprint=fingerprint,
        candidate_id=cid,
        measured_latency_us=lat,
    )


# ---------- Positive --------------------------------------------------


def test_empty_surrogate_returns_no_prediction():
    surr = Surrogate()
    pred = surr.predict(region_fingerprint="r1", candidate_id="c1")
    assert pred.predicted_latency_us is None
    assert pred.confidence == 0.0
    assert pred.n_neighbour_samples == 0


def test_tier1_exact_bucket_match():
    surr = Surrogate()
    surr.fit([
        _sample("r1", "c1", 10.0),
        _sample("r1", "c1", 12.0),
        _sample("r1", "c1", 14.0),
    ])
    pred = surr.predict(region_fingerprint="r1", candidate_id="c1")
    assert pred.predicted_latency_us == pytest.approx(12.0)
    assert pred.n_neighbour_samples == 3
    # Confidence proportional to sample count (capped at 20).
    assert pred.confidence == pytest.approx(3 / 20.0)


def test_tier2_per_candidate_fallback():
    surr = Surrogate()
    surr.fit([
        _sample("r_other", "c1", 100.0),
        _sample("r_other_too", "c1", 200.0),
    ])
    pred = surr.predict(region_fingerprint="r_new", candidate_id="c1")
    # We DON'T have r_new/c1 samples, so fall back to per-candidate
    # mean across all regions.
    assert pred.predicted_latency_us == pytest.approx(150.0)
    assert pred.confidence > 0.0
    assert pred.confidence <= 0.5  # half-credit tier


def test_tier3_global_mean_with_zero_confidence():
    surr = Surrogate()
    surr.fit([
        _sample("r1", "c_known", 50.0),
        _sample("r2", "c_known_too", 60.0),
    ])
    pred = surr.predict(region_fingerprint="r_unknown", candidate_id="c_unknown")
    assert pred.predicted_latency_us == pytest.approx(55.0)
    assert pred.confidence == 0.0  # honestly absent


def test_update_is_append_only():
    surr = Surrogate()
    surr.update(_sample("r1", "c1", 10.0))
    surr.update(_sample("r1", "c1", 20.0))
    assert surr.n_samples() == 2
    pred = surr.predict(region_fingerprint="r1", candidate_id="c1")
    assert pred.predicted_latency_us == pytest.approx(15.0)


def test_deterministic_across_reruns():
    samples = [
        _sample("r1", "c1", 10.0),
        _sample("r1", "c2", 5.0),
        _sample("r2", "c1", 12.0),
    ]
    s1 = Surrogate()
    s1.fit(samples)
    s2 = Surrogate()
    s2.fit(samples)
    p1 = s1.predict(region_fingerprint="r1", candidate_id="c1")
    p2 = s2.predict(region_fingerprint="r1", candidate_id="c1")
    assert p1.to_dict() == p2.to_dict()


def test_deltas_for_candidates_returns_dict():
    surr = Surrogate()
    surr.fit([_sample("r1", "a", 10.0), _sample("r1", "b", 20.0)])
    deltas = surr.deltas_for_candidates(
        region_fingerprint="r1", candidate_ids=["a", "b", "missing"]
    )
    assert deltas["a"] == pytest.approx(10.0)
    assert deltas["b"] == pytest.approx(20.0)
    # "missing" has no signal — but tier 3 global mean kicks in (15.0).
    # The presence in the output is acceptable; the CostPreview surfaces
    # it as a surrogate value with confidence=0.
    assert "missing" in deltas


def test_confidence_caps_at_one():
    surr = Surrogate(confidence_cap=5)
    # 10 samples on the same (region, candidate) pair.
    surr.fit([_sample("r1", "c1", 10.0)] * 10)
    pred = surr.predict(region_fingerprint="r1", candidate_id="c1")
    assert pred.confidence == pytest.approx(1.0)


# ---------- Integration with CostPreview -----------------------------


def test_cost_preview_consumes_surrogate_output():
    """End-to-end: surrogate produces deltas/confidence; CostPreview
    surfaces them on the right rows."""

    surr = Surrogate()
    surr.fit([
        _sample("r1", "a", 100.0),
        _sample("r1", "b", 50.0),
        _sample("r1", "c", 200.0),
    ])
    cands = [
        CandidateInput(candidate_id="a", delta_static=10.0),
        CandidateInput(candidate_id="b", delta_static=5.0),
        CandidateInput(candidate_id="c", delta_static=20.0),
    ]
    deltas = surr.deltas_for_candidates(
        region_fingerprint="r1", candidate_ids=[c.candidate_id for c in cands]
    )
    confs = surr.confidences_for_candidates(
        region_fingerprint="r1", candidate_ids=[c.candidate_id for c in cands]
    )
    previews = compute_cost_previews(
        cands, surrogate_deltas=deltas, confidence_by_id=confs
    )
    by_id = {p.candidate_id: p for p in previews}
    assert by_id["a"].delta_surrogate == pytest.approx(100.0)
    assert by_id["b"].delta_surrogate == pytest.approx(50.0)
    assert by_id["a"].confidence is not None
    assert 0.0 < by_id["a"].confidence <= 1.0


# ---------- Negative controls ----------------------------------------


def test_negative_latency_rejected():
    with pytest.raises(ValueError, match="measured_latency_us"):
        Sample(region_fingerprint="r", candidate_id="c", measured_latency_us=-1.0)
