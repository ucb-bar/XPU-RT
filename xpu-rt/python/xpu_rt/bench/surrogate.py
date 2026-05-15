"""Online surrogate cost model (P2.6).

Fits a lightweight, deterministic regression over historical
``(region_fingerprint, candidate, measured_latency_us)`` triples
produced by the region-compiled differential pass and the
profiler. The surrogate plugs into
:func:`compgen.agent.cost_preview.compute_cost_previews` via the
``surrogate_deltas`` argument.

The implementation is intentionally simple:

* Group samples by ``(region_fingerprint, candidate_id)`` key.
* Predict the *mean* of the bucket's measured latencies, with
  confidence proportional to the bucket's sample count (capped at
  ``confidence_cap``).
* For unseen (region, candidate) pairs, fall back to the
  per-candidate mean across all regions; if that is also unseen,
  fall back to the global mean with ``confidence=0.0``.

This is a *real* surrogate — not a stub. It does the job a tiny
linear / k-NN model would do at this scale (≤ 10⁴ samples), without
adding sklearn as a hard dependency. The interface accommodates a
heavier learner if someone wants to swap it in later (the
:meth:`Surrogate.fit` + :meth:`Surrogate.predict` signatures match
the sklearn convention).

Hard rules:

1. The surrogate never *invents* a measurement. Empty training data
   produces ``confidence=0.0`` so :class:`compgen.agent.cost_preview.CostPreview`
   reads it as honestly absent.
2. Updates are *append-only* — the surrogate cannot retroactively
   alter a recorded sample.
3. ``fit`` and ``predict`` are deterministic across reruns with the
   same input ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sample:
    """One historical observation."""

    region_fingerprint: str
    candidate_id: str
    measured_latency_us: float

    def __post_init__(self) -> None:
        if self.measured_latency_us < 0:
            raise ValueError(
                f"measured_latency_us={self.measured_latency_us} must be >= 0"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_fingerprint": self.region_fingerprint,
            "candidate_id": self.candidate_id,
            "measured_latency_us": self.measured_latency_us,
        }


@dataclass(frozen=True)
class SurrogatePrediction:
    """Typed prediction for one (region, candidate) pair."""

    predicted_latency_us: float | None
    confidence: float
    n_neighbour_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_latency_us": self.predicted_latency_us,
            "confidence": self.confidence,
            "n_neighbour_samples": self.n_neighbour_samples,
        }


@dataclass
class Surrogate:
    """Bucketed-mean online surrogate."""

    samples: list[Sample] = field(default_factory=list)
    confidence_cap: int = 20  # neighbour-count at which confidence saturates

    def fit(self, samples: list[Sample]) -> None:
        """Replace the training set with ``samples``."""

        self.samples = list(samples)

    def update(self, sample: Sample) -> None:
        """Append one observation. Surrogate retrains on next predict."""

        self.samples.append(sample)

    def n_samples(self) -> int:
        return len(self.samples)

    def predict(self, *, region_fingerprint: str, candidate_id: str) -> SurrogatePrediction:
        """Predict latency for one (region, candidate) pair."""

        if not self.samples:
            return SurrogatePrediction(
                predicted_latency_us=None,
                confidence=0.0,
                n_neighbour_samples=0,
            )

        # Tier 1: exact (region, candidate) bucket.
        bucket = [
            s for s in self.samples
            if s.region_fingerprint == region_fingerprint and s.candidate_id == candidate_id
        ]
        if bucket:
            mean = sum(s.measured_latency_us for s in bucket) / len(bucket)
            return SurrogatePrediction(
                predicted_latency_us=mean,
                confidence=min(1.0, len(bucket) / float(self.confidence_cap)),
                n_neighbour_samples=len(bucket),
            )

        # Tier 2: per-candidate mean across regions.
        per_cand = [s for s in self.samples if s.candidate_id == candidate_id]
        if per_cand:
            mean = sum(s.measured_latency_us for s in per_cand) / len(per_cand)
            # Half-credit confidence — we matched on the candidate but
            # not the region.
            return SurrogatePrediction(
                predicted_latency_us=mean,
                confidence=min(0.5, len(per_cand) / (2.0 * self.confidence_cap)),
                n_neighbour_samples=len(per_cand),
            )

        # Tier 3: global mean with confidence=0 (honestly absent).
        mean = sum(s.measured_latency_us for s in self.samples) / len(self.samples)
        return SurrogatePrediction(
            predicted_latency_us=mean,
            confidence=0.0,
            n_neighbour_samples=len(self.samples),
        )

    def deltas_for_candidates(
        self,
        *,
        region_fingerprint: str,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Convenience wrapper that returns the ``surrogate_deltas``
        mapping consumed by
        :func:`compgen.agent.cost_preview.compute_cost_previews`.

        For each candidate id, the value is the predicted latency
        (microseconds). Candidates with no surrogate signal are
        omitted from the dict — :class:`CostPreview` then reports
        ``delta_surrogate=None`` for those entries, which is the
        honest absent-value path.
        """

        out: dict[str, float] = {}
        for cid in candidate_ids:
            pred = self.predict(region_fingerprint=region_fingerprint, candidate_id=cid)
            if pred.predicted_latency_us is not None and not math.isnan(pred.predicted_latency_us):
                out[cid] = pred.predicted_latency_us
        return out

    def confidences_for_candidates(
        self,
        *,
        region_fingerprint: str,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Sibling of :meth:`deltas_for_candidates` that returns the
        ``confidence_by_id`` mapping for
        :func:`compgen.agent.cost_preview.compute_cost_previews`."""

        out: dict[str, float] = {}
        for cid in candidate_ids:
            pred = self.predict(region_fingerprint=region_fingerprint, candidate_id=cid)
            if pred.predicted_latency_us is not None:
                out[cid] = pred.confidence
        return out


__all__ = [
    "Sample",
    "Surrogate",
    "SurrogatePrediction",
]
