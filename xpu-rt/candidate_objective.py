"""Deciding whether a compiler change is an improvement, for a periodic workload.

The objective this replaces
---------------------------
The existing decision loop accepts a candidate when the *sum of per-op standalone
cycles of one network* goes down. That is wrong here in three separate ways, and
each one is worth naming because each produces a confident wrong answer:

* **It is a serial sum, so it cannot see parallelism.** Splitting an op to run
  on two cores makes the summed cycles *worse* -- measured on this project, an
  OC=8 tile costs 76% of the OC=16 op, so a 2-way split inflates total work
  ~53%. A criterion on summed cycles rejects every parallelism win by
  construction, including the one that takes DroNet from missing every deadline
  to nearly meeting its rate.
* **It looks at one network.** A fusion can make a model faster in isolation
  while creating a long non-preemptible dispatch that wrecks a co-running
  100 Hz control loop. Single-network scope cannot represent that.
* **It ignores deadlines entirely**, which is the only thing the workload is
  actually specified in.

So the order below is lexicographic, deadline-first, and standalone kernel
cycles come **last** -- as a tie-break among candidates that are otherwise
indistinguishable, never as the deciding term.

The two worked examples this must get right:

* a split making a kernel 5% slower in total cycles but letting DroNet meet
  30 Hz instead of missing 20% of deadlines is a **WIN**;
* a fusion making a model 10% faster in isolation but creating an 8 ms
  non-preemptible dispatch that breaks a 100 Hz MLP is a **LOSS**.

Why term 1 needs a tolerance
----------------------------
Deadline-miss counts are not stable on this hardware. Measured across seven runs
of an identical B4 schedule, MLP missed 7-9 deadlines out of 38 -- because MLP
completes at ~7 ms against a 10 ms deadline, so it sits on a knife edge and any
perturbation flips several instances. Makespan over the same runs was stable to
1.0%.

An earlier single run of that same schedule reported *1* miss, and reporting it
was a mistake: n=1 on a knife-edge metric is not a measurement. A comparator
that treats a 9-vs-7 difference as decisive would therefore make decisions on
noise, and would do so at the *highest-priority* term where nothing downstream
can correct it.

The treatment here: every term carries an explicit tolerance, and a difference
within tolerance is a TIE that falls through to the next term. For miss counts
the tolerance is expressed in instances and defaults to `max(1, 8% of
instances)` -- 8% because the observed spread was 5.3% and a tolerance has to
exceed the noise it absorbs, while still catching the 10/10-to-2/10 kind of
change that matters. Callers measuring repeated runs
should pass an observed spread instead of relying on the default.

This is deliberately conservative: it prefers "no evidence of improvement" over
"improvement", so a candidate has to earn acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- eligibility

#: Reasons a candidate is not comparable at all. These are gates, not objective
#: terms: a candidate that fails one is rejected without ever being scored,
#: because scoring it would mean comparing numbers that do not mean anything.
GATE_CORRECTNESS = "correctness"
GATE_LEGALITY = "hardware_legality"
GATE_PROFILE = "profile_validity"


@dataclass
class Ineligible:
    gate: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.gate}: {self.detail}"


# ---------------------------------------------------------------- the outcome

@dataclass
class ModelOutcome:
    """One model's measured behaviour over a repeated periodic run."""
    instances: int = 0
    deadline_misses: int = 0
    worst_lateness_ms: float = 0.0
    response_p99_ms: float = 0.0
    achieved_hz: float = 0.0
    required_hz: float = 0.0

    @property
    def miss_rate(self) -> float:
        return self.deadline_misses / self.instances if self.instances else 0.0

    @property
    def frequency_shortfall(self) -> float:
        """How far below the required rate, as a fraction. 0 when met."""
        if self.required_hz <= 0:
            return 0.0
        return max(0.0, (self.required_hz - self.achieved_hz) / self.required_hz)


@dataclass
class CandidateOutcome:
    """Everything the objective needs about one measured candidate.

    Built from `trace_metrics.summarise_trace` output plus, optionally, the
    standalone kernel cycles that the *old* objective used as its only term.
    """
    label: str
    per_model: Dict[str, ModelOutcome] = field(default_factory=dict)
    makespan_ms: float = 0.0
    #: Models whose deadlines are hard. Empty means "treat every model as hard",
    #: which is the safe reading when nobody has said otherwise.
    critical_models: Tuple[str, ...] = ()
    #: The heavy / background model, if the workload has one.
    heavy_model: Optional[str] = None
    heavy_max_latency_ms: float = 0.0
    heavy_throughput_hz: float = 0.0
    utilization_pct: Optional[float] = None
    standalone_cycles: Optional[int] = None
    ineligible: Optional[Ineligible] = None

    # -- helpers over the per-model view -----------------------------------

    def _hard(self) -> List[str]:
        if self.critical_models:
            return [m for m in self.critical_models if m in self.per_model]
        return sorted(self.per_model)

    def total_misses(self) -> int:
        return sum(self.per_model[m].deadline_misses for m in self._hard())

    def total_instances(self) -> int:
        return sum(self.per_model[m].instances for m in self._hard())

    def worst_lateness(self) -> float:
        return max((self.per_model[m].worst_lateness_ms for m in self._hard()),
                   default=0.0)

    def worst_frequency_shortfall(self) -> float:
        return max((self.per_model[m].frequency_shortfall for m in self._hard()),
                   default=0.0)

    def worst_p99(self) -> float:
        return max((self.per_model[m].response_p99_ms for m in self._hard()),
                   default=0.0)


def from_trace_summary(label: str, summary: dict,
                       critical_models: Tuple[str, ...] = (),
                       heavy_model: Optional[str] = None,
                       standalone_cycles: Optional[int] = None
                       ) -> CandidateOutcome:
    """Adapt `trace_metrics.summarise_trace` output into a CandidateOutcome."""
    per_model: Dict[str, ModelOutcome] = {}
    for name, d in (summary.get("per_model") or {}).items():
        per_model[name] = ModelOutcome(
            instances=int(d.get("instances", 0)),
            deadline_misses=int(d.get("instance_deadline_misses", 0)),
            worst_lateness_ms=float(d.get("worst_lateness_ms", 0.0)),
            response_p99_ms=float(d.get("response_p99_ms", 0.0)),
            achieved_hz=float(d.get("achieved_frequency_hz", 0.0)),
            required_hz=float(d.get("required_frequency_hz", 0.0)),
        )
    out = CandidateOutcome(
        label=label,
        per_model=per_model,
        makespan_ms=float(summary.get("makespan_us", 0.0)) / 1000.0,
        critical_models=critical_models,
        heavy_model=heavy_model,
        standalone_cycles=standalone_cycles,
    )
    if heavy_model and heavy_model in per_model:
        h = per_model[heavy_model]
        out.heavy_max_latency_ms = h.response_p99_ms
        out.heavy_throughput_hz = h.achieved_hz
    cl = summary.get("per_cluster_utilization_pct") or {}
    if cl:
        out.utilization_pct = sum(cl.values()) / len(cl)
    return out


# ---------------------------------------------------------------- tolerances

@dataclass
class Tolerances:
    """Per-term "difference too small to be evidence" thresholds.

    Every one of these exists because the corresponding quantity was observed to
    move on its own between identical runs. A difference within tolerance is
    reported as a TIE and the decision falls through to the next term -- it is
    never silently treated as an improvement.
    """
    #: Absolute instances. Default is applied as max(this, miss_rate_frac * N)
    #: so it scales with how many instances were actually released.
    #:
    #: miss_rate_frac is derived from measurement, not chosen for roundness.
    #: Seven runs of an identical B4 schedule gave MLP 7-9 misses of 38 -- a
    #: spread of 2 instances, i.e. 5.3%. A tolerance must EXCEED the observed
    #: noise to absorb it, so 5% would not have (5% of 38 = 1.9 < 2) and the
    #: noisiest quantity would have decided the highest-priority term. 8% gives
    #: 3.0 on 38, which covers the spread with margin while still leaving the
    #: differences that matter decisive: on a 10-instance model 8% is 0.8, so
    #: the floor of 1 applies and a 10/10-vs-2/10 change is still caught.
    miss_instances: int = 1
    miss_rate_frac: float = 0.08
    lateness_ms: float = 0.5
    frequency_frac: float = 0.02
    p99_ms: float = 0.5
    heavy_latency_ms: float = 1.0
    heavy_throughput_frac: float = 0.02
    #: Makespan was stable to 1.0% across repeated identical runs here.
    makespan_frac: float = 0.02
    utilization_pct: float = 2.0
    standalone_cycles_frac: float = 0.01

    def miss_tolerance(self, n_instances: int) -> float:
        return max(self.miss_instances, self.miss_rate_frac * n_instances)


DEFAULT_TOLERANCES = Tolerances()


# ---------------------------------------------------------------- comparison

#: (name, extractor, lower_is_better, tolerance_fn) in priority order.
def _terms(tol: Tolerances):
    return [
        ("hard deadline misses",
         lambda c: float(c.total_misses()), True,
         lambda a, b: tol.miss_tolerance(max(a.total_instances(),
                                             b.total_instances()))),
        ("max deadline lateness",
         lambda c: c.worst_lateness(), True,
         lambda a, b: tol.lateness_ms),
        ("frequency compliance",
         lambda c: c.worst_frequency_shortfall(), True,
         lambda a, b: tol.frequency_frac),
        ("p99 response of critical tasks",
         lambda c: c.worst_p99(), True,
         lambda a, b: tol.p99_ms),
        ("heavy-model max latency",
         lambda c: c.heavy_max_latency_ms, True,
         lambda a, b: tol.heavy_latency_ms),
        ("heavy-model throughput",
         lambda c: c.heavy_throughput_hz, False,
         lambda a, b: tol.heavy_throughput_frac
         * max(a.heavy_throughput_hz, b.heavy_throughput_hz, 1e-9)),
        ("makespan",
         lambda c: c.makespan_ms, True,
         lambda a, b: tol.makespan_frac
         * max(a.makespan_ms, b.makespan_ms, 1e-9)),
        ("utilization",
         lambda c: c.utilization_pct if c.utilization_pct is not None else 0.0,
         False, lambda a, b: tol.utilization_pct),
        # LAST. A candidate reaches this term only when it is indistinguishable
        # on every real-time property, which is exactly when raw kernel cost is
        # the right tie-break -- and never before.
        ("standalone kernel cycles",
         lambda c: float(c.standalone_cycles or 0), True,
         lambda a, b: tol.standalone_cycles_frac
         * max(a.standalone_cycles or 0, b.standalone_cycles or 0, 1)),
    ]


def compare(a: CandidateOutcome, b: CandidateOutcome,
            tol: Tolerances = DEFAULT_TOLERANCES) -> Tuple[int, str]:
    """Lexicographic order. Returns (-1 if a better, 1 if b better, 0 tie), why.

    Ineligibility is checked first: an ineligible candidate always loses, and
    two ineligible ones tie, because neither carries a meaningful number.
    """
    if a.ineligible and b.ineligible:
        return 0, f"both ineligible ({a.ineligible.gate}, {b.ineligible.gate})"
    if a.ineligible:
        return 1, f"{a.label} ineligible -- {a.ineligible}"
    if b.ineligible:
        return -1, f"{b.label} ineligible -- {b.ineligible}"

    for name, get, lower_better, tol_fn in _terms(tol):
        va, vb = get(a), get(b)
        t = tol_fn(a, b)
        if abs(va - vb) <= t:
            continue  # within noise -- fall through to the next term
        a_better = (va < vb) if lower_better else (va > vb)
        winner, loser = (a, b) if a_better else (b, a)
        wv, lv = (va, vb) if a_better else (vb, va)
        return (-1 if a_better else 1), (
            f"{name}: {winner.label}={wv:g} beats {loser.label}={lv:g} "
            f"(tolerance {t:g})")
    return 0, "indistinguishable on every term"


def accept(candidate: CandidateOutcome, baseline: CandidateOutcome,
           tol: Tolerances = DEFAULT_TOLERANCES) -> Tuple[bool, str]:
    """Should `candidate` replace `baseline`?

    Only a strict win is accepted. A tie is a rejection: a compiler change that
    cannot be shown to help is not worth the rebuild, the regenerated kernels,
    or the loss of a known-good baseline.
    """
    if candidate.ineligible:
        return False, f"rejected -- {candidate.ineligible}"
    order, why = compare(candidate, baseline, tol)
    if order < 0:
        return True, f"accepted -- {why}"
    if order == 0:
        return False, f"rejected -- {why}; a tie is not an improvement"
    return False, f"rejected -- {why}"


def rank(candidates: List[CandidateOutcome],
         tol: Tolerances = DEFAULT_TOLERANCES) -> List[CandidateOutcome]:
    """Best first. Stable for genuinely tied candidates."""
    import functools
    return sorted(candidates,
                  key=functools.cmp_to_key(
                      lambda x, y: compare(x, y, tol)[0]))
