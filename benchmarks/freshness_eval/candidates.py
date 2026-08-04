"""Precomputed candidate schedules, and the gate that decides whether a
candidate set is admissible.

TERMINOLOGY: these are **precomputed and empirically validated candidates**.
Not "certified" — no validation or proof procedure establishes a guarantee here.
`expected_*` fields are MEASURED outcomes from a named sweep, not promises, and
every one carries the run it was measured in.

WHY THERE IS A GATE
-------------------
Gate A produced a negative finding that changes how this phase has to work.
`static_conservative` — reserving the fast accelerator for the perception
producer, the protection mechanism the plan proposed — measured WORSE than doing
nothing at every contention level (output-valid 0.146 vs 0.220 at B=3). A
"protection" candidate is therefore not protective by construction, by
intention, or by name.

So `build_candidate_set` REFUSES to assemble a ladder whose protective rungs are
not measured to beat the nominal rung inside their own intended operating
region. Building a selector on an unvalidated ladder would produce a confident
adaptive-vs-static comparison in which the adaptive policy switches to
candidates that make things worse — and the result would look like a finding
about adaptation rather than what it is, a finding about a bad candidate.

A candidate set that fails the gate is a real, reportable result: it says the
mechanism search has not yet found a protective mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Imported lazily by callers that need them; kept here so the vocabulary is
# checked in one place.
from benchmarks.freshness_eval.run import ALL_POLICIES, MUTATION_KEYS


@dataclass(frozen=True)
class Candidate:
    """One precomputed schedule alternative.

    `intended_bursts` is the operating region the candidate is FOR, and is the
    region the validation gate scores it in. Claiming a wide region and
    performing well only in part of it is exactly what the gate is there to
    catch.
    """

    candidate_id: str
    protection_level: int
    intent: str
    intended_bursts: Tuple[int, ...]
    solver: str
    scheduler: str
    mutations: Dict[str, object]

    # Placement and rate description, recorded for the report rather than used
    # by the solver (the solver reads the materialised config).
    backend_assignment: str = "solver-chosen (no pin)"
    admission_policy: str = "admit all offered soft work"
    rates: str = "control 10 ms, perception 50 ms (unchanged)"

    # Measured, not promised. Filled by `attach_measurements`.
    expected_output_valid: Optional[Dict[int, float]] = None
    expected_soft_utility: Optional[Dict[int, float]] = None
    measured_in: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.mutations) - set(MUTATION_KEYS)
        if unknown:
            raise ValueError(
                f"{self.candidate_id}: unknown mutation(s) {sorted(unknown)}"
            )
        if not self.intended_bursts:
            raise ValueError(
                f"{self.candidate_id}: must declare an intended operating region; "
                f"a candidate with no region cannot be validated"
            )

    def describe(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "protection_level": self.protection_level,
            "intent": self.intent,
            "intended_bursts": list(self.intended_bursts),
            "solver": self.solver,
            "scheduler": self.scheduler,
            "mutations": self.mutations,
            "backend_assignment": self.backend_assignment,
            "admission_policy": self.admission_policy,
            "rates": self.rates,
            "expected_output_valid_by_burst": self.expected_output_valid,
            "expected_soft_utility_by_burst": self.expected_soft_utility,
            "measured_in": self.measured_in,
            "status": (
                "precomputed and empirically validated"
                if self.expected_output_valid else
                "precomputed, NOT yet validated"
            ),
        }


@dataclass
class GateResult:
    admissible: bool
    findings: List[str] = field(default_factory=list)
    per_candidate: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"candidate-set gate: {'PASS' if self.admissible else 'FAIL'}",
            "",
        ]
        for cid, d in self.per_candidate.items():
            lines.append(
                f"  {cid:<24} region={d['region']} "
                f"output_valid={d['candidate_output_valid']} "
                f"nominal={d['nominal_output_valid']} "
                f"margin={d['margin']} -> {d['verdict']}"
            )
        if self.findings:
            lines += ["", "findings:"] + [f"  - {f}" for f in self.findings]
        return "\n".join(lines)


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _rates_by_burst(rows, policy: str, phi: float, metric: str) -> Dict[int, float]:
    """Mean `metric` per contention level for one policy at one freshness window.

    Averages over seeds. Rows are the aggregate.csv dicts (all strings).
    """
    acc: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("policy") != policy:
            continue
        if abs(float(r["freshness_window"]) - phi) > 1e-6:
            continue
        b = int(float(r["contention_level"]))
        v = r.get(metric)
        if v in (None, ""):
            continue
        acc.setdefault(b, []).append(float(v))
    return {b: _mean(v) for b, v in sorted(acc.items()) if _mean(v) is not None}


def validate_candidate_set(
    candidates: Sequence[Candidate],
    aggregate_rows: Sequence[Dict[str, str]],
    *,
    phi: float,
    nominal_id: str,
    min_margin: float = 0.0,
    run_label: str = "",
) -> GateResult:
    """Score every protective candidate against the nominal one in its region.

    A protective candidate (protection_level > 0) is admissible only if its mean
    `output_valid_rate` over its intended bursts EXCEEDS the nominal candidate's
    by at least `min_margin`. `min_margin=0.0` means "must not be worse", which
    is the weakest defensible bar and the one Gate A's negative finding shows is
    not automatically met.
    """
    res = GateResult(admissible=True)
    nominal = _rates_by_burst(aggregate_rows, nominal_id, phi, "output_valid_rate")
    if not nominal:
        res.admissible = False
        res.findings.append(
            f"no rows for the nominal candidate {nominal_id!r} at phi={phi}; "
            f"nothing to validate against"
        )
        return res

    for c in candidates:
        cand = _rates_by_burst(aggregate_rows, c.candidate_id, phi, "output_valid_rate")
        region = [b for b in c.intended_bursts]
        missing = [b for b in region if b not in cand or b not in nominal]
        cv = _mean([cand[b] for b in region if b in cand])
        nv = _mean([nominal[b] for b in region if b in nominal])
        margin = None if (cv is None or nv is None) else cv - nv

        if c.protection_level == 0:
            verdict = "baseline"
        elif missing:
            verdict = f"UNSCORABLE (missing bursts {missing})"
            res.admissible = False
            res.findings.append(
                f"{c.candidate_id}: no measurement at burst(s) {missing} in its "
                f"declared region -- cannot be validated, so it is not admissible"
            )
        elif margin is None:
            verdict = "UNSCORABLE (no data)"
            res.admissible = False
            res.findings.append(f"{c.candidate_id}: no output_valid_rate data")
        elif margin > min_margin:
            verdict = "PASS"
        elif margin == 0.0 and min_margin == 0.0:
            # "Not worse" clears the weakest defensible bar, but a candidate
            # that buys nothing is worth naming rather than quietly passing:
            # a selector switching to it would spend utility for no gain.
            verdict = "PASS (no improvement, not worse)"
            res.findings.append(
                f"{c.candidate_id}: exactly matches nominal in its region "
                f"(margin 0.000) -- admissible but buys nothing"
            )
        else:
            verdict = "FAIL"
            res.admissible = False
            res.findings.append(
                f"{c.candidate_id}: output_valid {cv:.3f} vs nominal {nv:.3f} "
                f"(margin {margin:+.3f}) over bursts {region} -- a protection "
                f"candidate that does not protect. Do NOT build a selector on it."
            )

        res.per_candidate[c.candidate_id] = {
            "region": region,
            "candidate_output_valid": None if cv is None else round(cv, 4),
            "nominal_output_valid": None if nv is None else round(nv, 4),
            "margin": None if margin is None else round(margin, 4),
            "verdict": verdict,
            "by_burst": {b: round(cand[b], 4) for b in sorted(cand)},
            "run_label": run_label,
        }
    return res


def attach_measurements(
    candidate: Candidate,
    aggregate_rows: Sequence[Dict[str, str]],
    *,
    phi: float,
    run_label: str,
) -> Candidate:
    """Return a copy carrying its measured outcomes and their provenance."""
    from dataclasses import replace
    return replace(
        candidate,
        expected_output_valid=_rates_by_burst(
            aggregate_rows, candidate.candidate_id, phi, "output_valid_rate"),
        expected_soft_utility=_rates_by_burst(
            aggregate_rows, candidate.candidate_id, phi, "soft_instances_completed"),
        measured_in=run_label,
    )


def build_candidate_set(
    candidates: Sequence[Candidate],
    aggregate_rows: Sequence[Dict[str, str]],
    *,
    phi: float,
    run_label: str,
    min_margin: float = 0.0,
    strict: bool = True,
) -> Tuple[List[Candidate], GateResult]:
    """Validate, attach measurements, and return the ladder.

    With `strict=True` (the default) an inadmissible set raises rather than
    returning, so a selector can never be built on candidates that were not
    measured to help. Pass `strict=False` only to inspect a failing set.
    """
    levels = sorted(c.protection_level for c in candidates)
    if levels != list(range(len(candidates))):
        raise ValueError(
            f"protection levels must be 0..N-1 with no gaps/duplicates, got "
            f"{levels}. (This also guarantees exactly one nominal, level 0.)"
        )
    nominal = next(c for c in candidates if c.protection_level == 0)

    gate = validate_candidate_set(
        candidates, aggregate_rows, phi=phi,
        nominal_id=nominal.candidate_id, min_margin=min_margin,
        run_label=run_label,
    )
    if strict and not gate.admissible:
        raise ValueError(
            "candidate set is NOT admissible; refusing to build a selector on "
            "candidates that were not measured to help.\n\n" + gate.report()
        )
    out = [
        attach_measurements(c, aggregate_rows, phi=phi, run_label=run_label)
        for c in sorted(candidates, key=lambda c: c.protection_level)
    ]
    return out, gate
