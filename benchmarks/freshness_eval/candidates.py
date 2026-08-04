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
            # The baseline's "margin" is against itself and is always 0, so
            # flagging it as unquotable is noise.
            flag = ("" if d["verdict"] == "baseline"
                    or d.get("margin_epoch_comparable", True)
                    else " [margin NOT quotable]")
            lines.append(
                f"  {cid:<24} region={d['region']} "
                f"output_valid={d['candidate_output_valid']} "
                f"nominal={d['nominal_output_valid']} "
                f"margin={d['margin']}{flag} -> {d['verdict']}"
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


def _overrun_bursts(rows, policy: str, phi: float) -> List[int]:
    """Bursts where this policy's schedule does NOT fit the epoch.

    A rate measured on an overrunning schedule is computed over a longer trace
    with more consumer invocations, so it does not share a denominator with a
    rate from a fitting schedule. Margins spanning such a cell are directionally
    meaningful but not quotable, and the gate says so rather than printing a
    clean-looking number.
    """
    out = []
    for r in rows:
        if r.get("policy") != policy:
            continue
        if abs(float(r["freshness_window"]) - phi) > 1e-6:
            continue
        if r.get("fits_in_epoch") == "False":
            out.append(int(float(r["contention_level"])))
    return sorted(set(out))


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

        # A margin is only quotable if BOTH sides fit the epoch everywhere in the
        # region. Where they do not, the verdict still stands (the candidate is
        # better, and the nominal's overrun is itself a failure) but the number
        # must not be reported as a rate difference.
        nom_over = [b for b in _overrun_bursts(aggregate_rows, nominal_id, phi)
                    if b in region]
        cand_over = [b for b in _overrun_bursts(aggregate_rows, c.candidate_id, phi)
                     if b in region]
        comparable = not (nom_over or cand_over)
        if c.protection_level > 0 and not comparable:
            res.findings.append(
                f"{c.candidate_id}: margin {margin:+.3f} is NOT epoch-comparable "
                f"-- overrunning cells in the region: "
                f"nominal at B={nom_over or 'none'}, candidate at "
                f"B={cand_over or 'none'}. Those rates are over longer traces "
                f"with different invocation counts. Verdict stands (an overrun "
                f"is itself a failure); do not quote the margin."
            )

        res.per_candidate[c.candidate_id] = {
            "region": region,
            "candidate_output_valid": None if cv is None else round(cv, 4),
            "nominal_output_valid": None if nv is None else round(nv, 4),
            "margin": None if margin is None else round(margin, 4),
            "margin_epoch_comparable": comparable,
            "overrun_bursts_nominal": nom_over,
            "overrun_bursts_candidate": cand_over,
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


# --- the ladder for this workload -------------------------------------------
#
# Assembled only from mechanisms the probe sweep measured to work, with the
# region boundaries taken from where they were measured to work rather than from
# where they were hoped to.
#
# The shape is forced by three measurements, and the last one is inconvenient:
#
#  1. Soft deferral in [10, 18] ms weakly DOMINATES no deferral across all 20
#     measured (phi, B) cells at B<=3 -- equal at B=0, strictly better in the
#     other 16 -- while shedding no soft work and, at B=3, fitting the 300 ms
#     epoch where the baseline's schedule does not. There is therefore NO
#     operating region in which the unprotected schedule is the right choice.
#     C0 is kept as the reference the gate scores against, not as a rung a
#     selector would ever choose.
#
#  2. At B=4 that dominance ends: deferral alone overruns the epoch (806 ms
#     against a 300 ms budget) and so is not a deployable choice there at all,
#     whatever its rate. Admission capping ON TOP OF deferral is what survives:
#     at phi=A0+20, admit2 holds 0.867 and admit1 holds 0.900 at every burst.
#
#  3. Adaptive switching is therefore worth almost nothing on this workload, and
#     the bound is exact rather than a guess (see `adaptive.py`, which computes
#     it):
#
#       * over bursts 0..3, a single static rung is optimal at EVERY validity
#         target -- switching gains exactly zero;
#       * over bursts 0..4, switching gains exactly ONE soft instance out of the
#         10 offered (3/3 instead of 2/3 at B=3), and only for validity targets
#         <= 0.833. Above that it gains zero again.
#
#     The cause is that the protective mechanism is nearly free: deferral costs
#     no utility, so a permanently conservative rung is barely worse than the
#     best per-burst choice, and there is almost nothing for adaptation to
#     reclaim. That is a property of this workload, not a bug in the selector,
#     and it has to be reported as the headline for the adaptive phase rather
#     than buried under a selector that technically works.
LADDER_PHI_DELTA = 20.0     # phi = A0 + 20 ms, the primary reported operating point

C0_NOMINAL = Candidate(
    candidate_id="static_nominal",
    protection_level=0,
    intent=(
        "Reference, not a deployable rung: no protection. Measured to be weakly "
        "dominated by C1 at every contention level, so a selector has no reason "
        "to choose it."
    ),
    intended_bursts=(0, 1, 2, 3),
    solver="greedy", scheduler="mosek", mutations={},
)

C1_DEFER = Candidate(
    candidate_id="cand_c1_defer12",
    protection_level=1,
    intent=(
        "Perception protection by deferring the first soft release 12 ms, the "
        "centre of the measured [10, 18] ms plateau. Costs no soft utility."
    ),
    intended_bursts=(0, 1, 2, 3),
    admission_policy="admit all offered soft work",
    solver="greedy", scheduler="mosek",
    mutations={"soft_phase_ms": 12.0},
)

# NOTE on the id/level mismatch below: the `candidate_id`s are the policy keys the
# sweep was actually run under, and the measured rows in
# results/freshness_cand/*/aggregate.csv are keyed by them. Renaming them to match
# their protection level would orphan every measurement, so the ids stay as
# measured and the level is what orders the ladder.
C2_DEFER_ADMIT2 = Candidate(
    candidate_id="cand_c2_defer12_admit2",
    protection_level=2,
    intent=(
        "Degraded safety: deferral plus a 2-instance admission cap. Measured to "
        "be the lowest rung that is admissible across the WHOLE range 0..4 -- "
        "deferral alone overruns the epoch at B=4."
    ),
    intended_bursts=(3, 4),
    admission_policy="admit at most 2 soft instances",
    solver="greedy", scheduler="mosek",
    mutations={"soft_phase_ms": 12.0, "admit_cap": 2},
)

C3_DEFER_ADMIT1 = Candidate(
    candidate_id="cand_c2_defer12_admit1",
    protection_level=3,
    intent=(
        "Maximum protection: deferral plus a 1-instance admission cap. Holds a "
        "flat 0.900 output-valid rate at every burst 0..4 at phi=A0+20, the only "
        "rung that does, and pays for it by shedding the most soft work."
    ),
    intended_bursts=(2, 3, 4),
    admission_policy="admit at most 1 soft instance",
    solver="greedy", scheduler="mosek",
    mutations={"soft_phase_ms": 12.0, "admit_cap": 1},
)

# Ordered by protection level. Measured to be strictly monotone in BOTH
# directions at every burst: validity non-decreasing up the ladder, soft utility
# non-increasing. `test_candidate_ladder.py` pins that, because a ladder that is
# not monotone has no well-defined "next safer rung" for a selector to escalate
# to.
LADDER = (C0_NOMINAL, C1_DEFER, C2_DEFER_ADMIT2, C3_DEFER_ADMIT1)
