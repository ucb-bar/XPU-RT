"""`compile_advice.json` is a machine contract, so its shape has to be pinned.

WHAT THIS FILE GUARDS. The advice document is the only channel from "the
scheduler noticed something" to "the compiler does something about it". Nothing
on the consuming side validates it: `scripts/apply_compile_advice.py` reaches
straight into `item["evidence"]["proposed_impl"]`, and
`scripts/advice_to_fusion_hint.py` into `x["recommendation"]` and
`x["evidence"]`. A producer that renames a field, nests it one level deeper,
emits a recommendation word the consumer has never heard of, or drops the
numbers behind a claim, therefore fails at the consumer -- one repo away from
the change, in a run that has already burned a board slot.

Three classes of defect are specifically plausible here, and each has a test:

1. **`evidence.extra` not being flattened.** `Evidence` carries a free-form
   `extra` dict, and `Advice.as_dict()` merges it up into `evidence` on the way
   out. Every optional field a consumer reads -- `proposed_impl`, `n_cores`,
   `overrun_factor` -- lives in `extra`. A refactor that emitted `asdict()`
   directly would produce `evidence["extra"]["proposed_impl"]`, which is a
   `KeyError` at the consumer and looks like a valid document to everything
   else.

2. **A recommendation nobody can act on.** `RECOMMENDATIONS` is the vocabulary;
   `pin_core_class` and `coarsen` are in it and no producer emits them. The
   inverse -- a producer emitting a word not in the tuple -- is the one that
   breaks a consumer, and nothing checks it.

3. **A claim with no evidence.** The module docstring's first design rule is
   "evidence or nothing", because a recommendation nobody can audit still gets
   acted on. That is a property of every emitted item, so it belongs in a test
   that runs over every producer rather than in each producer's own tests.

The schema below is written out declaratively rather than derived from the
dataclasses on purpose: a test that reads the field list off the class it is
testing cannot notice a field being renamed, which is the change that breaks
consumers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

import compile_advice  # noqa: E402
from compile_advice import (  # noqa: E402
    RECOMMENDATIONS, SCHEMA_VERSION, Advice, Evidence, blocking_advice,
    implementation_advice, overhead_advice, shard_advice, write_advice,
)

# ---------------------------------------------------------------- the contract

#: Keys every advice item must carry, with the type a consumer may assume.
#: `dispatch_id` is deliberately `(int, str)`: `overhead_advice` addresses a
#: whole model with `"*"`, and a consumer that assumed int would crash on it.
REQUIRED_FIELDS = {
    "model": str,
    "dispatch_id": (int, str),
    "recommendation": str,
    "priority": int,
    "confidence": str,
    "evidence": dict,
    "constraints": dict,
    "rationale": str,
}

#: Keys every `evidence` block must carry, whatever else it adds.
REQUIRED_EVIDENCE_FIELDS = {
    "service_time_us": (int, float),
    "blocking_time_us": (int, float),
    "periodic_free_slot_us": (int, float),
    "deadline_misses_attributed": int,
    "on_critical_path": bool,
}

#: Priority is a small ordinal the consumer sorts on and truncates ("apply the
#: top N"). An out-of-range value silently reorders the whole document.
PRIORITY_RANGE = (1, 5)

CONFIDENCE_VALUES = {"high", "medium", "low"}

#: Evidence a consumer needs in order to *act* on each recommendation, as
#: opposed to merely parse it. Each entry is the set of keys the code that
#: applies that recommendation dereferences.
ACTIONABLE_EVIDENCE = {
    # scripts/apply_compile_advice.py: ev["proposed_impl"]
    "choose_implementation": {"proposed_impl", "baseline_impl", "gain_fraction"},
    # a split needs a target piece size; without the slot it is unmotivated
    "split": {"overrun_factor"},
    # a shard is a claim about measured scaling, so the measurement must be here
    "shard": {"n_cores", "measured_speedup", "parallel_efficiency"},
    "fuse_with_successor": {"n_dispatches", "estimated_overhead_fraction"},
    # An unfuse is only defensible when the FUSED kernel is the thing running
    # badly, so the evidence is which implementation each side actually ran --
    # a measured fact from the profile's `implementation` column, not a model.
    # Emitting it from a granularity verdict is how you get the 0.81x result.
    "unfuse": {"fused_impl", "constituent_impls"},
}

#: Constraints the consumer needs to know the change is legal / bounded.
REQUIRED_CONSTRAINTS = {
    "split": {"max_target_piece_us"},
    "shard": {"n_cores"},
    "choose_implementation": {"legal_resources"},
    "fuse_with_successor": {"requires_linear_chain"},
    # Undoing a fusion is only legal if a kernel exists for every constituent;
    # otherwise the restored ops fall back to the scalar reference, which is
    # exactly the failure this verb is meant to CURE.
    "unfuse": {"requires_constituent_kernels"},
}


def _validate(item: dict, testcase: unittest.TestCase, where: str = "") -> None:
    """Assert one advice dict conforms. Shared by every test below."""
    for key, typ in REQUIRED_FIELDS.items():
        testcase.assertIn(key, item, f"{where}: missing {key!r}")
        testcase.assertIsInstance(item[key], typ, f"{where}: {key!r}")

    rec = item["recommendation"]
    testcase.assertIn(rec, RECOMMENDATIONS,
                      f"{where}: {rec!r} is outside the vocabulary a consumer "
                      f"can dispatch on")
    testcase.assertIn(item["confidence"], CONFIDENCE_VALUES, where)
    lo, hi = PRIORITY_RANGE
    testcase.assertTrue(lo <= item["priority"] <= hi,
                        f"{where}: priority {item['priority']} out of range")
    testcase.assertTrue(item["rationale"].strip(),
                        f"{where}: empty rationale on a {rec}")

    ev = item["evidence"]
    testcase.assertNotIn(
        "extra", ev,
        f"{where}: `extra` was not flattened into `evidence`; every optional "
        f"field a consumer reads lives in it")
    for key, typ in REQUIRED_EVIDENCE_FIELDS.items():
        testcase.assertIn(key, ev, f"{where}: evidence missing {key!r}")
        testcase.assertIsInstance(ev[key], typ, f"{where}: evidence {key!r}")

    for key in ACTIONABLE_EVIDENCE.get(rec, ()):
        testcase.assertIn(key, ev,
                          f"{where}: a {rec} without {key!r} cannot be applied")
    for key in REQUIRED_CONSTRAINTS.get(rec, ()):
        testcase.assertIn(key, item["constraints"],
                          f"{where}: a {rec} without constraint {key!r}")

    # "Evidence or nothing": an actionable item must carry at least one measured
    # quantity, not just prose. `unchanged` is exempt from the numeric floor but
    # still has to say what was measured and refused.
    if rec != "unchanged":
        measured = [ev[k] for k in ("service_time_us", "blocking_time_us",
                                    "periodic_free_slot_us")]
        testcase.assertTrue(any(float(v) > 0 for v in measured),
                            f"{where}: {rec} carries no measured time")
    testcase.assertGreater(len(ev), len(REQUIRED_EVIDENCE_FIELDS),
                           f"{where}: {rec} adds nothing to the evidence "
                           f"beyond the empty defaults")


# ------------------------------------------------------------------- fixtures

def _jsonl_rec(did, ms, *, n_cores=1, cv=0.3, module=None):
    """One `profile.jsonl` record, the shape `_load_jsonl` returns."""
    return {"dispatch_id": did, "median_ms": ms, "n_cores": n_cores,
            "cv_pct": cv,
            "module_name": module or f"m$dispatch_{did}_rvv_x60_conv2d_s8_n{did}"}


def _profile(costs, **kw):
    return {did: _jsonl_rec(did, ms, **kw) for did, ms in costs.items()}


class EveryProducerConforms(unittest.TestCase):
    """One test per producer, all going through the same validator.

    Each producer is exercised on the case that makes it emit, so the schema is
    checked on real output rather than on a hand-written example that can drift
    from what the code does.
    """

    def test_implementation_advice_positive_and_negative(self):
        """Both branches: a switch worth making, and one measured and refused.

        The negative branch is the easy one to get wrong, because "unchanged"
        looks like it needs no payload -- but it exists precisely so a later
        round can see the change was already measured and rejected.
        """
        profs = {
            "rvv_x60": _profile({0: 2.0, 1: 0.5}),
            "scalar": _profile({0: 30.0, 1: 0.2}),
        }
        adv = implementation_advice("dronet", profs, "rvv_x60")
        self.assertEqual(len(adv), 2)
        recs = {a.recommendation for a in adv}
        self.assertEqual(recs, {"unchanged", "choose_implementation"},
                         "one dispatch improves, one does not")
        for a in adv:
            _validate(a.as_dict(), self, f"implementation_advice/{a.dispatch_id}")

    def test_overhead_advice(self):
        prof = _profile({i: 0.060 + 0.001 * i for i in range(6)})
        adv = overhead_advice("mlp_control", prof, chain=True)
        self.assertEqual(len(adv), 1)
        _validate(adv[0].as_dict(), self, "overhead_advice")

    def test_blocking_advice(self):
        prof = _profile({0: 22.9, 1: 0.06})
        adv = blocking_advice("dronet", prof, free_slot_ms=6.7, misses=3)
        self.assertEqual(len(adv), 1)
        _validate(adv[0].as_dict(), self, "blocking_advice")

    def test_shard_advice_positive_and_negative(self):
        """A 0.05 ms budget forces both dispatches past shard's first gate.

        Only then does the decision rest on the scaling measurement, which is
        the point: the conv goes 22.9 -> 6.1 ms on four cores and the tiny op
        gets *slower*, so one is recommended and one is refused-with-numbers.
        """
        by_cores = {
            1: _profile({0: 22.9, 1: 0.066}, n_cores=1),
            4: _profile({0: 6.1, 1: 0.094}, n_cores=4),
        }
        adv = shard_advice("dronet", by_cores, free_slot_ms=0.05)
        self.assertEqual({a.recommendation for a in adv},
                         {"shard", "unchanged"},
                         "the conv shards, the tiny op measurably does not")
        for a in adv:
            _validate(a.as_dict(), self, f"shard_advice/{a.dispatch_id}")


class ExtraMustBeFlattened(unittest.TestCase):
    """Defect class 1. The consumer reads `evidence["proposed_impl"]`."""

    def test_as_dict_lifts_extra_into_evidence(self):
        a = Advice(model="m", dispatch_id=0, recommendation="choose_implementation",
                   priority=1, confidence="high",
                   evidence=Evidence(service_time_us=1.0,
                                     extra={"proposed_impl": "IME"}),
                   rationale="because")
        d = a.as_dict()
        self.assertEqual(d["evidence"]["proposed_impl"], "IME")
        self.assertNotIn("extra", d["evidence"])

    def test_the_consumers_own_access_pattern_works(self):
        """Exactly what `apply_compile_advice.py` does, on real producer output.

        It builds `{(model, int(dispatch_id)): ev["proposed_impl"]}`. That line
        is the whole reason `extra` has to be flattened AND the reason
        `dispatch_id` has to be int-coercible for this recommendation.
        """
        profs = {"rvv_x60": _profile({7: 1.0}), "IME": _profile({7: 0.5})}
        adv = implementation_advice("dronet", profs, "rvv_x60")
        chosen = {}
        for item in (a.as_dict() for a in adv):
            if item["recommendation"] != "choose_implementation":
                continue
            chosen[(item["model"], int(item["dispatch_id"]))] = \
                item["evidence"]["proposed_impl"]
        self.assertEqual(chosen, {("dronet", 7): "IME"})

    def test_a_wildcard_dispatch_id_never_reaches_that_int_coercion(self):
        """`overhead_advice` addresses a model with `"*"`, not a dispatch.

        `apply_compile_advice.py` calls `int(item["dispatch_id"])`, guarded only
        by a `recommendation != "choose_implementation"` filter. So the moment
        any producer emits `"*"` with a `choose_implementation` (or any other
        recommendation that consumer handles), it raises `ValueError` on a
        document that is otherwise perfectly valid. Pin the invariant that keeps
        that from happening.
        """
        prof = _profile({i: 0.060 for i in range(4)})
        adv = overhead_advice("mlp_control", prof, chain=True)
        for a in adv:
            if a.dispatch_id == "*":
                self.assertTrue(a.recommendation.startswith("fuse_"),
                                f"{a.recommendation} with dispatch_id='*' "
                                f"would crash apply_compile_advice.py")
            else:
                int(a.dispatch_id)


class RecommendationVocabulary(unittest.TestCase):
    """Defect class 2."""

    def test_no_producer_invents_a_word(self):
        """Every producer, on inputs chosen so each emits something."""
        emitted = set()
        emitted |= {a.recommendation for a in implementation_advice(
            "m", {"a": _profile({0: 1.0}), "b": _profile({0: 0.1})}, "a")}
        emitted |= {a.recommendation for a in overhead_advice(
            "m", _profile({i: 0.06 for i in range(4)}), chain=True)}
        emitted |= {a.recommendation for a in blocking_advice(
            "m", _profile({0: 10.0}), free_slot_ms=1.0, misses=0)}
        emitted |= {a.recommendation for a in shard_advice(
            "m", {1: _profile({0: 20.0}, n_cores=1),
                  4: _profile({0: 5.0}, n_cores=4)}, free_slot_ms=10.0)}
        self.assertTrue(emitted)
        self.assertTrue(emitted <= set(RECOMMENDATIONS),
                        f"outside the vocabulary: {emitted - set(RECOMMENDATIONS)}")

    def test_the_vocabulary_has_no_duplicates(self):
        """A duplicate would make `in RECOMMENDATIONS` checks pass for two
        different intended meanings of the same word."""
        self.assertEqual(len(RECOMMENDATIONS), len(set(RECOMMENDATIONS)))


class MislabelledFieldsStayFixed(unittest.TestCase):
    """Two fields in this schema were previously carrying the wrong quantity."""

    def test_deadline_misses_attributed_is_the_measured_miss_count(self):
        """It used to be passed `len(profile)` -- a DISPATCH COUNT.

        Every split recommendation then carried the same number in a field named
        `deadline_misses_attributed`, and anything reading it downstream was
        reading a mislabelled constant. The honest value with no measured trace
        is 0, so a caller passing 0 must get 0 and not a fallback.
        """
        prof = _profile({0: 10.0, 1: 11.0, 2: 12.0})
        adv = blocking_advice("m", prof, free_slot_ms=1.0, misses=0)
        self.assertEqual(len(adv), 3)
        for a in adv:
            self.assertEqual(a.evidence.deadline_misses_attributed, 0)
        # And a real count is passed through unchanged, per item.
        adv = blocking_advice("m", prof, free_slot_ms=1.0, misses=7)
        self.assertEqual({a.evidence.deadline_misses_attributed for a in adv},
                         {7})

    def test_periodic_free_slot_is_a_slot_not_a_period(self):
        """`split` is only meaningful against the room actually available.

        The emitter previously passed `min(periods.values())` while calling it a
        slot, which overstates the budget by exactly the work the model already
        does. The evidence field has to report whatever budget the decision was
        made against, so the two can never disagree.
        """
        adv = blocking_advice("m", _profile({0: 10.0}), free_slot_ms=2.5,
                              misses=0)
        ev = adv[0].as_dict()["evidence"]
        self.assertAlmostEqual(ev["periodic_free_slot_us"], 2500.0, places=3)
        self.assertAlmostEqual(ev["overrun_factor"], 4.0, places=3)
        self.assertAlmostEqual(
            adv[0].constraints["max_target_piece_us"], 2500.0, places=3,
            msg="the target piece size must be the same budget the decision "
                "used, or the compiler cuts to a size that still does not fit")

    def test_no_advice_at_all_when_the_budget_is_unknown(self):
        """A zero/absent slot must produce silence, not advice against 0 ms.

        Comparing every dispatch against a 0 ms budget would flag the entire
        model, which reads as a catastrophic finding and is actually missing
        input.
        """
        self.assertEqual(
            blocking_advice("m", _profile({0: 10.0}), 0.0, misses=0), [])
        self.assertEqual(
            blocking_advice("m", _profile({0: 10.0}), -1.0, misses=0), [])


class TheDocumentOnDisk(unittest.TestCase):
    """`write_advice` output has to survive a JSON round trip untouched."""

    def _doc(self, advice, notes=None):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "compile_advice.json")
            write_advice(path, advice, schedule_id="sched.json", notes=notes)
            with open(path) as f:
                return json.load(f)

    def _all_advice(self):
        profs = {"rvv_x60": _profile({0: 22.9, 1: 0.066}),
                 "scalar": _profile({0: 300.0, 1: 0.06})}
        adv = implementation_advice("dronet", profs, "rvv_x60")
        adv += overhead_advice("mlp_control",
                               _profile({i: 0.06 for i in range(5)}), chain=True)
        adv += blocking_advice("dronet", _profile({0: 22.9}), 6.7, misses=2)
        adv += shard_advice("dronet",
                            {1: _profile({0: 22.9}, n_cores=1),
                             4: _profile({0: 6.1}, n_cores=4)}, 6.7)
        return adv

    def test_the_envelope_carries_a_version_and_a_schedule_id(self):
        """Advice about one schedule applied to another is unfalsifiable.

        The version pins the field layout; `schedule_id` pins WHICH schedule the
        numbers describe. Without the latter, an advice file from a previous
        round is indistinguishable from a current one.
        """
        doc = self._doc(self._all_advice())
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        self.assertIsInstance(doc["schema_version"], int)
        self.assertEqual(doc["schedule_id"], "sched.json")
        self.assertIsInstance(doc["advice"], list)

    def test_every_item_in_a_written_document_validates(self):
        doc = self._doc(self._all_advice())
        self.assertGreaterEqual(len(doc["advice"]), 5)
        for i, item in enumerate(doc["advice"]):
            _validate(item, self, f"advice[{i}]")

    def test_json_round_trip_changes_nothing_a_consumer_reads(self):
        """Nothing in the evidence may be a type JSON cannot express.

        `numpy` floats, sets and tuples all serialise (or fail) in ways that
        change what the consumer sees; the producers here are fed plain Python,
        but evidence is a free-form dict and a future producer sourcing values
        straight from a solver array is the obvious way in.
        """
        adv = self._all_advice()
        doc = self._doc(adv)
        in_memory = [a.as_dict() for a in adv]
        self.assertEqual(doc["advice"], in_memory)

    def test_notes_are_optional_and_omitted_when_empty(self):
        """A `notes` key present but empty reads as "nothing was profiled"."""
        self.assertNotIn("notes", self._doc(self._all_advice()))
        self.assertNotIn("notes", self._doc(self._all_advice(), notes={}))
        doc = self._doc(self._all_advice(), notes={"dronet": {"n_dispatches": 21}})
        self.assertEqual(doc["notes"]["dronet"]["n_dispatches"], 21)

    def test_an_empty_advice_list_is_still_a_valid_document(self):
        """"Nothing to change" has to be expressible.

        A round that finds no actionable advice must write a document saying so,
        not fail to write one -- otherwise "no advice" and "the emitter crashed"
        are the same observation.
        """
        doc = self._doc([])
        self.assertEqual(doc["advice"], [])
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)


class AgainstTheRealK1Fixture(unittest.TestCase):
    """The same validation, on advice derived from a real measured profile.

    The synthetic profiles above are chosen to make each producer emit. This
    runs the whole thing over `fixtures/k1_profile/` -- 21 real DroNet
    dispatches, two real implementations -- so the schema is also checked
    against the field set a real profile actually provides (notably: no
    `cv_pct`, because that run recorded a single sample per dispatch).
    """

    FIXTURE = os.path.join(_HERE, "fixtures", "k1_profile", "gen")

    def _profiles(self):
        return compile_advice.load_profiles_csv(
            self.FIXTURE, "spacemit_x60", "dronet", "dronet.int8",
            ["rvv_x60", "scalar"])

    def setUp(self):
        if not hasattr(compile_advice, "load_profiles_csv"):
            self.skipTest("compile_advice has no results.csv ingester")
        self.profs = self._profiles()
        if set(self.profs) != {"rvv_x60", "scalar"}:
            self.skipTest(f"fixture profiles not resolvable: {sorted(self.profs)}")

    def test_the_sub_microsecond_sigmoid_switch_is_refused(self):
        """A large percentage of a negligible cost is not an opportunity.

        20 of DroNet's 21 dispatches are faster in the rvv_x60 build (9.79 ms
        total against 157.38 ms scalar, 16x). The single exception is dispatch
        20, `sigmoid_s8_n1`: 1.42 us vs 1.21 us, because the LUT-gather kernel's
        setup costs more than a one-element sigmoid.

        A purely RELATIVE gate passed that: 14.7% clears any percentage floor
        you would set for a real win, and the advisor emitted "scalar is 14.7%
        faster (1us vs 1us)" -- two identical numbers, because the difference
        does not survive rounding to the unit it is reported in. Acting on it
        costs a Codex call, a rebuild, a board run and a re-solve to save
        0.21 us.

        It is not merely wasteful. The acceptance objective is lexicographic
        with standalone kernel cycles LAST, so a 0.21 us change cannot move any
        term above it: the best case is a no-op and the realistic case is that
        it perturbs a schedule currently missing zero deadlines.

        `min_gain_ms` is the floor that catches it, and it has to coexist with
        the relative one -- a 5% gain on a 200 ms dispatch is still worth
        having, so neither floor alone is sufficient.
        """
        adv = implementation_advice("dronet", self.profs, "rvv_x60")
        self.assertEqual(len(adv), 21)
        switches = [a for a in adv if a.recommendation == "choose_implementation"]
        self.assertEqual([a.dispatch_id for a in switches], [],
                         [a.rationale for a in switches])

    def test_a_large_relative_gain_alone_is_not_enough(self):
        """Directly: the relative floor passes and the absolute one refuses."""
        profs = {
            "rvv_x60": {0: {"median_ms": 0.00142, "module_name": "m$d0_x_sigmoid_s8_n1"}},
            "scalar":  {0: {"median_ms": 0.00121, "module_name": "m$d0_x_sigmoid_s8_n1"}},
        }
        gain = (0.00142 - 0.00121) / 0.00142
        self.assertGreater(gain, 0.05, "the relative floor really does pass")
        self.assertEqual(
            [a for a in implementation_advice("dronet", profs, "rvv_x60")
             if a.recommendation == "choose_implementation"], [])
        # ... and a gain of the same PROPORTION on real work is still taken.
        big = {
            "rvv_x60": {0: {"median_ms": 14.2, "module_name": "m$d0_x_conv2d_s8_N1"}},
            "scalar":  {0: {"median_ms": 12.1, "module_name": "m$d0_x_conv2d_s8_N1"}},
        }
        self.assertEqual(
            len([a for a in implementation_advice("dronet", big, "rvv_x60")
                 if a.recommendation == "choose_implementation"]), 1)

    def test_switching_the_baseline_recommends_the_vector_build_everywhere(self):
        """The mirror case, which is what makes the previous test meaningful.

        17, not 20: three of DroNet's dispatches are faster under rvv_x60 by
        less than `min_gain_ms`, so the absolute floor refuses them in this
        direction too. A floor that only applied to the answer we disliked
        would be a thumb on the scale.
        """
        adv = implementation_advice("dronet", self.profs, "scalar")
        switches = [a for a in adv if a.recommendation == "choose_implementation"]
        self.assertEqual(len(switches), 17)
        for a in switches:
            self.assertEqual(a.evidence.extra["proposed_impl"], "rvv_x60")
            self.assertGreaterEqual(a.evidence.extra["gain_ms"], 0.05)

    def test_confidence_is_not_high_when_the_profile_has_no_dispersion(self):
        """This run took one sample per dispatch, so `cv_pct` is absent.

        Absent is not the same as small. Reporting `confidence="high"` off a
        profile that cannot support it is the failure mode the fixture's missing
        `cv_pct` exists to catch -- and it must hold on BOTH branches, which is
        where the two used to disagree.
        """
        for baseline in ("rvv_x60", "scalar"):
            for a in implementation_advice("dronet", self.profs, baseline):
                self.assertEqual(a.confidence, "medium",
                                 f"{baseline}/{a.dispatch_id}: {a.rationale}")

    def test_advice_from_the_real_profile_validates(self):
        adv = implementation_advice("dronet", self.profs, "scalar")
        base = self.profs["rvv_x60"]
        # DroNet needs 9.79 ms against a 33.3 ms period, so nothing blocks at
        # that budget; a 1 ms budget is what a co-running 1 kHz control loop
        # leaves, and that does produce split advice.
        adv += blocking_advice("dronet", base, free_slot_ms=1.0, misses=4)
        self.assertGreater(len(adv), 21)
        for i, a in enumerate(adv):
            _validate(a.as_dict(), self, f"real/{i}/{a.recommendation}")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TheVocabularyDescribesWhatTheSystemCanSay(unittest.TestCase):
    """A verb with no producer is a word the system cannot say.

    RECOMMENDATIONS carried eight verbs; three of them -- `fuse_with_predecessor`,
    `pin_core_class`, `coarsen` -- were never emitted by any producer. A
    contract that advertises capabilities it does not have is worse than a
    smaller one, because a consumer written against it fails at runtime in
    another repo.

    These tests keep the annotation in compile_advice.py honest: every verb
    the contract offers must have a producer, and every verb a producer emits
    must be in the contract.
    """

    def _module_src(self):
        import compile_advice
        with open(compile_advice.__file__) as f:
            return f.read()

    def test_every_offered_verb_has_a_producer(self):
        import compile_advice
        src = self._module_src()
        for verb in compile_advice.RECOMMENDATIONS:
            if verb == "unchanged":
                continue          # every refusal branch emits it
            self.assertIn(f'recommendation="{verb}"', src,
                          f"{verb!r} is offered by RECOMMENDATIONS but no "
                          f"producer in this module emits it. Either wire a "
                          f"producer or move it to RETIRED_RECOMMENDATIONS.")

    def test_every_emitted_verb_is_in_the_contract(self):
        import re
        import compile_advice
        emitted = set(re.findall(r'recommendation="([a-z_]+)"', self._module_src()))
        for verb in emitted:
            self.assertIn(verb, compile_advice.RECOMMENDATIONS,
                          f"{verb!r} is emitted but not offered by the "
                          f"contract, so a consumer validating against "
                          f"RECOMMENDATIONS would reject it")

    def test_retired_verbs_are_not_also_offered(self):
        import compile_advice
        overlap = set(compile_advice.RECOMMENDATIONS) & set(
            compile_advice.RETIRED_RECOMMENDATIONS)
        self.assertEqual(overlap, set(),
                         "a verb cannot be both offered and retired")

    def test_every_actionable_verb_is_offered(self):
        import compile_advice
        for verb in set(ACTIONABLE_EVIDENCE) | set(REQUIRED_CONSTRAINTS):
            self.assertIn(verb, compile_advice.RECOMMENDATIONS,
                          f"{verb!r} has evidence/constraint requirements but "
                          f"is not in the contract")
