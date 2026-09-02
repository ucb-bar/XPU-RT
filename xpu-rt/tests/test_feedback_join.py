"""The measured run reaches the advice. Closing the loop's return edge.

WHAT WAS MISSING. `xpurt_feedback.json` had two producers -- the solver's
`--emit-feedback` and `streaming_feedback.py` -- and NO READER anywhere. A
producer with no consumer is exactly the shape of the problem the shard chain
was written to fix, pointing the other way, and it shipped that way until
someone asked whether the loop was actually closed.

WHY THE CONSUMER IS THE ADVICE PRODUCER. Turning a runtime hint into a graph
rewrite needs things the hint does not carry: a split factor needs the
periodic budget, a fusion needs the group of dispatches, `pin_target` names a
machine combination rather than a kernel implementation. Only
`emit_compile_advice` holds the graph and the budget, so a direct
feedback-to-hint bridge would have to invent the numbers the loop exists to
measure.

So the run CORROBORATES or CONTRADICTS advice derived from profiles. It never
manufactures it. These tests pin that boundary, because a join that quietly
started inventing advice would look like a working loop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir))

import compile_advice                                   # noqa: E402
import feedback_join                                    # noqa: E402


def _advice(rec, did=0, model="dronet", confidence="medium", priority=1):
    return compile_advice.Advice(
        model=model, dispatch_id=did, recommendation=rec, priority=priority,
        confidence=confidence, rationale="static reason",
        evidence=compile_advice.Evidence(service_time_us=1000.0))


def _doc(hints_by_key, run_id="r1"):
    return {"schema_version": 1, "run_id": run_id,
            "dispatches": {k: {"hints": v} for k, v in hints_by_key.items()}}


class TheRunCorroborates(unittest.TestCase):

    def test_prefer_finer_raises_the_confidence_of_a_split(self):
        adv = [_advice("split")]
        out, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(counts["corroborated"], 1)
        self.assertEqual(out[0].confidence, "high")
        self.assertEqual(out[0].recommendation, "split",
                         "corroboration must not change WHAT is advised")
        self.assertIn("corroborated", out[0].rationale)

    def test_the_evidence_records_what_the_run_said(self):
        """A reader must be able to see why the confidence is what it is."""
        adv = [_advice("split")]
        out, _ = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_finer"]}, run_id="k1_x"),
            "dronet", {"dronet"})
        ex = out[0].evidence.extra
        self.assertEqual(ex["runtime_hints"], ["prefer_finer"])
        self.assertEqual(ex["corroborated_by_measurement"], ["prefer_finer"])
        self.assertEqual(ex["runtime_run_id"], "k1_x")

    def test_confidence_saturates_rather_than_running_off_the_scale(self):
        adv = [_advice("split", confidence="high")]
        out, _ = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(out[0].confidence, "high")


class TheRunContradicts(unittest.TestCase):

    def test_prefer_coarser_demotes_a_split_to_unchanged(self):
        adv = [_advice("split")]
        out, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_coarser"]}), "dronet",
            {"dronet"})
        self.assertEqual(counts["contradicted"], 1)
        self.assertEqual(out[0].recommendation, "unchanged")
        self.assertIn("DEMOTED", out[0].rationale)
        self.assertEqual(out[0].evidence.extra["demoted_by_measurement"],
                         ["prefer_coarser"])

    def test_prefer_finer_does_NOT_contradict_a_fusion(self):
        """The contradiction table is deliberately not the complement.

        A dispatch can be both slower than predicted AND worth fusing with a
        neighbour. Treating the hints as opposites would suppress correct
        advice on exactly the dispatches under most pressure.
        """
        adv = [_advice("fuse")]
        out, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(out[0].recommendation, "fuse")
        self.assertEqual(counts["contradicted"], 0)


class ItNeverInventsAdvice(unittest.TestCase):
    """The boundary that makes this honest."""

    def test_a_hint_about_a_dispatch_with_no_advice_adds_nothing(self):
        adv = [_advice("split", did=0)]
        out, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_7": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(len(out), 1, "the run must not create advice items")
        self.assertEqual(counts["silent"], 1)

    def test_an_empty_feedback_document_changes_nothing(self):
        adv = [_advice("split")]
        before = (adv[0].recommendation, adv[0].confidence, adv[0].rationale)
        out, counts = feedback_join.join(adv, None, "dronet", {"dronet"})
        self.assertEqual(
            (out[0].recommendation, out[0].confidence, out[0].rationale),
            before, "absent feedback must be byte-identical to no feedback")
        self.assertEqual(counts, {"corroborated": 0, "contradicted": 0,
                                  "not_applicable": 0, "silent": 0})


class SilenceAndIrrelevanceAreDifferentFacts(unittest.TestCase):
    """"the run said nothing" and "the run said something unrelated" are not
    the same, and only the first means the measurement missed the dispatch."""

    def test_no_hints_for_the_dispatch_is_silence(self):
        adv = [_advice("split", did=0)]
        _, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_9": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(counts["silent"], 1)
        self.assertEqual(counts["not_applicable"], 0)

    def test_hints_that_do_not_bear_on_the_recommendation_are_not_silence(self):
        adv = [_advice("unchanged", did=0)]
        _, counts = feedback_join.join(
            adv, _doc({"dronet0_dispatch_0": ["prefer_finer"]}), "dronet",
            {"dronet"})
        self.assertEqual(counts["not_applicable"], 1)
        self.assertEqual(counts["silent"], 0)
        self.assertEqual(adv[0].evidence.extra["runtime_hints"],
                         ["prefer_finer"],
                         "the run's report is recorded even when it does not "
                         "change the verdict")


class InstancesAreUnionedOntoDispatches(unittest.TestCase):
    """Feedback is per INSTANCE; advice is per DISPATCH."""

    def test_a_hint_in_any_instance_reaches_the_dispatch(self):
        adv = [_advice("split", did=3)]
        out, counts = feedback_join.join(
            adv,
            _doc({"dronet0_dispatch_3": [],
                  "dronet7_dispatch_3": ["prefer_finer"]}),
            "dronet", {"dronet"})
        self.assertEqual(counts["corroborated"], 1)
        self.assertEqual(out[0].confidence, "high")

    def test_another_models_hints_are_not_read(self):
        adv = [_advice("split", did=0, model="dronet")]
        out, counts = feedback_join.join(
            adv, _doc({"mlp_control0_dispatch_0": ["prefer_coarser"]}),
            "dronet", {"dronet", "mlp_control"})
        self.assertEqual(out[0].recommendation, "split")
        self.assertEqual(counts["silent"], 1)

    def test_a_digit_ending_network_is_split_at_the_right_place(self):
        """The same trap as everywhere else: the key carries the instance."""
        adv = [_advice("split", did=2, model="yolov8_nano_64x96")]
        out, counts = feedback_join.join(
            adv, _doc({"yolov8_nano_64x960_dispatch_2": ["prefer_finer"]}),
            "yolov8_nano_64x96", {"yolov8_nano_64x96"})
        self.assertEqual(counts["corroborated"], 1,
                         "instance 0 of yolov8_nano_64x96, not instance 960 "
                         "of yolov8_nano_64x")


class LoadIsForgiving(unittest.TestCase):

    def test_absent_is_None_not_an_error(self):
        self.assertIsNone(feedback_join.load("/nonexistent/xpurt_feedback.json"))

    def test_malformed_is_None_not_an_exception(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fb.json"
            p.write_text("{not json")
            self.assertIsNone(feedback_join.load(p))
            p.write_text('{"schema_version": 1}')
            self.assertIsNone(feedback_join.load(p))


if __name__ == "__main__":
    unittest.main()
