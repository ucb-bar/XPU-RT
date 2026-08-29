"""The streaming feedback channel, which until now had neither end connected.

`streaming_feedback.py` shipped with no importer, no test, and both of its
endpoints missing: it tailed a JSON-Lines format that no runner in either repo
emitted (`TelemetrySink::EmitDispatchEnd` existed nowhere), and posted the
result through an `ingest_xpurt_feedback` MCP tool that was not in the merlin
checkout either. It was a plausible-looking module that could not have run.

The producer now exists -- `generate_xpurt_main.py` emits one JSON line per
dispatch END when built with `-DMODELBLASTER_XPURT_STREAM` -- and the consumer
writes a file rather than calling a tool.

THE TEST THAT MATTERS MOST is `test_the_emitters_fields_are_the_parsers_fields`.
The producer is generated C and the consumer is Python; nothing but agreement
between two files keeps them speaking the same language, and a rename on either
side would silently yield zero hints rather than an error. So the field names
are checked against the actual generated source.
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

import streaming_feedback as sf  # noqa: E402


def _line(net="dronet", inst=0, did=3, impl="rvv",
          pred_ms=1.0, t0=0, t1=24_000):
    """One walker telemetry line. 24_000 ticks = 1.000 ms at 24 MHz."""
    return {"entry_id": did, "network": net, "instance": inst,
            "dispatch_id": did, "impl": impl, "hart": 0,
            "predicted_start_ms": 0.0, "predicted_duration_ms": pred_ms,
            "start_ticks": t0, "end_ticks": t1}


class TicksAreRdtimeAt24MHz(unittest.TestCase):

    def test_a_tick_window_becomes_microseconds(self):
        ev = sf.normalise_event(_line(t0=0, t1=24_000))
        self.assertAlmostEqual(ev["run_us"], 1000.0)

    def test_the_constant_is_24_not_the_core_clock(self):
        """1.6 GHz or 1 MHz would both produce plausible-looking numbers."""
        self.assertEqual(sf.TICKS_PER_US, 24.0)

    def test_a_line_that_is_not_a_dispatch_end_is_skipped_not_crashed(self):
        self.assertIsNone(sf.normalise_event({"hello": "world"}))
        self.assertIsNone(sf.normalise_event({"network": "d", "instance": 0}))


class ADeadlineTheBoardDoesNotKnowIsUnknownNotZero(unittest.TestCase):
    """The failure this guards against has already happened once.

    A network whose name ended in a digit had its instance index misparsed, so
    its deadline became ~48 s and it could never miss. The reported zero was
    structural, not measured, and it looked exactly like a pass.
    """

    def test_without_a_spec_the_miss_is_None(self):
        self.assertIsNone(sf.normalise_event(_line())["deadline_miss"])

    def test_None_is_not_counted_as_a_miss_nor_as_a_pass(self):
        window = [sf.normalise_event(_line(inst=i)) for i in range(10)]
        payload = sf._derive_streaming_hints(window, run_id="r")
        for rec in payload["dispatches"].values():
            self.assertNotIn("consider_fuse_with_pred", rec["hints"],
                             "an unknown miss rate must not fire the "
                             "miss-driven hints")

    def test_with_a_spec_a_late_dispatch_misses(self):
        # instance 0, deadline 1 ms, finishing at 2 ms.
        ev = sf.normalise_event(_line(t1=48_000), windows_ms={"dronet": 1.0})
        self.assertTrue(ev["deadline_miss"])
        ev2 = sf.normalise_event(_line(t1=12_000), windows_ms={"dronet": 1.0})
        self.assertFalse(ev2["deadline_miss"])


class TheHintsFollowMeasuredAgainstPredicted(unittest.TestCase):

    def _hints(self, ratio):
        """`ratio` is measured/predicted."""
        window = [sf.normalise_event(_line(inst=i, pred_ms=1.0,
                                           t1=int(24_000 * ratio)))
                  for i in range(8)]
        payload = sf._derive_streaming_hints(window, run_id="r")
        recs = list(payload["dispatches"].values())
        return recs[0]["hints"] if recs else []

    def test_slower_than_predicted_asks_for_finer(self):
        self.assertIn("prefer_finer", self._hints(2.0))

    def test_much_faster_than_predicted_asks_for_coarser(self):
        self.assertIn("prefer_coarser", self._hints(0.2))

    def test_on_prediction_says_nothing(self):
        self.assertEqual([h for h in self._hints(1.0)
                          if not h.startswith("pin_target")], [])


class TheFileMergesRatherThanOverwrites(unittest.TestCase):
    """A long run posts many times; instance 40 being quiet must not erase
    what instance 4 established."""

    def _payload(self, run_id, hints):
        return {"schema_version": 1, "run_id": run_id,
                "source_schedule": "streaming_feedback",
                "model_signals": {},
                "dispatches": {"dronet0_dispatch_3": {"hints": list(hints)}}}

    def test_same_run_id_unions_the_hints(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fb.json"
            sf.write_payload(self._payload("r1", ["prefer_finer"]), out)
            sf.write_payload(self._payload("r1", ["pin_target=ime"]), out)
            got = json.loads(out.read_text())
            self.assertEqual(
                sorted(got["dispatches"]["dronet0_dispatch_3"]["hints"]),
                ["pin_target=ime", "prefer_finer"])

    def test_a_different_run_id_starts_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "fb.json"
            sf.write_payload(self._payload("r1", ["prefer_finer"]), out)
            sf.write_payload(self._payload("r2", ["prefer_coarser"]), out)
            got = json.loads(out.read_text())
            self.assertEqual(
                got["dispatches"]["dronet0_dispatch_3"]["hints"],
                ["prefer_coarser"],
                "a new campaign must not inherit the last one's conclusions")


class TheProducerAndTheConsumerAgree(unittest.TestCase):

    def test_the_emitters_fields_are_the_parsers_fields(self):
        """Generated C on one side, Python on the other, nothing between."""
        mb = Path(__file__).resolve().parents[2] / "ModelBlaster"
        if not (mb / "pipeline" / "generate_xpurt_main.py").exists():
            raise unittest.SkipTest("ModelBlaster not checked out")
        sys.path.insert(0, str(mb))
        sys.path.insert(0, str(mb / "src"))
        try:
            from pipeline import generate_xpurt_main as gen
        except ImportError as e:                       # pragma: no cover
            raise unittest.SkipTest(f"cannot import the generator: {e}")

        # The GENERATED C, not the generator's source: the escaping between
        # the two is exactly where a field name would be lost.
        src = gen._emit(networks=["m"], schedule_name="s",
                        dispatch_table_header="t.h", core_kinds=["rvv"],
                        backends=["rvv_x60"], pool_sizes=[1],
                        n_instances={"m": 1}, platform="linux")
        block = src[src.index("MODELBLASTER_XPURT_STREAM"):]
        block = block[:block.index("#endif")]
        for field in ("entry_id", "network", "instance", "dispatch_id",
                      "impl", "hart", "predicted_start_ms",
                      "predicted_duration_ms", "start_ticks", "end_ticks"):
            self.assertIn(f'\\"{field}\\"', block,
                          f"the walker no longer emits {field!r}, which "
                          f"normalise_event reads")

    def test_the_line_the_walker_emits_round_trips(self):
        """A literal line of the emitted shape must normalise cleanly."""
        raw = json.loads(
            '{"entry_id":7,"network":"dronet","instance":2,"dispatch_id":13,'
            '"impl":"ime","hart":3,"predicted_start_ms":1.250000,'
            '"predicted_duration_ms":0.500000,'
            '"start_ticks":1000,"end_ticks":13000}')
        ev = sf.normalise_event(raw)
        self.assertEqual(ev["dispatch_id"], "dronet2_dispatch_13")
        self.assertEqual(ev["target"], "ime")
        self.assertAlmostEqual(ev["run_us"], 500.0)
        self.assertAlmostEqual(ev["planned_duration_us"], 500.0)


if __name__ == "__main__":
    unittest.main()
