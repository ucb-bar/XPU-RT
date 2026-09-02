"""`unfuse` must fire on one measured failure and refuse everything else.

WHY THE GATE IS NARROW. Undoing a fusion is almost always wrong. The curated
`conv2d_batchnorm2d_silu_s8` kernel is 97% of yolov8n's runtime and applies BN
and SiLU as a table lookup inside the conv's OC-blocked register tile; unfusing
it loses the epilogue fusion, takes yolov8n from 63 to 177 dispatches, and adds
two full passes over the output tensor. `advisor.py` reading this workload
reports `granularity: too_fine` -- the loop's own measurement wants FEWER
dispatches, not more.

WHY IT EXISTS ANYWAY. Curated kernels are looked up by exact op name, so a
fused op matches no per-constituent kernel and falls back to the scalar
reference *inside a build labelled rvv_x60*. That is not hypothetical:

    yolov8_nano  rvv_x60   57 of 90 dispatches on reference, 99.8% of 4974.8 ms
                           -- 0.81x against the pure-scalar build

When each constituent does have a vector kernel, unfusing turns one scalar
dispatch into three vector ones. Both facts are readable from artifacts: the
profile's `implementation` column says what ran, the kernels directory says
what exists. So the trigger is measured, not modelled.

The tests below pin both directions, because a verb that fires too readily here
reintroduces the 0.81x result and a verb that never fires is dead vocabulary.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile_advice import unfuse_advice  # noqa: E402


def _fused(kind="conv2d_batchnorm2d_silu_s8",
           subs=("conv2d_s8", "batchnorm2d_s8", "silu_s8")):
    return {"op": kind, "sub_ops": [{"op": s} for s in subs]}


def _kernels(*names):
    d = tempfile.mkdtemp()
    for n in names:
        open(os.path.join(d, n), "w").close()
    return d


class ItFiresOnlyOnAReferenceFallback(unittest.TestCase):

    def test_a_small_working_fused_kernel_is_left_alone(self):
        """A working fused kernel that is a SMALL share of the model.

        This used to assert that any working fused kernel is left alone, on
        the reasoning that a curated fused kernel beats its constituents. That
        reasoning was disproved on the board -- see
        `AWorkingFusedKernelIsAQuestionNotAnAnswer` below -- so the rule now
        turns on whether the op is worth a board rung, not on whether the
        kernel is a fallback.
        """
        kd = _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                      "rvv_silu_s8_a.c")
        prof = {0: {"median_ms": 3.8,
                    "implementation": "curated[rvv]/oc_blocked"},
                1: {"median_ms": 96.2, "implementation": "curated[rvv]/x"}}
        a = unfuse_advice("y", prof, {0: _fused()}, kd)
        self.assertEqual([x.recommendation for x in a], ["unchanged"])
        self.assertIn("not a reference fallback", a[0].evidence.extra["reason"])
        self.assertLess(a[0].evidence.extra["kind_runtime_share"], 0.25)

    def test_a_reference_fallback_with_covered_constituents_fires(self):
        kd = _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                      "rvv_silu_s8_a.c")
        a = unfuse_advice("y", {0: {"median_ms": 87.3,
                                    "implementation": "reference/scalar_ref"}},
                          {0: _fused()}, kd)
        self.assertEqual(a[0].recommendation, "unfuse")
        ev = a[0].evidence.extra
        self.assertEqual(ev["fused_impl"], "reference/scalar_ref")
        self.assertEqual(set(ev["constituent_impls"]),
                         {"conv2d_s8", "batchnorm2d_s8", "silu_s8"})
        self.assertTrue(all(ev["constituent_impls"].values()))
        self.assertTrue(a[0].constraints["requires_constituent_kernels"])

    def test_a_fallback_whose_constituents_also_lack_kernels_is_refused(self):
        """Otherwise it is the same problem with more dispatches."""
        kd = _kernels("rvv_conv2d_s8_a.c")          # no batchnorm/silu kernel
        a = unfuse_advice("y", {0: {"median_ms": 87.3,
                                    "implementation": "reference/scalar_ref"}},
                          {0: _fused()}, kd)
        self.assertEqual(a[0].recommendation, "unchanged")
        self.assertIn("no curated kernel", a[0].evidence.extra["reason"])

    def test_without_a_kernels_dir_it_refuses_rather_than_assumes(self):
        """Unverifiable is not the same as satisfied."""
        a = unfuse_advice("y", {0: {"median_ms": 87.3,
                                    "implementation": "reference/scalar_ref"}},
                          {0: _fused()}, kernels_dir=None)
        self.assertEqual(a[0].recommendation, "unchanged")
        self.assertIn("cannot verify", a[0].evidence.extra["reason"])

    def test_an_unfused_op_is_not_a_candidate(self):
        kd = _kernels("rvv_conv2d_s8_a.c")
        a = unfuse_advice("y", {0: {"median_ms": 1.0,
                                    "implementation": "reference/scalar_ref"}},
                          {0: {"op": "conv2d_s8"}}, kd)
        self.assertEqual(a, [], "a plain op has nothing to unfuse")


class TheEvidenceIsMeasuredNotModelled(unittest.TestCase):

    def test_every_record_names_what_actually_ran(self):
        """The whole gate rests on `implementation`, so it must be carried."""
        kd = _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                      "rvv_silu_s8_a.c")
        for impl in ("curated[rvv]/x", "reference/scalar_ref"):
            a = unfuse_advice("y", {0: {"median_ms": 5.0,
                                        "implementation": impl}},
                              {0: _fused()}, kd)
            self.assertIn("fused_impl", a[0].evidence.extra, impl)

    def test_no_recommendation_is_derived_from_dispatch_count(self):
        """A granularity verdict must not be able to trigger this.

        Two fused ops identical except for `implementation`; only the
        reference one may fire. If dispatch count or op-kind could trigger it,
        both would.
        """
        kd = _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                      "rvv_silu_s8_a.c")
        # Op 0 is deliberately a SMALL share (5 of 100 ms), so the
        # runtime-share probe cannot fire on it and the only thing that could
        # make it fire is its `implementation` -- which is what this test is
        # about.
        prof = {0: {"median_ms": 5.0, "implementation": "curated[rvv]/x"},
                1: {"median_ms": 5.0, "implementation": "reference/scalar_ref"},
                2: {"median_ms": 90.0, "implementation": "curated[rvv]/x"}}
        a = unfuse_advice("y", prof, {0: _fused(), 1: _fused()}, kd)
        kinds = {x.dispatch_id: x.recommendation for x in a}
        self.assertEqual(kinds, {0: "unchanged", 1: "unfuse"},
                         "op 0 runs a curated kernel and is a small share, so "
                         "it stays unchanged; op 1 is a reference fallback")


class AWorkingFusedKernelIsAQuestionNotAnAnswer(unittest.TestCase):
    """The rule this file used to encode was disproved on the board.

    `unfuse_advice` refused to propose anything whose fused kernel was not a
    reference fallback, reasoning that a curated fused kernel beats its
    constituents. Measured on yolov8_nano, rvv_x60, K1:

        curated fused conv+BN+SiLU   218.128 ms
        unfused into constituents    176.370 ms   -19.1%, ACCEPTED at term 5

    The lost epilogue fusion costs 4.655 ms (57 BN + 57 SiLU passes) and buys
    46.5 ms, because the fused kernel's conv inner loop is slower than
    `rvv_conv2d_s8_rvv_vsmul_vnclip.c`. The old gate was ASSERTING an outcome
    the loop exists to MEASURE, and the price of that certainty was a 19% win
    that nothing could propose.

    So a big enough fused op with covered constituents is now a PROBE: low
    confidence, priority 3, `measure_before_adopting`. If the fused kernel is
    genuinely better the loop rejects it, which costs one rung and is the loop
    working rather than failing.
    """

    def _kd(self):
        return _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                        "rvv_silu_s8_a.c")

    def test_a_dominant_working_fused_kernel_is_proposed_as_a_probe(self):
        a = unfuse_advice("y", {0: {"median_ms": 97.0,
                                    "implementation": "curated[rvv]/oc_blocked"}},
                          {0: _fused()}, self._kd())
        self.assertEqual(a[0].recommendation, "unfuse")
        self.assertEqual(a[0].evidence.extra["trigger"], "runtime_share_probe")
        self.assertEqual(a[0].confidence, "low",
                         "a probe must not claim the confidence of a measured "
                         "fallback")
        self.assertTrue(a[0].constraints["measure_before_adopting"])

    def test_the_probe_needs_covered_constituents(self):
        """Without kernels to land on, unfusing reproduces the 0.81x result."""
        kd = _kernels("rvv_conv2d_s8_a.c")          # no bn, no silu
        a = unfuse_advice("y", {0: {"median_ms": 97.0,
                                    "implementation": "curated[rvv]/x"}},
                          {0: _fused()}, kd)
        self.assertEqual(a[0].recommendation, "unchanged")

    def test_the_share_is_aggregated_over_the_op_KIND(self):
        """The unit is the kernel, not the dispatch.

        yolov8_nano's fused convs are 1.69% of runtime EACH and 96.2%
        TOGETHER. Any per-dispatch threshold low enough to catch them fires on
        everything; any threshold selective enough never fires. Measured on the
        shipping build, a per-dispatch rule proposed nothing at all.
        """
        prof = {i: {"median_ms": 10.0, "implementation": "curated[rvv]/x"}
                for i in range(10)}
        ops = {i: _fused() for i in range(10)}
        a = unfuse_advice("y", prof, ops, self._kd())
        self.assertEqual({x.recommendation for x in a}, {"unfuse"},
                         "10 dispatches of one kind at 10% each are 100% of "
                         "the model as a KIND, which is the unit that decides")
        self.assertEqual(a[0].evidence.extra["kind_dispatch_count"], 10)

    def test_a_kind_that_is_a_small_share_of_the_model_proposes_nothing(self):
        """One rung per probe, so only a kernel worth a rung may propose one."""
        prof = {0: {"median_ms": 5.0, "implementation": "curated[rvv]/x"},
                1: {"median_ms": 95.0, "implementation": "curated[rvv]/other"}}
        ops = {0: _fused(), 1: {"op": "conv2d_s8"}}
        a = unfuse_advice("y", prof, ops, self._kd())
        self.assertEqual([x.recommendation for x in a], ["unchanged"],
                         "5% of the model is not worth a board rung")


if __name__ == "__main__":
    unittest.main()
