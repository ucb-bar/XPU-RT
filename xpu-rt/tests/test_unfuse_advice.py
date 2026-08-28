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

    def test_a_working_fused_kernel_is_left_alone(self):
        """The common case, and the one that must never be disturbed."""
        kd = _kernels("rvv_conv2d_s8_a.c", "rvv_batchnorm2d_s8_a.c",
                      "rvv_silu_s8_a.c")
        a = unfuse_advice("y", {0: {"median_ms": 3.8,
                                    "implementation": "curated[rvv]/oc_blocked"}},
                          {0: _fused()}, kd)
        self.assertEqual([x.recommendation for x in a], ["unchanged"])
        self.assertIn("not a reference fallback", a[0].evidence.extra["reason"])

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
        prof = {0: {"median_ms": 5.0, "implementation": "curated[rvv]/x"},
                1: {"median_ms": 5.0, "implementation": "reference/scalar_ref"}}
        a = unfuse_advice("y", prof, {0: _fused(), 1: _fused()}, kd)
        kinds = {x.dispatch_id: x.recommendation for x in a}
        self.assertEqual(kinds, {0: "unchanged", 1: "unfuse"})


if __name__ == "__main__":
    unittest.main()
