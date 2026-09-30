"""What the width clamp is allowed to take away.

The clamp makes a solved schedule buildable: a packed convolution gets one shard width
across its periodic instances, and that width has to divide the dispatch's output
channels. Both rules are real. What is not a rule is "narrow it because I could not find
the output-channel count", and that is what the clamp did to 57 of this network's 63
packed convolutions -- a fused `conv2d_batchnorm2d_silu_s8` carries no shape of its own,
because the numbers belong to the `conv2d_s8` it fused and live one level down in
`sub_ops`. Every greedy and soft table in the repo was executed with its widths removed
that way, and on the chain spec it is the difference between 36 ms and 467 ms.

These pin the three sources of an output-channel count and the rules that stay.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

from clamp_schedule_widths import oc_of, oc_from_module_name   # noqa: E402

MODULE = ("yolov8_nano_64x96$dispatch_0_rvv_x60_conv2d_batchnorm2d_silu_s8_"
          "N1xIC3xIH64xIW96xOC16xOH32xOW48xKH3xKW3xSH2xSW2xPH1xPW1")


class OutputChannels(unittest.TestCase):
    def test_top_level_shape(self):
        self.assertEqual(oc_of({"op": "conv2d_s8", "shape": {"OC": 32}}), 32)

    def test_shape_as_a_string(self):
        self.assertEqual(oc_of({"op": "conv2d_s8", "shape": "N=1;IC=3;OC=16;OH=32"}), 16)

    def test_fused_op_keeps_its_shape_one_level_down(self):
        # The case the clamp missed: no shape at the top, the convolution's shape in sub_ops.
        op = {"op": "conv2d_batchnorm2d_silu_s8", "name": "l0.conv",
              "sub_ops": [{"op": "conv2d_s8", "shape": {"IC": 3, "OC": 16}}]}
        self.assertEqual(oc_of(op), 16)

    def test_module_name_is_the_second_source(self):
        self.assertEqual(oc_from_module_name(MODULE), 16)
        self.assertEqual(oc_from_module_name("no shape here"), 0)

    def test_nothing_to_find_still_reports_nothing(self):
        # Honest zero: the caller narrows on it, which is the conservative thing to do.
        self.assertEqual(oc_of({"op": "conv2d_s8"}), 0)


class ClampRules(unittest.TestCase):
    """End to end on a two-instance schedule, through the script's own main()."""

    def _run(self, ir_ops, targets):
        import subprocess
        d = tempfile.mkdtemp()
        ir = os.path.join(d, "graph.json"); sched = os.path.join(d, "s.json"); out = os.path.join(d, "o.json")
        json.dump({"ops": ir_ops}, open(ir, "w"))
        # a schedule's `dispatches` is keyed by entry, and `id` is the dispatch within the
        # network, shared by every periodic instance of it
        json.dump({"dispatches": {str(i): {
            "id": 0, "job_name": f"net{i}", "module_name": MODULE, "hardware_target": t,
            "start_time": 0.0, "duration": 1.0} for i, t in enumerate(targets)}}, open(sched, "w"))
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "clamp_schedule_widths.py"),
                        sched, f"net:{ir}", "--out", out], check=True, capture_output=True)
        return [e["hardware_target"] for e in json.load(open(out))["dispatches"].values()]

    def test_fused_conv_keeps_a_width_that_divides_oc(self):
        ir = [{"dispatch_id": 0, "op": "conv2d_batchnorm2d_silu_s8",
               "sub_ops": [{"op": "conv2d_s8", "shape": {"OC": 16}}]}]
        got = self._run(ir, ["CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3"] * 2)
        self.assertEqual(got, ["CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3"] * 2)

    def test_width_that_does_not_divide_oc_is_lowered(self):
        ir = [{"dispatch_id": 0, "op": "conv2d_s8", "shape": {"OC": 2}}]
        got = self._run(ir, ["CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3"] * 2)
        self.assertEqual(got, ["CPU_P#0+CPU_P#1"] * 2)   # 4 does not divide 2, 2 does

    def test_instances_are_brought_to_the_narrowest(self):
        ir = [{"dispatch_id": 0, "op": "conv2d_s8", "shape": {"OC": 16}}]
        got = self._run(ir, ["CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3", "CPU_P#0+CPU_P#1"])
        self.assertEqual(got, ["CPU_P#0+CPU_P#1", "CPU_P#0+CPU_P#1"])

    def test_width_is_never_raised(self):
        ir = [{"dispatch_id": 0, "op": "conv2d_s8", "shape": {"OC": 64}}]
        got = self._run(ir, ["CPU_P#0"] * 2)
        self.assertEqual(got, ["CPU_P#0"] * 2)


if __name__ == "__main__":
    unittest.main()


class MissingKernels(unittest.TestCase):
    """The gate that stops a board FATAL before the cross-build.

    `find_illegal_implementations` asks whether the CORE can run the implementation, and a
    convolution placed on `ime` at CPU_P#0 passes it: cluster 0 has the IME. Whether the
    implementation has a kernel for that op is a different question, and the answer for every
    convolution is no -- the IME has `linear_s8`/matmul only. Without this gate the run dies
    at entry 21 of yolov8_nano_64x96, after the solve, the clamp, the cross-build and the deploy.
    """

    def _dispatches(self, impl):
        return {"net0_dispatch_0": {"id": 0, "job_name": "net0", "impl": impl,
                                    "hardware_target": "CPU_P#0", "start_time": 0.0,
                                    "duration": 1.0}}

    def _tree(self, impl_dir, implementation):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "profile", impl_dir, "tgt", "net", "spec", "topo_0")
        os.makedirs(p)
        with open(os.path.join(p, "results.csv"), "w") as fh:
            fh.write("dispatch_id,mean_time,implementation\n0,1.0,%s\n" % implementation)
        return d

    def test_an_ime_row_naming_an_rvv_kernel_is_not_an_ime_kernel(self):
        from check_schedule_feasibility import find_missing_kernels
        root = self._tree("ime_x60", "curated[rvv]/rvv_oc_blocked_bn_silu_epilogue")
        bad = find_missing_kernels(self._dispatches("ime"), gen_root=root, target="tgt")
        self.assertEqual(len(bad), 1)
        self.assertIn("curated[rvv]", bad[0]["why"])

    def test_a_real_ime_kernel_passes(self):
        from check_schedule_feasibility import find_missing_kernels
        root = self._tree("ime_x60", "curated[ime]/ime_vmadot_4x4x8")
        self.assertEqual(find_missing_kernels(self._dispatches("ime"), gen_root=root, target="tgt"), [])

    def test_rvv_is_never_questioned(self):
        from check_schedule_feasibility import find_missing_kernels
        root = self._tree("ime_x60", "curated[rvv]/anything")
        self.assertEqual(find_missing_kernels(self._dispatches("rvv"), gen_root=root, target="tgt"), [])


class WidthProfile(unittest.TestCase):
    """A shard lever costed from a tree with no width dimension is decorative, and says so.

    Enabling sharding against tables that are copies of the one-hart table looks exactly like
    a solver that does not shard: every width prices the same and the search correctly never
    widens. On the deployed chain that was a 45.3 ms YOLO where the hardware does 23.6.
    """

    def _tree(self, sums):
        d = tempfile.mkdtemp()
        for tag, tot in sums.items():
            p = os.path.join(d, "profile", "rvv_x60", "tgt", "net", "spec", tag)
            os.makedirs(p)
            with open(os.path.join(p, "results.csv"), "w") as fh:
                fh.write("dispatch_id,mean_time\n0,%f\n" % tot)
        return d

    def test_flat_tables_are_reported(self):
        from profile_loader import report_width_scaling
        root = self._tree({"topo_0": 55.7, "topo_0_1": 55.6, "topo_0_1_2_3": 55.7})
        said = []
        flat = report_width_scaling(root, "tgt", ["net"], log=said.append)
        self.assertEqual(flat, ["net"])
        self.assertIn("FLAT", said[0])

    def test_a_real_width_dimension_is_not(self):
        from profile_loader import report_width_scaling
        root = self._tree({"topo_0": 55.7, "topo_0_1": 44.5, "topo_0_1_2_3": 24.7})
        said = []
        self.assertEqual(report_width_scaling(root, "tgt", ["net"], log=said.append), [])
        self.assertIn("2.26x", said[0].replace("2.25x", "2.26x"))

    def test_one_table_is_not_evidence_either_way(self):
        from profile_loader import report_width_scaling
        root = self._tree({"topo_0": 55.7})
        self.assertEqual(report_width_scaling(root, "tgt", ["net"], log=lambda *_: None), [])


class QueueDepth(unittest.TestCase):
    """What the Gantt records beyond the bars, once the baseline uses the whole machine.

    An eight-lane ROS row at 55 % on every hart looks healthier than a four-lane one at 88 %, while
    its chain is seven times longer. Bars cannot show that; the number of frames released and not
    yet answered can, and it is the queue the default QoS depth lets build.
    """

    def _rows(self, releases, spans, net="yolov8_nano_64x96"):
        return [{"net": net, "inst": i, "rel": rel, "s": a, "e": b}
                for i, (rel, (a, b)) in enumerate(zip(releases, spans))]

    def test_depth_counts_frames_released_but_not_answered(self):
        import make_measured_gantt_pair as G
        # three frames released 10 ms apart, each taking 25 ms: they overlap two deep
        rows = self._rows([0.0, 10.0, 20.0], [(0.0, 25.0), (10.0, 35.0), (20.0, 45.0)])
        d = dict(G.queue_depth(rows, 0.0, 40.0, step_ms=5.0))
        self.assertEqual(d[0.0], 1)
        self.assertEqual(d[20.0], 3)      # all three in flight
        self.assertEqual(d[40.0], 1)      # only the last is still going

    def test_a_frame_leaves_the_queue_when_its_own_perception_ends(self):
        import make_measured_gantt_pair as G
        rows = self._rows([0.0], [(0.0, 10.0)])
        d = dict(G.queue_depth(rows, 0.0, 20.0, step_ms=5.0))
        self.assertEqual(d[5.0], 1)
        self.assertEqual(d[10.0], 0)      # the end is exclusive: it is answered

    def test_late_frames_carry_how_far_over_they_are(self):
        import make_measured_gantt_pair as G
        rows = self._rows([0.0, 10.0], [(0.0, 30.0), (10.0, 20.0)])
        late = G.late_frames_in_window(rows, 0.0, 40.0, yolo_win=22.0)
        self.assertEqual([x["inst"] for x in late], [0])       # only the 30 ms one
        self.assertAlmostEqual(late[0]["over_ms"], 8.0)

    def test_frames_released_outside_the_drawn_window_are_not_marked(self):
        import make_measured_gantt_pair as G
        rows = self._rows([0.0, 500.0], [(0.0, 90.0), (500.0, 900.0)])
        late = G.late_frames_in_window(rows, 0.0, 100.0, yolo_win=22.0)
        self.assertEqual([x["inst"] for x in late], [0])
