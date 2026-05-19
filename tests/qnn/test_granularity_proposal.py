"""Unit tests for the granularity proposal builders."""

from __future__ import annotations

import dataclasses

from xpu_rt.targets.backends.qnn.granularity_proposal import (
    ArtifactRef, CostCell, GranularityProposal, Island,
    annotate_with_analytical_bounds, propose_fusions, propose_per_op,
    propose_shards, propose_whole_net,
)
from xpu_rt.targets.backends.qnn.target_spec import OpFootprint


def test_whole_net_has_single_island_covering_all_ops():
    p = propose_whole_net("yolov8n", ["op0", "op1", "op2"])
    assert len(p.islands) == 1
    assert p.islands[0].op_ids == ("op0", "op1", "op2")
    assert p.islands[0].kind == "whole_net"


def test_per_op_has_one_island_per_op_with_chain():
    p = propose_per_op("yolov8n", ["op0", "op1", "op2"])
    assert len(p.islands) == 3
    assert all(i.kind == "op" for i in p.islands)
    # Chain: op1 ← op0; op2 ← op1.
    assert p.islands[0].predecessor_island_ids == ()
    assert p.islands[1].predecessor_island_ids == (p.islands[0].island_id,)
    assert p.islands[2].predecessor_island_ids == (p.islands[1].island_id,)


def test_fusion_groups_two_ops_singleton_rest():
    p = propose_fusions("yolov8n", ["a", "b", "c", "d"],
                        fusion_groups=[("b", "c")])
    # a (singleton), b+c (fused), d (singleton).
    assert len(p.islands) == 3
    assert p.islands[0].op_ids == ("a",)
    assert p.islands[1].op_ids == ("b", "c")
    assert p.islands[1].kind == "fused"
    assert p.islands[2].op_ids == ("d",)


def test_sharded_proposal_has_k_parallel_islands():
    p = propose_shards("net", "matmul0", n_shards=4)
    assert len(p.islands) == 4
    assert all(i.kind == "sharded" for i in p.islands)
    # Sharded siblings have no predecessor edges between them.
    for i in p.islands:
        assert i.predecessor_island_ids == ()


def test_schedulable_backends_requires_measured_AND_artifact():
    # Island with a measured CPU cell but no artifact → not schedulable.
    isl = Island(
        island_id="i0", op_ids=("a",), kind="op", workload_id="w",
        cost={"CPU": CostCell(mean_us=100, provenance="measured")},
        executor_artifact={"CPU": None},
    )
    assert isl.schedulable_backends() == []
    # Add an artifact → schedulable.
    isl2 = dataclasses.replace(isl, executor_artifact={
        "CPU": ArtifactRef(remote_path="/r/a.dlc", kind="dlc", backend="CPU"),
    })
    assert isl2.schedulable_backends() == ["CPU"]


def test_planner_visible_includes_unbuilt():
    isl = Island(
        island_id="i0", op_ids=("a",), kind="op", workload_id="w",
        cost={"CPU": CostCell(mean_us=100, provenance="measured"),
              "GPU": CostCell(mean_us=80, provenance="analytical_bound")},
        executor_artifact={"CPU": None, "GPU": None},
    )
    pv = set(isl.planner_visible_backends())
    assert pv == {"CPU", "GPU"}
    assert isl.schedulable_backends() == []


def test_annotate_with_analytical_bounds_adds_only_missing_cells():
    p = propose_per_op("net", ["op0"])
    # Pre-fill op0 with a measured CPU cell.
    isl = p.islands[0]
    p2 = dataclasses.replace(p, islands=(dataclasses.replace(
        isl, cost={"CPU": CostCell(mean_us=42, provenance="measured")},
    ),))
    footprints = {"op0": OpFootprint(flops=1e9, bytes_read=1e6, bytes_written=1e6)}
    annotated = annotate_with_analytical_bounds(p2, footprints, ["CPU", "GPU"])
    cell_cpu = annotated.islands[0].cost["CPU"]
    cell_gpu = annotated.islands[0].cost["GPU"]
    # CPU stays measured; GPU now an analytical bound.
    assert cell_cpu.provenance == "measured"
    assert cell_cpu.mean_us == 42
    assert cell_gpu.provenance == "analytical_bound"
    assert cell_gpu.mean_us > 0
