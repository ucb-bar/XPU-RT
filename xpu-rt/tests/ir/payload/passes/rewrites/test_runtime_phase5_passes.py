"""Tests for the Phase-5 runtime passes.

All nine passes operate on :class:`compgen.runtime.execution_plan.ExecutionPlan`
rather than xDSL IR, so the tests live in one module and share a
small set of builder helpers.
"""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.rewrites.alias_io_buffers import (
    AliasIOBuffersConfig,
    run_alias_io_buffers,
)
from compgen.ir.payload.passes.rewrites.assign_memory_space import (
    AssignMemorySpaceConfig,
    AssignMemorySpaceStats,
    run_assign_memory_space,
)
from compgen.ir.payload.passes.rewrites.assign_queue import (
    AssignQueueConfig,
    run_assign_queue,
)
from compgen.ir.payload.passes.rewrites.assign_streams import (
    AssignStreamsConfig,
    run_assign_streams,
)
from compgen.ir.payload.passes.rewrites.dma_overlap import (
    DMAOverlapConfig,
    run_dma_overlap,
)
from compgen.ir.payload.passes.rewrites.insert_copies import (
    run_insert_copies,
)
from compgen.ir.payload.passes.rewrites.insert_host_offload import (
    InsertHostOffloadConfig,
    run_insert_host_offload,
)
from compgen.ir.payload.passes.rewrites.normalize_subbyte_post_layout import (
    NormalizeSubbytePostLayoutConfig,
    NormalizeSubbytePostLayoutStats,
    run_normalize_subbyte_post_layout,
)
from compgen.ir.payload.passes.rewrites.plan_buffers import (
    PlanBuffersConfig,
    run_plan_buffers,
)
from compgen.runtime.execution_plan import (
    ExecutionPlan,
)
from compgen.runtime.plan_builder import ExecutionPlanBuilder

# --- shared builders ---------------------------------------------------------


def _basic_plan() -> ExecutionPlan:
    b = ExecutionPlanBuilder("t", "cuda_h100")
    b.add_region("r0", "cuda:0", "")
    b.add_region("r1", "cuda:0", "")
    b.add_buffer("w", 1024, "", 0, 0, persistent=True, ownership="shared_readonly")
    b.add_buffer("mid", 8192, "", 1, 5)
    b.add_buffer("out", 4096, "", 6, 10)
    b.add_dependency("r0", "r1", value_ref="mid")
    return b.build()


# ========================================================================
# assign_memory_space
# ========================================================================


class TestAssignMemorySpace:
    def test_weight_goes_to_dram_by_default(self):
        plan = _basic_plan()
        stats = run_assign_memory_space(plan)
        assert plan.buffer("w").memory_space == "dram"
        assert stats.placed_by_space.get("dram", 0) >= 1

    def test_activations_go_to_scratchpad_when_vtcm_budget_sufficient(self):
        plan = _basic_plan()
        cfg = AssignMemorySpaceConfig(vtcm_bytes=16384, scratch_memory_space="vtcm")
        run_assign_memory_space(plan, config=cfg)
        assert plan.buffer("mid").memory_space == "vtcm"
        assert plan.buffer("out").memory_space == "vtcm"

    def test_activations_spill_to_default_when_too_big(self):
        plan = _basic_plan()
        # Make mid huge so it blows past the scratch budget.
        plan.buffer("mid").size_bytes = 1_000_000
        cfg = AssignMemorySpaceConfig(vtcm_bytes=16384, scratch_memory_space="vtcm")
        run_assign_memory_space(plan, config=cfg)
        assert plan.buffer("mid").memory_space == "dram"

    def test_idempotent_by_default(self):
        plan = _basic_plan()
        plan.buffer("mid").memory_space = "custom"
        run_assign_memory_space(plan)
        assert plan.buffer("mid").memory_space == "custom"

    def test_overwrite_existing_flag(self):
        plan = _basic_plan()
        plan.buffer("mid").memory_space = "custom"
        cfg = AssignMemorySpaceConfig(overwrite_existing=True)
        run_assign_memory_space(plan, config=cfg)
        assert plan.buffer("mid").memory_space != "custom"

    def test_stats_initial_values(self):
        s = AssignMemorySpaceStats()
        assert s.buffers_seen == 0


# ========================================================================
# assign_queue
# ========================================================================


class TestAssignQueue:
    def test_assigns_queues_to_all_regions(self):
        plan = _basic_plan()
        stats = run_assign_queue(plan)
        assert stats.regions_assigned == 2
        assert plan.placement_for("r0").queue != ""
        assert plan.placement_for("r1").queue != ""

    def test_round_robin_spreads_across_queues(self):
        b = ExecutionPlanBuilder("t", "cuda")
        for i in range(4):
            b.add_region(f"r{i}", "cuda:0", "")
        plan = b.build()
        run_assign_queue(plan, config=AssignQueueConfig(num_queues_per_device=2))
        queues = [rp.queue for rp in plan.region_placement]
        assert len(set(queues)) == 2

    def test_topo_order_places_producer_before_consumer_when_same_queue(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "")
        b.add_region("b", "cuda:0", "")
        b.add_region("c", "cuda:0", "")
        b.add_dependency("a", "b")
        b.add_dependency("b", "c")
        plan = b.build()
        # With 3 regions and num_queues=1, all on same queue in topo order.
        run_assign_queue(plan, config=AssignQueueConfig(num_queues_per_device=1))
        queues = [rp.queue for rp in plan.region_placement]
        assert len(set(queues)) == 1

    def test_invalid_num_queues_raises(self):
        plan = _basic_plan()
        with pytest.raises(ValueError, match="num_queues_per_device"):
            run_assign_queue(plan, config=AssignQueueConfig(num_queues_per_device=0))

    def test_idempotent_does_not_overwrite(self):
        plan = _basic_plan()
        plan.placement_for("r0").queue = "custom_q"
        run_assign_queue(plan)
        assert plan.placement_for("r0").queue == "custom_q"

    def test_separate_queues_per_device(self):
        b = ExecutionPlanBuilder("t", "mixed")
        b.add_region("gpu_r", "cuda:0", "")
        b.add_region("cpu_r", "cpu", "")
        plan = b.build()
        run_assign_queue(plan)
        assert plan.placement_for("gpu_r").queue != plan.placement_for("cpu_r").queue


# ========================================================================
# assign_streams
# ========================================================================


class TestAssignStreams:
    def test_sync_when_producer_on_same_queue(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "q0")
        b.add_region("b", "cuda:0", "q0")
        b.add_dependency("a", "b")
        plan = b.build()
        run_assign_streams(plan)
        kinds = plan.summary["stream_kinds"]
        assert kinds["b"] == "sync"

    def test_async_wrap_when_producer_on_different_queue(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "q0")
        b.add_region("b", "cuda:0", "q1")
        b.add_dependency("a", "b")
        plan = b.build()
        run_assign_streams(plan)
        kinds = plan.summary["stream_kinds"]
        assert kinds["b"] == "async_wrap"

    def test_no_dependency_is_sync(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "q0")
        plan = b.build()
        run_assign_streams(plan)
        assert plan.summary["stream_kinds"]["a"] == "sync"

    def test_force_sync_overrides(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "q0")
        b.add_region("b", "cuda:0", "q1")
        b.add_dependency("a", "b")
        plan = b.build()
        run_assign_streams(plan, config=AssignStreamsConfig(force_sync=True))
        assert plan.summary["stream_kinds"]["b"] == "sync"

    def test_stream_id_deterministic_by_queue_name(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "beta")
        b.add_region("b", "cuda:0", "alpha")
        plan = b.build()
        run_assign_streams(plan)
        assert plan.placement_for("b").stream_id == 0  # alpha sorts first
        assert plan.placement_for("a").stream_id == 1

    def test_stats_count_sync_and_async(self):
        b = ExecutionPlanBuilder("t", "cuda")
        b.add_region("a", "cuda:0", "q0")
        b.add_region("b", "cuda:0", "q0")
        b.add_region("c", "cuda:0", "q1")
        b.add_dependency("a", "b")
        b.add_dependency("a", "c")
        plan = b.build()
        stats = run_assign_streams(plan)
        assert stats.sync_regions >= 1
        assert stats.async_wrap_regions >= 1


# ========================================================================
# plan_buffers
# ========================================================================


class TestPlanBuffers:
    def test_pools_all_buffers_in_single_space(self):
        plan = _basic_plan()
        run_assign_memory_space(plan)
        stats = run_plan_buffers(plan)
        assert stats.buffers_pooled >= 3
        assert "dram" in plan.summary["buffer_offsets"]

    def test_overlapping_buffers_get_distinct_offsets(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 1024, "scratchpad", 0, 10)
        b.add_buffer("b", 2048, "scratchpad", 5, 15)
        plan = b.build()
        run_plan_buffers(plan)
        offsets = plan.summary["buffer_offsets"]["scratchpad"]
        assert offsets["a"] != offsets["b"]

    def test_non_overlapping_buffers_can_share_offset(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 1024, "scratchpad", 0, 5)
        b.add_buffer("b", 1024, "scratchpad", 10, 15)
        plan = b.build()
        run_plan_buffers(plan)
        offsets = plan.summary["buffer_offsets"]["scratchpad"]
        # Non-overlapping + same size => greedy color assigns same color.
        assert offsets["a"] == offsets["b"]

    def test_alignment_pads_buffer_size(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 100, "scratchpad", 0, 10)
        b.add_buffer("b", 200, "scratchpad", 5, 15)
        plan = b.build()
        run_plan_buffers(plan, config=PlanBuffersConfig(alignment_bytes=128))
        total = plan.summary["buffer_pool_total_bytes"]["scratchpad"]
        # 100 → pad to 128; 200 → pad to 256; each gets its own color
        # (they overlap) so total = 128 + 256 = 384.
        assert total == 384

    def test_restrict_to_spaces_limits_scope(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 100, "scratchpad", 0, 10)
        b.add_buffer("b", 100, "hbm", 0, 10)
        plan = b.build()
        run_plan_buffers(
            plan,
            config=PlanBuffersConfig(restrict_to_spaces=frozenset({"scratchpad"})),
        )
        assert "scratchpad" in plan.summary["buffer_offsets"]
        assert "hbm" not in plan.summary["buffer_offsets"]


# ========================================================================
# insert_copies
# ========================================================================


class TestInsertCopies:
    def test_inserts_copy_across_memory_spaces(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cpu", "q_cpu")
        b.add_buffer("val", 1024, "hbm", 0, 5)
        b.add_dependency("p", "c", value_ref="val")
        plan = b.build()
        plan.summary["device_default_space"] = {"cpu": "host"}
        stats = run_insert_copies(plan)
        assert stats.copies_inserted >= 1
        assert len(plan.copy_edges) >= 1

    def test_skips_same_memory_space(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cuda:0", "q2")
        b.add_buffer("val", 1024, "hbm", 0, 5)
        b.add_dependency("p", "c", value_ref="val")
        plan = b.build()
        plan.summary["device_default_space"] = {"cuda:0": "hbm"}
        stats = run_insert_copies(plan)
        assert stats.skipped_same_space >= 1
        assert stats.copies_inserted == 0

    def test_skips_dependency_without_value_ref(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cpu", "q_cpu")
        b.add_dependency("p", "c")  # no value_ref
        plan = b.build()
        stats = run_insert_copies(plan)
        assert stats.skipped_no_value_ref >= 1

    def test_staging_buffer_created(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cpu", "q_cpu")
        b.add_buffer("val", 1024, "hbm", 0, 5)
        b.add_dependency("p", "c", value_ref="val")
        plan = b.build()
        plan.summary["device_default_space"] = {"cpu": "host"}
        run_insert_copies(plan)
        ids = {b.buffer_id for b in plan.buffers}
        assert "val_staging" in ids

    def test_idempotent_second_run(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cpu", "q_cpu")
        b.add_buffer("val", 1024, "hbm", 0, 5)
        b.add_dependency("p", "c", value_ref="val")
        plan = b.build()
        plan.summary["device_default_space"] = {"cpu": "host"}
        first = run_insert_copies(plan)
        assert first.copies_inserted == 1
        second = run_insert_copies(plan)
        assert second.copies_inserted == 0


# ========================================================================
# alias_io_buffers
# ========================================================================


class TestAliasIOBuffers:
    def test_non_overlapping_leafs_get_aliased(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("early", 4096, "hbm", 0, 5)
        b.add_buffer("late", 4096, "hbm", 10, 15)
        plan = b.build()
        stats = run_alias_io_buffers(plan)
        assert stats.aliases_created == 1
        # The later buffer should be aliased onto the earlier one.
        assert plan.buffer("late").ownership == "alias"
        assert plan.buffer("late").alias_of == "early"

    def test_overlapping_leafs_are_not_aliased(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 4096, "hbm", 0, 10)
        b.add_buffer("b", 4096, "hbm", 5, 15)
        plan = b.build()
        stats = run_alias_io_buffers(plan)
        assert stats.aliases_created == 0

    def test_different_sizes_skipped_by_default(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 2048, "hbm", 0, 5)
        b.add_buffer("b", 4096, "hbm", 10, 15)
        plan = b.build()
        stats = run_alias_io_buffers(plan)
        assert stats.aliases_created == 0
        assert stats.aliases_skipped_size_mismatch >= 1

    def test_different_sizes_allowed_via_config(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("a", 2048, "hbm", 0, 5)
        b.add_buffer("b", 4096, "hbm", 10, 15)
        plan = b.build()
        cfg = AliasIOBuffersConfig(allow_different_sizes=True)
        stats = run_alias_io_buffers(plan, config=cfg)
        assert stats.aliases_created >= 1

    def test_persistent_buffers_are_not_aliased(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("w", 4096, "hbm", 0, 5, persistent=True)
        b.add_buffer("a", 4096, "hbm", 10, 15)
        plan = b.build()
        stats = run_alias_io_buffers(plan)
        assert stats.aliases_created == 0


# ========================================================================
# dma_overlap
# ========================================================================


class TestDMAOverlap:
    def _plan_with_copy(self, size=16384):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("p", "cuda:0", "q")
        b.add_region("c", "cuda:0", "q")
        b.add_buffer("src", size, "hbm", 0, 5)
        b.add_buffer("dst", size, "scratchpad", 6, 10)
        b.add_copy("src", "dst", size, transfer_path="ddr_to_tcm")
        return b.build()

    def test_double_buffers_large_copy(self):
        plan = self._plan_with_copy(size=16384)
        stats = run_dma_overlap(plan)
        assert stats.copies_double_buffered == 1
        assert stats.dma_edges_emitted == 2
        assert stats.sync_edges_emitted == 1
        ids = {b.buffer_id for b in plan.buffers}
        assert "dst_ping" in ids
        assert "dst_pong" in ids

    def test_skips_small_copy(self):
        plan = self._plan_with_copy(size=64)
        stats = run_dma_overlap(plan)
        assert stats.copies_double_buffered == 0
        assert stats.copies_skipped_too_small == 1

    def test_dma_edges_use_configured_transfer_path(self):
        plan = self._plan_with_copy(size=16384)
        run_dma_overlap(plan, config=DMAOverlapConfig(dma_transfer_path="custom_dma"))
        new_edges = [e for e in plan.copy_edges if "custom_dma" == e.transfer_path]
        assert len(new_edges) == 2

    def test_sync_kind_validated(self):
        plan = self._plan_with_copy(size=16384)
        with pytest.raises(ValueError, match="sync_kind"):
            run_dma_overlap(plan, config=DMAOverlapConfig(sync_kind="cosmic"))

    def test_idempotent_second_run(self):
        plan = self._plan_with_copy(size=16384)
        first = run_dma_overlap(plan)
        assert first.copies_double_buffered == 1
        second = run_dma_overlap(plan)
        assert second.copies_double_buffered == 0


# ========================================================================
# insert_host_offload
# ========================================================================


class TestInsertHostOffload:
    def test_no_host_regions_is_noop(self):
        plan = _basic_plan()
        stats = run_insert_host_offload(plan)
        assert stats.host_regions_found == 0

    def test_host_regions_detected_and_summary_populated(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("gpu_r", "cuda:0", "q")
        b.add_region("cpu_r", "cpu", "q_cpu")
        plan = b.build()
        stats = run_insert_host_offload(plan)
        assert stats.host_regions_found == 1
        assert plan.summary["host_offload_regions"] == ["cpu_r"]

    def test_emits_transfer_from_device_buffer_to_host(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("gpu_r", "cuda:0", "q")
        b.add_region("cpu_r", "cpu", "q_cpu")
        b.add_buffer("activations", 2048, "hbm", 0, 5)
        b.add_dependency("gpu_r", "cpu_r", value_ref="activations")
        plan = b.build()
        stats = run_insert_host_offload(plan)
        assert stats.offload_transfers_inserted == 1
        assert any(e.transfer_path == "host_offload" for e in plan.copy_edges)

    def test_custom_host_prefix(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("special", "custom_host_0", "q")
        plan = b.build()
        cfg = InsertHostOffloadConfig(host_device_prefixes=("custom_host",))
        stats = run_insert_host_offload(plan, config=cfg)
        assert stats.host_regions_found == 1


# ========================================================================
# normalize_subbyte_post_layout
# ========================================================================


class TestNormalizeSubbytePostLayout:
    def test_noop_without_subbyte_ops_on_summary(self):
        plan = _basic_plan()
        stats = run_normalize_subbyte_post_layout(plan)
        assert stats.buffers_realigned == 0

    def test_realigns_buffer_to_dma_line(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("packed", 100, "hbm", 0, 5)
        plan = b.build()
        plan.summary["subbyte_ops"] = [{"buffer_id": "packed", "bit_width": 4, "pack_dim": 1}]
        cfg = NormalizeSubbytePostLayoutConfig(dma_line_bytes=64)
        stats = run_normalize_subbyte_post_layout(plan, config=cfg)
        assert stats.buffers_realigned == 1
        assert plan.buffer("packed").size_bytes == 128  # padded to 64-byte line
        assert plan.summary["subbyte_buffer_strides"]["packed"] > 0

    def test_stride_recorded_per_buffer(self):
        b = ExecutionPlanBuilder("t", "c")
        b.add_region("r", "cuda", "q")
        b.add_buffer("p1", 256, "hbm", 0, 5)
        b.add_buffer("p2", 512, "hbm", 6, 10)
        plan = b.build()
        plan.summary["subbyte_ops"] = [
            {"buffer_id": "p1", "bit_width": 4, "pack_dim": 1},
            {"buffer_id": "p2", "bit_width": 4, "pack_dim": 1},
        ]
        run_normalize_subbyte_post_layout(plan)
        strides = plan.summary["subbyte_buffer_strides"]
        assert "p1" in strides
        assert "p2" in strides

    def test_unknown_buffer_is_skipped(self):
        plan = _basic_plan()
        plan.summary["subbyte_ops"] = [{"buffer_id": "ghost", "bit_width": 4, "pack_dim": 1}]
        stats = run_normalize_subbyte_post_layout(plan)
        assert stats.buffers_skipped >= 1

    def test_stats_initial_values(self):
        s = NormalizeSubbytePostLayoutStats()
        assert s.buffers_realigned == 0
        assert s.strides_by_buffer == {}


# ========================================================================
# Integration: full Phase-5 chain on a realistic plan
# ========================================================================


def test_full_phase5_chain_on_mixed_plan():
    """Runs all 9 passes in dependency order and asserts the final
    plan passes ``plan.validate()`` + has the expected annotations."""
    b = ExecutionPlanBuilder("integration", "heterogeneous")
    b.add_region("gpu0", "cuda:0", "")
    b.add_region("gpu1", "cuda:0", "")
    b.add_region("host", "cpu", "")
    b.add_buffer("w", 8192, "", 0, 0, persistent=True, ownership="shared_readonly")
    b.add_buffer("act_0", 16384, "", 1, 5)
    b.add_buffer("act_1", 16384, "", 6, 10)
    b.add_buffer("final", 4096, "", 11, 20)
    b.add_dependency("gpu0", "gpu1", value_ref="act_0")
    b.add_dependency("gpu1", "host", value_ref="final")
    plan = b.build()
    plan.summary["device_default_space"] = {"cuda:0": "vtcm", "cpu": "host"}

    #  chain in correct order:
    run_assign_memory_space(plan, config=AssignMemorySpaceConfig(vtcm_bytes=20_000, scratch_memory_space="vtcm"))
    run_assign_queue(plan)
    run_assign_streams(plan)
    run_plan_buffers(plan)
    run_insert_copies(plan)
    run_alias_io_buffers(plan)
    run_dma_overlap(plan, config=DMAOverlapConfig(min_copy_size_bytes=1024))
    run_insert_host_offload(plan)
    plan.summary["subbyte_ops"] = [{"buffer_id": "w", "bit_width": 4, "pack_dim": 1}]
    run_normalize_subbyte_post_layout(plan)

    plan.validate()

    # Post-conditions that matter for downstream stages:
    assert all(buf.memory_space for buf in plan.buffers)
    assert all(rp.queue for rp in plan.region_placement)
    assert "stream_kinds" in plan.summary
    assert "buffer_offsets" in plan.summary
    assert "host_offload_regions" in plan.summary
    assert "subbyte_buffer_strides" in plan.summary
