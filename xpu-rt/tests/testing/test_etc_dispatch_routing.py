"""Phase-7 routing tests (CPU-only).

Pin the contract changes #032 introduced:

- ``compgen.api._ETC_DISPATCH_READY`` is True (Phase 7 routed).
- ``compgen.testing.workloads.WORKLOAD_FACTORIES`` registers
  ``"diamond_dag"``.
- The diamond_dag workload factory returns a structurally valid
  :class:`Workload`: 4 CallDeviceOp-equivalent tasks, 4 event
  tensors, real CUDA-C++ device function bodies (no stubs).
- :class:`compgen.testing.etc_dispatch.EtcDispatchError` exists and
  is a RuntimeError subclass for the harness to catch.

GPU end-to-end (NVRTC compile + cooperative launch + correctness +
1.2× speedup gate) lives behind ``requires_gpu`` and runs on bwell
via the conformance harness. This module is the contract gate that
runs in every CI loop on the dev box.
"""

from __future__ import annotations

import pytest


class TestEtcDispatchReadiness:
    def test_etc_dispatch_ready_flag_is_true(self) -> None:
        from compgen import api

        assert api._ETC_DISPATCH_READY is True

    def test_workload_factory_registry_has_diamond_dag(self) -> None:
        from compgen.testing.workloads import WORKLOAD_FACTORIES

        assert "diamond_dag" in WORKLOAD_FACTORIES
        factory = WORKLOAD_FACTORIES["diamond_dag"]
        assert callable(factory)

    def test_etc_dispatch_error_is_runtime_subclass(self) -> None:
        from compgen.testing.etc_dispatch import EtcDispatchError

        assert issubclass(EtcDispatchError, RuntimeError)


class TestDiamondDagFactory:
    def test_build_returns_structured_workload(self) -> None:
        from compgen.testing.workloads.diamond_dag import Workload, build

        wl = build(dtype="fp32", num_gpus=1)
        assert isinstance(wl, Workload)
        assert wl.model is not None
        assert len(wl.sample_inputs) == 1
        # Shape comes from the workload module's _BATCH × _IN_DIM
        # constants — keep the test loose-but-anchored: positive 2D
        # tensor whose row count matches the model's batch arg.
        assert wl.sample_inputs[0].dim() == 2
        assert wl.sample_inputs[0].shape[0] >= 1
        assert wl.sample_inputs[0].shape[1] == wl.model.a.in_features

    def test_megakernel_graph_has_4_calls_4_events_with_tile_tasks(self) -> None:
        """The diamond now uses 4 :class:`DeviceCall`s with
        ``task_shape=(NUM_TILES,)``. Each call expands to NUM_TILES
        tile-tasks, so the static scheduler enumerates 4 × NUM_TILES
        flat tasks. Four event tensors of shape ``(NUM_TILES,)`` —
        one cell per tile — give each tile its own dedicated edge."""
        from compgen.testing.workloads.diamond_dag import (
            _NUM_TILES,
            build,
        )

        wl = build(dtype="fp32", num_gpus=1)
        graph = wl.build_megakernel_graph(wl.model, wl.sample_inputs)
        names = [c.name for c in graph.calls]
        assert names == ["linear_a", "linear_b", "add_op", "relu_op"]
        for call in graph.calls:
            assert call.task_shape == (_NUM_TILES,), (
                f"{call.name} should have task_shape=(_NUM_TILES,) for tile-level scheduling"
            )
        assert sorted(graph.event_tensors) == ["ev_a", "ev_add", "ev_b", "ev_done"]
        for ev in graph.event_tensors.values():
            assert ev.shape == (_NUM_TILES,), (
                "event tensors must be shape (NUM_TILES,) so each tile-task has its own cell"
            )

    def test_device_function_bodies_have_real_cuda(self) -> None:
        """No placeholder bodies — every device function must contain
        the actual GEMM/elementwise CUDA C++ that will run on silicon."""
        from compgen.testing.workloads.diamond_dag import build

        wl = build(dtype="fp32", num_gpus=1)
        bodies = wl.device_function_sources
        assert set(bodies) == {"linear_a", "linear_b", "add_op", "relu_op"}
        # GEMM bodies: shared-memory tiled with k-tile inner reduction.
        for name in ("linear_a", "linear_b"):
            body = bodies[name].body
            assert "__shared__" in body, f"{name} missing shared-mem tile"
            assert "A_tile" in body and "W_tile" in body, f"{name} missing tile arrays"
            assert "k_tile += TK" in body, f"{name} missing K-tile loop"
            # Accumulator update can be `acc += a*b` or `acc = fmaf(a,b,acc)`.
            assert "acc +=" in body or "fmaf(" in body, f"{name} missing accumulator update"
        # Add: elementwise sum of ya + yb into yadd.
        assert "ya[idx] + yb[idx]" in bodies["add_op"].body
        # Relu: max(0, x).
        assert "v > 0.0f ? v : 0.0f" in bodies["relu_op"].body

    def test_user_buffer_layout_matches_bodies(self) -> None:
        """The buffer-layout names in Workload.user_buffer_layout
        must match the buffers[N] indexing in the bodies."""
        from compgen.testing.workloads.diamond_dag import build

        wl = build(dtype="fp32", num_gpus=1)
        # Layout order corresponds to buffers[0], buffers[1], ...
        assert wl.user_buffer_layout == ("x", "wa", "wb", "ya", "yb", "yadd", "yout")
        # Spot-check: linear_a reads buffers[0] (x), buffers[1] (wa)
        # and writes buffers[3] (ya).
        body_a = wl.device_function_sources["linear_a"].body
        assert "buffers[0]" in body_a  # x
        assert "buffers[1]" in body_a  # wa
        assert "buffers[3]" in body_a  # ya
        # relu reads buffers[5] (yadd), writes buffers[6] (yout).
        body_relu = wl.device_function_sources["relu_op"].body
        assert "buffers[5]" in body_relu
        assert "buffers[6]" in body_relu

    def test_multi_gpu_rejected(self) -> None:
        from compgen.testing.workloads.diamond_dag import build

        with pytest.raises(ValueError, match="single-GPU"):
            build(dtype="fp32", num_gpus=2)


class TestDecoderLayerFactory:
    """Pin the decoder_layer (v1: FFN) workload's structural shape.

    GPU correctness + perf gate runs in the conformance harness on
    bwell; these CPU-only tests guard the ABI.
    """

    def test_build_returns_workload(self) -> None:
        from compgen.testing.workloads.decoder_layer import Workload, build

        wl = build(dtype="fp32", num_gpus=1)
        assert isinstance(wl, Workload)
        assert hasattr(wl.model, "up")
        assert hasattr(wl.model, "down")
        assert wl.user_buffer_layout == (
            "x",
            "w_up",
            "w_down",
            "y_up",
            "y_relu",
            "y_out",
        )

    def test_megakernel_graph_topology(self) -> None:
        from compgen.testing.workloads.decoder_layer import (
            _NUM_DOWN_TILES,
            _NUM_UP_TILES,
            _UP_TILES_PER_ROW,
            build,
        )

        wl = build(dtype="fp32", num_gpus=1)
        graph = wl.build_megakernel_graph(wl.model, wl.sample_inputs)
        names = [c.name for c in graph.calls]
        assert names == ["up_proj", "relu_op", "down_proj"]

        up, relu, down = graph.calls
        assert up.task_shape == (_NUM_UP_TILES,)
        assert relu.task_shape == (_NUM_UP_TILES,)
        assert down.task_shape == (_NUM_DOWN_TILES,)

        # down_proj waits on _UP_TILES_PER_ROW relu cells per task —
        # the K-axis dependency that diamond_dag doesn't exercise.
        assert len(down.in_edges) == _UP_TILES_PER_ROW
        assert all(e.event_name == "ev_relu" for e in down.in_edges)

    def test_factory_registered(self) -> None:
        from compgen.testing.workloads import WORKLOAD_FACTORIES

        assert "decoder_layer" in WORKLOAD_FACTORIES

    def test_device_function_bodies_are_real(self) -> None:
        """All three bodies must have shared-mem tile loops + fmaf
        accumulators — no stubs."""
        from compgen.testing.workloads.decoder_layer import build

        wl = build(dtype="fp32", num_gpus=1)
        bodies = wl.device_function_sources
        assert set(bodies) == {"up_proj", "relu_op", "down_proj"}
        for name in ("up_proj", "down_proj"):
            body = bodies[name].body
            assert "__shared__" in body, f"{name} missing shared-mem tile"
            assert "fmaf(" in body, f"{name} missing fmaf accumulator"
        # relu is elementwise and uses the in_bounds gate.
        assert "v > 0.0f ? v : 0.0f" in bodies["relu_op"].body

    def test_multi_gpu_rejected(self) -> None:
        from compgen.testing.workloads.decoder_layer import build

        with pytest.raises(ValueError, match="single-GPU"):
            build(dtype="fp32", num_gpus=2)


class TestGemmReduceScatterFactory:
    """Pin the gemm_reduce_scatter (Phase 4b v1) workload's structural
    shape. GPU correctness + perf live in the conformance harness on
    bwell with --num-gpus=2; these CPU-only tests guard the ABI."""

    def test_build_returns_workload(self) -> None:
        from compgen.testing.workloads.gemm_reduce_scatter import Workload, build

        wl = build(dtype="fp32", num_gpus=2)
        assert isinstance(wl, Workload)
        assert wl.num_ranks == 2
        assert wl.user_buffer_layout == ("x_shard", "w_shard", "y_partial")
        # multi_rank_collective tells the harness which NCCL primitive
        # to drive between the per-rank megakernel launches.
        assert wl.multi_rank_collective == "allreduce_sum"

    def test_megakernel_graph_is_single_op(self) -> None:
        from compgen.testing.workloads.gemm_reduce_scatter import (
            _NUM_TILES_PER_RANK,
            build,
        )

        wl = build(dtype="fp32", num_gpus=2)
        graph = wl.build_megakernel_graph(wl.model, wl.sample_inputs)
        assert [c.name for c in graph.calls] == ["gemm_local"]
        assert graph.calls[0].task_shape == (_NUM_TILES_PER_RANK,)

    def test_factory_registered(self) -> None:
        from compgen.testing.workloads import WORKLOAD_FACTORIES

        assert "gemm_rs" in WORKLOAD_FACTORIES

    def test_only_two_gpus_allowed_in_v1(self) -> None:
        from compgen.testing.workloads.gemm_reduce_scatter import build

        with pytest.raises(ValueError, match="num_gpus=2"):
            build(dtype="fp32", num_gpus=1)
        with pytest.raises(ValueError, match="num_gpus=2"):
            build(dtype="fp32", num_gpus=4)

    def test_unsupported_dtype_raises(self) -> None:
        from compgen.testing.workloads.gemm_reduce_scatter import build

        with pytest.raises(NotImplementedError, match="tensor-core"):
            build(dtype="bf16", num_gpus=2)


class TestPerWorkloadGateOverride:
    """Pin :data:`WORKLOAD_GATES` + :func:`gate_for` semantics."""

    def test_default_gate_keeps_perf_floor(self) -> None:
        from compgen.testing.etc_conformance import gate_for

        gate = gate_for("decoder_layer")
        assert gate.min_speedup_vs_eager == 1.2

    def test_diamond_dag_gate_is_stress_test_framed(self) -> None:
        """diamond_dag is documented as an internal stress test —
        its gate has perf floor 0.0 with a recorded rationale."""
        from compgen.testing.etc_conformance import gate_for

        gate = gate_for("diamond_dag")
        assert gate.min_speedup_vs_eager == 0.0
        assert gate.rationale is not None
        assert "stress test" in gate.rationale.lower()
        # Correctness + launch + atomics gates remain at the
        # default — the override only relaxes the perf floor.
        assert gate.correctness_atol == 1e-3
        assert gate.correctness_rtol == 1e-3
        assert gate.require_atomics is True
        assert gate.max_launches_static == 1

    def test_gemm_rs_gate_is_v1_framed(self) -> None:
        """gemm_rs v1 = per-rank cooperative launches + host-side
        AllReduce (2 launches total). Override allows that, with a
        documented rationale that v2 collapses to 1 launch."""
        from compgen.testing.etc_conformance import gate_for

        gate = gate_for("gemm_rs")
        assert gate.max_launches_static == 2
        assert gate.min_speedup_vs_eager == 0.0
        assert gate.rationale is not None
        assert "v1" in gate.rationale.lower()

    def test_atomic_gate_passes_with_only_notify(self) -> None:
        """Single-producer workloads (gemm_rs v1) have notify_atomics
        but zero wait_sites — that's correct topology, not a stub.
        Gate must accept any-direction atomics rather than requiring
        both."""
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            PassGate,
            _evaluate_gate,
        )

        gate = PassGate(min_speedup_vs_eager=0.0, max_launches_static=2)
        correctness = {
            "max_abs_err": 1e-6,
            "max_rel_err": 0.001,
            "num_failing_elements": 0.0,
            "num_inputs": 16.0,
        }
        timing = {"speedup_vs_eager": 0.1}
        launch_profile = {
            "num_launches": 2,
            "notify_atomics": 32,
            "wait_sites": 0,
        }
        errors: list[str] = []
        ok = _evaluate_gate(
            ConformanceWorkload.GEMM_REDUCE_SCATTER,
            correctness,
            timing,
            launch_profile,
            gate,
            errors,
        )
        assert ok is True, errors

    def test_atomic_gate_fails_when_both_zero(self) -> None:
        """Conversely, zero-atomic + zero-wait IS a real bypass — must
        fail the gate. Pin the boundary."""
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            PassGate,
            _evaluate_gate,
        )

        gate = PassGate(min_speedup_vs_eager=0.0)
        correctness = {
            "max_abs_err": 1e-6,
            "max_rel_err": 0.001,
            "num_failing_elements": 0.0,
            "num_inputs": 16.0,
        }
        timing = {"speedup_vs_eager": 0.1}
        launch_profile = {
            "num_launches": 1,
            "notify_atomics": 0,
            "wait_sites": 0,
        }
        errors: list[str] = []
        ok = _evaluate_gate(
            ConformanceWorkload.DIAMOND_DAG,
            correctness,
            timing,
            launch_profile,
            gate,
            errors,
        )
        assert ok is False
        assert any("lacks ETC primitives" in e for e in errors)

    def test_allclose_count_short_circuits_strict_max_rel(self) -> None:
        """Per #035: max_rel can blow up on tiny outputs (denominator
        ~ULP) even when |a-b| is well below atol. The gate should
        defer to ``num_failing_elements`` (allclose semantics) when
        the harness reports it."""
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            PassGate,
            _evaluate_gate,
        )

        gate = PassGate(
            correctness_atol=1e-3,
            correctness_rtol=1e-3,
            min_speedup_vs_eager=1.2,
        )
        # Diagnostic shows large max_rel (3.5%), but allclose says
        # zero elements failed → gate passes correctness.
        correctness = {
            "max_abs_err": 1e-6,
            "max_rel_err": 0.035,
            "num_failing_elements": 0.0,
            "num_inputs": 16.0,
        }
        timing = {"speedup_vs_eager": 1.5, "etc_us": 50.0, "eager_us": 75.0}
        launch_profile = {"num_launches": 1, "notify_atomics": 4, "wait_sites": 3}
        errors: list[str] = []
        ok = _evaluate_gate(ConformanceWorkload.DIAMOND_DAG, correctness, timing, launch_profile, gate, errors)
        assert ok is True, errors

    def test_allclose_count_fails_when_elements_violate(self) -> None:
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            PassGate,
            _evaluate_gate,
        )

        gate = PassGate()
        correctness = {
            "max_abs_err": 5e-3,
            "max_rel_err": 0.1,
            "num_failing_elements": 17.0,
            "num_inputs": 16.0,
        }
        timing = {"speedup_vs_eager": 1.5}
        launch_profile = {"num_launches": 1, "notify_atomics": 4, "wait_sites": 3}
        errors: list[str] = []
        ok = _evaluate_gate(ConformanceWorkload.DECODER_LAYER, correctness, timing, launch_profile, gate, errors)
        assert ok is False
        assert any("17 elements failed" in e for e in errors)


class TestConformanceHarnessRouting:
    """The harness's `_check_etc_routing_ready` should now return True
    + the missing-workload short-circuit fires for unwired workloads."""

    def test_routing_ready_returns_true(self) -> None:
        from compgen.testing.etc_conformance import _check_etc_routing_ready

        errors: list[str] = []
        assert _check_etc_routing_ready(errors) is True
        assert errors == []

    def test_unwired_workload_short_circuits_with_typed_error(self, tmp_path) -> None:
        """A workload not in WORKLOAD_FACTORIES (e.g. moe_fwd, which
        depends on Phase 1's TriggerOp wiring) reports cleanly via the
        errors list rather than crashing."""
        from compgen.testing.etc_conformance import (
            ConformanceWorkload,
            _compile_and_evaluate,
        )

        errors: list[str] = []
        result = _compile_and_evaluate(
            workload=ConformanceWorkload.MOE_FWD,
            model=None,
            sample_inputs=(),
            dtype="bf16",
            device_index=0,
            num_correctness_inputs=1,
            num_benchmark_iters=1,
            output_path=tmp_path,
            errors=errors,
        )
        assert result == ({}, {}, {}, None)
        assert any("not yet implemented" in e for e in errors)
