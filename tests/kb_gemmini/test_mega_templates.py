"""Render-time tests for :mod:`xpu_rt.kb_gemmini.mega_templates`.

Like the vanilla-chain harness tests, these do not compile or run on
Spike — they verify the structure of the emitted ``init.c`` and
``driver.c`` so we can catch template regressions before any compile
attempt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xpu_rt.ir.payload.contract_graph import (
    ContractEdge,
    ContractNode,
    build_contract_graph_from_nodes,
)
from xpu_rt.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement
from xpu_rt.kb_gemmini.mega_templates import (
    FusedKernelArtifacts,
    render_fused_artifacts,
    render_fused_driver_c,
    render_fused_init_c,
    stage_mega_contract_dir,
)
from xpu_rt.kernels.contract_v3 import Granularity, HardwareEnvelope
from xpu_rt.kernels.fusion_planner import FusionCluster
from xpu_rt.kernels.mega_contract_emitter import emit_mega_contract


def _gemmini_env() -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name="gemmini_mx",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
    )


def _matmul(region: str, a: tuple[int, int], b: tuple[int, int], out: tuple[int, int]) -> KernelContract:
    return KernelContract(
        op_name="matmul",
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR), LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"},
        cost=CostEstimate(flops=out[0] * out[1] * a[1] * 2),
        fusable=False,
        metadata={
            "input_shapes": [a, b],
            "output_shapes": [out],
            "region_id": region,
            "dispatch_id": region,
        },
    )


def _silu(region: str, shape: tuple[int, int]) -> KernelContract:
    return KernelContract(
        op_name="silu",
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"},
        cost=CostEstimate(flops=shape[0] * shape[1]),
        fusable=True,
        metadata={
            "input_shapes": [shape],
            "output_shapes": [shape],
            "region_id": region,
            "dispatch_id": region,
        },
    )


def _build_mega_for_mlp_chain():
    env = _gemmini_env()
    m1 = _matmul("m1", (64, 720), (720, 1440), (64, 1440))
    silu = _silu("s1", (64, 1440))
    m2 = _matmul("m2", (64, 1440), (1440, 720), (64, 720))
    nodes = [
        ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="m1"),
        ContractNode(op_id="n_s1", contract=silu, op_name="silu", region_id="s1"),
        ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="m2"),
    ]
    edges = [
        ContractEdge(producer_id="n_m1", consumer_id="n_s1", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
        ContractEdge(producer_id="n_s1", consumer_id="n_m2", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)
    cluster = FusionCluster(
        cluster_id="cluster_mlp",
        member_op_ids=("n_m1", "n_s1", "n_m2"),
        rationale="MLP chain test",
        estimated_speedup=2.0,
    )
    result = emit_mega_contract(cluster, graph, env)
    return result.contract


def test_render_fused_init_c_preserves_signature() -> None:
    mega = _build_mega_for_mlp_chain()
    src = render_fused_init_c(mega)
    # The vanilla KB launch_gpu_implementation signature MUST be
    # preserved — the driver's extern depends on it.
    assert "void launch_gpu_implementation(void *output," in src
    assert "void *input_A," in src
    assert "void *input_B," in src
    assert "int64_t M, int64_t K, int64_t N" in src
    # The starter doc-comment must mention the chain.
    assert "matmul -> silu -> matmul" in src


def test_render_fused_driver_emits_single_launch_call_and_chain_ref() -> None:
    mega = _build_mega_for_mlp_chain()
    drv = render_fused_driver_c(mega)
    # MEGA harness: exactly one launch_gpu_implementation call (the
    # whole point — fused single kernel). Count is 2: one extern
    # declaration + one call site.
    assert drv.count("launch_gpu_implementation(") == 2
    # The extern declaration must appear at the top, before main().
    extern_pos = drv.index("extern void launch_gpu_implementation(")
    main_pos = drv.index("int main(")
    assert extern_pos < main_pos
    # And the call site appears inside main, exactly once.
    body_after_main = drv[main_pos:]
    assert body_after_main.count("launch_gpu_implementation(") == 1
    # The chain reference must produce 2 matmul snippets + 1 activation.
    assert drv.count("matmul reference") == 2
    # The activation reference: relu-style i32 clamp.
    assert "v > 0 ? v : 0" in drv
    # Counter wiring matches the vanilla path for fair comparison.
    assert "MAIN_LD_ST_EX_CYCLES" in drv
    assert "counter_snapshot_reset();" in drv
    # Output protocol matches the single-op harness.
    assert 'printf("mismatches=%d/%d' in drv
    assert 'printf("cycles=%lld' in drv
    # MEGA-specific marker for the parser.
    assert 'printf("mega=1 nodes=' in drv


def test_render_fused_artifacts_returns_both() -> None:
    mega = _build_mega_for_mlp_chain()
    art = render_fused_artifacts(mega)
    assert isinstance(art, FusedKernelArtifacts)
    assert "launch_gpu_implementation" in art.init_c
    assert "launch_gpu_implementation" in art.driver_c


def test_stage_mega_contract_dir_writes_both_files(tmp_path: Path) -> None:
    mega = _build_mega_for_mlp_chain()
    out_dir = tmp_path / "mega_stage"
    p = stage_mega_contract_dir(out_dir, mega)
    assert p == out_dir
    init_c = (out_dir / "init.cu").read_text()
    driver_c = (out_dir / "driver.cpp").read_text()
    assert "launch_gpu_implementation" in init_c
    assert "launch_gpu_implementation" in driver_c


def test_render_rejects_non_mega_contract() -> None:
    """Single-op contracts must take the single-op path, not the
    MEGA renderer."""
    # Quick way to construct a NORMAL v3 contract for the negative test.
    from xpu_rt.ir.payload.contract_graph import ContractNode, build_contract_graph_from_nodes

    c = _silu("solo", (16, 16))
    n = ContractNode(op_id="solo", contract=c, op_name="silu", region_id="solo")
    graph = build_contract_graph_from_nodes([n], [])
    # Build an isolated NORMAL v3 by lifting via the planner helper.
    from xpu_rt.kernels.fusion_planner import lift_v1_to_v3

    v3 = lift_v1_to_v3(c, _gemmini_env(), op_id_hint="solo")
    assert v3.granularity is not Granularity.MEGA
    with pytest.raises(ValueError, match="MEGA"):
        render_fused_init_c(v3)
    with pytest.raises(ValueError, match="MEGA"):
        render_fused_driver_c(v3)
