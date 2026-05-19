"""Render-time tests for :mod:`xpu_rt.kb_gemmini.multiop_harness`.

These tests do NOT compile or run on Spike — they only verify the
emitted C source has the right structure (externs, allocations,
sequential calls, scalar-reference chain, counter wiring). A full
end-to-end Spike compile-and-run smoke test lives outside the pytest
suite because it requires the riscv-tools toolchain and is gated
behind a separate marker.
"""

from __future__ import annotations

from pathlib import Path

from xpu_rt.ir.payload.contract_graph import (
    ContractEdge,
    ContractNode,
    build_contract_graph_from_nodes,
)
from xpu_rt.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement
from xpu_rt.kb_gemmini.multiop_harness import (
    KernelBinding,
    PipelineHarnessSpec,
    render_pipeline_driver_c,
    stage_pipeline_harness,
)


def _v1_matmul_contract(region_id: str, a: tuple[int, int], b: tuple[int, int], out: tuple[int, int]) -> KernelContract:
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
            "region_id": region_id,
            "dispatch_id": region_id,
        },
    )


def _v1_silu_contract(region_id: str, shape: tuple[int, int]) -> KernelContract:
    return KernelContract(
        op_name="silu",
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={"i32"},
        cost=CostEstimate(flops=shape[0] * shape[1]),
        fusable=True,
        metadata={
            "input_shapes": [shape],
            "output_shapes": [shape],
            "region_id": region_id,
            "dispatch_id": region_id,
        },
    )


def _two_matmul_chain_spec() -> PipelineHarnessSpec:
    """The smoke-test chain referenced by the plan:
    M=64, K1=720, N1=K2=32, N2=320 — both shapes are 7/14 winners."""
    m1 = _v1_matmul_contract("m1", (64, 720), (720, 32), (64, 32))
    m2 = _v1_matmul_contract("m2", (64, 32), (32, 320), (64, 320))
    n1 = ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="m1")
    n2 = ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="m2")
    edge = ContractEdge(
        producer_id="n_m1",
        consumer_id="n_m2",
        operand_index=0,
        tensor_shape=(64, 32),
        dtype="i32",
        bytes_per_element=4,
    )
    graph = build_contract_graph_from_nodes([n1, n2], [edge])
    bindings = (
        KernelBinding(op_id="n_m1", function_name="launch_m1", kernel_source_path=Path("/tmp/__missing")),
        KernelBinding(op_id="n_m2", function_name="launch_m2", kernel_source_path=Path("/tmp/__missing")),
    )
    return PipelineHarnessSpec(graph=graph, bindings=bindings, external_input_shapes={})


def test_render_two_matmul_chain_emits_expected_skeleton() -> None:
    spec = _two_matmul_chain_spec()
    src = render_pipeline_driver_c(spec)
    # The driver must declare both kernels as externs.
    assert "extern void launch_m1(" in src
    assert "extern void launch_m2(" in src
    # The chain ordering must put m1's call before m2's.
    m1_pos = src.index("launch_m1((void *)")
    m2_pos = src.index("launch_m2((void *)")
    assert m1_pos < m2_pos
    # m2's first input must be m1's output buffer (intra-graph edge).
    m2_call_line = src[m2_pos:].split(";", 1)[0]
    assert "buf_out_n_m1_0" in m2_call_line
    # The whole-block scalar-reference chain must compute the matmul
    # twice — once for each ref buffer. (no silu, just matmul refs.)
    assert src.count("matmul reference") == 2
    # Gemmini-style counter wiring must be present.
    assert "MAIN_LD_ST_EX_CYCLES" in src
    assert "counter_snapshot_reset();" in src
    # Block-end correctness print uses the tail node's buffers.
    assert "buf_ref_n_m2_0" in src
    assert "buf_out_n_m2_0" in src


def test_render_mlp_chain_includes_silu_reference() -> None:
    """matmul → silu → matmul. The whole-block scalar reference must
    include one silu-style activation snippet — the harness exists
    precisely so this end-to-end diff catches wiring errors that
    per-kernel correctness checks would miss."""
    m1 = _v1_matmul_contract("m1", (64, 720), (720, 1440), (64, 1440))
    silu = _v1_silu_contract("s1", (64, 1440))
    m2 = _v1_matmul_contract("m2", (64, 1440), (1440, 720), (64, 720))
    n1 = ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="m1")
    ns = ContractNode(op_id="n_s1", contract=silu, op_name="silu", region_id="s1")
    n2 = ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="m2")
    edges = [
        ContractEdge(producer_id="n_m1", consumer_id="n_s1", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i32", bytes_per_element=4),
        ContractEdge(producer_id="n_s1", consumer_id="n_m2", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i32", bytes_per_element=4),
    ]
    graph = build_contract_graph_from_nodes([n1, ns, n2], edges)
    bindings = (
        KernelBinding(op_id="n_m1", function_name="launch_m1", kernel_source_path=Path("/tmp/__missing")),
        KernelBinding(op_id="n_s1", function_name="launch_silu", kernel_source_path=Path("/tmp/__missing")),
        KernelBinding(op_id="n_m2", function_name="launch_m2", kernel_source_path=Path("/tmp/__missing")),
    )
    spec = PipelineHarnessSpec(graph=graph, bindings=bindings, external_input_shapes={})

    src = render_pipeline_driver_c(spec)
    assert "activation reference" in src  # silu's snippet
    assert src.count("matmul reference") == 2
    # Three externs, three device calls.
    assert "extern void launch_m1(" in src
    assert "extern void launch_silu(" in src
    assert "extern void launch_m2(" in src


def test_stage_pipeline_harness_writes_driver_and_kernels(tmp_path: Path) -> None:
    """The stager must drop one .c per kernel + a driver.c. We feed
    fake kernel sources so we don't need the toolchain in unit tests."""
    spec = _two_matmul_chain_spec()
    # Materialise the fake kernel source files so the stager can copy them.
    k1 = tmp_path / "k1.c"
    k1.write_text("void launch_m1(void *a, void *b, void *c, long M, long K, long N){}\n")
    k2 = tmp_path / "k2.c"
    k2.write_text("void launch_m2(void *a, void *b, void *c, long M, long K, long N){}\n")
    bindings = (
        KernelBinding(op_id="n_m1", function_name="launch_m1", kernel_source_path=k1),
        KernelBinding(op_id="n_m2", function_name="launch_m2", kernel_source_path=k2),
    )
    spec2 = PipelineHarnessSpec(graph=spec.graph, bindings=bindings, external_input_shapes={})
    out_dir = tmp_path / "staged"
    p = stage_pipeline_harness(out_dir, spec2)
    assert p == out_dir
    assert (out_dir / "launch_m1.c").is_file()
    assert (out_dir / "launch_m2.c").is_file()
    assert (out_dir / "driver.c").is_file()
    driver_src = (out_dir / "driver.c").read_text()
    assert "extern void launch_m1" in driver_src
    assert "extern void launch_m2" in driver_src
