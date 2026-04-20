"""Tests for kernel contracts."""

from __future__ import annotations

from compgen.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement


def test_layout_kind_values() -> None:
    assert LayoutKind.ROW_MAJOR.value == "row_major"
    assert LayoutKind.COLUMN_MAJOR.value == "column_major"
    assert LayoutKind.CUSTOM_STRIDES.value == "custom_strides"
    assert LayoutKind.ANY.value == "any"


def test_layout_requirement_defaults() -> None:
    req = LayoutRequirement()
    assert req.kind == LayoutKind.ANY
    assert req.strides is None
    assert req.alignment == 1


def test_cost_estimate_defaults() -> None:
    cost = CostEstimate()
    assert cost.flops == 0
    assert cost.bytes_read == 0
    assert cost.bytes_written == 0
    assert cost.latency_us is None


def test_kernel_contract_construction() -> None:
    layout = LayoutRequirement(kind=LayoutKind.ROW_MAJOR, alignment=64)
    cost = CostEstimate(flops=1024, bytes_read=512, bytes_written=256)
    contract = KernelContract(
        op_name="linalg.matmul",
        input_layouts=[layout, layout],
        output_layouts=[layout],
        cost=cost,
        fusable=False,
    )
    assert contract.op_name == "linalg.matmul"
    assert len(contract.input_layouts) == 2
    assert contract.cost.flops == 1024
    assert contract.fusable is False
    assert "float32" in contract.supported_dtypes


def test_extract_contracts() -> None:
    """extract_contracts should walk an xDSL module and emit KernelContracts."""
    from compgen.ir.payload.contracts import extract_contracts
    from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import MatmulOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region

    f32 = Float32Type()
    lhs_type = TensorType(f32, [64, 128])
    rhs_type = TensorType(f32, [128, 256])
    out_type = TensorType(f32, [64, 256])

    # Build a minimal function with a matmul
    block = Block(arg_types=[lhs_type, rhs_type])
    empty = EmptyOp([], out_type)
    matmul = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[empty.results[0]],
        res=[out_type],
    )
    ret = ReturnOp(matmul)
    block.add_ops([empty, matmul, ret])
    func_op = FuncOp("main", ([lhs_type, rhs_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    contracts = extract_contracts(module)
    # Should have contracts for EmptyOp and MatmulOp (skip FuncOp/ReturnOp/ModuleOp)
    assert len(contracts) >= 1
    # Find the matmul contract
    matmul_contracts = [c for c in contracts if "matmul" in c.op_name]
    assert len(matmul_contracts) == 1
    mc = matmul_contracts[0]
    assert mc.fusable is False  # matmul is a kernel boundary
    assert mc.cost.flops > 0
    assert mc.cost.bytes_read > 0


def test_kernel_contract_yaml_serialization() -> None:
    """KernelContract should be serializable to YAML for LLM context."""
    import yaml

    cost = CostEstimate(flops=2048, bytes_read=1024, bytes_written=512)
    contract = KernelContract(
        op_name="linalg.matmul",
        input_layouts=[
            LayoutRequirement(kind=LayoutKind.ROW_MAJOR, alignment=64),
            LayoutRequirement(kind=LayoutKind.ROW_MAJOR, alignment=64),
        ],
        output_layouts=[
            LayoutRequirement(kind=LayoutKind.ROW_MAJOR, alignment=64),
        ],
        cost=cost,
        fusable=False,
    )
    # Build a plain-dict representation suitable for safe YAML round-tripping
    # (enums and sets don't survive yaml.safe_load).
    data = {
        "op_name": contract.op_name,
        "fusable": contract.fusable,
        "cost": {
            "flops": contract.cost.flops,
            "bytes_read": contract.cost.bytes_read,
            "bytes_written": contract.cost.bytes_written,
            "latency_us": contract.cost.latency_us,
        },
        "input_layouts": [{"kind": l.kind.value, "alignment": l.alignment} for l in contract.input_layouts],
        "output_layouts": [{"kind": l.kind.value, "alignment": l.alignment} for l in contract.output_layouts],
        "supported_dtypes": sorted(contract.supported_dtypes),
    }
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=True)
    assert "linalg.matmul" in yaml_text
    assert "flops: 2048" in yaml_text
    # Deserialize back
    recovered = yaml.safe_load(yaml_text)
    assert recovered["op_name"] == "linalg.matmul"
    assert recovered["cost"]["flops"] == 2048
    assert recovered["fusable"] is False
