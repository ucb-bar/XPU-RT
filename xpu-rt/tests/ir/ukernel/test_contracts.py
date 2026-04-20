"""Tests for ukernel interface contracts."""

from __future__ import annotations

from compgen.ir.ukernel.contracts import UkernelContract


def test_ukernel_contract() -> None:
    c = UkernelContract(kernel_name="my_matmul", supported_dtypes={"float32", "float16"}, perf_bound_us=100.0)
    assert "float16" in c.supported_dtypes
    assert c.perf_bound_us == 100.0


def test_contract_matching() -> None:
    """A call should match its contract."""
    from compgen.ir.ukernel.contracts import check_contract
    from compgen.ir.ukernel.ops import UkernelCallOp

    contract = UkernelContract(
        kernel_name="my_matmul",
        input_layouts=["row_major", "col_major"],
        output_layouts=["row_major"],
        supported_dtypes={"float32", "float16"},
        max_workspace_bytes=4096,
    )
    call = UkernelCallOp(
        kernel_name="my_matmul",
        operands=["a", "b"],
        results=["c"],
        workspace_bytes=1024,
        metadata={"dtype": "float32"},
    )
    assert check_contract(call, contract) is True

    # Wrong name
    bad_name = UkernelCallOp(kernel_name="other_kernel", operands=["a", "b"], results=["c"])
    assert check_contract(bad_name, contract) is False

    # Too many operands
    bad_ops = UkernelCallOp(kernel_name="my_matmul", operands=["a", "b", "extra"], results=["c"])
    assert check_contract(bad_ops, contract) is False

    # Workspace exceeds limit
    bad_ws = UkernelCallOp(kernel_name="my_matmul", operands=["a", "b"], results=["c"], workspace_bytes=8192)
    assert check_contract(bad_ws, contract) is False

    # Unsupported dtype
    bad_dtype = UkernelCallOp(
        kernel_name="my_matmul",
        operands=["a", "b"],
        results=["c"],
        metadata={"dtype": "int8"},
    )
    assert check_contract(bad_dtype, contract) is False
