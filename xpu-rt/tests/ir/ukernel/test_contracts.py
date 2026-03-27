"""Tests for ukernel interface contracts."""

from __future__ import annotations

import pytest
from compgen.ir.ukernel.contracts import UkernelContract


def test_ukernel_contract() -> None:
    c = UkernelContract(kernel_name="my_matmul", supported_dtypes={"float32", "float16"}, perf_bound_us=100.0)
    assert "float16" in c.supported_dtypes
    assert c.perf_bound_us == 100.0


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_contract_matching() -> None:
    """A call should match its contract."""
