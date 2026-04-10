"""Tests for LLM-guided kernel strategy selection (Unit 6)."""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock

from compgen.kernels.selector import KernelSelector, KernelStrategy


def _make_mock_llm(strategy: str = "autocomp", reason: str = "test"):
    """Create a mock LLM client that returns a fixed strategy."""
    mock = MagicMock()
    response = MagicMock()
    response.raw_text = json.dumps({"strategy": strategy, "reason": reason})
    mock.generate_structured.return_value = response
    mock.generate.return_value = response
    return mock


def _make_spec(op_name: str = "linalg.matmul", flops: int = 500):
    """Create a minimal KernelSpec for testing."""
    from compgen.kernels.contracts import KernelSpec
    from unittest.mock import MagicMock
    contract = MagicMock()
    contract.op_name = op_name
    contract.cost.flops = flops
    contract.cost.bytes_read = 0
    contract.cost.bytes_written = 0
    contract.supported_dtypes = ("float32",)
    return KernelSpec(contract=contract, input_shapes=[(1, 64), (64, 64)])


def _make_target():
    from unittest.mock import MagicMock
    target = MagicMock()
    target.name = "test_gpu"
    device = MagicMock()
    device.device_type = "gpu"
    device.compute_units = []
    device.supported_ops = []
    target.devices = [device]
    return target


class TestLLMSelector:
    def test_without_llm_uses_heuristic(self):
        target = _make_target()
        selector = KernelSelector(target=target)
        spec = _make_spec(flops=500)  # Below 1000 threshold
        decisions = selector.select([spec])
        assert len(decisions) == 1
        # Without LLM, low FLOP op gets FALLBACK (after library/ukernel checks fail)
        assert decisions[0].strategy in (KernelStrategy.FALLBACK, KernelStrategy.LIBRARY)

    def test_with_llm_overrides_to_autocomp(self):
        target = _make_target()
        mock_llm = _make_mock_llm("autocomp", "worth searching despite low FLOPs")
        selector = KernelSelector(target=target, llm_client=mock_llm)
        spec = _make_spec(op_name="custom.op", flops=500)
        decisions = selector.select([spec])
        assert len(decisions) == 1
        assert decisions[0].strategy == KernelStrategy.AUTOCOMP
        assert "LLM" in decisions[0].reason

    def test_with_llm_selects_fallback(self):
        target = _make_target()
        mock_llm = _make_mock_llm("fallback", "too trivial")
        selector = KernelSelector(target=target, llm_client=mock_llm)
        spec = _make_spec(op_name="custom.op", flops=500)
        decisions = selector.select([spec])
        assert decisions[0].strategy == KernelStrategy.FALLBACK

    def test_native_ops_bypass_llm(self):
        target = _make_target()
        mock_llm = _make_mock_llm("autocomp")
        selector = KernelSelector(target=target, llm_client=mock_llm)
        spec = _make_spec(op_name="arith.addf", flops=10)
        decisions = selector.select([spec])
        # Native ops are detected before LLM is consulted
        assert decisions[0].strategy == KernelStrategy.NATIVE
        mock_llm.generate_structured.assert_not_called()
