"""Tests for kernels/validate.py -- kernel correctness validation."""

from __future__ import annotations

import torch
from compgen.ir.payload.contracts import KernelContract
from compgen.kernels.contracts import KernelSpec
from compgen.kernels.validate import KernelValidationResult, KernelValidator, validate_kernel


def _make_spec(op_name: str = "test_op") -> KernelSpec:
    return KernelSpec(contract=KernelContract(op_name=op_name))


def test_kernel_validation_result_construction() -> None:
    result = KernelValidationResult(passed=True, correct=True)
    assert result.passed is True
    assert result.correct is True
    assert result.l2_error == 0.0
    assert result.max_abs_error == 0.0
    assert result.compiles is True
    assert result.within_perf_budget is True
    assert result.measured_latency_us is None
    assert result.diagnostics == []


def test_kernel_validator_defaults() -> None:
    v = KernelValidator()
    assert v.tolerance == 1e-5


def test_validate_correct_kernel() -> None:
    """A kernel that produces the correct output should pass."""
    kernel_code = """
def kernel(x, y):
    return x + y
"""
    spec = _make_spec("add")
    test_inputs = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
    reference = torch.tensor([5.0, 7.0, 9.0])

    result = validate_kernel(kernel_code, spec, test_inputs, reference)
    assert result.passed
    assert result.correct
    assert result.compiles
    assert result.max_abs_error < 1e-6


def test_validate_incorrect_kernel() -> None:
    """A kernel that produces wrong output should fail correctness."""
    kernel_code = """
def kernel(x, y):
    return x * y  # Wrong! Should be x + y
"""
    spec = _make_spec("add")
    test_inputs = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
    reference = torch.tensor([5.0, 7.0, 9.0])

    result = validate_kernel(kernel_code, spec, test_inputs, reference)
    assert not result.correct
    assert not result.passed
    assert result.max_abs_error > 0


def test_validate_non_compiling_kernel() -> None:
    """A kernel with syntax errors should fail compilation."""
    kernel_code = "def kernel(: broken"
    spec = _make_spec()
    result = validate_kernel(kernel_code, spec, torch.zeros(1), torch.zeros(1))
    assert not result.compiles
    assert not result.passed


def test_validate_no_callable() -> None:
    """Code without a callable function should fail."""
    kernel_code = "x = 42"
    spec = _make_spec()
    result = validate_kernel(kernel_code, spec, torch.zeros(1), torch.zeros(1))
    assert not result.passed
    assert "No callable" in result.diagnostics[0]


def test_validate_kernel_convenience() -> None:
    """validate_kernel convenience function works."""
    code = "def kernel(x): return x * 2"
    spec = _make_spec()
    inputs = torch.tensor([1.0, 2.0])
    ref = torch.tensor([2.0, 4.0])
    result = validate_kernel(code, spec, (inputs,), ref)
    assert result.passed
