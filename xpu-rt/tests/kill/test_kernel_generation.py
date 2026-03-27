"""Kill Test 2: Kernel Generation Usefulness.

Validates that the kernel validation pipeline works correctly:
kernel code → compile → correctness check → pass/fail.

This test uses KernelValidator with trivial Python kernels (no GPU/API needed).
Real Autocomp search is tested separately with @requires_gpu marker.
"""

from __future__ import annotations

import torch
from compgen.ir.payload.contracts import KernelContract
from compgen.kernels.contracts import KernelSpec
from compgen.kernels.validate import KernelValidator, validate_kernel


def _make_spec(op_name: str = "test_matmul") -> KernelSpec:
    return KernelSpec(contract=KernelContract(op_name=op_name))


def test_matmul_kernel_validation() -> None:
    """A correct matmul kernel passes validation."""
    kernel_code = """
def kernel(a, b):
    return torch.matmul(a, b)
"""
    spec = _make_spec("matmul")
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    ref = torch.matmul(a, b)

    result = validate_kernel(kernel_code, spec, (a, b), ref)
    assert result.passed, f"Kernel validation failed: {result.diagnostics}"
    assert result.correct
    assert result.compiles
    assert result.max_abs_error < 1e-5


def test_fused_reduction_kernel_validation() -> None:
    """A correct layernorm-like kernel passes validation."""
    kernel_code = """
def kernel(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5)
"""
    spec = _make_spec("layernorm")
    x = torch.randn(4, 64)
    ref = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
        x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
    )

    result = validate_kernel(kernel_code, spec, (x,), ref)
    assert result.passed, f"Kernel validation failed: {result.diagnostics}"
    assert result.correct


def test_incorrect_kernel_fails_validation() -> None:
    """An incorrect kernel must fail correctness check."""
    kernel_code = """
def kernel(a, b):
    return a * b  # Wrong: should be matmul
"""
    spec = _make_spec("matmul")
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    ref = torch.matmul(a, b)

    result = validate_kernel(kernel_code, spec, (a, b), ref)
    assert not result.correct


def test_kernel_go_no_go() -> None:
    """Aggregate: validation pipeline must handle correct and incorrect kernels."""
    validator = KernelValidator()

    # Test 3 kernels: 2 correct, 1 wrong
    results = []

    # Correct add
    code1 = "def kernel(a, b): return a + b"
    r1 = validator.validate(code1, _make_spec("add"), (torch.ones(4), torch.ones(4)), torch.ones(4) * 2)
    results.append(r1.correct)

    # Correct mul
    code2 = "def kernel(a, b): return a * b"
    r2 = validator.validate(code2, _make_spec("mul"), (torch.ones(4) * 3, torch.ones(4) * 2), torch.ones(4) * 6)
    results.append(r2.correct)

    # Wrong sub (should be add)
    code3 = "def kernel(a, b): return a - b"
    r3 = validator.validate(code3, _make_spec("add"), (torch.ones(4) * 3, torch.ones(4) * 2), torch.ones(4) * 5)
    results.append(r3.correct)

    pass_rate = sum(results) / len(results)
    assert pass_rate >= 0.6, f"Pass rate {pass_rate} < 0.6"
    # At least 2 of 3 should pass (the correct ones)
    assert sum(results) >= 2
