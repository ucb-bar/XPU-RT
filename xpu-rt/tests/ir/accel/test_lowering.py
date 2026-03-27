"""Tests for accelerator dialect lowering."""

from __future__ import annotations

import pytest
from compgen.ir.accel.lowering import lower_accel_to_llvm


def test_lower_accel_to_llvm_exists() -> None:
    assert callable(lower_accel_to_llvm)


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_accel_to_llvm_basic() -> None:
    """lower_accel_to_llvm should lower accel ops to LLVM dialect ops."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_accel_to_llvm_with_target_triple() -> None:
    """lower_accel_to_llvm should respect the target_triple parameter."""
