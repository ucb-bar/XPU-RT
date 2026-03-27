"""Tests for ukernel lowering to concrete calls."""

from __future__ import annotations

import pytest
from compgen.ir.ukernel.lower import lower_ukernel_to_call


def test_lower_ukernel_to_call_exists() -> None:
    assert callable(lower_ukernel_to_call)


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_ukernel_to_call_c_backend() -> None:
    """lower_ukernel_to_call with backend='c' should produce C function calls."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_ukernel_to_call_triton_backend() -> None:
    """lower_ukernel_to_call with backend='triton' should produce Triton kernel launches."""
