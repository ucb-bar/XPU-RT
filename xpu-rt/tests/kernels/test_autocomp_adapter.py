"""Tests for autocomp adapter integration."""

from __future__ import annotations

import pytest
from compgen.kernels.autocomp_adapter import AutocompAdapter


def test_adapter_instantiation() -> None:
    """AutocompAdapter should be instantiable with defaults."""
    adapter = AutocompAdapter()
    assert adapter.beam_size == 4
    assert adapter.max_iterations == 10


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_adapter_translate_profile() -> None:
    """_translate_profile should map TargetProfile to autocomp HardwareConfig."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_adapter_search_kernel() -> None:
    """search_kernel should run autocomp beam search and return AutocompResult."""
