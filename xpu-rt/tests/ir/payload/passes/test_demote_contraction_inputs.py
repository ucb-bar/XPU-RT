"""Tests for DemoteContractionInputs MVP port."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes import DemoteContractionInputs
from xdsl.dialects.builtin import ModuleOp


def test_empty_module_produces_zero_count() -> None:
    mod = ModuleOp([])
    DemoteContractionInputs().run(mod)
    count = mod.attributes["compgen.demote_contraction_inputs.count"]
    assert int(count.value.data) == 0


def test_dtype_whitelist_rejects_unknown() -> None:
    mod = ModuleOp([])
    with pytest.raises(ValueError, match="dtype must be one of"):
        DemoteContractionInputs().run(mod, dtype="unheardof")


def test_dtype_whitelist_accepts_variants() -> None:
    for dt in ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2"):
        DemoteContractionInputs().run(ModuleOp([]), dtype=dt)


def test_targets_filter_accepted() -> None:
    mod = ModuleOp([])
    for filt in ("all_contractions", "matmul_only", "conv_only"):
        DemoteContractionInputs().run(mod, targets=filt)


def test_registered_as_real_tool() -> None:
    import compgen.ir.payload.passes  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    tool = r.lookup_tool("demote_contraction_inputs", phase=2)
    assert tool is not None and tool.is_stub is False
