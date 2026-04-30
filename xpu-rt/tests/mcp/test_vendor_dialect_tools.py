"""Tests for the entry-point-driven vendor dialect MCP tools.

Covers ``compgen_list_vendor_dialects`` and
``compgen_compile_torch_model_with_vendor`` — the surface that lets a
PyPI user + Claude Code agent drive a registered vendor adapter
without forking CompGen.
"""

from __future__ import annotations

from typing import Any

import pytest
from compgen.extensions.vendor_dialect.adapter import (
    LoweringResult,
    VendorDialectAdapter,
)
from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.registry import (
    register_adapter,
    reset_registry,
)
from compgen.mcp.tools.vendor_dialect import (
    VENDOR_DIALECT_TOOLS,
    compgen_compile_torch_model_with_vendor,
    compgen_list_vendor_dialects,
)
from compgen.targets.backend import CompiledArtifact


class _StubSessionManager:
    """Sentinel — these tools are session-less."""


# --------------------------------------------------------------------------- #
# Mock adapter
# --------------------------------------------------------------------------- #


class _MockVendorAdapter(VendorDialectAdapter):
    """Minimal adapter that returns the payload IR text unchanged."""

    version: str = "0.1.0"

    def lower_payload(self, payload_mlir, *, output_dir, options=None):
        return LoweringResult(
            vendor_mlir=payload_mlir,
            metadata={"options": options or {}},
        )

    def emit_artifact(self, lowering, *, output_dir, options=None):
        return CompiledArtifact(
            code=lowering.vendor_mlir,
            format="mock",
            target_name=self.target,
        )

    def capabilities(self) -> dict[str, Any]:
        return {"supports_matmul": True, "tile_shapes": ["128x128"]}


def _descriptor(name: str, target: str) -> VendorDialectDescriptor:
    return VendorDialectDescriptor(
        name=name,
        package_name=f"compgen_{name}",
        repo_path=f"/tmp/{name}",
        target=target,
        compile_entry=CompileEntry(),
        lowering=LoweringStrategy(),
        bundle=BundlePlan(),
        verification=VerificationPlan(),
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


# --------------------------------------------------------------------------- #
# Tool registry sanity
# --------------------------------------------------------------------------- #


def test_new_tools_registered() -> None:
    """Both new tools are exported and have the expected schema shape."""
    by_name = {t["name"]: t for t in VENDOR_DIALECT_TOOLS}
    assert "compgen_list_vendor_dialects" in by_name
    assert "compgen_compile_torch_model_with_vendor" in by_name

    list_tool = by_name["compgen_list_vendor_dialects"]
    assert callable(list_tool["handler"])
    assert list_tool["input_schema"]["required"] == []

    compile_tool = by_name["compgen_compile_torch_model_with_vendor"]
    assert callable(compile_tool["handler"])
    assert set(compile_tool["input_schema"]["required"]) == {
        "model_pickle_b64",
        "sample_input_pickle_b64",
        "output_dir",
        "vendor_name",
    }


# --------------------------------------------------------------------------- #
# compgen_list_vendor_dialects
# --------------------------------------------------------------------------- #


def test_list_with_no_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing on the entry-point group, the tool returns []."""

    class _EmptyEPSet:
        def __iter__(self):
            return iter([])

    def _fake_eps(*, group: str | None = None, **_: object):
        return _EmptyEPSet()

    monkeypatch.setattr(
        "compgen.mcp.tools.vendor_dialect.importlib_metadata.entry_points",
        _fake_eps,
    )

    res = compgen_list_vendor_dialects(_StubSessionManager())
    assert res == {"vendor_dialects": []}


def test_list_surfaces_registered_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mock adapter advertised on the entry-point group is surfaced
    with name / target / version / capabilities."""
    descriptor = _descriptor("mock_vendor", "mock-target:v1")
    adapter = _MockVendorAdapter(descriptor)

    class _FakeEntryPoint:
        name = "mock_vendor"
        value = "compgen_mock_vendor.adapter:make_adapter"
        group = "compgen.vendor_dialects"

        def load(self):  # noqa: N805 — fake importlib EntryPoint shim
            return lambda: adapter

    def _fake_eps(*, group: str | None = None, **_: object):
        if group == "compgen.vendor_dialects":
            return [_FakeEntryPoint()]
        return []

    monkeypatch.setattr(
        "compgen.mcp.tools.vendor_dialect.importlib_metadata.entry_points",
        _fake_eps,
    )

    res = compgen_list_vendor_dialects(_StubSessionManager())
    assert "vendor_dialects" in res
    assert len(res["vendor_dialects"]) == 1

    rec = res["vendor_dialects"][0]
    assert rec["name"] == "mock_vendor"
    assert rec["target"] == "mock-target:v1"
    assert rec["version"] == "0.1.0"
    assert rec["capabilities"] == {
        "supports_matmul": True,
        "tile_shapes": ["128x128"],
    }
    assert rec["module"] == "compgen_mock_vendor.adapter:make_adapter"
    assert rec["error"] is None


def test_list_reports_broken_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point that fails to load is still reported, with
    ``error`` populated — the tool never raises."""

    class _BrokenEntryPoint:
        name = "broken"
        value = "compgen_broken.adapter:explode"
        group = "compgen.vendor_dialects"

        def load(self):  # noqa: N805 — fake importlib EntryPoint shim
            raise ImportError("simulated broken adapter")

    def _fake_eps(*, group: str | None = None, **_: object):
        return [_BrokenEntryPoint()]

    monkeypatch.setattr(
        "compgen.mcp.tools.vendor_dialect.importlib_metadata.entry_points",
        _fake_eps,
    )

    res = compgen_list_vendor_dialects(_StubSessionManager())
    assert len(res["vendor_dialects"]) == 1
    rec = res["vendor_dialects"][0]
    assert rec["name"] == "broken"
    assert rec["error"] is not None
    assert "simulated broken adapter" in rec["error"]
    assert rec["target"] is None
    assert rec["capabilities"] is None


# --------------------------------------------------------------------------- #
# compgen_compile_torch_model_with_vendor
# --------------------------------------------------------------------------- #


def test_compile_with_unknown_vendor_returns_vendor_not_found(tmp_path) -> None:
    """A vendor name absent from the registry surfaces as
    ``vendor_not_found`` plus the list of available names; the tool
    never raises."""
    # Pre-register one adapter so ``available`` is non-trivial.
    register_adapter(_MockVendorAdapter(_descriptor("mock_vendor", "mock:v1")))

    res = compgen_compile_torch_model_with_vendor(
        _StubSessionManager(),
        model_pickle_b64="",
        sample_input_pickle_b64="",
        output_dir=str(tmp_path),
        vendor_name="nonexistent",
        vendor_options=None,
    )
    assert res["status"] == "vendor_not_found"
    assert res["bundle_dir"] is None
    assert res["vendor_name"] == "nonexistent"
    assert "available" in res
    assert "mock_vendor" in res["available"]
    assert res["error"] is not None
    assert isinstance(res["elapsed_ms"], float)
