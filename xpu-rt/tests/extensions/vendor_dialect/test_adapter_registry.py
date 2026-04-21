"""Adapter registry: register / get / find-by-target / reset."""

from __future__ import annotations

import pytest

from compgen.extensions.vendor_dialect.adapter import LoweringResult, VendorDialectAdapter
from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.registry import (
    adapters_for_target,
    available_adapters,
    get_adapter,
    register_adapter,
    reset_registry,
)
from compgen.targets.backend import CompiledArtifact


class DummyAdapter(VendorDialectAdapter):
    def lower_payload(self, payload_mlir, *, output_dir, options=None):
        return LoweringResult(vendor_mlir=payload_mlir)

    def emit_artifact(self, lowering, *, output_dir, options=None):
        return CompiledArtifact(
            code=lowering.vendor_mlir,
            format="dummy",
            target_name=self.target,
        )


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


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_register_and_get() -> None:
    adapter = DummyAdapter(_descriptor("alpha", "t1"))
    register_adapter(adapter)
    assert "alpha" in available_adapters()
    assert get_adapter("alpha") is adapter


def test_duplicate_register_without_replace_raises() -> None:
    a1 = DummyAdapter(_descriptor("alpha", "t1"))
    a2 = DummyAdapter(_descriptor("alpha", "t1"))
    register_adapter(a1)
    with pytest.raises(ValueError):
        register_adapter(a2)


def test_duplicate_register_with_replace() -> None:
    a1 = DummyAdapter(_descriptor("alpha", "t1"))
    a2 = DummyAdapter(_descriptor("alpha", "t1"))
    register_adapter(a1)
    register_adapter(a2, replace=True)
    assert get_adapter("alpha") is a2


def test_find_for_target() -> None:
    a1 = DummyAdapter(_descriptor("alpha", "t1"))
    a2 = DummyAdapter(_descriptor("beta", "t2"))
    register_adapter(a1)
    register_adapter(a2)
    matched = adapters_for_target("t2")
    assert [a.name for a in matched] == ["beta"]


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_adapter("nope")
