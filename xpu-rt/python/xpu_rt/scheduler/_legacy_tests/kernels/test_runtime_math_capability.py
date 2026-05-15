"""Tests for REQ-025 — spec-level runtime_math capability flag.

A spec can declare ``runtime.math.has_libm`` / ``has_libc`` /
``intrinsics``; the loader routes the values through
``TargetProfile.metadata['runtime_math']``; ``spec_to_provider_contract``
surfaces them as ``KernelContract.runtime`` so a provider's
``accepts_contract`` / ``search`` can branch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.contracts import build_kernel_contracts, spec_to_provider_contract
from compgen.kernels.provider import RuntimeCapabilities
from compgen.targetgen.hardware_spec import RuntimeMathSpec
from compgen.targetgen.load import extract_target_profile, load_hardware_spec


def _write_spec(tmp_path: Path, runtime_block: str) -> Path:
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "platform:\n"
        "  vendor: vendor_x\n"
        "  family: fam_x\n"
        "  chip_name: chip_x\n"
        "execution_model:\n"
        "  model: simt_gpu\n"
        "engine_geometry:\n"
        "  max_warp_size: 8\n"
        f"runtime_contract:\n{runtime_block}\n"
    )
    return spec


def test_runtime_math_defaults_to_no_libm_no_intrinsics() -> None:
    """A spec that says nothing about runtime → all flags False."""
    s = RuntimeMathSpec()
    assert s.has_libm is False
    assert s.has_libc is False
    assert s.intrinsics == []


def test_loader_reads_runtime_math_block(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        runtime_block=(
            "  math:\n    has_libm: true\n    has_libc: false\n    intrinsics:\n      - mu_fexp\n      - mu_fnexp\n"
        ),
    )
    spec = load_hardware_spec(str(spec_path))
    assert spec.runtime_contract.math.has_libm is True
    assert spec.runtime_contract.math.has_libc is False
    assert spec.runtime_contract.math.intrinsics == ["mu_fexp", "mu_fnexp"]


def test_loader_handles_missing_runtime_math_block(tmp_path: Path) -> None:
    """No ``math:`` block → defaults are honoured (everything False)."""
    spec_path = _write_spec(tmp_path, runtime_block="  calling_convention: c_abi")
    spec = load_hardware_spec(str(spec_path))
    assert spec.runtime_contract.math.has_libm is False
    assert spec.runtime_contract.math.intrinsics == []


def test_extract_target_profile_routes_runtime_math_into_metadata(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        runtime_block=("  math:\n    has_libm: true\n    intrinsics: [mu_fexp]\n"),
    )
    spec = load_hardware_spec(str(spec_path))
    profile = extract_target_profile(spec)
    rm = profile.metadata["runtime_math"]
    assert rm["has_libm"] is True
    assert rm["intrinsics"] == ["mu_fexp"]


def test_provider_contract_surfaces_runtime_math(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        runtime_block=("  math:\n    has_libm: false\n    intrinsics: [mu_fexp, mu_fnexp]\n"),
    )
    spec = load_hardware_spec(str(spec_path))
    profile = extract_target_profile(spec)

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    ep = capture_model(Add(), (torch.randn(4), torch.randn(4)))
    module, _ = fx_to_xdsl(ep)
    specs = build_kernel_contracts(module, profile, None)
    pc = spec_to_provider_contract(specs[0], "r0", profile)

    assert isinstance(pc.runtime, RuntimeCapabilities)
    assert pc.runtime.has_libm is False
    assert "mu_fexp" in pc.runtime.intrinsics
    assert "mu_fnexp" in pc.runtime.intrinsics


def test_provider_can_branch_on_runtime_capabilities(tmp_path: Path) -> None:
    """Concrete worked example: a softmax provider that needs ``expf``
    rejects a target without libm but accepts one with it."""
    from compgen.kernels.codegen_fallback import run_provider_fallback
    from compgen.kernels.provider import (
        KernelContract as ProviderContract,
    )
    from compgen.kernels.provider import (
        KnowledgeExport,
        ProviderResult,
        SearchBudget,
    )

    class _NeedsLibmProvider:
        name = "needs_libm"

        def accepts_contract(self, c: ProviderContract) -> bool:
            return c.runtime.has_libm or "mu_fexp" in c.runtime.intrinsics

        def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
            return ProviderResult(
                found=True,
                kernel_code="// expf or mu_fexp\n",
                language="cpp",
                correct=True,
            )

        def export_knowledge(self) -> list[KnowledgeExport]:
            return []

    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare_spec = _write_spec(bare_dir, runtime_block="  math:\n    has_libm: false\n")
    libm_dir = tmp_path / "libm_target"
    libm_dir.mkdir()
    libm_spec = _write_spec(libm_dir, runtime_block="  math:\n    has_libm: true\n")

    bare_profile = extract_target_profile(load_hardware_spec(str(bare_spec)))
    libm_profile = extract_target_profile(load_hardware_spec(str(libm_spec)))

    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    ep = capture_model(Mul(), (torch.randn(4), torch.randn(4)))
    module, _ = fx_to_xdsl(ep)

    bare_out = run_provider_fallback(
        module,
        bare_profile,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_NeedsLibmProvider()],
    )
    libm_out = run_provider_fallback(
        module,
        libm_profile,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_NeedsLibmProvider()],
    )

    assert bare_out == [], "bare-metal target without libm should reject"
    assert libm_out, "target with libm should accept"
    assert libm_out[0]["provider"] == "needs_libm"
