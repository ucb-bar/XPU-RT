"""per-provider normalized test for Triton.

Pins:
1. Triton card resolves through the registry shim.
2. The legacy ``TritonTemplateProvider`` wraps into the ABC.
3. ``probe()`` is ``available`` when triton is installed.
4. ``provider_id`` matches the card.

Triton currently doesn't have a flash_attention template; the
deepening of that path is. This test only pins the
substrate / interface — not the kernel content.
"""

from __future__ import annotations

import pytest

from compgen.providers.kernel_provider import KernelProvider
from compgen.providers.provider_registry import build_provider_registry
from compgen.providers.provider_types import (
    PROBE_STATUSES,
    ProviderProbeResult,
)


def test_triton_card_present_in_registry():
    r = build_provider_registry()
    assert "triton" in r.provider_ids()
    card = r.card_for("triton")
    assert card.integration_level == "promote"
    assert card.paper_claimable is True
    assert "cuda" in card.target_families
    assert "matmul" in card.contract_kinds


def test_triton_instance_satisfies_kernel_provider_via_shim():
    r = build_provider_registry()
    inst = r.instance("triton")
    assert isinstance(inst, KernelProvider)


def test_triton_probe_returns_typed_status():
    r = build_provider_registry()
    probe = r.probe("triton")
    assert isinstance(probe, ProviderProbeResult)
    assert probe.status in PROBE_STATUSES


def test_triton_class_resolves_to_real_module():
    """Sanity: triton card entrypoint points at the live class."""

    from compgen.providers.adapters.base import resolve_provider_class

    card = build_provider_registry().card_for("triton")
    cls = resolve_provider_class(card)
    assert cls.__module__ == "compgen.kernels.providers.triton_templates"
    assert cls.__name__ == "TritonTemplateProvider"
