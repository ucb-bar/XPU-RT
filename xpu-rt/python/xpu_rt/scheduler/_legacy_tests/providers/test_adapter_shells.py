"""every shipped card resolves to a real KernelProvider class."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from compgen.providers.adapters.base import (
    AdapterResolutionError,
    resolve_provider_class,
)
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
)
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.legacy_shim import wrap_legacy
from compgen.providers.provider_types import PROBE_STATUSES
from compgen.providers.result_v1 import ProviderResultV1


# ---------------------------------------------------------------------------
# Every shipped card resolves
# ---------------------------------------------------------------------------


def test_every_provider_card_entrypoint_resolves():
    failures = []
    for c in iter_provider_cards():
        try:
            resolve_provider_class(c)
        except AdapterResolutionError as exc:
            failures.append((c.provider_id, exc.reason))
    assert not failures, f"provider entrypoints did not resolve: {failures}"


def test_every_dialect_card_entrypoint_resolves():
    failures = []
    for c in iter_dialect_cards():
        try:
            mod_path, _, sym = c.entrypoint.partition(":")
            mod = importlib.import_module(mod_path)
            getattr(mod, sym)
        except Exception as exc:
            failures.append((c.dialect_provider_id, repr(exc)))
    assert not failures, f"dialect entrypoints did not resolve: {failures}"


# ---------------------------------------------------------------------------
# Every shell satisfies the ABC (or is shimmable)
# ---------------------------------------------------------------------------


_KWARGS_REQUIRED = {"claude_kernel"}  # legacy constructor needs kwargs


def test_every_provider_satisfies_kernel_provider_or_shims():
    """For each card, the entrypoint either:
      a) directly satisfies the KernelProvider ABC; or
      b) is a legacy class that wrap_legacy(card, instance) shims into one.
    """

    for c in iter_provider_cards():
        cls = resolve_provider_class(c)
        if c.provider_id in _KWARGS_REQUIRED:
            # Skip instantiation; the class is real and resolvable.
            continue
        inst = cls()
        if isinstance(inst, KernelProvider):
            continue
        # Legacy class — shim must wrap it cleanly.
        shimmed = wrap_legacy(c, inst)
        assert isinstance(shimmed, KernelProvider), (
            f"{c.provider_id}: neither direct ABC nor shimmable"
        )


def test_every_dialect_provider_satisfies_kernel_provider():
    for c in iter_dialect_cards():
        mod_path, _, sym = c.entrypoint.partition(":")
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, sym)
        inst = cls()
        assert isinstance(inst, KernelProvider), (
            f"dialect {c.dialect_provider_id}: not a KernelProvider"
        )


# ---------------------------------------------------------------------------
# Every shell's probe() returns a typed status
# ---------------------------------------------------------------------------


def test_every_shell_probe_returns_typed_status():
    """No probe may raise; every result has a typed status."""

    for c in iter_provider_cards():
        if c.provider_id in _KWARGS_REQUIRED:
            continue
        cls = resolve_provider_class(c)
        inst = cls()
        if isinstance(inst, KernelProvider):
            probe = inst.probe()
        else:
            probe = wrap_legacy(c, inst).probe()
        assert probe.status in PROBE_STATUSES, (
            f"{c.provider_id}: probe status {probe.status!r} not in enum"
        )


def test_every_dialect_shell_probe_returns_typed_status():
    for c in iter_dialect_cards():
        mod_path, _, sym = c.entrypoint.partition(":")
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, sym)
        inst = cls()
        probe = inst.probe()
        assert probe.status in PROBE_STATUSES, (
            f"dialect {c.dialect_provider_id}: probe status {probe.status!r}"
        )


# ---------------------------------------------------------------------------
# Shells decline contracts honestly; propose() returns ProviderResultV1
# ---------------------------------------------------------------------------


class _Target:
    name = "host_cpu"


def test_shell_propose_returns_v1_result(tmp_path: Path):
    """Pick a representative shell (cuda_tile_ir) and verify propose
    returns a typed v1 result with status=blocked."""

    from compgen.providers.adapters.cuda_tile_ir import CudaTileIRProvider

    provider = CudaTileIRProvider()
    req = KernelCodegenRequest(
        task_id="kcodegen_test",
        contract=None,
        target=_Target(),
        artifact_dir=str(tmp_path),
    )
    result = provider.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "blocked"
    assert result.detail


def test_shell_propose_never_raises():
    """Every shell's propose() returns a typed result rather than raising."""

    from compgen.providers.adapters.bitblas import BitBlasProvider
    from compgen.providers.adapters.cutlass_cute import CutlassCuteProvider
    from compgen.providers.adapters.kernelbench_caesar import (
        KernelBenchCaesarProvider,
    )
    from compgen.providers.adapters.thunderkittens import ThunderKittensProvider

    for cls in (
        BitBlasProvider,
        CutlassCuteProvider,
        KernelBenchCaesarProvider,
        ThunderKittensProvider,
    ):
        inst = cls()
        req = KernelCodegenRequest(
            task_id="t",
            contract=None,
            target=_Target(),
            artifact_dir="/tmp",
        )
        # Must return ProviderResultV1, must not raise.
        result = inst.propose(req)
        assert isinstance(result, ProviderResultV1)
        assert result.status == "blocked"


def test_inventory_completeness():
    """All 19 provider cards + 10 dialect cards present and resolvable."""

    pids = {c.provider_id for c in iter_provider_cards()}
    dids = {c.dialect_provider_id for c in iter_dialect_cards()}
    assert len(pids) == 19, f"expected 19 provider cards, got {len(pids)}"
    assert len(dids) == 10, f"expected 10 dialect cards, got {len(dids)}"
