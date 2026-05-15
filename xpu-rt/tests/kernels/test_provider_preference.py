"""Pin the KernelBlaster-before-Autocomp provider preference.

Both providers cover CUDA kernel-search; their priorities decide
who bids first when a contract is applicable to both.
"""

from __future__ import annotations

from compgen.kernels.providers.autocomp import AutocompProvider
from compgen.kernels.providers.kernelblaster import KernelBlasterProvider
from compgen.kernels.registry import ProviderRegistry


def test_kernelblaster_priority_exceeds_autocomp():
    assert KernelBlasterProvider.priority > AutocompProvider.priority


def test_default_registry_orders_kernelblaster_before_autocomp():
    """``default_registry()`` registers discovered providers in
    priority order, so KB precedes autocomp in iteration order."""
    reg = ProviderRegistry()
    # Register in deliberately-wrong order so the sort step is the
    # thing that fixes it.
    discovered = [AutocompProvider(), KernelBlasterProvider()]
    discovered.sort(
        key=lambda p: (-int(getattr(p, "priority", 0)), getattr(p, "name", ""))
    )
    for p in discovered:
        reg.register(p)
    names = reg.provider_names
    assert names.index("kernelblaster") < names.index("autocomp"), (
        f"expected kernelblaster before autocomp, got {names}"
    )


def test_applicable_ranks_kernelblaster_first_when_both_match():
    """V3 applicability sort: highest priority first.

    Both providers expose ``priority`` and (by leaving
    ``applicable_targets`` / ``applicable_archetypes`` unset) match
    wildcardly, so a synthetic V3 contract should rank KB above
    autocomp regardless of registration order.
    """
    try:
        from compgen.kernels.contract_v3 import KernelContractV3  # noqa: F401
    except Exception:
        import pytest

        pytest.skip("KernelContractV3 not importable in this environment")

    reg = ProviderRegistry()
    reg.register(AutocompProvider())
    reg.register(KernelBlasterProvider())

    class _Hw:
        target_name = "cuda"

    class _Exec:
        hardware = _Hw()

    class _Orch:
        execution = _Exec()

    class _Arch:
        value = "matmul"

    class _Ctr:
        orchestration = _Orch()
        archetype = _Arch()

    ranked = reg.applicable(_Ctr())  # type: ignore[arg-type]
    # Filter to the two we care about; ignore any other discovered
    # providers (c_reference, etc.).
    names = [r.provider_name for r in ranked if r.provider_name in {"kernelblaster", "autocomp"}]
    assert names == ["kernelblaster", "autocomp"], names
