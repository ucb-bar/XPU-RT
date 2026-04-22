"""KernelBlaster kernel provider.

Exposes NVlabs' KernelBlaster (https://github.com/NVlabs/KernelBlaster) to
CompGen's :class:`~compgen.kernels.registry.ProviderRegistry` through the
:class:`~compgen.kernels.provider.KernelProvider` protocol. The heavy
lifting — input staging, docker/shell invocation, output parsing — lives
in :mod:`compgen.kernels.kernelblaster_adapter`; this module is the
protocol-level glue.

Acceptance gate — KernelBlaster is CUDA-only and requires a CUDA kernel
source + C++ harness in ``contract.constraints.kernelblaster``. Contracts
that lack either of those fall through to the next provider.
"""

from __future__ import annotations

import structlog

from compgen.kernels.kernelblaster_adapter import (
    KernelBlasterAdapter,
    KernelBlasterConfig,
    KernelBlasterUnavailable,
)
from compgen.kernels.provider import (
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


_CUDA_TARGET_KEYWORDS = ("cuda", "gpu", "h100", "a100", "l40", "l40s", "hopper", "ampere", "ada")


class KernelBlasterProvider:
    """KernelBlaster as a first-class :class:`KernelProvider`.

    Contracts flow in from :class:`ProviderRegistry`; this class gates
    on CUDA targets + required KB payloads, delegates the search to
    :class:`KernelBlasterAdapter`, and reports graceful failure when
    KernelBlaster isn't installed on the host.

    Instance state is the accumulated knowledge from every search the
    provider has performed this process — flushed on
    :meth:`export_knowledge` so the caller (usually
    :meth:`ProviderRegistry.ingest_knowledge`) can persist it into
    :class:`~compgen.memory.store.CompilerMemory`.
    """

    def __init__(
        self,
        *,
        config: KernelBlasterConfig | None = None,
        adapter: KernelBlasterAdapter | None = None,
    ) -> None:
        self._adapter = adapter or KernelBlasterAdapter(
            config=config or KernelBlasterConfig.from_env(),
        )
        self._accumulated_knowledge: list[KnowledgeExport] = []

    @property
    def name(self) -> str:
        return "kernelblaster"

    @property
    def config(self) -> KernelBlasterConfig:
        return self._adapter.config

    def accepts_contract(self, contract: KernelContract) -> bool:
        """True iff the contract is CUDA-shaped and carries KB inputs.

        Host-level availability (repo cloned, docker present,
        ``OPENAI_API_KEY`` set) is checked at search time so that a
        correctly-shaped contract still falls through cleanly when the
        host isn't provisioned.
        """
        target = contract.target_name.lower()
        hardware = contract.hardware_key.lower()
        is_cuda = any(kw in target or kw in hardware for kw in _CUDA_TARGET_KEYWORDS)
        if not is_cuda:
            return False
        kb = contract.constraints.get("kernelblaster") or {}
        return bool(kb.get("init_cu")) and bool(kb.get("driver_cpp"))

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Delegate to the adapter; translate failures into not-found results."""
        try:
            result = self._adapter.search_kernel(contract, budget)
        except KernelBlasterUnavailable as exc:
            log.warning("kernelblaster.provider.unavailable", reason=str(exc))
            return ProviderResult(
                found=False,
                metadata={"provider": "kernelblaster", "reason": str(exc)},
            )
        except ValueError as exc:
            # Contract was gated in but lost its payload between
            # accepts_contract() and search() — surface for debugging.
            log.warning("kernelblaster.provider.bad_contract", reason=str(exc))
            return ProviderResult(
                found=False,
                metadata={"provider": "kernelblaster", "reason": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("kernelblaster.provider.search_failed", error=str(exc))
            return ProviderResult(
                found=False,
                metadata={"provider": "kernelblaster", "reason": f"{type(exc).__name__}: {exc}"},
            )

        if result.knowledge_exports:
            self._accumulated_knowledge.extend(result.knowledge_exports)
        return result

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Drain accumulated knowledge for ingestion into CompilerMemory."""
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports


__all__ = ["KernelBlasterProvider"]
