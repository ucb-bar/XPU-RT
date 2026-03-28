"""Bridge between UkernelRegistry and KernelProvider protocol.

Allows ukernels to be discovered by the ``ProviderRegistry`` alongside
autocomp, triton_templates, and other kernel providers. The bridge
translates between the ukernel selection API and the KernelProvider
search interface.
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.ir.ukernel.constraints import ConstraintContext
from compgen.ir.ukernel.registry import UkernelRegistry
from compgen.kernels.provider import (
    KernelProvider,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

# Use the provider-level KernelContract (not IR-level)
from compgen.kernels.provider import KernelContract

log = structlog.get_logger()


def _context_from_contract(contract: KernelContract) -> ConstraintContext:
    """Build a ConstraintContext from a provider-level KernelContract."""
    shapes: dict[str, int] = {}
    if contract.input_shapes:
        first = contract.input_shapes[0]
        if len(first) >= 2:
            shapes["M"] = first[0]
            shapes["K"] = first[1]
        if len(contract.input_shapes) > 1:
            second = contract.input_shapes[1]
            if len(second) >= 2:
                shapes["N"] = second[1]

    features: set[str] = set()
    if contract.hardware_key:
        features.add(f"has_{contract.hardware_key}")

    return ConstraintContext(
        shapes=shapes,
        dtypes=contract.dtypes,
        target_features=frozenset(features),
        device_type=contract.target_name.split("_")[0] if contract.target_name else "",
        layouts={"lhs": contract.layout} if contract.layout else {},
    )


class UkernelProvider:
    """KernelProvider that serves ukernels from a UkernelRegistry.

    Sits alongside autocomp, triton_templates in the ProviderRegistry.
    Does NOT replace other providers — it adds a ukernel selection lane.
    """

    def __init__(self, registry: UkernelRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "ukernel"

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Check if any registered ukernel can handle this contract."""
        if not contract.op_family:
            return False
        context = _context_from_contract(contract)
        result = self._registry.select_ukernel(contract.op_family, context)
        return result is not None

    def search(
        self,
        contract: KernelContract,
        budget: SearchBudget | None = None,
    ) -> ProviderResult:
        """Select the best ukernel for this contract."""
        context = _context_from_contract(contract)
        decl = self._registry.select_ukernel(contract.op_family, context)

        if decl is None:
            return ProviderResult(found=False)

        # Find the best body
        target_family = contract.target_name or "any"
        body = self._registry.select_body(decl.kernel_name, target_family)

        kernel_code = ""
        language = decl.body_kind
        if body is not None:
            kernel_code = body.inline_body or body.source_ref
            language = body.body_kind

        log.debug(
            "ukernel.provider.search",
            kernel=decl.kernel_name,
            transparency=decl.transparency,
            body_kind=language,
            body_found=body is not None,
        )

        return ProviderResult(
            found=True,
            kernel_code=kernel_code,
            language=language,
            correct=True,
            metadata={
                "kernel_name": decl.kernel_name,
                "transparency": decl.transparency,
                "body_kind": language,
                "target_family": body.target_family if body else "any",
                "preferred_layouts": list(decl.preferred_layouts),
                "tile_family": decl.tile_family,
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export ukernel registry knowledge."""
        exports = []
        for decl in self._registry.all_decls():
            exports.append(KnowledgeExport(
                kind="ukernel_decl",
                scope="global",
                scope_key=decl.kernel_name,
                content=f"{decl.kernel_name}: {decl.transparency} {decl.body_kind}",
                confidence=1.0,
            ))
        return exports


__all__ = ["UkernelProvider"]
