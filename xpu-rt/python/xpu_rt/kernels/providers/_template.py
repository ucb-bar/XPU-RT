"""Template: Custom Kernel Generator Provider

Copy this file into the ``providers/`` directory and implement the
``KernelProvider`` protocol.  Your provider will be auto-discovered
by the ``ProviderRegistry`` when registered.

See ``compgen.kernels.provider`` for the full protocol definition.
See ``autocomp.py`` for a working example (LLM-driven search).
See ``triton_templates.py`` for another example (parameterized templates).

Steps:
    1. Copy this file: ``cp _template.py my_provider.py``
    2. Implement ``accepts_contract()``, ``search()``, ``export_knowledge()``
    3. Register: ``ProviderRegistry().register(MyProvider())``
"""

from __future__ import annotations

from typing import Any

from compgen.kernels.provider import KernelContract, ProviderResult


class TemplateKernelProvider:
    """Template kernel generation provider.

    Replace this with your own implementation. A provider:
    - Receives a ``KernelContract`` describing what to compute
    - Generates kernel code (any language: Python, C, Triton, ISA)
    - Returns a ``ProviderResult`` with the code and correctness info
    """

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Return True for contracts this provider can handle.

        Filter by op_family, dtypes, shapes, target, etc.
        """
        # Example: only handle matmul ops
        return contract.op_family == "matmul"

    def search(self, contract: KernelContract, budget: int = 50) -> ProviderResult:
        """Search for an optimized kernel implementation.

        Args:
            contract: What to compute (shapes, dtypes, target).
            budget: Max iterations for the search.

        Returns:
            ProviderResult with kernel code and metadata.
        """
        # TODO: Replace with your kernel generation logic
        kernel_code = f"# Kernel for {contract.op_family}: {contract.input_shapes}"
        return ProviderResult(
            found=False,
            kernel_code=kernel_code,
            language="python",
            correct=False,
        )

    def export_knowledge(self) -> list[Any]:
        """Export learned optimization knowledge for the memory system."""
        return []
