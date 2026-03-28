"""Optional IREE backend adapter.

Provides integration with IREE's compiler and runtime for:
- Ahead-of-time (AOT) compilation to VMFB artifacts
- HAL device abstraction for heterogeneous execution
- Multi-device scheduling via Stream/HAL dialect concepts

This is NOT a core dependency. IREE is used when the deployment target
requires packaged, low-overhead artifacts or IREE's HAL driver ecosystem.

Invariants:
    - IREE is imported at call time, not at module level.
    - ImportError is caught and produces a clear diagnostic.
    - The adapter translates CompGen's ExecutionPlan to IREE concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class IREEAdapter:
    """Optional IREE backend adapter.

    Attributes:
        target_backends: IREE target backends for compilation (e.g., ``["llvm-cpu"]``).
    """

    target_backends: list[str] = field(default_factory=lambda: ["llvm-cpu"])

    def compile_to_vmfb(self, mlir_text: str, output_path: str | None = None) -> str:
        """Compile MLIR text to an IREE VMFB artifact.

        Args:
            mlir_text: The MLIR module text to compile.
            output_path: Optional filesystem path to write the VMFB bytes.

        Returns:
            The *output_path* if provided, otherwise ``"<in-memory>"``.

        Raises:
            RuntimeError: If ``iree-compiler`` is not installed.
        """
        try:
            import iree.compiler.tools as iree_tools  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Install iree-compiler: pip install iree-compiler") from e

        log.info("iree.compile", backends=self.target_backends)
        vmfb = iree_tools.compile_str(mlir_text, target_backends=self.target_backends)
        if output_path:
            Path(output_path).write_bytes(vmfb)
            return output_path
        return "<in-memory>"

    def execute_vmfb(self, vmfb_path: str, inputs: list[Any] | None = None) -> list[Any]:
        """Execute a VMFB artifact via IREE runtime.

        Args:
            vmfb_path: Filesystem path to the VMFB file.
            inputs: Positional inputs to the ``main`` function.

        Returns:
            List of output arrays.

        Raises:
            RuntimeError: If ``iree-runtime`` is not installed.
        """
        try:
            import iree.runtime as iree_rt  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Install iree-runtime: pip install iree-runtime") from e

        log.info("iree.execute", vmfb_path=vmfb_path)
        config = iree_rt.Config("local-task")
        ctx = iree_rt.SystemContext(config=config)
        vm_module = iree_rt.VmModule.copy_buffer(ctx.instance, Path(vmfb_path).read_bytes())
        ctx.add_vm_module(vm_module)
        f = ctx.modules.module.main
        return list(f(*(inputs or [])))


__all__ = ["IREEAdapter"]
