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

TODO: Implement IREEAdapter with AOT compilation path.
TODO: Implement VMFB packaging.
TODO: Implement HAL device mapping from TargetProfile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compgen.runtime.bundle import Bundle


@dataclass
class IREEAdapter:
    """Optional IREE backend adapter.

    TODO: Implement compile_to_vmfb() for AOT compilation.
    TODO: Implement execute_vmfb() for IREE runtime execution.
    """

    def compile_to_vmfb(self, bundle: Bundle) -> Any:
        """Compile a bundle to IREE VMFB format.

        TODO: Import iree.compiler, translate IR, compile to VMFB.
        """
        raise NotImplementedError("IREEAdapter.compile_to_vmfb is not yet implemented")

    def execute_vmfb(self, vmfb_path: str, inputs: Any = None) -> Any:
        """Execute a VMFB artifact via IREE runtime.

        TODO: Import iree.runtime, load VMFB, execute.
        """
        raise NotImplementedError("IREEAdapter.execute_vmfb is not yet implemented")


__all__ = ["IREEAdapter"]
