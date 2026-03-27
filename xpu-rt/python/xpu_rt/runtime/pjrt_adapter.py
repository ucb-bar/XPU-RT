"""Optional PJRT backend adapter.

Provides integration with OpenXLA's PJRT plugin interface for:
- Cross-framework device API compatibility (JAX, PyTorch/XLA)
- Plugin-based new hardware bring-up
- StableHLO interchange

This is NOT a core dependency. PJRT is used when targeting the
OpenXLA ecosystem or when building device plugins.

Invariants:
    - PJRT is imported at call time, not at module level.
    - The adapter exposes PJRT plugin scaffolding for new targets.

TODO: Implement PJRTAdapter with plugin scaffold generation.
TODO: Implement conformance test runner for PJRT plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compgen.targets.schema import TargetProfile


@dataclass
class PJRTAdapter:
    """Optional PJRT backend adapter.

    TODO: Implement generate_plugin_scaffold() for new device targets.
    TODO: Implement run_conformance() for plugin testing.
    """

    def generate_plugin_scaffold(self, target: TargetProfile, output_dir: str) -> None:
        """Generate a PJRT plugin scaffold for a new target.

        TODO: Create C API stubs implementing GetPjRtApi.
        TODO: Generate build configuration.
        TODO: Generate basic conformance test harness.
        """
        raise NotImplementedError("PJRTAdapter.generate_plugin_scaffold is not yet implemented")

    def run_conformance(self, plugin_path: str) -> Any:
        """Run PJRT conformance tests on a plugin.

        TODO: Load plugin, run standard PJRT test cases.
        """
        raise NotImplementedError("PJRTAdapter.run_conformance is not yet implemented")


__all__ = ["PJRTAdapter"]
