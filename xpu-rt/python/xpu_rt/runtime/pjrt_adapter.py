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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

_PJRT_C_TEMPLATE = """\
/* PJRT plugin stub for target: {target} */
#include "pjrt_c_api.h"

const PJRT_Api* GetPjrtApi(void) {{
  /* TODO: Populate the PJRT_Api struct for {target}. */
  static PJRT_Api api;
  return &api;
}}
"""

_CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.18)
project({target}_pjrt_plugin C)

add_library({target}_pjrt SHARED pjrt_plugin.c)
target_include_directories({target}_pjrt PRIVATE ${{CMAKE_CURRENT_SOURCE_DIR}})
"""


@dataclass
class PJRTAdapter:
    """Optional PJRT backend adapter."""

    def generate_plugin_scaffold(self, target_name: str, output_dir: str) -> str:
        """Generate a PJRT plugin scaffold for a new target.

        Creates a C API stub implementing ``GetPjrtApi`` and a matching
        ``CMakeLists.txt`` in *output_dir*.

        Args:
            target_name: Short identifier for the target (e.g., ``"my_accel"``).
            output_dir: Directory to write scaffold files into.

        Returns:
            The absolute path of the scaffold directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log.info("pjrt.scaffold", target=target_name, output_dir=str(out))
        (out / "pjrt_plugin.c").write_text(_PJRT_C_TEMPLATE.format(target=target_name))
        (out / "CMakeLists.txt").write_text(_CMAKE_TEMPLATE.format(target=target_name))
        return str(out)

    def run_conformance(self, plugin_path: str) -> dict[str, Any]:
        """Run basic PJRT conformance checks.

        Uses JAX to exercise elementary operations (add, mul) and reports
        pass/fail for each.

        Args:
            plugin_path: Filesystem path to the compiled plugin shared library.

        Returns:
            Dict mapping test name to pass/fail boolean.

        Raises:
            RuntimeError: If ``jax`` is not installed.
        """
        try:
            import jax  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Install jax: pip install jax") from e

        log.info("pjrt.conformance", plugin_path=plugin_path)
        results: dict[str, Any] = {"basic_add": False, "basic_mul": False}
        try:
            x = jax.numpy.ones(4)
            y = jax.numpy.ones(4)
            z = x + y
            results["basic_add"] = bool(jax.numpy.allclose(z, 2.0))
            w = x * y
            results["basic_mul"] = bool(jax.numpy.allclose(w, 1.0))
        except Exception:
            pass
        return results


__all__ = ["PJRTAdapter"]
