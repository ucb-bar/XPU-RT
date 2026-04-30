"""Target management subsystem.

Manages the full target lifecycle: profile, capability, maturity, and package.

Modules:
    schema       -- TargetProfile dataclass and YAML loading
    validate     -- Profile validation (schema + semantic)
    calibrate    -- Hardware calibration and profiling
    capability   -- CapabilitySpec, target classification (Triton/accel/ukernel/hybrid)
    maturity     -- Target maturity levels (L0-L3)
    package      -- TargetPackage generation, loading, validation

The key abstraction is the **target package**: CompGen generates a target
enablement package (NOT a full compiler). The package includes profile,
capability map, recipe library, IR dialect skeleton, kernel paths, runtime
integration, and verification suite.

For targets with existing backends (Merlin, IREE, XLA), CompGen generates
an integration layer, not a duplicate backend.

Wave 1.10/1.11/1.12 — unified target hierarchy:

  targets/
    gpu/        contracts.py ← class-level Protocols
      nvidia/   __init__.py registers vendor-common entry
        common/ any-NVIDIA-GPU code (Wave 1.14 migration target)
        blackwell/ arch leaf — sm_100/sm_120, cuBLASDx, cluster-launch
        hopper/    arch leaf — sm_90, wgmma
        ampere/    arch leaf — sm_80/86, older mma atoms
      amd/      (placeholder)
      intel/    (placeholder)
    cpu/        contracts.py
      x86/      vendor entry (Wave 1.15 fills in real adapters)
      arm/      (placeholder)
    tpu/        contracts.py
    custom/     MCP-registered user targets (session scope)

Importing this package triggers registration of all in-tree
leaves into ``compgen.targets.registry``:

  >>> import compgen.targets
  >>> from compgen.targets.registry import registry
  >>> sorted(registry().classes())
  ['cpu', 'gpu']
"""

from __future__ import annotations


def _register_in_tree() -> None:
    """Import / re-import every in-tree target package so each
    one's registration side-effect runs.

    First call: ``__import__`` triggers the side-effect.
    Subsequent calls (e.g. tests after ``registry().reset()``):
    ``importlib.reload`` re-runs the registration. Best-effort —
    a broken package shouldn't kill the whole import."""
    import importlib
    import sys

    in_tree_modules = (
        "compgen.targets.gpu.nvidia",
        "compgen.targets.gpu.nvidia.blackwell",
        "compgen.targets.gpu.nvidia.hopper",
        "compgen.targets.gpu.nvidia.ampere",
        "compgen.targets.cpu.x86",
    )
    for mod_path in in_tree_modules:
        try:
            if mod_path in sys.modules:
                importlib.reload(sys.modules[mod_path])
            else:
                __import__(mod_path)
        except Exception:  # noqa: BLE001
            # Per the registry's robustness contract: a single
            # broken target package shouldn't prevent the rest
            # from registering.
            pass


_register_in_tree()

__all__: list[str] = []
