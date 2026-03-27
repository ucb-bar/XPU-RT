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
"""

from __future__ import annotations

__all__: list[str] = []
