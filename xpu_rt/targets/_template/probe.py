"""Template Probe — fill in for your target's hardware detection.

A probe answers: "is this target reachable on the current host?"
The Wave 1.10 class-level Protocols (``GpuProbe`` /
``CpuProbe``-equivalent) are what every probe satisfies.
"""

from __future__ import annotations

from typing import Any


class TemplateProbe:
    """Replace with ``YourArchProbe``. Methods are stubs that
    document the contract — fill each one in for your target."""

    def is_available(self) -> bool:
        """Cheap probe — returns True iff this target's runtime is
        reachable. Hosts that can't run this target return False
        without raising."""
        return False

    def device_arch(self) -> str:
        """Vendor-specific arch tag the universal compile path
        passes back to your other adapters. NVIDIA returns
        ``"sm_100"``; AMD returns ``"gfx942"``; etc."""
        return "template_arch"

    def supports_clusters(self) -> bool:
        """True iff multi-block-per-task cooperative primitive
        exists. NVIDIA cluster-launch on sm_90+, etc."""
        return False

    def supports_tensor_cores(self) -> bool:
        return False

    def library_paths(self) -> dict[str, str | None]:
        """Vendor-specific library paths NVRTC / hipcc / clang
        need. Universal modules pass these through verbatim."""
        return {}

    def vendor_extras(self) -> dict[str, Any]:
        """Anything else the vendor wants surfaced for the agent's
        audit query. Lands in ``BackendChoice.vendor_extras[vendor_id]``."""
        return {}
