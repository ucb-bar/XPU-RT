"""Profiler integration framework.

Provides a spec-driven, target-agnostic profiling system.  The
:class:`~compgen.targetgen.hardware_spec.ProfilingSpec` declares what
the hardware CAN expose.  The scaffold provides the adapter protocol.
The agentic LLM generates target-specific hooks and selects which
counters to enable.

Subpackages:
    adapters/     Concrete profiler adapter implementations.
"""

from __future__ import annotations

__all__: list[str] = []
