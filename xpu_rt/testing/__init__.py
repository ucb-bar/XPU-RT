"""Public testing surface — importable by the remote agent.

Exposes the Event Tensor Compiler conformance harness so a remote
Claude Code agent on a Blackwell box can install ``xpu_rt[cuda]``
from PyPI, import :mod:`xpu_rt.testing.etc_conformance`, and run
the paper's reference workloads without ever cloning the source repo.

Public names:

- :class:`xpu_rt.testing.etc_conformance.ConformanceWorkload`
- :class:`xpu_rt.testing.etc_conformance.ConformanceReport`
- :func:`xpu_rt.testing.etc_conformance.run_conformance`
- :func:`xpu_rt.testing.etc_conformance.summarize_reports`
"""

from __future__ import annotations

from xpu_rt.testing.etc_conformance import (
    ConformanceReport,
    ConformanceWorkload,
    PassGate,
    run_conformance,
    summarize_reports,
)

__all__ = [
    "ConformanceReport",
    "ConformanceWorkload",
    "PassGate",
    "run_conformance",
    "summarize_reports",
]
