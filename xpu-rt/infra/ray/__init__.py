"""Ray-based distributed control plane for XPU-RT.

Ray is OPTIONAL.  This package is only usable when ``ray`` is installed
(``pip install 'xpu_rt[ray]'``).  The core ``xpu_rt`` package never
imports from this module at the top level.

Subsystems:
    actors/   — long-lived stateful services (registry, broker, index)
    tasks/    — one-shot distributed jobs (compile, benchmark, verify)
    serve/    — Ray Serve deployments (REST API, MCP gateway)
    tune/     — Ray Tune search experiments (tile, eqsat, evolutionary)
"""

from __future__ import annotations

__all__: list[str] = []
