"""Drop-in Gemmini backend for vanilla KernelBlaster.

Reproduces vanilla KB's compile + run microservice API (same FastAPI
routes, same request / response shapes) but with the bodies swapped
to target Gemmini via Spike+gemmini-extension instead of CUDA via
nvcc+NCU. The intent: run **vanilla** KernelBlaster (its agent /
LangGraph workflow / strategy bandit / optimization-database lookup
/ multi-round repair loop) end-to-end without changing any of KB's
code — only the eval backbone.

See plan 2 § Option A in /home/agustin/.claude/plans/floofy-foraging-matsumoto.md.
"""

from __future__ import annotations

__all__: list[str] = []
