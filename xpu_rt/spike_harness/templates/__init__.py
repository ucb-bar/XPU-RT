"""Per-target ``init.c`` + ``driver.c`` renderers.

Dispatch entry points: :func:`render_init_c`, :func:`render_driver_c`,
:func:`stage_contract_dir`. Each routes by ``target_id`` to the
per-target snippet module (``gemmini.py`` / ``saturn.py``) declared
in :data:`xpu_rt.spike_harness.targets.SPIKE_TARGETS`.

Adding a target = author one ``templates/<id>.py`` exposing the same
three functions (``render_init_c() -> str``,
``render_driver_c(M, K, N) -> str``,
``stage_contract_dir(out_dir, M, K, N) -> Path``) and register the
module path in :data:`SPIKE_TARGETS`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from xpu_rt.spike_harness.targets import resolve_target


def _load(target_id: str):
    spec = resolve_target(target_id)
    return importlib.import_module(spec.templates_module)


def render_init_c(target_id: str) -> str:
    """Return the starter ``init.c`` for ``target_id``. KB rewrites the
    body of ``launch_gpu_implementation`` during the agent loop."""
    return _load(target_id).render_init_c()


def render_driver_c(target_id: str, M: int, K: int, N: int) -> str:
    """Return the per-shape ``driver.c`` for ``target_id`` with the
    constants ``M``, ``K``, ``N`` stamped in."""
    return _load(target_id).render_driver_c(M=M, K=K, N=N)


def stage_contract_dir(target_id: str, out_dir: Path, *, M: int, K: int, N: int) -> Path:
    """Write ``init.cu`` + ``driver.cpp`` into ``out_dir`` for the
    given target × shape. KB-vanilla's filename convention; the
    compile server treats both as C regardless of extension."""
    return _load(target_id).stage_contract_dir(out_dir, M=M, K=K, N=N)


__all__ = ["render_driver_c", "render_init_c", "stage_contract_dir"]
