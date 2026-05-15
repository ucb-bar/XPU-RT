"""Lazy Ray import guard — shared by all infra/ray modules.

Every file in ``infra/ray/`` calls :func:`require_ray` at the top of
each function that needs Ray.  This keeps Ray as an optional dependency
and provides a clear error message when it is missing.
"""

from __future__ import annotations

from typing import Any

_INSTALL_MSG = (
    "Ray is required for the CompGen infrastructure layer. "
    "Install with: pip install 'compgen[ray]'"
)


def require_ray() -> Any:
    """Import and return the ``ray`` module.

    Raises:
        ImportError: If Ray is not installed, with installation instructions.
    """
    try:
        import ray

        return ray
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def require_serve() -> Any:
    """Import and return ``ray.serve``."""
    try:
        from ray import serve

        return serve
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def require_tune() -> Any:
    """Import and return ``ray.tune``."""
    try:
        from ray import tune

        return tune
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def ensure_ray_initialized(address: str = "auto") -> None:
    """Ensure ``ray.init()`` has been called.

    Args:
        address: Ray cluster address.  ``"auto"`` connects to a running
            cluster or starts a local one.
    """
    ray = require_ray()
    if not ray.is_initialized():
        ray.init(address=address, namespace="compgen")
