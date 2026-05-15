"""Adapter resolution helpers.

A ``ProviderCard.entrypoint`` is a ``module:Symbol`` string. The
resolver imports the module lazily (hard rule 8 — optional
providers must not be imported at core module import time) and
returns the named class.

Resolution failures map to ``AdapterResolutionError`` carrying the
typed reason so callers can short-circuit cleanly without
swallowing exceptions.
"""

from __future__ import annotations

import importlib
from typing import Any

from compgen.providers.provider_types import ProviderCard


class AdapterResolutionError(RuntimeError):
    """Raised when ``ProviderCard.entrypoint`` cannot be resolved."""

    def __init__(self, *, provider_id: str, entrypoint: str, reason: str) -> None:
        self.provider_id = provider_id
        self.entrypoint = entrypoint
        self.reason = reason
        super().__init__(
            f"adapter_resolution_error: provider_id={provider_id!r} "
            f"entrypoint={entrypoint!r} reason={reason!r}"
        )


def _split_entrypoint(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"entrypoint {spec!r} must use 'module:Symbol' form"
        )
    module_path, _, symbol = spec.partition(":")
    if not module_path or not symbol:
        raise ValueError(
            f"entrypoint {spec!r} must use 'module:Symbol' form"
        )
    return module_path, symbol


def resolve_provider_class(card: ProviderCard) -> Any:
    """Resolve ``card.entrypoint`` to its provider class.

    Raises ``AdapterResolutionError`` with a typed ``reason`` on
    failure:

    * ``bad_entrypoint_syntax`` — not of the form ``module:Symbol``.
    * ``module_not_importable`` — the module side fails to import.
    * ``symbol_not_in_module`` — module imports but the symbol is
      missing.

    Never raises ``ImportError``; resolution errors are typed.
    """

    try:
        module_path, symbol = _split_entrypoint(card.entrypoint)
    except ValueError as exc:
        raise AdapterResolutionError(
            provider_id=card.provider_id,
            entrypoint=card.entrypoint,
            reason="bad_entrypoint_syntax",
        ) from exc

    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AdapterResolutionError(
            provider_id=card.provider_id,
            entrypoint=card.entrypoint,
            reason="module_not_importable",
        ) from exc

    try:
        return getattr(mod, symbol)
    except AttributeError as exc:
        raise AdapterResolutionError(
            provider_id=card.provider_id,
            entrypoint=card.entrypoint,
            reason="symbol_not_in_module",
        ) from exc
