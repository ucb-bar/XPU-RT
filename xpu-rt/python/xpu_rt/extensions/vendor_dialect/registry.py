"""Vendor dialect adapter registry.

Two complementary discovery paths:

1. **Entry point discovery** via ``compgen.vendor_dialects`` in the
   consumer package's ``pyproject.toml``. This is how user-space packages
   (e.g. ``compgen_cuda_tile``) advertise themselves once ``pip install``-ed.
2. **Runtime registration** via :func:`register_adapter` — used by tests,
   demos, and in-process composition.

The registry is a singleton per process. Callers fetch adapters by name
or by target.
"""

from __future__ import annotations

import threading
from importlib import metadata as importlib_metadata

import structlog

from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter

log = structlog.get_logger()


_ENTRY_POINT_GROUP = "compgen.vendor_dialects"


class VendorAdapterRegistry:
    """Process-wide registry of vendor dialect adapters.

    Prefer the module-level helpers (:func:`register_adapter`,
    :func:`get_adapter`, :func:`available_adapters`) over touching
    this class directly — they share a single global instance.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, VendorDialectAdapter] = {}
        self._lock = threading.Lock()
        self._entry_points_loaded = False

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, adapter: VendorDialectAdapter, *, replace: bool = False) -> None:
        name = adapter.name
        with self._lock:
            if name in self._by_name and not replace:
                raise ValueError(
                    f"vendor adapter {name!r} already registered; "
                    f"pass replace=True to override"
                )
            self._by_name[name] = adapter
        log.info("vendor_registry.register", name=name, target=adapter.target)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._by_name.pop(name, None)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> VendorDialectAdapter:
        self._ensure_entry_points_loaded()
        with self._lock:
            if name not in self._by_name:
                raise KeyError(
                    f"vendor adapter {name!r} not registered "
                    f"(available: {sorted(self._by_name)})"
                )
            return self._by_name[name]

    def find_for_target(self, target: str) -> list[VendorDialectAdapter]:
        self._ensure_entry_points_loaded()
        with self._lock:
            return [a for a in self._by_name.values() if a.supports_target(target)]

    def names(self) -> list[str]:
        self._ensure_entry_points_loaded()
        with self._lock:
            return sorted(self._by_name)

    # ------------------------------------------------------------------ #
    # Entry-point discovery
    # ------------------------------------------------------------------ #

    def _ensure_entry_points_loaded(self) -> None:
        if self._entry_points_loaded:
            return
        with self._lock:
            if self._entry_points_loaded:
                return
            self._entry_points_loaded = True
        try:
            eps = importlib_metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception as exc:  # pragma: no cover — importlib.metadata oddities
            log.warning("vendor_registry.entry_points.error", error=str(exc))
            return
        for ep in eps:
            try:
                factory = ep.load()
                adapter = factory() if callable(factory) else factory
                if not isinstance(adapter, VendorDialectAdapter):
                    log.warning(
                        "vendor_registry.entry_points.bad_type",
                        ep=ep.name,
                        type=type(adapter).__name__,
                    )
                    continue
                self.register(adapter, replace=True)
            except Exception as exc:
                log.warning(
                    "vendor_registry.entry_points.load_failed",
                    ep=ep.name,
                    error=str(exc),
                )

    # ------------------------------------------------------------------ #
    # Testing helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Drop all registered adapters and clear the entry-point cache.

        Intended for tests only.
        """
        with self._lock:
            self._by_name.clear()
            self._entry_points_loaded = False


# --------------------------------------------------------------------------- #
# Module-level singleton + helpers
# --------------------------------------------------------------------------- #


_REGISTRY = VendorAdapterRegistry()


def register_adapter(adapter: VendorDialectAdapter, *, replace: bool = False) -> None:
    """Register a vendor adapter in the process-wide registry."""
    _REGISTRY.register(adapter, replace=replace)


def get_adapter(name: str) -> VendorDialectAdapter:
    """Fetch a registered adapter by canonical name."""
    return _REGISTRY.get(name)


def available_adapters() -> list[str]:
    """List names of all currently registered adapters."""
    return _REGISTRY.names()


def adapters_for_target(target: str) -> list[VendorDialectAdapter]:
    """All registered adapters that claim to support ``target``."""
    return _REGISTRY.find_for_target(target)


def reset_registry() -> None:
    """Testing helper: wipe the registry."""
    _REGISTRY.reset()


__all__ = [
    "VendorAdapterRegistry",
    "adapters_for_target",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "reset_registry",
]
