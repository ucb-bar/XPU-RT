"""First-party reference adapters bundled with CompGen.

Third-party adapters (e.g. ``compgen_cuda_tile`` on a Blackwell box)
discover via the ``compgen.vendor_dialects`` entry-point group. The
adapters here are **reference implementations** that ship with the
CompGen wheel itself — pip-installing CompGen gets them without any
extra dependency. They serve three roles:

1. **Reference for users authoring third-party adapters** — read the
   source to see the canonical shape of ``lower_payload`` /
   ``emit_artifact``.
2. **Anchor for unit tests** — the lowering produces deterministic
   MLIR text that can be regression-tested without a vendor toolchain
   on PATH. Toolchain-driven gates are opt-in via fixture markers.
3. **PyPI surface guarantee** — ``compgen_compile_torch_model_with_vendor``
   has at least one working adapter to point at out of the box.

Reference adapters do NOT auto-register on import. Callers opt in via
:func:`register_builtin_adapter`. This keeps the registry minimal for
users who don't need the reference.
"""

from __future__ import annotations

from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter
from compgen.extensions.vendor_dialect.builtins.cuda_tile import (
    make_adapter as make_cuda_tile_adapter,
)
from compgen.extensions.vendor_dialect.registry import register_adapter

_BUILTIN_FACTORIES: dict[str, callable] = {
    "cuda_tile": make_cuda_tile_adapter,
}


def list_builtin_adapters() -> list[str]:
    """Names of reference adapters bundled with CompGen."""
    return sorted(_BUILTIN_FACTORIES)


def make_builtin_adapter(name: str) -> VendorDialectAdapter:
    """Construct a fresh reference adapter by name.

    Raises:
        KeyError: ``name`` is not a known reference adapter.
    """
    try:
        factory = _BUILTIN_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown builtin adapter {name!r}; available: {list_builtin_adapters()}") from exc
    return factory()


def register_builtin_adapter(name: str, *, replace: bool = False) -> VendorDialectAdapter:
    """Construct and register a reference adapter in the process registry.

    Returns the registered adapter so callers can immediately compile.
    """
    adapter = make_builtin_adapter(name)
    register_adapter(adapter, replace=replace)
    return adapter


__all__ = [
    "list_builtin_adapters",
    "make_builtin_adapter",
    "register_builtin_adapter",
]
