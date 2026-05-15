"""Target registry — the discovery + introspection surface.

The hierarchy `targets/{class}/{vendor}/{arch}/` is a filesystem
layout. The registry is the in-process API that mirrors it:

    >>> from compgen.targets.registry import registry
    >>> reg = registry()
    >>> reg.classes()
    ('cpu', 'gpu', 'tpu')
    >>> reg.vendors('gpu')
    ('amd', 'intel', 'nvidia')
    >>> reg.arches('gpu', 'nvidia')
    ('ampere', 'blackwell', 'hopper')
    >>> pkg = reg.get('gpu.nvidia.blackwell')
    >>> pkg.body_emitter
    <NvidiaBlackwellBodyEmitter ...>

Three registration paths land in the same registry:

1. **In-tree**: a target package's ``__init__.py`` calls
   :func:`register_target` at import time. The package is loaded
   when the module is first touched (lazy).
2. **Entry-point**: third-party packages declare a
   ``compgen.runtime.lowerings`` or ``compgen.targets`` entry-point
   in their ``pyproject.toml``; :func:`discover_entry_points`
   loads them at probe-device time.
3. **MCP-driven**: the ``compgen_register_target`` MCP tool calls
   :func:`register_target` directly with adapter callables — the
   user's agent can extend the registry at session scope without
   editing source.

All three produce :class:`TargetPackage` entries with the same
shape, so universal compile / dispatch paths can't tell which
registration path produced a given target.

The registry is the agentic-compilation answer to "easy to explore":

- ``reg.tree()`` returns a nested dict for the whole hierarchy.
- ``reg.find(predicate)`` filters by capability.
- ``reg.describe('gpu.nvidia.blackwell')`` returns the README +
  rationale + adapter classes for an audit query.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetPackage:
    """One registered target — class.vendor.arch tuple + adapters.

    Adapters are the four Protocols from the class-level
    ``contracts.py`` (or CPU/TPU equivalents). Universal modules
    consume only these adapters; vendor-specific types stay
    inside the package.

    Attributes:
        target_class: Top-level family (``"gpu"``, ``"cpu"``, ``"tpu"``).
        vendor: Vendor under the class (``"nvidia"``, ``"amd"``).
        arch: Arch under the vendor (``"blackwell"``, ``"hopper"``).
            Empty string means "vendor-common" (any-arch within
            the vendor).
        target_id: ``"{class}.{vendor}.{arch}"`` — canonical
            dotted path. Empty arch means ``"{class}.{vendor}"``.
        probe: Compile-time hardware/library probe.
        body_emitter: Per-op kernel body emitter.
        runtime: JIT compile + dispatch.
        cost_model: TFLOPS / overhead / launch numbers for the
            roofline predictor.
        rationale: Human-readable description for the audit query.
        registration_path: ``"in_tree"`` | ``"entry_point"`` |
            ``"mcp"``. Surfaces in audit queries so the agent
            knows where the target came from.
        metadata: Per-target free-form data (paper references,
            release dates, supported dtypes etc.). Vendor-defined.
    """

    target_class: str
    vendor: str
    arch: str
    probe: Any
    body_emitter: Any
    runtime: Any
    cost_model: Any
    rationale: str = ""
    registration_path: str = "in_tree"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_id(self) -> str:
        if self.arch:
            return f"{self.target_class}.{self.vendor}.{self.arch}"
        return f"{self.target_class}.{self.vendor}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_class": self.target_class,
            "vendor": self.vendor,
            "arch": self.arch,
            "rationale": self.rationale,
            "registration_path": self.registration_path,
            "metadata": dict(self.metadata),
            "adapters": {
                "probe": _typename(self.probe),
                "body_emitter": _typename(self.body_emitter),
                "runtime": _typename(self.runtime),
                "cost_model": _typename(self.cost_model),
            },
        }


def _typename(obj: Any) -> str:
    if obj is None:
        return "None"
    return f"{type(obj).__module__}.{type(obj).__name__}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class _TargetRegistry:
    """Process-wide target registry. Constructed once via
    :func:`registry`; tests reset via :func:`reset`."""

    def __init__(self) -> None:
        # target_id → TargetPackage. Invariant: target_id matches
        # ``f"{class}.{vendor}.{arch}"`` (or ``f"{class}.{vendor}"``
        # for vendor-common entries).
        self._by_id: dict[str, TargetPackage] = {}

    # ----- Registration -----

    def register(self, pkg: TargetPackage) -> None:
        """Add a TargetPackage. Re-registering the same target_id
        replaces the previous entry — the agent uses this to
        override an in-tree default at session scope."""
        self._by_id[pkg.target_id] = pkg

    def unregister(self, target_id: str) -> None:
        self._by_id.pop(target_id, None)

    def reset(self) -> None:
        """Clear all entries — for tests."""
        self._by_id.clear()

    # ----- Lookup -----

    def get(self, target_id: str) -> TargetPackage | None:
        """Return the package for ``"gpu.nvidia.blackwell"`` etc.,
        or None when not registered. Falls back to the
        vendor-common entry (``"gpu.nvidia"``) when the arch-leaf
        isn't registered — handy for vendor-shared adapters."""
        pkg = self._by_id.get(target_id)
        if pkg is not None:
            return pkg
        # Fallback to vendor-common.
        parts = target_id.split(".")
        if len(parts) == 3:
            vendor_id = ".".join(parts[:2])
            return self._by_id.get(vendor_id)
        return None

    def all(self) -> tuple[TargetPackage, ...]:
        """Every registered package, sorted by target_id."""
        return tuple(sorted(self._by_id.values(), key=lambda p: p.target_id))

    # ----- Navigation -----

    def classes(self) -> tuple[str, ...]:
        """All registered top-level classes (``"gpu"``, ``"cpu"``,
        ``"tpu"``, ...). Sorted for determinism."""
        return tuple(sorted({p.target_class for p in self._by_id.values()}))

    def vendors(self, target_class: str) -> tuple[str, ...]:
        """All vendors registered under ``target_class``."""
        return tuple(sorted({p.vendor for p in self._by_id.values() if p.target_class == target_class}))

    def arches(self, target_class: str, vendor: str) -> tuple[str, ...]:
        """All arch-leaves registered under ``{class}/{vendor}/``.
        Excludes the vendor-common entry (which has empty arch)."""
        return tuple(
            sorted(
                {
                    p.arch
                    for p in self._by_id.values()
                    if p.target_class == target_class and p.vendor == vendor and p.arch
                }
            )
        )

    def tree(self) -> dict[str, dict[str, list[str]]]:
        """Nested dict view of the whole hierarchy. Useful for the
        agent's "show me what's available" query.

        ``{
            "gpu": {
                "nvidia": ["blackwell", "hopper", "ampere"],
                "amd": [],
            },
            "cpu": {
                "x86": [],
                "arm": [],
            },
        }``
        """
        out: dict[str, dict[str, list[str]]] = {}
        for cls in self.classes():
            out[cls] = {}
            for vendor in self.vendors(cls):
                out[cls][vendor] = list(self.arches(cls, vendor))
        return out

    # ----- Filtering -----

    def find(self, predicate: Callable[[TargetPackage], bool]) -> tuple[TargetPackage, ...]:
        """Return packages where ``predicate(pkg)`` is True.
        Caller-defined filter — e.g. ``find(lambda p: p.vendor ==
        "nvidia" and p.metadata.get("supports_tensor_cores"))``."""
        return tuple(p for p in self.all() if predicate(p))

    def describe(self, target_id: str) -> dict[str, Any]:
        """Audit-query payload for one target. Returns the
        ``to_dict()`` form when the target is registered, an empty
        dict when not.

        The agent's "why was X picked?" / "what does X do?"
        queries land here — same shape as
        :meth:`BackendChoice.to_dict` so the surfaces compose."""
        pkg = self.get(target_id)
        if pkg is None:
            return {}
        return pkg.to_dict()


# Process-wide singleton.
_REGISTRY = _TargetRegistry()


def registry() -> _TargetRegistry:
    """Return the process-wide :class:`_TargetRegistry`."""
    return _REGISTRY


def reset() -> None:
    """Clear all registrations. For tests."""
    _REGISTRY.reset()


# ---------------------------------------------------------------------------
# Programmatic registration API (used by in-tree packages + MCP tool)
# ---------------------------------------------------------------------------


def register_target(
    *,
    target_class: str,
    vendor: str,
    arch: str = "",
    probe: Any = None,
    body_emitter: Any = None,
    runtime: Any = None,
    cost_model: Any = None,
    rationale: str = "",
    registration_path: str = "in_tree",
    metadata: dict[str, Any] | None = None,
) -> TargetPackage:
    """Register a target package into the process-wide registry.

    All five adapters (probe, body_emitter, runtime, cost_model,
    plus rationale) are optional at registration time so a
    placeholder package can register early and fill in later. The
    universal compile/dispatch paths surface a typed error when an
    unconfigured adapter is invoked.

    Three callers:

    1. **In-tree**: a package's ``__init__.py`` calls this at
       import time, with concrete adapter instances. Default
       ``registration_path="in_tree"``.
    2. **Entry-point**: :func:`discover_entry_points` calls this
       with adapters loaded from a third-party wheel.
       ``registration_path="entry_point"``.
    3. **MCP tool**: ``compgen_register_target`` calls this with
       adapters built from user-supplied callables.
       ``registration_path="mcp"``.

    Returns the registered :class:`TargetPackage` for chaining.
    """
    pkg = TargetPackage(
        target_class=target_class,
        vendor=vendor,
        arch=arch,
        probe=probe,
        body_emitter=body_emitter,
        runtime=runtime,
        cost_model=cost_model,
        rationale=rationale,
        registration_path=registration_path,
        metadata=dict(metadata or {}),
    )
    _REGISTRY.register(pkg)
    return pkg


def discover_entry_points(*, group: str = "compgen.targets") -> int:
    """Load third-party target packages declared as entry points.

    Each entry-point's loaded value must be a zero-arg callable
    that registers itself via :func:`register_target` (typically
    its ``__init__.py``'s side effect). Returns the number of
    entry points successfully loaded.

    Idempotent — re-running picks up newly-installed packages
    without duplicating already-registered ones.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return 0

    loaded = 0
    try:
        eps = entry_points().select(group=group)
    except (AttributeError, KeyError):
        try:
            eps = entry_points().get(group, [])
        except Exception:  # noqa: BLE001
            return 0

    for ep in eps:
        try:
            obj = ep.load()
            if callable(obj):
                obj()
            loaded += 1
        except Exception:  # noqa: BLE001
            # Surface in logs but don't crash the registry —
            # one broken third-party package shouldn't kill
            # discovery of the rest.
            continue
    return loaded
