"""Local (``~/.compgen/extensions/``) extension loader.

Discovers every ``*.py`` file in ``~/.compgen/extensions/`` (overridable
via ``COMPGEN_EXTENSIONS_DIR``) and gives it a chance to register
tools or invent-slots against the global registry.

Loading contract — each file may:

1. Define ``def register(registry):`` which will be called with the
   live :class:`~compgen.llm.registry.Registry`.
2. *Or* define module-level constants ``TOOL`` / ``TOOLS`` (iterable
   of :class:`~compgen.llm.registry.Tool`) and / or ``SLOT`` /
   ``SLOTS`` (iterable of :class:`~compgen.llm.registry.InventSlot`),
   which will be auto-registered.

Failures never raise — one broken file must not prevent the registry
from coming up. Instead, the failure is captured in the returned
:class:`LocalExtensionLoadResult` so callers (including the CLI and
the MCP server) can surface it to the user.

Loading is idempotent: a state file
``~/.compgen/extensions/_state.json`` records which (module, registry-
epoch) pairs have already been loaded, and repeat calls are no-ops
until the registry is :meth:`~compgen.llm.registry.Registry.clear`-ed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog

if TYPE_CHECKING:   # pragma: no cover
    from compgen.llm.registry import InventSlot, Registry, Tool

log = structlog.get_logger()


DEFAULT_ROOT = Path("~/.compgen/extensions").expanduser()
STATE_FILENAME = "_state.json"
ENV_VAR = "COMPGEN_EXTENSIONS_DIR"


@dataclass(frozen=True)
class LocalExtension:
    """One user-authored extension file that was processed."""

    path: Path
    module_name: str
    tools_registered: tuple[str, ...] = ()
    slots_registered: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class LocalExtensionLoadResult:
    """Summary of one call to :func:`load_local_extensions`."""

    root: Path
    extensions: list[LocalExtension] = field(default_factory=list)

    def ok(self) -> bool:
        return all(e.ok for e in self.extensions)

    def errors(self) -> list[LocalExtension]:
        return [e for e in self.extensions if not e.ok]

    def tool_names(self) -> list[str]:
        return [t for e in self.extensions for t in e.tools_registered]

    def slot_names(self) -> list[str]:
        return [s for e in self.extensions for s in e.slots_registered]


# ---------------------------------------------------------------------------
# Idempotence state
# ---------------------------------------------------------------------------


def _state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def _load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:   # noqa: BLE001
        return {}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        _state_path(root).write_text(json.dumps(state, indent=2, default=str))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def _ext_root(override: Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_ROOT


def _discover(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        p for p in root.glob("*.py")
        if p.is_file() and not p.name.startswith("_")
    )


def _import_file(path: Path) -> Any:
    """Import ``path`` as a throwaway module under ``compgen_ext.<stem>``."""
    mod_name = f"compgen_ext.{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _iter_collected(
    module: Any, single_attr: str, plural_attr: str,
) -> list[Any]:
    items: list[Any] = []
    single = getattr(module, single_attr, None)
    plural = getattr(module, plural_attr, None)
    if single is not None:
        items.append(single)
    if plural is not None:
        try:
            items.extend(list(plural))
        except TypeError:
            pass
    return items


def _register_from_module(
    module: Any, registry: "Registry",
) -> tuple[list[str], list[str]]:
    """Return (tool_names, slot_names) registered from ``module``."""
    from compgen.llm.registry import InventSlot, Tool

    tool_names: list[str] = []
    slot_names: list[str] = []

    reg_fn: Callable[[Any], Any] | None = getattr(module, "register", None)
    if callable(reg_fn):
        before_tools = {
            t.name for t in registry.list_tools()
        }
        before_slots = {
            s.name for s in registry.list_invent_slots()
        }
        reg_fn(registry)
        after_tools = {t.name for t in registry.list_tools()}
        after_slots = {s.name for s in registry.list_invent_slots()}
        tool_names.extend(sorted(after_tools - before_tools))
        slot_names.extend(sorted(after_slots - before_slots))
        return tool_names, slot_names

    # Fallback: module-level TOOL/TOOLS/SLOT/SLOTS constants.
    for item in _iter_collected(module, "TOOL", "TOOLS"):
        if isinstance(item, Tool):
            registry.register_tool(item)
            tool_names.append(item.name)
    for item in _iter_collected(module, "SLOT", "SLOTS"):
        if isinstance(item, InventSlot):
            registry.register_invent_slot(item)
            slot_names.append(item.name)

    return tool_names, slot_names


def load_local_extensions(
    registry: "Registry",
    root: Path | str | None = None,
    *,
    force: bool = False,
) -> LocalExtensionLoadResult:
    """Load every ``~/.compgen/extensions/*.py`` into ``registry``.

    Args:
        registry: Live registry to mutate.
        root: Directory to scan; defaults to ``$COMPGEN_EXTENSIONS_DIR``
            or ``~/.compgen/extensions``.
        force: When True, reload even if already loaded this process.

    Returns:
        A :class:`LocalExtensionLoadResult` describing every file found
        and what each one registered (or the error it raised).
    """
    ext_root = _ext_root(root if isinstance(root, Path | type(None)) else Path(root))
    result = LocalExtensionLoadResult(root=ext_root)

    paths = _discover(ext_root)
    if not paths:
        return result

    state = _load_state(ext_root)
    loaded_already: set[str] = set(state.get("loaded_modules", []))

    new_loaded: list[str] = []
    for path in paths:
        mod_name = f"compgen_ext.{path.stem}"
        if mod_name in loaded_already and not force:
            # Already processed in a prior call; skip so we don't
            # double-register and hit the ValueError in the registry.
            continue
        try:
            module = _import_file(path)
            tools, slots = _register_from_module(module, registry)
            result.extensions.append(
                LocalExtension(
                    path=path,
                    module_name=mod_name,
                    tools_registered=tuple(tools),
                    slots_registered=tuple(slots),
                )
            )
            new_loaded.append(mod_name)
            log.info(
                "extensions.loaded",
                path=str(path),
                tools=len(tools),
                slots=len(slots),
            )
        except Exception as exc:   # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            result.extensions.append(
                LocalExtension(
                    path=path,
                    module_name=mod_name,
                    error=err,
                )
            )
            log.warning("extensions.load_failed", path=str(path), error=str(exc))

    # Update state file.
    if new_loaded:
        state["loaded_modules"] = sorted(set(loaded_already) | set(new_loaded))
        _save_state(ext_root, state)

    return result


def record_accepted_invocation(
    root: Path | str | None,
    slot_or_tool_name: str,
    invocation: dict[str, Any],
) -> None:
    """Append an accepted invocation to the per-root state file.

    Used by the driver to build up the history the P3 ``compgen
    contrib draft`` command turns into a regression test. Intentionally
    loose schema — callers write whatever JSON-serialisable payload
    reproduces the invocation.
    """
    ext_root = _ext_root(Path(root) if root is not None else None)
    state = _load_state(ext_root)
    accepted = state.setdefault("accepted_invocations", {})
    bucket = accepted.setdefault(slot_or_tool_name, [])
    bucket.append(invocation)
    # Cap at 32 per slot to keep the state file small.
    if len(bucket) > 32:
        del bucket[: len(bucket) - 32]
    _save_state(ext_root, state)


__all__ = [
    "DEFAULT_ROOT",
    "ENV_VAR",
    "LocalExtension",
    "LocalExtensionLoadResult",
    "load_local_extensions",
    "record_accepted_invocation",
]
