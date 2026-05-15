"""Plugin discovery via Python entry points.

User packages can extend CompGen without forking the repo by exposing
entry points in their ``pyproject.toml``. CompGen discovers them at
import time and registers them into the appropriate runtime registry.

Supported entry-point groups:

  * ``compgen.kernels.providers``      — user ``KernelProvider`` impls
  * ``compgen.transforms.decompositions`` — user FX-decomp functions
  * ``compgen.kernels.fusion_rules``   — user fusion-rule predicates
  * ``compgen.targets.backends``       — user ``TargetBackendProtocol`` impls
  * ``compgen.kernels.contracts``      — user ``KernelContractV3`` factories

Example user-side ``pyproject.toml``::

    [project.entry-points."compgen.kernels.providers"]
    my_custom_kernel = "mypkg.providers:MyKernelProvider"

CompGen calls ``compgen.plugins.discover_all()`` once at startup
(via the agent loop's bootstrap) — the result is a populated registry
the rest of the system reads.

Discovery is best-effort: a failing plugin logs a warning and is
skipped, never crashes the host. Validation per group ensures the
loaded object satisfies the expected protocol before registration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

log = logging.getLogger(__name__)


# Entry-point group names — keep in sync with the docstring above.
GROUP_KERNEL_PROVIDERS = "compgen.kernels.providers"
GROUP_DECOMPOSITIONS = "compgen.transforms.decompositions"
GROUP_FUSION_RULES = "compgen.kernels.fusion_rules"
GROUP_TARGET_BACKENDS = "compgen.targets.backends"
GROUP_KERNEL_CONTRACTS = "compgen.kernels.contracts"
GROUP_MCP_TOOLS = "compgen.mcp.tools"
# User-supplied pattern matchers + custom MLIR-dialect lowerings.
# The agentic-compilation surface for "I have a custom dialect"
# (cuda-tile, etc.). Registered functions are tried by
# :func:`compgen.runtime.lowering.lower_torch_to_megakernel` before
# the built-in diamond/FFN matchers, so a user can plug in a
# domain-specific pattern without forking the matcher list.
GROUP_LOWERINGS = "compgen.runtime.lowerings"


KNOWN_GROUPS = (
    GROUP_KERNEL_PROVIDERS,
    GROUP_DECOMPOSITIONS,
    GROUP_FUSION_RULES,
    GROUP_TARGET_BACKENDS,
    GROUP_KERNEL_CONTRACTS,
    GROUP_MCP_TOOLS,
    GROUP_LOWERINGS,
)


# ---------------------------------------------------------------------------
# Validators per group — invariants the discovered object must satisfy
# ---------------------------------------------------------------------------


def _validate_kernel_provider(obj: Any) -> tuple[bool, str]:
    """Must look like a ``KernelProvider`` (Protocol)."""
    required = ("name", "accepts_contract", "search", "export_knowledge")
    missing = [m for m in required if not hasattr(obj, m)]
    if missing:
        return (False, f"missing KernelProvider methods: {missing}")
    return (True, "")


def _validate_decomposition(obj: Any) -> tuple[bool, str]:
    """Must be callable: ``(operands, meta, node_name) -> DecompResult``."""
    if not callable(obj):
        return (False, "decomposition entry must be callable")
    return (True, "")


def _validate_fusion_rule(obj: Any) -> tuple[bool, str]:
    """Must be callable: ``(producer_v3, consumer_v3) -> bool | FusionVerdict``."""
    if not callable(obj):
        return (False, "fusion-rule entry must be callable")
    return (True, "")


def _validate_target_backend(obj: Any) -> tuple[bool, str]:
    required = ("supports_target", "get_options", "get_compilation_stages", "compile_stage", "validate")
    missing = [m for m in required if not hasattr(obj, m)]
    if missing:
        return (False, f"missing TargetBackend methods: {missing}")
    return (True, "")


def _validate_kernel_contract_factory(obj: Any) -> tuple[bool, str]:
    if not callable(obj):
        return (False, "kernel-contract factory must be callable")
    return (True, "")


def _validate_lowering(obj: Any) -> tuple[bool, str]:
    """User-supplied pattern matcher / MLIR-dialect lowering.

    Contract: ``obj(model, sample_inputs, *, backend_choice=None) ->
    LoweringResult``. The function may raise
    :class:`compgen.runtime.lowering.UnsupportedShape` to indicate
    "this pattern doesn't match", letting the matcher cascade
    continue to the next registered lowering or the built-ins.
    """
    if not callable(obj):
        return (False, "lowering entry must be callable")
    import inspect

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        # Built-in or C-extension callable; trust it.
        return (True, "")
    params = list(sig.parameters.values())
    # Need at least 2 positional args (model, sample_inputs).
    pos_count = sum(
        1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    )
    if pos_count < 2:
        return (
            False,
            f"lowering must accept (model, sample_inputs); got {pos_count} positional",
        )
    return (True, "")


def _validate_mcp_tool(obj: Any) -> tuple[bool, str]:
    """Each entry must be a tool-dict or iterable of tool-dicts.

    A tool-dict has keys ``name``, ``description``, ``input_schema``,
    ``handler`` (callable), and ``phase``. Shape mirrors what the
    in-tree ``compgen.mcp.tools`` modules already export.
    """
    items = obj if isinstance(obj, (list, tuple)) else [obj]
    required = ("name", "description", "input_schema", "handler", "phase")
    for item in items:
        if not isinstance(item, dict):
            return (False, f"mcp tool entry must be a dict, got {type(item).__name__}")
        missing = [k for k in required if k not in item]
        if missing:
            return (False, f"mcp tool dict missing keys: {missing}")
        if not callable(item.get("handler")):
            return (False, "mcp tool 'handler' must be callable")
    return (True, "")


_VALIDATORS: dict[str, Callable[[Any], tuple[bool, str]]] = {
    GROUP_KERNEL_PROVIDERS: _validate_kernel_provider,
    GROUP_DECOMPOSITIONS: _validate_decomposition,
    GROUP_FUSION_RULES: _validate_fusion_rule,
    GROUP_TARGET_BACKENDS: _validate_target_backend,
    GROUP_KERNEL_CONTRACTS: _validate_kernel_contract_factory,
    GROUP_MCP_TOOLS: _validate_mcp_tool,
    GROUP_LOWERINGS: _validate_lowering,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class LoadedPlugin:
    group: str
    name: str  # entry-point name (user-chosen)
    object: Any  # the loaded callable / class / instance
    distribution: str = ""  # package that supplied it
    notes: str = ""


@dataclass
class ExtensionRegistry:
    """In-process registry of loaded plugins, keyed by entry-point group."""

    by_group: dict[str, list[LoadedPlugin]] = field(default_factory=dict)
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    # ^ (group, name, reason) for each plugin that didn't load

    def add(self, plugin: LoadedPlugin) -> None:
        self.by_group.setdefault(plugin.group, []).append(plugin)

    def get(self, group: str) -> list[LoadedPlugin]:
        return list(self.by_group.get(group, ()))

    def names_in(self, group: str) -> list[str]:
        return [p.name for p in self.by_group.get(group, ())]

    def total_loaded(self) -> int:
        return sum(len(v) for v in self.by_group.values())


# Process-wide registry singleton.
_REGISTRY = ExtensionRegistry()


def registry() -> ExtensionRegistry:
    """Return the process-wide ExtensionRegistry."""
    return _REGISTRY


def reset_registry() -> None:
    """Clear all loaded plugins. Used by tests."""
    global _REGISTRY
    _REGISTRY = ExtensionRegistry()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _try_load(ep: EntryPoint, group: str) -> LoadedPlugin | None:
    try:
        obj = ep.load()
    except Exception as exc:  # noqa: BLE001
        _REGISTRY.failures.append((group, ep.name, f"load: {type(exc).__name__}: {exc}"))
        log.warning(
            "extension.load_failed",
            extra={
                "group": group,
                # ``name`` collides with LogRecord's reserved attribute
                # and turns the original load failure into a noisy
                # KeyError that masks what really went wrong.
                "plugin_name": ep.name,
                "error": str(exc),
            },
        )
        return None

    validator = _VALIDATORS.get(group)
    if validator is not None:
        ok, reason = validator(obj)
        if not ok:
            _REGISTRY.failures.append((group, ep.name, f"validate: {reason}"))
            log.warning(
                "extension.validate_failed",
                extra={
                    "group": group,
                    "plugin_name": ep.name,
                    "reason": reason,
                },
            )
            return None

    dist = ""
    try:
        dist = ep.dist.name if ep.dist is not None else ""  # type: ignore[union-attr]
    except Exception:
        pass

    return LoadedPlugin(group=group, name=ep.name, object=obj, distribution=dist)


def discover_all() -> ExtensionRegistry:
    """Scan installed packages for CompGen entry points + populate the registry.

    Idempotent: re-running picks up newly-installed packages but doesn't
    duplicate already-registered plugins (matched by ``(group, name)``).
    Returns the populated registry.
    """
    eps = entry_points()
    for group in KNOWN_GROUPS:
        try:
            group_eps = eps.select(group=group)
        except AttributeError:
            # Older importlib.metadata API
            group_eps = eps.get(group, [])
        existing_names = set(_REGISTRY.names_in(group))
        for ep in group_eps:
            if ep.name in existing_names:
                continue
            plugin = _try_load(ep, group)
            if plugin is not None:
                _REGISTRY.add(plugin)
    return _REGISTRY


def discover_group(group: str) -> list[LoadedPlugin]:
    """Convenience: scan one group + return what loaded."""
    if group not in KNOWN_GROUPS:
        raise ValueError(f"unknown extension group: {group!r} (known: {KNOWN_GROUPS})")
    discover_all()
    return _REGISTRY.get(group)


# ---------------------------------------------------------------------------
# Manual registration (for testing + for when you want to register
# a plugin programmatically without going through entry points)
# ---------------------------------------------------------------------------


def register(group: str, name: str, obj: Any) -> LoadedPlugin:
    """Register ``obj`` under ``(group, name)`` after validating it.

    Raises ValueError if the object fails the group's validator.
    """
    if group not in KNOWN_GROUPS:
        raise ValueError(f"unknown extension group: {group!r}")
    validator = _VALIDATORS.get(group)
    if validator is not None:
        ok, reason = validator(obj)
        if not ok:
            raise ValueError(f"validation failed for {name!r} in {group!r}: {reason}")
    plugin = LoadedPlugin(group=group, name=name, object=obj, notes="manual-register")
    _REGISTRY.add(plugin)
    return plugin


# ---------------------------------------------------------------------------
# Unified discovery facade
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryReport:
    """Snapshot of what every discovery path produced in one pass.

    Returned by :func:`discover_everything` so the CLI + MCP server can
    surface a single consolidated view of "what extensions is this
    process aware of" without knowing about the two underlying registries.
    """

    entry_point_plugins: dict[str, list[str]] = field(default_factory=dict)
    vendor_dialects: list[str] = field(default_factory=list)
    user_space_tools: list[str] = field(default_factory=list)
    user_space_slots: list[str] = field(default_factory=list)
    user_space_root: str = ""
    user_space_errors: list[tuple[str, str]] = field(default_factory=list)
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def total(self) -> int:
        return (
            sum(len(v) for v in self.entry_point_plugins.values())
            + len(self.vendor_dialects)
            + len(self.user_space_tools)
            + len(self.user_space_slots)
        )


def discover_everything() -> DiscoveryReport:
    """Trigger every discovery path CompGen knows about and summarise it.

    Combines:

    1. Entry-point plugins across :data:`KNOWN_GROUPS` (this module).
    2. Vendor dialect adapters (``compgen.vendor_dialects`` entry points).
    3. User-space ``~/.compgen/extensions/*.py`` tools + invent-slots
       (via :mod:`compgen.agent.extensions.local_loader`; idempotent —
       the LLM registry re-uses whatever it already loaded).

    Returns a :class:`DiscoveryReport`. Safe to call repeatedly; each
    underlying registry is idempotent.
    """
    discover_all()
    report = DiscoveryReport(
        entry_point_plugins={g: _REGISTRY.names_in(g) for g in KNOWN_GROUPS},
        failures=list(_REGISTRY.failures),
    )

    try:
        from compgen.extensions.vendor_dialect.registry import available_adapters

        report.vendor_dialects = list(available_adapters())
    except Exception as exc:  # noqa: BLE001
        log.warning("plugins.vendor_dialect.error", extra={"error": str(exc)})

    try:
        from compgen.agent.extensions.local_loader import load_local_extensions
        from compgen.llm.registry import get_registry

        registry_ = get_registry()
        result = load_local_extensions(registry_)
        report.user_space_root = str(result.root)
        report.user_space_tools = list(result.tool_names())
        report.user_space_slots = list(result.slot_names())
        report.user_space_errors = [(str(e.path), e.error or "unknown") for e in result.errors()]
    except Exception as exc:  # noqa: BLE001
        log.warning("plugins.user_space.error", extra={"error": str(exc)})

    return report


__all__ = [
    "DiscoveryReport",
    "ExtensionRegistry",
    "GROUP_DECOMPOSITIONS",
    "GROUP_FUSION_RULES",
    "GROUP_KERNEL_CONTRACTS",
    "GROUP_KERNEL_PROVIDERS",
    "GROUP_MCP_TOOLS",
    "GROUP_TARGET_BACKENDS",
    "KNOWN_GROUPS",
    "LoadedPlugin",
    "discover_all",
    "discover_everything",
    "discover_group",
    "register",
    "registry",
    "reset_registry",
]
