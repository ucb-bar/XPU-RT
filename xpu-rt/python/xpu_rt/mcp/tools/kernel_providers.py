"""M-62 — MCP tools for kernel-provider discovery + user-kernel indexing.

Three tools surface the registry + indexer to a Claude Code agent:

- ``compgen_list_kernel_providers(target?)`` — enumerate every provider
  the registry would consider applicable for a target.
- ``compgen_describe_kernel_provider(provider_id)`` — return manifest
  + metadata for one provider; for user-path providers, includes the
  indexed signature.
- ``compgen_discover_user_kernels(path)`` — walk a user-supplied
  directory, validate every ``kernel_manifest.yaml``, and persist the
  derived index under ``.compgen/user_kernel_index/``.

The tools are read-only over the registry; ``discover_user_kernels``
is the only write — and it only writes under
``.compgen/user_kernel_index/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compgen_list_kernel_providers(
    *,
    target: str | None = None,
) -> dict[str, Any]:
    """Enumerate the registered kernel providers.

    Args:
        target: Optional ``hardware_envelope.target_name`` filter
            (e.g. ``"host_cpu"``). When provided, only providers whose
            ``applicable_targets`` includes the target (or is wildcard)
            are listed.

    Returns:
        ``{ok, providers: [{provider_id, source, priority, applicable_targets,
        applicable_archetypes, summary}]}``.
    """
    try:
        from compgen.kernels.registry import default_registry

        reg = default_registry()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    rows: list[dict[str, Any]] = []
    for p in reg._providers:
        applicable_targets = tuple(getattr(p, "applicable_targets", ()) or ())
        applicable_archetypes = tuple(getattr(p, "applicable_archetypes", ()) or ())
        priority = int(getattr(p, "priority", 0))
        source = str(getattr(p, "_compgen_source", "unknown"))
        # Filter by target when requested.
        if target and applicable_targets and target not in applicable_targets:
            continue
        rows.append(
            {
                "provider_id": p.name,
                "source": source,
                "priority": priority,
                "applicable_targets": list(applicable_targets),
                "applicable_archetypes": list(applicable_archetypes),
                "summary": _provider_summary(p),
            }
        )
    rows.sort(key=lambda r: (-r["priority"], r["provider_id"]))
    return {"ok": True, "providers": rows}


def compgen_describe_kernel_provider(*, provider_id: str) -> dict[str, Any]:
    """Read-only deep view of one provider.

    For ``UserKernelProvider`` the response includes the indexed
    manifests so the agent can see exactly which user kernels are on
    deck.
    """
    try:
        from compgen.kernels.registry import default_registry

        reg = default_registry()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    for p in reg._providers:
        if p.name != provider_id:
            continue
        body: dict[str, Any] = {
            "ok": True,
            "provider_id": p.name,
            "source": str(getattr(p, "_compgen_source", "unknown")),
            "priority": int(getattr(p, "priority", 0)),
            "applicable_targets": list(getattr(p, "applicable_targets", ()) or ()),
            "applicable_archetypes": list(getattr(p, "applicable_archetypes", ()) or ()),
            "summary": _provider_summary(p),
        }
        # M-62 — surface user-path index entries when the provider is
        # the user-space kind.
        entries = getattr(p, "_entries", None)
        if entries:
            body["indexed_kernels"] = [
                {
                    "index_id": e.index_id,
                    "source_dir": e.source_dir,
                    "op_name": e.manifest.op_name,
                    "archetype": e.manifest.archetype,
                    "target_name": e.manifest.target_name,
                    "language": e.manifest.language,
                    "entry_symbol": e.manifest.entry_symbol,
                    "indexed_at_utc": e.indexed_at_utc,
                    "perf_priors": dict(e.manifest.perf_priors or {}),
                }
                for e in entries
            ]
        return body
    return {
        "ok": False,
        "error": f"provider {provider_id!r} not found",
        "provider_id": provider_id,
    }


def compgen_discover_user_kernels(*, path: str) -> dict[str, Any]:
    """Walk a user-supplied directory + populate the index.

    Args:
        path: Filesystem path to a directory containing one or more
            ``kernel_manifest.yaml`` files (each with a sibling kernel
            source). Paths are resolved relative to the current
            working directory.

    Returns:
        ``{ok, indexed_count, manifests_written, errors, registry_path,
        index_root}``. ``errors`` lists per-manifest validation
        failures with typed ``error_kind`` so the agent can fix the
        offending manifest and re-discover.
    """
    try:
        from compgen.kernels.user_kernel_index import (
            default_index_root,
            reindex,
        )

        search_path = Path(path).resolve()
        if not search_path.exists():
            return {
                "ok": False,
                "error": f"path {str(search_path)!r} does not exist",
                "path": str(search_path),
            }
        index_root = default_index_root()
        result = reindex(search_path=search_path, index_root=index_root)
        return {
            "ok": True,
            "path": str(search_path),
            "index_root": str(index_root),
            **result,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": path,
        }


def _provider_summary(p: Any) -> str:
    """One-line summary of a provider for list views."""
    docstring = (getattr(p, "__doc__", None) or "").strip().splitlines()
    if docstring:
        return docstring[0].strip()
    cls_doc = (type(p).__doc__ or "").strip().splitlines()
    if cls_doc:
        return cls_doc[0].strip()
    return type(p).__name__


__all__ = [
    "compgen_describe_kernel_provider",
    "compgen_discover_user_kernels",
    "compgen_list_kernel_providers",
]
