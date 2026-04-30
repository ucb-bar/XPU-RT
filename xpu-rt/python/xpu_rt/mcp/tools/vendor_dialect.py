"""MCP tools: scan / propose / scaffold / verify a vendor MLIR dialect,
plus discovery + invocation tools for already-registered vendor
adapters.

The first four tools (``scan_vendor_repo``, ``propose_vendor_spec``,
``scaffold_vendor_package``, ``verify_vendor_package``) are the
MCP-visible surface of the vendor integration agent. They are thin
wrappers over :mod:`compgen.extensions.vendor_dialect` and
:mod:`compgen.agent.vendor_integration`.

The two newer tools (``compgen_list_vendor_dialects``,
``compgen_compile_torch_model_with_vendor``) let a remote agent
discover and drive vendor adapters that were registered through the
``compgen.vendor_dialects`` entry-point group — i.e. user-space
packages installed via pip, with no CompGen fork required.

Each tool follows the repo-wide MCP convention:

* signature ``(sm: SessionManager, **kwargs) -> dict[str, Any]``
* returns a JSON-serialisable dict with an ``ok`` boolean and tool-
  specific payload
* never prints — errors come back in the dict

The tools intentionally do NOT require a CompGen session to be open.
Vendor integration happens *before* a workload is loaded: the user
points the MCP at a repo, reviews the proposed spec, approves, and
scaffolds a package that can later be used in a real session.
"""

from __future__ import annotations

import base64
import time
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.vendor_integration.explore import explore_vendor_repo
from compgen.agent.vendor_integration.propose_adapter import propose_adapter_layout
from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter
from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor
from compgen.extensions.vendor_dialect.registry import (
    available_adapters,
    get_adapter,
)
from compgen.extensions.vendor_dialect.scaffold import scaffold_package
from compgen.extensions.vendor_dialect.verify import verify_package
from compgen.mcp.session import SessionManager

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Tool handlers
# --------------------------------------------------------------------------- #


def scan_vendor_repo(
    sm: SessionManager,
    *,
    repo_path: str,
    target: str,
    workloads: list[str] | None = None,
) -> dict[str, Any]:
    """Scan a third-party MLIR dialect repo and propose a descriptor.

    This combines the deterministic scanner with a (LLM-optional)
    classifier and returns:

    * ``scan`` — the structured scanner summary
    * ``descriptor_yaml`` — the proposed :class:`VendorDialectDescriptor`
      serialised as YAML for human review
    * ``proposal`` — a :class:`ProposedAdapter` layout hint

    The caller gates on the YAML and calls
    :func:`scaffold_vendor_package` once satisfied.
    """
    del sm  # session-less tool
    workloads_t = tuple(workloads or ())
    explore = explore_vendor_repo(repo_path, target=target, workloads=workloads_t)
    proposal = propose_adapter_layout(explore.descriptor, workloads=workloads_t)
    return {
        "ok": True,
        "scan": explore.scan.summary(),
        "descriptor_yaml": explore.descriptor.to_yaml(),
        "descriptor": explore.descriptor.to_dict(),
        "proposal": {
            "rules": [asdict(r) for r in proposal.rules],
            "risks": proposal.risks,
            "verification_hooks": proposal.verification_hooks,
            "llm_used": proposal.llm_used,
        },
        "llm_used": explore.llm_used,
    }


def propose_vendor_spec(
    sm: SessionManager,
    *,
    repo_path: str,
    target: str,
    workloads: list[str] | None = None,
    package_name: str | None = None,
    vendor_name: str | None = None,
) -> dict[str, Any]:
    """Re-run classification with explicit overrides.

    Useful when the caller wants to pin the canonical vendor name or
    change the target without altering the scanner output. Equivalent to
    :func:`scan_vendor_repo` with extra overrides.
    """
    del sm
    explore = explore_vendor_repo(
        repo_path,
        target=target,
        workloads=tuple(workloads or ()),
        package_name=package_name,
        vendor_name=vendor_name,
    )
    return {
        "ok": True,
        "descriptor_yaml": explore.descriptor.to_yaml(),
        "descriptor": explore.descriptor.to_dict(),
        "llm_used": explore.llm_used,
    }


def scaffold_vendor_package(
    sm: SessionManager,
    *,
    descriptor_yaml: str | None = None,
    descriptor_path: str | None = None,
    out_dir: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render a user-space adapter package from a descriptor.

    Pass either ``descriptor_yaml`` (inline YAML) or ``descriptor_path``
    (an on-disk YAML file produced by :func:`scan_vendor_repo`).
    """
    del sm
    descriptor = _load_descriptor(descriptor_yaml, descriptor_path)
    out = Path(out_dir).expanduser().resolve()
    result = scaffold_package(descriptor, out, overwrite=overwrite)
    return {
        "ok": True,
        "package_dir": str(result.package_dir),
        "descriptor_path": str(result.descriptor_path),
        "files_written": [str(p) for p in result.files_written],
        "next_step": (
            f"pip install -e {result.package_dir} and call "
            f"compgen.api.compile_with_vendor(model, adapter=get_adapter('{descriptor.name}'), ...)"
        ),
    }


def verify_vendor_package(
    sm: SessionManager,
    *,
    package_dir: str,
    run_workload_gate: bool | None = None,
) -> dict[str, Any]:
    """Run the verification ladder against a scaffolded package."""
    del sm
    report = verify_package(package_dir, run_workload_gate=run_workload_gate)
    return {
        "ok": report.passed,
        "report": report.to_dict(),
    }


_ENTRY_POINT_GROUP = "compgen.vendor_dialects"


def compgen_list_vendor_dialects(sm: SessionManager) -> dict[str, Any]:
    """List every vendor dialect adapter discoverable in this process.

    Walks the ``compgen.vendor_dialects`` entry-point group and returns
    one record per advertised adapter. For each entry point we attempt
    to ``ep.load()`` and (if it's a callable factory) instantiate it,
    then surface the adapter's ``name``, ``target``, optional
    ``version`` attribute, and the result of ``capabilities()`` when
    that method exists.

    Failures are NOT raised — a broken adapter is reported with an
    ``error`` field so the agent can see what couldn't load. The tool
    always succeeds with a list (possibly empty).

    Returns:
        ``{"vendor_dialects": [{"name": ..., "target": ...,
            "version": ..., "capabilities": ... | None,
            "module": "<entry point value>", "error": ... | None}, ...]}``
    """
    del sm
    out: list[dict[str, Any]] = []
    try:
        eps = importlib_metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001
        log.warning("vendor_dialect.list.entry_points_error", error=str(exc))
        return {"vendor_dialects": []}

    for ep in eps:
        record: dict[str, Any] = {
            "name": ep.name,
            "target": None,
            "version": None,
            "capabilities": None,
            "module": getattr(ep, "value", None) or str(ep),
            "error": None,
        }
        try:
            factory = ep.load()
            adapter = factory() if callable(factory) else factory
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue

        if not isinstance(adapter, VendorDialectAdapter):
            record["error"] = (
                f"entry point {ep.name!r} did not yield a VendorDialectAdapter (got {type(adapter).__name__})"
            )
            out.append(record)
            continue

        # Canonical name comes from the adapter, not the entry point —
        # the registry resolves by adapter.name.
        try:
            record["name"] = adapter.name
            record["target"] = adapter.target
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue

        # Optional ``version`` — adapters may expose it as an attribute
        # on the instance or on the descriptor's extras dict.
        version = getattr(adapter, "version", None)
        if version is None:
            extras = getattr(getattr(adapter, "descriptor", None), "extras", None) or {}
            version = extras.get("version") if isinstance(extras, dict) else None
        record["version"] = str(version) if version is not None else None

        # Optional ``capabilities()`` — call when present, otherwise
        # leave as None.
        capabilities_method = getattr(adapter, "capabilities", None)
        if callable(capabilities_method):
            try:
                caps = capabilities_method()
                record["capabilities"] = caps if caps is None else dict(caps)
            except Exception as exc:  # noqa: BLE001
                record["capabilities"] = None
                record["error"] = f"capabilities() raised: {type(exc).__name__}: {exc}"

        out.append(record)

    # Per bridge #134: union with the in-process registry. The
    # entry-point walk above only sees pip-installed packages; the
    # ``register_adapter()`` API populates a process-local registry
    # that ``compile_with_vendor`` already consults via
    # ``get_adapter()``. Without this union, an agent that
    # programmatically registers an adapter (e.g. a test fixture or
    # a notebook session) sees ``compile`` succeed but ``list`` /
    # ``describe`` come up empty — agentic-compilation discovery
    # loop is broken.
    seen_names = {r["name"] for r in out if r.get("name")}
    for name in available_adapters():
        if name in seen_names:
            continue
        record: dict[str, Any] = {
            "name": name,
            "target": None,
            "version": None,
            "capabilities": None,
            "module": "<in-process>",
            "error": None,
        }
        try:
            adapter = get_adapter(name)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue
        try:
            record["target"] = adapter.target
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue
        version = getattr(adapter, "version", None)
        if version is None:
            extras = getattr(getattr(adapter, "descriptor", None), "extras", None) or {}
            version = extras.get("version") if isinstance(extras, dict) else None
        record["version"] = str(version) if version is not None else None
        capabilities_method = getattr(adapter, "capabilities", None)
        if callable(capabilities_method):
            try:
                caps = capabilities_method()
                record["capabilities"] = caps if caps is None else dict(caps)
            except Exception as exc:  # noqa: BLE001
                record["capabilities"] = None
                record["error"] = f"capabilities() raised: {type(exc).__name__}: {exc}"
        out.append(record)

    return {"vendor_dialects": out}


def compgen_describe_vendor_dialect(
    sm: SessionManager,
    *,
    vendor_name: str,
) -> dict[str, Any]:
    """Return a single vendor adapter's full descriptor + capabilities.

    Per bridge #129: pre-screening tool — the agent asks "given the
    workload I'm about to compile, can ``vendor_name`` lower it?"
    before committing to ``compgen_compile_torch_model_with_vendor``.

    Resolves ``vendor_name`` against the same
    ``compgen.vendor_dialects`` entry-point group as
    :func:`compgen_list_vendor_dialects` and returns the adapter's
    descriptor (target + extras) plus the result of its
    ``capabilities()`` method when implemented. Adapters predating
    the capabilities Protocol fall back to the descriptor's static
    extras dict.

    Returns:
        On success: ``{"status": "ok", "name", "target", "version",
        "module", "capabilities": ... | None, "descriptor_extras": {...}}``.
        On miss: ``{"status": "vendor_not_found", "vendor_name",
        "available": [...]}``.
        On adapter load failure:
        ``{"status": "load_failed", "vendor_name", "error"}``.
    """
    del sm
    try:
        eps = list(importlib_metadata.entry_points(group=_ENTRY_POINT_GROUP))
    except Exception as exc:  # noqa: BLE001
        log.warning("vendor_dialect.describe.entry_points_error", error=str(exc))
        return {
            "status": "load_failed",
            "vendor_name": vendor_name,
            "error": f"{type(exc).__name__}: {exc}",
        }

    available_names: list[str] = []
    target_ep = None
    for ep in eps:
        # Match either the entry-point name or the adapter's own
        # canonical name once loaded — the registry resolves by the
        # latter, but agents commonly query by the former.
        if ep.name == vendor_name:
            target_ep = ep
        available_names.append(ep.name)

    if target_ep is None:
        # Try canonical-name match by loading each ep until we find
        # one whose adapter.name == vendor_name.
        for ep in eps:
            try:
                factory = ep.load()
                adapter = factory() if callable(factory) else factory
                if isinstance(adapter, VendorDialectAdapter) and adapter.name == vendor_name:
                    target_ep = ep
                    break
            except Exception:  # noqa: BLE001
                continue

    if target_ep is None:
        # Per bridge #134: fall back to the in-process registry.
        # ``register_adapter()`` -populated adapters don't appear in
        # entry points but are valid lookup targets for compile_with_vendor.
        registry_names = available_adapters()
        if vendor_name in registry_names:
            try:
                adapter = get_adapter(vendor_name)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "load_failed",
                    "vendor_name": vendor_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            descriptor = getattr(adapter, "descriptor", None)
            descriptor_extras: dict[str, Any] = {}
            if descriptor is not None:
                extras = getattr(descriptor, "extras", None) or {}
                if isinstance(extras, dict):
                    descriptor_extras = dict(extras)
            version = getattr(adapter, "version", None) or descriptor_extras.get("version")
            capabilities: dict[str, Any] | None = None
            capabilities_method = getattr(adapter, "capabilities", None)
            if callable(capabilities_method):
                try:
                    caps = capabilities_method()
                    capabilities = dict(caps) if caps is not None else None
                except Exception as exc:  # noqa: BLE001
                    return {
                        "status": "load_failed",
                        "vendor_name": vendor_name,
                        "error": (f"capabilities() raised: {type(exc).__name__}: {exc}"),
                    }
            return {
                "status": "ok",
                "name": adapter.name,
                "target": adapter.target,
                "version": str(version) if version is not None else None,
                "module": "<in-process>",
                "capabilities": capabilities,
                "descriptor_extras": descriptor_extras,
            }
        # Surface BOTH discovery sources so the agent's correction
        # path sees everything it could have called.
        return {
            "status": "vendor_not_found",
            "vendor_name": vendor_name,
            "available": sorted(set(available_names) | set(registry_names)),
        }

    try:
        factory = target_ep.load()
        adapter = factory() if callable(factory) else factory
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "load_failed",
            "vendor_name": vendor_name,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(adapter, VendorDialectAdapter):
        return {
            "status": "load_failed",
            "vendor_name": vendor_name,
            "error": (
                f"entry point {target_ep.name!r} did not yield a VendorDialectAdapter (got {type(adapter).__name__})"
            ),
        }

    descriptor = getattr(adapter, "descriptor", None)
    descriptor_extras: dict[str, Any] = {}
    if descriptor is not None:
        extras = getattr(descriptor, "extras", None) or {}
        if isinstance(extras, dict):
            descriptor_extras = dict(extras)

    version = getattr(adapter, "version", None) or descriptor_extras.get("version")

    capabilities: dict[str, Any] | None = None
    capabilities_method = getattr(adapter, "capabilities", None)
    if callable(capabilities_method):
        try:
            caps = capabilities_method()
            capabilities = dict(caps) if caps is not None else None
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "load_failed",
                "vendor_name": vendor_name,
                "error": (f"capabilities() raised: {type(exc).__name__}: {exc}"),
            }

    return {
        "status": "ok",
        "name": adapter.name,
        "target": adapter.target,
        "version": str(version) if version is not None else None,
        "module": getattr(target_ep, "value", None) or str(target_ep),
        "capabilities": capabilities,
        "descriptor_extras": descriptor_extras,
    }


def compgen_compile_torch_model_with_vendor(
    sm: SessionManager,
    *,
    model_pickle_b64: str,
    sample_input_pickle_b64: str,
    output_dir: str,
    vendor_name: str,
    vendor_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a base64-pickled torch model through a registered vendor adapter.

    Resolves ``vendor_name`` against the process-wide
    :class:`VendorAdapterRegistry` (which auto-loads entry points), then
    drives :func:`compgen.api.compile_with_vendor`. ``vendor_options``
    is forwarded as the adapter's ``options`` dict (the registry stores
    pre-instantiated adapters; per-call configuration goes through the
    ``compile`` boundary).

    Args:
        sm: Session manager (unused — vendor compile is session-less).
        model_pickle_b64: ``base64(pickle.dumps(model))``.
        sample_input_pickle_b64: ``base64(pickle.dumps((x,)))``.
        output_dir: Destination for the vendor bundle artifacts.
        vendor_name: Registry key (e.g. ``"cuda_tile"``).
        vendor_options: Forwarded to ``vendor_adapter.compile`` as the
            ``options`` argument.

    Returns:
        ``{
            "status": "ok" | "vendor_not_found" | "lowering_failed"
                     | "load_failed",
            "bundle_dir": <str | None>,
            "vendor_name": <str>,
            "lowering_summary": <dict>,
            "elapsed_ms": <float>,
            "error": <str | None>,
            "available": [<str>, ...]   # only when vendor_not_found
        }``

    Never raises — exceptions are caught at the boundary and surfaced
    via ``status``/``error``.
    """
    del sm
    t0 = time.perf_counter()
    elapsed = lambda: (time.perf_counter() - t0) * 1000.0  # noqa: E731

    # 1. Resolve the adapter via the registry. Entry-point discovery is
    # triggered on first lookup.
    try:
        adapter = get_adapter(vendor_name)
    except KeyError:
        return {
            "status": "vendor_not_found",
            "bundle_dir": None,
            "vendor_name": vendor_name,
            "lowering_summary": {},
            "elapsed_ms": elapsed(),
            "error": f"vendor adapter {vendor_name!r} is not registered",
            "available": available_adapters(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "load_failed",
            "bundle_dir": None,
            "vendor_name": vendor_name,
            "lowering_summary": {},
            "elapsed_ms": elapsed(),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # 2. Unpickle inputs. Decode failures land as ``load_failed``.
    try:
        import pickle

        model = pickle.loads(base64.b64decode(model_pickle_b64))
        sample_inputs = pickle.loads(base64.b64decode(sample_input_pickle_b64))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "load_failed",
            "bundle_dir": None,
            "vendor_name": vendor_name,
            "lowering_summary": {},
            "elapsed_ms": elapsed(),
            "error": f"failed to deserialize inputs: {type(exc).__name__}: {exc}",
        }

    # 3. Drive the vendor compile pipeline.
    try:
        from compgen.api import compile_with_vendor

        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        artifact = compile_with_vendor(
            model,
            adapter,
            sample_inputs=tuple(sample_inputs) if isinstance(sample_inputs, (list, tuple)) else (sample_inputs,),
            output_dir=out_path,
            options=vendor_options or {},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "vendor_dialect.compile.lowering_failed",
            vendor=vendor_name,
            error=str(exc),
        )
        return {
            "status": "lowering_failed",
            "bundle_dir": None,
            "vendor_name": vendor_name,
            "lowering_summary": {},
            "elapsed_ms": elapsed(),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Build a JSON-safe lowering summary from the CompiledArtifact.
    summary: dict[str, Any] = {}
    for key in ("format", "target_name", "metadata"):
        value = getattr(artifact, key, None)
        if value is not None:
            summary[key] = value
    code = getattr(artifact, "code", None)
    if isinstance(code, str):
        summary["code_size"] = len(code)
    elif isinstance(code, (bytes, bytearray)):
        summary["code_size"] = len(code)

    return {
        "status": "ok",
        "bundle_dir": str(out_path),
        "vendor_name": vendor_name,
        "lowering_summary": summary,
        "elapsed_ms": elapsed(),
        "error": None,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_descriptor(descriptor_yaml: str | None, descriptor_path: str | None) -> VendorDialectDescriptor:
    if descriptor_yaml and descriptor_path:
        raise ValueError("pass descriptor_yaml OR descriptor_path, not both")
    if descriptor_yaml:
        return VendorDialectDescriptor.from_yaml(descriptor_yaml)
    if descriptor_path:
        return VendorDialectDescriptor.load(descriptor_path)
    raise ValueError("must pass descriptor_yaml or descriptor_path")


# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #


VENDOR_DIALECT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "scan_vendor_repo",
        "description": (
            "Scan a third-party MLIR dialect repo and propose a frozen "
            "VendorDialectDescriptor (YAML) plus a lowering proposal. "
            "Review the YAML, then call scaffold_vendor_package."
        ),
        "phase": "inspect",
        "handler": scan_vendor_repo,
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "target": {"type": "string"},
                "workloads": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["repo_path", "target"],
        },
    },
    {
        "name": "propose_vendor_spec",
        "description": (
            "Re-run the vendor classifier with explicit vendor/package overrides. "
            "Returns a VendorDialectDescriptor YAML ready for review."
        ),
        "phase": "inspect",
        "handler": propose_vendor_spec,
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "target": {"type": "string"},
                "workloads": {"type": "array", "items": {"type": "string"}},
                "package_name": {"type": "string"},
                "vendor_name": {"type": "string"},
            },
            "required": ["repo_path", "target"],
        },
    },
    {
        "name": "scaffold_vendor_package",
        "description": (
            "Render a pip-installable user-space adapter package from a "
            "reviewed VendorDialectDescriptor. Pass descriptor_yaml OR "
            "descriptor_path."
        ),
        "phase": "transform",
        "handler": scaffold_vendor_package,
        "input_schema": {
            "type": "object",
            "properties": {
                "descriptor_yaml": {"type": "string"},
                "descriptor_path": {"type": "string"},
                "out_dir": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["out_dir"],
        },
    },
    {
        "name": "verify_vendor_package",
        "description": (
            "Run the verification ladder (structural / matmul / workload) against a scaffolded adapter package."
        ),
        "phase": "inspect",
        "handler": verify_vendor_package,
        "input_schema": {
            "type": "object",
            "properties": {
                "package_dir": {"type": "string"},
                "run_workload_gate": {"type": "boolean"},
            },
            "required": ["package_dir"],
        },
    },
    {
        "name": "compgen_list_vendor_dialects",
        "description": (
            "List every vendor dialect adapter advertised on the "
            "'compgen.vendor_dialects' entry-point group. Returns one "
            "record per adapter with name / target / version / "
            "capabilities() / module path. Adapters that fail to load "
            "are reported with an error field rather than being skipped."
        ),
        "phase": "inspect",
        "handler": compgen_list_vendor_dialects,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compgen_describe_vendor_dialect",
        "description": (
            "Return one vendor adapter's full descriptor + capabilities. "
            "Pre-screening tool the agent calls before committing to "
            "compgen_compile_torch_model_with_vendor — given a workload's "
            "shape/dtype, the agent asks 'can vendor X lower this?'. "
            "Status='vendor_not_found' (with the available registry "
            "names), 'load_failed', or 'ok'."
        ),
        "phase": "inspect",
        "handler": compgen_describe_vendor_dialect,
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": (
                        "Registry key for the vendor adapter. Use compgen_list_vendor_dialects to discover names."
                    ),
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "compgen_compile_torch_model_with_vendor",
        "description": (
            "Compile a base64-pickled torch nn.Module through a "
            "registered vendor dialect adapter. Resolves the adapter "
            "via the 'compgen.vendor_dialects' entry-point registry, "
            "then drives compgen.api.compile_with_vendor. Returns "
            "status='vendor_not_found' (with the available registry "
            "names), 'load_failed', 'lowering_failed', or 'ok'. Never "
            "raises — failures land in status/error."
        ),
        "phase": "compile",
        "handler": compgen_compile_torch_model_with_vendor,
        "input_schema": {
            "type": "object",
            "properties": {
                "model_pickle_b64": {
                    "type": "string",
                    "description": "base64-encoded pickle.dumps(nn.Module).",
                },
                "sample_input_pickle_b64": {
                    "type": "string",
                    "description": ("base64-encoded pickle.dumps((x,)) — a tuple of sample inputs for torch.export."),
                },
                "output_dir": {
                    "type": "string",
                    "description": "Filesystem path for the vendor bundle.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": (
                        "Registry key for the vendor adapter (e.g. "
                        "'cuda_tile'). Use compgen_list_vendor_dialects "
                        "to discover available names."
                    ),
                },
                "vendor_options": {
                    "type": ["object", "null"],
                    "description": (
                        "Free-form options dict forwarded to the adapter's compile() call. Adapter-specific."
                    ),
                },
            },
            "required": [
                "model_pickle_b64",
                "sample_input_pickle_b64",
                "output_dir",
                "vendor_name",
            ],
        },
    },
]


__all__ = [
    "VENDOR_DIALECT_TOOLS",
    "compgen_compile_torch_model_with_vendor",
    "compgen_describe_vendor_dialect",
    "compgen_list_vendor_dialects",
    "propose_vendor_spec",
    "scaffold_vendor_package",
    "scan_vendor_repo",
    "verify_vendor_package",
]
