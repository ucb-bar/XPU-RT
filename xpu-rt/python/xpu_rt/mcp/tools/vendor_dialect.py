"""MCP tools: scan / propose / scaffold / verify a vendor MLIR dialect.

These four tools are the MCP-visible surface of the vendor integration
agent. They are thin wrappers over
:mod:`compgen.extensions.vendor_dialect` and
:mod:`compgen.agent.vendor_integration`.

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

from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.vendor_integration.explore import explore_vendor_repo
from compgen.agent.vendor_integration.propose_adapter import propose_adapter_layout
from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_descriptor(
    descriptor_yaml: str | None, descriptor_path: str | None
) -> VendorDialectDescriptor:
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
            "Run the verification ladder (structural / matmul / workload) "
            "against a scaffolded adapter package."
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
]


__all__ = [
    "VENDOR_DIALECT_TOOLS",
    "propose_vendor_spec",
    "scaffold_vendor_package",
    "scan_vendor_repo",
    "verify_vendor_package",
]
