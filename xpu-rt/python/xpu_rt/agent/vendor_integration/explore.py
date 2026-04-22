"""Explore a vendor MLIR repo and produce a ``VendorDialectDescriptor``.

Pipeline:

1. :func:`compgen.extensions.vendor_dialect.scanner.scan_repo` collects
   deterministic facts.
2. The prompt template at ``prompts/explore_vendor.md`` is rendered with
   those facts.
3. The LLM returns a JSON blob; we parse it and fold it into a descriptor
   that merges scan output (authoritative) with LLM classifications.

Fallback behaviour: when no LLM client is available (no API keys, offline
tests), :func:`explore_vendor_repo` still returns a best-effort descriptor
derived from the scan plus conservative defaults. This keeps the agent
loop usable in CI even without network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    OpEntry,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.scanner import ScanResult, scan_repo

log = structlog.get_logger()


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "explore_vendor.md"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExploreResult:
    """Bundle of outputs from the explore phase."""

    descriptor: VendorDialectDescriptor
    scan: ScanResult
    llm_classification: dict[str, Any]
    llm_used: bool


def explore_vendor_repo(
    repo_path: str | Path,
    *,
    target: str,
    workloads: tuple[str, ...] = (),
    package_name: str | None = None,
    vendor_name: str | None = None,
    llm_client: object | None = None,
) -> ExploreResult:
    """Scan a vendor repo and synthesise a proposed descriptor.

    Args:
        repo_path: Path to the vendor repository (cloned locally).
        target: CompGen target name the adapter will bind to.
        workloads: Names of workloads the adapter must ultimately run.
        package_name: Override for the user-space package name. Defaults
            to ``compgen_<vendor_name>``.
        vendor_name: Override for the canonical dialect name. Defaults
            to the first dialect the scanner detected, or the repo dir
            name as a last resort.
        llm_client: Optional client implementing either
            ``chat(prompt, num_samples=...) -> list[str]`` (autocomp) or
            ``chat(prompt) -> list[str]`` (MockLLMClient).
    """
    scan = scan_repo(repo_path)
    inferred_name = vendor_name or _infer_vendor_name(scan)
    pkg = package_name or f"compgen_{_sanitize(inferred_name)}"

    classification, llm_used = _classify_with_llm(scan, llm_client)
    descriptor = _assemble_descriptor(
        scan=scan,
        vendor_name=inferred_name,
        package_name=pkg,
        target=target,
        workloads=workloads,
        classification=classification,
    )
    log.info(
        "vendor_explore.done",
        vendor=inferred_name,
        llm_used=llm_used,
        kernel_auth=descriptor.kernel_authoring_required,
    )
    return ExploreResult(
        descriptor=descriptor,
        scan=scan,
        llm_classification=classification,
        llm_used=llm_used,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _classify_with_llm(scan: ScanResult, llm_client: object | None) -> tuple[dict[str, Any], bool]:
    """Run the explore prompt; return classification + whether LLM was used."""
    if llm_client is None:
        return _default_classification(scan), False

    prompt = _render_prompt(scan)
    try:
        responses = _chat(llm_client, prompt)
    except Exception as exc:
        log.warning("vendor_explore.llm_error", error=str(exc))
        return _default_classification(scan), False

    for resp in responses:
        parsed = _parse_json(resp)
        if parsed:
            merged = _default_classification(scan)
            merged.update({k: v for k, v in parsed.items() if v is not None})
            return merged, True
    log.warning("vendor_explore.no_parsable_response")
    return _default_classification(scan), False


def _render_prompt(scan: ScanResult) -> str:
    template = _PROMPT_PATH.read_text()
    readme_excerpt = scan.readme_text[:2000]
    td_ops = "\n".join(f"- {op.name}: {op.summary}" for op in scan.td_ops[:30])
    cli_tools = "\n".join(f"- {t}" for t in scan.cli_tools)
    return template.format(
        scanner_summary=json.dumps(scan.summary(), indent=2),
        readme_excerpt=readme_excerpt,
        td_ops=td_ops or "(none detected)",
        cli_tools=cli_tools or "(none detected)",
    )


def _chat(client: object, prompt: str) -> list[str]:
    if hasattr(client, "chat"):
        try:
            return list(client.chat(prompt, num_samples=1))
        except TypeError:
            return list(client.chat(prompt))
    if callable(client):
        out = client(prompt)
        return out if isinstance(out, list) else [str(out)]
    raise TypeError(f"unsupported LLM client type: {type(client).__name__}")


def _parse_json(response: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(response)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _default_classification(scan: ScanResult) -> dict[str, Any]:
    """Conservative classification used when the LLM is unavailable.

    We assume:
    * The vendor accepts some generic upstream IR when there is a CLI tool
      whose name contains ``linalg`` — otherwise we fall back to
      ``kernel_authoring``.
    * ``bundle_steps`` reference the CLI tools in their scan order.
    """
    has_linalg_tool = any("linalg" in t for t in scan.cli_tools)
    input_ir = ["linalg"] if has_linalg_tool else ["<vendor-native>"]
    mode = "direct_linalg" if has_linalg_tool else "kernel_authoring"
    op_families = _guess_op_families(scan)
    bundle_steps = [f"{t} <input>" for t in scan.cli_tools[:3]]
    return {
        "input_ir": input_ir,
        "output_format": _guess_output_format(scan),
        "kernel_authoring_required": mode == "kernel_authoring",
        "lowering_mode": mode,
        "op_families": op_families,
        "bundle_steps": bundle_steps,
        "runtime_entry": "",
        "notes": "default classification (no LLM)",
    }


def _guess_op_families(scan: ScanResult) -> list[str]:
    known = ["matmul", "softmax", "rmsnorm", "layernorm", "flash_attn", "conv", "reduce"]
    hits: list[str] = []
    haystack = " ".join(op.name.lower() + " " + op.summary.lower() for op in scan.td_ops)
    for fam in known:
        if fam in haystack or fam.replace("_", "") in haystack:
            hits.append(fam)
    return hits


def _guess_output_format(scan: ScanResult) -> str:
    lo = scan.readme_text.lower()
    for marker, fmt in (
        ("cubin", "cubin"),
        ("ptx", "ptx"),
        ("bytecode", "bytecode"),
        ("hexagon", "hexagon_elf"),
        ("llvm", "llvm_ir"),
    ):
        if marker in lo:
            return fmt
    return "binary"


def _assemble_descriptor(
    *,
    scan: ScanResult,
    vendor_name: str,
    package_name: str,
    target: str,
    workloads: tuple[str, ...],
    classification: dict[str, Any],
) -> VendorDialectDescriptor:
    op_registry = tuple(
        OpEntry(
            name=op.name,
            summary=op.summary,
            source_file=op.source_file,
        )
        for op in scan.td_ops
    )
    compile_entry = CompileEntry(
        cli_tools=tuple(scan.cli_tools),
        python_module="",
        python_symbols=(),
    )
    lowering = LoweringStrategy(
        mode=str(classification.get("lowering_mode", "direct_linalg")),
        op_families=tuple(classification.get("op_families", ()) or ()),
        template_ops=(),
        notes=str(classification.get("notes", "")),
    )
    bundle = BundlePlan(
        steps=tuple(classification.get("bundle_steps", ()) or ()),
        output_format=str(classification.get("output_format", "binary")),
        runtime_entry=str(classification.get("runtime_entry", "")),
    )
    verification = VerificationPlan(
        structural=True,
        matmul_diff_test=True,
        workload_diff_test=bool(workloads),
        workloads=tuple(workloads),
    )
    return VendorDialectDescriptor(
        name=vendor_name,
        package_name=package_name,
        repo_path=scan.repo_path,
        target=target,
        input_ir=tuple(classification.get("input_ir", ()) or ()),
        output_format=str(classification.get("output_format", "binary")),
        compile_entry=compile_entry,
        td_files=tuple(scan.td_files),
        op_registry=op_registry,
        lowering=lowering,
        bundle=bundle,
        verification=verification,
        kernel_authoring_required=bool(classification.get("kernel_authoring_required", False)),
        dependencies=(),
        license=scan.license_spdx,
        extras={"explore_notes": str(classification.get("notes", ""))},
    )


def _infer_vendor_name(scan: ScanResult) -> str:
    if scan.dialect_names:
        return _sanitize(scan.dialect_names[0])
    return _sanitize(Path(scan.repo_path).name)


_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _sanitize(name: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", name.lower()).strip("_")
    return cleaned or "vendor"


__all__ = ["ExploreResult", "explore_vendor_repo"]
