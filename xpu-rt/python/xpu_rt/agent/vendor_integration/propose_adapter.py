"""Turn a frozen ``VendorDialectDescriptor`` into an adapter layout proposal.

This runs *after* the exploration phase and *before* scaffolding. The
output is a :class:`ProposedAdapter` that summarises what the scaffold
engine will produce: which op families get hand-crafted templates, which
delegate to the LLM-backed kernel provider, and which risks the
integration carries.

The LLM is optional: a deterministic proposer runs first and the LLM
refines it; if no LLM is available, the deterministic proposal is
returned as-is. This matches how :mod:`explore` handles fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor

log = structlog.get_logger()


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "propose_lowering.md"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LoweringRule:
    op_family: str
    strategy: str  # "template" | "llm" | "passthrough"
    rationale: str = ""


@dataclass
class ProposedAdapter:
    """Adapter layout proposal — fed to the scaffold engine."""

    rules: list[LoweringRule] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    verification_hooks: list[str] = field(default_factory=list)
    llm_used: bool = False


def propose_adapter_layout(
    descriptor: VendorDialectDescriptor,
    *,
    workloads: tuple[str, ...] = (),
    llm_client: object | None = None,
) -> ProposedAdapter:
    """Produce a :class:`ProposedAdapter` for the scaffold engine."""
    baseline = _deterministic_proposal(descriptor)

    if llm_client is None:
        return baseline

    prompt = _render_prompt(descriptor, workloads)
    try:
        responses = _chat(llm_client, prompt)
    except Exception as exc:
        log.warning("propose_adapter.llm_error", error=str(exc))
        return baseline

    for resp in responses:
        parsed = _parse_json(resp)
        if parsed:
            return _merge_proposal(baseline, parsed)
    return baseline


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _deterministic_proposal(descriptor: VendorDialectDescriptor) -> ProposedAdapter:
    mode = descriptor.lowering.mode
    rules: list[LoweringRule] = []
    for fam in descriptor.lowering.op_families or ("matmul",):
        if mode == "direct_linalg":
            strategy = "passthrough"
            rationale = "vendor accepts linalg directly"
        elif mode == "kernel_authoring":
            strategy = "llm"
            rationale = "no upstream ingress — LLM-backed kernel required"
        else:
            strategy = "llm"
            rationale = f"mode={mode} defaults to llm-backed lowering"
        rules.append(LoweringRule(op_family=fam, strategy=strategy, rationale=rationale))

    risks: list[str] = []
    if descriptor.kernel_authoring_required and not descriptor.op_registry:
        risks.append("kernel authoring required but op registry is empty")
    if not descriptor.bundle.steps:
        risks.append("bundle plan has no steps — vendor toolchain not discovered")
    if not descriptor.compile_entry.cli_tools and not descriptor.compile_entry.python_module:
        risks.append("no CLI tools or Python bindings detected")

    hooks = ["structural"]
    if descriptor.verification.matmul_diff_test:
        hooks.append("matmul_diff")
    if descriptor.verification.workload_diff_test:
        hooks.append("workload_diff")

    return ProposedAdapter(rules=rules, risks=risks, verification_hooks=hooks, llm_used=False)


def _render_prompt(descriptor: VendorDialectDescriptor, workloads: tuple[str, ...]) -> str:
    template = _PROMPT_PATH.read_text()
    return template.format(
        descriptor_yaml=descriptor.to_yaml(),
        workloads=", ".join(workloads) or "(none)",
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


def _merge_proposal(baseline: ProposedAdapter, payload: dict[str, Any]) -> ProposedAdapter:
    rules = [
        LoweringRule(
            op_family=str(r.get("op_family", "")),
            strategy=str(r.get("strategy", "llm")),
            rationale=str(r.get("rationale", "")),
        )
        for r in payload.get("rules", []) or []
        if r.get("op_family")
    ]
    if not rules:
        rules = baseline.rules
    risks = [str(x) for x in payload.get("risks", []) or []] or baseline.risks
    hooks = [str(x) for x in payload.get("verification_hooks", []) or []] or baseline.verification_hooks
    return ProposedAdapter(rules=rules, risks=risks, verification_hooks=hooks, llm_used=True)


__all__ = ["LoweringRule", "ProposedAdapter", "propose_adapter_layout"]
