"""Claude-backed kernel provider for vendor-dialect kernel authoring.

When a third-party MLIR vendor has no direct linalg/stablehlo ingress
(e.g. NVIDIA CUDA Tile IR), CompGen has to *author* kernels directly in
the vendor's dialect. The obvious tool for that is the same LLM that
drives the rest of CompGen.

This provider is a ``KernelProvider`` that:

1. Takes a :class:`PromptPack` pinning dialect-specific prompts + guard
   rails (the vendor adapter supplies this).
2. Invokes ``autocomp.common.llm_utils.LLMClient`` — we reuse autocomp's
   client per the repo rule against duplicating LLM plumbing.
3. Runs a bounded retry loop:
   * generate candidate MLIR
   * run a caller-supplied structural gate (``cuda-tile-opt --verify``,
     or a lightweight parser for tests)
   * stop on first accepted candidate; otherwise refine
4. Exports a :class:`KnowledgeExport` that records the winning candidate
   per op family so later searches can reuse it.

Autocomp's own loop is a superset of this, but it is CUDA-specific and
couples shape/op family assumptions into its plan/code phases. This
provider is the minimal, vendor-agnostic surface that user-space
adapters compose on top of.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Prompt pack
# --------------------------------------------------------------------------- #


@dataclass
class PromptPack:
    """Dialect-specific prompts + guard rails.

    Attributes:
        name: Dialect identifier (e.g. ``"cuda_tile"``).
        system: System message pinning the dialect conventions.
        op_templates: Per-op-family prompt templates. Each template may
            reference ``{shapes}``, ``{dtypes}``, ``{layout}``, ``{hints}``,
            ``{region_id}``.
        default_template: Used when no op-specific template matches.
        forbidden_substrings: Heuristic filter — reject candidates that
            contain any of these strings.
        code_fence_language: Pulled out of the LLM response. Empty string
            means "accept any fenced block".
        max_iterations: Default retry budget when the caller does not
            override via :class:`SearchBudget`.
    """

    name: str
    system: str = "You emit MLIR in a specified dialect. No prose."
    op_templates: dict[str, str] = field(default_factory=dict)
    default_template: str = (
        "Emit a {dialect} MLIR module for {op_family} with "
        "input shapes {shapes} and dtypes {dtypes}. Return ONLY the module, "
        "fenced with triple backticks."
    )
    forbidden_substrings: tuple[str, ...] = ()
    code_fence_language: str = "mlir"
    max_iterations: int = 4


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


StructuralGate = Callable[[str, KernelContract], tuple[bool, str]]
"""``structural_gate(candidate_mlir, contract) -> (accepted, diagnostic)``.

If ``accepted`` is False, ``diagnostic`` is fed back into the next LLM
turn as context so the model can correct itself. The gate is caller-
supplied; adapters typically pass a thin wrapper around
``<vendor>-opt --verify``.
"""


class ClaudeKernelProvider:
    """Vendor-agnostic, LLM-driven kernel provider.

    Despite the name, the underlying ``LLMClient`` can target any
    provider autocomp supports (OpenAI, Anthropic/Bedrock, Gemini,
    Together). The default model is the latest Anthropic Claude on the
    assumption that vendor-dialect authoring is a long-context reasoning
    task.
    """

    _CODE_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z_+-]*)\n(?P<code>.*?)```", re.DOTALL)
    _DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(
        self,
        *,
        name: str,
        prompt_pack: PromptPack,
        target_name: str,
        model: str | None = None,
        llm_client: object | None = None,
        structural_gate: StructuralGate | None = None,
    ) -> None:
        self._name = name
        self._pack = prompt_pack
        self._target_name = target_name
        self._model = model or self._DEFAULT_MODEL
        self._explicit_client = llm_client
        self._gate = structural_gate or _permissive_gate
        self._accumulated_knowledge: list[KnowledgeExport] = []

    # ------------------------------------------------------------------ #
    # KernelProvider protocol
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name

    def accepts_contract(self, contract: KernelContract) -> bool:
        if not contract.op_family:
            return False
        if contract.target_name and contract.target_name != self._target_name:
            return False
        return True

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        client = self._llm_client()
        if client is None:
            log.warning("claude_kernel.no_client", provider=self._name)
            return ProviderResult(found=False, metadata={"reason": "no_llm_client"})

        max_iters = min(budget.max_iterations or self._pack.max_iterations, self._pack.max_iterations)
        deadline = time.monotonic() + max(1, budget.max_time_ms) / 1000.0

        prompt = self._build_prompt(contract)
        diagnostic: str = ""
        candidate: str = ""

        for iteration in range(1, max_iters + 1):
            if time.monotonic() > deadline:
                log.info("claude_kernel.deadline", provider=self._name, iteration=iteration)
                break
            turn_prompt = (
                prompt
                if not diagnostic
                else (
                    f"{prompt}\n\n---\nPrevious attempt failed the structural gate "
                    f"with:\n{diagnostic}\nFix the issue and re-emit the module."
                )
            )
            try:
                responses = _client_chat(client, turn_prompt, model=self._model)
            except Exception as exc:
                log.warning("claude_kernel.llm_error", provider=self._name, error=str(exc))
                return ProviderResult(found=False, metadata={"reason": "llm_error", "error": str(exc)})

            for resp in responses:
                extracted = self._extract_code(resp)
                if not extracted:
                    diagnostic = "no fenced code block found"
                    continue
                if self._hits_forbidden(extracted):
                    diagnostic = "candidate contained a forbidden substring"
                    continue
                accepted, note = self._gate(extracted, contract)
                if accepted:
                    candidate = extracted
                    diagnostic = note
                    break
                diagnostic = note or "structural gate rejected candidate"

            if candidate:
                break

        if not candidate:
            return ProviderResult(
                found=False,
                iterations_used=iteration,
                metadata={"reason": "no_accepted_candidate", "last_diagnostic": diagnostic},
            )

        knowledge = KnowledgeExport(
            kind="vendor_kernel",
            scope="op_family",
            scope_key=f"{self._pack.name}/{contract.op_family}",
            content=candidate,
            metadata={
                "target": self._target_name,
                "shapes": str(contract.input_shapes),
                "dtypes": ",".join(contract.dtypes),
                "summary": f"{self._pack.name} kernel for {contract.op_family}",
            },
            confidence=0.6,
        )
        self._accumulated_knowledge.append(knowledge)
        return ProviderResult(
            found=True,
            kernel_code=candidate,
            language="mlir",
            correct=True,
            iterations_used=iteration,
            knowledge_exports=[knowledge],
            contract_feedback=self._derive_feedback(contract, candidate),
            metadata={"provider": self._name, "dialect": self._pack.name, "model": self._model},
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_prompt(self, contract: KernelContract) -> str:
        template = self._pack.op_templates.get(contract.op_family, self._pack.default_template)
        hints = ", ".join(f"{k}={v}" for k, v in contract.provider_hints.items()) or "(none)"
        body = template.format(
            dialect=self._pack.name,
            op_family=contract.op_family,
            shapes=list(contract.input_shapes),
            dtypes=list(contract.dtypes),
            layout=contract.layout,
            hints=hints,
            region_id=contract.region_id,
        )
        return f"{self._pack.system}\n\n{body}"

    def _extract_code(self, response: str) -> str:
        for m in self._CODE_FENCE_RE.finditer(response):
            lang = (m.group("lang") or "").strip().lower()
            if self._pack.code_fence_language and lang and lang != self._pack.code_fence_language:
                continue
            code = m.group("code").strip()
            if code:
                return code
        return ""

    def _hits_forbidden(self, code: str) -> bool:
        return any(tok and tok in code for tok in self._pack.forbidden_substrings)

    def _derive_feedback(self, contract: KernelContract, candidate: str) -> list[ContractFeedback]:
        feedback: list[ContractFeedback] = []
        # Minimal: record the layout the candidate appears to target.
        if contract.layout == "row_major" and "col_major" in candidate:
            feedback.append(
                ContractFeedback(
                    field="layout",
                    current_value="row_major",
                    suggested_value="col_major",
                    reason="kernel emitted in col_major ordering",
                    measured_gain=0.0,
                )
            )
        return feedback

    def _llm_client(self):
        if self._explicit_client is not None:
            return self._explicit_client
        try:
            from autocomp.common.llm_utils import LLMClient  # type: ignore[import-not-found]
        except Exception as exc:
            log.info("claude_kernel.autocomp_unavailable", error=str(exc))
            return None
        if not _provider_keys_available(self._model):
            log.info("claude_kernel.no_keys", model=self._model)
            return None
        try:
            return LLMClient(model=self._model)
        except Exception as exc:
            log.warning("claude_kernel.client_init_failed", error=str(exc))
            return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _permissive_gate(candidate: str, contract: KernelContract) -> tuple[bool, str]:
    del contract
    return (bool(candidate.strip()), "")


def _client_chat(client: object, prompt: str, *, model: str) -> list[str]:
    """Invoke the LLM client.

    Supports two shapes:

    * autocomp's ``LLMClient.chat(prompt, num_samples, temperature)`` →
      ``list[str]``
    * a minimal shim exposing ``chat(prompt) -> list[str]`` (used by the
      ``MockLLMClient`` in tests).
    """
    if hasattr(client, "chat"):
        try:
            return list(client.chat(prompt, num_samples=1))
        except TypeError:
            return list(client.chat(prompt))
    if callable(client):
        out = client(prompt)
        return out if isinstance(out, list) else [str(out)]
    raise TypeError(f"unsupported LLM client type: {type(client).__name__}")


def _provider_keys_available(model: str) -> bool:
    """Best-effort key check — matches autocomp's provider-from-model heuristic."""
    if "claude" in model:
        return bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(os.environ.get("AWS_ACCESS_KEY_ID"))
    if "gemini" in model:
        return bool(os.environ.get("GOOGLE_API_KEY")) or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return bool(os.environ.get("OPENAI_API_KEY"))
    return True


__all__ = ["ClaudeKernelProvider", "PromptPack", "StructuralGate"]
