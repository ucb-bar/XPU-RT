"""LLM-driven unsupported-operator recovery orchestrator.

Used by :func:`compgen.api_llm.compile_with_llm` when
``recover_unsupported=True``. Given a :class:`CaptureArtifact` whose
``unsupported_resolutions`` list is non-empty, picks a strategy for
each op — optionally asking the LLM when the dossier's own
classification is ambiguous — and returns a plan the caller can
apply before the FX → xDSL import step.

No LLM call is issued unless the deterministic classifier is
uncertain (``confidence == "low"``). This keeps the common-case path
silent and cheap; the LLM is only invoked where its judgement
actually matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.capture.torch_export import CaptureArtifact
from compgen.capture.unsupported import UnsupportedOpResolution
from compgen.capture.unsupported.synthesize_decomp import (
    synthesize_export_decomposition,
)
from compgen.capture.unsupported.synthesize_translation import (
    synthesize_payload_translation,
)
from compgen.llm.base import CompGenLLMProtocol

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class OpRecoveryDecision:
    """One resolved target — picked strategy + evidence."""

    target: str
    strategy: str  # decomp | translation | blackbox | none
    source: str  # classifier | llm | override | fallback
    ok: bool
    detail: str = ""
    error: str = ""


@dataclass
class RecoveryPlan:
    """Aggregate: how every unsupported op will be handled."""

    decisions: list[OpRecoveryDecision] = field(default_factory=list)
    llm_consulted: int = 0
    skipped: int = 0

    def ok(self) -> bool:
        return all(d.ok for d in self.decisions)

    def by_strategy(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for d in self.decisions:
            out.setdefault(d.strategy, []).append(d.target)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok(),
            "num_issues": len(self.decisions),
            "llm_consulted": self.llm_consulted,
            "skipped": self.skipped,
            "decisions": [
                {
                    "target": d.target,
                    "strategy": d.strategy,
                    "source": d.source,
                    "ok": d.ok,
                    "detail": d.detail,
                    "error": d.error,
                }
                for d in self.decisions
            ],
        }


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------


def _llm_pick_strategy(
    resolution: UnsupportedOpResolution,
    llm_client: CompGenLLMProtocol,
) -> tuple[str, str]:
    """Ask the LLM which bucket (decomp|translation|blackbox) to use.

    Returns ``(strategy, reasoning)``. On any LLM failure returns
    ``("fallback", str(err))`` — callers then route to the
    deterministic default.
    """
    from compgen.llm.base import GenerationRequest, LLMConfig, PromptContext

    target = resolution.target
    dossier = resolution.dossier
    cls = resolution.classification

    prompt = (
        "An unsupported PyTorch operator was detected during capture. "
        "Pick ONE of (decomp | translation | blackbox).\n\n"
        f"Target: {target}\n"
        f"Schema: {dossier.schema}\n"
        f"is_aten: {dossier.is_aten}, is_custom: {dossier.is_custom}, "
        f"is_torchao_like: {dossier.is_torchao_like}\n"
        f"Classifier bucket: {cls.bucket}, strategy: {cls.strategy}, "
        f"confidence: {cls.confidence}\n"
        f"Classifier reason: {cls.reason}\n\n"
        "decomp = ATen allow-list decomposition (best when the op is a "
        "common ATen tensor op).\n"
        "translation = external-call lowering (best when the op has a "
        "simple Tensor-in / Tensor-out schema).\n"
        "blackbox = opaque boundary (falls back to the eager op at "
        "runtime; choose when the op is quantised or has complex "
        "side-effects).\n\n"
        "Respond with ONLY the single strategy word, optionally "
        "followed by a short rationale on the next line."
    )

    request = GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="",
            target_profile_summary="",
            available_transforms=["decomp", "translation", "blackbox"],
            kernel_contracts=[],
            objective="latency",  # type: ignore[arg-type]
        ),
        config=LLMConfig(
            model=str(getattr(llm_client, "model", "default")),
            temperature=0.1,
            max_tokens=128,
        ),
    )

    try:
        response = llm_client.generate(request)
    except Exception as exc:  # noqa: BLE001
        return "fallback", f"llm_call_failed: {exc}"

    text = (response.raw_text or "").strip().lower()
    first_word = text.split()[0] if text else ""
    # Allow the LLM to say "blackbox", "black box", "black-box"...
    if "decomp" in first_word:
        return "decomp", text
    if "translation" in first_word or "translate" in first_word:
        return "translation", text
    if "blackbox" in first_word.replace("-", "").replace(" ", ""):
        return "blackbox", text
    return "fallback", f"unrecognised_llm_answer: {text!r}"


def _deterministic_default(
    resolution: UnsupportedOpResolution,
) -> str:
    """The rule we fall back on when the LLM isn't consulted or errs."""
    cls = resolution.classification
    if cls.strategy == "known_payload_decomposition":
        return "none"
    if cls.strategy == "synthesized_external_call":
        return "translation"
    if resolution.dossier.is_aten:
        return "decomp"
    return "blackbox"


def _apply_strategy(
    strategy: str,
    resolution: UnsupportedOpResolution,
) -> tuple[bool, str, str]:
    """Try to apply ``strategy``; return (ok, detail, error)."""
    if strategy == "none":
        return True, "already covered by registered decomposition", ""

    if strategy == "decomp":
        decomp = synthesize_export_decomposition(
            resolution.target,
            resolution.dossier,
        )
        if decomp is None:
            return False, "", "not_on_allow_list"
        return True, decomp.description, ""

    if strategy == "translation":
        from compgen.capture.unsupported.classify import UnsupportedClassification

        # Try both the existing classification and a forced eligibility.
        forced = UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="low",
            reason="forced by LLM-driven recovery orchestrator",
        )
        translation = synthesize_payload_translation(
            resolution.issue,
            resolution.dossier,
            resolution.classification if resolution.classification.strategy == "synthesized_external_call" else forced,
        )
        if translation is None:
            return False, "", "translation_not_eligible"
        return True, f"{translation.kind}:{translation.callee_name}", ""

    if strategy == "blackbox":
        # Blackbox registration is metadata only — always ok.
        return True, f"blackbox cache_key={resolution.promotion.cache_key}", ""

    return False, "", f"unknown_strategy:{strategy}"


def plan_recovery(
    artifact: CaptureArtifact,
    *,
    llm_client: CompGenLLMProtocol | None = None,
    consult_llm_on: tuple[str, ...] = ("low",),
) -> RecoveryPlan:
    """Decide + apply a recovery strategy for every unsupported op.

    Args:
        artifact: The capture artifact from :func:`capture_frontend_artifact`.
        llm_client: Optional LLM backend. When absent, the deterministic
            classifier rules are used.
        consult_llm_on: Confidence levels that trigger an LLM consult.
            Defaults to ``("low",)`` — the classifier only calls the LLM
            on ambiguous cases.

    Returns:
        A :class:`RecoveryPlan` with one :class:`OpRecoveryDecision` per
        target. ``plan.ok()`` is True iff every decision applied cleanly.
    """
    plan = RecoveryPlan()

    for resolution in artifact.unsupported_resolutions:
        strategy = _deterministic_default(resolution)
        source = "classifier"

        cls_conf = resolution.classification.confidence
        if llm_client is not None and cls_conf in consult_llm_on and strategy != "none":
            picked, reasoning = _llm_pick_strategy(resolution, llm_client)
            plan.llm_consulted += 1
            log.debug(
                "recovery.llm_consulted",
                target=resolution.target,
                picked=picked,
                reasoning=reasoning[:80],
            )
            if picked == "fallback":
                source = "fallback"
            else:
                strategy = picked
                source = "llm"

        ok, detail, error = _apply_strategy(strategy, resolution)

        # If the LLM's pick failed to apply, retry once with the
        # deterministic default so we don't lose the resolution.
        if not ok and source == "llm":
            fallback = _deterministic_default(resolution)
            if fallback != strategy:
                ok2, detail2, error2 = _apply_strategy(fallback, resolution)
                if ok2:
                    strategy = fallback
                    source = "fallback"
                    ok, detail, error = ok2, detail2, ""
                else:
                    error = f"{error}; fallback={error2}"

        plan.decisions.append(
            OpRecoveryDecision(
                target=resolution.target,
                strategy=strategy,
                source=source,
                ok=ok,
                detail=detail,
                error=error,
            )
        )
        if not ok:
            plan.skipped += 1

    return plan


__all__ = [
    "OpRecoveryDecision",
    "RecoveryPlan",
    "plan_recovery",
]
