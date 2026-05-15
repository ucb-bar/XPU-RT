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
class RegionReplanEvent:
    """One typed replan event recorded by the G3 wire-in.

    Captures what happened when a per-op recovery failure was routed
    through :func:`compgen.agent.plan.replan_on_reject`. The event
    carries the rejection_class, the rung that was walked, and the new
    plan version — enough for an evidence-pack consumer to reconstruct
    the ladder traversal without re-running the recovery loop.
    """

    target: str
    region_id: str
    rejection_class: str  # tactic_fatal | tactic_recoverable | surprising
    rung_before: str
    rung_after: str
    plan_version_before: int
    plan_version_after: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "region_id": self.region_id,
            "rejection_class": self.rejection_class,
            "rung_before": self.rung_before,
            "rung_after": self.rung_after,
            "plan_version_before": self.plan_version_before,
            "plan_version_after": self.plan_version_after,
            "detail": self.detail,
        }


@dataclass
class RecoveryPlan:
    """Aggregate: how every unsupported op will be handled."""

    decisions: list[OpRecoveryDecision] = field(default_factory=list)
    llm_consulted: int = 0
    skipped: int = 0
    # G3 wire-in: typed replan events recorded when a Plan-driven
    # recovery walked the fallback ladder. Empty when the caller
    # did not supply a region Plan (legacy path).
    region_replan_events: list[RegionReplanEvent] = field(default_factory=list)

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
            "region_replan_events": [
                e.to_dict() for e in self.region_replan_events
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
    region_plan: Any | None = None,
    region_id_for_target: dict[str, str] | None = None,
) -> RecoveryPlan:
    """Decide + apply a recovery strategy for every unsupported op.

    Args:
        artifact: The capture artifact from :func:`capture_frontend_artifact`.
        llm_client: Optional LLM backend. When absent, the deterministic
            classifier rules are used.
        consult_llm_on: Confidence levels that trigger an LLM consult.
            Defaults to ``("low",)`` — the classifier only calls the LLM
            on ambiguous cases.
        region_plan: Optional :class:`compgen.agent.plan.Plan` whose
            fallback ladders describe per-region recovery rungs. When
            supplied, per-op failures trigger
            :func:`compgen.agent.plan.replan_on_reject` and the walk
            is logged in :attr:`RecoveryPlan.region_replan_events`.
            Backward-compatible: omit to get the legacy flat recovery.
        region_id_for_target: Optional map from ``resolution.target``
            (the FX node name) to its enclosing region id. Used only
            when ``region_plan`` is supplied. Targets without a region
            mapping are skipped for replan accounting (the original
            recovery decision is still recorded).

    Returns:
        A :class:`RecoveryPlan` with one :class:`OpRecoveryDecision` per
        target. ``plan.ok()`` is True iff every decision applied cleanly.
    """
    plan = RecoveryPlan()
    region_id_for_target = region_id_for_target or {}

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
            # G3 wire-in: when a region Plan is supplied, walk the
            # fallback ladder for this region. The rejection_class is
            # `tactic_fatal` for a hard recovery failure (no apply
            # path landed) — the Strategist should drop this rung.
            if region_plan is not None:
                rid = region_id_for_target.get(resolution.target, "")
                if rid:
                    region_plan = _record_region_replan(
                        plan, region_plan, rid, resolution.target,
                        rejection_class="tactic_fatal",
                        detail=error or "recovery_strategy_did_not_apply",
                    )

    return plan


def _record_region_replan(
    recovery: RecoveryPlan,
    region_plan: Any,
    region_id: str,
    target: str,
    *,
    rejection_class: str,
    detail: str,
) -> Any:
    """Apply :func:`compgen.agent.plan.replan_on_reject` and record
    the event on the RecoveryPlan. Returns the new plan version so
    subsequent rejections walk further down the ladder.

    On a malformed input (unknown region, unknown rejection class)
    the helper returns the input plan unchanged and records nothing —
    the existing recovery decision is the canonical signal.
    """

    try:
        from compgen.agent.plan import PlanError, replan_on_reject
    except ImportError:
        return region_plan

    try:
        region_before = region_plan.get_region(region_id)
    except PlanError:
        return region_plan

    try:
        new_plan = replan_on_reject(
            region_plan, region_id=region_id, rejection_class=rejection_class
        )
    except PlanError:
        return region_plan

    try:
        region_after = new_plan.get_region(region_id)
    except PlanError:
        return new_plan

    recovery.region_replan_events.append(
        RegionReplanEvent(
            target=target,
            region_id=region_id,
            rejection_class=rejection_class,
            rung_before=region_before.tactic,
            rung_after=region_after.tactic,
            plan_version_before=region_plan.plan_version,
            plan_version_after=new_plan.plan_version,
            detail=detail,
        )
    )
    return new_plan


__all__ = [
    "OpRecoveryDecision",
    "RecoveryPlan",
    "RegionReplanEvent",
    "plan_recovery",
]
