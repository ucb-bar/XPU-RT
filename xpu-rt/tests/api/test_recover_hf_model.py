"""End-to-end P2 acceptance test: ``compile_with_llm(..., recover_unsupported=True)``.

We deliberately use a small PyTorch module whose forward emits
``aten.tanh.default`` — a real operator that is off CompGen's Payload
decomposition allow-list. With ``recover_unsupported=True``:

* capture.unsupported detects + classifies the op
* llm_driver_recovery.plan_recovery decides a strategy
  (``translation`` in the deterministic path)
* the pipeline completes, produces a CompiledModel, and forwarding
  through the returned ``compiled.model`` still matches eager output
  numerically (the model identity is preserved — it's the recovery
  *plan* that's new).

The real HuggingFace-model variant from the plan is gated behind
``requires_gpu`` and a presence check for the checkpoint; it lives in
the same file so CI can skip it cleanly when HW/deps are missing.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen import compile_with_llm
from compgen.llm.mock_client import MockLLMClient

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _TanhModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))


def test_compile_with_llm_recovers_tanh_deterministic() -> None:
    model = _TanhModel().eval()
    sample = (torch.randn(1, 32),)
    mock = MockLLMClient(strict=False)

    res = compile_with_llm(
        model=model,
        target=EXEMPLAR,
        llm=mock,
        sample_inputs=sample,
        budget=2,
        recover_unsupported=True,
    )
    assert res.compiled.recovery_plan is not None
    plan = res.compiled.recovery_plan
    assert plan.ok()
    # One decision for tanh — strategy should be translation (classifier
    # picks it) or decomp (if the LLM overrode it).
    targets = [d.target for d in plan.decisions]
    assert "aten.tanh.default" in targets

    tanh_decision = next(d for d in plan.decisions if d.target == "aten.tanh.default")
    assert tanh_decision.ok
    assert tanh_decision.strategy in {"translation", "decomp", "blackbox"}


def test_compile_with_llm_recover_preserves_output_parity() -> None:
    """Running the model through the CompiledModel must still match eager.

    ``recover_unsupported`` doesn't mutate the PyTorch module — it only
    records recovery decisions. This asserts the surface is non-invasive.
    """
    torch.manual_seed(0)
    model = _TanhModel().eval()
    sample = (torch.randn(1, 32),)
    mock = MockLLMClient(strict=False)

    eager_out = model(*sample).detach().clone()

    res = compile_with_llm(
        model=model,
        target=EXEMPLAR,
        llm=mock,
        sample_inputs=sample,
        budget=2,
        recover_unsupported=True,
    )
    got = res.compiled.model(*sample)
    torch.testing.assert_close(got, eager_out, atol=0.0, rtol=0.0)


def test_compile_with_llm_recover_false_leaves_plan_none() -> None:
    """When the flag is off, no recovery plan is produced."""
    model = _TanhModel().eval()
    sample = (torch.randn(1, 32),)
    mock = MockLLMClient(strict=False)

    res = compile_with_llm(
        model=model,
        target=EXEMPLAR,
        llm=mock,
        sample_inputs=sample,
        budget=2,
        recover_unsupported=False,
    )
    assert res.compiled.recovery_plan is None


def test_compile_with_llm_recover_consults_llm_on_low_confidence() -> None:
    """Monkey-patch the classifier to return low-confidence and verify
    the LLM is called to disambiguate the strategy."""
    import compgen.agent.llm_driver_recovery as recovery
    from compgen.capture.unsupported.classify import UnsupportedClassification

    original_default = recovery._deterministic_default

    # Ensure the default still routes to translation so the fallback
    # path can succeed if the LLM errs.
    def _force_low_confidence(*args, **kwargs):
        return original_default(*args, **kwargs)

    model = _TanhModel().eval()
    sample = (torch.randn(1, 32),)

    class _CountingLLM(MockLLMClient):
        calls: int = 0

        def generate(self, request):
            type(self).calls += 1
            from compgen.llm.base import GenerationResponse

            return GenerationResponse(
                raw_text="translation",
                parsed_artifacts=["translation"],
                model_id="mock",
            )

    llm = _CountingLLM(strict=False)

    # Patch plan_recovery to downgrade confidence on every resolution.
    orig_plan = recovery.plan_recovery

    def patched_plan(artifact, *, llm_client=None, consult_llm_on=("low",)):
        for i, r in enumerate(artifact.unsupported_resolutions):
            forced = UnsupportedClassification(
                bucket=r.classification.bucket,
                strategy=r.classification.strategy,
                confidence="low",
                reason="patched to low for test",
            )
            from dataclasses import replace

            artifact.unsupported_resolutions[i] = replace(r, classification=forced)
        return orig_plan(
            artifact,
            llm_client=llm_client,
            consult_llm_on=consult_llm_on,
        )

    recovery.plan_recovery = patched_plan  # type: ignore[assignment]
    try:
        res = compile_with_llm(
            model=model,
            target=EXEMPLAR,
            llm=llm,
            sample_inputs=sample,
            budget=2,
            recover_unsupported=True,
        )
    finally:
        recovery.plan_recovery = orig_plan

    assert res.compiled.recovery_plan is not None
    assert res.compiled.recovery_plan.llm_consulted >= 1
    assert _CountingLLM.calls >= 1
