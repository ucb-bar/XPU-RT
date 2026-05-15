"""Tests for ``compgen.agent.hw_aware_dispatch``.

Locks in:
  * deterministic fallback (no LLM) returns oracle's prior + a
    legality-validated decision per target
  * single-target single-op MICRO/NORMAL choice flows through
  * MEGA dispatch model is rejected on a CPU adapter and the
    decision falls back to NORMAL
  * the LLM path overrides oracle when the response is parseable +
    legal; falls back to oracle when LLM picks an illegal granularity
  * best_target selection picks the LLM's pick if legal, else the
    highest-throughput legal target
"""

from __future__ import annotations

import pytest
from compgen.agent.hw_aware_dispatch import (
    MultiTargetDispatchDecision,
    decide_dispatch,
)
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    Granularity,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    OrchestrationSpec,
    ShapeClass,
    TensorIO,
)
from compgen.llm.base import (
    GenerationRequest,
    GenerationResponse,
    Objective,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _envelope(name: str, *, peak: float = 672.0, lanes: int = 64) -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name=name,
        vector_lanes=lanes,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16", "f32"),
        peak_bandwidth_gbps=peak,
    )


def _normal_pointwise(target: str = "cuda-a100") -> KernelContractV3:
    env = _envelope(target)
    return KernelContractV3(
        op_name="addf",
        archetype=KernelArchetype.POINTWISE,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_decide_dispatch_requires_at_least_one_envelope() -> None:
    with pytest.raises(ValueError, match="at least one HardwareEnvelope"):
        decide_dispatch([_normal_pointwise()], envelopes=[])


# ---------------------------------------------------------------------------
# Deterministic fallback (no LLM)
# ---------------------------------------------------------------------------


def test_decide_dispatch_no_llm_uses_oracle_prior() -> None:
    region = [_normal_pointwise("cuda-a100")]
    env = _envelope("cuda-a100")
    out = decide_dispatch(region, envelopes=[env])
    assert isinstance(out, MultiTargetDispatchDecision)
    assert "cuda-a100" in out.per_target
    decision = out.per_target["cuda-a100"]
    assert decision.target == "cuda-a100"
    assert decision.adapter_name == "cuda"
    # Oracle picks NORMAL/MICRO depending on register-fit. f32 with
    # unknown dim → working set is small, so MICRO is plausible.
    assert decision.granularity in (Granularity.NORMAL, Granularity.MICRO)
    assert decision.legal
    assert out.used_llm is False


def test_decide_dispatch_picks_highest_throughput_when_no_llm() -> None:
    """With two legal targets, the heuristic ranker prefers the one
    with higher peak bandwidth."""
    region = [_normal_pointwise("cuda-a100")]
    fast = _envelope("cuda-a100", peak=2000.0)
    slow = _envelope("cpu-host", peak=50.0)
    out = decide_dispatch(region, envelopes=[fast, slow])
    assert out.best_target == "cuda-a100"


# ---------------------------------------------------------------------------
# Legality enforcement
# ---------------------------------------------------------------------------


def test_decide_dispatch_falls_back_to_normal_when_micro_illegal_on_cpu() -> None:
    """CPU adapter rejects INLINE (MICRO). The oracle prior on a small
    pointwise often picks MICRO; the legality fallback chain must
    coerce to NORMAL (universally supported) and mark the decision
    legal with a rationale describing the fallback."""
    a = _normal_pointwise("cpu-host")
    cpu_env = _envelope("cpu-host", peak=50.0)
    out = decide_dispatch([a], envelopes=[cpu_env])
    decision = out.per_target["cpu-host"]
    assert decision.legal
    # Oracle could pick NORMAL outright OR MICRO (then fallback to NORMAL).
    assert decision.granularity is Granularity.NORMAL
    if "falling back to" in decision.rationale:
        assert "normal" in decision.rationale.lower()


# ---------------------------------------------------------------------------
# LLM-driven path
# ---------------------------------------------------------------------------


class _StaticLLM:
    """Tiny LLM mock that returns a fixed JSON dispatch decision."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import json as _json

        return GenerationResponse(
            raw_text=_json.dumps(self._payload),
            parsed_artifacts=[],
            model_id="static-mock",
        )

    def generate_structured(self, request, schema):  # noqa: D401, ANN001
        return self.generate(request)


def test_llm_decision_overrides_oracle_when_legal() -> None:
    region = [_normal_pointwise("cuda-a100")]
    env = _envelope("cuda-a100")
    llm = _StaticLLM(
        {
            "per_target": {
                "cuda-a100": {
                    "granularity": "normal",
                    "rationale": "LLM picked NORMAL because boundary fusion enabled",
                }
            },
            "best_target": "cuda-a100",
            "best_rationale": "only target available",
        }
    )
    out = decide_dispatch(region, envelopes=[env], llm=llm)
    assert out.used_llm is True
    decision = out.per_target["cuda-a100"]
    assert decision.granularity is Granularity.NORMAL
    assert "LLM picked NORMAL" in decision.rationale


def test_llm_picking_illegal_granularity_falls_back_through_chain() -> None:
    """LLM picks MEGA on CPU (illegal); fallback chain coerces through
    oracle prior and finally NORMAL (universally legal)."""
    a = _normal_pointwise("cpu-host")
    cpu_env = _envelope("cpu-host", peak=50.0)
    llm = _StaticLLM(
        {
            "per_target": {"cpu-host": {"granularity": "mega", "rationale": "fictitious mega"}},
            "best_target": "cpu-host",
            "best_rationale": "only target",
        }
    )
    out = decide_dispatch([a], envelopes=[cpu_env], llm=llm)
    decision = out.per_target["cpu-host"]
    # NORMAL is the universal fallback that CPU always supports.
    assert decision.granularity is Granularity.NORMAL
    assert decision.legal
    assert "falling back" in decision.rationale


def test_llm_response_with_garbage_text_falls_back_to_oracle() -> None:
    region = [_normal_pointwise("cuda-a100")]
    env = _envelope("cuda-a100")

    class _GarbageLLM:
        def generate(self, request):
            return GenerationResponse(
                raw_text="lorem ipsum, no JSON here",
                parsed_artifacts=[],
                model_id="garbage",
            )

        def generate_structured(self, *a, **kw):
            return self.generate(None)

    out = decide_dispatch(region, envelopes=[env], llm=_GarbageLLM())
    # used_llm stays False because the JSON parse failed.
    assert out.used_llm is False
    assert out.per_target["cuda-a100"].legal


# ---------------------------------------------------------------------------
# Best-target selection across multiple targets
# ---------------------------------------------------------------------------


def test_llm_best_target_overrides_throughput_ranker_when_legal() -> None:
    region = [_normal_pointwise("cuda-a100")]
    cuda = _envelope("cuda-a100", peak=2000.0)
    cpu = _envelope("cpu-host", peak=50.0)
    llm = _StaticLLM(
        {
            "per_target": {
                "cuda-a100": {"granularity": "normal", "rationale": "fast"},
                "cpu-host": {"granularity": "normal", "rationale": "fits budget"},
            },
            "best_target": "cpu-host",
            "best_rationale": "energy budget tight; CPU is enough",
        }
    )
    out = decide_dispatch(region, envelopes=[cuda, cpu], llm=llm)
    assert out.best_target == "cpu-host"
    assert "energy" in out.best_rationale.lower()


def test_per_target_decisions_carry_oracle_prior() -> None:
    region = [_normal_pointwise("cuda-a100")]
    env = _envelope("cuda-a100")
    out = decide_dispatch(region, envelopes=[env])
    decision = out.per_target["cuda-a100"]
    assert decision.deterministic_prior is not None
    # The fallback rationale should include the oracle's reason.
    assert decision.rationale  # non-empty


def test_objective_is_passed_through_to_prompt_with_llm() -> None:
    """When the LLM is called, the prompt mentions the objective. We
    indirectly verify by capturing the prompt text in a custom mock."""
    region = [_normal_pointwise("cuda-a100")]
    env = _envelope("cuda-a100")

    captured: dict[str, str] = {}

    class _CapturingLLM:
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            captured["prompt"] = request.prompt_template
            import json as _json

            return GenerationResponse(
                raw_text=_json.dumps(
                    {
                        "per_target": {"cuda-a100": {"granularity": "normal", "rationale": "x"}},
                        "best_target": "cuda-a100",
                        "best_rationale": "y",
                    }
                ),
                parsed_artifacts=[],
                model_id="cap",
            )

        def generate_structured(self, *a, **kw):
            return self.generate(*a, **kw)

    decide_dispatch(region, envelopes=[env], objective=Objective.ENERGY, llm=_CapturingLLM())
    assert "energy" in captured["prompt"].lower()
