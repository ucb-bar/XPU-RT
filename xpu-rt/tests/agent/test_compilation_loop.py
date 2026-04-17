"""Tests for the agentic compilation loop."""

from __future__ import annotations

from types import SimpleNamespace

from compgen.agent.analyzer import GraphAnalysisDossier, NetworkAnalysis, RegionDossier
from compgen.agent.loop import AgenticCompilationLoop, CompilationResult
from compgen.agent.env import CompilerEnv
from compgen.llm.base import GenerationRequest, GenerationResponse
from compgen.llm.mock_client import MockLLMClient
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, c)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx, idx], [idx]), Region([block]))])


def _make_mock_client() -> MockLLMClient:
    """Create a mock LLM that returns valid analysis + refinement responses."""
    client = MockLLMClient(strict=False)
    # Analysis response
    analysis = '[{"action_type": "eqsat", "target": "all", "reason": "algebraic", "expected_improvement": 5.0}]'
    client.add_response("optimization", analysis)
    # Refinement response
    refine = '{"action_type": "eqsat", "target_region": "", "parameters": {}, "reasoning": "try more"}'
    client.add_response("iteratively", refine)
    return client


def test_compilation_loop_runs() -> None:
    """The agentic loop runs without error on a simple module."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = _make_mock_client()
    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=5)
    result = loop.run(target)

    assert isinstance(result, CompilationResult)
    assert result.iterations_run >= 0
    assert result.initial_cost_us >= 0


def test_compilation_loop_tracks_history() -> None:
    """The loop records iteration history."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = _make_mock_client()
    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=3)
    result = loop.run(target)

    assert len(result.history) == result.iterations_run
    for record in result.history:
        assert record.action_type


def test_compilation_loop_stops_on_noop() -> None:
    """The loop stops when LLM suggests noop."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = MockLLMClient(strict=False)
    # Return empty analysis → no proposals → falls to refinement → returns noop
    client.add_response("optimization", "[]")
    client.add_response("iteratively", '{"action_type": "noop"}')

    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=10)
    result = loop.run(target)
    # Should stop early (not run all 10 iterations)
    assert result.iterations_run < 10


def test_compilation_loop_with_real_eqsat() -> None:
    """The loop can drive real eqsat optimization."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = MockLLMClient(strict=False)
    resp = '[{"action_type": "eqsat", "target": "all", "reason": "commute", "expected_improvement": 2.0}]'
    client.add_response("optimization", resp)

    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=2)
    result = loop.run(target)

    # At least 1 iteration should have applied eqsat
    eqsat_iters = [r for r in result.history if r.action_type == "eqsat"]
    assert len(eqsat_iters) >= 1


def test_compilation_loop_passes_frontend_and_dossier_context_to_llm() -> None:
    """The loop should expose capture and dossier evidence through structured LLM requests."""

    class _SpyLLMClient:
        model = "spy-model"

        def __init__(self) -> None:
            self.structured_requests: list[tuple[GenerationRequest, dict[str, object]]] = []
            self.raw_requests: list[GenerationRequest] = []

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.raw_requests.append(request)
            return GenerationResponse(raw_text='{"action_type":"noop","target_region":"","parameters":{},"reasoning":"stop"}',
                                      parsed_artifacts=[], model_id=self.model)

        def generate_structured(
            self, request: GenerationRequest, schema: dict[str, object]
        ) -> GenerationResponse:
            self.structured_requests.append((request, schema))
            if request.prompt_template.startswith("You are an expert ML compiler optimizer"):
                return GenerationResponse(
                    raw_text="[]",
                    parsed_artifacts=["[]"],
                    model_id=self.model,
                )
            return GenerationResponse(
                raw_text='{"action_type":"noop","target_region":"","parameters":{},"reasoning":"stop"}',
                parsed_artifacts=['{"action_type":"noop","target_region":"","parameters":{},"reasoning":"stop"}'],
                model_id=self.model,
            )

    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    dossier = GraphAnalysisDossier(
        model_name="test-model",
        op_histogram={"arith.addi": 1, "arith.muli": 1},
        repeated_patterns={"elementwise_chain": 2},
        total_regions=1,
        total_flops=2,
        total_bytes=24,
        critical_path=("r0",),
        independent_region_sets=(("r0",),),
        dynamic_shape_regions=(),
        unsupported_targets=("aten.sin.default",),
        regions=(
            RegionDossier(
                region_id="r0",
                kind="elementwise_chain",
                node_names=("add", "mul"),
                repeated_count=2,
                flops=2,
                bytes=24,
                arithmetic_intensity=0.1,
                dynamic_shapes=False,
                producers=(),
                consumers=(),
                parallelizable_with=("r1",),
                layout_candidates=("rowmajor",),
                backend_viability=("cpu", "triton"),
                best_device="cpu",
                local_memory_fit={"cpu": True},
            ),
        ),
    )
    env._analysis = NetworkAnalysis(
        model_name="test-model",
        total_params=0,
        total_flops=2,
        total_bytes=24,
        clusters=[],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=[],
        optimization_opportunities=[],
        dossier=dossier,
    )
    env._capture_artifact = SimpleNamespace(
        diagnostics=SimpleNamespace(graph_breaks=["break"], guard_observations=["guard"]),
        unsupported_resolutions=[SimpleNamespace(target="aten.sin.default")],
    )

    client = _SpyLLMClient()
    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=1)
    result = loop.run(target)

    assert isinstance(result, CompilationResult)
    assert len(client.structured_requests) >= 2
    request, _schema = client.structured_requests[0]
    assert "graph_breaks=1" in request.context.frontend_diagnostics_summary
    assert "repeated_patterns=elementwise_chain:2" in request.context.analysis_dossier_summary
    assert "aten.sin.default" in request.context.unsupported_operator_summary
    assert '"graph_break_count": 1' in request.context.evidence_json


def test_compilation_loop_skips_repair_when_verification_has_no_counterexample() -> None:
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = MockLLMClient(strict=False)
    client.add_response(
        "optimization",
        '[{"action_type": "eqsat", "target": "all", "reason": "algebraic", "expected_improvement": 1.0}]',
    )

    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=1)
    loop._run_per_step_verification = lambda action, obs, target: {  # type: ignore[method-assign]
        "passed": False,
        "status": "unknown",
        "region_id": action.region_id,
        "counterexample": None,
    }

    result = loop.run(target)
    assert result.best_observation is not None
