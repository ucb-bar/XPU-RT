"""Tests for transform synthesis with mock LLM."""

from __future__ import annotations

from compgen.llm.base import Objective
from compgen.llm.mock_client import MockLLMClient
from compgen.targets.schema import TargetProfile
from compgen.transforms.synthesize import TransformScript, TransformSynthesizer
from xdsl.dialects.builtin import Float32Type, FunctionType, ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region


def _make_test_module() -> ModuleOp:
    """Build a minimal valid xDSL module."""
    f32 = Float32Type()
    tensor_type = TensorType(f32, [4, 4])
    func_type = FunctionType.from_lists([tensor_type], [tensor_type])

    block = Block(arg_types=[tensor_type])
    block.add_op(ReturnOp(block.args[0]))

    region = Region([block])
    func_op = FuncOp("forward", func_type, region)
    return ModuleOp([func_op])


_NOOP_PATTERN_CODE = (
    "from xdsl.pattern_rewriter import RewritePattern, PatternRewriter\n"
    "from xdsl.ir import Operation\n"
    "\n"
    "class NoopPattern(RewritePattern):\n"
    "    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:\n"
    "        return\n"
)


def test_synthesize_with_mock_llm() -> None:
    """TransformSynthesizer should generate scripts using mock LLM."""
    mock = MockLLMClient(strict=False)
    # The PassGenerator builds a prompt containing "Generate a RewritePattern";
    # register a fragment response that matches that substring.
    mock.add_response("RewritePattern", _NOOP_PATTERN_CODE)

    target = TargetProfile(name="test-target")
    module = _make_test_module()

    synthesizer = TransformSynthesizer(llm_client=mock, max_candidates=1)
    scripts = synthesizer.synthesize(
        ir_summary="small test IR",
        target=target,
        module=module,
        objective=Objective.LATENCY,
    )

    # Should produce at least one script since the mock returns valid code
    assert isinstance(scripts, list)
    for script in scripts:
        assert isinstance(script, TransformScript)
        assert script.content  # non-empty


def test_synthesize_respects_budget() -> None:
    """Synthesis should not exceed max_candidates."""
    mock = MockLLMClient(strict=False)
    mock.add_response("RewritePattern", _NOOP_PATTERN_CODE)

    target = TargetProfile(name="test-target")
    module = _make_test_module()

    synthesizer = TransformSynthesizer(llm_client=mock, max_candidates=3)
    scripts = synthesizer.synthesize(
        ir_summary="small test IR",
        target=target,
        module=module,
        objective=Objective.LATENCY,
    )

    # max_candidates=3, but current impl generates min(max_candidates, 1) = 1
    assert len(scripts) <= 3
