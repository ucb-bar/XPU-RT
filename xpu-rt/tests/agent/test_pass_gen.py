"""Tests for LLM-driven pass generation and verification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from compgen.agent.env import CompilerEnv, GeneratePassAction
from compgen.agent.pass_gen import PassGenerator
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.llm import create_llm_client
from compgen.llm.gemini_client import GeminiClient
from compgen.llm.mock_client import MockLLMClient
from compgen.targets.schema import load_profile

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


def _get_module_and_ep():
    sys.path.insert(0, str(EXAMPLES / "models"))
    from simple_mlp import SimpleMLP, get_sample_inputs

    ep = capture_model(SimpleMLP(), get_sample_inputs())
    module, _ = fx_to_xdsl(ep)
    return module, ep


def _get_target():
    return load_profile(EXAMPLES / "target_profiles" / "cuda_a100.yaml")


# ---- PassGenerator unit tests ----


def test_pass_generator_valid_code() -> None:
    """PassGenerator should validate hand-written valid code."""
    module, _ = _get_module_and_ep()

    # Manually test validation with known-good code
    gen = PassGenerator(llm_client=GeminiClient())  # client unused for manual test

    code = """
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter
from xdsl.ir import Operation
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.builtin import StringAttr

class TagMatmulPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, MatmulOp):
            return
        op.attributes["compgen.tagged"] = StringAttr("by_test")
"""
    result = gen._validate("tag_matmul", code, module)
    assert result.verified, f"Should be verified: {result.verification_error}"
    assert result.pattern_class is not None


def test_pass_generator_invalid_code() -> None:
    """PassGenerator should reject code that doesn't parse."""
    module, _ = _get_module_and_ep()
    gen = PassGenerator(llm_client=GeminiClient())

    result = gen._validate("bad_code", "def this is not valid python!!!", module)
    assert not result.verified
    assert "Syntax error" in result.verification_error


def test_pass_generator_no_pattern_class() -> None:
    """PassGenerator should reject code without a RewritePattern subclass."""
    module, _ = _get_module_and_ep()
    gen = PassGenerator(llm_client=GeminiClient())

    result = gen._validate("no_pattern", "x = 42\nprint(x)", module)
    assert not result.verified
    assert "No RewritePattern" in result.verification_error


def test_pass_generator_injects_guard_ref() -> None:
    module, _ = _get_module_and_ep()
    gen = PassGenerator(llm_client=GeminiClient())

    code = """
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter
from xdsl.ir import Operation

class GuardedNoopPattern(RewritePattern):
    def __init__(self, guard_ref=None):
        self.guard_ref = guard_ref

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        return
"""
    result = gen._validate("guarded_noop", code, module, guard_refs=("guard.local_mem",))
    assert result.verified
    assert result.guard_refs == ("guard.local_mem",)


# ---- Real LLM generation test ----


def test_generate_pass_via_llm() -> None:
    """Opt-in real LLM call to generate a pass."""
    if os.environ.get("COMPGEN_RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("Set COMPGEN_RUN_REAL_LLM_TESTS=1 to enable real LLM smoke tests.")
    module, _ = _get_module_and_ep()
    backend = os.environ.get("COMPGEN_REAL_LLM_BACKEND", "gemini")
    model = os.environ.get("COMPGEN_REAL_LLM_MODEL")
    client = create_llm_client(backend, model=model, working_dir=Path.cwd())
    gen = PassGenerator(llm_client=client)

    result = gen.generate(
        description="Add a 'compgen.optimized' annotation to all MatmulOp operations",
        target_pattern="linalg.MatmulOp",
        expected_effect="Each MatmulOp gets a StringAttr 'compgen.optimized' = 'true'",
        module=module,
    )

    # The LLM might or might not produce valid code, but it should not crash
    assert result.source_code != ""
    if result.verified:
        assert result.pattern_class is not None
        # Apply it
        success, err = gen.apply_generated_pass(result.name, module.clone())
        assert success, f"Application failed: {err}"


# ---- Integration via CompilerEnv ----


def test_generate_pass_action_in_env() -> None:
    """GeneratePassAction should call LLM and validate the result."""
    module, ep = _get_module_and_ep()
    env = CompilerEnv()
    env.reset(module, _get_target(), exported_program=ep, budget=5)
    client = MockLLMClient(strict=False)
    client.add_response(
        "You are generating an xDSL compiler pass",
        """```python
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter
from xdsl.ir import Operation

class NoopPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        return
```""",
    )
    env.attach_llm_client(client)

    result = env.step(
        GeneratePassAction(
            description="Tag all MatmulOp with a 'compgen.llm_generated' StringAttr annotation",
            target_pattern="MatmulOp",
            expected_effect="MatmulOps get compgen.llm_generated attribute",
        )
    )

    # Should have diagnostics regardless of success/failure
    assert len(result.info.diagnostics) > 0
    assert any("PASS_GEN" in d for d in result.info.diagnostics)


def test_generate_pass_without_description_fails() -> None:
    """GeneratePassAction without description should fail."""
    module, ep = _get_module_and_ep()
    env = CompilerEnv()
    env.reset(module, _get_target(), exported_program=ep)

    result = env.step(GeneratePassAction(description=""))
    assert not result.info.action_applied
    assert "description" in result.info.error.lower()
