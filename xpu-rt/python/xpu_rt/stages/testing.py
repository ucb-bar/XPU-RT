"""Non-negotiable test suite for compilation stages.

Every (stage, target) combination must pass these tests.
These are the **contracts** — a generated plugin is only promoted
if ALL contract tests pass.

Usage::

    class TestEncodingCudaA100(StageContractTestSuite):
        @pytest.fixture(autouse=True)
        def setup(self, cuda_a100_target, simple_mlp_module):
            self.stage = EncodingStage()
            self.target = cuda_a100_target
            self.capabilities = infer_capabilities(cuda_a100_target)
            self.sample_module = simple_mlp_module
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp

from compgen.stages.base import CompilationStage, StageResult
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


class StageContractTestSuite:
    """Non-negotiable test suite that every stage + target must pass.

    Subclasses set ``stage``, ``target``, ``capabilities``, and
    ``sample_module`` as instance attributes (e.g., via a pytest fixture).
    """

    stage: CompilationStage
    target: TargetProfile
    capabilities: CapabilitySpec
    sample_module: ModuleOp

    def test_input_contract_accepts_valid_ir(self) -> None:
        """The input contract must accept canonical IR."""
        violations = self.stage.verify_contract(
            self.sample_module, self.stage.input_contract()
        )
        assert violations == [], f"Valid IR rejected by input contract: {violations}"

    def test_output_contract_holds_after_run(self) -> None:
        """After running, the output contract must hold."""
        result = self.stage.run(
            self.sample_module.clone(), self.target, self.capabilities
        )
        assert result.passed, f"Stage failed: {result.contract_violations}"
        if result.module is not None:
            violations = self.stage.verify_contract(
                result.module, self.stage.output_contract()
            )
            assert violations == [], f"Output contract violated: {violations}"

    def test_xdsl_verifier_passes(self) -> None:
        """The output IR must pass the xDSL structural verifier."""
        result = self.stage.run(
            self.sample_module.clone(), self.target, self.capabilities
        )
        if result.module is not None:
            result.module.verify()

    def test_semantic_preservation(self) -> None:
        """The stage must preserve function signatures."""
        original = self.sample_module.clone()
        result = self.stage.run(
            self.sample_module.clone(), self.target, self.capabilities
        )
        if result.module is not None:
            orig_funcs = [op for op in original.walk() if isinstance(op, FuncOp)]
            trans_funcs = [op for op in result.module.walk() if isinstance(op, FuncOp)]
            assert len(orig_funcs) == len(trans_funcs), "Function count changed"
            for of, tf in zip(orig_funcs, trans_funcs):
                assert of.function_type == tf.function_type, (
                    f"Function signature changed: {of.function_type} → {tf.function_type}"
                )

    def test_stage_is_idempotent(self) -> None:
        """Running the stage twice should not break anything."""
        result1 = self.stage.run(
            self.sample_module.clone(), self.target, self.capabilities
        )
        if result1.module is not None and result1.passed:
            result2 = self.stage.run(
                result1.module.clone(), self.target, self.capabilities
            )
            assert result2.passed, f"Stage not idempotent: {result2.contract_violations}"

    def test_graceful_without_plugin(self) -> None:
        """Stage must work without a target plugin (graceful degradation)."""
        stage_copy = type(self.stage)()
        result = stage_copy.run(
            self.sample_module.clone(), self.target, self.capabilities
        )
        assert isinstance(result, StageResult)
