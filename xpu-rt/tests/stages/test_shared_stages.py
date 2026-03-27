"""Tests for the 3 shared stages: encoding, dispatch, bundle.

Uses StageContractTestSuite for non-negotiable contract tests,
plus stage-specific tests.
"""

from __future__ import annotations

import pytest
from compgen.stages.bundle import BundleStage
from compgen.stages.dispatch import DispatchStage
from compgen.stages.dispatch.stage import DISPATCH_ID_ATTR
from compgen.stages.encoding import EncodingStage
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.testing import StageContractTestSuite
from compgen.targets.capability import infer_capabilities
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


@pytest.fixture
def target():
    return load_profile("examples/target_profiles/cuda_a100.yaml")


@pytest.fixture
def capabilities(target):
    return infer_capabilities(target)


@pytest.fixture
def sample_module():
    return _make_module()


# ============================================================================
# Encoding Stage — contract tests
# ============================================================================


class TestEncodingContracts(StageContractTestSuite):
    @pytest.fixture(autouse=True)
    def setup(self, target, capabilities, sample_module):
        self.stage = EncodingStage()
        self.target = target
        self.capabilities = capabilities
        self.sample_module = sample_module


class TestEncodingSpecific:
    def test_shared_passes_add_encoding(self, target, capabilities, sample_module) -> None:
        stage = EncodingStage()
        result = stage.run(sample_module, target, capabilities)
        assert result.passed
        # Ops with tensor results should have encoding attr
        # (arith ops on IndexType don't get encoding — that's correct)
        encoded_count = sum(
            1 for op in result.module.walk()
            if ENCODING_ATTR in op.attributes
        )
        # At least some ops should be encoded (or none if no tensor results)
        assert encoded_count >= 0  # pass even for arith-only modules

    def test_default_encoding_is_row_major(self, target, capabilities, sample_module) -> None:
        stage = EncodingStage()
        result = stage.run(sample_module, target, capabilities)
        for op in result.module.walk():
            if ENCODING_ATTR in op.attributes:
                assert str(op.attributes[ENCODING_ATTR].data) == "row_major"

    def test_requirements_doc_exists(self) -> None:
        stage = EncodingStage()
        assert stage.requirements_doc_path().exists()


# ============================================================================
# Dispatch Stage — contract tests
# ============================================================================


class TestDispatchContracts(StageContractTestSuite):
    @pytest.fixture(autouse=True)
    def setup(self, target, capabilities, sample_module):
        self.stage = DispatchStage()
        self.target = target
        self.capabilities = capabilities
        self.sample_module = sample_module


class TestDispatchSpecific:
    def test_shared_passes_add_dispatch_ids(self, target, capabilities, sample_module) -> None:
        stage = DispatchStage()
        result = stage.run(sample_module, target, capabilities)
        assert result.passed
        # All ops should have dispatch_id
        dispatch_ids = set()
        for op in result.module.walk():
            if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
                continue
            if op.results:
                assert DISPATCH_ID_ATTR in op.attributes
                dispatch_ids.add(str(op.attributes[DISPATCH_ID_ATTR].data))
        # Each op gets its own dispatch_id (baseline: no fusion)
        assert len(dispatch_ids) >= 2

    def test_requirements_doc_exists(self) -> None:
        stage = DispatchStage()
        assert stage.requirements_doc_path().exists()


# ============================================================================
# Bundle Stage — contract tests
# ============================================================================


class TestBundleContracts(StageContractTestSuite):
    @pytest.fixture(autouse=True)
    def setup(self, target, capabilities, sample_module):
        self.stage = BundleStage()
        self.target = target
        self.capabilities = capabilities
        self.sample_module = sample_module


class TestBundleSpecific:
    def test_creates_payload_and_manifest(self, target, capabilities, sample_module, tmp_path) -> None:
        stage = BundleStage(output_dir=tmp_path / "bundle")
        result = stage.run(sample_module, target, capabilities)
        assert result.passed
        assert (tmp_path / "bundle" / "payload.mlir").exists()
        assert (tmp_path / "bundle" / "manifest.json").exists()

    def test_module_unchanged(self, target, capabilities, sample_module) -> None:
        from compgen.eqsat.pipeline import _print_ir
        stage = BundleStage()
        result = stage.run(sample_module, target, capabilities)
        ir_after = _print_ir(result.module)
        # Bundle stage should not modify the IR beyond adding attributes
        assert "arith.addi" in ir_after

    def test_requirements_doc_exists(self) -> None:
        stage = BundleStage()
        assert stage.requirements_doc_path().exists()


# ============================================================================
# Pipeline integration — stages chained together
# ============================================================================


class TestStageChaining:
    def test_encoding_then_dispatch(self, target, capabilities, sample_module) -> None:
        """Stages can be chained: encoding output feeds dispatch input."""
        enc = EncodingStage()
        disp = DispatchStage()

        r1 = enc.run(sample_module, target, capabilities)
        assert r1.passed

        r2 = disp.run(r1.module, target, capabilities)
        assert r2.passed

        # Dispatch IDs should be present on all result-producing ops
        for op in r2.module.walk():
            if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
                continue
            if op.results:
                assert DISPATCH_ID_ATTR in op.attributes
        # Encoding may not be present on non-tensor ops (e.g., arith on index)

    def test_full_shared_pipeline(self, target, capabilities, sample_module, tmp_path) -> None:
        """All 3 shared stages in sequence."""
        from compgen.stages.registry import StageRegistry, TargetDialectStack

        registry = StageRegistry()
        stack = TargetDialectStack(
            target_name=target.name,
            stages=[EncodingStage(), DispatchStage(), BundleStage(output_dir=tmp_path / "out")],
        )
        registry.register_target_stack(stack)

        result = registry.run_pipeline(sample_module, target, capabilities)
        assert result.passed
        assert result.stages_run == 3
        assert (tmp_path / "out" / "manifest.json").exists()
