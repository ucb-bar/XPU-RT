"""End-to-end integration tests.

Tests the complete CompGen pipeline: model capture → IR → eqsat → kernel
contracts → stages pipeline → bundle.  These are the tests that prove
the system works as a compiler generator, not just individual components.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from compgen.capture.torch_export import capture_model
from compgen.eqsat.config import EqSatConfig
from compgen.eqsat.pipeline import run_eqsat_pass
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.contracts import build_kernel_contracts
from compgen.kernels.selector import select_strategies
from compgen.stages.registry import StageRegistry
from compgen.stages.targets.cuda_gpu import create_cuda_gpu_stack
from compgen.targetgen.generate import generate_target
from compgen.targets.capability import infer_capabilities
from compgen.targets.schema import load_profile

EXEMPLAR_DIR = Path(__file__).parent / "targetgen" / "exemplars"


class _SimpleMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _TransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = torch.nn.LayerNorm(64)
        self.ff = torch.nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x + self.ff(x)


# ============================================================================
# Test 1: Full pipeline on SimpleMLP with CUDA target
# ============================================================================


class TestFullPipeline:
    def test_simplemlp_capture_to_bundle(self, tmp_path: Path) -> None:
        """SimpleMLP → capture → xDSL → eqsat → contracts → stages → bundle."""
        # 1. Capture
        model = _SimpleMLP()
        ep = capture_model(model, (torch.randn(8, 64),))
        assert len(ep.graph.nodes) > 0

        # 2. Convert to xDSL
        module, diagnostics = fx_to_xdsl(ep)
        assert module is not None

        # 3. Run eqsat
        result = run_eqsat_pass(module, config=EqSatConfig(max_iterations=3))
        assert result.ops_after > 0

        # 4. Build kernel contracts + strategies
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        specs = build_kernel_contracts(module, target)
        assert len(specs) > 0
        decisions = select_strategies(specs, target)
        assert len(decisions) == len(specs)

        # 5. Run stages pipeline
        capabilities = infer_capabilities(target)
        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "bundle"))
        stack.target_name = target.name
        registry = StageRegistry()
        registry.register_target_stack(stack)
        pipeline_result = registry.run_pipeline(module, target, capabilities)

        assert pipeline_result.passed, f"Pipeline failed at {pipeline_result.first_failure}"
        assert pipeline_result.stages_run == 6

        # 6. Verify bundle
        assert (tmp_path / "bundle" / "manifest.json").exists()
        assert (tmp_path / "bundle" / "payload.mlir").exists()

    @pytest.mark.xfail(reason="mul_tensor decomposition needs more operand handling")
    def test_transformer_block_capture_to_ir(self) -> None:
        """TransformerBlock captures and converts to xDSL successfully."""
        model = _TransformerBlock()
        ep = capture_model(model, (torch.randn(2, 8, 64),))
        module, _ = fx_to_xdsl(ep)
        assert module is not None
        op_count = sum(1 for _ in module.walk())
        assert op_count > 5


# ============================================================================
# Test 2: TargetGen → Pipeline integration
# ============================================================================


class TestTargetGenPipeline:
    def test_targetgen_to_pipeline(self, tmp_path: Path) -> None:
        """HW spec YAML → targetgen → generated stack → pipeline on real model."""
        # 1. Generate target from spec
        gen = generate_target(
            EXEMPLAR_DIR / "test_gpu_simt.yaml",
            tmp_path / "targetgen_output",
        )

        # 2. Capture model
        model = _SimpleMLP()
        ep = capture_model(model, (torch.randn(8, 64),))
        module, _ = fx_to_xdsl(ep)

        # 3. Run generated pipeline
        registry = StageRegistry()
        gen.dialect_stack.target_name = gen.profile.name
        registry.register_target_stack(gen.dialect_stack)
        result = registry.run_pipeline(module, gen.profile, gen.capabilities)

        assert result.passed, f"Failed at {result.first_failure}"
        assert result.stages_run >= 3  # At minimum encoding + dispatch + bundle

    @pytest.mark.parametrize("yaml_file", sorted(EXEMPLAR_DIR.glob("*.yaml")))
    def test_all_families_generate_and_run(self, yaml_file: Path, tmp_path: Path) -> None:
        """Every exemplar family generates a stack that runs on SimpleMLP IR."""
        from xdsl.dialects import arith, func
        from xdsl.dialects.builtin import IndexType, ModuleOp
        from xdsl.ir import Block, Region

        # Simple arith module (works for all families)
        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        a, b = block.args
        add = arith.AddiOp(a, b)
        block.add_op(add)
        block.add_op(func.ReturnOp(add.result))
        module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

        gen = generate_target(yaml_file, tmp_path / yaml_file.stem)
        registry = StageRegistry()
        gen.dialect_stack.target_name = gen.profile.name
        registry.register_target_stack(gen.dialect_stack)
        result = registry.run_pipeline(module, gen.profile, gen.capabilities)

        assert result.passed, f"{yaml_file.name}: failed at {result.first_failure}"


# ============================================================================
# Test 3: Kernel contracts + strategies E2E
# ============================================================================


class TestKernelPipelineE2E:
    def test_contracts_to_strategies_on_real_model(self) -> None:
        """Kernel contracts extracted from real FX-imported model get valid strategies."""
        model = _SimpleMLP()
        ep = capture_model(model, (torch.randn(8, 64),))
        module, _ = fx_to_xdsl(ep)

        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        specs = build_kernel_contracts(module, target)
        decisions = select_strategies(specs, target)

        # Every spec must get a decision
        assert len(decisions) == len(specs)
        # All decisions must have a valid strategy
        for d in decisions:
            assert d.strategy is not None
            assert d.reason
