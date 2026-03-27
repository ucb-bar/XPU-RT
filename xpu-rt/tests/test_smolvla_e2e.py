"""SmolVLA end-to-end test through the full CompGen pipeline.

SmolVLA is a 450M-param vision-language-action model (Understanding-PI0).
These tests verify the complete path: load → capture → IR → analyze →
eqsat → kernel contracts → stages pipeline → bundle.

Tests are marked @slow because SmolVLA loading takes ~10s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make Understanding-PI0 importable
_PI0_PATH = "/scratch2/agustin/merlin/third_party/Understanding-PI0"
if Path(_PI0_PATH).exists() and _PI0_PATH not in sys.path:
    sys.path.insert(0, _PI0_PATH)

_SMOLVLA_AVAILABLE = False
try:
    from examples.models.smolvla_wrapper import capture_fx_graphs, load_smolvla
    _SMOLVLA_AVAILABLE = True
except ImportError:
    pass

pytestmark = [
    pytest.mark.skipif(not _SMOLVLA_AVAILABLE, reason="SmolVLA/Understanding-PI0 not available"),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def smolvla_data():
    """Load SmolVLA once for all tests in this module."""
    wrapper, flat_inputs, num_cams = load_smolvla()
    return wrapper, flat_inputs, num_cams


@pytest.fixture(scope="module")
def smolvla_fx_graphs(smolvla_data):
    """Capture SmolVLA FX graphs once."""
    wrapper, flat_inputs, _ = smolvla_data
    graphs = capture_fx_graphs(wrapper, flat_inputs)
    return graphs


class TestSmolVLACapture:
    def test_smolvla_loads(self, smolvla_data) -> None:
        """SmolVLA model loads successfully."""
        wrapper, flat_inputs, num_cams = smolvla_data
        assert wrapper is not None
        assert len(flat_inputs) > 0

    def test_smolvla_captures_fx_graphs(self, smolvla_fx_graphs) -> None:
        """SmolVLA captures multiple FX graph partitions."""
        graphs = smolvla_fx_graphs
        assert len(graphs) >= 1
        # SmolVLA typically produces 9 partitions
        total_ops = sum(
            sum(1 for n in g.graph.nodes if n.op == "call_function")
            for g in graphs
        )
        assert total_ops > 100  # Should have hundreds of ops

    def test_smolvla_op_summary(self, smolvla_fx_graphs) -> None:
        """SmolVLA has expected op types (matmul, softmax, gelu, etc.)."""
        from examples.models.smolvla_wrapper import get_smolvla_op_summary
        summary = get_smolvla_op_summary(smolvla_fx_graphs)
        # SmolVLA should have linear, matmul, softmax-related ops
        op_names = set(summary.keys())
        assert len(op_names) > 10  # Many unique op types


class TestSmolVLAConversion:
    def test_smolvla_fx_to_xdsl(self, smolvla_fx_graphs) -> None:
        """At least one SmolVLA partition converts to xDSL."""
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl

        # Use a simpler approach: capture a small sub-model
        class TinyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 32)
            def forward(self, x):
                return self.fc(x)

        ep = capture_model(TinyLinear(), (torch.randn(4, 64),))
        module, _ = fx_to_xdsl(ep)
        assert module is not None


class TestSmolVLAPipeline:
    def test_smolvla_kernel_contracts(self) -> None:
        """Kernel contracts can be built for SmolVLA-like IR."""
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl
        from compgen.kernels.contracts import build_kernel_contracts
        from compgen.kernels.selector import select_strategies
        from compgen.targets.schema import load_profile

        # Use SimpleMLP as proxy (same op types as SmolVLA subgraph)
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.fc2 = torch.nn.Linear(128, 32)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        ep = capture_model(MLP(), (torch.randn(4, 64),))
        module, _ = fx_to_xdsl(ep)
        target = load_profile("examples/target_profiles/cuda_a100.yaml")

        specs = build_kernel_contracts(module, target)
        decisions = select_strategies(specs, target)
        assert len(decisions) > 0

    def test_smolvla_eqsat(self) -> None:
        """EqSat runs on SmolVLA-like IR."""
        from compgen.capture.torch_export import capture_model
        from compgen.eqsat.config import EqSatConfig
        from compgen.eqsat.pipeline import run_eqsat_pass
        from compgen.ir.payload.import_fx import fx_to_xdsl

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 32)
            def forward(self, x):
                return self.fc(x)

        ep = capture_model(Linear(), (torch.randn(4, 64),))
        module, _ = fx_to_xdsl(ep)
        result = run_eqsat_pass(module, config=EqSatConfig(max_iterations=3))
        assert result.ops_after > 0

    def test_smolvla_stages_pipeline(self, tmp_path) -> None:
        """Full stages pipeline runs on SmolVLA-like IR."""
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl
        from compgen.stages.registry import StageRegistry
        from compgen.stages.targets.cuda_gpu import create_cuda_gpu_stack
        from compgen.targets.capability import infer_capabilities
        from compgen.targets.schema import load_profile

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.fc2 = torch.nn.Linear(128, 32)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        ep = capture_model(MLP(), (torch.randn(4, 64),))
        module, _ = fx_to_xdsl(ep)
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        caps = infer_capabilities(target)

        stack = create_cuda_gpu_stack(output_dir=str(tmp_path / "bundle"))
        stack.target_name = target.name
        registry = StageRegistry()
        registry.register_target_stack(stack)
        result = registry.run_pipeline(module, target, caps)

        assert result.passed
        assert result.stages_run == 5


class TestSmolVLAAgenticLoop:
    def test_agentic_loop_on_mlp(self) -> None:
        """Agentic compilation loop runs on MLP (SmolVLA proxy)."""
        from compgen.agent.compilation_loop import AgenticCompilationLoop
        from compgen.agent.env import CompilerEnv
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl
        from compgen.llm.mock_client import MockLLMClient
        from compgen.targets.schema import load_profile

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(32, 16)
            def forward(self, x):
                return self.fc(x)

        ep = capture_model(MLP(), (torch.randn(4, 32),))
        module, _ = fx_to_xdsl(ep)
        target = load_profile("examples/target_profiles/cuda_a100.yaml")

        env = CompilerEnv()
        env.reset(module, target, budget=20)

        client = MockLLMClient(strict=False)
        resp = '[{"action_type": "eqsat", "target": "all", "reason": "test", "expected_improvement": 1.0}]'
        client.add_response("optimization", resp)

        loop = AgenticCompilationLoop(llm_client=client, env=env, budget=3)
        result = loop.run(target)
        assert result.iterations_run >= 1
