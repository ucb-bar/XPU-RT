"""explore_vendor_repo → VendorDialectDescriptor, with and without LLM."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.agent.vendor_integration.explore import explore_vendor_repo
from compgen.agent.vendor_integration.propose_adapter import propose_adapter_layout

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "fake_vendor"


class _LLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def chat(self, prompt, num_samples: int = 1):
        self.calls.append(prompt)
        return [self._responses.pop(0)]


def test_explore_without_llm_returns_conservative_descriptor() -> None:
    result = explore_vendor_repo(FIXTURE, target="toy-target", workloads=("tinyllama",))
    d = result.descriptor
    assert not result.llm_used
    assert d.name in ("fake", "fake_vendor")
    assert d.package_name.startswith("compgen_")
    assert d.target == "toy-target"
    assert d.verification.workload_diff_test  # workloads was set
    # Scanner authoritative fields
    assert set(d.compile_entry.cli_tools) >= {"fake-opt", "fake-translate"}
    # LLM-less classification flagged kernel_authoring (no linalg tool present)
    assert d.kernel_authoring_required
    assert d.lowering.mode == "kernel_authoring"


def test_explore_with_mock_llm_prefers_llm_classification() -> None:
    payload = {
        "input_ir": ["linalg"],
        "output_format": "cubin",
        "kernel_authoring_required": False,
        "lowering_mode": "direct_linalg",
        "op_families": ["matmul", "softmax"],
        "bundle_steps": ["fake-opt --canonicalize", "fake-translate -o out.cubin"],
        "runtime_entry": "fake::launch",
        "notes": "cli tools + tutorial show a linalg ingress",
    }
    llm = _LLM([f"Here is my answer:\n{json.dumps(payload)}"])
    result = explore_vendor_repo(FIXTURE, target="nvidia-h100", workloads=("tinyllama",), llm_client=llm)
    d = result.descriptor
    assert result.llm_used
    assert d.input_ir == ("linalg",)
    assert d.output_format == "cubin"
    assert d.lowering.mode == "direct_linalg"
    assert not d.kernel_authoring_required
    assert "matmul" in d.lowering.op_families
    assert d.bundle.steps[0] == "fake-opt --canonicalize"


def test_explore_ignores_unparseable_llm_response() -> None:
    llm = _LLM(["I am not JSON and never will be"])
    result = explore_vendor_repo(FIXTURE, target="nvidia-h100", workloads=(), llm_client=llm)
    assert not result.llm_used  # fell back to deterministic classification


def test_propose_adapter_deterministic() -> None:
    result = explore_vendor_repo(FIXTURE, target="toy-target")
    proposal = propose_adapter_layout(result.descriptor, workloads=("tinyllama",))
    assert [r.op_family for r in proposal.rules]
    # Fixture has no linalg tool, so default proposal uses llm strategy.
    assert any(r.strategy == "llm" for r in proposal.rules)
    assert "structural" in proposal.verification_hooks
    assert not proposal.llm_used


def test_propose_adapter_llm_overrides_strategies() -> None:
    result = explore_vendor_repo(FIXTURE, target="toy-target")
    payload = {
        "rules": [
            {"op_family": "matmul", "strategy": "template", "rationale": "hand-tuned"},
            {"op_family": "softmax", "strategy": "llm", "rationale": "fallback"},
        ],
        "risks": ["vendor lacks f16 support"],
        "verification_hooks": ["structural", "matmul_diff"],
    }
    llm = _LLM([json.dumps(payload)])
    proposal = propose_adapter_layout(result.descriptor, workloads=("tinyllama",), llm_client=llm)
    assert proposal.llm_used
    by_fam = {r.op_family: r.strategy for r in proposal.rules}
    assert by_fam == {"matmul": "template", "softmax": "llm"}
    assert proposal.risks == ["vendor lacks f16 support"]
