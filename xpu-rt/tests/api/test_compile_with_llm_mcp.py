"""Tests for the W8.1 ``mcp_session`` arg on ``compile_with_llm``.

When the caller supplies an MCP session + a contracts list, the W7
``optimize_via_mcp`` loop runs after the standard pipeline and the
returned ``LLMCompileResult.mcp_optimized`` field carries the result.

Without those args, ``compile_with_llm`` behaves exactly like before
(``mcp_optimized is None``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.api_llm import LLMCompileResult, compile_with_llm
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope, HardwareEnvelope, IOContract, KernelArchetype,
    KernelContractV3, OrchestrationSpec, ShapeClass, TensorIO,
)
from compgen.kernels.store import KernelStore, set_shared_store
from compgen.mcp.session import SessionManager
from compgen.memory.kernel_db import KernelDB, set_shared_db


EXEMPLAR_TARGET = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


@pytest.fixture(autouse=True)
def isolated_kernel_store(tmp_path: Path):
    set_shared_store(KernelStore(root=tmp_path / "kernel_store"))
    yield
    set_shared_store(None)


@pytest.fixture
def isolated_db(tmp_path: Path):
    db = KernelDB(path=tmp_path / "kernel_db.sqlite")
    set_shared_db(db)
    yield db
    set_shared_db(None)


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _matmul_contract(target: str = "test-gpu-simt") -> KernelContractV3:
    env = HardwareEnvelope(
        target_name=target, vector_lanes=64,
        scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f16",), peak_bandwidth_gbps=672.0,
    )
    return KernelContractV3(
        op_name="matmul", archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="lhs", shape=ShapeClass(dims=(64, 64)),
                         dtype_class=("f16",)),
                TensorIO(name="rhs", shape=ShapeClass(dims=(64, 64)),
                         dtype_class=("f16",)),
            ),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(64, 64)),
                              dtype_class=("f16",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


def test_compile_with_llm_runs_mcp_optimizer_when_session_supplied(
    sm, isolated_db, tmp_path,
) -> None:
    """The mcp_session+mcp_contracts opt-in must populate
    ``mcp_optimized`` on the returned result. We use the mock LLM so
    the test runs without API credentials and skips the agentic loop."""
    contracts = [_matmul_contract()]
    res = compile_with_llm(
        model=_Identity(),
        target=EXEMPLAR_TARGET,
        llm="mock",
        sample_inputs=(torch.randn(1, 64),),
        budget=0,
        transcript_dir=tmp_path / "transcripts",
        mcp_session=(sm, "sess1"),
        mcp_contracts=contracts,
        mcp_perf_budget_us=100.0,
    )
    assert isinstance(res, LLMCompileResult)
    assert res.mcp_optimized is not None
    assert len(res.mcp_optimized.decisions) == 1
    # The OptimizedModel's target field is the device profile name.
    assert res.mcp_optimized.target  # non-empty


def test_compile_with_llm_no_mcp_session_leaves_mcp_optimized_none(
    sm, isolated_db, tmp_path,
) -> None:
    """Backwards-compat: omitting the new args preserves legacy behaviour."""
    res = compile_with_llm(
        model=_Identity(),
        target=EXEMPLAR_TARGET,
        llm="mock",
        sample_inputs=(torch.randn(1, 64),),
        budget=0,
        transcript_dir=tmp_path / "transcripts",
    )
    assert res.mcp_optimized is None
