"""M-55 — ProviderRegistry wired into the Phase C kernel-codegen path.

Three test classes, mirroring the M-42 layout:

- ``TestApplicableMethod`` — unit tests for
  :meth:`compgen.kernels.registry.ProviderRegistry.applicable` over a
  KernelContractV3 fixture. Wildcards, target match, archetype match,
  ordering by priority.
- ``TestRegistryResolutionEmit`` — the M-42 pipeline stage now also
  writes ``04_kernel_codegen/registry_resolution.json``. The file
  exists; schema_version matches; ``providers_considered`` is sorted
  deterministically.
- ``TestEndToEndUnchanged`` — today's behaviour preserved: when no
  applicable providers are registered, ``fallback_used: true`` and the
  M-43 commit path stays the canonical one (as proven by the existence
  of ``requests/<task_id>.request.json``).

No new flag, no new boundary. M-55 is a strict generalisation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_v3_contract():
    """Build a KernelContractV3 fixture mirroring merlin_mlp_wide
    matmul_0 (host_cpu, COMPUTE_TILED, NORMAL granularity, SYNC).
    """
    from compgen.kernels.contract_v3 import (
        AliasPair,
        ConcurrencyUnit,
        DispatchModel,
        DispatchSpec,
        EventDecl,
        ExecutionEnvelope,
        FusionPolicy,
        Granularity,
        HardwareEnvelope,
        IOContract,
        KernelArchetype,
        KernelContractV3,
        LayoutKind,
        MemorySpec,
        NumericsSpec,
        ObservabilitySpec,
        OrchestrationSpec,
        PaddingPolicy,
        PerformancePriority,
        ShapeClass,
        StaticAttr,
        SyncSpec,
        TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="host_cpu",
        vector_lanes=8,
        scratchpad_bytes=65_536,
        register_bytes=16,
        native_dtypes=("f32",),
    )
    exe = ExecutionEnvelope(
        hardware=hw,
        memory_budget_bytes=49_152,
        concurrency_unit=ConcurrencyUnit.WARP,
        padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(
        name="A",
        shape=ShapeClass(dims=(16, 16)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    B = TensorIO(
        name="B",
        shape=ShapeClass(dims=(16, 32)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    Y = TensorIO(
        name="Y",
        shape=ShapeClass(dims=(16, 32)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    io = IOContract(
        inputs=(A, B),
        outputs=(Y,),
        attributes=(StaticAttr(name="transpose_b", value=False),),
        numerics=NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=0.0,
            deterministic=True,
        ),
    )
    orch = OrchestrationSpec(
        execution=exe,
        sync=SyncSpec(event_decls=(EventDecl(name="matmul_done"),)),
        memory=MemorySpec(),
        fusion=FusionPolicy(is_boundary=True),
        dispatch=DispatchSpec(model=DispatchModel.SYNC),
        observability=ObservabilitySpec(emit_completion_event=True),
    )
    return KernelContractV3(
        op_name="matmul_0",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=io,
        granularity=Granularity.NORMAL,
        orchestration=orch,
    )


class _StubProvider:
    """Minimal KernelProvider-conformant stub for unit tests."""

    def __init__(
        self,
        *,
        name: str,
        applicable_targets: tuple[str, ...] = (),
        applicable_archetypes: tuple[str, ...] = (),
        priority: int = 0,
        source: str = "in_tree",
    ) -> None:
        self._name = name
        self.applicable_targets = applicable_targets
        self.applicable_archetypes = applicable_archetypes
        self.priority = priority
        self._compgen_source = source

    @property
    def name(self) -> str:
        return self._name

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ProviderResult

        return ProviderResult(found=False)

    def export_knowledge(self):  # noqa: D401
        return []


# --------------------------------------------------------------------------- #
# Unit — applicable() filter
# --------------------------------------------------------------------------- #


class TestApplicableMethod:
    def test_wildcard_provider_matches(self) -> None:
        from compgen.kernels.registry import ProviderRegistry

        reg = ProviderRegistry()
        reg.register(_StubProvider(name="wild"))
        rows = reg.applicable(_build_v3_contract())
        assert len(rows) == 1
        assert rows[0].applicable is True
        assert rows[0].match_reason == "wildcard"

    def test_target_filter_matches_exactly(self) -> None:
        from compgen.kernels.registry import ProviderRegistry

        reg = ProviderRegistry()
        reg.register(
            _StubProvider(
                name="cpu_only",
                applicable_targets=("host_cpu",),
            )
        )
        reg.register(
            _StubProvider(
                name="gpu_only",
                applicable_targets=("cuda_sm75", "cuda_sm89"),
            )
        )
        rows = reg.applicable(_build_v3_contract())
        by_name = {r.provider_name: r for r in rows}
        assert by_name["cpu_only"].applicable is True
        assert by_name["gpu_only"].applicable is False
        assert "target='host_cpu' not in" in by_name["gpu_only"].match_reason

    def test_archetype_filter(self) -> None:
        from compgen.kernels.registry import ProviderRegistry

        reg = ProviderRegistry()
        reg.register(
            _StubProvider(
                name="pointwise_only",
                applicable_archetypes=("pointwise",),
            )
        )
        rows = reg.applicable(_build_v3_contract())
        assert rows[0].applicable is False
        assert "archetype='compute_tiled'" in rows[0].match_reason

    def test_sort_priority_then_name(self) -> None:
        from compgen.kernels.registry import ProviderRegistry

        reg = ProviderRegistry()
        reg.register(_StubProvider(name="b_low", priority=1))
        reg.register(_StubProvider(name="a_high", priority=10))
        reg.register(_StubProvider(name="z_low", priority=1))
        rows = reg.applicable(_build_v3_contract())
        assert [r.provider_name for r in rows] == ["a_high", "b_low", "z_low"]


# --------------------------------------------------------------------------- #
# Unit — registry_resolution.json emit
# --------------------------------------------------------------------------- #


def _invoke_pipeline(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-codegen-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class TestRegistryResolutionEmit:
    def test_resolution_file_emitted_alongside_request(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(model="merlin_mlp_wide", out_dir=tmp_path / "run")
        assert result.returncode == 0, result.stderr

        resolution = tmp_path / "run" / "04_kernel_codegen" / "registry_resolution.json"
        assert resolution.exists(), "M-55 must emit registry_resolution.json"
        body = json.loads(resolution.read_text())

        assert body["schema_version"] == "registry_resolution_v1"
        assert body["request_kind"] == "kernel_codegen"
        assert body["region_id"]
        assert body["contract_hash"]
        assert "providers_considered" in body
        assert "applicable_provider_names" in body
        assert "fallback_used" in body

    def test_byte_stable_across_reruns(self, tmp_path: Path) -> None:
        runs = [tmp_path / "run1", tmp_path / "run2"]
        for r in runs:
            res = _invoke_pipeline(model="merlin_mlp_wide", out_dir=r)
            assert res.returncode == 0, res.stderr
        a = json.loads((runs[0] / "04_kernel_codegen" / "registry_resolution.json").read_text())
        b = json.loads((runs[1] / "04_kernel_codegen" / "registry_resolution.json").read_text())
        # generated_at_utc differs; everything else must match.
        for k in (
            "schema_version", "task_id", "region_id", "candidate_id",
            "contract_hash", "request_kind", "providers_considered",
            "applicable_provider_names", "fallback_used",
        ):
            assert a[k] == b[k], f"{k} drifted across reruns"


# --------------------------------------------------------------------------- #
# E2E — today's path unchanged when no applicable providers
# --------------------------------------------------------------------------- #


class TestEndToEndUnchanged:
    def test_request_still_emitted_and_fallback_recorded(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(model="merlin_mlp_wide", out_dir=tmp_path / "run")
        assert result.returncode == 0, result.stderr

        # Today's M-42 request still lands.
        requests_dir = tmp_path / "run" / "04_kernel_codegen" / "requests"
        request_files = list(requests_dir.glob("*.request.json"))
        assert len(request_files) == 1, "M-42 request must still be emitted"

        # Resolution file says fallback_used (no applicable providers
        # in a clean checkout — Phase D wires real providers in M-56+).
        body = json.loads(
            (tmp_path / "run" / "04_kernel_codegen" / "registry_resolution.json").read_text()
        )
        if not body["applicable_provider_names"]:
            assert body["fallback_used"] is True
            assert body["fallback_path"] == "claude_code_subagent_via_m43"
