"""User-space kernel-provider discovery + indexing tests.

Coverage:

- ``TestManifestSchema`` — round-trip + validation for the user-facing
  ``user_kernel_manifest_v1`` schema.
- ``TestIndexer`` — walk a tmp directory, validate every manifest,
  persist locked-file SHAs, register summary in ``registry.yaml``.
- ``TestLockedFilesAudit`` — tamper detection raises typed
  ``UserKernelHashDriftError``.
- ``TestProviderBid`` — UserKernelProvider matches contract by
  archetype + target + dtype + layout; exact-dim match → high
  confidence, compat → mid confidence, mismatch → 0.
- ``TestProviderSearch`` — fulfilling a matched bid reads the kernel
  source from disk; tampering refuses.
- ``TestMcpTools`` — list/describe/discover MCP tools surface the
  expected data.
- ``TestEndToEndAuction`` — auction with both CReferenceProvider +
  UserKernelProvider; the user kernel wins on confidence when its
  manifest matches the contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


_MATMUL_C_SRC = """\
#include <string.h>
void user_matmul_f32(const float* A, const float* B, float* Y,
                     int M, int N, int K) {
    memset(Y, 0, (size_t)M * (size_t)N * sizeof(float));
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                Y[i * N + j] += a * B[k * N + j];
        }
}
"""


def _write_user_kernel(
    *,
    root: Path,
    op_name: str = "linalg.matmul",
    target: str = "host_cpu",
    dims_lhs: list[int] | None = None,
    dims_rhs: list[int] | None = None,
    dims_out: list[int] | None = None,
    perf_priors: dict | None = None,
) -> Path:
    """Build a ``kernel_manifest.yaml`` + ``kernel.c`` pair under root."""
    root.mkdir(parents=True, exist_ok=True)
    src_path = root / "kernel.c"
    src_path.write_text(_MATMUL_C_SRC, encoding="utf-8")

    manifest = {
        "schema_version": "user_kernel_manifest_v1",
        "op_name": op_name,
        "archetype": "compute_tiled",
        "target_name": target,
        "language": "c",
        "kernel_source": "kernel.c",
        "entry_symbol": "user_matmul_f32",
        "inputs": [
            {"name": "lhs", "dtype": "f32", "layout": "row_major",
             "dims": dims_lhs if dims_lhs is not None else [16, 16]},
            {"name": "rhs", "dtype": "f32", "layout": "row_major",
             "dims": dims_rhs if dims_rhs is not None else [16, 32]},
        ],
        "outputs": [
            {"name": "out", "dtype": "f32", "layout": "row_major",
             "dims": dims_out if dims_out is not None else [16, 32]},
        ],
        "numerics": {"accumulator_dtype": "f32",
                     "expected_numerics": "tolerance_eps"},
        "dispatch_model": "sync",
        "perf_priors": perf_priors or {"estimated_us": 5.0, "confidence": 0.95},
    }
    manifest_path = root / "kernel_manifest.yaml"
    try:
        import yaml
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
    except ImportError:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )
    return manifest_path


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestManifestSchema:
    def test_round_trip(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
        )

        body = {
            "schema_version": "user_kernel_manifest_v1",
            "op_name": "linalg.matmul",
            "archetype": "compute_tiled",
            "target_name": "host_cpu",
            "language": "c",
            "kernel_source": "kernel.c",
            "entry_symbol": "x",
            "inputs": [{"name": "lhs", "dtype": "f32", "layout": "row_major"}],
            "outputs": [{"name": "out", "dtype": "f32", "layout": "row_major"}],
            "numerics": {"accumulator_dtype": "f32"},
        }
        m = UserKernelManifest.from_dict(body)
        assert m.op_name == "linalg.matmul"
        assert m.archetype == "compute_tiled"
        # Round-trip preserves every declared field.
        round_tripped = UserKernelManifest.from_dict(m.to_dict())
        assert round_tripped == m

    def test_missing_required_raises(self) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
            UserKernelManifestError,
        )

        with pytest.raises(UserKernelManifestError, match="missing required fields"):
            UserKernelManifest.from_dict(
                {"schema_version": "user_kernel_manifest_v1", "op_name": "x"}
            )

    def test_unknown_schema_raises(self) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
            UserKernelManifestError,
        )

        with pytest.raises(UserKernelManifestError, match="unknown schema_version"):
            UserKernelManifest.from_dict({"schema_version": "v9999"})


# --------------------------------------------------------------------------- #
# Indexer
# --------------------------------------------------------------------------- #


class TestIndexer:
    def test_indexes_one_manifest(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import index_one_manifest

        kernel_dir = tmp_path / "src" / "matmul_f32"
        manifest_path = _write_user_kernel(root=kernel_dir)
        index_root = tmp_path / "index"

        entry = index_one_manifest(
            manifest_path=manifest_path, index_root=index_root,
        )
        assert entry.manifest.op_name == "linalg.matmul"
        assert entry.manifest.target_name == "host_cpu"
        # Locked-files map carries SHAs for both manifest + source.
        assert "kernel_manifest.yaml" in entry.locked_files
        assert "kernel.c" in entry.locked_files
        # Index file written to disk.
        idx_file = index_root / entry.index_id / "manifest.yaml"
        assert idx_file.exists()

    def test_reindex_writes_registry(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import reindex

        # Two kernels in the search path.
        _write_user_kernel(root=tmp_path / "k1")
        _write_user_kernel(
            root=tmp_path / "k2", op_name="aten_pointwise_mul",
        )

        result = reindex(
            search_path=tmp_path, index_root=tmp_path / "index",
        )
        assert result["indexed_count"] == 2
        assert len(result["manifests_written"]) == 2
        assert result["errors"] == []
        assert Path(result["registry_path"]).exists()


# --------------------------------------------------------------------------- #
# Locked-files audit
# --------------------------------------------------------------------------- #


class TestLockedFilesAudit:
    def test_clean_audit_returns(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            audit_locked_files,
            index_one_manifest,
        )

        manifest_path = _write_user_kernel(root=tmp_path / "k")
        entry = index_one_manifest(
            manifest_path=manifest_path, index_root=tmp_path / "index",
        )
        # Should not raise.
        audit_locked_files(entry)

    def test_tampered_source_raises_typed(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelHashDriftError,
            audit_locked_files,
            index_one_manifest,
        )

        manifest_path = _write_user_kernel(root=tmp_path / "k")
        entry = index_one_manifest(
            manifest_path=manifest_path, index_root=tmp_path / "index",
        )

        # Tamper the kernel source.
        (tmp_path / "k" / "kernel.c").write_text(
            "/* edited post-index */\n", encoding="utf-8",
        )
        with pytest.raises(UserKernelHashDriftError, match="locked-files audit"):
            audit_locked_files(entry)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelHashDriftError,
            audit_locked_files,
            index_one_manifest,
        )

        manifest_path = _write_user_kernel(root=tmp_path / "k")
        entry = index_one_manifest(
            manifest_path=manifest_path, index_root=tmp_path / "index",
        )
        (tmp_path / "k" / "kernel.c").unlink()
        with pytest.raises(UserKernelHashDriftError):
            audit_locked_files(entry)


# --------------------------------------------------------------------------- #
# Provider bid + search
# --------------------------------------------------------------------------- #


def _build_v3_matmul(*, M=16, K=16, N=32):
    from compgen.kernels.contract_v3 import (
        ConcurrencyUnit, DispatchModel, DispatchSpec, EventDecl,
        ExecutionEnvelope, FusionPolicy, Granularity, HardwareEnvelope,
        IOContract, KernelArchetype, KernelContractV3, LayoutKind, MemorySpec,
        NumericsSpec, ObservabilitySpec, OrchestrationSpec, PaddingPolicy,
        PerformancePriority, ShapeClass, StaticAttr, SyncSpec, TensorIO,
    )

    hw = HardwareEnvelope(
        target_name="host_cpu", vector_lanes=8, scratchpad_bytes=65536,
        register_bytes=16, native_dtypes=("f32",),
    )
    exe = ExecutionEnvelope(
        hardware=hw, memory_budget_bytes=49152,
        concurrency_unit=ConcurrencyUnit.WARP, padding=PaddingPolicy.NONE,
        priority=PerformancePriority.LATENCY,
    )
    A = TensorIO(name="lhs", shape=ShapeClass(dims=(M, K)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    B = TensorIO(name="rhs", shape=ShapeClass(dims=(K, N)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    Y = TensorIO(name="out", shape=ShapeClass(dims=(M, N)),
                 dtype_class=("f32",), layout=LayoutKind.ROW_MAJOR,
                 alignment_bytes=64)
    return KernelContractV3(
        op_name="linalg.matmul", archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(A, B), outputs=(Y,),
            attributes=(StaticAttr(name="t_b", value=False),),
            numerics=NumericsSpec(
                accumulator_dtype="f32", fast_math=False,
                max_relative_error=0.0, deterministic=True,
            ),
        ),
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=exe,
            sync=SyncSpec(event_decls=(EventDecl(name="d"),)),
            memory=MemorySpec(),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.SYNC),
            observability=ObservabilitySpec(emit_completion_event=True),
        ),
    )


class TestProviderBid:
    def test_no_index_returns_zero_confidence(self, tmp_path: Path) -> None:
        from compgen.kernels.providers.user_path import UserKernelProvider

        # Empty index dir.
        p = UserKernelProvider(index_root=tmp_path / "empty_index")
        bid = p.bid(_build_v3_matmul())
        assert bid.confidence == 0.0
        assert bid.rationale == "no_indexed_kernels"

    def test_exact_dim_match_high_confidence(self, tmp_path: Path) -> None:
        from compgen.kernels.providers.user_path import UserKernelProvider
        from compgen.kernels.user_kernel_index import reindex

        _write_user_kernel(root=tmp_path / "k", dims_lhs=[16, 16],
                           dims_rhs=[16, 32], dims_out=[16, 32])
        reindex(search_path=tmp_path, index_root=tmp_path / "index")

        p = UserKernelProvider(index_root=tmp_path / "index")
        bid = p.bid(_build_v3_matmul(M=16, K=16, N=32))
        assert bid.rationale.startswith("exact_match:")
        assert bid.confidence >= 0.9
        assert bid.cache_hit is True

    def test_compat_match_mid_confidence(self, tmp_path: Path) -> None:
        from compgen.kernels.providers.user_path import UserKernelProvider
        from compgen.kernels.user_kernel_index import reindex

        # Manifest declares (32,32)/(32,64) but contract is (16,16)/(16,32).
        _write_user_kernel(
            root=tmp_path / "k", dims_lhs=[32, 32], dims_rhs=[32, 64],
            dims_out=[32, 64],
            perf_priors={"estimated_us": 10.0, "confidence": 0.6},
        )
        reindex(search_path=tmp_path, index_root=tmp_path / "index")

        p = UserKernelProvider(index_root=tmp_path / "index")
        bid = p.bid(_build_v3_matmul(M=16, K=16, N=32))
        assert bid.rationale.startswith("compat_match:")
        assert 0.0 < bid.confidence < 0.9

    def test_target_mismatch_zero_confidence(self, tmp_path: Path) -> None:
        from compgen.kernels.providers.user_path import UserKernelProvider
        from compgen.kernels.user_kernel_index import reindex

        # Manifest is for cuda_sm75; contract is host_cpu.
        _write_user_kernel(root=tmp_path / "k", target="cuda_sm75")
        reindex(search_path=tmp_path, index_root=tmp_path / "index")

        p = UserKernelProvider(index_root=tmp_path / "index")
        bid = p.bid(_build_v3_matmul())
        assert bid.confidence == 0.0


class TestProviderSearch:
    def test_search_after_bid_returns_kernel(self, tmp_path: Path) -> None:
        from compgen.kernels.provider import KernelContract, SearchBudget
        from compgen.kernels.providers.user_path import UserKernelProvider
        from compgen.kernels.user_kernel_index import reindex

        _write_user_kernel(root=tmp_path / "k")
        reindex(search_path=tmp_path, index_root=tmp_path / "index")

        p = UserKernelProvider(index_root=tmp_path / "index")
        # Bid first to record the match.
        p.bid(_build_v3_matmul())
        result = p.search(
            KernelContract(target_name="host_cpu", op_family="linalg.matmul"),
            SearchBudget(),
        )
        assert result.found is True
        assert result.language == "c"
        assert "user_matmul_f32" in result.kernel_code
        assert result.metadata["entry_symbol"] == "user_matmul_f32"

    def test_search_without_bid_returns_not_found(self, tmp_path: Path) -> None:
        from compgen.kernels.provider import KernelContract, SearchBudget
        from compgen.kernels.providers.user_path import UserKernelProvider

        p = UserKernelProvider(index_root=tmp_path / "empty")
        result = p.search(
            KernelContract(target_name="host_cpu"), SearchBudget(),
        )
        assert result.found is False
        assert "no bid match" in result.metadata.get("reason", "")

    def test_tampered_source_refuses(self, tmp_path: Path) -> None:
        from compgen.kernels.provider import KernelContract, SearchBudget
        from compgen.kernels.providers.user_path import UserKernelProvider
        from compgen.kernels.user_kernel_index import (
            UserKernelHashDriftError,
            reindex,
        )

        _write_user_kernel(root=tmp_path / "k")
        reindex(search_path=tmp_path, index_root=tmp_path / "index")

        p = UserKernelProvider(index_root=tmp_path / "index")
        p.bid(_build_v3_matmul())

        # Tamper after bid.
        (tmp_path / "k" / "kernel.c").write_text(
            "/* tampered */\n", encoding="utf-8",
        )
        with pytest.raises(UserKernelHashDriftError):
            p.search(KernelContract(target_name="host_cpu"), SearchBudget())


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #


class TestMcpTools:
    def test_discover_walks_path(self, tmp_path: Path, monkeypatch) -> None:
        from compgen.mcp.tools.kernel_providers import (
            compgen_discover_user_kernels,
        )

        monkeypatch.chdir(tmp_path)
        _write_user_kernel(root=tmp_path / "src" / "k1")
        result = compgen_discover_user_kernels(path=str(tmp_path / "src"))
        assert result["ok"] is True
        assert result["indexed_count"] == 1
        assert (tmp_path / ".compgen" / "user_kernel_index").exists()

    def test_discover_missing_path(self, tmp_path: Path) -> None:
        from compgen.mcp.tools.kernel_providers import (
            compgen_discover_user_kernels,
        )

        result = compgen_discover_user_kernels(
            path=str(tmp_path / "nonexistent"),
        )
        assert result["ok"] is False
        assert "does not exist" in result["error"]

    def test_list_includes_user_path_after_discovery(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from compgen.mcp.tools.kernel_providers import (
            compgen_discover_user_kernels,
            compgen_list_kernel_providers,
        )

        monkeypatch.chdir(tmp_path)
        _write_user_kernel(root=tmp_path / "src" / "k1")
        compgen_discover_user_kernels(path=str(tmp_path / "src"))

        result = compgen_list_kernel_providers()
        assert result["ok"] is True
        ids = {r["provider_id"] for r in result["providers"]}
        # CReferenceProvider is always present; UserKernelProvider when
        # the index has at least one entry.
        assert "c_reference" in ids
        assert "user_path" in ids

    def test_describe_user_path_surfaces_indexed_kernels(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from compgen.mcp.tools.kernel_providers import (
            compgen_describe_kernel_provider,
            compgen_discover_user_kernels,
        )

        monkeypatch.chdir(tmp_path)
        _write_user_kernel(root=tmp_path / "src" / "k1")
        compgen_discover_user_kernels(path=str(tmp_path / "src"))

        result = compgen_describe_kernel_provider(provider_id="user_path")
        assert result["ok"] is True
        assert "indexed_kernels" in result
        assert len(result["indexed_kernels"]) == 1
        ik = result["indexed_kernels"][0]
        assert ik["op_name"] == "linalg.matmul"
        assert ik["target_name"] == "host_cpu"
        assert ik["entry_symbol"] == "user_matmul_f32"


# --------------------------------------------------------------------------- #
# Auction with user kernel
# --------------------------------------------------------------------------- #


class TestEndToEndAuction:
    def test_user_kernel_wins_against_c_reference(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """End-to-end: pipeline runs to kernel-codegen-request,
        user kernel is indexed, auction runs and the user kernel
        wins on confidence."""
        # Bootstrap the run directory. CWD must remain repo root so
        # the model-config's relative module path resolves.
        bootstrap = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
                "--out", str(tmp_path / "run"),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
                "--auction-mode", "disabled",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert bootstrap.returncode == 0, bootstrap.stderr

        # Index a user kernel that exactly matches merlin_mlp_wide's
        # matmul_0 contract: lhs=(16,16), rhs=(16,32), out=(16,32),
        # perf_priors confidence=0.99.
        _write_user_kernel(
            root=tmp_path / "user_kernels" / "matmul",
            perf_priors={"estimated_us": 1.5, "confidence": 0.99},
        )

        # Now chdir to tmp so default_registry()/default_index_root()
        # finds the local index.
        monkeypatch.chdir(tmp_path)

        from compgen.kernels.user_kernel_index import (
            default_index_root,
            reindex,
        )

        reindex(
            search_path=tmp_path / "user_kernels",
            index_root=default_index_root(),
        )

        # Run the auction in-process (consumes the index from CWD's
        # .compgen/user_kernel_index/).
        from compgen.graph_compilation.kernel_auction import run_kernel_auction

        result = run_kernel_auction(
            run_dir=tmp_path / "run", mode="multi-bidder", bid_cutoff=3,
        )

        # The user kernel should have bid; CReferenceProvider also bid.
        bid_providers = {b.provider_name for b in result.bids}
        assert "user_path" in bid_providers
        assert "c_reference" in bid_providers
        # Both fulfilled + verified.
        assert result.overall == "pass"
        # User kernel should win because its perf_estimate (1.5us) +
        # high confidence beats c_reference (~2us, 0.85 confidence).
        assert result.winner_provider == "user_path"
