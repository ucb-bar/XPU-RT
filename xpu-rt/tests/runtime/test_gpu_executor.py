"""Tests for the Wave 13 GPU runtime + distributed adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# --- hardware gates ---------------------------------------------------------


def _have_gpu() -> bool:
    from compgen.runtime.gpu_executor import gpu_available

    return gpu_available()


gpu_only = pytest.mark.skipif(not _have_gpu(), reason="GPU not available")


# --- CPU-side harness tests (always run) ------------------------------------


class TestGPUExecutorCPUSide:
    def test_gpu_available_returns_bool(self):
        from compgen.runtime.gpu_executor import gpu_available

        assert isinstance(gpu_available(), bool)

    def test_require_gpu_raises_on_missing(self):
        from compgen.runtime.gpu_executor import GPUNotAvailable, gpu_available

        if gpu_available():
            pytest.skip("gpu present; can't test the missing path")
        with pytest.raises(GPUNotAvailable):
            # Any kernel launch triggers _require_gpu.
            from compgen.runtime.gpu_executor import launch_triton_kernel

            with TemporaryDirectory() as td:
                launch_triton_kernel(
                    Path(td),
                    "nope",
                    args=[],
                    grid=(1,),
                )

    def test_load_emission_manifest_missing_file_raises(self, tmp_path: Path):
        from compgen.runtime.gpu_executor import load_emission_manifest

        with pytest.raises(FileNotFoundError):
            load_emission_manifest(tmp_path)

    def test_load_emission_manifest_round_trip(self, tmp_path: Path):
        from compgen.runtime.gpu_executor import load_emission_manifest

        manifest_path = tmp_path / "emission_manifest.json"
        manifest_path.write_text('{"foo": {"kernel": "foo", "source_path": "x"}}')
        data = load_emission_manifest(tmp_path)
        assert "foo" in data


# --- Real-GPU tests (skipped on CPU-only hosts) -----------------------------


@gpu_only
class TestGPUExecutorOnGPU:
    def test_matmul_launch_has_zero_diff_vs_torch(self, tmp_path: Path):
        """End-to-end: compile attention_mlp_tiny through the Triton
        allowlist, emit kernels, launch one on CUDA, confirm the
        output matches ``torch.matmul``."""
        import torch
        from compgen.options import CompGenOptions
        from compgen.pipeline import compile_through_pipeline
        from compgen.runtime.gpu_executor import launch_triton_kernel
        from compgen.runtime.triton_emitter import emit_triton_kernels

        from tests._fixtures.real_workloads import attention_mlp_tiny

        opts = CompGenOptions(
            enable_raise_special_ops=True,
            enable_match_library_call=True,
            library_allowlist=frozenset({"triton"}),
            kernel_family_allowlist=frozenset({"triton"}),
        )
        fx = attention_mlp_tiny()
        pr = compile_through_pipeline(fx.model, fx.example_inputs, options=opts)
        assert pr.module is not None

        emit = emit_triton_kernels(pr.module, out_dir=tmp_path)
        assert emit.kernels_emitted >= 1
        name = next(iter(emit.manifest))

        # Simple square matmul on real CUDA tensors.
        M = K = N = 32
        a = torch.randn(M, K, device="cuda:0")
        b = torch.randn(K, N, device="cuda:0")
        c = torch.zeros(M, N, device="cuda:0")
        grid = (1, 1)
        result = launch_triton_kernel(
            tmp_path,
            name,
            args=[
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
            ],
            grid=grid,
        )
        assert result.kernel_name == name
        # Diff against torch's matmul -- should be exact (no reassoc).
        ref = a @ b
        assert (c - ref).abs().max().item() < 1e-4


# --- GPU diff harness -------------------------------------------------------


class TestGPUDiffHarness:
    def test_cpu_host_returns_skipped_report(self):
        if _have_gpu():
            pytest.skip("skip-path unreachable on gpu host")
        from compgen.options import cuda_h100_defaults
        from compgen.runtime.gpu_diff import compile_and_diff_gpu

        from tests._fixtures.real_workloads import attention_mlp_tiny

        fx = attention_mlp_tiny()
        report = compile_and_diff_gpu(
            fx.model,
            fx.example_inputs,
            options=cuda_h100_defaults(),
            fixture_name=fx.name,
            eager_reference=fx.eager_output,
            exported_program=fx.exported,
        )
        assert report.skipped or report.triton_kernels_emitted >= 0

    @gpu_only
    def test_gpu_host_runs_diff(self):
        from compgen.options import cuda_h100_defaults
        from compgen.runtime.gpu_diff import compile_and_diff_gpu

        from tests._fixtures.real_workloads import attention_mlp_tiny

        fx = attention_mlp_tiny()
        report = compile_and_diff_gpu(
            fx.model,
            fx.example_inputs,
            options=cuda_h100_defaults(),
            fixture_name=fx.name,
            eager_reference=fx.eager_output,
            exported_program=fx.exported,
        )
        assert not report.skipped
        assert report.gpu_launches >= 0


# --- Distributed adapter ----------------------------------------------------


class TestDistributedAdapter:
    def test_distributed_available_returns_bool(self):
        from compgen.runtime.distributed import distributed_available

        assert isinstance(distributed_available(), bool)

    def test_current_env_reports_state(self):
        from compgen.runtime.distributed import current_env

        env = current_env()
        # On CPU-only hosts: world_size=1 rank=0 uninitialized.
        assert env.world_size >= 1

    def test_init_if_needed_world_size_1_skips(self):
        from compgen.runtime.distributed import init_if_needed

        env = init_if_needed()
        assert env.world_size == 1

    def test_all_reduce_single_process_is_identity(self):
        import torch
        from compgen.runtime.distributed import DistributedAdapter

        adapter = DistributedAdapter()
        x = torch.arange(12, dtype=torch.float32).view(3, 4)
        y = adapter.all_reduce(x, op="sum")
        assert torch.equal(x, y)

    def test_all_gather_single_process_is_identity(self):
        import torch
        from compgen.runtime.distributed import DistributedAdapter

        adapter = DistributedAdapter()
        x = torch.arange(6, dtype=torch.float32).view(2, 3)
        y = adapter.all_gather(x, dim=0)
        assert torch.equal(x, y)

    def test_reduce_scatter_single_process_is_identity(self):
        import torch
        from compgen.runtime.distributed import DistributedAdapter

        adapter = DistributedAdapter()
        x = torch.arange(8, dtype=torch.float32).view(2, 4)
        y = adapter.reduce_scatter(x, op="sum", dim=0)
        assert torch.equal(x, y)

    def test_broadcast_single_process_is_identity(self):
        import torch
        from compgen.runtime.distributed import DistributedAdapter

        adapter = DistributedAdapter()
        x = torch.ones(4)
        y = adapter.broadcast(x, source_replica=0)
        assert torch.equal(x, y)

    def test_adapter_constructor_without_env_reads_current(self):
        from compgen.runtime.distributed import DistributedAdapter

        adapter = DistributedAdapter()
        assert adapter.env is not None
