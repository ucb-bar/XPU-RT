"""Phase-4 ctypes-wrapper API tests (CPU-only).

Validates the Python surface of ``compgen.runtime.native.cuda`` —
imports, struct layouts, and the typed-error path on hosts without
the CUDA-built library.

GPU correctness tests for the .cu primitives live behind
``@pytest.mark.requires_gpu`` and run on the bwell box via the
conformance harness.
"""

from __future__ import annotations

import pytest


class TestModuleSurface:
    def test_top_level_imports(self) -> None:
        from compgen.runtime.native.cuda import (
            CudaCommGroup,
            CudaDeviceProbe,
            CudaDynamicQueue,
            CudaEventTensor,
            CudaMegakernelLauncher,
            CudaModule,
            CudaUnavailableError,
        )

        assert CudaCommGroup is not None
        assert CudaDeviceProbe is not None
        assert CudaDynamicQueue is not None
        assert CudaEventTensor is not None
        assert CudaMegakernelLauncher is not None
        assert CudaModule is not None
        assert issubclass(CudaUnavailableError, RuntimeError)

    def test_cuda_module_accepts_extra_include_paths(self) -> None:
        """Phase 10b contract — CudaModule.__init__ accepts
        ``extra_include_paths: tuple[str, ...]`` so cuBLASDx /
        CUTLASS / other header-only backends can have their
        include directory passed to NVRTC. Pin the kwarg name +
        default so a refactor can't silently drop it."""
        import inspect

        from compgen.runtime.native.cuda import CudaModule

        sig = inspect.signature(CudaModule.__init__)
        assert "extra_include_paths" in sig.parameters
        param = sig.parameters["extra_include_paths"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default == ()

    def test_cublasdx_discovery_helpers_exposed(self) -> None:
        """Phase 10b discovery — ``discover_cublasdx_include`` returns
        a path or None; ``cublasdx_available`` is a thin bool wrapper.
        On a CPU host without nvidia-mathdx these return ``None`` and
        ``False`` respectively — never raise."""
        from compgen.runtime.native.cuda import (
            cublasdx_available,
            discover_cublasdx_include,
        )

        path = discover_cublasdx_include()
        assert path is None or isinstance(path, str)
        assert isinstance(cublasdx_available(), bool)
        assert cublasdx_available() == (path is not None)

    def test_cutlass_discovery_helpers_exposed(self) -> None:
        """Round 2c — cuBLASDx pulls in ``cutlass/numeric_types.h``
        from the CUTLASS sublibrary that ``nvidia-mathdx`` vendors
        under ``external/cutlass/include``. Discovery must surface
        its location (or None) so the smoke + body emitter pass it
        as a separate ``-I`` to NVRTC."""
        from compgen.runtime.native.cuda import (
            cutlass_available,
            discover_cutlass_include,
        )

        path = discover_cutlass_include()
        assert path is None or isinstance(path, str)
        assert isinstance(cutlass_available(), bool)
        assert cutlass_available() == (path is not None)
        if path is not None:
            from pathlib import Path

            assert (Path(path) / "cutlass" / "numeric_types.h").is_file()

    def test_cu13_nvrtc_discovery_helpers_exposed(self) -> None:
        """Phase 10c+ — cu13 NVRTC discovery surfaces a path or None;
        ``cu13_nvrtc_available`` is a thin bool wrapper. On a CPU host
        without cu13 NVRTC these return ``None`` / ``False``,
        never raise."""
        from compgen.runtime.native.cuda import (
            _resolve_cu13_nvrtc_lib_path,
            cu13_nvrtc_available,
        )

        path = _resolve_cu13_nvrtc_lib_path()
        assert path is None or isinstance(path, str)
        assert isinstance(cu13_nvrtc_available(), bool)
        assert cu13_nvrtc_available() == (path is not None)

    def test_cu13_nvrtc_discovery_honors_env_var(self, tmp_path, monkeypatch) -> None:
        """Bridge #091 path #1 — `$COMPGEN_CU13_NVRTC_LIB_PATH` env
        override takes priority over package discovery so callers
        (and bwell's role-isolated agent) can point the wrapper at
        an arbitrary libnvrtc.so.13 without touching site-packages."""
        from compgen.runtime.native.cuda import _resolve_cu13_nvrtc_lib_path

        # Empty file is enough to pass the .is_file() check; we don't
        # actually load it here.
        fake_lib = tmp_path / "fake_libnvrtc.so.13"
        fake_lib.write_bytes(b"")
        monkeypatch.setenv("COMPGEN_CU13_NVRTC_LIB_PATH", str(fake_lib))
        assert _resolve_cu13_nvrtc_lib_path() == str(fake_lib)

        # Non-existent file → env override is ignored, falls through
        # to package discovery (which on CPU hosts returns None).
        monkeypatch.setenv("COMPGEN_CU13_NVRTC_LIB_PATH", str(tmp_path / "nope"))
        path = _resolve_cu13_nvrtc_lib_path()
        assert path is None or isinstance(path, str)

    def test_cu13_nvrtc_discovery_searches_unified_and_split_layouts(self) -> None:
        """Bridge #091 path #2 + #3 — discovery must search both
        the unified ``nvidia.cu13`` layout (torch≥2.6 meta-wheel)
        and the older split ``nvidia.cuda_nvrtc`` layout. Either
        present should resolve."""
        import importlib.util

        from compgen.runtime.native.cuda import _resolve_cu13_nvrtc_lib_path

        # If neither package is on this CPU host, the helper returns
        # None — which is the contract.
        unified = importlib.util.find_spec("nvidia.cu13")
        split = importlib.util.find_spec("nvidia.cuda_nvrtc")
        path = _resolve_cu13_nvrtc_lib_path()
        if unified is None and split is None:
            assert path is None
        else:
            # On hosts where one is installed, the helper either
            # finds the .so or returns None (e.g. wheel is partial).
            assert path is None or isinstance(path, str)

    def test_cuda_module_accepts_use_cu13_nvrtc(self) -> None:
        """Phase 10c+ contract — CudaModule.__init__ accepts
        ``use_cu13_nvrtc: bool`` so the megakernel can be compiled
        with the cu13 NVRTC (knows sm_100/sm_120) instead of the
        cu12 NVRTC bundled with cuda-python. Pin the kwarg name +
        default since silently dropping it regresses the
        tensor-core path."""
        import inspect

        from compgen.runtime.native.cuda import CudaModule

        sig = inspect.signature(CudaModule.__init__)
        assert "use_cu13_nvrtc" in sig.parameters
        param = sig.parameters["use_cu13_nvrtc"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is False

    def test_libcudacxx_discovery_helpers_exposed(self) -> None:
        """Round 2b/c — cuBLASDx's commondx layer pulls in
        ``cuda/std/type_traits`` etc. from libcudacxx; the discovery
        helpers must surface its location (or None) so the smoke
        tool + body emitter can pass it as a separate ``-I`` to
        NVRTC."""
        from compgen.runtime.native.cuda import (
            discover_libcudacxx_include,
            libcudacxx_available,
        )

        path = discover_libcudacxx_include()
        assert path is None or isinstance(path, str)
        assert isinstance(libcudacxx_available(), bool)
        assert libcudacxx_available() == (path is not None)
        # When found, the sentinel header must actually exist —
        # validates the search is checking content, not just paths.
        if path is not None:
            from pathlib import Path

            assert (Path(path) / "cuda" / "std" / "type_traits").is_file()

    def test_device_create_classmethod_exists(self) -> None:
        """The Phase-5 GPU smoke (and any consumer code) calls
        ``Device.create("cuda:0")``. Pin the classmethod's existence
        + signature shape here so a future refactor doesn't silently
        drop it again — the GPU test wouldn't catch this on a CPU
        host, but this CPU test will."""
        from compgen.runtime.native.device import Device

        assert hasattr(Device, "create")
        # Signature: ``Device.create(target: str) -> Device``.
        import inspect

        sig = inspect.signature(Device.create)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].name == "target"


class TestProbeStructLayout:
    def test_probe_struct_field_count_matches_header(self) -> None:
        """The C struct cg_rt_cuda_probe_t has 26 fields ordered as
        declared in compgen_rt.h. Drift between the header and the
        Python mirror would silently shred probe values across CUDA
        SDK upgrades, so we pin the count."""
        from compgen.runtime.native.cuda import _CudaProbeStruct

        # 1 char[128] + 25 int/longlong fields.
        assert len(_CudaProbeStruct._fields_) == 26

    def test_probe_struct_critical_fields_present(self) -> None:
        from compgen.runtime.native.cuda import _CudaProbeStruct

        names = {name for name, _ in _CudaProbeStruct._fields_}
        for required in (
            "device_name",
            "compute_capability_major",
            "compute_capability_minor",
            "sm_count",
            "cluster_launch",
            "supports_clusters",
            "supports_tma",
            "supports_fp8",
            "supports_fp4",
            "supports_ondevice_scheduler",
            "max_shared_memory_per_block_optin_bytes",
            "l2_cache_bytes",
            "driver_version",
            "runtime_version",
        ):
            assert required in names, f"probe struct missing {required}"


class TestLaunchConfigStruct:
    def test_launch_config_field_count(self) -> None:
        from compgen.runtime.native.cuda import _LaunchConfigStruct

        # kernel_handle + 3-axis grid/block/cluster + shared mem = 11.
        assert len(_LaunchConfigStruct._fields_) == 11

    def test_launch_config_critical_fields_present(self) -> None:
        from compgen.runtime.native.cuda import _LaunchConfigStruct

        names = {name for name, _ in _LaunchConfigStruct._fields_}
        assert {
            "kernel_handle",
            "grid_dim_x",
            "grid_dim_y",
            "grid_dim_z",
            "block_dim_x",
            "block_dim_y",
            "block_dim_z",
            "cluster_dim_x",
            "cluster_dim_y",
            "cluster_dim_z",
            "shared_mem_bytes",
        } <= names


def _has_cuda_runtime() -> bool:
    try:
        import compgen
    except Exception:
        return False
    return bool(compgen.has_cuda_runtime())


@pytest.mark.skipif(
    _has_cuda_runtime(),
    reason="fixture asserts absence of native HAL .so — only valid on CPU-only hosts",
)
class TestCudaUnavailableOnCpuHost:
    """On a host without the CUDA-built library, every constructor
    must raise ``CudaUnavailableError`` with a useful message — not
    a bare ImportError or AttributeError."""

    def test_probe_raises(self) -> None:
        # Force a fresh load attempt by clearing the cache; ensures
        # the error path runs even after probe_via_torch warmed
        # up another lib loader path elsewhere in the test session.
        import compgen.runtime.native.cuda as cuda_mod
        from compgen.runtime.native.cuda import (
            CudaDeviceProbe,
            CudaUnavailableError,
        )

        cuda_mod._CACHED_LIB = None

        # The library isn't present in the dev tree (`make build-cuda-rt`
        # never ran on Garden). Constructor must raise the typed
        # error with install instructions.
        with pytest.raises(CudaUnavailableError, match="libcompgen_rt"):
            CudaDeviceProbe()

    def test_event_tensor_raises(self) -> None:
        from compgen.runtime.native.cuda import (
            CudaEventTensor,
            CudaUnavailableError,
        )

        with pytest.raises(CudaUnavailableError):
            CudaEventTensor(num_cells=4)

    def test_dynamic_queue_raises(self) -> None:
        from compgen.runtime.native.cuda import (
            CudaDynamicQueue,
            CudaUnavailableError,
        )

        with pytest.raises(CudaUnavailableError):
            CudaDynamicQueue(capacity=16)

    def test_comm_group_raises_when_no_nccl(self) -> None:
        """When libcompgen_rt isn't loadable (CPU host) OR is built
        without NCCL, CudaCommGroup must raise the typed error with
        a clear rebuild hint — never bare ImportError."""
        from compgen.runtime.native.cuda import (
            CudaCommGroup,
            CudaUnavailableError,
        )

        with pytest.raises(CudaUnavailableError, match="NCCL|libcompgen_rt"):
            CudaCommGroup(device_indices=[0, 1])


@pytest.mark.skipif(
    _has_cuda_runtime(),
    reason="fixture asserts absence of native HAL .so — only valid on CPU-only hosts",
)
class TestNativeHalProbeFallback:
    """``probe_via_native_hal`` must surface the typed
    ``_NativeHalUnavailable`` when the CUDA build isn't there, so
    ``probe_cuda_device`` falls through to torch."""

    def test_native_hal_raises_native_hal_unavailable(self) -> None:
        import compgen.runtime.native.cuda as cuda_mod
        from compgen.runtime.probe import (
            _NativeHalUnavailable,
            probe_via_native_hal,
        )

        cuda_mod._CACHED_LIB = None

        with pytest.raises(_NativeHalUnavailable):
            probe_via_native_hal(0)

    def test_probe_cuda_device_falls_through_to_torch(self) -> None:
        """When native HAL is unavailable the umbrella probe lands in
        the torch path with ``probe_source="torch"`` (or "fallback"
        on a CPU-only host)."""
        from compgen.runtime.probe import probe_cuda_device

        result = probe_cuda_device(0)
        assert result["probe_source"] in ("torch", "fallback")
