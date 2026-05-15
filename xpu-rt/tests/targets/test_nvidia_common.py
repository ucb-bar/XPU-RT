"""Wave 1.14 — NVIDIA-vendor-common code migration tests.

Pin the post-migration import paths so future moves don't drift.
Each migrated symbol lives at its new home AND re-exports from
the original location for backward compatibility.
"""

from __future__ import annotations


class TestDiscoveryMigration:
    """Discovery helpers (cuBLASDx / libcudacxx / CUTLASS) moved
    from ``runtime/native/cuda.py`` to
    ``targets/gpu/nvidia/common/discovery.py`` (Wave 1.14)."""

    def test_new_public_location_works(self) -> None:
        from compgen.targets.gpu.nvidia.common.discovery import (
            cublasdx_available,
            cutlass_available,
            discover_cublasdx_include,
            discover_cutlass_include,
            discover_libcudacxx_include,
            libcudacxx_available,
        )

        # All callable — return None or a path string, never raise.
        assert callable(discover_cublasdx_include)
        assert callable(discover_libcudacxx_include)
        assert callable(discover_cutlass_include)

        for fn in (discover_cublasdx_include, discover_libcudacxx_include, discover_cutlass_include):
            result = fn()
            assert result is None or isinstance(result, str)

        # Bool wrappers consistent with the discoverers.
        assert cublasdx_available() == (discover_cublasdx_include() is not None)
        assert libcudacxx_available() == (discover_libcudacxx_include() is not None)
        assert cutlass_available() == (discover_cutlass_include() is not None)

    def test_old_location_still_works_via_reexport(self) -> None:
        from compgen.runtime.native.cuda import (
            cublasdx_available as old_cublasdx_avail,
        )
        from compgen.runtime.native.cuda import (
            discover_cublasdx_include as old_discover,
        )
        from compgen.targets.gpu.nvidia.common.discovery import (
            cublasdx_available as new_cublasdx_avail,
        )
        from compgen.targets.gpu.nvidia.common.discovery import (
            discover_cublasdx_include as new_discover,
        )

        # Single source of truth — both paths return the same callable.
        assert old_discover is new_discover
        assert old_cublasdx_avail is new_cublasdx_avail


class TestArchCostTableMigration:
    """Per-arch TFLOPS tables migrated to
    ``targets/gpu/nvidia/{blackwell,hopper,ampere}/cost.py`` (Wave 1.14c).
    Universal ``etc_predict`` queries the leaves via
    ``_lookup_arch_tflops`` with a local-table fallback.
    """

    def test_blackwell_leaf_constants(self) -> None:
        from compgen.targets.gpu.nvidia.blackwell import cost

        assert cost.PEAK_FP32_TFLOPS_PER_SM == 4.5
        # Bridge #095 — bf16+fp32-acc tensor-core peak.
        assert cost.PEAK_BF16_TC_TFLOPS_PER_SM == 50.0
        assert cost.SM_COUNT_DEFAULT["sm_100"] == 132
        assert cost.SM_COUNT_DEFAULT["sm_120"] == 188

    def test_hopper_leaf_constants(self) -> None:
        from compgen.targets.gpu.nvidia.hopper import cost

        assert cost.PEAK_FP32_TFLOPS_PER_SM == 4.0
        assert cost.PEAK_BF16_TC_TFLOPS_PER_SM == 40.0

    def test_ampere_leaf_constants(self) -> None:
        from compgen.targets.gpu.nvidia.ampere import cost

        assert cost.PEAK_FP32_TFLOPS_PER_SM == 3.0
        assert cost.PEAK_BF16_TC_TFLOPS_PER_SM == 16.0

    def test_lookup_helper_uses_leaf(self) -> None:
        """``_lookup_arch_tflops("100", tensor_core=True)`` should
        find Blackwell's leaf value, not the local-table fallback."""
        from compgen.kernels.cost.etc_predict import _lookup_arch_tflops

        # sm_100 → blackwell leaf → 50.0 bf16+fp32-acc
        assert _lookup_arch_tflops("100", tensor_core=True) == 50.0
        # sm_90 → hopper leaf → 40.0
        assert _lookup_arch_tflops("90", tensor_core=True) == 40.0
        # sm_80 → ampere leaf → 16.0
        assert _lookup_arch_tflops("80", tensor_core=True) == 16.0

    def test_lookup_helper_falls_back_for_unknown_arch(self) -> None:
        """An arch with no leaf returns the conservative default —
        50 TFLOPS for tensor-core, 4 TFLOPS for SIMT. Lets the
        predictor handle future arches before their leaves land."""
        from compgen.kernels.cost.etc_predict import _lookup_arch_tflops

        result_tc = _lookup_arch_tflops("999", tensor_core=True)
        result_simt = _lookup_arch_tflops("999", tensor_core=False)
        # Defaults from the fallback path.
        assert result_tc == 50.0
        assert result_simt == 4.0


class TestCu13NvrtcMigration:
    """cu13 NVRTC code moved from ``runtime/native/cuda.py`` to
    ``targets/gpu/nvidia/blackwell/cu13_nvrtc.py`` (Wave 1.14b).
    Lives in ``blackwell/`` because it's the JIT toolchain
    Blackwell tcgen05 needs — Hopper / Ampere don't need it."""

    def test_new_blackwell_location_works(self) -> None:
        from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (
            _compile_via_cu13_nvrtc,
            _load_cu13_nvrtc,
            _resolve_cu13_nvrtc_lib_path,
            cu13_nvrtc_available,
        )

        for fn in (
            _resolve_cu13_nvrtc_lib_path,
            cu13_nvrtc_available,
            _load_cu13_nvrtc,
            _compile_via_cu13_nvrtc,
        ):
            assert callable(fn)

        # Probes never raise, even on CPU host without cu13 NVRTC.
        path = _resolve_cu13_nvrtc_lib_path()
        assert path is None or isinstance(path, str)
        assert isinstance(cu13_nvrtc_available(), bool)
        assert cu13_nvrtc_available() == (path is not None)

    def test_old_location_still_works_via_reexport(self) -> None:
        from compgen.runtime.native.cuda import (
            _compile_via_cu13_nvrtc as old_compile,
        )
        from compgen.runtime.native.cuda import (
            _resolve_cu13_nvrtc_lib_path as old_resolve,
        )
        from compgen.runtime.native.cuda import (
            cu13_nvrtc_available as old_avail,
        )
        from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (
            _compile_via_cu13_nvrtc as new_compile,
        )
        from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (
            _resolve_cu13_nvrtc_lib_path as new_resolve,
        )
        from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (
            cu13_nvrtc_available as new_avail,
        )

        # Single source of truth — both paths return the same callable.
        assert old_resolve is new_resolve
        assert old_avail is new_avail
        assert old_compile is new_compile


class TestSmTagMigration:
    """``_arch_to_cublasdx_sm`` moved from
    ``runtime/lowering/fx_to_megakernel.py`` to
    ``targets/gpu/nvidia/common/sm_tag.py`` (Wave 1.14)."""

    def test_new_public_location_works(self) -> None:
        from compgen.targets.gpu.nvidia.common import arch_to_cublasdx_sm

        assert callable(arch_to_cublasdx_sm)
        assert arch_to_cublasdx_sm("sm_100") == 1000
        assert arch_to_cublasdx_sm("sm_90") == 900
        assert arch_to_cublasdx_sm("sm_120") == 1000  # falls back

    def test_old_private_location_still_works(self) -> None:
        """Re-export shim — existing callers shouldn't break."""
        from compgen.runtime.lowering.fx_to_megakernel import (
            _arch_to_cublasdx_sm,
        )

        assert _arch_to_cublasdx_sm("sm_100") == 1000

    def test_both_paths_return_same_function(self) -> None:
        """The re-export is identity — single source of truth."""
        from compgen.runtime.lowering.fx_to_megakernel import (
            _arch_to_cublasdx_sm as old_path,
        )
        from compgen.targets.gpu.nvidia.common.sm_tag import (
            arch_to_cublasdx_sm as new_path,
        )

        assert old_path is new_path
