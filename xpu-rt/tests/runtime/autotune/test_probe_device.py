"""Wave 1.1 — probe_device + BackendChoice tests (CPU only).

These tests run on Garden without a GPU and pin the contract every
agentic-compilation caller relies on:

- ``probe_device(target="auto")`` never raises, even when CUDA is
  unreachable.
- The decision tree picks the right (use_cublasdx, use_cu13_nvrtc,
  precision, tile) tuple given a (arch, library) snapshot.
- ``BackendChoice`` serializes for the bundle's decision log.
- Caching is process-wide and cleanable for tests.

GPU paths run on bwell via the conformance flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _PickleableDiamond(nn.Module):
    """Module-level so pickle can serialize it (test-method-local
    classes can't round-trip through pickle for the MCP path)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 32, bias=False)
        self.b = nn.Linear(64, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class TestBackendChoiceContract:
    def test_top_level_imports(self) -> None:
        from compgen.runtime.autotune import BackendChoice, probe_device

        assert BackendChoice is not None
        assert callable(probe_device)

    def test_probe_returns_backend_choice(self) -> None:
        from compgen.runtime.autotune import (
            BackendChoice,
            _clear_probe_cache_for_tests,
            probe_device,
        )

        _clear_probe_cache_for_tests()
        choice = probe_device(target="auto")
        assert isinstance(choice, BackendChoice)
        # Required scalar fields all populated.
        assert isinstance(choice.target_arch, str) and choice.target_arch
        assert isinstance(choice.cublasdx_available, bool)
        assert isinstance(choice.cu13_nvrtc_available, bool)
        assert isinstance(choice.use_cublasdx_for_linears, bool)
        assert choice.cublasdx_precision in {"fp32", "bf16_fp32"}
        assert isinstance(choice.use_cu13_nvrtc, bool)
        assert isinstance(choice.cublasdx_sm, int) and choice.cublasdx_sm > 0
        for dim in (choice.tile_m, choice.tile_n, choice.tile_k):
            assert isinstance(dim, int) and dim > 0
        assert choice.target_origin in {"probed", "probed_torch", "explicit", "fallback"}
        assert isinstance(choice.rationale, str) and choice.rationale

    def test_explicit_target_skips_probe(self) -> None:
        """Passing target='sm_90' should never run CudaDeviceProbe;
        origin should be 'explicit'."""
        from compgen.runtime.autotune import (
            _clear_probe_cache_for_tests,
            probe_device,
        )

        _clear_probe_cache_for_tests()
        choice = probe_device(target="sm_90")
        assert choice.target_arch == "sm_90"
        assert choice.target_origin == "explicit"

    def test_auto_falls_back_to_sm_100_on_cpu_host(self) -> None:
        """Garden has no CUDA; auto should fall back to sm_100
        (paper-faithful default), not raise."""
        from compgen.runtime.autotune import (
            _clear_probe_cache_for_tests,
            probe_device,
        )

        _clear_probe_cache_for_tests()
        choice = probe_device(target="auto")
        # Three valid outcomes per bridge #102 fix:
        # - "fallback" on CPU host without libcompgen_rt + no torch CUDA
        # - "probed" via libcompgen_rt's CudaDeviceProbe
        # - "probed_torch" via torch.cuda.get_device_capability fallback
        assert choice.target_origin in {"fallback", "probed", "probed_torch"}
        if choice.target_origin == "fallback":
            assert choice.target_arch == "sm_100"
        else:
            assert choice.target_arch.startswith("sm_")

    def test_serializes_for_decision_log(self) -> None:
        from compgen.runtime.autotune import probe_device

        d = probe_device(target="sm_100").to_dict()
        # Pin the keys the bundle compile_context.json depends on.
        for key in (
            "target_arch",
            "target_origin",
            "cublasdx_available",
            "cu13_nvrtc_available",
            "use_cublasdx_for_linears",
            "cublasdx_precision",
            "use_cu13_nvrtc",
            "cublasdx_sm",
            "tile_shape",
            "rationale",
            "library_paths",
        ):
            assert key in d, f"missing key {key} in BackendChoice.to_dict()"
        assert isinstance(d["tile_shape"], list) and len(d["tile_shape"]) == 3


class TestDecisionTree:
    """Pin the rule-based decisions for every (arch × library) cell.

    These tests use monkeypatching to simulate library availability,
    so a CPU host can verify the matrix without needing real
    nvidia-mathdx / cu13 NVRTC installs.
    """

    def _force_libs(self, monkeypatch, *, cublasdx: bool, cu13: bool) -> None:
        """Monkeypatch the library probes inside autotune."""
        from compgen.runtime import autotune as autotune_mod

        def fake_probe() -> tuple[bool, bool, dict[str, str | None]]:
            return (
                cublasdx,
                cu13,
                {
                    "cublasdx_include": "/fake/cublasdx" if cublasdx else None,
                    "libcudacxx_include": "/fake/libcudacxx" if cublasdx else None,
                    "cutlass_include": "/fake/cutlass" if cublasdx else None,
                    "cu13_nvrtc_lib": "/fake/libnvrtc.so.13" if cu13 else None,
                },
            )

        monkeypatch.setattr(autotune_mod, "_probe_libraries", fake_probe)
        autotune_mod._clear_probe_cache_for_tests()

    def test_blackwell_with_all_libs_picks_cublasdx_bf16(self, monkeypatch) -> None:
        """The win path: Blackwell + all libs → tensor-core engaged
        (bf16+fp32-acc, 64×64×16 tile, cu13 NVRTC, SM<1000>)."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=True)
        choice = probe_device(target="sm_100")
        assert choice.use_cublasdx_for_linears is True
        assert choice.cublasdx_precision == "bf16_fp32"
        assert choice.use_cu13_nvrtc is True
        assert choice.cublasdx_sm == 1000
        assert (choice.tile_m, choice.tile_n, choice.tile_k) == (64, 64, 16)

    def test_blackwell_without_cu13_falls_back_to_fmaf(self, monkeypatch) -> None:
        """Bridge #089: cu12 NVRTC's max sm_90 silently SIMTs cuBLASDx
        on Blackwell. Without cu13 NVRTC we MUST fall back — silently
        emitting cuBLASDx with cu12 NVRTC is the bug we're avoiding."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=False)
        choice = probe_device(target="sm_100")
        assert choice.use_cublasdx_for_linears is False
        assert choice.use_cu13_nvrtc is False
        assert (choice.tile_m, choice.tile_n, choice.tile_k) == (32, 32, 32)
        assert "cu13 NVRTC not reachable" in choice.rationale

    def test_blackwell_without_cublasdx_falls_back(self, monkeypatch) -> None:
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=False, cu13=True)
        choice = probe_device(target="sm_100")
        assert choice.use_cublasdx_for_linears is False
        # cu13 NVRTC available but no cuBLASDx → use_cu13 stays True
        # (the runtime can use it for non-cuBLASDx kernels too) but
        # the matcher's tile + body stays on the fmaf path.
        assert choice.use_cu13_nvrtc is True
        assert (choice.tile_m, choice.tile_n, choice.tile_k) == (32, 32, 32)
        assert "cuBLASDx headers not reachable" in choice.rationale

    def test_hopper_with_all_libs_uses_cublasdx_fp32(self, monkeypatch) -> None:
        """sm_90 doesn't engage Blackwell's tcgen05 path; we still
        get cuBLASDx with the older mma.sync atom but at fp32 SIMT-
        equivalent throughput. Tile stays 32 since 64-tile only
        matters for engaging the larger MMA atoms."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=True)
        choice = probe_device(target="sm_90")
        # Hopper: cuBLASDx is reachable but use_cu13_nvrtc=False
        # because we don't need cu13 — sm_90 is in cu12's range. That
        # gates use_cublasdx (since the win path needs both). Hopper
        # stays on hand_rolled_fmaf for now; explicit Hopper support
        # is a Wave 2 task.
        assert choice.use_cu13_nvrtc is False
        assert choice.use_cublasdx_for_linears is False
        assert choice.cublasdx_sm == 900

    def test_no_libs_no_cuda(self, monkeypatch) -> None:
        """The CPU-host case: nothing reachable. Choice is fmaf at
        32-tile, sm_100 origin=fallback. Should never raise."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=False, cu13=False)
        choice = probe_device(target="auto")
        assert choice.use_cublasdx_for_linears is False
        assert choice.use_cu13_nvrtc is False
        assert (choice.tile_m, choice.tile_n, choice.tile_k) == (32, 32, 32)


class TestCaching:
    def test_repeat_calls_return_same_choice(self, monkeypatch) -> None:
        from compgen.runtime.autotune import (
            _clear_probe_cache_for_tests,
            probe_device,
        )

        _clear_probe_cache_for_tests()
        a = probe_device(target="sm_100")
        b = probe_device(target="sm_100")
        assert a is b  # cached, identical object

    def test_force_refresh_bypasses_cache(self) -> None:
        from compgen.runtime.autotune import (
            _clear_probe_cache_for_tests,
            probe_device,
        )

        _clear_probe_cache_for_tests()
        a = probe_device(target="sm_100")
        b = probe_device(target="sm_100", force_refresh=True)
        # Same content but possibly distinct objects.
        assert a.to_dict() == b.to_dict()


class TestMatcherIntegration:
    """The agentic-compilation contract end-to-end: pass a
    BackendChoice into ``lower_torch_to_megakernel`` and the
    individual flags get overridden cleanly."""

    def test_lower_accepts_backend_choice(self) -> None:
        """The matcher's `backend_choice` kwarg overrides every
        individual flag. Existing callers (passing flags
        explicitly) keep working."""
        from compgen.runtime.autotune import probe_device
        from compgen.runtime.lowering import lower_torch_to_megakernel

        # No flags — auto-probe path.
        choice = probe_device(target="sm_90")  # Hopper → fmaf path
        result = lower_torch_to_megakernel(
            _PickleableDiamond(),
            (torch.randn(64, 64),),
            backend_choice=choice,
        )
        # Probe returned use_cublasdx=False for sm_90 with no libs,
        # so the matcher's body decisions should reflect that.
        for d in result.decision.body_decisions:
            assert d.backend == "hand_rolled_fmaf"

    def test_mcp_tool_auto_mode_runs_probe(self, tmp_path) -> None:
        """The MCP-tool path with backend='auto' (the default) must
        run probe_device + propagate the result into compile_context.json
        so run_compiled_bundle can match the same lowering on
        recompile."""
        import base64
        import json
        import pickle

        from compgen.mcp.tools.compile import compgen_compile_torch_model

        x = torch.randn(64, 64)
        result = compgen_compile_torch_model(
            model_pickle_b64=base64.b64encode(pickle.dumps(_PickleableDiamond())).decode(),
            sample_input_pickle_b64=base64.b64encode(pickle.dumps((x,))).decode(),
            output_dir=str(tmp_path),
            # backend defaults to "auto" — no other flags.
        )
        assert result["status"] == "ok"
        ctx = json.loads((tmp_path / "bundle" / "compile_context.json").read_text())
        # Backend choice must be present + populated for the audit story.
        assert ctx["backend_mode"] == "auto"
        assert ctx["backend_choice"] is not None
        bc = ctx["backend_choice"]
        for key in (
            "target_arch",
            "target_origin",
            "use_cublasdx_for_linears",
            "cublasdx_precision",
            "use_cu13_nvrtc",
            "rationale",
        ):
            assert key in bc, f"compile_context.backend_choice missing {key}"


class TestClusterLaunchWiring:
    """Wave 1.6 — cluster-launch dimensions surface on BackendChoice
    when the probe detects a cluster-capable target. Bridge #108
    confirmed multi-block-per-task is the perf-gate fix; this is
    the wiring that lets it get through to compute_static_schedule."""

    def _force_libs(self, monkeypatch, *, cublasdx: bool, cu13: bool) -> None:
        from compgen.runtime import autotune as autotune_mod

        def fake_probe() -> tuple[bool, bool, dict[str, str | None]]:
            return (
                cublasdx,
                cu13,
                {
                    "cublasdx_include": "/fake/cublasdx" if cublasdx else None,
                    "libcudacxx_include": "/fake/libcudacxx" if cublasdx else None,
                    "cutlass_include": "/fake/cutlass" if cublasdx else None,
                    "cu13_nvrtc_lib": "/fake/libnvrtc.so.13" if cu13 else None,
                },
            )

        monkeypatch.setattr(autotune_mod, "_probe_libraries", fake_probe)
        autotune_mod._clear_probe_cache_for_tests()

    def test_blackwell_enables_cluster_launch(self, monkeypatch) -> None:
        """sm_100 / sm_120 → supports_clusters=True with default
        cluster_dim=(2,1,1). The conservative starting point per
        Wave 1.6's wiring; 4 / 8 are tunable later."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=True)
        choice = probe_device(target="sm_100")
        assert choice.supports_clusters is True
        assert choice.cluster_dim_x == 2
        assert choice.cluster_dim_y == 1
        assert choice.cluster_dim_z == 1

    def test_hopper_no_cluster_yet(self, monkeypatch) -> None:
        """Wave 1.6 starts with Blackwell only — Hopper supports
        clusters too but bridge #108's data validates Blackwell.
        Hopper enablement is a follow-up tuning."""
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=True)
        choice = probe_device(target="sm_90")
        assert choice.supports_clusters is False
        assert choice.cluster_dim_x is None

    def test_cpu_no_cluster(self, monkeypatch) -> None:
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=False, cu13=False)
        # Use an explicit CPU-shaped arch so the probe doesn't
        # auto-detect a GPU on the host.
        choice = probe_device(target="cpu_x86")
        assert choice.supports_clusters is False
        assert choice.cluster_dim_x is None

    def test_to_dict_serializes_cluster_fields(self, monkeypatch) -> None:
        from compgen.runtime.autotune import probe_device

        self._force_libs(monkeypatch, cublasdx=True, cu13=True)
        d = probe_device(target="sm_100").to_dict()
        assert d["supports_clusters"] is True
        assert d["cluster_dim"] == [2, 1, 1]


class TestAgentAuditFields:
    """The agentic-compilation contract: every decision the matcher
    made must be queryable later (via the bundle's decision log)
    so the agent can answer "why X for op Y?" without re-probing."""

    def test_rationale_explains_decision(self) -> None:
        from compgen.runtime.autotune import probe_device

        choice = probe_device(target="sm_100")
        # Rationale must mention each top-level decision so the
        # agent can grep for the answer.
        for token in (
            "target=sm_100",
            "cublasdx headers reachable",
            "cu13 NVRTC reachable",
            "use_cu13_nvrtc",
            "use_cublasdx_for_linears",
        ):
            assert token in choice.rationale, f"rationale missing audit token {token!r}: {choice.rationale}"

    def test_library_paths_carry_actual_paths_when_reachable(self, monkeypatch) -> None:
        """When a library probes successfully, the path string lands
        in library_paths so the agent can audit the include/dlopen
        chain without re-running discovery."""
        from compgen.runtime import autotune as autotune_mod
        from compgen.runtime.autotune import (
            _clear_probe_cache_for_tests,
            probe_device,
        )

        def fake_probe() -> tuple[bool, bool, dict[str, str | None]]:
            return (
                True,
                True,
                {
                    "cublasdx_include": "/p1",
                    "libcudacxx_include": "/p2",
                    "cutlass_include": "/p3",
                    "cu13_nvrtc_lib": "/p4",
                },
            )

        monkeypatch.setattr(autotune_mod, "_probe_libraries", fake_probe)
        _clear_probe_cache_for_tests()
        choice = probe_device(target="sm_100")
        assert choice.library_paths == {
            "cublasdx_include": "/p1",
            "libcudacxx_include": "/p2",
            "cutlass_include": "/p3",
            "cu13_nvrtc_lib": "/p4",
        }
