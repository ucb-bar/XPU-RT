"""Wave 1.4 — `compgen.compile_to_megakernel` agentic-compilation entry.

CPU-only tests. The flagless contract: a PyPI user (or their agent)
calls ``compile_to_megakernel(model, sample_inputs)`` with no
backend flags and gets a :class:`MegakernelBundle` whose
``backend_choice`` snapshots how the probe resolved every knob.

GPU dispatch lives in :func:`MegakernelBundle.dispatch` — exercised
on bwell via the conformance flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Diamond(nn.Module):
    """Module-level so pickle round-trips through compile_context.json."""

    def __init__(self, in_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.a = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class _DataDependent(nn.Module):
    """Tensor-valued control flow — ``torch.fx.symbolic_trace`` rejects
    this, so even the Wave 2.2 generic FX fallback can't lower it.
    Used as the typed-error canary for the public-API surface tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x + 1
        return x - 1


class TestPublicSurface:
    def test_top_level_imports(self) -> None:
        import compgen

        assert hasattr(compgen, "compile_to_megakernel")
        assert hasattr(compgen, "MegakernelBundle")
        assert callable(compgen.compile_to_megakernel)

    def test_compile_to_megakernel_returns_bundle(self, tmp_path) -> None:
        """The agentic-compilation contract: zero flags, get a
        MegakernelBundle back."""
        import compgen

        model = _Diamond()
        x = torch.randn(64, 64)
        bundle = compgen.compile_to_megakernel(
            model,
            (x,),
            output_dir=str(tmp_path),
        )
        assert isinstance(bundle, compgen.MegakernelBundle)
        assert bundle.bundle_dir.is_dir()
        assert (bundle.bundle_dir / "megakernel" / "source.cu").is_file()
        assert (bundle.bundle_dir / "megakernel" / "manifest.yaml").is_file()
        assert (bundle.bundle_dir / "compile_context.json").is_file()
        assert bundle.kernel_name
        assert bundle.elapsed_ms > 0

    def test_bundle_carries_backend_choice(self, tmp_path) -> None:
        """``backend_choice`` is the agent-audit surface — every
        decision the probe made lands here."""
        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        assert bundle.backend_choice is not None
        for key in (
            "target_arch",
            "target_origin",
            "use_cublasdx_for_linears",
            "cublasdx_precision",
            "use_cu13_nvrtc",
            "rationale",
            "library_paths",
        ):
            assert key in bundle.backend_choice, f"backend_choice missing audit field {key}"

    def test_compile_context_round_trips(self, tmp_path) -> None:
        """compile_context.json preserves the resolved choice so
        ``compgen_run_compiled_bundle`` re-compiles against the same
        backend selection on bwell."""
        import json

        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        ctx = json.loads((bundle.bundle_dir / "compile_context.json").read_text())
        assert ctx["backend_mode"] in {"auto", "auto+overrides"}
        assert ctx["backend_choice"] is not None
        # Match top-level fields between in-memory bundle + on-disk ctx.
        assert ctx["target_arch"] == bundle.backend_choice["target_arch"]
        assert ctx["use_cu13_nvrtc"] == bundle.backend_choice["use_cu13_nvrtc"]

    def test_compile_context_carries_cost_prediction(self, tmp_path) -> None:
        """Bridge #129 — cost_prediction lives in compile_context.json
        AND verification_report.json so ``compgen_run_compiled_bundle``
        + audit-via-MCP queries can read it without re-running the
        predictor. Pin the schema so future schema drift fails here.
        """
        import json

        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        ctx = json.loads((bundle.bundle_dir / "compile_context.json").read_text())
        # Cost prediction stamped in.
        assert "cost_prediction" in ctx, (
            "compile_context.json must carry cost_prediction so the "
            "dispatch path can audit without re-running the predictor"
        )
        cp = ctx["cost_prediction"]
        # Top-level fields from EtcCostPrediction.
        for k in ("etc_us", "eager_us", "speedup", "passes_gate", "components", "reason"):
            assert k in cp, f"cost_prediction missing {k!r}"
        # Components include the per-pool breakdown that bridge #129
        # added (so the audit query can show wave fan-out).
        for k in (
            "num_linear_tasks",
            "num_pointwise_tasks",
            "num_linear_waves",
            "num_pointwise_waves",
            "cooperative_sync_us",
            "eager_dtype",
        ):
            assert k in cp["components"], f"cost_prediction.components missing {k!r}"
        # In-memory and on-disk should agree.
        assert cp["etc_us"] == bundle.cost_prediction["etc_us"]
        assert cp["passes_gate"] == bundle.cost_prediction["passes_gate"]

    def test_dispatch_method_exists(self, tmp_path) -> None:
        """The bundle exposes ``.dispatch(*args)`` — wraps the MCP
        run tool. Actual GPU dispatch tested on bwell; here we just
        pin the surface."""
        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        assert callable(bundle.dispatch)


class TestBackendOverrides:
    """Per-flag overrides on top of the probe."""

    def test_overrides_reach_backend_choice(self, tmp_path) -> None:
        """When the user sets ``backend_overrides``, those values
        win over the probe — but un-specified keys keep the probe's
        choice. This is the "advanced caller can pin one knob"
        story."""
        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
            backend_overrides={"cublasdx_precision": "fp32"},
        )
        # Override applied:
        assert bundle.backend_choice["cublasdx_precision"] == "fp32"

    def test_overrides_dont_change_probe_for_other_keys(self, tmp_path) -> None:
        import compgen

        b1 = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path / "b1"),
        )
        b2 = compgen.compile_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path / "b2"),
            backend_overrides={"cublasdx_precision": "fp32"},
        )
        # Only the overridden field differs.
        assert b1.backend_choice["target_arch"] == b2.backend_choice["target_arch"]
        assert b1.backend_choice["use_cu13_nvrtc"] == b2.backend_choice["use_cu13_nvrtc"]


class TestClusterLaunchEndToEnd:
    """Wave 1.6 — cluster-launch wiring carries through from the
    probe into the bundle's launch_config + compile_context.

    The agent reads the cluster shape via ``backend_choice.cluster_dim``
    in the response; the bundle's manifest carries the resolved
    cluster shape into the runtime launcher via ``cuLaunchKernelEx``.
    """

    def test_cluster_dim_lands_in_backend_choice(self, tmp_path) -> None:
        """Default Blackwell-targeting probe → cluster_dim populated
        in backend_choice. Validates Wave 1.6's plumbing through
        compile_to_megakernel."""
        import compgen

        bundle = compgen.compile_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        # The probe-detected target's supports_clusters surfaces in
        # the dict. Could be True (real Blackwell host) or False
        # (CPU host, or non-Blackwell GPU). Either way the field
        # must be present.
        assert "supports_clusters" in bundle.backend_choice
        if bundle.backend_choice["supports_clusters"]:
            assert bundle.backend_choice["cluster_dim"] is not None
            assert len(bundle.backend_choice["cluster_dim"]) == 3


class TestErrorPaths:
    def test_unsupported_shape_raises_typed(self, tmp_path) -> None:
        """Models that even the generic FX fallback (Wave 2.2) can't
        trace raise ``UnsupportedShape``, not a generic Exception.

        Pre-Wave-2.2 this used a simple chained-linear model; Wave 2.2
        added :func:`lower_generic_fx` which lowers any FX-traceable
        model. To still pin the typed-error path we use a model with
        tensor-valued control flow that ``torch.fx.symbolic_trace``
        rejects.
        """
        import compgen
        import pytest
        from compgen.runtime.lowering import UnsupportedShape

        with pytest.raises(UnsupportedShape):
            compgen.compile_to_megakernel(
                _DataDependent(),
                (torch.randn(64, 64),),
                output_dir=str(tmp_path),
            )
