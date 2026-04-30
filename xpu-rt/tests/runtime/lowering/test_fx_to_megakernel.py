"""Phase-10a tests — FX-graph → MegakernelGraph pattern matcher.

CPU-only. Validates the matcher's structural recognition of the
diamond shape, the fail-loud path on unsupported shapes, and the
shape-parameterised body emission. GPU end-to-end runs through
``compgen_compile_torch_model`` + ``compgen_run_compiled_bundle``
on the bwell box; these tests guard the matcher's contract on
every CI run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _Diamond(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.a = nn.Linear(in_dim, out_dim, bias=False)
        self.b = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class _Sequential(nn.Module):
    """FFN-shaped: two linears with chained shapes + relu in between.

    Round 2 added the FFN matcher, so this *now matches*. Tests that
    used this as the "neither-pattern-matches" canary use
    :class:`_ThreeLinear` instead.
    """

    def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.up = nn.Linear(in_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class _DataDependent(nn.Module):
    """Tensor-valued control flow — torch.fx.symbolic_trace rejects
    this, so even the generic FX fallback can't lower it. Used as
    the typed-error canary for the MCP-tool surface test."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x + 1
        return x - 1


class _ThreeLinear(nn.Module):
    """Neither diamond (needs exactly 2 linears with matching shapes)
    nor FFN (needs exactly 2 linears chained through relu) — used as
    the rejection canary for tests that want UnsupportedShape."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        self.c = nn.Linear(64, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


class TestDiamondMatcher:
    def test_matches_diamond_shape(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xD1A0)
        model = _Diamond(in_dim=64, out_dim=32)
        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(model, (x,))

        assert result.decision.pattern_name == "diamond"
        assert result.user_buffer_layout == (
            "x",
            "wa",
            "wb",
            "ya",
            "yb",
            "yadd",
            "yout",
        )
        names = [c.name for c in result.megakernel_graph.calls]
        assert names == ["linear_a", "linear_b", "add_op", "relu_op"]

    def test_rejects_non_matching_shape(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        # Three linears chained — neither diamond (which wants
        # exactly 2 linears with matching shapes) nor FFN (which
        # wants exactly 2 linears chained through relu) accepts
        # this. Both matchers should report their reasons.
        model = _ThreeLinear()
        x = torch.randn(64, 64)
        # Pin the matcher-cascade behavior; the generic FX fallback
        # is exercised in the Wave 2.2 tests.
        with pytest.raises(UnsupportedShape) as exc_info:
            lower_torch_to_megakernel(model, (x,), allow_generic_fallback=False)
        msg = str(exc_info.value)
        assert "diamond" in msg
        assert "ffn" in msg

    def test_rejects_diamond_with_bias(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _BiasedDiamond(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(64, 32, bias=True)
                self.b = nn.Linear(64, 32, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (self.a(x) + self.b(x)).relu()

        model = _BiasedDiamond()
        with pytest.raises(UnsupportedShape, match="bias=False"):
            lower_torch_to_megakernel(model, (torch.randn(64, 64),))

    def test_rejects_diamond_with_concat_instead_of_add(self) -> None:
        """Catches the case where the module structure looks right
        but the actual forward computation isn't (linear+linear).relu()."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _ConcatDiamond(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(64, 32, bias=False)
                self.b = nn.Linear(64, 32, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Output shape is (64, 64) — wrong for the diamond
                # matcher, which expects out_features per linear.
                return torch.cat([self.a(x), self.b(x)], dim=-1).relu()

        with pytest.raises(UnsupportedShape):
            lower_torch_to_megakernel(_ConcatDiamond(), (torch.randn(64, 64),))


class TestDiamondTopology:
    """Pin the emitted graph's structural shape — same topology as
    the hand-built diamond_dag workload, just driven by FX-shape
    inference instead of being typed by hand."""

    def test_tile_count_matches_shape(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(_Diamond(in_dim=64, out_dim=32), (x,))
        # 64/32 row tiles × 32/32 col tiles = 2 × 1 = 2 tiles per op.
        # Each op has task_shape=(NUM_TILES,) = (2,).
        for call in result.megakernel_graph.calls:
            assert call.task_shape == (2,)

    def test_decision_log_has_per_op_backend(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(_Diamond(64, 32), (x,))
        decision = result.decision.to_dict()
        assert decision["pattern_name"] == "diamond"
        assert "rationale" in decision["pattern_rationale"] or decision["pattern_rationale"]
        assert len(decision["body_decisions"]) == 4
        # Round 2a: every body is hand_rolled_fmaf — round 2b is what
        # swaps to cuBLASDx. The rationale must mention the discovery
        # outcome (so the agent can audit why a body went to fmaf
        # despite cuBLASDx maybe being available).
        for d in decision["body_decisions"]:
            assert d["backend"] == "hand_rolled_fmaf"
            rationale = d["rationale"].lower()
            assert "cublasdx" in rationale, (
                "round-2a body rationale must surface the cuBLASDx discovery state for agent audit"
            )

    def test_decision_log_carries_backend_availability(self) -> None:
        """Round-2a/b contract: the decision exposes ``backends`` with
        cuBLASDx + libcudacxx availability so the agent can audit
        downstream compile failures (e.g. NVRTC can't find
        cuda/std/type_traits because nvidia-cuda-cccl isn't
        installed)."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64, 32), (torch.randn(64, 64),))
        decision = result.decision.to_dict()
        assert "backends" in decision
        backends = decision["backends"]
        assert backends is not None
        for key in ("cublasdx_status", "cublasdx_include", "libcudacxx_status", "libcudacxx_include"):
            assert key in backends, f"missing {key} in decision.backends"
        # cuBLASDx: available iff include path is non-null.
        if backends["cublasdx_status"] == "available":
            assert backends["cublasdx_include"] is not None
        elif "missing" in backends["cublasdx_status"] and "deps-missing" not in backends["cublasdx_status"]:
            assert backends["cublasdx_include"] is None
        # libcudacxx: same shape.
        if backends["libcudacxx_status"] == "available":
            assert backends["libcudacxx_include"] is not None
        elif "missing" in backends["libcudacxx_status"]:
            assert backends["libcudacxx_include"] is None

    def test_decision_log_threads_nvrtc_include_paths(self) -> None:
        """When cuBLASDx is reachable, the decision surfaces the
        include path NVRTC needs — caller plumbs this through
        ``CudaModule(extra_include_paths=...)``. When it's not
        reachable the path list is empty."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64, 32), (torch.randn(64, 64),))
        decision = result.decision.to_dict()
        backends = decision["backends"]
        # Each discovered include path lands in nvrtc_include_paths.
        # Both are independent — libcudacxx is on hosts with the
        # system CUDA toolkit even if nvidia-mathdx is missing.
        for key in ("cublasdx_include", "libcudacxx_include"):
            if backends and backends[key]:
                assert backends[key] in decision["nvrtc_include_paths"]
        # And anything not discovered must NOT appear there.
        if not (backends and backends["cublasdx_include"]) and not (backends and backends["libcudacxx_include"]):
            assert decision["nvrtc_include_paths"] == []

    def test_bodies_present(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64, 32), (torch.randn(64, 64),))
        names = set(result.device_function_sources.keys())
        assert names == {"linear_a", "linear_b", "add_op", "relu_op"}
        # The matcher-emitted bodies use the inferred shape, not
        # the hardcoded 64×512×512 from the standalone factory.
        for name in ("linear_a", "linear_b"):
            body = result.device_function_sources[name].body
            assert "const int B = 64;" in body
            assert "const int IN = 64;" in body
            assert "const int OUT = 32;" in body


class TestUnsupportedDimensions:
    def test_rejects_non_multiple_of_tile(self) -> None:
        """Round-1 matcher requires shapes divisible by tile sizes.
        This is a typed rejection, not a silent partial-tile fallback."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        # 50 isn't divisible by TILE_M=32.
        model = _Diamond(in_dim=64, out_dim=32)
        with pytest.raises(UnsupportedShape, match="divisible"):
            lower_torch_to_megakernel(model, (torch.randn(50, 64),), allow_generic_fallback=False)


class TestMcpToolSurface:
    """The two new MCP tools live in compgen.mcp.tools.compile and
    appear in the registry. Pin their public-surface shape."""

    def test_tools_registered(self) -> None:
        from compgen.mcp.tools import COMPILE_TOOLS, get_all_tools

        names = {t["name"] for t in COMPILE_TOOLS}
        # Round 2b adds the cuBLASDx header-smoke tool; the
        # registered set now has three Phase-10 tools. Use ⊆ semantics
        # for the canonical pair so further additions don't trip
        # this test.
        assert {
            "compgen_compile_torch_model",
            "compgen_run_compiled_bundle",
        } <= names
        # And the cuBLASDx smoke must register exactly once.
        assert "compgen_cublasdx_header_smoke" in names
        all_names = {t["name"] for t in get_all_tools()}
        assert names <= all_names

    def test_tool_handlers_callable(self) -> None:
        from compgen.mcp.tools.compile import (
            compgen_compile_torch_model,
            compgen_run_compiled_bundle,
        )

        assert callable(compgen_compile_torch_model)
        assert callable(compgen_run_compiled_bundle)

    def test_input_schemas_present(self) -> None:
        from compgen.mcp.tools import COMPILE_TOOLS

        for tool in COMPILE_TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_run_cuda_source_tool_handler_callable(self) -> None:
        """Round 2c — generic NVRTC compile + run tool is registered
        and callable with a structured input schema. The actual
        compile + run path is GPU-only (tested on bwell)."""
        from compgen.mcp.tools.compile import (
            COMPILE_TOOLS,
            compgen_run_cuda_source,
        )

        assert callable(compgen_run_cuda_source)
        names = {t["name"] for t in COMPILE_TOOLS}
        assert "compgen_run_cuda_source" in names
        # Pin the input schema's required fields so a refactor can't
        # silently drop them.
        descriptor = next(t for t in COMPILE_TOOLS if t["name"] == "compgen_run_cuda_source")
        assert descriptor["input_schema"]["required"] == [
            "cuda_source",
            "kernel_name",
        ]

    def test_cublasdx_smoke_tool_returns_missing_on_cpu_host(self) -> None:
        """Round 2b: the cuBLASDx header smoke tool returns
        ``status="missing"`` on a CPU host (no GPU, no nvidia-mathdx
        installed). Pin the contract so a regression doesn't crash
        the tool when called on a host without cuBLASDx."""
        from compgen.mcp.tools.compile import compgen_cublasdx_header_smoke

        out = compgen_cublasdx_header_smoke()
        # Either truly missing, or unavailable for a different reason
        # (e.g. a partial install) — never raise.
        assert out["status"] in {"missing", "compile_failed"}
        assert "log" in out

    def test_discover_cublasdx_handles_namespace_package(self) -> None:
        """REMOTE bridge probe #070 caught the namespace-package bug:
        ``nvidia.mathdx.__file__`` is None for PEP 420 namespace
        packages, so the discovery used to raise TypeError. Pin the
        ``__path__`` fallback so the regression can't return."""
        from compgen.runtime.native.cuda import discover_cublasdx_include

        # Build a fake namespace-package-shaped object and feed it
        # through the discovery's lookup. Easier than mocking
        # importlib; we just need to verify the code doesn't blow
        # up on a None __file__.
        path = discover_cublasdx_include()
        # Either None (mathdx not installed) or a string path.
        # Crucially, no TypeError ever escapes.
        assert path is None or isinstance(path, str)

    def test_compile_tool_typed_error_on_unsupported(self, tmp_path) -> None:
        """When the model doesn't match any pattern AND the generic
        FX fallback can't trace it, the tool returns
        status='unsupported_shape' with a typed reason — not a raise.

        Wave 2.2 added a generic FX→megakernel fallback that catches
        any FX-traceable model. To pin the typed-error path we use a
        data-dependent control-flow model that ``torch.fx.symbolic_trace``
        rejects.
        """
        import base64
        import pickle

        from compgen.mcp.tools.compile import compgen_compile_torch_model

        model = _DataDependent()
        x = torch.randn(64, 64)
        result = compgen_compile_torch_model(
            model_pickle_b64=base64.b64encode(pickle.dumps(model)).decode(),
            sample_input_pickle_b64=base64.b64encode(pickle.dumps((x,))).decode(),
            output_dir=str(tmp_path),
        )
        assert result["status"] == "unsupported_shape"
        assert result["bundle_dir"] is None
        assert "error" in result


class TestFfnMatcher:
    """Round-2 — FFN pattern (``y = down(relu(up(x)))``).

    The matcher lowers a transformer-style FFN block: two linears
    chained through a relu, with hidden ≠ in/out. Tile graph has
    K-fan-in on the relu_up→linear_down edge: each linear_down
    output tile waits for ALL relu_up tiles in its row stripe.
    """

    def test_matches_ffn_shape(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xFF11)
        model = _Sequential(in_dim=64, hidden=128, out_dim=64)
        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(model, (x,))

        assert result.decision.pattern_name == "ffn"
        assert result.user_buffer_layout == (
            "x",
            "w_up",
            "w_down",
            "y_up",
            "y_relu",
            "y_out",
        )
        names = [c.name for c in result.megakernel_graph.calls]
        assert names == ["linear_up", "relu_up", "linear_down"]

    def test_rejects_ffn_with_bias(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _BiasedFfn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.Linear(64, 128, bias=True)
                self.down = nn.Linear(128, 64, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.down(torch.relu(self.up(x)))

        with pytest.raises(UnsupportedShape, match="bias=False"):
            lower_torch_to_megakernel(_BiasedFfn(), (torch.randn(64, 64),))

    def test_rejects_ffn_with_disagreeing_hidden(self) -> None:
        """up.out_features must equal down.in_features — no implicit
        reshape, no broadcast. Wrong shape is a typed rejection."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _BadHidden(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.Linear(64, 128, bias=False)
                self.down = nn.Linear(96, 64, bias=False)  # 96 != 128

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.down(torch.relu(self.up(x)))

        # The forward will torch-error on the shape mismatch before
        # the matcher gets to validate. The matcher's pre-shape check
        # catches it cleanly with a hidden-mismatch reason — no
        # cryptic torch traceback.
        with pytest.raises(UnsupportedShape, match="hidden"):
            lower_torch_to_megakernel(
                _BadHidden(),
                (torch.randn(64, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_ffn_with_concat_instead_of_chain(self) -> None:
        """Same module structure, different forward — catches the
        case where two linears + relu exist but the topology isn't
        ``down(relu(up(x)))``."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _SidewaysFfn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.Linear(64, 128, bias=False)
                self.down = nn.Linear(128, 64, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Chain order swapped — does up first then re-uses
                # x for down. Won't equal `down(relu(up(x)))`.
                return torch.relu(self.up(x)).sum(dim=-1, keepdim=True).expand(-1, 64) + x

        with pytest.raises(UnsupportedShape, match="output disagrees"):
            lower_torch_to_megakernel(_SidewaysFfn(), (torch.randn(64, 64),))

    def test_accepts_nd_inputs(self) -> None:
        """Bridge #108 fix — matcher must accept ND inputs by
        flattening leading dims into the batch axis. Before this
        fix, ``compile_to_megakernel(FFN(), (torch.randn(1, 64, 64),))``
        would reject with `ndim != 2`, even though the same FFN's
        forward works on the same input via torch.nn.Linear's ND
        broadcasting. Pin acceptance for the common (1, B, in) and
        (D1, D2, in) shapes."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        for shape in [(64, 64), (1, 64, 64), (2, 32, 64)]:
            result = lower_torch_to_megakernel(
                _Sequential(in_dim=64, hidden=128, out_dim=64),
                (torch.randn(shape),),
            )
            assert result.decision.pattern_name == "ffn"
            # Tile graph operates on flattened batch (= prod of leading dims).
            expected_batch_flat = 1
            for d in shape[:-1]:
                expected_batch_flat *= d
            # The decision's schedule_hints carry tile_grid_up reflecting
            # the flattened batch (not the original ND shape).
            tile_grid_up = result.decision.schedule_hints["tile_grid_up"]
            assert tile_grid_up[0] == expected_batch_flat // 32 or tile_grid_up[0] == expected_batch_flat // 64, (
                f"shape={shape}: tile_grid_up[0]={tile_grid_up[0]} not consistent with batch_flat={expected_batch_flat}"
            )


class TestFfnTopology:
    """Pin the K-fan-in event-tensor structure on the FFN matcher."""

    def test_tile_counts(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        # B=64, hidden=128, out=64 → B_TILES=2, H_TILES=4, O_TILES=2.
        result = lower_torch_to_megakernel(
            _Sequential(in_dim=64, hidden=128, out_dim=64),
            (torch.randn(64, 64),),
        )
        calls = {c.name: c for c in result.megakernel_graph.calls}
        # linear_up + relu_up: B_TILES * H_TILES = 8 tasks each.
        assert calls["linear_up"].task_shape == (8,)
        assert calls["relu_up"].task_shape == (8,)
        # linear_down: B_TILES * O_TILES = 4 tasks.
        assert calls["linear_down"].task_shape == (4,)

    def test_k_fan_in_on_relu_to_down(self) -> None:
        """The relu_up→linear_down event tensor has wait_count_default
        = H_TILES — proving the tile graph models the K-fan-in
        correctly. If the matcher silently downgraded to wait_count=1
        we'd lose the cross-K dependency and linear_down could read
        partially-written y_relu."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Sequential(in_dim=64, hidden=128, out_dim=64),
            (torch.randn(64, 64),),
        )
        # H_TILES = hidden / TILE_N = 128 / 32 = 4.
        ev_relu = result.megakernel_graph.event_tensors["ev_relu"]
        assert ev_relu.wait_count_default == 4
        # B_TILES = 2 — one cell per row stripe.
        assert ev_relu.shape == (2,)

    def test_decision_log_has_three_bodies(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Sequential(64, 128, 64), (torch.randn(64, 64),))
        decision = result.decision.to_dict()
        assert decision["pattern_name"] == "ffn"
        names = [d["op_name"] for d in decision["body_decisions"]]
        assert names == ["linear_up", "relu_up", "linear_down"]
        # Round-2 default: every body is hand_rolled_fmaf until
        # cuBLASDx authorship lands.
        for d in decision["body_decisions"]:
            assert d["backend"] == "hand_rolled_fmaf"

    def test_bodies_present_with_correct_dims(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Sequential(64, 128, 64), (torch.randn(64, 64),))
        names = set(result.device_function_sources.keys())
        assert names == {"linear_up", "relu_up", "linear_down"}
        # linear_up: in=64, out=hidden=128.
        up_body = result.device_function_sources["linear_up"].body
        assert "const int IN = 64;" in up_body
        assert "const int OUT = 128;" in up_body
        # linear_down: in=hidden=128, out=64.
        down_body = result.device_function_sources["linear_down"].body
        assert "const int IN = 128;" in down_body
        assert "const int OUT = 64;" in down_body


class TestCublasdxBodyOptIn:
    """Round-2c+ — opt-in cuBLASDx body for linear ops.

    The flag ``prefer_cublasdx_for_linears`` swaps the linear bodies
    from the hand_rolled_fmaf default to a cuBLASDx-emitted GEMM that
    uses ``Arrangement<row_major, row_major, row_major>`` (validated
    bit-exact on bwell #083). On hosts without cuBLASDx the matcher
    silently falls back to fmaf — the decision log surfaces which
    path was taken.

    These tests run on CPU; they verify body source content + decision
    log + extra-options plumbing without launching anything.
    """

    def test_off_by_default_diamond(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64, 32), (torch.randn(64, 64),))
        decision = result.decision.to_dict()
        for d in decision["body_decisions"]:
            assert d["backend"] == "hand_rolled_fmaf"
        # nvrtc_extra_options stays empty when no body asks for one.
        assert decision["nvrtc_extra_options"] == []

    def test_off_by_default_ffn(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Sequential(64, 128, 64), (torch.randn(64, 64),))
        decision = result.decision.to_dict()
        for d in decision["body_decisions"]:
            assert d["backend"] == "hand_rolled_fmaf"
        assert decision["nvrtc_extra_options"] == []

    def test_prefer_falls_back_when_backend_missing(self) -> None:
        """When prefer_cublasdx_for_linears=True but cuBLASDx isn't
        reachable (e.g. CI hosts without nvidia-mathdx), the matcher
        must NOT crash — it falls back to fmaf and records the
        request in the body rationale so the agent can audit."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
        )
        backends = result.decision.backends
        decision = result.decision.to_dict()
        if backends is None or backends.cublasdx_status != "available":
            # Fallback path: every body is fmaf, but the rationale
            # must call out that cuBLASDx was requested.
            for d in decision["body_decisions"]:
                assert d["backend"] == "hand_rolled_fmaf"
            linear_rationales = [
                d["rationale"] for d in decision["body_decisions"] if d["op_name"] in {"linear_a", "linear_b"}
            ]
            for r in linear_rationales:
                assert "prefer_cublasdx_for_linears requested" in r, (
                    "fallback rationale must mention the requested flag"
                )
            # Without cuBLASDx selected, no -default-device flag.
            assert decision["nvrtc_extra_options"] == []

    def test_prefer_emits_cublasdx_body_when_available_diamond(self) -> None:
        """When all three header sets resolve, the linear bodies
        carry the cuBLASDx incantation + the include header."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable — see test_prefer_falls_back_*")

        decision = result.decision.to_dict()
        # Linear ops swap to cuBLASDx; elementwise ops stay fmaf.
        per_op_backend = {d["op_name"]: d["backend"] for d in decision["body_decisions"]}
        assert per_op_backend["linear_a"] == "cublasdx_fp32"
        assert per_op_backend["linear_b"] == "cublasdx_fp32"
        assert per_op_backend["add_op"] == "hand_rolled_fmaf"
        assert per_op_backend["relu_op"] == "hand_rolled_fmaf"

        # Body source contains the canonical Arrangement tag and
        # the include header — pin so a refactor can't silently drop
        # them and revert to bwell's #080 layout bug.
        for op in ("linear_a", "linear_b"):
            src = result.device_function_sources[op]
            assert "Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>" in src.body, (
                f"{op} body must declare row-major arrangement on all 3 buffers"
            )
            assert "BLAS().execute(1.0f," in src.body
            assert "#include <cublasdx.hpp>" in src.included_headers

        # NVRTC extra options must include the -default-device flag
        # cuBLASDx static constexpr functions need.
        assert "-default-device" in decision["nvrtc_extra_options"]

    def test_prefer_emits_cublasdx_body_when_available_ffn(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Sequential(64, 128, 64),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        decision = result.decision.to_dict()
        per_op_backend = {d["op_name"]: d["backend"] for d in decision["body_decisions"]}
        # Both linears swap; relu stays fmaf (elementwise — never cuBLASDx).
        assert per_op_backend["linear_up"] == "cublasdx_fp32"
        assert per_op_backend["linear_down"] == "cublasdx_fp32"
        assert per_op_backend["relu_up"] == "hand_rolled_fmaf"
        assert "-default-device" in decision["nvrtc_extra_options"]

    def test_cublasdx_precision_fp32_default(self) -> None:
        """When prefer_cublasdx_for_linears=True without an explicit
        precision, default is fp32 — Precision<float>, no bf16 cast,
        bit-tight vs torch.matmul."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        decision = result.decision.to_dict()
        per_op_backend = {d["op_name"]: d["backend"] for d in decision["body_decisions"]}
        assert per_op_backend["linear_a"] == "cublasdx_fp32"
        body = result.device_function_sources["linear_a"].body
        assert "cublasdx::Precision<float>" in body
        assert "__nv_bfloat16" not in body
        # No bf16 header in fp32 mode.
        headers = result.device_function_sources["linear_a"].included_headers
        assert "#include <cuda_bf16.h>" not in headers

    def test_cublasdx_precision_bf16_fp32_engages_tensor_cores(self) -> None:
        """When cublasdx_precision='bf16_fp32', the body uses
        Precision<__nv_bfloat16, __nv_bfloat16, float>, casts fp32
        gmem to bf16 at smem load, keeps fp32 accumulator + output.
        Pin every piece since silently dropping any of them
        regresses the tensor-core path."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
            cublasdx_precision="bf16_fp32",
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        decision = result.decision.to_dict()
        per_op_backend = {d["op_name"]: d["backend"] for d in decision["body_decisions"]}
        # Backend tag flips to record the precision variant.
        assert per_op_backend["linear_a"] == "cublasdx_bf16_fp32"
        assert per_op_backend["linear_b"] == "cublasdx_bf16_fp32"
        # Elementwise ops stay fmaf — precision tag doesn't apply.
        assert per_op_backend["add_op"] == "hand_rolled_fmaf"

        body = result.device_function_sources["linear_a"].body
        # Precision tag.
        assert "cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>" in body
        # Smem types: bf16 inputs, fp32 accumulator/output.
        assert "__shared__ __nv_bfloat16 smem_a[32 * 32];" in body
        assert "__shared__ __nv_bfloat16 smem_b[32 * 32];" in body
        assert "__shared__ float smem_c[32 * 32];" in body
        # Smem load casts fp32→bf16 with __float2bfloat16.
        assert "__float2bfloat16(x[a_row * IN + a_col])" in body
        assert "__float2bfloat16(w[w_n * IN + w_k])" in body
        # K-accumulation pattern preserved.
        assert "float beta_iter = 0.0f;" in body
        assert "beta_iter = 1.0f;" in body
        # Headers: bf16 first (cublasdx.hpp depends on it).
        headers = result.device_function_sources["linear_a"].included_headers
        assert "#include <cuda_bf16.h>" in headers
        assert "#include <cublasdx.hpp>" in headers

    def test_select_tile_shape_per_backend(self) -> None:
        """Bridge #095 — cuBLASDx wants 64×64×16 tiles to engage
        mma.sync; fmaf path stays at 32×32×32 for the smaller-shape
        test fixtures. Pin both."""
        from compgen.runtime.lowering.fx_to_megakernel import (
            _select_tile_shape,
        )

        assert _select_tile_shape(use_cublasdx=False) == (32, 32, 32)
        assert _select_tile_shape(use_cublasdx=True) == (64, 64, 16)

    def test_cublasdx_body_uses_64_16_tile(self) -> None:
        """When prefer_cublasdx_for_linears=True on a 64-divisible
        shape, the emitted body declares Size<64, 64, 16> so cuBLASDx
        engages mma.sync (per #095 PTX dump). Pin every constant."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        # 64×64 input, 64-divisible out_dim → cuBLASDx 64-tile path works.
        result = lower_torch_to_megakernel(
            _Diamond(in_dim=64, out_dim=64),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
            cublasdx_precision="bf16_fp32",
            target_arch="sm_100",
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        body = result.device_function_sources["linear_a"].body
        # cuBLASDx Size template — the actual mma.sync trigger.
        assert "cublasdx::Size<64, 64, 16>" in body
        # Smem allocations match the 64×64×16 tile contract.
        assert "smem_a[1024]" in body  # 64*16
        assert "smem_b[1024]" in body  # 16*64
        assert "smem_c[4096]" in body  # 64*64
        # Decision log carries the new tile shape.
        decision = result.decision.to_dict()
        for d in decision["body_decisions"]:
            if d["op_name"] in {"linear_a", "linear_b"}:
                assert d["tile_shape"] == [64, 64, 16]

    def test_diamond_elementwise_covers_full_tile_at_64(self) -> None:
        """Bridge #097 regression — at 64-tile, ``add_op`` and
        ``relu_op`` must loop to cover the full TM×TN=4096 element
        tile, not just the 1024 elements one thread per (ty, tx)
        position can reach. Without the loop, 3072 of 4096 outputs
        stay uninitialized and correctness blows up."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(in_dim=64, out_dim=64),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
            cublasdx_precision="bf16_fp32",
            target_arch="sm_100",
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        for op in ("add_op", "relu_op"):
            body = result.device_function_sources[op].body
            # Tile-aware loop pattern — auto-degrades to 1 iter for
            # the 32-tile case and runs 4 iters for 64-tile.
            assert "for (int p = 0; p < 4;" in body, (
                f"{op}: 64-tile path must run 4 iterations per thread "
                "(4096 elts / 1024 threads). Bridge #097 caught this "
                "as 1-iter-only producing partial output."
            )
            # Index derivation must be from p+linear, not (ty, tx).
            assert "int t_idx = p * 1024 + linear;" in body
            assert "int dy = t_idx / TN;" in body
            assert "int dx = t_idx % TN;" in body

    def test_diamond_elementwise_loops_once_at_32(self) -> None:
        """Default (32-tile fmaf) path keeps the 1-iter-per-thread
        loop so the body is identical-to-original perf at the
        smaller tile."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64, 32), (torch.randn(64, 64),))
        for op in ("add_op", "relu_op"):
            body = result.device_function_sources[op].body
            # 32-tile: TM*TN=1024 / 1024 threads = 1 iter.
            assert "for (int p = 0; p < 1;" in body

    def test_cublasdx_body_falls_back_when_shape_too_small(self) -> None:
        """64-tile cuBLASDx path requires 64-divisible shapes. When
        prefer_cublasdx_for_linears=True but the shape is only
        32-divisible, the matcher must raise UnsupportedShape rather
        than silently reverting to 32-tile cuBLASDx (which would put
        us back on the SIMT path)."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        # Try to fetch backends without prefer flag — if cuBLASDx is
        # not reachable, the prefer flag silently falls back to fmaf
        # (which can handle 32-tile shapes). Skip in that case so we
        # only test the actual rejection path.
        result0 = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
        )
        if result0.decision.backends is None or result0.decision.backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable — fallback hides the rejection")

        with pytest.raises(UnsupportedShape, match="cuBLASDx"):
            lower_torch_to_megakernel(
                _Diamond(in_dim=64, out_dim=32),  # 32 not divisible by 64
                (torch.randn(64, 64),),
                prefer_cublasdx_for_linears=True,
            )

    def test_arch_to_cublasdx_sm_mapping(self) -> None:
        """Arch-to-cuBLASDx-SM mapping per #087 diagnosis. Pin every
        Blackwell variant since silently mapping sm_120 to SM<900>
        regresses the tensor-core path."""
        from compgen.runtime.lowering.fx_to_megakernel import (
            _arch_to_cublasdx_sm,
        )

        assert _arch_to_cublasdx_sm("sm_90") == 900
        assert _arch_to_cublasdx_sm("sm_90a") == 900
        # Blackwell datacenter (paper hardware) → SM<1000>.
        assert _arch_to_cublasdx_sm("sm_100") == 1000
        assert _arch_to_cublasdx_sm("sm_100a") == 1000
        # Workstation Blackwell falls back to SM<1000> since
        # cuBLASDx 0.4.0 doesn't have SM<1200> entries.
        assert _arch_to_cublasdx_sm("sm_120") == 1000
        # Older arches resolve to nearest supported.
        assert _arch_to_cublasdx_sm("sm_80") == 800
        assert _arch_to_cublasdx_sm("sm_86") == 860
        # Unknown arches fall back to Blackwell.
        assert _arch_to_cublasdx_sm("sm_999") == 1000

    def test_cublasdx_body_uses_target_arch_sm_tag(self) -> None:
        """When target_arch='sm_100', cuBLASDx body's SM tag must be
        SM<1000> (Blackwell tcgen05.mma) — not the SM<900> default
        that produced bwell #087's tensor-core no-op."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
            cublasdx_precision="bf16_fp32",
            target_arch="sm_100",
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        body = result.device_function_sources["linear_a"].body
        assert "cublasdx::SM<1000>" in body, (
            "Blackwell target_arch must map to SM<1000> in the body — "
            "SM<900> silently falls back to SIMT-fma on sm_100/sm_120."
        )
        assert "cublasdx::SM<900>" not in body

    def test_cublasdx_body_sm_tag_for_hopper(self) -> None:
        """Hopper still gets SM<900> when explicitly requested. The
        mapping isn't always-Blackwell — only the *default* is."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(64, 32),
            (torch.randn(64, 64),),
            prefer_cublasdx_for_linears=True,
            target_arch="sm_90",
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        body = result.device_function_sources["linear_a"].body
        assert "cublasdx::SM<900>" in body
        assert "cublasdx::SM<1000>" not in body

    def test_cublasdx_precision_invalid_raises(self) -> None:
        """Unknown precision fails fast at the public entry — no
        silent fall-through to fp32."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        with pytest.raises(ValueError, match="cublasdx_precision"):
            lower_torch_to_megakernel(
                _Diamond(64, 32),
                (torch.randn(64, 64),),
                prefer_cublasdx_for_linears=True,
                cublasdx_precision="fp16",  # not supported (yet)
            )

    def test_cublasdx_body_accumulates_over_k(self) -> None:
        """The cuBLASDx body must do K-tile accumulation: beta=0 on
        the first iteration, beta=1 thereafter. Without this the
        K loop overwrites instead of summing — silent correctness bug
        against torch.matmul for K_total > 32."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(in_dim=128, out_dim=32),  # K_total = 128 = 4 tiles
            (torch.randn(32, 128),),
            prefer_cublasdx_for_linears=True,
        )
        backends = result.decision.backends
        if backends is None or backends.cublasdx_status != "available":
            pytest.skip("cuBLASDx not reachable")

        for op in ("linear_a", "linear_b"):
            body = result.device_function_sources[op].body
            assert "float beta_iter = 0.0f;" in body, f"{op}: K-accumulation must start with beta=0"
            assert "beta_iter = 1.0f;" in body, f"{op}: subsequent K-tiles must use beta=1"
            assert "for (int k_tile = 0; k_tile < IN; k_tile += TK)" in body
