"""Wave 1.3 — ETC vs. eager cost prediction tests.

CPU-only. Pin the predictor's input/output contract + the
empirical regimes from bridge #099:

- Small per-task work + many tasks → ETC loses (scheduling dominates)
- Large per-task work + few tasks → ETC wins (scheduling amortizes)
- Tensor-core path on Blackwell → predicted speedup higher than fp32 SIMT

The numbers are conservative roofline bounds; actual perf will
differ. The contract this module pins is the **shape of the answer**
the agent gets, not the exact magnitudes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _PickleableDiamond(nn.Module):
    """Module-level so pickle can serialize it (test-method-local
    classes can't round-trip)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 32, bias=False)
        self.b = nn.Linear(64, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class TestPublicSurface:
    def test_top_level_imports(self) -> None:
        from compgen.kernels.cost import (
            EtcCostPrediction,
            WontWinError,
            predict_etc_dispatch,
        )

        assert EtcCostPrediction is not None
        assert WontWinError is not None
        assert callable(predict_etc_dispatch)
        assert issubclass(WontWinError, RuntimeError)


class TestPredictionContract:
    """The agent reads `cost_prediction["passes_gate"]` and decides
    dispatch path. Pin every field the agent might query."""

    def _fake_decision(
        self,
        *,
        num_tasks: int = 8,
        k_per_op: int = 64,
        backend: str = "cublasdx_bf16_fp32",
    ) -> dict:
        return {
            "pattern_name": "diamond",
            "body_decisions": [
                {"op_name": "linear_a", "backend": backend, "tile_shape": [64, 64, 16]},
                {"op_name": "linear_b", "backend": backend, "tile_shape": [64, 64, 16]},
                {"op_name": "add_op", "backend": "hand_rolled_fmaf", "tile_shape": [64, 64, 16]},
                {"op_name": "relu_op", "backend": "hand_rolled_fmaf", "tile_shape": [64, 64, 16]},
            ],
            "schedule_hints": {
                "tile_grid": [num_tasks // 4 // 1, 1],  # rough approximation
                "k_tiles": k_per_op // 16,
            },
            "total_tile_tasks": num_tasks,
        }

    def _fake_choice(
        self,
        *,
        arch: str = "sm_100",
        use_cublasdx: bool = True,
        precision: str = "bf16_fp32",
        tile: tuple[int, int, int] = (64, 64, 16),
    ) -> dict:
        return {
            "target_arch": arch,
            "tile_shape": list(tile),
            "use_cublasdx_for_linears": use_cublasdx,
            "cublasdx_precision": precision,
            "use_cu13_nvrtc": True,
        }

    def test_returns_prediction_with_required_fields(self) -> None:
        from compgen.kernels.cost import predict_etc_dispatch

        pred = predict_etc_dispatch(
            sample_input_shape=(64, 64),
            decision=self._fake_decision(),
            backend_choice=self._fake_choice(),
        )
        assert pred.etc_us > 0
        assert pred.eager_us > 0
        assert pred.speedup > 0
        assert pred.threshold == 1.2
        assert isinstance(pred.passes_gate, bool)
        assert pred.reason
        # Component breakdown is the agent's audit surface.
        for key in (
            "per_task_gemm_us",
            "per_task_overhead_us",
            "num_linear_tasks",
            "eager_gemm_us",
            "sm_count",
            "tile_shape",
            "use_cublasdx",
            "precision",
        ):
            assert key in pred.components, f"missing component {key}"

    def test_threshold_changes_passes_gate(self) -> None:
        """The same prediction with a different threshold flips
        the gate cleanly. Agent can ask "would 1.0× pass?" by
        re-running with threshold=1.0."""
        from compgen.kernels.cost import predict_etc_dispatch

        decision = self._fake_decision(num_tasks=4, k_per_op=512)
        choice = self._fake_choice()
        # Strict gate — likely fails.
        strict = predict_etc_dispatch(
            sample_input_shape=(64, 64),
            decision=decision,
            backend_choice=choice,
            threshold=2.0,
        )
        # Loose gate — same numbers, threshold lower.
        loose = predict_etc_dispatch(
            sample_input_shape=(64, 64),
            decision=decision,
            backend_choice=choice,
            threshold=0.5,
        )
        assert strict.speedup == pytest.approx(loose.speedup, rel=1e-9)
        # Either both pass or strict fails / loose passes.
        if strict.passes_gate:
            assert loose.passes_gate
        # Loose should be at-least as permissive.
        assert loose.passes_gate or not strict.passes_gate


class TestEmpiricalRegimes:
    """Pin the directional behavior matching bridge #099's empirics:
    small workloads with many tasks lose; large workloads with few
    tasks win on Blackwell."""

    def _diamond_choice_blackwell(self) -> dict:
        return {
            "target_arch": "sm_100",
            "tile_shape": [64, 64, 16],
            "use_cublasdx_for_linears": True,
            "cublasdx_precision": "bf16_fp32",
            "use_cu13_nvrtc": True,
        }

    def test_many_small_tasks_lose(self) -> None:
        """The #099 regime: 256 tasks each doing 64×64×16 GEMM. Per-
        task work is small (~1µs at TC) and per-task overhead is
        ~1µs, so total ≈ 512µs. Eager does the equivalent in one
        cuBLAS call across 132 SMs at TC throughput → ≈ 1µs.
        Predicted speedup << 1.0× — predictor must say so."""
        from compgen.kernels.cost import predict_etc_dispatch

        decision = {
            "pattern_name": "ffn",
            "body_decisions": [
                {"op_name": "linear_up", "backend": "cublasdx_bf16_fp32", "tile_shape": [64, 64, 16]},
                {"op_name": "relu_up", "backend": "hand_rolled_fmaf", "tile_shape": [64, 64, 16]},
                {"op_name": "linear_down", "backend": "cublasdx_bf16_fp32", "tile_shape": [64, 64, 16]},
            ],
            "schedule_hints": {
                "tile_grid": [8, 8],  # 64 tiles per linear
                "k_tiles_up": 4,
                "k_tiles_down": 4,
            },
            "total_tile_tasks": 256,
        }
        pred = predict_etc_dispatch(
            sample_input_shape=(64, 64),
            decision=decision,
            backend_choice=self._diamond_choice_blackwell(),
        )
        # The empirical loss regime — predictor surfaces
        # "scheduling dominates" in the reason.
        assert pred.passes_gate is False
        assert "scheduling" in pred.reason.lower() or "overhead" in pred.reason.lower()

    def test_few_large_tasks_win(self) -> None:
        """The hypothesized win regime: 4 tasks each doing a very-
        large-K GEMM. Per-task work scales linearly with K; overhead
        is fixed ~1µs. At big enough K the GEMM dominates and the
        bundle's scheduling cost amortizes.

        Using K=65536 (paper's MLP-1 scale): 64*64*65536 = 268M FLOPs
        / 50 TFLOPS at TC = ~5.4µs per task. Overhead = 1µs. 5.4× margin.
        """
        from compgen.kernels.cost import predict_etc_dispatch

        decision = {
            "pattern_name": "diamond",
            "body_decisions": [
                {"op_name": "linear_a", "backend": "cublasdx_bf16_fp32", "tile_shape": [64, 64, 16]},
                {"op_name": "linear_b", "backend": "cublasdx_bf16_fp32", "tile_shape": [64, 64, 16]},
                {"op_name": "add_op", "backend": "hand_rolled_fmaf", "tile_shape": [64, 64, 16]},
                {"op_name": "relu_op", "backend": "hand_rolled_fmaf", "tile_shape": [64, 64, 16]},
            ],
            "schedule_hints": {
                "tile_grid": [1, 1],  # 1 tile per linear
                "k_tiles": 4096,  # K=65536 → 4096 K-tiles
            },
            "total_tile_tasks": 4,
        }
        pred = predict_etc_dispatch(
            sample_input_shape=(64, 65536),
            decision=decision,
            backend_choice=self._diamond_choice_blackwell(),
        )
        # Per-task GEMM should now dominate the overhead.
        assert pred.components["per_task_gemm_us"] > pred.components["per_task_overhead_us"]

    def test_tensor_core_predicts_higher_speedup_than_simt(self) -> None:
        """On the same workload, the bf16+fp32-acc path should
        predict more compute throughput than fp32 SIMT — so ETC's
        per-task GEMM is faster, more competitive."""
        from compgen.kernels.cost import predict_etc_dispatch

        decision = {
            "pattern_name": "diamond",
            "body_decisions": [
                {"op_name": "linear_a", "backend": "cublasdx_bf16_fp32", "tile_shape": [64, 64, 16]},
            ],
            "schedule_hints": {"tile_grid": [2, 2], "k_tiles": 16},
            "total_tile_tasks": 8,
        }
        bf16 = predict_etc_dispatch(
            sample_input_shape=(64, 256),
            decision=decision,
            backend_choice={
                "target_arch": "sm_100",
                "tile_shape": [64, 64, 16],
                "use_cublasdx_for_linears": True,
                "cublasdx_precision": "bf16_fp32",
                "use_cu13_nvrtc": True,
            },
        )
        fp32 = predict_etc_dispatch(
            sample_input_shape=(64, 256),
            decision=decision,
            backend_choice={
                "target_arch": "sm_100",
                "tile_shape": [32, 32, 32],
                "use_cublasdx_for_linears": True,
                "cublasdx_precision": "fp32",
                "use_cu13_nvrtc": True,
            },
        )
        # bf16+TC has ~10× the throughput → per-task GEMM is ~10×
        # smaller → ETC total drops.
        assert bf16.components["per_task_gemm_us"] < fp32.components["per_task_gemm_us"]


class TestWontWinErrorIntegration:
    """Wire-up between ``compile_to_megakernel`` and the predictor."""

    def test_compile_stamps_cost_prediction(self, tmp_path) -> None:
        """The compile path attaches the prediction to the bundle.
        The agent reads ``bundle.cost_prediction["passes_gate"]`` to
        decide dispatch."""
        import compgen

        bundle = compgen.compile_to_megakernel(
            _PickleableDiamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        assert bundle.cost_prediction is not None
        for key in (
            "etc_us",
            "eager_us",
            "speedup",
            "threshold",
            "passes_gate",
            "components",
            "reason",
        ):
            assert key in bundle.cost_prediction

    def test_compile_writes_verification_report(self, tmp_path) -> None:
        """The prediction also lands in
        ``bundle/verification_report.json`` for the agent's audit
        query — same shape as the in-memory ``cost_prediction`` field."""
        import json

        import compgen

        bundle = compgen.compile_to_megakernel(
            _PickleableDiamond(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        report = json.loads((bundle.bundle_dir / "verification_report.json").read_text())
        assert "cost_prediction" in report
        assert report["cost_prediction"]["passes_gate"] == bundle.cost_prediction["passes_gate"]

    def test_fail_when_wont_win_raises_typed(self, tmp_path) -> None:
        """When the user asks to fail-fast on a losing prediction,
        WontWinError is raised. Carries the full prediction for
        audit."""
        import compgen
        from compgen.kernels.cost import WontWinError

        # Use threshold=1000× so any prediction loses.
        with pytest.raises(WontWinError) as exc_info:
            compgen.compile_to_megakernel(
                _PickleableDiamond(),
                (torch.randn(64, 64),),
                output_dir=str(tmp_path),
                fail_when_wont_win=True,
                perf_threshold=1000.0,
            )
        # Prediction is preserved on the exception for audit.
        assert exc_info.value.prediction is not None
        assert exc_info.value.prediction.passes_gate is False
        assert exc_info.value.prediction.reason
