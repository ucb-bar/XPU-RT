"""Wave 2.5 — FFN epilogue fusion (per bridge #137).

Folds the relu into ``linear_up``'s MMA epilogue so the tile graph
shrinks from three ops to two (``linear_up_relu`` + ``linear_down``)
and the ``y_up`` round-trip + the entire pointwise pool disappear.
The motivating data: at MLP-1 paper shapes the cost model from #126
reported ``overhead_share=57%`` mostly pointwise, and cluster-locality
intra_frac was 1.8-5.0% — well below the partitioner gate. Fusion
leapfrogs that whole class of perf problems by removing the
bipartite layer structure outright.

These tests cover the matcher + emitter + decision metadata. CUDA
correctness lives behind ``requires_gpu`` in the conformance harness;
here we exercise the structural shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Ffn(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.up = nn.Linear(in_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class TestFusedTopology:
    def test_drops_relu_op(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF1)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        names = [c.name for c in result.megakernel_graph.calls]
        assert names == ["linear_up_relu", "linear_down"]

    def test_drops_y_up_buffer(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF2)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        assert result.user_buffer_layout == (
            "x",
            "w_up",
            "w_down",
            "y_relu",
            "y_out",
        )

    def test_drops_ev_up_event_tensor(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF3)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        ev_names = set(result.megakernel_graph.event_tensors)
        assert ev_names == {"ev_relu", "ev_done"}

    def test_total_tile_tasks_drops_pointwise_pool(self) -> None:
        """3-op topology: n_up + n_up + n_down = 8+8+4 = 20.
        Fused: n_up + n_down = 8 + 4 = 12."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF4)
        unfused = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=False,
        )
        fused = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        assert fused.decision.total_tile_tasks < unfused.decision.total_tile_tasks
        # The eliminated pointwise pool is exactly n_up tasks.
        b_tiles, h_tiles = fused.decision.schedule_hints["tile_grid_up"]
        n_up = b_tiles * h_tiles
        assert unfused.decision.total_tile_tasks - fused.decision.total_tile_tasks == n_up

    def test_decision_carries_fusion_marker(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF5)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        assert result.decision.schedule_hints.get("epilogue_fusion") == "relu_into_linear_up"
        assert "Wave 2.5" in result.decision.pattern_rationale


class TestFusedBodies:
    def test_linear_up_relu_writes_y_relu(self) -> None:
        """Buffer layout dropped y_up; linear_up_relu must store to
        buffer index 3 (y_relu in the 5-buffer layout)."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF6)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        body = result.device_function_sources["linear_up_relu"].body
        # Output buffer index resolves to 3 (y_relu) in fused layout.
        assert "buffers[3]" in body

    def test_linear_up_relu_applies_relu_in_epilogue(self) -> None:
        """The store epilogue must clamp negative accumulator values
        to zero. fmaf path uses a ternary; cuBLASDx path uses fmaxf."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF7)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        body = result.device_function_sources["linear_up_relu"].body
        # fmaf path stores `(acc > 0.0f ? acc : 0.0f)`. cuBLASDx
        # path stores `fmaxf(smem_c[idx], 0.0f)`.
        assert ("acc > 0.0f" in body) or ("fmaxf" in body)

    def test_linear_down_reads_y_relu_at_buffer_3(self) -> None:
        """In the fused 5-buffer layout, linear_down's input buffer
        index is 3 (was 4 in the unfused 6-buffer layout because
        y_up sat at slot 3 and y_relu was bumped to 4)."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF8)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
            fuse_epilogue=True,
        )
        body = result.device_function_sources["linear_down"].body
        # x_buf=3, w_buf=2, out_buf=4 in the fused layout.
        assert "buffers[3]" in body
        assert "buffers[4]" in body


class TestDefaultUnchanged:
    """fuse_epilogue=False (the default) must produce the original
    3-op topology so the existing 33-test FFN suite + downstream
    consumers keep working."""

    def test_default_keeps_three_ops(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xF9)
        result = lower_torch_to_megakernel(
            _Ffn(),
            (torch.randn(64, 64),),
        )
        names = [c.name for c in result.megakernel_graph.calls]
        assert names == ["linear_up", "relu_up", "linear_down"]
        assert result.user_buffer_layout == (
            "x",
            "w_up",
            "w_down",
            "y_up",
            "y_relu",
            "y_out",
        )
