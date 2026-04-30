"""Wave 2.1 — pattern-catalog matcher tests.

Pins the matcher cascade for the three transformer-shaped patterns
added in Wave 2.1 (residual+norm, MHA, MoE). Each pattern gets:

* a *match test* — model with the right shape lowers to the expected
  ``pattern_name``,
* a *reject test* — variant model that should fall through (e.g.
  bias=True, mismatched head dim, missing top_k),
* a *composition test* — residual+norm over an FFN/diamond produces
  the ``residual_norm@<sub>`` pattern,
* a *schedule sanity* — ``total_tasks`` is reasonable for a small
  shape, plus (for MoE) the dynamic-schedule pass builds without
  raising on the (8-token, 4-expert, top_k=2) shape the user pinned.

Wave-1 contract preserved: 3-D inputs ``(B, S, D)`` work the same
way they do in FFN per bridge #108. We pin acceptance for both
``(B, S, D)`` and ``(B*S, D)`` everywhere it makes sense.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_tasks(graph) -> int:
    """Sum of ``prod(task_shape)`` across every DeviceCall."""
    total = 0
    for c in graph.calls:
        n = 1
        for d in c.task_shape:
            n *= int(d)
        total += n
    return total


class _FFN(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class _Diamond(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


# ---------------------------------------------------------------------------
# residual+norm
# ---------------------------------------------------------------------------


class _ResidualFFN(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.sub = _FFN(d, hidden)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.sub(x))


class _ResidualDiamond(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.sub = _Diamond(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.sub(x))


class TestResidualNormMatcher:
    def test_matches_residual_over_ffn(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xDA1)
        model = _ResidualFFN(d=64, hidden=128)
        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(model, (x,))

        assert result.decision.pattern_name == "residual_norm@ffn"
        # Sublayer's calls are renamed with ``sub__`` prefix; tail
        # ops are residual_add + ln_mean + ln_var + ln_affine.
        names = [c.name for c in result.megakernel_graph.calls]
        for tail in ("residual_add", "ln_mean", "ln_var", "ln_affine"):
            assert tail in names
        assert any(n.startswith("sub__") for n in names), (
            "sublayer calls must be prefixed to avoid event-name collisions"
        )
        # Sublayer's body sources flow through with the prefix.
        body_names = set(result.device_function_sources.keys())
        assert "residual_add" in body_names
        assert "ln_affine" in body_names
        # Sublayer pattern flagged on schedule_hints.
        assert result.decision.schedule_hints["sublayer_pattern"] == "ffn"

    def test_matches_residual_over_diamond(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xDA2)
        model = _ResidualDiamond(d=64)
        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(model, (x,))

        assert result.decision.pattern_name == "residual_norm@diamond"
        assert result.decision.schedule_hints["sublayer_pattern"] == "diamond"

    def test_rejects_residual_with_layernorm_no_affine(self) -> None:
        """Wave 2.1 requires LayerNorm with affine=True (γ + β). A
        plain affine=False is a typed rejection so the agent's audit
        sees why it didn't lower."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = _FFN(64, 128)
                self.norm = nn.LayerNorm(64, elementwise_affine=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x + self.sub(x))

        with pytest.raises(UnsupportedShape, match="affine"):
            lower_torch_to_megakernel(
                M(),
                (torch.randn(64, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_residual_with_concat_instead_of_add(self) -> None:
        """LayerNorm(cat([x, sub(x)])) doesn't match — the matcher's
        forward equality check rejects."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = _FFN(64, 128)
                self.norm = nn.LayerNorm(64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Note: same shape as input, but content differs from
                # x + sub(x).
                return self.norm(x * self.sub(x))

        with pytest.raises(UnsupportedShape):
            lower_torch_to_megakernel(
                M(),
                (torch.randn(64, 64),),
                allow_generic_fallback=False,
            )

    def test_residual_total_tasks_reasonable(self) -> None:
        """Sublayer tasks + 4 tail tasks per row tile."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_ResidualFFN(d=64, hidden=128), (torch.randn(64, 64),))
        total = _total_tasks(result.megakernel_graph)
        # FFN tasks: linear_up=8, relu_up=8, linear_down=4 → 20.
        # Tail: 4 ops × 2 row tiles = 8. Sum = 28.
        assert total == 28

    def test_residual_accepts_nd_inputs(self) -> None:
        """Bridge #108 contract — ND inputs flatten leading dims."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        for shape in [(64, 64), (1, 64, 64), (2, 32, 64)]:
            result = lower_torch_to_megakernel(
                _ResidualFFN(d=64, hidden=128),
                (torch.randn(shape),),
            )
            assert result.decision.pattern_name == "residual_norm@ffn"


# ---------------------------------------------------------------------------
# MHA
# ---------------------------------------------------------------------------


class _HandRolledMHA(nn.Module):
    """Hand-rolled q/k/v/o module — alternative MHA shape the matcher
    must recognise alongside ``nn.MultiheadAttention``."""

    def __init__(self, d: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads
        d_h = D // H
        q = self.q(x).view(B, S, H, d_h).transpose(1, 2)
        k = self.k(x).view(B, S, H, d_h).transpose(1, 2)
        v = self.v(x).view(B, S, H, d_h).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (d_h**0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)


class TestMhaMatcher:
    def test_matches_nn_multihead_attention(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xA77E)
        m = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            bias=False,
            batch_first=True,
        )
        x = torch.randn(2, 32, 64)
        result = lower_torch_to_megakernel(m, (x,))

        assert result.decision.pattern_name == "mha"
        names = [c.name for c in result.megakernel_graph.calls]
        # 4 linears + 1 batched-matmul (Q@K^T) + 4 softmax tile-tasks +
        # 1 batched-matmul (attn@V).
        assert "q_proj" in names
        assert "k_proj" in names
        assert "v_proj" in names
        assert "qk_matmul" in names
        assert "softmax_max" in names
        assert "softmax_exp" in names
        assert "softmax_sum" in names
        assert "softmax_div" in names
        assert "av_matmul" in names
        assert "o_proj" in names

        hints = result.decision.schedule_hints
        assert hints["num_heads"] == 4
        assert hints["head_dim"] == 16
        assert hints["mha_causal"] is False

    def test_matches_hand_rolled_mha(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xA77F)
        m = _HandRolledMHA(d=64, num_heads=4)
        x = torch.randn(2, 32, 64)
        result = lower_torch_to_megakernel(m, (x,))
        assert result.decision.pattern_name == "mha"

    def test_rejects_mha_with_mismatched_head_dim(self) -> None:
        """embed_dim must be divisible by num_heads."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        # nn.MultiheadAttention itself rejects this in __init__ — the
        # matcher's check is for the hand-rolled form.
        class _BadMHA(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 64 not divisible by 5 → matcher rejection.
                self.q = nn.Linear(64, 64, bias=False)
                self.k = nn.Linear(64, 64, bias=False)
                self.v = nn.Linear(64, 64, bias=False)
                self.o = nn.Linear(64, 64, bias=False)
                self.num_heads = 5

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.o(x)  # arbitrary; matcher rejects before forward

        with pytest.raises(UnsupportedShape, match="divisible by num_heads"):
            lower_torch_to_megakernel(
                _BadMHA(),
                (torch.randn(2, 32, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_mha_with_bias(self) -> None:
        """Wave-2.1 simplification: bias=False on QKVO. nn.MHA with
        bias=True falls through."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        m = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            bias=True,
            batch_first=True,
        )
        with pytest.raises(UnsupportedShape):
            lower_torch_to_megakernel(
                m,
                (torch.randn(2, 32, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_mha_with_2d_input(self) -> None:
        """MHA wants 3-D ``(B, S, D)``. 2-D is rejected."""
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        m = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            bias=False,
            batch_first=True,
        )
        with pytest.raises(UnsupportedShape, match="3-D"):
            lower_torch_to_megakernel(
                m,
                (torch.randn(64, 64),),
                allow_generic_fallback=False,
            )

    def test_mha_total_tasks_reasonable(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        # B=2, S=32, D=64, H=4 → s_tiles=1, d_tiles=2.
        # qkv tiles per linear = B*s*d = 4. score tiles = B*H*s*s = 8.
        # softmax row tiles = B*H*s = 8.
        m = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            bias=False,
            batch_first=True,
        )
        result = lower_torch_to_megakernel(m, (torch.randn(2, 32, 64),))
        total = _total_tasks(result.megakernel_graph)
        # All tile-tasks present; bound it to a sane cap.
        assert 0 < total < 1000

    def test_mha_causal_flag_propagates(self) -> None:
        """Hand-rolled module exposing ``is_causal=True`` lands in the
        schedule_hints so the kernel emitter (Wave 2.2) picks the
        masked-softmax variant."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        m = _HandRolledMHA(d=64, num_heads=4)
        m.is_causal = True
        result = lower_torch_to_megakernel(m, (torch.randn(2, 32, 64),))
        assert result.decision.schedule_hints["mha_causal"] is True


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class _MoE(nn.Module):
    def __init__(
        self,
        d: int = 64,
        n_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([_FFN(d, hidden=2 * d) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trivial passthrough — Wave-2.1 matcher's forward probe just
        # validates that the output is same-shape; the dynamic
        # dispatch lives in the lowered megakernel, not the
        # python-level forward.
        return x


class TestMoeMatcher:
    def test_matches_moe_shape(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        torch.manual_seed(0xE0E)
        m = _MoE(d=64, n_experts=4, top_k=2)
        x = torch.randn(8, 64)
        result = lower_torch_to_megakernel(m, (x,))

        assert result.decision.pattern_name == "moe"
        # Dynamic-schedule path.
        assert result.megakernel_graph.policy == "dynamic"
        assert result.decision.schedule_hints["requires_ondevice_scheduler"] is True

    def test_moe_emits_one_trigger_per_expert(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel
        from compgen.runtime.lowering.pattern_catalog import (
            build_moe_trigger_generators,
        )

        m = _MoE(d=64, n_experts=4, top_k=2)
        result = lower_torch_to_megakernel(m, (torch.randn(8, 64),))
        trigs = build_moe_trigger_generators(result.decision)
        assert len(trigs) == 4
        # Each trigger names a distinct expert event.
        target_events = {t.target_event for t in trigs}
        assert target_events == {f"ev_expert_{e}" for e in range(4)}
        # And points back at the matching device-func.
        for t in trigs:
            assert t.target_device_func.startswith("expert_")
            assert t.source_tensor == "topk_indices"

    def test_moe_dynamic_schedule_builds(self) -> None:
        """The (8 token, 4 expert, top_k=2) shape pinned by Wave 2.1
        must produce a schedule without raising. Validates the matcher
        + dynamic-schedule pass compose end-to-end."""
        from compgen.runtime.lowering import lower_torch_to_megakernel
        from compgen.runtime.lowering.pattern_catalog import (
            build_moe_trigger_generators,
        )
        from compgen.transforms.event_dynamic_schedule import (
            compute_dynamic_schedule,
        )

        m = _MoE(d=64, n_experts=4, top_k=2)
        result = lower_torch_to_megakernel(m, (torch.randn(8, 64),))
        trigs = build_moe_trigger_generators(result.decision)

        sched = compute_dynamic_schedule(
            result.megakernel_graph,
            sm_count=8,
            trigger_generators=trigs,
            supports_ondevice_scheduler=True,
        )
        # Schedule has tasks + ready queue + the trigger generators
        # threaded through.
        assert len(sched.tasks) > 0
        assert len(sched.trigger_generators) == 4
        # router_proj has 0 predecessors → seeds the queue.
        assert len(sched.ready_queue.initial_task_ids) >= 1

    def test_rejects_moe_without_topk(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = nn.Linear(64, 4, bias=False)
                self.experts = nn.ModuleList([_FFN(64, 128) for _ in range(4)])
                # No top_k attribute.

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises(UnsupportedShape, match="top_k"):
            lower_torch_to_megakernel(
                M(),
                (torch.randn(8, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_moe_with_router_bias(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = nn.Linear(64, 4, bias=True)
                self.experts = nn.ModuleList([_FFN(64, 128) for _ in range(4)])
                self.top_k = 2

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises(UnsupportedShape, match="router"):
            lower_torch_to_megakernel(
                M(),
                (torch.randn(8, 64),),
                allow_generic_fallback=False,
            )

    def test_rejects_moe_with_topk_exceeding_n_experts(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        m = _MoE(d=64, n_experts=4, top_k=5)
        with pytest.raises(UnsupportedShape, match="top_k"):
            lower_torch_to_megakernel(
                m,
                (torch.randn(8, 64),),
                allow_generic_fallback=False,
            )

    def test_moe_total_tasks_reasonable(self) -> None:
        """8 tokens, 4 experts, top_k=2 → small task count.

        At n_tokens=8 < tile_m=32, n_token_tiles=1.
        Tasks: router_proj(1) + router_topk(1) + 4×expert(1) +
        combine(1) = 7.
        """
        from compgen.runtime.lowering import lower_torch_to_megakernel

        m = _MoE(d=64, n_experts=4, top_k=2)
        result = lower_torch_to_megakernel(m, (torch.randn(8, 64),))
        total = _total_tasks(result.megakernel_graph)
        assert total == 7


# ---------------------------------------------------------------------------
# Composition + cascade order
# ---------------------------------------------------------------------------


class TestPatternCascadeOrder:
    """Pin that the Wave-2.1 matchers run AFTER the Wave-1 matchers
    (a plain FFN still matches as ``"ffn"``, not ``"residual_norm@..."``
    or anything else)."""

    def test_plain_ffn_still_matches_as_ffn(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_FFN(64, 128), (torch.randn(64, 64),))
        assert result.decision.pattern_name == "ffn"

    def test_plain_diamond_still_matches_as_diamond(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(_Diamond(64), (torch.randn(64, 64),))
        assert result.decision.pattern_name == "diamond"
