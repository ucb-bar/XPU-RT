"""Dynamic-schedule megakernel regression tests (dynamic row-sum, MoE).

Like ``test_static_schedule.py``, every test in this file
executes the **actually-emitted** persistent megakernel produced by
:func:`compgen.ir.tile.lower_megakernel_dynamic.lower_megakernel_dynamic`
on a real GPU and compares to a trustworthy PyTorch reference.  No
hand-written Triton, no protocol stubs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require CUDA")


# ---------------------------------------------------------------------------
# Dynamic scheduler validation: row-sum (paper Fig. 3 with policy=dynamic)
# ---------------------------------------------------------------------------


def test_dynamic_row_sum_matches_torch_sum() -> None:
    from examples.event_tensor.row_sum_dynamic_megakernel import (
        compile_dynamic_megakernel,
        reference,
        run_dynamic_megakernel,
    )

    compiled = compile_dynamic_megakernel(n_row_blocks=8, j_chunks=4)
    torch.manual_seed(0)
    a = torch.randn(
        (compiled.n_row_blocks * compiled.block_m, compiled.j_chunks * compiled.block_k),
        dtype=torch.float32,
        device="cuda",
    )
    got = run_dynamic_megakernel(compiled, a)
    ref = reference(a)
    assert (got - ref).abs().max().item() < 1e-3


def test_dynamic_emitted_kernel_uses_push_pop_protocol() -> None:
    """The emitted source must use the on-GPU MPMC queue protocol --
    QUEUE_HEAD_PTR + QUEUE_TAIL_PTR + per-slot QUEUE_VALID_PTR --
    not a precomputed per-SM table."""
    from examples.event_tensor.row_sum_dynamic_megakernel import (
        compile_dynamic_megakernel,
    )

    compiled = compile_dynamic_megakernel(n_row_blocks=2, j_chunks=2)
    src = compiled.kernel_source
    assert "QUEUE_HEAD_PTR" in src
    assert "QUEUE_TAIL_PTR" in src
    assert "QUEUE_VALID_PTR" in src
    # No precomputed per-SM task table (that's the static emitter's marker).
    assert "Per-SM task table baked" not in src
    # Both atomic_xchg (publish) and atomic_or (acquire-load) must appear.
    assert "tl.atomic_xchg" in src
    assert "tl.atomic_or" in src


def test_dynamic_initial_queue_seeds_only_root_tasks() -> None:
    """Tasks with no in_edges must be the only entries in initial_queue."""
    from examples.event_tensor.row_sum_dynamic_megakernel import (
        compile_dynamic_megakernel,
    )

    compiled = compile_dynamic_megakernel(n_row_blocks=4, j_chunks=2)
    initial = compiled.lowering.initial_queue
    # 8 producers (4 row-blocks * 2 chunks), all with kind == partial_sum.
    assert len(initial) == 8
    # final_sum tasks have in_edges; must NOT be in the initial seed.
    expert_kind = compiled.lowering.device_function_table
    partial_kind = next(k for k, fn in expert_kind.items() if fn == "partial_sum")
    final_kind = next(k for k, fn in expert_kind.items() if fn == "final_sum")
    initial_kinds = {k for _tid, k in initial}
    assert initial_kinds == {partial_kind}
    assert final_kind not in initial_kinds


# ---------------------------------------------------------------------------
# Real MoE workload: data-dep event.update / event.trigger semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_experts,n_tokens,top_k,head_dim",
    [
        (4, 8, 2, 32),
        (8, 16, 2, 32),
    ],
)
def test_moe_megakernel_matches_pytorch_reference(
    n_experts: int,
    n_tokens: int,
    top_k: int,
    head_dim: int,
) -> None:
    from examples.event_tensor.moe_megakernel import (
        compile_moe_megakernel,
        reference_moe,
        route_tokens,
        run_moe_megakernel,
    )

    torch.manual_seed(99)
    x = torch.randn((n_tokens, head_dim), dtype=torch.float32, device="cuda")
    w = (
        torch.randn(
            (n_experts, head_dim, head_dim),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.05
    )
    router = torch.randn((n_tokens, n_experts), dtype=torch.float32, device="cuda")
    routing = route_tokens(router, n_experts, top_k, max_slots_per_expert=16)

    compiled = compile_moe_megakernel(
        n_experts=n_experts,
        n_tokens=n_tokens,
        top_k=top_k,
        head_dim=head_dim,
        max_slots_per_expert=16,
    )
    got = run_moe_megakernel(compiled, x, w, routing)
    ref = reference_moe(x, w, routing)
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"MoE diverges by {err}"


def test_moe_per_expert_counts_drive_event_initialization() -> None:
    """The host-seeded event counter values must match the runtime
    per-expert token count -- this is the paper's data-dep `event.update`."""
    from examples.event_tensor.moe_megakernel import route_tokens

    torch.manual_seed(7)
    router = torch.randn((16, 8), dtype=torch.float32, device="cuda")
    routing = route_tokens(router, n_experts=8, top_k=2, max_slots_per_expert=16)
    # Sum of per-expert counts must equal total gather tasks (T*K).
    assert int(routing.per_expert_count.sum().item()) == 16 * 2
    # exp_indptr must be a strict prefix sum of per_expert_count.
    expected = torch.zeros((9,), dtype=torch.int32, device="cuda")
    expected[1:] = torch.cumsum(routing.per_expert_count, dim=0)
    assert torch.all(routing.exp_indptr == expected)


def test_moe_emitted_kernel_handles_two_device_functions() -> None:
    """The MoE megakernel must emit both gather_tile and expert_compute
    bodies in the same persistent kernel."""
    from examples.event_tensor.moe_megakernel import compile_moe_megakernel

    compiled = compile_moe_megakernel(n_experts=4, n_tokens=8, top_k=2, head_dim=32)
    src = compiled.kernel_source
    assert "_run_gather_tile" in src
    assert "_run_expert_compute" in src
    # Both functions appear in the dispatch table.
    assert sorted(compiled.lowering.device_function_table.values()) == [
        "expert_compute",
        "gather_tile",
    ]
