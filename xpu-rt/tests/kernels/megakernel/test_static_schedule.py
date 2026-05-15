"""Static-schedule megakernel regression tests (row-sum, attention, Llama MLP).

These tests execute the **actually-emitted** persistent megakernels --
no hand-written Triton, no protocol stubs.  Each example builds the
event.graph, runs Algorithm 1 (StaticMegakernelSchedule), lowers via
``lower_megakernel`` with real ``DeviceFunctionSpec`` bodies, imports
the emitted source as a module, launches the kernel on the GPU, and
compares the output to a trustworthy PyTorch reference.

The bodies live in ``examples/event_tensor/`` so they are reusable
outside the test suite (the user can ``python -m examples...`` them
to inspect what the compiler emitted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "event_tensor"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require a CUDA device")


# ---------------------------------------------------------------------------
# Example 1: row-sum (paper Fig. 3 verbatim)
# ---------------------------------------------------------------------------


def test_row_sum_megakernel_matches_torch_sum() -> None:
    from examples.event_tensor.row_sum_megakernel import (
        compile_megakernel,
        reference,
        run_megakernel,
    )

    compiled = compile_megakernel(n_row_blocks=8, j_chunks=4)
    torch.manual_seed(0)
    a = torch.randn(
        (compiled.n_row_blocks * compiled.block_m, compiled.j_chunks * compiled.block_k),
        dtype=torch.float32,
        device="cuda",
    )
    got = run_megakernel(compiled, a)
    ref = reference(a)
    assert (got - ref).abs().max().item() < 1e-3


def test_row_sum_megakernel_kernel_source_is_emitted_not_handwritten() -> None:
    """Sanity: the kernel source must contain the emitter's marker comments."""
    from examples.event_tensor.row_sum_megakernel import compile_megakernel

    compiled = compile_megakernel(n_row_blocks=2, j_chunks=2)
    src = compiled.kernel_source
    assert "Per-SM task table baked into the megakernel at compile time." in src
    assert "Persistent megakernel emitted by ETC Algorithm 1" in src
    assert "_run_partial_sum" in src and "_run_final_sum" in src


# ---------------------------------------------------------------------------
# Example 2: Llama / Gemma attention block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_heads,seq_len,head_dim",
    [
        (2, 32, 32),
        (4, 64, 32),
    ],
)
def test_attention_megakernel_matches_torch_sdpa(
    n_heads: int,
    seq_len: int,
    head_dim: int,
) -> None:
    from examples.event_tensor.attention_megakernel import (
        compile_attention_megakernel,
        reference_attention,
        run_attention_megakernel,
    )

    compiled = compile_attention_megakernel(
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        q_tile_size=16,
    )
    torch.manual_seed(123)
    q = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device="cuda")
    k = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device="cuda")
    v = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device="cuda")

    got = run_attention_megakernel(compiled, q, k, v)
    ref = reference_attention(q, k, v)
    err = (got - ref).abs().max().item()
    assert err < 1e-3, f"attention output diverges by {err}"


def test_attention_megakernel_emits_real_softmax_dot_chain() -> None:
    """The emitted kernel must call tl.dot, tl.exp, and tl.sum -- proves the
    real attention math is in the emitted source, not a placeholder."""
    from examples.event_tensor.attention_megakernel import (
        compile_attention_megakernel,
    )

    compiled = compile_attention_megakernel(n_heads=2, seq_len=32, head_dim=32)
    src = compiled.kernel_source
    assert "tl.dot" in src
    assert "tl.exp" in src
    assert "tl.sum" in src


# ---------------------------------------------------------------------------
# Example 3: Llama / Gemma SwiGLU MLP block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K,I,N",
    [
        (16, 32, 64, 32),
        (32, 64, 128, 64),
    ],
)
def test_llama_mlp_megakernel_matches_pytorch_reference(
    M: int,
    K: int,
    I: int,
    N: int,
) -> None:
    from examples.event_tensor.llama_mlp_megakernel import (
        compile_mlp_megakernel,
        reference_mlp,
        run_mlp_megakernel,
    )

    compiled = compile_mlp_megakernel(M=M, K=K, I=I, N=N)
    torch.manual_seed(7)
    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_gate = torch.randn((I, K), dtype=torch.float32, device="cuda") * 0.05
    w_up = torch.randn((I, K), dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn((N, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_mlp_megakernel(compiled, x, w_gate, w_up, w_down)
    ref = reference_mlp(x, w_gate, w_up, w_down)
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"MLP output diverges by {err}"


def test_llama_mlp_megakernel_uses_two_event_tensors() -> None:
    """Sanity: the MLP graph declares EG (gate) + EU (up); the emitter
    must thread both pointers through the persistent kernel signature."""
    from examples.event_tensor.llama_mlp_megakernel import compile_mlp_megakernel

    compiled = compile_mlp_megakernel(M=16, K=32, I=64, N=32)
    src = compiled.kernel_source
    assert "EG_ptr" in src
    assert "EU_ptr" in src
    assert "_run_gate_proj_tile" in src
    assert "_run_up_proj_tile" in src
    assert "_run_down_proj_tile" in src
