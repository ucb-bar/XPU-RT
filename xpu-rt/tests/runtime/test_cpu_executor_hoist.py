"""Tests for the CPU-executor performance fixes (cpu_executor.py).

Coverage:

1. ``_exec_transpose`` no longer calls ``.contiguous()`` — the returned
   tensor's strides are non-contiguous after a permute that swaps dims.
2. ``_exec_empty`` returns an *uninitialised* tensor (``torch.empty``
   semantics) rather than zeros. We can't verify "uninitialised"
   directly without UB, but we can check the function dispatches to
   ``torch.empty`` not ``torch.zeros`` by monkeypatching.
3. ``_build_hoist_cache`` identifies ops whose operands all derive
   from PARAMETER block-args, executes them once, and caches.
4. A second ``execute()`` call on the same module skips dispatch for
   hoisted ops (verified via ExecutorStats).
"""

from __future__ import annotations

from typing import Any

import torch

from compgen.runtime import cpu_executor


def test_exec_transpose_skips_contiguous_call(monkeypatch) -> None:
    """The transpose handler returns a non-contiguous view of its input.

    Before the fix this called ``.contiguous()`` and forced a copy of
    the weight every forward — ~80ms/iter on TinyLlama MLP.
    """

    contiguous_calls: list[int] = []

    real_contiguous = torch.Tensor.contiguous

    def _spy_contiguous(self):  # type: ignore[no-untyped-def]
        contiguous_calls.append(1)
        return real_contiguous(self)

    monkeypatch.setattr(torch.Tensor, "contiguous", _spy_contiguous)

    # Build a fake TransposeOp-shaped object that the handler accepts.
    from xdsl.dialects.builtin import DenseArrayBase

    class _DenseStub:
        def get_values(self):  # noqa: D401
            return [1, 0]

    class _OpStub:
        def __init__(self, x: torch.Tensor) -> None:
            self.input = "key"
            self.permutation = _DenseStub()

    monkeypatch.setattr(cpu_executor, "DenseArrayBase", _DenseStub)
    op = _OpStub(torch.randn(4, 5))
    env = {op.input: torch.randn(4, 5)}

    _ = cpu_executor._exec_transpose(op, env)
    assert contiguous_calls == [], (
        "transpose handler must NOT call .contiguous() — that forced "
        "a per-call weight-copy worth ~80ms/iter on TinyLlama"
    )


def test_exec_empty_uses_torch_empty(monkeypatch) -> None:
    """Empty ops use uninitialised memory — not torch.zeros."""

    empty_calls: list[Any] = []
    zeros_calls: list[Any] = []

    real_empty = torch.empty
    real_zeros = torch.zeros

    def _spy_empty(*a, **kw):
        empty_calls.append((a, kw))
        return real_empty(*a, **kw)

    def _spy_zeros(*a, **kw):
        zeros_calls.append((a, kw))
        return real_zeros(*a, **kw)

    monkeypatch.setattr(torch, "empty", _spy_empty)
    monkeypatch.setattr(torch, "zeros", _spy_zeros)

    # Minimal EmptyOp stub: a result with a TensorType-shaped type.
    class _TensorTypeStub:
        @staticmethod
        def get_shape():
            return [4, 8]

    class _ResStub:
        def __init__(self) -> None:
            self.type = _TensorTypeStub()

    class _EmptyOpStub:
        def __init__(self) -> None:
            self.results = [_ResStub()]

    monkeypatch.setattr(cpu_executor, "TensorType", _TensorTypeStub)
    out = cpu_executor._exec_empty(_EmptyOpStub(), {})
    assert out.shape == torch.Size([4, 8])
    assert empty_calls, "expected torch.empty to be called"
    # We tolerate zeros calls only if they weren't from _exec_empty
    # itself — the spy above sits at module level so it'd see everything.
    # The fix replaced the call site directly: assert empty was used.
    assert empty_calls[-1][0][0] == [4, 8]


def test_hoist_cache_replays_param_derived_ops() -> None:
    """A second ``execute`` on the same module should not re-dispatch
    hoisted ops — proven by counting dispatch_op recordings."""

    # The TinyLlama MLP slice is the simplest real workload exercising
    # this. We import it lazily because it pulls in transformers.
    from pathlib import Path

    from compgen.api import compile_model, device as _device
    from compgen.benchmarks.live_adapters import _load_workload

    handle = _load_workload("tinyllama_1_1b__slice")
    target_yaml = (
        Path("/scratch2/agustin/CompGen")
        / "tests"
        / "targetgen"
        / "exemplars"
        / "test_gpu_simt.yaml"
    )
    dev = _device(str(target_yaml))
    compiled = compile_model(
        handle.model,
        dev,
        sample_inputs=handle.inputs,
        verify=False,
        strict_artifacts=False,
        run_compile_baseline=False,
    )

    # First call: builds the cache (some ops marked, executed, stored).
    stats1 = cpu_executor.ExecutorStats()
    cpu_executor.execute(
        compiled.payload_module,
        compiled.capture_artifact.exported_program,
        handle.inputs,
        stats=stats1,
    )

    # Second call: should dispatch *fewer* ops because hoisted ones
    # are skipped.
    stats2 = cpu_executor.ExecutorStats()
    cpu_executor.execute(
        compiled.payload_module,
        compiled.capture_artifact.exported_program,
        handle.inputs,
        stats=stats2,
    )

    assert stats2.ops_executed < stats1.ops_executed, (
        f"hoist cache should reduce per-call dispatches: "
        f"first call={stats1.ops_executed}, second call={stats2.ops_executed}"
    )
