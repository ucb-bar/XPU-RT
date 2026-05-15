"""CUDA-graph capture wrapper around an arbitrary forward function.

Replays the entire forward as ONE CUDA graph launch instead of
``N kernels × per-launch overhead``. At small sequence lengths where
TinyLlama's forward is dominated by per-kernel launch latency, this
is the single biggest perf lever on consumer GPUs.

Usage::

    wrapper = CudaGraphCaptureWrapper(model_fn, sample_inputs)
    out = wrapper(input_ids)        # first call: capture; subsequent: replay

Edge cases:
  * **Dynamic shapes**: one graph per ``(B, S)`` tuple, cached.
    Different shapes pay capture cost the first time only.
  * **Side-effects in capture**: anything that mutates Python state,
    allocates new tensors mid-forward, or branches on tensor values
    will break capture. Caller must keep the forward pure on a fixed
    input/weight set.
  * **Memory pool**: capture uses a private memory pool so allocations
    inside the captured region are deterministic on replay.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class _CapturedGraph:
    """One captured graph keyed by (B, S) shape tuple."""

    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor  # the graph reads from this tensor
    static_output: torch.Tensor  # the graph writes to this tensor
    pool: Any = None  # private memory pool handle


@dataclass
class CudaGraphCaptureWrapper:
    """Wrap a forward function with shape-keyed CUDA-graph caching.

    Attributes:
        model_fn: ``forward(input_tensor) -> output_tensor`` callable.
            Pure on a fixed weight set; no Python branches on tensor
            values inside.
        warmup_iters: How many warmup runs before capture (lets autotune
            sweeps + JIT compiles complete; capture would otherwise
            include their cost).
    """

    model_fn: Callable[[torch.Tensor], torch.Tensor]
    warmup_iters: int = 3
    _graphs: dict[tuple, _CapturedGraph] = field(default_factory=dict)
    _enabled: bool = True

    def disable(self) -> None:
        """Force-disable graph replay (e.g. for debugging)."""
        self._enabled = False

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return self.model_fn(x)
        if not torch.cuda.is_available() or x.device.type != "cuda":
            return self.model_fn(x)

        key = tuple(x.shape)
        cached = self._graphs.get(key)
        if cached is None:
            cached = self._capture(x)
            self._graphs[key] = cached

        cached.static_input.copy_(x)
        cached.graph.replay()
        return cached.static_output.clone()  # clone so caller can mutate

    def _capture(self, sample: torch.Tensor) -> _CapturedGraph:
        """Run warmup + capture for one input shape."""
        log.debug("cuda_graph.capture", extra={"shape": tuple(sample.shape)})

        static_input = sample.clone().detach()

        # 1. Warmup on the side stream so the default stream stays clean
        # for capture. CUDA-graph capture refuses to run on the default stream.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                out = self.model_fn(static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # 2. Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = self.model_fn(static_input)

        return _CapturedGraph(
            graph=graph,
            static_input=static_input,
            static_output=captured_out,
        )

    def num_cached_graphs(self) -> int:
        return len(self._graphs)

    def cached_shapes(self) -> list[tuple]:
        return list(self._graphs.keys())


__all__ = ["CudaGraphCaptureWrapper"]
