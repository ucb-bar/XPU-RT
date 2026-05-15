"""Distributed runtime adapter for ``compgen.collective`` ops.

Wraps PyTorch's ``torch.distributed`` primitives so
``compgen.collective.{all_reduce, all_gather, reduce_scatter,
broadcast}`` ops can actually execute on a multi-GPU host.

Graceful degradation on single-process / CPU hosts:

- When ``torch.distributed`` is not initialized, collectives
  behave as single-process identity ops (the sane semantic for
  world_size=1).
- When ``world_size > 1`` but the user hasn't called ``init()``,
  we init on ``gloo`` (CPU) or ``nccl`` (if CUDA) automatically.

Usage::

    from compgen.runtime.distributed import (
        DistributedAdapter, init_if_needed,
    )
    init_if_needed(backend="nccl")  # or "gloo" on CPU
    adapter = DistributedAdapter()
    y = adapter.all_reduce(x, op="sum")

This module is the concrete integration point for the
distributed passes. The xDSL IR side is complete (shard_tensors_spmd,
insert_all_reduce, etc.); this is how you execute it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


_REDUCE_OP_MAP = {
    "sum": "SUM",
    "mean": "AVG",  # on NCCL >= 2.10, else fallback to SUM/world_size
    "max": "MAX",
    "min": "MIN",
    "prod": "PRODUCT",
}


@dataclass
class DistributedEnv:
    world_size: int = 1
    rank: int = 0
    backend: str = "none"
    initialized: bool = False


def distributed_available() -> bool:
    """Whether ``torch.distributed`` + a backend are importable."""
    try:
        import torch

        return hasattr(torch, "distributed") and torch.distributed.is_available()
    except ImportError:
        return False


def current_env() -> DistributedEnv:
    """Snapshot of the current ``torch.distributed`` state."""
    env = DistributedEnv()
    if not distributed_available():
        return env
    import torch.distributed as dist

    if dist.is_initialized():
        env.initialized = True
        env.world_size = dist.get_world_size()
        env.rank = dist.get_rank()
        env.backend = dist.get_backend()
    return env


def init_if_needed(
    *,
    backend: str = "auto",
    init_method: str | None = None,
    world_size: int | None = None,
    rank: int | None = None,
) -> DistributedEnv:
    """Initialize torch.distributed if it isn't already.

    ``backend="auto"`` picks ``nccl`` when CUDA is available, else
    ``gloo``. When torch.distributed isn't importable, returns a
    single-process env with ``initialized=False``.
    """
    if not distributed_available():
        return DistributedEnv()
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return current_env()

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Set sensible defaults when the env vars aren't present.
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if init_method is None:
        init_method = os.environ.get("COMPGEN_INIT_METHOD", "env://")

    # For world_size=1 we can skip init entirely -- the adapter's
    # ops will fall through to identity.
    if world_size <= 1:
        log.info("distributed.skip_init_single_process")
        return DistributedEnv(world_size=1, rank=0, backend="none")

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("distributed.init_failed", error=str(exc))
        return DistributedEnv()

    return current_env()


class DistributedAdapter:
    """Thin wrapper around ``torch.distributed`` collectives.

    Every method degrades gracefully to an identity no-op when
    ``world_size=1`` or when ``torch.distributed`` isn't initialized.
    """

    def __init__(self, env: DistributedEnv | None = None) -> None:
        self.env = env if env is not None else current_env()

    def _is_single(self) -> bool:
        return not self.env.initialized or self.env.world_size <= 1

    def all_reduce(self, tensor: Any, *, op: str = "sum") -> Any:
        if self._is_single():
            return tensor
        import torch.distributed as dist

        op_attr = getattr(dist.ReduceOp, _REDUCE_OP_MAP.get(op, "SUM"), dist.ReduceOp.SUM)
        dist.all_reduce(tensor, op=op_attr)
        return tensor

    def all_gather(
        self,
        tensor: Any,
        *,
        dim: int = 0,
    ) -> Any:
        if self._is_single():
            return tensor
        import torch
        import torch.distributed as dist

        gathered = [torch.zeros_like(tensor) for _ in range(self.env.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)

    def reduce_scatter(
        self,
        tensor: Any,
        *,
        op: str = "sum",
        dim: int = 0,
    ) -> Any:
        if self._is_single():
            return tensor
        import torch
        import torch.distributed as dist

        # Split along dim into world_size shards.
        shards = list(torch.chunk(tensor, self.env.world_size, dim=dim))
        out = torch.zeros_like(shards[0])
        op_attr = getattr(
            dist.ReduceOp,
            _REDUCE_OP_MAP.get(op, "SUM"),
            dist.ReduceOp.SUM,
        )
        dist.reduce_scatter(out, shards, op=op_attr)
        return out

    def broadcast(self, tensor: Any, *, source_replica: int = 0) -> Any:
        if self._is_single():
            return tensor
        import torch.distributed as dist

        dist.broadcast(tensor, src=source_replica)
        return tensor


__all__ = [
    "DistributedAdapter",
    "DistributedEnv",
    "current_env",
    "distributed_available",
    "init_if_needed",
]
