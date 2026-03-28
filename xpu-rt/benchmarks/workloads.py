"""Benchmark workload definitions and model loaders."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from compgen.models import ModelSpec, build_default_model_catalog

LoaderFn = Callable[[], tuple[nn.Module, tuple[Any, ...]]]


MODEL_CATALOG = build_default_model_catalog()


def _example_loader(
    module_name: str,
    *,
    factory_name: str = "get_model_and_inputs",
    model_kwargs: dict[str, Any] | None = None,
    input_kwargs: dict[str, Any] | None = None,
) -> LoaderFn:
    """Create a lazy loader around an example module."""

    model_kwargs = model_kwargs or {}
    input_kwargs = input_kwargs or {}

    def _load() -> tuple[nn.Module, tuple[Any, ...]]:
        module = importlib.import_module(module_name)
        if hasattr(module, factory_name):
            factory = getattr(module, factory_name)
            if model_kwargs or input_kwargs:
                if module_name.endswith("simple_mlp") or module_name.endswith("quantized_mlp"):
                    model = module.SimpleMLP(**model_kwargs)
                elif module_name.endswith("transformer_block"):
                    model = module.TransformerBlock(**model_kwargs)
                else:
                    built_model, _ = factory()
                    model = type(built_model)(**model_kwargs)
                inputs = getattr(module, "get_sample_inputs")(**input_kwargs)
                return model, inputs
            return factory()
        raise AttributeError(f"{module_name} does not expose {factory_name}()")

    return _load


class MatmulBiasGELU(nn.Module):
    def __init__(self, m: int = 128, k: int = 256, n: int = 128) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, n))
        self.bias = nn.Parameter(torch.randn(n))
        self.m = m
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.weight + self.bias)


class MatmulAddRelu(nn.Module):
    def __init__(self, m: int = 128, k: int = 256, n: int = 128) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, n))
        self.residual = nn.Parameter(torch.randn(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x @ self.weight + self.residual)


class LayerNormChain(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = F.gelu(self.proj(x))
        return self.norm2(x + 0.125)


class SoftmaxElemwise(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.softmax(x, dim=-1)
        return y * self.scale + 0.5


class TransposePingPong(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(-1, -2)
        z = y.transpose(-1, -2)
        return self.proj(z)


class CopyBoundaryHeavy(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(self.proj(x), 2, dim=-1)
        return torch.cat([a.relu(), b.sigmoid()], dim=-1)


class ScanSmallKernels(nn.Module):
    def __init__(self, hidden_dim: int = 128, steps: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim))
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for _ in range(self.steps):
            y = torch.tanh(y + self.weight)
        return y


class ReductionBlock(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        return y.mean(dim=-1, keepdim=True) + y.sum(dim=-1, keepdim=True) * 1e-3


def _micro_loader(module_cls: type[nn.Module], *inputs: torch.Tensor, **kwargs: Any) -> LoaderFn:
    """Create a loader for a local microbenchmark module."""

    def _load() -> tuple[nn.Module, tuple[Any, ...]]:
        return module_cls(**kwargs), tuple(inputs)

    return _load


def _catalog_loader(model_id: str) -> LoaderFn:
    """Adapt a model-catalog spec to the legacy workload loader signature."""

    def _load() -> tuple[nn.Module, tuple[Any, ...]]:
        return MODEL_CATALOG.get(model_id).load()

    return _load


DEFAULT_LOADERS: dict[str, LoaderFn] = {
    "simple_mlp": _example_loader("examples.models.simple_mlp"),
    "transformer_block": _example_loader("examples.models.transformer_block"),
    "quantized_mlp": _example_loader("examples.models.quantized_mlp"),
    "simple_mlp_batch16": _example_loader(
        "examples.models.simple_mlp",
        model_kwargs={"input_dim": 768, "hidden_dim": 3072, "output_dim": 768},
        input_kwargs={"batch_size": 16, "input_dim": 768},
    ),
    "simple_mlp_batch32": _example_loader(
        "examples.models.simple_mlp",
        model_kwargs={"input_dim": 768, "hidden_dim": 3072, "output_dim": 768},
        input_kwargs={"batch_size": 32, "input_dim": 768},
    ),
    "transformer_block_seq8": _example_loader(
        "examples.models.transformer_block",
        model_kwargs={"d_model": 512, "num_heads": 8, "d_ff": 2048},
        input_kwargs={"batch_size": 2, "seq_len": 8, "d_model": 512},
    ),
    "transformer_block_seq32": _example_loader(
        "examples.models.transformer_block",
        model_kwargs={"d_model": 512, "num_heads": 8, "d_ff": 2048},
        input_kwargs={"batch_size": 2, "seq_len": 32, "d_model": 512},
    ),
    "quantized_mlp_batch16": _example_loader(
        "examples.models.quantized_mlp",
        model_kwargs={"input_dim": 768, "hidden_dim": 3072, "output_dim": 768},
        input_kwargs={"batch_size": 16, "input_dim": 768},
    ),
    "matmul_bias_gelu": _micro_loader(MatmulBiasGELU, torch.randn(8, 256), m=8, k=256, n=128),
    "matmul_add_relu": _micro_loader(MatmulAddRelu, torch.randn(8, 256), m=8, k=256, n=128),
    "layernorm_chain": _micro_loader(LayerNormChain, torch.randn(4, 32, 256), hidden_dim=256),
    "softmax_elemwise": _micro_loader(SoftmaxElemwise, torch.randn(4, 32, 256), hidden_dim=256),
    "transpose_pingpong": _micro_loader(TransposePingPong, torch.randn(4, 64, 128), hidden_dim=128),
    "copy_boundary_heavy": _micro_loader(CopyBoundaryHeavy, torch.randn(4, 64, 128), hidden_dim=128),
    "scan_small_kernels": _micro_loader(ScanSmallKernels, torch.randn(4, 64, 128), hidden_dim=128),
    "reduction_block": _micro_loader(ReductionBlock, torch.randn(4, 64, 128), hidden_dim=128),
    "llama31_decoder_block": _catalog_loader("llama31_decoder_block"),
    "llama31_8b_slice": _catalog_loader("llama31_8b_slice"),
    "llama4_moe_router_expert_block": _catalog_loader("llama4_moe_router_expert_block"),
    "dlrmv3_ranking_block": _catalog_loader("dlrmv3_ranking_block"),
    "mamba_block": _catalog_loader("mamba_block"),
    "convnext_stage": _catalog_loader("convnext_stage"),
    "smolvla_one_step": _catalog_loader("smolvla_one_step"),
    "groot_policy_step": _catalog_loader("groot_policy_step"),
    "cosmos_reason2": _catalog_loader("cosmos_reason2"),
    "cosmos_predict2_5": _catalog_loader("cosmos_predict2_5"),
    "cosmos_transfer2_5": _catalog_loader("cosmos_transfer2_5"),
}


@dataclass(frozen=True)
class LoaderEntry:
    """Named workload loader entry."""

    workload_id: str
    loader: LoaderFn


def get_loader(workload_id: str) -> LoaderFn:
    """Return a loader for a known workload id."""

    if workload_id not in DEFAULT_LOADERS:
        raise KeyError(f"Unknown workload id: {workload_id}")
    return DEFAULT_LOADERS[workload_id]


def get_model_spec(workload_id: str) -> ModelSpec | None:
    """Return the model-catalog entry for a workload when one exists."""

    try:
        return MODEL_CATALOG.get(workload_id)
    except KeyError:
        return None


__all__ = ["DEFAULT_LOADERS", "LoaderEntry", "LoaderFn", "MODEL_CATALOG", "get_loader", "get_model_spec"]
