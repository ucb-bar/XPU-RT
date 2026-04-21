"""Robotics and embodied-model adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from compgen.models.core import CaptureMode, ModelSource, ModelSpec, ReadinessLevel

if TYPE_CHECKING:
    from benchmarks.spec import WorkspaceConfig


def _iter_workspace_roots(workspace: WorkspaceConfig | None, keys: tuple[str, ...]) -> list[Path]:
    if workspace is None:
        return []
    roots: list[Path] = []
    for key in keys:
        if key in workspace.external_roots:
            roots.append(workspace.external_roots[key])
    return roots


def _resolve_existing_root(
    workspace: WorkspaceConfig | None, *, keys: tuple[str, ...], defaults: tuple[str, ...]
) -> Path:
    candidates = _iter_workspace_roots(workspace, keys)
    candidates.extend(Path(path) for path in defaults)
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded
    raise FileNotFoundError(f"Unable to resolve external repo for keys={keys}")


def _ensure_sys_path(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def _hf_cache_repo_dir(repo_id: str) -> Path:
    org, name = repo_id.split("/", 1)
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"


def load_smolvla_bundle(
    workspace: WorkspaceConfig | None = None,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[torch.Tensor, ...], int]:
    """Load SmolVLA and return the wrapped one-step module plus flattened inputs."""

    root = _resolve_existing_root(
        workspace,
        keys=("understanding_pi0", "Understanding-PI0"),
        defaults=(
            "/scratch2/agustin/merlin/third_party/Understanding-PI0",
            "/scratch2/agustin/merlin/third_party/understanding_pi0",
        ),
    )
    _ensure_sys_path(root)

    # ``understanding_pi0`` imports ``lerobot``; if it isn't installed
    # in the active venv, resolve the sibling source checkout and add
    # its ``src/`` to sys.path so the import works.
    try:
        import lerobot  # noqa: F401
    except ImportError:
        try:
            lerobot_root = _resolve_existing_root(
                workspace,
                keys=("lerobot",),
                defaults=(
                    "/scratch2/agustin/merlin/third_party/lerobot",
                    "/scratch2/agustin/experimental/Understanding-PI0/lerobot",
                ),
            )
            lerobot_src = lerobot_root / "src"
            _ensure_sys_path(lerobot_src if lerobot_src.exists() else lerobot_root)
        except FileNotFoundError:
            pass  # falls through; the next import will raise the original error

    # Bypass lerobot.policies.__init__ which imports the broken GR00T dataclass.
    # We directly import the smolvla submodule without triggering the full
    # policy registry.
    import sys

    # Pre-register stub modules to prevent lerobot.policies.__init__ from
    # importing the broken GR00T package at all.
    _stubs_needed = [
        "lerobot.policies.groot",
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.groot_n1",
    ]
    import types as _types

    for _mod_name in _stubs_needed:
        if _mod_name not in sys.modules:
            _stub = _types.ModuleType(_mod_name)
            _stub.__path__ = []  # type: ignore[attr-defined]
            # Add dummy attributes that __init__.py expects
            if _mod_name.endswith(".configuration_groot"):
                _stub.GrootConfig = type("GrootConfig", (), {})  # type: ignore[attr-defined]
            sys.modules[_mod_name] = _stub

    from understanding_pi0.smolvla_mx.loader import build_dummy_processed_inputs, load_smolvla_policy
    from understanding_pi0.smolvla_mx.wrappers import SmolVLAOneStepNoCacheWrapper, flatten_processed_inputs

    policy = load_smolvla_policy(model_id="lerobot/smolvla_base", device=device)
    processed = build_dummy_processed_inputs(policy, batch_size=1, prompt_len=8, device=device)
    flat_inputs = flatten_processed_inputs(processed)
    num_cams = (len(flat_inputs) - 5) // 2
    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=num_cams).eval()
    return wrapper, flat_inputs, num_cams


def load_smolvla(
    workspace: WorkspaceConfig | None = None,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[Any, ...]]:
    """Load the SmolVLA one-step wrapper and example inputs."""

    wrapper, flat_inputs, _ = load_smolvla_bundle(workspace, device=device)
    return wrapper, flat_inputs


def load_smolvla_quantized_bundle(
    workspace: WorkspaceConfig | None = None,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[torch.Tensor, ...], int]:
    """Load SmolVLA with FP8 E4M3 po2 quantization for NPU deployment.

    Applies the NPU quantization recipe (all matmuls FP8, vector ops BF16,
    softmax BF16) and rewrites modules for export compatibility.

    Returns:
        (quantized_wrapper, flat_inputs, num_cams)
    """
    wrapper, flat_inputs, num_cams = load_smolvla_bundle(workspace, device=device)

    from compgen.quantization.export_wrappers import rewrite_for_export
    from compgen.quantization.smolvla_recipe import apply_smolvla_quantization, default_npu_recipe

    recipe = default_npu_recipe()
    apply_smolvla_quantization(wrapper, recipe)
    rewrite_for_export(wrapper)

    return wrapper, flat_inputs, num_cams


def load_smolvla_quantized(
    workspace: WorkspaceConfig | None = None,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, tuple[Any, ...]]:
    """Load the quantized SmolVLA wrapper and example inputs."""

    wrapper, flat_inputs, _ = load_smolvla_quantized_bundle(workspace, device=device)
    return wrapper, flat_inputs


def _load_groot(_workspace: WorkspaceConfig | None = None) -> tuple[nn.Module, tuple[Any, ...]]:
    model_id = "nvidia/GR00T-N1.6-3B"
    cache_dir = _hf_cache_repo_dir(model_id)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cached GR00T checkpoint: {cache_dir}")
    raise RuntimeError(
        "GR00T workload is registered for probe coverage, but a tensor-input wrapper "
        "for TorchDynamo capture has not been implemented yet."
    )


def _load_cosmos(_workspace: WorkspaceConfig | None = None) -> tuple[nn.Module, tuple[Any, ...]]:
    raise RuntimeError(
        "Cosmos workloads are registered for probe coverage, but no runnable PyTorch "
        "wrapper is available in this repository yet."
    )


def get_graph_op_summary(graphs: list[torch.fx.GraphModule]) -> dict[str, int]:
    """Count call targets across captured graph partitions."""

    targets: dict[str, int] = {}
    for gm in graphs:
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = str(node.target)
            targets[target] = targets.get(target, 0) + 1
    return dict(sorted(targets.items(), key=lambda item: (-item[1], item[0])))


def build_robotics_model_specs() -> list[ModelSpec]:
    """Return heavyweight robotics model specs."""

    return [
        ModelSpec(
            model_id="smolvla_one_step",
            family="robotics_vla",
            description="SmolVLA one-step wrapper captured via TorchDynamo partitioning",
            loader=load_smolvla,
            source=ModelSource(
                kind="external_repo",
                identifier="Understanding-PI0",
                repo_name="understanding_pi0",
                notes="Uses Understanding-PI0 + LeRobot smolvla wrapper",
            ),
            source_model_id="lerobot/smolvla_base",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.ANALYSIS_ONLY,
            expected_status="pass",
            tags=("frontier", "robotics", "smolvla"),
            requirements=("understanding_pi0", "lerobot[smolvla]"),
        ),
        ModelSpec(
            model_id="smolvla_fp8_npu",
            family="robotics_vla",
            description="SmolVLA quantized to FP8 E4M3 (po2 scaling) for NPU deployment",
            loader=load_smolvla_quantized,
            source=ModelSource(
                kind="external_repo",
                identifier="Understanding-PI0",
                repo_name="understanding_pi0",
                notes="Uses Understanding-PI0 + LeRobot smolvla wrapper, FP8 E4M3 po2 quantization",
            ),
            source_model_id="lerobot/smolvla_base",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.ANALYSIS_ONLY,
            expected_status="pass",
            tags=("frontier", "robotics", "smolvla", "fp8", "npu"),
            requirements=("understanding_pi0", "lerobot[smolvla]", "torchao>=0.16"),
        ),
        ModelSpec(
            model_id="groot_policy_step",
            family="robotics_vla",
            description="GR00T policy probe for NVIDIA robotics integration coverage",
            loader=_load_groot,
            source=ModelSource(
                kind="external_repo",
                identifier="lerobot",
                repo_name="lerobot",
                notes="Relies on LeRobot's GR00T integration and cached NVIDIA checkpoints",
            ),
            source_model_id="nvidia/GR00T-N1.6-3B",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.PROBE_ONLY,
            expected_status="xfail",
            tags=("frontier", "robotics", "groot"),
            requirements=("lerobot[groot]",),
        ),
        ModelSpec(
            model_id="cosmos_reason2",
            family="world_model",
            description="Cosmos Reason 2 world-model probe placeholder",
            loader=_load_cosmos,
            source=ModelSource(kind="external_hub", identifier="NVIDIA Cosmos Reason 2"),
            source_model_id="nvidia/Cosmos-Reason2",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.PROBE_ONLY,
            expected_status="xfail",
            tags=("frontier", "robotics", "cosmos"),
        ),
        ModelSpec(
            model_id="cosmos_predict2_5",
            family="world_model",
            description="Cosmos Predict 2.5 world-model probe placeholder",
            loader=_load_cosmos,
            source=ModelSource(kind="external_hub", identifier="NVIDIA Cosmos Predict 2.5"),
            source_model_id="nvidia/Cosmos-Predict2.5",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.PROBE_ONLY,
            expected_status="xfail",
            tags=("frontier", "robotics", "cosmos"),
        ),
        ModelSpec(
            model_id="cosmos_transfer2_5",
            family="world_model",
            description="Cosmos Transfer 2.5 world-model probe placeholder",
            loader=_load_cosmos,
            source=ModelSource(kind="external_hub", identifier="NVIDIA Cosmos Transfer 2.5"),
            source_model_id="nvidia/Cosmos-Transfer2.5",
            capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
            readiness=ReadinessLevel.PROBE_ONLY,
            expected_status="xfail",
            tags=("frontier", "robotics", "cosmos"),
        ),
    ]
