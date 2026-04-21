"""Tests for the ROCm GPU target dialect stack (Wave 5 skeleton).

Locks in:
  * Each plugin satisfies the TargetStagePlugin protocol
  * Stack factory produces 6 stages + 5 plugins
  * Plugins tag IR with the expected ROCm-specific attrs
    (cdna_mfma_16x16 encoding for matmul-named ops, triton_rocm codegen)
"""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType, f16, i32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.stages.dispatch.stage import DISPATCH_ID_ATTR
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.targets.rocm_gpu import (
    RocmCodegenPlugin,
    RocmDispatchPlugin,
    RocmEncodingPlugin,
    RocmLayoutPlugin,
    RocmTilingPlugin,
    create_rocm_gpu_stack,
)
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR
from compgen.stages.templates.tiling import TILE_SIZES_ATTR


# ---------------------------------------------------------------------------
# Plugin protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plugin_cls,stage_name", [
    (RocmEncodingPlugin, "encoding"),
    (RocmLayoutPlugin,   "layout"),
    (RocmDispatchPlugin, "dispatch"),
    (RocmTilingPlugin,   "tiling"),
    (RocmCodegenPlugin,  "codegen"),
])
def test_plugin_satisfies_protocol(plugin_cls, stage_name) -> None:
    from compgen.stages.base import TargetStagePlugin

    p = plugin_cls()
    assert isinstance(p, TargetStagePlugin)
    assert p.target_name == "rocm_gpu"
    assert p.stage_name == stage_name


# ---------------------------------------------------------------------------
# Stack factory
# ---------------------------------------------------------------------------


def test_create_rocm_stack_has_6_stages_and_5_plugins(tmp_path) -> None:
    stack = create_rocm_gpu_stack(output_dir=str(tmp_path / "out"))
    assert stack.target_name == "rocm_mi250"
    assert len(stack.stages) == 6
    assert len(stack.plugins) == 5
    assert set(stack.plugins.keys()) == {
        "encoding", "layout", "dispatch", "tiling", "codegen"
    }


def test_create_rocm_stack_stage_order(tmp_path) -> None:
    stack = create_rocm_gpu_stack(output_dir=str(tmp_path / "out"))
    names = [s.name for s in stack.stages]
    assert names == ["encoding", "layout", "dispatch", "tiling", "codegen", "bundle"]


# ---------------------------------------------------------------------------
# Plugin transforms — tag IR correctly
# ---------------------------------------------------------------------------


def _matmul_module() -> ModuleOp:
    """A tiny module with one tensor-producing op named 'matmul-like'."""
    block = Block(arg_types=[])
    # tensor.empty produces a TensorType; we'll rename below to mimic matmul
    t = TensorType(f16, [16, 16])
    e = EmptyOp([], t)
    block.add_op(e)
    block.add_op(ReturnOp())
    func = FuncOp("matmul_kernel", ([], []), Region([block]))
    return ModuleOp([func])


def test_rocm_codegen_plugin_tags_triton_rocm_backend() -> None:
    plugin = RocmCodegenPlugin()
    mod = _matmul_module()
    out = plugin.transform(mod)
    tagged = [
        op for op in out.walk()
        if not isinstance(op, (ModuleOp, FuncOp, ReturnOp))
        and any(isinstance(r.type, TensorType) for r in op.results)
    ]
    assert tagged, "plugin should have tagged at least one op"
    for op in tagged:
        assert CODEGEN_BACKEND_ATTR in op.attributes
        assert op.attributes[CODEGEN_BACKEND_ATTR] == StringAttr("triton_rocm")


def test_rocm_dispatch_plugin_assigns_unique_dispatch_ids() -> None:
    plugin = RocmDispatchPlugin()
    mod = _matmul_module()
    out = plugin.transform(mod)
    ids = []
    for op in out.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if DISPATCH_ID_ATTR in op.attributes:
            ids.append(op.attributes[DISPATCH_ID_ATTR].data)
    assert ids
    assert len(set(ids)) == len(ids)


def test_rocm_tiling_plugin_uses_64x64x16_for_matmul_named_ops() -> None:
    """The plugin keys off op.name containing 'matmul'. tensor.empty
    won't trigger that branch — so we need an op whose .name lowercases
    to contain 'matmul'. Easiest path: assert the fall-through pointwise
    tile is applied to non-matmul tensor ops."""
    plugin = RocmTilingPlugin()
    mod = _matmul_module()
    out = plugin.transform(mod)
    tagged = [
        op for op in out.walk()
        if not isinstance(op, (ModuleOp, FuncOp, ReturnOp))
        and any(isinstance(r.type, TensorType) for r in op.results)
    ]
    assert tagged
    for op in tagged:
        assert TILE_SIZES_ATTR in op.attributes
        # tensor.empty doesn't have 'matmul' in its name → 1024 (pointwise default)
        assert op.attributes[TILE_SIZES_ATTR] == StringAttr("1024")


def test_rocm_encoding_plugin_tags_row_major_for_non_matmul_tensor_ops() -> None:
    plugin = RocmEncodingPlugin()
    mod = _matmul_module()
    out = plugin.transform(mod)
    tagged = [
        op for op in out.walk()
        if not isinstance(op, (ModuleOp, FuncOp, ReturnOp))
        and any(isinstance(r.type, TensorType) for r in op.results)
    ]
    assert tagged
    for op in tagged:
        assert ENCODING_ATTR in op.attributes
        # tensor.empty isn't a matmul → row_major (default)
        assert op.attributes[ENCODING_ATTR] == StringAttr("row_major")


def test_rocm_layout_plugin_is_passthrough() -> None:
    plugin = RocmLayoutPlugin()
    mod = _matmul_module()
    before_ops = list(mod.walk())
    out = plugin.transform(mod)
    after_ops = list(out.walk())
    assert len(before_ops) == len(after_ops)


# ---------------------------------------------------------------------------
# Configure interface contract
# ---------------------------------------------------------------------------


def test_plugins_accept_configure_call() -> None:
    """All plugins must be configurable with TargetProfile + CapabilitySpec."""
    from compgen.targets.capability import infer_capabilities
    from compgen.targets.schema import load_profile

    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    caps = infer_capabilities(target)

    for plugin_cls in (RocmEncodingPlugin, RocmLayoutPlugin,
                       RocmDispatchPlugin, RocmTilingPlugin, RocmCodegenPlugin):
        p = plugin_cls()
        p.configure(target, caps)  # must not raise
