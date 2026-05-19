"""Unit tests for the SmolVLA subset selector.

Drives :class:`SubsetSelector` against a hand-built minimal ``nn.Module``
that mimics SmolVLA's relevant naming: action-expert layers + an action
head. No model download, no LLM, no torch.compile.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xpu_rt.benchmarks.smolvla_subset import (
    DEFAULT_SEQ_LEN,
    SubsetSelector,
    _action_expert_layer_index_for_path,
    _component_for_path,
    _COMPONENT_SUBSTRINGS,
    load_contracts,
)
from xpu_rt.promotion.region_signature import hash_region_signature, make_region_signature


# ---------------------------------------------------------------------------
# Tiny stand-in module — mirrors SmolVLA's naming exactly
# ---------------------------------------------------------------------------


def _build_fake_smolvla() -> nn.Module:
    """Module tree with two action-expert layers + an action head.

    Names match SmolVLA's real paths so the substring filter exercises
    the production code paths verbatim.
    """

    class GemmaMLP(nn.Module):
        def __init__(self, d: int, ff: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(d, ff, bias=False)
            self.up_proj = nn.Linear(d, ff, bias=False)
            self.down_proj = nn.Linear(ff, d, bias=False)

    class GemmaAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

    class GemmaLayer(nn.Module):
        def __init__(self, d: int, ff: int) -> None:
            super().__init__()
            self.self_attn = GemmaAttn(d)
            self.mlp = GemmaMLP(d, ff)

    class LMExpert(nn.Module):
        def __init__(self, n_layers: int, d: int, ff: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [GemmaLayer(d, ff) for _ in range(n_layers)]
            )

    class VLMWithExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_expert = LMExpert(n_layers=5, d=768, ff=3072)

    class FakeSmolVLA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm_with_expert = VLMWithExpert()
            self.action_in_proj = nn.Linear(32, 720)
            self.action_out_proj = nn.Linear(720, 32)

    return FakeSmolVLA()


# ---------------------------------------------------------------------------
# Path-resolution helpers
# ---------------------------------------------------------------------------


def test_component_substrings_match_real_smolvla_paths() -> None:
    """Catches a regression if someone renames a substring."""
    paths_for_expert = [
        "vlm_with_expert.lm_expert.layers.0.mlp.gate_proj",
        "vlm_with_expert.lm_expert.layers.5.self_attn.q_proj",
    ]
    paths_for_head = [
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
    ]
    for p in paths_for_expert:
        assert "action_expert" in _component_for_path(p)
    for p in paths_for_head:
        assert "action_head" in _component_for_path(p)


def test_action_expert_layer_index_parses() -> None:
    assert _action_expert_layer_index_for_path("vlm_with_expert.lm_expert.layers.3.mlp.gate_proj") == 3
    assert _action_expert_layer_index_for_path("vlm_with_expert.lm_expert.layers.12.self_attn.o_proj") == 12
    assert _action_expert_layer_index_for_path("action_in_proj") is None


# ---------------------------------------------------------------------------
# End-to-end enumeration on the fake module
# ---------------------------------------------------------------------------


def test_enumerate_dedups_identical_shapes() -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(action_expert_layers=None, components=("action_expert",))
    unique, report = sel.enumerate_unique_contracts(model)
    assert report.total_linears == 5 * 7 + 2  # 7 linears per gemma layer + 2 head linears
    assert report.total_passing_filter == 5 * 7  # only the lm_expert layers pass
    assert report.unique_after_dedup < report.total_passing_filter
    # 7 linears per layer collapse to 3 shapes:
    #   QKVO (d×d, 4 occurrences per layer)
    #   gate/up (d×ff, 2 occ per layer)
    #   down (ff×d, 1 occ per layer)
    assert report.unique_after_dedup == 3
    by_shape = sorted(
        (
            tuple(u.contract.input_shapes[1]),  # the (K, N) of weight
            u.occurrences,
        )
        for u in unique
    )
    # Expect (768,768)x20, (768,3072)x10, (3072,768)x5 with 5 layers.
    assert by_shape == [((768, 768), 20), ((768, 3072), 10), ((3072, 768), 5)]


def test_enumerate_layer_filter_clips_to_first_n() -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(
        action_expert_layers=(0, 1),
        components=("action_expert",),
    )
    unique, report = sel.enumerate_unique_contracts(model)
    assert report.total_passing_filter == 2 * 7
    # Each shape now has half the occurrences.
    occ = {tuple(u.contract.input_shapes[1]): u.occurrences for u in unique}
    assert occ == {(768, 768): 8, (768, 3072): 4, (3072, 768): 2}


def test_enumerate_picks_up_action_head() -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(
        action_expert_layers=(0,),
        components=("action_expert", "action_head"),
    )
    unique, report = sel.enumerate_unique_contracts(model)
    head = [u for u in unique if "action_head" in u.components]
    assert len(head) == 2
    expert = [u for u in unique if "action_expert" in u.components]
    assert len(expert) == 3


def test_enumerate_quantize_to_i8_stamps_correct_dtypes() -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(action_expert_layers=None, components=("action_expert",), quantize_to_i8=True)
    unique, _ = sel.enumerate_unique_contracts(model)
    for u in unique:
        assert u.contract.dtypes == ("i8", "i8", "i32")


def test_enumerate_no_quant_uses_module_dtype() -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(action_expert_layers=None, components=("action_expert",), quantize_to_i8=False)
    unique, _ = sel.enumerate_unique_contracts(model)
    # Default torch dtype is fp32 unless changed; the module's nn.Linear weights default to fp32.
    for u in unique:
        assert u.contract.dtypes[0] in ("fp32", "fp16", "bf16")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(action_expert_layers=(0,), components=("action_expert",))
    unique, report = sel.enumerate_unique_contracts(model)
    selected = sel.select_subset(unique, limit=2)
    manifest_path = sel.save(selected, report, out_dir=tmp_path)
    assert manifest_path.is_file()
    body = json.loads(manifest_path.read_text())
    assert body["report"]["selected"] == 2
    assert len(body["contracts"]) == 2

    reloaded = load_contracts(manifest_path)
    assert len(reloaded) == 2
    # Region signature is stable across save/load.
    for c, u in zip(reloaded, selected):
        sig = make_region_signature(
            op_family=c.op_family,
            dtype=c.dtypes[0],
            layout=c.layout,
            dims=[
                int(d) for s in (*c.input_shapes, *c.output_shapes) for d in s
            ],
            target_class=sel.target_class,
        )
        assert hash_region_signature(sig) == u.region_sig_hash


def test_seq_len_stamped_on_M(tmp_path: Path) -> None:
    model = _build_fake_smolvla()
    sel = SubsetSelector(
        action_expert_layers=(0,),
        components=("action_expert",),
        seq_len=128,
    )
    unique, _ = sel.enumerate_unique_contracts(model)
    for u in unique:
        assert u.contract.input_shapes[0][0] == 128
