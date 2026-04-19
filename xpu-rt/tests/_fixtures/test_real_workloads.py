"""Tests for the real-workload fixtures themselves.

These are smoke tests — the fixtures are the *test data* for Wave 2+
passes, so we only need to confirm they build, run on CPU, and
expose the contract each pass relies on (model, inputs,
eager_output, exported).
"""

from __future__ import annotations

import pytest
import torch

from tests._fixtures.real_workloads import (
    ALL_FIXTURE_FNS,
    RealWorkloadFixture,
    gemma_decode_tiny,
    qwen_moe_tiny,
    smolvla_tiny,
)


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_fixture_builds_and_runs(fn):
    fx = fn()
    assert isinstance(fx, RealWorkloadFixture)
    assert fx.model is not None
    assert len(fx.example_inputs) >= 1


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_eager_output_is_finite(fn):
    fx = fn()
    assert fx.eager_output.isfinite().all()


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_eager_output_is_deterministic(fn):
    a = fn()
    b = fn()
    assert torch.allclose(a.eager_output, b.eager_output, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_exported_program_is_valid(fn):
    fx = fn()
    # ExportedProgram has a ``.graph`` attribute and callable.
    assert hasattr(fx.exported, "graph")
    # Run the exported module against its example inputs and compare.
    out = fx.exported.module()(*fx.example_inputs)
    assert torch.allclose(out, fx.eager_output, rtol=1e-5, atol=1e-6)


def test_smolvla_output_shape():
    fx = smolvla_tiny()
    assert tuple(fx.eager_output.shape) == (1, 8, 128)


def test_gemma_decode_has_rope_inputs():
    fx = gemma_decode_tiny()
    # inputs: (x, cos, sin)
    assert len(fx.example_inputs) == 3
    cos = fx.example_inputs[1]
    sin = fx.example_inputs[2]
    assert cos.shape == sin.shape
    assert cos.ndim == 4  # [B, n_heads, T, half_head_dim]


def test_qwen_moe_has_two_experts():
    fx = qwen_moe_tiny()
    # router projects to n_experts=2.
    assert fx.model.n_experts == 2
    assert len(fx.model.experts) == 2


def test_fixture_names_are_unique():
    names = [fn().name for fn in ALL_FIXTURE_FNS]
    assert len(set(names)) == len(names)
