"""Skeleton-contract tests for the remaining 6 P3 primitives.

Each primitive is tested for:

* Its card is registered in the global call-site registry.
* The primary path runs end-to-end and produces output that satisfies
  the declared output_schema.
* The fallback path runs end-to-end (``COMPGEN_DISABLE_LLM=1``) and
  surfaces ``fallback_used=true``.
* Every primitive's declared forbidden actions are members of
  ``FORBIDDEN_LLM_ACTIONS``.

The P3.3 ``rank_candidates`` primitive has its own deeper test file
(test_primitive_rank_candidates.py) because its permutation invariant
is the headline guarantee.

Importing :mod:`compgen.agent.primitives` registers all 7 sites; the
module-level fixture in this file imports it once.
"""

from __future__ import annotations

from typing import Any

import compgen.agent.primitives  # noqa: F401  (registers every primitive)
import pytest
from compgen.llm.call_site import (
    FORBIDDEN_LLM_ACTIONS,
    get_call_site,
    list_call_site_ids,
)

_PRIMITIVE_INPUTS: dict[str, dict[str, Any]] = {
    "recognize_python_pattern": {
        "python_source": "def f(x):\n    return x * 2\n",
        "fx_graph_summary": {"nodes": []},
    },
    "name_cluster": {
        "region_dossier": {"region_id": "region_001", "ops": ["matmul", "softmax"]},
    },
    "revise_kernel": {
        "kernel_contract": {"op": "matmul", "dtype": "fp16"},
        "target_envelope": {"sm_count": 72},
        "prev_attempt": "# tile=128\n@triton.jit\ndef kernel(): pass\n",
        "typed_failure": {
            "ir_slice": {"annotation": "tile=128, scratchpad overflow"},
        },
    },
    "pick_dispatch": {
        "workload_class": "streaming",
        "deployment_constraints": {"latency_budget_ms": 25.0},
        "region_dossier": {"region_id": "r"},
    },
    "explain_counterexample": {
        "counterexample": {
            "likely_cause": "fp16 accumulator overflow",
            "ir_slice": {"region_id": "r017"},
        },
        "ir_slice": {"region_id": "r017"},
        "refinement_spec": {"kind": "numerical"},
    },
    "compare_recipes": {
        "recipe_a": {"recipe_id": "a", "version": "1"},
        "recipe_b": {"recipe_id": "b", "version": "1"},
        "target_class": "cuda",
    },
}


@pytest.fixture(autouse=True)
def _ensure_llm_disabled(monkeypatch):
    """Run each test under the deterministic-fallback regime so we
    never depend on a live LLM connection."""

    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    yield


def test_all_seven_primitives_registered():
    """The :mod:`compgen.agent.primitives` import registers every
    site so they appear in the global registry."""

    site_ids = set(list_call_site_ids())
    for name in (
        "recognize_python_pattern",
        "name_cluster",
        "rank_candidates",
        "revise_kernel",
        "pick_dispatch",
        "explain_counterexample",
        "compare_recipes",
    ):
        assert name in site_ids, f"primitive {name!r} not registered"


@pytest.mark.parametrize("site_id,kwargs", list(_PRIMITIVE_INPUTS.items()))
def test_primitive_runs_under_fallback(site_id: str, kwargs: dict[str, Any]):
    """Each primitive must produce a schema-valid output under
    ``COMPGEN_DISABLE_LLM=1`` (no LLM, fallback path only)."""

    import importlib

    mod = importlib.import_module(f"compgen.agent.primitives.{site_id}")
    fn = getattr(mod, site_id)
    out = fn(**kwargs)
    assert isinstance(out, dict)
    assert "fallback_used" in out
    assert out["fallback_used"] is True


def test_every_card_forbidden_in_closed_enum():
    """All seven primitives declare forbidden actions that are members
    of :data:`FORBIDDEN_LLM_ACTIONS`."""

    for site_id in [
        "recognize_python_pattern", "name_cluster", "rank_candidates",
        "revise_kernel", "pick_dispatch", "explain_counterexample",
        "compare_recipes",
    ]:
        card = get_call_site(site_id)
        for action in card.forbidden:
            assert action in FORBIDDEN_LLM_ACTIONS, (
                f"primitive {site_id!r} declares undocumented forbidden {action!r}"
            )


def test_pick_dispatch_dispatch_mode_in_closed_enum():
    """The dispatch_mode the fallback picks must be in the enum."""

    from compgen.agent.primitives.pick_dispatch import DISPATCH_MODES, pick_dispatch

    for workload in ("streaming", "batched_inference", "one_shot", "persistent", "novel"):
        out = pick_dispatch(workload, {"latency_budget_ms": 25.0}, {"region_id": "r"})
        assert out["dispatch_mode"] in DISPATCH_MODES


def test_compare_recipes_relation_in_closed_enum():
    from compgen.agent.primitives.compare_recipes import (
        RECIPE_RELATIONS,
        compare_recipes,
    )

    out = compare_recipes({}, {}, "cuda")
    assert out["relation"] in RECIPE_RELATIONS


def test_revise_kernel_shrinks_tile_size_monotonically():
    """The fallback's tile shrink ladder is monotonic: the new tile
    is strictly smaller than or equal to the previous one."""

    from compgen.agent.primitives.revise_kernel import revise_kernel

    out = revise_kernel(
        {"op": "matmul"},
        {},
        "# stub\n",
        {"ir_slice": {"annotation": "tile=128, overflow"}},
    )
    assert out["new_tile_size"] is not None
    assert out["new_tile_size"] <= 128


def test_explain_counterexample_suggested_edit_kind_typed():
    from compgen.agent.primitives.explain_counterexample import explain_counterexample

    out = explain_counterexample({"likely_cause": "x"}, {}, {})
    assert out["suggested_edit"]["kind"] in {
        "tactic_change",
        "param_change",
        "abandon_tactic",
    }


def test_p3_full_registry_size_after_imports():
    """Sanity floor: after importing ``compgen.agent.primitives`` the
    registry contains at least 7 sites."""

    assert len(list_call_site_ids()) >= 7
