"""Tests for :mod:`compgen.ir.recipe.llm_view`.

Invariants:
- View is deterministic (same module → same hash).
- Total rows never exceed ``max_ops``.
- Token estimate stays below a tight budget on real recipes.
- ``diff_views`` detects additions/removals.
"""

from __future__ import annotations

import pytest
from compgen.ir.recipe.llm_view import (
    diff_views,
    estimate_tokens,
    recipe_to_llm_view,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region


def _empty_module() -> ModuleOp:
    return ModuleOp(Region([Block()]))


def test_view_of_empty_module() -> None:
    view = recipe_to_llm_view(_empty_module())
    assert view["total_ops"] == 0
    assert view["banner"] == []
    assert view["middle"] == []
    assert view["hash"].startswith("sha256:")


def test_view_is_deterministic() -> None:
    m1 = _empty_module()
    m2 = _empty_module()
    v1 = recipe_to_llm_view(m1)
    v2 = recipe_to_llm_view(m2)
    assert v1["hash"] == v2["hash"]


def test_view_respects_max_ops() -> None:
    """Even with a populated recipe, max_ops caps rows."""
    from pathlib import Path

    from compgen.api import device
    from compgen.ir.recipe.seed import generate_seed_recipe

    exemplar = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
    dev = device(exemplar)

    # Build a tiny Payload module so generate_seed_recipe has something
    # to produce ops over. Seed emits > 10 ops for a non-trivial target.
    payload = _empty_module()
    recipe = generate_seed_recipe(payload, dev.profile, "latency")

    view = recipe_to_llm_view(recipe, max_ops=12)
    total_rows = len(view["banner"]) + sum(1 for r in view["middle"] if "_truncated" not in r)
    assert total_rows <= 12


def test_view_truncation_marker() -> None:
    """When the recipe has more ops than max_ops, a _truncated row appears."""
    from pathlib import Path

    from compgen.api import device
    from compgen.ir.recipe.seed import generate_seed_recipe

    exemplar = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
    dev = device(exemplar)
    recipe = generate_seed_recipe(_empty_module(), dev.profile, "latency")

    # Aggressively low cap so truncation triggers.
    view = recipe_to_llm_view(recipe, max_ops=3)
    if view["total_ops"] > 3:
        last = view["middle"][-1]
        assert "_truncated" in last


def test_token_budget_under_2k_for_small_recipe() -> None:
    """A small recipe's default view should fit well under 2k tokens."""
    from pathlib import Path

    from compgen.api import device
    from compgen.ir.recipe.seed import generate_seed_recipe

    exemplar = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
    dev = device(exemplar)
    recipe = generate_seed_recipe(_empty_module(), dev.profile, "latency")

    view = recipe_to_llm_view(recipe, max_ops=80)
    tokens = estimate_tokens(view)
    assert tokens < 2000, f"view is {tokens} estimated tokens, over budget"


def test_diff_views_detects_added_and_removed() -> None:
    """Synthesise two views by hand-editing banner rows and check the diff."""
    view_a = {
        "hash": "sha256:A",
        "counts": {},
        "total_ops": 2,
        "banner": [
            {"op_id": "op_aaaa", "_op": "recipe.tile"},
            {"op_id": "op_bbbb", "_op": "recipe.fuse"},
        ],
        "middle": [],
    }
    view_b = {
        "hash": "sha256:B",
        "counts": {},
        "total_ops": 2,
        "banner": [
            {"op_id": "op_aaaa", "_op": "recipe.tile"},
            {"op_id": "op_cccc", "_op": "recipe.vectorize"},
        ],
        "middle": [],
    }
    diff = diff_views(view_a, view_b)
    added = {e["op_id"] for e in diff["added"]}
    removed = {e["op_id"] for e in diff["removed"]}
    assert added == {"op_cccc"}
    assert removed == {"op_bbbb"}
    assert diff["unchanged_count"] == 1
    assert diff["hash_before"] == "sha256:A"
    assert diff["hash_after"] == "sha256:B"


def test_diff_views_on_unchanged() -> None:
    view = {
        "hash": "sha256:same",
        "counts": {},
        "total_ops": 1,
        "banner": [{"op_id": "op_x", "_op": "recipe.tile"}],
        "middle": [],
    }
    diff = diff_views(view, view)
    assert diff["added"] == []
    assert diff["removed"] == []
    assert diff["unchanged_count"] == 1


def test_focus_inlines_named_op() -> None:
    """When `focus=op_id` is supplied, the named op appears in `focused`."""
    from pathlib import Path

    from compgen.api import device
    from compgen.ir.recipe.seed import generate_seed_recipe

    exemplar = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
    dev = device(exemplar)
    recipe = generate_seed_recipe(_empty_module(), dev.profile, "latency")
    view = recipe_to_llm_view(recipe, max_ops=80)

    all_rows = view["banner"] + [r for r in view["middle"] if "op_id" in r]
    if not all_rows:
        pytest.skip("seed recipe has no ops")
    pick = all_rows[0]["op_id"]
    focused_view = recipe_to_llm_view(recipe, max_ops=80, focus=pick)
    assert "focused" in focused_view
    assert pick in focused_view["focused"]
