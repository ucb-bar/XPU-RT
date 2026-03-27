"""Tests for Recipe IR dialect registration.

Verifies dialect name, op/attr counts, and that all expected operation
names are present.
"""

from __future__ import annotations

from compgen.ir.recipe.dialect import ALL_ATTRS, ALL_OPS, Recipe


def test_dialect_name() -> None:
    """Dialect name is 'recipe'."""
    assert Recipe.name == "recipe"


def test_dialect_op_count() -> None:
    """Dialect contains exactly 46 operations (44 original + 2 Exo ops)."""
    assert len(ALL_OPS) == 46


def test_dialect_attr_count() -> None:
    """Dialect contains exactly 5 custom attributes."""
    assert len(ALL_ATTRS) == 5


def test_all_scope_op_names_present() -> None:
    """Family A (scope) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.region",
        "recipe.segment",
        "recipe.anchor",
        "recipe.bind_payload",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_fact_op_names_present() -> None:
    """Family B (fact) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.fact.backend_available",
        "recipe.fact.kernel_contract",
        "recipe.fact.transfer_cost",
        "recipe.fact.local_mem_fit",
        "recipe.fact.fusible_with",
        "recipe.fact.calibration",
        "recipe.fact.export_issue",
        "recipe.fact.graph_break",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_candidate_op_names_present() -> None:
    """Family C (candidate) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.tile",
        "recipe.fuse",
        "recipe.vectorize",
        "recipe.reassociate",
        "recipe.layout_normalize",
        "recipe.lower_to_accel",
        "recipe.request_triton_kernel",
        "recipe.materialize_ukernel",
        "recipe.place_on_device",
        "recipe.insert_copy_boundary",
        "recipe.segment_boundary",
        "recipe.blackbox",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_choice_op_names_present() -> None:
    """Family D (choice) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.alternatives",
        "recipe.rank",
        "recipe.search_budget",
        "recipe.require_eqsat",
        "recipe.require_solver",
        "recipe.defer_choice",
        "recipe.promote_candidate",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_verify_op_names_present() -> None:
    """Family E (verify) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.require_diff_test",
        "recipe.require_translation_validation",
        "recipe.require_layout_invariant",
        "recipe.require_memory_bound",
        "recipe.require_check_file",
        "recipe.require_profile_budget",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_provenance_op_names_present() -> None:
    """Family F (provenance) ops are in the dialect."""
    names = {op.name for op in ALL_OPS}
    for expected in (
        "recipe.from_agent",
        "recipe.from_eqsat",
        "recipe.from_template",
        "recipe.feedback",
        "recipe.reject",
        "recipe.promote",
        "recipe.lineage",
    ):
        assert expected in names, f"{expected} missing from ALL_OPS"


def test_all_attr_names_present() -> None:
    """All 5 custom attributes are present."""
    names = {attr.name for attr in ALL_ATTRS}
    for expected in (
        "recipe.shape_summary",
        "recipe.effect_class",
        "recipe.cost",
        "recipe.provenance",
        "recipe.device_ref",
    ):
        assert expected in names, f"{expected} missing from ALL_ATTRS"


def test_dialect_ops_match_all_ops() -> None:
    """Dialect._operations matches ALL_OPS list."""
    dialect_op_names = {op.name for op in Recipe._operations}
    all_op_names = {op.name for op in ALL_OPS}
    assert dialect_op_names == all_op_names


def test_dialect_attrs_match_all_attrs() -> None:
    """Dialect._attributes matches ALL_ATTRS list."""
    dialect_attr_names = {attr.name for attr in Recipe._attributes}
    all_attr_names = {attr.name for attr in ALL_ATTRS}
    assert dialect_attr_names == all_attr_names
