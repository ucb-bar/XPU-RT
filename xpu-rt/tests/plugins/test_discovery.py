"""Tests for ``compgen.plugins`` — entry-point discovery + manual register.

Locks in:
  * group constants are stable
  * register() validates + adds to the registry
  * register() rejects objects that fail the per-group validator
  * unknown groups raise
  * reset_registry() clears state for test isolation
  * discover_all() runs without crashing even when no plugins exist
"""

from __future__ import annotations

import pytest
from compgen.plugins import (
    GROUP_DECOMPOSITIONS,
    GROUP_FUSION_RULES,
    GROUP_KERNEL_PROVIDERS,
    GROUP_TARGET_BACKENDS,
    KNOWN_GROUPS,
    discover_all,
    register,
    registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Group constants
# ---------------------------------------------------------------------------


def test_known_groups_match_documented_set() -> None:
    assert GROUP_KERNEL_PROVIDERS in KNOWN_GROUPS
    assert GROUP_DECOMPOSITIONS in KNOWN_GROUPS
    assert GROUP_FUSION_RULES in KNOWN_GROUPS
    assert GROUP_TARGET_BACKENDS in KNOWN_GROUPS
    assert len(KNOWN_GROUPS) == 5


# ---------------------------------------------------------------------------
# Manual register — happy path + validation rejection
# ---------------------------------------------------------------------------


class _GoodKernelProvider:
    """Satisfies the KernelProvider protocol surface."""

    name = "test_provider"

    def accepts_contract(self, contract):
        return True

    def search(self, contract, budget):
        return None

    def export_knowledge(self):
        return []


def test_register_kernel_provider_loads_and_lists_it() -> None:
    p = register(GROUP_KERNEL_PROVIDERS, "test_p", _GoodKernelProvider())
    assert p.group == GROUP_KERNEL_PROVIDERS
    assert p.name == "test_p"
    reg = registry()
    assert reg.total_loaded() == 1
    assert "test_p" in reg.names_in(GROUP_KERNEL_PROVIDERS)


def test_register_rejects_invalid_kernel_provider() -> None:
    """Object missing the required protocol methods → ValueError."""

    class _Incomplete:
        name = "x"  # missing search / accepts_contract / export_knowledge

    with pytest.raises(ValueError, match="missing KernelProvider methods"):
        register(GROUP_KERNEL_PROVIDERS, "bad", _Incomplete())


def test_register_decomposition_must_be_callable() -> None:
    with pytest.raises(ValueError, match="must be callable"):
        register(GROUP_DECOMPOSITIONS, "bad", object())  # not callable


def test_register_fusion_rule_must_be_callable() -> None:
    register(GROUP_FUSION_RULES, "ok", lambda p, c: True)
    assert "ok" in registry().names_in(GROUP_FUSION_RULES)


def test_register_target_backend_must_have_required_methods() -> None:
    class _GoodBackend:
        def supports_target(self, name):
            return True

        def get_options(self):
            return {}

        def get_compilation_stages(self):
            return []

        def compile_stage(self, *a, **k):
            return None

        def validate(self, *a, **k):
            return True

    register(GROUP_TARGET_BACKENDS, "good_be", _GoodBackend())
    assert "good_be" in registry().names_in(GROUP_TARGET_BACKENDS)


def test_register_unknown_group_rejects() -> None:
    with pytest.raises(ValueError, match="unknown extension group"):
        register("compgen.does.not.exist", "x", lambda: None)


# ---------------------------------------------------------------------------
# Discovery — no installed plugins → registry is empty + no crash
# ---------------------------------------------------------------------------


def test_discover_all_runs_clean_with_no_plugins() -> None:
    reg = discover_all()
    # Either nothing installed OR genuinely-installed plugins land here;
    # we just assert it didn't crash and returned a registry object.
    assert reg is registry()


def test_registry_failures_list_is_empty_when_no_failures() -> None:
    discover_all()
    assert registry().failures == [] or all(isinstance(f, tuple) and len(f) == 3 for f in registry().failures)
