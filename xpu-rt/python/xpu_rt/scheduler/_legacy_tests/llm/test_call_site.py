"""Tests for :mod:`compgen.llm.call_site` (P3.0).

Coverage:

Positive:
* A decorated primary runs end-to-end when the LLM is enabled, the
  output is schema-validated, and the wrapper exposes ``.card`` /
  ``.fallback`` / ``.primary`` attributes for introspection.
* When ``COMPGEN_DISABLE_LLM=1`` is set, the fallback runs and the
  primary is never invoked.
* When the primary raises, the wrapper transparently falls back.
* ``list_call_sites`` returns alphabetised, every site card.

Negative controls:
* A site referencing an unregistered fallback raises at decoration.
* Re-registering the same site id raises.
* Re-registering a fallback name with a different function raises.
* An unknown forbidden action in the card raises.
* The decorator rejects an output_schema that isn't a JSON object.
* The wrapper raises :class:`LLMOutputSchemaError` when the primary's
  output violates the schema.
* The wrapper raises :class:`LLMOutputSchemaError` when the fallback's
  output violates the schema (no silent pass-through).
"""

from __future__ import annotations

from typing import Any

import pytest
from compgen.llm.call_site import (
    FORBIDDEN_LLM_ACTIONS,
    LLMCallSiteError,
    LLMOutputSchemaError,
    _reset_registry_for_tests,
    get_call_site,
    list_call_site_ids,
    list_call_sites,
    llm_call_site,
    register_fallback,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a clean global registry; restore on teardown so
    other test modules find the production primitives intact."""

    import compgen.llm.call_site as cs

    saved_sites = dict(cs._CALL_SITES)
    saved_fallbacks = dict(cs._FALLBACKS)
    _reset_registry_for_tests()
    try:
        yield
    finally:
        _reset_registry_for_tests()
        cs._CALL_SITES.update(saved_sites)
        cs._FALLBACKS.update(saved_fallbacks)


SIMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["value"],
    "properties": {"value": {"type": "integer", "minimum": 0}},
    "additionalProperties": False,
}


# ---------- Positive --------------------------------------------------


def test_primary_runs_when_llm_enabled(monkeypatch):
    monkeypatch.delenv("COMPGEN_DISABLE_LLM", raising=False)

    @register_fallback("fb_simple")
    def _fb(x: int) -> dict[str, Any]:
        return {"value": 0}

    @llm_call_site(
        site_id="test_primary_runs",
        leverage="echo a positive int",
        inputs=["int"],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_simple",
    )
    def echo(x: int) -> dict[str, Any]:
        return {"value": x}

    assert echo(7) == {"value": 7}
    assert echo.card.site_id == "test_primary_runs"
    assert callable(echo.fallback)
    assert callable(echo.primary)


def test_fallback_runs_when_disabled(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    primary_calls: list[int] = []

    @register_fallback("fb_zero")
    def _fb(x: int) -> dict[str, Any]:
        return {"value": 0}

    @llm_call_site(
        site_id="test_fallback_runs",
        leverage="echo a positive int",
        inputs=["int"],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_zero",
    )
    def echo(x: int) -> dict[str, Any]:
        primary_calls.append(x)
        return {"value": x}

    assert echo(7) == {"value": 0}
    assert primary_calls == []


def test_primary_exception_triggers_fallback(monkeypatch):
    monkeypatch.delenv("COMPGEN_DISABLE_LLM", raising=False)

    @register_fallback("fb_safe")
    def _fb(x: int) -> dict[str, Any]:
        return {"value": x * 2}

    @llm_call_site(
        site_id="test_exc_fallback",
        leverage="double, falling back gracefully",
        inputs=["int"],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_safe",
    )
    def boom(x: int) -> dict[str, Any]:
        raise RuntimeError("intentional")

    assert boom(3) == {"value": 6}


def test_list_call_sites_alphabetical(monkeypatch):
    monkeypatch.delenv("COMPGEN_DISABLE_LLM", raising=False)

    @register_fallback("fb_a")
    def _a(*a, **k):
        return {"value": 0}

    @llm_call_site(
        site_id="zeta", leverage="z", inputs=[], output_schema=SIMPLE_SCHEMA,
        forbidden=[], fallback="fb_a",
    )
    def z():
        return {"value": 0}

    @llm_call_site(
        site_id="alpha", leverage="a", inputs=[], output_schema=SIMPLE_SCHEMA,
        forbidden=[], fallback="fb_a",
    )
    def a():
        return {"value": 0}

    assert list_call_site_ids() == ["alpha", "zeta"]
    cards = list_call_sites()
    assert [c.site_id for c in cards] == ["alpha", "zeta"]


def test_get_call_site_returns_card():
    @register_fallback("fb_get")
    def _fb(*a, **k):
        return {"value": 0}

    @llm_call_site(
        site_id="getme",
        leverage="get a thing",
        inputs=["x"],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_get",
    )
    def fn():
        return {"value": 0}

    card = get_call_site("getme")
    assert card.site_id == "getme"
    assert card.leverage == "get a thing"


def test_forbidden_actions_enum_is_documented():
    assert "invent_candidate_not_in_input_list" in FORBIDDEN_LLM_ACTIONS
    assert "be_sole_correctness_decider" in FORBIDDEN_LLM_ACTIONS


# ---------- Negative controls ----------------------------------------


def test_unregistered_fallback_raises():
    with pytest.raises(LLMCallSiteError, match="unknown fallback"):

        @llm_call_site(
            site_id="bad",
            leverage="x",
            inputs=[],
            output_schema=SIMPLE_SCHEMA,
            forbidden=[],
            fallback="does_not_exist",
        )
        def _f():
            return {"value": 0}


def test_double_registration_raises():
    @register_fallback("fb_dup")
    def _fb(*a, **k):
        return {"value": 0}

    @llm_call_site(
        site_id="dup",
        leverage="x",
        inputs=[],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_dup",
    )
    def _f():
        return {"value": 0}

    with pytest.raises(LLMCallSiteError, match="already registered"):

        @llm_call_site(
            site_id="dup",
            leverage="x",
            inputs=[],
            output_schema=SIMPLE_SCHEMA,
            forbidden=[],
            fallback="fb_dup",
        )
        def _f2():
            return {"value": 0}


def test_fallback_redefinition_raises():
    @register_fallback("fb_redef")
    def _first(*a, **k):
        return {"value": 0}

    with pytest.raises(LLMCallSiteError, match="already registered"):

        @register_fallback("fb_redef")
        def _second(*a, **k):
            return {"value": 1}


def test_unknown_forbidden_action_raises():
    @register_fallback("fb_bad")
    def _fb(*a, **k):
        return {"value": 0}

    with pytest.raises(LLMCallSiteError, match="forbidden action"):

        @llm_call_site(
            site_id="bad_forbidden",
            leverage="x",
            inputs=[],
            output_schema=SIMPLE_SCHEMA,
            forbidden=["totally_made_up"],
            fallback="fb_bad",
        )
        def _f():
            return {"value": 0}


def test_output_schema_must_be_object():
    @register_fallback("fb_obj")
    def _fb(*a, **k):
        return ["nope"]

    with pytest.raises(LLMCallSiteError, match="output_schema must be a JSON-schema object"):

        @llm_call_site(
            site_id="bad_schema",
            leverage="x",
            inputs=[],
            output_schema={"type": "array"},
            forbidden=[],
            fallback="fb_obj",
        )
        def _f():
            return ["nope"]


def test_primary_violating_schema_raises(monkeypatch):
    monkeypatch.delenv("COMPGEN_DISABLE_LLM", raising=False)

    @register_fallback("fb_violate_p")
    def _fb(*a, **k):
        return {"value": 0}

    @llm_call_site(
        site_id="violate_p",
        leverage="break the schema",
        inputs=[],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_violate_p",
    )
    def bad():
        return {"value": -1}  # minimum=0

    with pytest.raises(LLMOutputSchemaError, match="schema validation"):
        bad()


def test_fallback_violating_schema_raises(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")

    @register_fallback("fb_violate_f")
    def _fb(*a, **k):
        return {"value": -5}  # invalid

    @llm_call_site(
        site_id="violate_f",
        leverage="break via fallback",
        inputs=[],
        output_schema=SIMPLE_SCHEMA,
        forbidden=[],
        fallback="fb_violate_f",
    )
    def primary():
        return {"value": 0}  # never invoked

    with pytest.raises(LLMOutputSchemaError, match="schema validation"):
        primary()


def test_missing_leverage_raises():
    @register_fallback("fb_lev")
    def _fb(*a, **k):
        return {"value": 0}

    with pytest.raises(LLMCallSiteError, match="leverage"):

        @llm_call_site(
            site_id="no_lev",
            leverage="",
            inputs=[],
            output_schema=SIMPLE_SCHEMA,
            forbidden=[],
            fallback="fb_lev",
        )
        def _f():
            return {"value": 0}


def test_get_unknown_call_site_raises():
    with pytest.raises(LLMCallSiteError, match="unknown call site"):
        get_call_site("never_registered")
