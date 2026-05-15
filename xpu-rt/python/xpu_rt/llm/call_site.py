"""LLM call-site decorator + registry (P3.0).

Every place in the compiler that hands a decision to an LLM declares
itself here. The decorator captures the typed contract — site id,
leverage statement, input/output schemas, forbidden invariants,
deterministic fallback — and the registry exposes the closed list to
the audit + the MCP bridge.

Hard rules (enforced):

1. **Every decorated site has a fallback.** ``COMPGEN_DISABLE_LLM=1``
   in the environment forces the fallback path; the wrapped function
   never sees an LLM call. CI runs this way by default so every
   primitive must be runnable deterministically.
2. **No invent of candidates.** The forbidden list is a *declaration*;
   per-primitive bodies are responsible for honoring it. The audit
   re-asserts at the call-site test level (e.g. ``rank_candidates``
   tests assert the output is a permutation of the input).
3. **Output_schema is enforced.** After the call (LLM or fallback)
   the decorator validates the return value against the declared
   output_schema and raises a typed error on violation. Silent
   pass-through is forbidden.

The registry is global (module-level dict). Re-registering a site id
raises — sites are *declared* once and never re-bound. Tests register
their own site ids to keep the global namespace clean.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import jsonschema

FORBIDDEN_LLM_ACTIONS: Final[tuple[str, ...]] = (
    "invent_candidate_not_in_input_list",
    "invent_numerical_threshold",
    "invent_tile_size_outside_candidate_set",
    "be_sole_correctness_decider",
    "pick_dispatch_without_cost_table_check",
    "emit_certificate",
)


class LLMCallSiteError(ValueError):
    """An LLM call-site decorator or registry rejected a request."""


class LLMOutputSchemaError(ValueError):
    """An LLM call-site's return value violated its output_schema."""


@dataclass(frozen=True)
class LLMCallSiteCard:
    """Frozen declaration of one LLM call site.

    The card is *what* the site promises; the registered Python
    callable is *how* it currently delivers. Tests verify both.
    """

    site_id: str
    leverage: str
    inputs: tuple[str, ...]
    output_schema: dict[str, Any]
    forbidden: tuple[str, ...]
    fallback: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.site_id or not isinstance(self.site_id, str):
            raise LLMCallSiteError(f"site_id must be a non-empty string; got {self.site_id!r}")
        if not self.leverage:
            raise LLMCallSiteError(f"site {self.site_id!r}: leverage must be a non-empty sentence")
        for a in self.forbidden:
            if a not in FORBIDDEN_LLM_ACTIONS:
                raise LLMCallSiteError(
                    f"site {self.site_id!r}: forbidden action {a!r} not in {FORBIDDEN_LLM_ACTIONS}"
                )
        if not self.fallback:
            raise LLMCallSiteError(
                f"site {self.site_id!r}: a deterministic fallback name is required"
            )
        if not isinstance(self.output_schema, dict) or self.output_schema.get("type") != "object":
            raise LLMCallSiteError(
                f"site {self.site_id!r}: output_schema must be a JSON-schema object"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "leverage": self.leverage,
            "inputs": list(self.inputs),
            "output_schema": dict(self.output_schema),
            "forbidden": list(self.forbidden),
            "fallback": self.fallback,
            "description": self.description,
        }


@dataclass
class _CallSiteEntry:
    card: LLMCallSiteCard
    primary: Callable[..., Any]
    fallback_fn: Callable[..., Any]


_CALL_SITES: dict[str, _CallSiteEntry] = {}
_FALLBACKS: dict[str, Callable[..., Any]] = {}


def register_fallback(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a deterministic fallback callable.

    Fallbacks are looked up by name when a call site is constructed,
    so the resolution order is fallback-first: declare the fallback,
    then declare the site that references it. The fallback signature
    matches the primary's exactly.
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _FALLBACKS and _FALLBACKS[name] is not fn:
            raise LLMCallSiteError(
                f"fallback {name!r} already registered; re-registration with a "
                f"different function is forbidden (sites are declared once)"
            )
        _FALLBACKS[name] = fn
        return fn

    return _wrap


def _resolve_output_schema(output: Any, site_id: str, schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=output, schema=schema)
    except jsonschema.ValidationError as exc:
        raise LLMOutputSchemaError(
            f"site {site_id!r} output failed schema validation: {exc.message} "
            f"(path={list(exc.absolute_path)})"
        ) from exc


def llm_call_site(
    *,
    site_id: str,
    leverage: str,
    inputs: list[str] | tuple[str, ...],
    output_schema: dict[str, Any],
    forbidden: list[str] | tuple[str, ...],
    fallback: str,
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers an LLM call site and enforces its contract.

    The decorated function becomes the *primary* path. The named
    ``fallback`` (registered via :func:`register_fallback`) is the
    deterministic path used when ``COMPGEN_DISABLE_LLM=1`` is set or
    when the primary raises. Both paths produce values validated
    against ``output_schema``; a violation raises
    :class:`LLMOutputSchemaError` regardless of which path produced it.

    The returned wrapper carries ``.card`` and ``.fallback`` attributes
    so callers (and tests) can introspect the registered contract.
    """

    if site_id in _CALL_SITES:
        raise LLMCallSiteError(
            f"site {site_id!r} already registered; re-declaration is forbidden"
        )
    if fallback not in _FALLBACKS:
        raise LLMCallSiteError(
            f"site {site_id!r} references unknown fallback {fallback!r}; "
            f"register it with @register_fallback({fallback!r}) first"
        )
    card = LLMCallSiteCard(
        site_id=site_id,
        leverage=leverage,
        inputs=tuple(inputs),
        output_schema=output_schema,
        forbidden=tuple(forbidden),
        fallback=fallback,
        description=description,
    )

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fallback_fn = _FALLBACKS[fallback]
        entry = _CallSiteEntry(card=card, primary=fn, fallback_fn=fallback_fn)
        _CALL_SITES[site_id] = entry

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            disable = os.environ.get("COMPGEN_DISABLE_LLM") == "1"
            if disable:
                out = fallback_fn(*args, **kwargs)
            else:
                try:
                    out = fn(*args, **kwargs)
                except Exception:
                    out = fallback_fn(*args, **kwargs)
            _resolve_output_schema(out, site_id, output_schema)
            return out

        wrapper.card = card  # type: ignore[attr-defined]
        wrapper.fallback = fallback_fn  # type: ignore[attr-defined]
        wrapper.primary = fn  # type: ignore[attr-defined]
        return wrapper

    return _decorate


def get_call_site(site_id: str) -> LLMCallSiteCard:
    if site_id not in _CALL_SITES:
        raise LLMCallSiteError(f"unknown call site {site_id!r}")
    return _CALL_SITES[site_id].card


def list_call_sites() -> list[LLMCallSiteCard]:
    """Return every registered call site, alphabetically by id."""

    return [entry.card for _, entry in sorted(_CALL_SITES.items())]


def list_call_site_ids() -> list[str]:
    return sorted(_CALL_SITES.keys())


def _reset_registry_for_tests() -> None:
    """Wipe the registry. ONLY for use by tests in tests/llm/.

    Production code must never call this — sites are declared once.
    """

    _CALL_SITES.clear()
    _FALLBACKS.clear()


__all__ = [
    "FORBIDDEN_LLM_ACTIONS",
    "LLMCallSiteCard",
    "LLMCallSiteError",
    "LLMOutputSchemaError",
    "_reset_registry_for_tests",
    "get_call_site",
    "list_call_site_ids",
    "list_call_sites",
    "llm_call_site",
    "register_fallback",
]
