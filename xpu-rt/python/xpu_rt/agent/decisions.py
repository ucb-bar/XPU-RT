"""Decision-site registry — the agent's write path.

The earlier architecture had stage plugins write IR attributes directly
based on hardcoded heuristics or oracle picks. That made the oracle the
decider and gave the LLM no way to intervene. This module inverts that
relationship.

**Contract.** Every place the compiler has a choice to make is a
:class:`DecisionSite`. A site declares:

* A stable ``site_id`` (e.g. ``"stage.encoding.matmul_0.encoding"``).
* A list of :class:`DecisionCandidate` objects (oracle-sourced + any
  candidates the LLM inserted via :func:`register_custom_candidate`).
* A ``context`` dict with just enough data for a reader to understand
  the choice (shapes, dtypes, envelope snippets).
* The oracle's recommended candidate id — **non-binding**.

The pipeline never applies a decision directly. It calls
:meth:`DecisionRegistry.resolve(site)` which:

1. Checks whether an agent has pre-applied a choice via
   :meth:`DecisionRegistry.apply`.
2. If not, uses the oracle's recommendation (``source="fallback_oracle"``).
3. Emits a ``decision_site`` trace event at enqueue time and a
   ``decision`` event at resolution time, with the ``source`` field
   distinguishing agent picks from oracle fallbacks.

MCP tools (:mod:`compgen.mcp.tools.decisions`) expose the registry to
the agent: ``list_decisions``, ``apply_decision``, ``override_decision``.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionCandidate:
    """One option at a decision site.

    Attributes:
        id: Stable identifier within the site's candidate list.
        value: The actual payload applied when this candidate wins
            (e.g. ``"tiled_16x16x16"`` for an encoding choice).
        source: Where this candidate came from. Common values:
            ``"oracle:fusion"``, ``"oracle:tile"``, ``"oracle:granularity"``,
            ``"cost_model"``, ``"invent"`` (LLM-synthesized),
            ``"previous_session"`` (knowledge-store replay).
        oracle_verdict: The oracle's stance on this candidate
            (``"recommended"`` | ``"allowed"`` | ``"discouraged"``).
        oracle_reason: Short rationale from the oracle.
        oracle_confidence: 0..1.
        cost_breakdown: Optional per-component cost estimate
            (``{dram_savings_us: ..., launch_savings_us: ...}``).
        knowledge_brief: Markdown excerpt of relevant lessons pulled
            from :mod:`compgen.memory.knowledge`.
        evidence: Free-form auxiliary data (bench history, prior
            autotune winners, etc).
    """

    id: str
    value: Any
    source: str
    oracle_verdict: str = ""
    oracle_reason: str = ""
    oracle_confidence: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    knowledge_brief: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionSite:
    """A place where the compiler must choose among candidates.

    ``status`` transitions: ``pending`` → (``resolved`` | ``overridden``).
    A site is ``overridden`` when the agent replaces the outcome AFTER
    it was already resolved; it is ``resolved`` on first resolution.
    """

    site_id: str
    kind: str  # "encoding" | "tile" | "fusion" | "granularity" | "kernel_choice" | "dispatch_group"
    context: dict[str, Any]
    candidates: tuple[DecisionCandidate, ...]
    oracle_recommended_id: str = ""
    trace_event_id: str = ""
    status: str = "pending"
    outcome: "DecisionOutcome | None" = None

    def candidate_by_id(self, cid: str) -> DecisionCandidate | None:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "kind": self.kind,
            "context": dict(self.context),
            "candidates": [asdict(c) for c in self.candidates],
            "oracle_recommended_id": self.oracle_recommended_id,
            "trace_event_id": self.trace_event_id,
            "status": self.status,
            "outcome": (self.outcome.to_dict() if self.outcome else None),
        }


@dataclass
class DecisionOutcome:
    """The resolved choice for a site.

    ``source`` records how the choice was made:

    * ``"agent"`` — an LLM called :meth:`DecisionRegistry.apply` with
      a rationale.
    * ``"fallback_oracle"`` — no agent pick was registered; the
      oracle's recommendation was applied automatically.
    * ``"override"`` — an agent replaced a previously-resolved outcome.
    * ``"invent"`` — the agent chose a novel candidate not enumerated
      by the oracle.
    """

    site_id: str
    chosen_id: str
    chosen_value: Any
    source: str
    rationale: str = ""
    llm_turn_id: str = ""
    decision_event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "chosen_id": self.chosen_id,
            "chosen_value": self.chosen_value,
            "source": self.source,
            "rationale": self.rationale,
            "llm_turn_id": self.llm_turn_id,
            "decision_event_id": self.decision_event_id,
        }


class DecisionRegistry:
    """Per-session registry.

    Lifetime: one per MCP session. Stored on :class:`McpSession` so the
    agent's applies are scoped to its compile.
    """

    def __init__(self) -> None:
        self._sites: dict[str, DecisionSite] = {}
        # Pre-applied decisions keyed by site_id the agent registered
        # BEFORE the pipeline enqueues the site. ``resolve`` drains
        # these into outcomes.
        self._preapplied: dict[str, DecisionOutcome] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------
    # Stage-plugin side
    # ---------------------------------------------------------------

    def enqueue(self, site: DecisionSite) -> DecisionSite:
        """Register a new decision site. Emits a ``decision_site`` trace event.

        Returns the site with ``trace_event_id`` populated. If the site_id
        already exists, returns the existing site unchanged (idempotent).
        """
        with self._lock:
            existing = self._sites.get(site.site_id)
            if existing is not None:
                return existing
            self._sites[site.site_id] = site
        # Trace event is emitted outside the lock to avoid deadlocks if
        # a publisher ever acquires more locks.
        from compgen.trace import DecisionSitePublisher

        event_id = DecisionSitePublisher.emit(
            site_id=site.site_id,
            kind=site.kind,
            context=site.context,
            candidate_ids=[c.id for c in site.candidates],
            oracle_recommended_id=site.oracle_recommended_id,
        )
        site.trace_event_id = event_id or ""
        return site

    def resolve(self, site_id: str) -> DecisionOutcome:
        """Resolve a site. Uses a pre-applied agent pick if present,
        else falls back to the oracle recommendation.

        Emits a ``decision`` trace event carrying source and rationale.
        """
        with self._lock:
            site = self._sites.get(site_id)
            if site is None:
                raise KeyError(f"unknown decision site: {site_id!r}")
            if site.outcome is not None:
                return site.outcome
            preapplied = self._preapplied.pop(site_id, None)

        if preapplied is not None:
            # When the agent pre-applied BEFORE the site existed, we
            # couldn't validate the ``chosen_id`` or derive
            # ``chosen_value`` from the candidate list. Fill those in
            # now that the site is live. Novel ``invent:`` values are
            # passed through unchanged.
            if preapplied.chosen_value is None and not preapplied.chosen_id.startswith("invent:"):
                candidate = site.candidate_by_id(preapplied.chosen_id)
                if candidate is None:
                    raise RuntimeError(
                        f"pre-applied candidate {preapplied.chosen_id!r} not in site "
                        f"{site_id!r}; valid ids: {[c.id for c in site.candidates]}"
                    )
                preapplied = DecisionOutcome(
                    site_id=preapplied.site_id,
                    chosen_id=preapplied.chosen_id,
                    chosen_value=candidate.value,
                    source=preapplied.source,
                    rationale=preapplied.rationale,
                    llm_turn_id=preapplied.llm_turn_id,
                )
            return self._commit_outcome(site, preapplied)

        # Oracle fallback — use the recommended candidate; if none is
        # recommended, pick the first candidate as a safe default.
        chosen_id = site.oracle_recommended_id or (site.candidates[0].id if site.candidates else "")
        candidate = site.candidate_by_id(chosen_id) if chosen_id else None
        if candidate is None:
            raise RuntimeError(
                f"site {site_id!r} has no candidates and no oracle recommendation; "
                "cannot resolve"
            )
        outcome = DecisionOutcome(
            site_id=site_id,
            chosen_id=candidate.id,
            chosen_value=candidate.value,
            source="fallback_oracle",
            rationale=candidate.oracle_reason
            or f"oracle recommended {candidate.id!r} (no agent override)",
        )
        return self._commit_outcome(site, outcome)

    # ---------------------------------------------------------------
    # Agent side (drives from MCP tools)
    # ---------------------------------------------------------------

    def apply(
        self,
        site_id: str,
        *,
        chosen_id: str,
        rationale: str,
        chosen_value: Any = None,
        source: str = "agent",
        llm_turn_id: str = "",
    ) -> DecisionOutcome:
        """Record a pre-emptive agent decision.

        Two code paths:

        * Site **not yet enqueued** — stash the outcome in a side-table
          so the next ``enqueue``+``resolve`` applies it. The value is
          taken from the named candidate when ``chosen_value`` is None;
          otherwise the caller supplies a novel value (``"invent"`` mode).
        * Site **already enqueued but not resolved** — resolve it now
          with this outcome.
        * Site **already resolved** — see :meth:`override`.
        """
        with self._lock:
            site = self._sites.get(site_id)
            if site is not None and site.outcome is not None:
                raise RuntimeError(
                    f"site {site_id!r} already resolved; use override() to replace"
                )

            value = chosen_value
            if site is not None and value is None:
                candidate = site.candidate_by_id(chosen_id)
                if candidate is None and not chosen_id.startswith("invent:"):
                    raise KeyError(
                        f"candidate {chosen_id!r} not in site {site_id!r}; "
                        f"valid ids: {[c.id for c in site.candidates]}"
                    )
                if candidate is not None:
                    value = candidate.value

            outcome = DecisionOutcome(
                site_id=site_id,
                chosen_id=chosen_id,
                chosen_value=value,
                source=source or ("invent" if chosen_id.startswith("invent:") else "agent"),
                rationale=rationale,
                llm_turn_id=llm_turn_id,
            )

            if site is None:
                self._preapplied[site_id] = outcome
                return outcome

        # Commit outside the lock so the trace publisher can take the bus lock.
        return self._commit_outcome(site, outcome)

    def override(
        self,
        site_id: str,
        *,
        chosen_id: str,
        rationale: str,
        chosen_value: Any = None,
        llm_turn_id: str = "",
    ) -> DecisionOutcome:
        """Replace an already-resolved outcome."""
        with self._lock:
            site = self._sites.get(site_id)
            if site is None:
                raise KeyError(f"unknown decision site: {site_id!r}")
            value = chosen_value
            if value is None:
                candidate = site.candidate_by_id(chosen_id)
                if candidate is None and not chosen_id.startswith("invent:"):
                    raise KeyError(
                        f"candidate {chosen_id!r} not in site {site_id!r}"
                    )
                if candidate is not None:
                    value = candidate.value
            outcome = DecisionOutcome(
                site_id=site_id,
                chosen_id=chosen_id,
                chosen_value=value,
                source="override",
                rationale=rationale,
                llm_turn_id=llm_turn_id,
            )
            site.outcome = outcome
            site.status = "overridden"
        self._emit_decision_event(site, outcome)
        return outcome

    def list_pending(self) -> list[DecisionSite]:
        with self._lock:
            return [s for s in self._sites.values() if s.status == "pending"]

    def list_all(self) -> list[DecisionSite]:
        with self._lock:
            return list(self._sites.values())

    def get(self, site_id: str) -> DecisionSite | None:
        with self._lock:
            return self._sites.get(site_id)

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _commit_outcome(self, site: DecisionSite, outcome: DecisionOutcome) -> DecisionOutcome:
        site.outcome = outcome
        site.status = "resolved" if outcome.source != "override" else "overridden"
        self._emit_decision_event(site, outcome)
        return outcome

    def _emit_decision_event(self, site: DecisionSite, outcome: DecisionOutcome) -> None:
        from compgen.trace import DecisionPublisher

        event_id = DecisionPublisher.emit(
            decision_type=site.kind,
            site_id=site.site_id,
            chosen=outcome.chosen_id,
            chosen_value=outcome.chosen_value,
            source=outcome.source,
            rationale=outcome.rationale,
            candidates=[c.id for c in site.candidates],
            oracle_recommended_id=site.oracle_recommended_id,
            llm_turn_id=outcome.llm_turn_id,
        )
        outcome.decision_event_id = event_id or ""


# ---------------------------------------------------------------------------
# Active registry (process-level + ContextVar)
# ---------------------------------------------------------------------------

import contextvars as _contextvars

_active_registry: _contextvars.ContextVar["DecisionRegistry | None"] = _contextvars.ContextVar(
    "compgen_decision_registry", default=None
)
_process_registry: "DecisionRegistry | None" = None


def install_registry(registry: DecisionRegistry | None) -> None:
    """Install ``registry`` as the active one for this context AND the process.

    Mirrors :func:`compgen.trace.install_bus` — ContextVar for task isolation,
    process-level fallback so async boundaries don't lose the registry.
    """
    global _process_registry
    _active_registry.set(registry)
    _process_registry = registry


def get_active_registry() -> DecisionRegistry | None:
    """Return the active registry, or ``None`` when no session owns one."""
    r = _active_registry.get()
    if r is not None:
        return r
    return _process_registry


__all__ = [
    "DecisionCandidate",
    "DecisionOutcome",
    "DecisionRegistry",
    "DecisionSite",
    "get_active_registry",
    "install_registry",
]
