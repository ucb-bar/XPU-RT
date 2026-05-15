"""Solver backend registry.

. Auto-registers every shipped backend (Z3, OR-Tools CP-SAT,
MOSEK, HiGHS), caches probe results, and exposes typed lookup.

The registry is the single point that knows which backends are
installed and licensed on this host. Routing
(:mod:`compgen.solve.routing`) consults it before dispatching a
problem.

Probe results are cached per-registry-instance. Tests can construct
a fresh registry to bypass the cache, or monkey-patch a backend's
``probe`` to force ``import_missing``.
"""

from __future__ import annotations

import threading
from typing import Iterable

from compgen.solve.backends.base import SolverBackend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
)

__all__ = ["SolverBackendRegistry", "default_registry"]


class SolverBackendRegistry:
    """Holds registered :class:`SolverBackend` instances + probe cache."""

    def __init__(self) -> None:
        self._backends: dict[SolverBackendName, SolverBackend] = {}
        self._probe_cache: dict[SolverBackendName, BackendProbeResult] = {}
        self._lock = threading.Lock()

    def register(self, backend: SolverBackend) -> None:
        """Register a backend. Replaces any prior entry with the same name."""

        with self._lock:
            self._backends[backend.name] = backend
            self._probe_cache.pop(backend.name, None)

    def names(self) -> tuple[SolverBackendName, ...]:
        """All registered backend names, sorted by enum order."""

        return tuple(sorted(self._backends.keys(), key=lambda n: n.value))

    def get_backend(self, name: SolverBackendName) -> SolverBackend | None:
        return self._backends.get(name)

    def probe(self, name: SolverBackendName, *, force: bool = False) -> BackendProbeResult:
        """Probe one backend, caching the result.

        Args:
            name: Which backend to probe.
            force: If True, ignore the cache and re-probe.
        """

        backend = self._backends.get(name)
        if backend is None:
            return BackendProbeResult(
                backend=name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail="backend not registered",
            )
        if not force and name in self._probe_cache:
            return self._probe_cache[name]
        result = backend.probe()
        with self._lock:
            self._probe_cache[name] = result
        return result

    def probe_all(self, *, force: bool = False) -> dict[SolverBackendName, BackendProbeResult]:
        return {name: self.probe(name, force=force) for name in self.names()}

    def available_backends(self) -> tuple[SolverBackendName, ...]:
        """Names of backends whose probe returned ``available``."""

        return tuple(
            name
            for name in self.names()
            if self.probe(name).availability is BackendAvailabilityStatus.AVAILABLE
        )

    def supports_for(
        self,
        problem_kind: SolverProblemKind,
        *,
        only_available: bool = True,
    ) -> tuple[SolverBackendName, ...]:
        """Names of backends whose ``supports(kind)`` is True.

        ``only_available=True`` filters by ``probe()`` first.
        """

        candidates: Iterable[SolverBackendName] = (
            self.available_backends() if only_available else self.names()
        )
        return tuple(
            name for name in candidates if self._backends[name].supports(problem_kind)
        )

    def reset_cache(self) -> None:
        with self._lock:
            self._probe_cache.clear()


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_REGISTRY: SolverBackendRegistry | None = None


def default_registry() -> SolverBackendRegistry:
    """Process-wide default registry with all shipped backends registered.

    Imports backend implementations lazily so that callers who only
    want the type envelope do not pay the dep-check cost.
    """

    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is not None:
            return _DEFAULT_REGISTRY

        registry = SolverBackendRegistry()
        # Lazy imports — each backend handles its own optional dep.
        from compgen.solve.backends.highs_backend import HighsBackend
        from compgen.solve.backends.mosek_backend import MosekBackend
        from compgen.solve.backends.ortools_cp_sat_backend import OrToolsCpSatBackend
        from compgen.solve.backends.z3_backend import Z3Backend

        registry.register(Z3Backend())
        registry.register(OrToolsCpSatBackend())
        registry.register(MosekBackend())
        registry.register(HighsBackend())

        _DEFAULT_REGISTRY = registry
        return registry
