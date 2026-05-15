"""Typed envelope for solver requests, responses, and routing.

. The compiler treats every solver invocation (Z3 proof,
OR-Tools placement/schedule, MOSEK / HiGHS LP-MILP) as the same
shape:

    SolverRequest --(routing)--> SolverBackend.solve --> SolverResponse

Every response carries ``formulation_hash`` (canonical-JSON SHA256 of
the request's ``formulation`` payload), ``selected_backend``,
``status``, ``time_ms`` and either a solution / solution_path /
counterexample / infeasibility_reason. The hash is byte-stable across
reruns, so audit gates can confirm a re-run did not silently mutate
the problem.

Hard rules:

* ``status`` is typed (``SolverStatus``); never raw exceptions.
* ``optimal`` / ``proved`` are reserved for backend-confirmed
  outcomes — never assigned as a default.
* ``blocked`` is the honest answer when no backend supports the
  ``SolverProblemKind`` on this host. The caller must NOT fall back
  to a greedy heuristic and claim success.

Used by ``backend_registry`` (registers backends), ``routing``
(deterministic kind -> backend table), and every planner under
``compgen.solve`` (memory, placement, overlap, Z3 obligations).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "SolverProblemKind",
    "SolverBackendName",
    "SolverStatus",
    "BackendAvailabilityStatus",
    "BackendProbeResult",
    "SolverRequest",
    "SolverResponse",
    "compute_formulation_hash",
    "canonical_formulation_dump",
    "SOLVER_REQUEST_SCHEMA_VERSION",
    "SOLVER_RESPONSE_SCHEMA_VERSION",
]


SOLVER_REQUEST_SCHEMA_VERSION = "solver_request_v1"
SOLVER_RESPONSE_SCHEMA_VERSION = "solver_response_v1"


class SolverProblemKind(str, Enum):
    """Discrete kinds of problems the solver layer routes."""

    # Proof / verification (Z3)
    PEEPHOLE_VERIFY = "peephole_verify"
    RECIPE_REFINEMENT = "recipe_refinement"
    TRANSLATION_VALIDATION = "translation_validation"
    SHAPE_PREDICATE_VERIFY = "shape_predicate_verify"
    PLAN_INVARIANT_VERIFY = "plan_invariant_verify"

    # Combinatorial planning (OR-Tools CP-SAT)
    PLACEMENT = "placement"
    SCHEDULE = "schedule"
    NO_OVERLAP_SCHEDULE = "no_overlap_schedule"
    EVENT_ORDERING = "event_ordering"
    OVERLAP_PLANNING = "overlap_planning"

    # Numeric / MILP (MOSEK preferred, HiGHS fallback)
    MEMORY_ALLOCATION = "memory_allocation"
    BUFFER_ALIASING = "buffer_aliasing"
    BANDWIDTH_ALLOCATION = "bandwidth_allocation"
    COST_MODEL_FIT = "cost_model_fit"

    # Meta
    BACKEND_PROBE = "backend_probe"


class SolverBackendName(str, Enum):
    """Names of registered solver backends."""

    Z3 = "z3"
    ORTOOLS_CP_SAT = "ortools_cp_sat"
    MOSEK = "mosek"
    HIGHS = "highs"
    OSQP_OPTIONAL = "osqp_optional"
    CLARABEL_OPTIONAL = "clarabel_optional"


class SolverStatus(str, Enum):
    """Outcome of a solver call."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    PROVED = "proved"
    SAT_COUNTEREXAMPLE = "sat_counterexample"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    NOT_PROVED = "not_proved"


class BackendAvailabilityStatus(str, Enum):
    """Why a backend is or is not usable on this host."""

    AVAILABLE = "available"
    IMPORT_MISSING = "import_missing"
    LICENSE_MISSING = "license_missing"
    LICENSE_TOKEN_UNAVAILABLE = "license_token_unavailable"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    PROBE_ERROR = "probe_error"


@dataclass(frozen=True)
class BackendProbeResult:
    """Result of ``SolverBackend.probe()``: typed availability."""

    backend: SolverBackendName
    availability: BackendAvailabilityStatus
    version: str | None = None
    supports: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "availability": self.availability.value,
            "version": self.version,
            "supports": list(self.supports),
            "detail": self.detail,
        }


def canonical_formulation_dump(formulation: Any) -> str:
    """Canonical JSON dump used for ``formulation_hash``.

    Sort keys, no whitespace, ``ensure_ascii`` False so unicode in
    operand names is byte-stable. Floats are converted via
    ``repr`` to avoid platform-specific double formatting from
    ``json.dumps``.
    """

    def _normalize(value: Any) -> Any:
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, dict):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        return value

    return json.dumps(_normalize(formulation), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_formulation_hash(formulation: Any) -> str:
    """SHA256[:16] of the canonical formulation dump."""

    payload = canonical_formulation_dump(formulation).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class SolverRequest:
    """A typed solver request.

    ``formulation`` is opaque to the envelope: each backend reads it
    according to ``problem_kind``. We compute the canonical hash here
    so call sites cannot forget to commit one.

    Attributes:
        problem_id: Caller-chosen identifier (e.g. ``"merlin_mlp_wide_memory_v1"``).
        problem_kind: One of ``SolverProblemKind``.
        formulation: Backend-specific problem data (JSON-serializable).
        time_budget_ms: Wall-clock budget; backends MUST enforce.
        optimality_required: If True, backends return ``optimal``
            or ``infeasible``; ``feasible`` is converted to
            ``timeout`` when proof of optimality is unavailable.
        backend_preference: Override routing; ``None`` for default.
        artifact_dir: Where to write solution / certificate files.
        source_artifact_hashes: SHA refs to inputs (e.g. recipe,
            target profile) for traceability.
        metadata: Free-form; never affects routing.
    """

    problem_id: str
    problem_kind: SolverProblemKind
    formulation: Any
    time_budget_ms: int = 30_000
    optimality_required: bool = False
    backend_preference: SolverBackendName | None = None
    artifact_dir: str | None = None
    source_artifact_hashes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SOLVER_REQUEST_SCHEMA_VERSION

    @property
    def formulation_hash(self) -> str:
        return compute_formulation_hash(self.formulation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "problem_kind": self.problem_kind.value,
            "formulation": self.formulation,
            "formulation_hash": self.formulation_hash,
            "time_budget_ms": self.time_budget_ms,
            "optimality_required": self.optimality_required,
            "backend_preference": self.backend_preference.value if self.backend_preference else None,
            "artifact_dir": self.artifact_dir,
            "source_artifact_hashes": list(self.source_artifact_hashes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> SolverRequest:
        return cls(
            problem_id=body["problem_id"],
            problem_kind=SolverProblemKind(body["problem_kind"]),
            formulation=body["formulation"],
            time_budget_ms=int(body.get("time_budget_ms", 30_000)),
            optimality_required=bool(body.get("optimality_required", False)),
            backend_preference=(
                SolverBackendName(body["backend_preference"])
                if body.get("backend_preference")
                else None
            ),
            artifact_dir=body.get("artifact_dir"),
            source_artifact_hashes=tuple(body.get("source_artifact_hashes", ())),
            metadata=dict(body.get("metadata", {})),
            schema_version=body.get("schema_version", SOLVER_REQUEST_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class SolverResponse:
    """A typed solver response.

    Fields are deliberately all optional except the envelope ones
    (``problem_id`` / ``problem_kind`` / ``selected_backend`` /
    ``status`` / ``time_ms`` / ``formulation_hash``); the right set is
    populated based on ``status``:

      - OPTIMAL / FEASIBLE: ``objective_value``, ``solution`` or
        ``solution_path``, optional ``lower_bound`` / ``upper_bound``
        / ``gap_from_optimum``.
      - PROVED / NOT_PROVED: ``certificate_path``.
      - SAT_COUNTEREXAMPLE: ``counterexample``.
      - INFEASIBLE / BLOCKED / UNSUPPORTED / TIMEOUT / ERROR:
        ``infeasibility_reason`` (string), optional ``logs``.

    Callers MUST consult ``status`` first; relying on
    ``objective_value`` without checking status is a silent-failure
    bug.
    """

    problem_id: str
    problem_kind: SolverProblemKind
    selected_backend: SolverBackendName
    backend_availability: BackendAvailabilityStatus
    status: SolverStatus
    formulation_hash: str
    time_ms: float
    objective_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    gap_from_optimum: float | None = None
    solution: Any = None
    solution_path: str | None = None
    certificate_path: str | None = None
    counterexample: dict[str, Any] | None = None
    infeasibility_reason: str | None = None
    logs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    schema_version: str = SOLVER_RESPONSE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["problem_kind"] = self.problem_kind.value
        body["selected_backend"] = self.selected_backend.value
        body["backend_availability"] = self.backend_availability.value
        body["status"] = self.status.value
        body["logs"] = list(self.logs)
        body["caveats"] = list(self.caveats)
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> SolverResponse:
        return cls(
            problem_id=body["problem_id"],
            problem_kind=SolverProblemKind(body["problem_kind"]),
            selected_backend=SolverBackendName(body["selected_backend"]),
            backend_availability=BackendAvailabilityStatus(body["backend_availability"]),
            status=SolverStatus(body["status"]),
            formulation_hash=body["formulation_hash"],
            time_ms=float(body["time_ms"]),
            objective_value=body.get("objective_value"),
            lower_bound=body.get("lower_bound"),
            upper_bound=body.get("upper_bound"),
            gap_from_optimum=body.get("gap_from_optimum"),
            solution=body.get("solution"),
            solution_path=body.get("solution_path"),
            certificate_path=body.get("certificate_path"),
            counterexample=body.get("counterexample"),
            infeasibility_reason=body.get("infeasibility_reason"),
            logs=tuple(body.get("logs", ())),
            caveats=tuple(body.get("caveats", ())),
            schema_version=body.get("schema_version", SOLVER_RESPONSE_SCHEMA_VERSION),
        )
