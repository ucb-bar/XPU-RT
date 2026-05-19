"""Per-(workload, backend) two-term calibration model for the QNN cost matrix.

A4 (`build/experiments/exp9_calibration/report.md`) found that the raw
per-op cost matrix systematically undercounts DSP/GPU wall time by 5–6×
because per-op profiling captures only kernel time, not dispatch/launch/
transfer overhead. The closed-loop EMA contention factors silently
absorbed this gap, producing 13–40% per-round prediction error.

The calibration model has gone through four iterations:

* **v1** — single per-*backend* overhead constant. exp18 test E showed
  this constant overfits to the largest workload — yolov8n's DSP-side
  ``e2e_solo - Σ per_op`` delta is 226 ms, but dronet's *entire* DSP
  chain-sum is < 8 ms, so blindly adding 226 ms produces a 43%+ error
  on dronet.
* **v2** — per-*(workload, backend)* base overhead. Drops cross-workload
  mean and indexes by ``(workload_id, backend)``. Solo E2E error
  collapses to 0% by construction. But the contended closed-loop
  yolov8n-DSP error grows to 29.3% (vs v1's 13.5%) because v2 has no
  way to absorb the multi-tenant residual: under contention yolov8n
  runs *faster* than its solo baseline (DSP fully available while
  CPU runs 12× dronet), and v2's solo-fit overshoots.
* **v3** — *two-term* model::

      predicted = (chain_sum + base_overhead[w, b]) * contention_factor[w, b]

  ``base_overhead[w][b] = max(0, e2e_solo[w][b] - Σ per_op_costs[w][*][b])``
  is the same v2 quantity, fit from solo E2E.
  ``contention_factor[w][b] = mean over closed-loop rounds of
  (measured / predicted_base)`` for cells where contended ground truth
  exists; 1.0 (with provenance ``"default_no_data"``) elsewhere.
* **v4** (this module) — adds a ``deployment_mode`` axis on top of v3.
  v3 collapsed *cold-start* (qnn-net-run includes graph init) and
  *warm-loop* (cached-context binaries, pre-allocated tensor buffers,
  SCHED_FIFO+mlockall — see ``realtime_qnn/REPLICATION.md``) into one
  overhead constant. For yolov8n on DSP that constant is ~295 ms,
  dominated by graph init that the warm bundle amortises to zero per
  iter; the same loop run with warm-bundle ground truth measures
  ~55 ms/iter. v4 keeps the v3 two-term shape but stores parallel
  ``overhead_us_warm`` / ``contention_factor_warm`` maps so callers
  can pick the right constants per deployment mode. The mapping
  ``techniques → mode`` lives in :func:`techniques_to_mode`; any
  technique set containing ``cached_context_binary`` is treated as
  warm (the simplification can be refined later when richer modes are
  needed).

The two terms decouple: base overhead absorbs the solo dispatch/launch
gap; contention absorbs the multi-tenant residual. The deployment-mode
axis decouples one-shot graph-init cost (cold) from steady-state
per-iter cost (warm). The schema bumps to ``calibration_model_v4`` and
refuses to load v1/v2/v3 files (no automatic upgrade — re-run
:func:`bootstrap_from_solo_measurements`,
:func:`bootstrap_warm_from_csv_traces`, and
:func:`bootstrap_contention_from_closed_loop`).
"""

from __future__ import annotations

import csv
import json
import statistics
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

CALIBRATION_SCHEMA_VERSION = "calibration_model_v4"
LEGACY_SCHEMA_VERSIONS: tuple[str, ...] = (
    "calibration_model_v1",
    "calibration_model_v2",
    "calibration_model_v3",
)

_DEFAULT_BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")

# Provenance tags for ``contention_factor`` cells. ``"measured"`` means
# the cell was fit from at least one closed-loop observation;
# ``"default_no_data"`` means we have no contended ground truth and
# defaulted to 1.0 (no-op multiplier) so the gap is visible in audits.
PROVENANCE_MEASURED = "measured"
PROVENANCE_DEFAULT_NO_DATA = "default_no_data"

# Deployment-mode tokens (v4). ``cold_start`` matches qnn-net-run-style
# measurements where graph init is included in per-iter wall time;
# ``warm_loop`` matches the realtime_qnn bundle which uses cached
# context binaries + pre-allocated tensor buffers so init cost is
# amortised to zero per iter.
DEPLOYMENT_MODE_COLD = "cold_start"
DEPLOYMENT_MODE_WARM = "warm_loop"

# Canonical deployment-technique tokens. The set used in the realtime_qnn
# bundle (see ``realtime_qnn/REPLICATION.md``) maps to ``warm_loop``;
# any future mode would extend this enum.
TECHNIQUE_CACHED_CONTEXT = "cached_context_binary"
TECHNIQUE_PREALLOC_BUFFERS = "preallocated_tensor_buffers"
TECHNIQUE_NO_FILE_IO = "no_file_io_in_loop"
TECHNIQUE_SCHED_FIFO = "sched_fifo_mlockall"
TECHNIQUE_TIMERFD_ABSTIME = "timerfd_clock_monotonic_abstime"
TECHNIQUE_PER_SENSOR_ROTATION = "per_sensor_input_rotation"
TECHNIQUE_FULL_BUFFER_REWRITE = "full_buffer_rewrite_per_frame"

WARM_TECHNIQUES: frozenset[str] = frozenset(
    {
        TECHNIQUE_CACHED_CONTEXT,
        TECHNIQUE_PREALLOC_BUFFERS,
        TECHNIQUE_NO_FILE_IO,
        TECHNIQUE_SCHED_FIFO,
        TECHNIQUE_TIMERFD_ABSTIME,
    }
)

# CSV → (workload, backend) registry for the bundle traces. Extending
# this is a follow-up when more (workload, backend) traces ship.
_BUNDLE_CSV_REGISTRY: dict[str, tuple[str, str]] = {
    "rt_yolo_f.csv": ("yolov8n", "DSP"),
    "rt_drone_f.csv": ("dronet", "GPU"),
}


def techniques_to_mode(techniques: Iterable[str]) -> str:
    """Map a deployment-technique set to ``cold_start`` or ``warm_loop``.

    v4 simplification: any technique set containing
    :data:`TECHNIQUE_CACHED_CONTEXT` is treated as warm; everything else
    is cold. Later iterations may introduce finer modes (e.g., a
    cached-but-not-pinned mode); the call site only needs to flip the
    string returned here.
    """

    return DEPLOYMENT_MODE_WARM if TECHNIQUE_CACHED_CONTEXT in set(techniques) else DEPLOYMENT_MODE_COLD


def _select_mode_dicts(
    model: CalibrationModel,
    deployment_mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Return ``(overhead, contention, provenance)`` for the requested mode.

    Raises ``ValueError`` for unknown modes; the typed error makes the
    calling-site bug obvious instead of silently falling through to the
    cold-start path.
    """

    if deployment_mode == DEPLOYMENT_MODE_COLD:
        return model.overhead_us, model.contention_factor, model.contention_provenance
    if deployment_mode == DEPLOYMENT_MODE_WARM:
        return (
            model.overhead_us_warm,
            model.contention_factor_warm,
            model.contention_provenance_warm,
        )
    raise ValueError(
        f"deployment_mode must be {DEPLOYMENT_MODE_COLD!r} or "
        f"{DEPLOYMENT_MODE_WARM!r}, got {deployment_mode!r}"
    )


class CalibrationSchemaMismatchError(ValueError):
    """Raised when a calibration file uses a schema this module cannot read."""


@dataclass(frozen=True)
class CalibrationRound:
    """One observation absorbed into the calibration model."""

    round: int
    workload_id: str
    backend: str
    predicted_us: float
    measured_us: float
    per_op_sum_us: float
    delta_overhead_us: float
    timestamp: str


@dataclass(frozen=True)
class CalibrationModel:
    """Per-target two-term calibration state with a deployment-mode axis.

    ``overhead_us[w][b]`` / ``contention_factor[w][b]`` hold the
    *cold-start* constants — what qnn-net-run-style measurements (graph
    init included in per-iter wall time) imply. ``overhead_us_warm`` /
    ``contention_factor_warm`` hold the *warm-loop* constants — what the
    realtime_qnn bundle (cached context binaries, pre-allocated tensors,
    SCHED_FIFO) achieves once init is amortised away.

    A cell is fit from solo / closed-loop E2E and **not** folded into
    per-op costs (see :func:`apply` for why).

    The ``_warm`` maps default to empty so v3-bootstrapped models keep
    working in cold-only contexts; warm callers must seed them with
    :func:`bootstrap_warm_from_csv_traces` or
    :func:`bootstrap_warm_from_measurements` before requesting
    ``deployment_mode="warm_loop"``.

    ``contention_provenance`` (and its ``_warm`` twin) records
    ``"measured"`` / ``"default_no_data"`` per (workload, backend) cell
    so the audit layer can see which factors are bootstrapped vs default.
    """

    schema_version: str
    target_id: str
    overhead_us: dict[str, dict[str, float]]
    contention_factor: dict[str, dict[str, float]]
    history: tuple[CalibrationRound, ...]
    created_at: str
    contention_provenance: dict[str, dict[str, str]] = field(default_factory=dict)
    overhead_us_warm: dict[str, dict[str, float]] = field(default_factory=dict)
    contention_factor_warm: dict[str, dict[str, float]] = field(default_factory=dict)
    contention_provenance_warm: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementRecord:
    """A single (workload, backend) measurement to feed into EMA.

    ``concurrent_workloads`` distinguishes the two regimes the v3 model
    represents: solo (empty tuple — fits ``overhead_us``) vs contended
    (non-empty — fits ``contention_factor``). The EMA target differs
    between regimes; folding both into the same update conflates the
    additive and multiplicative terms (see the v3 trial dry-run).

    ``deployment_techniques`` (v4) names the warm-loop techniques in
    effect during the measurement. The update path routes the EMA to
    the cold or warm dict by inspecting this tuple (via
    :func:`techniques_to_mode`); the empty default keeps v3-shape
    measurements behaving exactly as before.
    """

    workload_id: str
    backend: str
    measured_us: float
    per_op_sum_us: float
    predicted_us: float
    concurrent_workloads: tuple[str, ...] = ()
    deployment_techniques: tuple[str, ...] = ()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _per_op_sum_us(raw_cost_matrix: dict[str, Any], workload: str, backend: str) -> tuple[float, int]:
    """Return (sum_us, covered_op_count) for one (workload, backend)."""

    ops = raw_cost_matrix.get(workload, {})
    total = 0.0
    covered = 0
    for _op_name, costs in ops.items():
        if not isinstance(costs, dict):
            continue
        if backend in costs:
            total += float(costs[backend])
            covered += 1
    return total, covered


def bootstrap_from_solo_measurements(
    raw_cost_matrix: dict,
    e2e_measurements: dict,
    target_id: str = "qrb5165",
    seed_contention: dict[str, dict[str, float]] | None = None,
) -> CalibrationModel:
    """Seed a fresh calibration model from solo whole-network measurements.

    For every (workload ``w``, backend ``b``) pair with a solo whole-net
    E2E measurement and at least one covered per-op cost::

        overhead_us[w][b] = max(0, e2e_solo[w, b] - Σ per_op_costs[w][*][b])

    Workloads/backends without a usable measurement get a 0.0 overhead
    entry (so the cost-matrix view never silently drops them).
    Contention factors seed to 1.0 per (workload, backend) with
    ``"default_no_data"`` provenance unless ``seed_contention`` supplies
    a value (in which case provenance is ``"measured"``).

    Args:
        raw_cost_matrix: ``qnn_cost_matrix_v1``-shaped dict (top-level
            workload keys, each mapping op_id → {backend: us}).
        e2e_measurements: ``qnn_e2e/measurements.json``-shaped dict with
            a top-level ``matrix`` key.
        target_id: Identifier for the hardware target.
        seed_contention: Optional initial contention factors keyed by
            ``[workload_id][backend]``. Missing cells default to 1.0.

    Returns:
        A frozen CalibrationModel with empty history and 1.0 contention
        on every cell (call :func:`bootstrap_contention_from_closed_loop`
        next to fit per-(w, b) contention from real rounds).
    """

    matrix = e2e_measurements.get("matrix", {})
    workloads = [w for w in raw_cost_matrix.keys() if not w.startswith("_")]
    backend_set: set[str] = set()
    for w in workloads:
        for costs in raw_cost_matrix[w].values():
            if isinstance(costs, dict):
                backend_set.update(costs.keys())
    backends = sorted(backend_set) if backend_set else list(_DEFAULT_BACKENDS)

    overhead: dict[str, dict[str, float]] = {}
    for w in workloads:
        per_workload: dict[str, float] = {}
        for b in backends:
            wl_e2e = matrix.get(w, {}).get(b)
            if not wl_e2e or not wl_e2e.get("ok", True):
                per_workload[b] = 0.0
                continue
            measured_us = float(wl_e2e.get("mean_us", 0.0))
            if measured_us <= 0.0:
                per_workload[b] = 0.0
                continue
            per_op_sum, covered = _per_op_sum_us(raw_cost_matrix, w, b)
            if covered == 0:
                per_workload[b] = 0.0
                continue
            per_workload[b] = max(0.0, measured_us - per_op_sum)
        overhead[w] = per_workload

    contention: dict[str, dict[str, float]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for w in workloads:
        per_workload_c: dict[str, float] = {}
        per_workload_p: dict[str, str] = {}
        seed_w = (seed_contention or {}).get(w, {}) if seed_contention else {}
        for b in backends:
            if b in seed_w:
                per_workload_c[b] = float(seed_w[b])
                per_workload_p[b] = PROVENANCE_MEASURED
            else:
                per_workload_c[b] = 1.0
                per_workload_p[b] = PROVENANCE_DEFAULT_NO_DATA
        contention[w] = per_workload_c
        provenance[w] = per_workload_p

    model = CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id=target_id,
        overhead_us=overhead,
        contention_factor=contention,
        history=(),
        created_at=_utc_now_iso(),
        contention_provenance=provenance,
        overhead_us_warm={},
        contention_factor_warm={},
        contention_provenance_warm={},
    )
    log.info(
        "calibration_bootstrap",
        target_id=target_id,
        overhead_us=overhead,
        workloads=workloads,
    )
    return model


def bootstrap_warm_from_measurements(
    model: CalibrationModel,
    warm_measurements: dict[str, Any],
    raw_cost_matrix: dict[str, Any],
) -> CalibrationModel:
    """Seed ``overhead_us_warm`` from a warm-loop measurements JSON.

    Expects the schema produced by post-processing the realtime_qnn
    bundle CSVs (``xpu-rt/data/profiled/qnn_warm/measurements.json``):
    top-level ``matrix`` mapping ``workload → backend → {mean_us,
    p50_us, p99_us, n, ok}``. Per (workload, backend) cell::

        warm_overhead = max(0, p50_us - Σ per_op_costs[w][*][b])

    Cells without a usable measurement are not populated (so the audit
    layer can distinguish "no warm data" from "warm overhead is zero").

    Args:
        model: An existing v4 calibration model (typically the output of
            :func:`bootstrap_from_solo_measurements`).
        warm_measurements: Loaded warm measurements JSON.
        raw_cost_matrix: Same cost matrix used to bootstrap cold; used
            to compute chain-sums.

    Returns:
        A new model with ``overhead_us_warm`` populated. Contention is
        left at the existing (typically empty) warm dict; warm
        contention is a follow-up when we have warm contended ground
        truth.
    """

    matrix = warm_measurements.get("matrix", {})
    warm_overhead: dict[str, dict[str, float]] = {
        wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()
    }
    for w, per_backend in matrix.items():
        if not isinstance(per_backend, dict):
            continue
        warm_overhead.setdefault(w, {})
        for b, cell in per_backend.items():
            if not isinstance(cell, dict) or not cell.get("ok", True):
                continue
            p50 = float(cell.get("p50_us") or cell.get("mean_us") or 0.0)
            if p50 <= 0.0:
                continue
            chain_us, covered = _per_op_sum_us(raw_cost_matrix, w, b)
            if covered == 0:
                continue
            warm_overhead[w][b] = max(0.0, p50 - chain_us)
            log.info(
                "calibration_bootstrap_warm",
                workload=w,
                backend=b,
                p50_us=p50,
                chain_us=chain_us,
                warm_overhead_us=warm_overhead[w][b],
            )
    return CalibrationModel(
        schema_version=model.schema_version,
        target_id=model.target_id,
        overhead_us={wid: dict(per_b) for wid, per_b in model.overhead_us.items()},
        contention_factor={wid: dict(per_b) for wid, per_b in model.contention_factor.items()},
        history=model.history,
        created_at=model.created_at,
        contention_provenance={wid: dict(per_b) for wid, per_b in model.contention_provenance.items()},
        overhead_us_warm=warm_overhead,
        contention_factor_warm={
            wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()
        },
        contention_provenance_warm={
            wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()
        },
    )


def _aggregate_csv_exec_us(
    csv_path: Path,
    *,
    drop_warmup: int,
    aggregator: str,
) -> tuple[float, int]:
    """Return ``(aggregate_us, n_used)`` from a bundle CSV.

    Reads the ``exec_us`` column (per-iter execute wall time) from the
    realtime_qnn CSV format. ``drop_warmup`` skips the first N rows so
    init-warm transients don't pollute steady-state stats.
    """

    rows: list[float] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(float(row["exec_us"]))
            except (KeyError, ValueError, TypeError):
                continue
    if drop_warmup > 0:
        rows = rows[drop_warmup:]
    if not rows:
        return (0.0, 0)
    if aggregator == "mean":
        agg = float(statistics.fmean(rows))
    elif aggregator == "median":
        agg = float(statistics.median(rows))
    else:
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")
    return (agg, len(rows))


def bootstrap_warm_from_csv_traces(
    model: CalibrationModel,
    csv_paths: Iterable[Path | str],
    raw_cost_matrix: dict[str, Any],
    *,
    drop_warmup: int = 5,
    aggregator: str = "median",
) -> CalibrationModel:
    """Seed ``overhead_us_warm`` from realtime_qnn bundle CSV traces.

    Each CSV's filename is looked up in :data:`_BUNDLE_CSV_REGISTRY` to
    resolve ``(workload, backend)``. Per cell::

        warm_overhead = max(0, agg(exec_us) - Σ per_op_costs[w][*][b])

    where ``agg`` is ``median`` (default; robust to tail outliers) or
    ``mean``. The first ``drop_warmup`` iters are skipped to avoid
    JIT/page-fault warmup transients.

    Args:
        model: v4 calibration model to extend.
        csv_paths: Iterable of paths to ``rt_*.csv`` files; unknown
            filenames are skipped with a warning.
        raw_cost_matrix: Cost matrix for chain-sum lookup.
        drop_warmup: Number of leading rows to drop (default ``5``).
        aggregator: ``"mean"`` or ``"median"`` (default ``"median"``).

    Returns:
        A new model with ``overhead_us_warm`` populated for every CSV
        that resolved to a registered ``(workload, backend)`` pair.
    """

    warm_overhead: dict[str, dict[str, float]] = {
        wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()
    }
    for raw_path in csv_paths:
        path = Path(raw_path)
        key = path.name
        if key not in _BUNDLE_CSV_REGISTRY:
            log.warning(
                "calibration_bootstrap_warm_unknown_csv",
                csv=str(path),
                known=list(_BUNDLE_CSV_REGISTRY.keys()),
            )
            continue
        workload, backend = _BUNDLE_CSV_REGISTRY[key]
        agg_us, n_used = _aggregate_csv_exec_us(
            path, drop_warmup=drop_warmup, aggregator=aggregator
        )
        if n_used == 0:
            log.warning(
                "calibration_bootstrap_warm_empty_csv", csv=str(path), workload=workload, backend=backend
            )
            continue
        chain_us, covered = _per_op_sum_us(raw_cost_matrix, workload, backend)
        if covered == 0:
            log.warning(
                "calibration_bootstrap_warm_no_chain",
                csv=str(path),
                workload=workload,
                backend=backend,
            )
            continue
        ovh = max(0.0, agg_us - chain_us)
        warm_overhead.setdefault(workload, {})[backend] = ovh
        log.info(
            "calibration_bootstrap_warm_csv",
            workload=workload,
            backend=backend,
            csv=str(path),
            aggregator=aggregator,
            n_used=n_used,
            agg_us=agg_us,
            chain_us=chain_us,
            warm_overhead_us=ovh,
        )
    return CalibrationModel(
        schema_version=model.schema_version,
        target_id=model.target_id,
        overhead_us={wid: dict(per_b) for wid, per_b in model.overhead_us.items()},
        contention_factor={wid: dict(per_b) for wid, per_b in model.contention_factor.items()},
        history=model.history,
        created_at=model.created_at,
        contention_provenance={wid: dict(per_b) for wid, per_b in model.contention_provenance.items()},
        overhead_us_warm=warm_overhead,
        contention_factor_warm={
            wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()
        },
        contention_provenance_warm={
            wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()
        },
    )


def bootstrap_contention_from_closed_loop(
    model: CalibrationModel,
    rounds_data: list[dict[str, Any]],
    raw_cost_matrix: dict,
    *,
    aggregator: str = "mean",
) -> CalibrationModel:
    """Fit per-(workload, backend) contention from closed-loop rounds.

    For each round, this computes a residual ratio::

        residual_ratio = measured_us / (chain_sum[w][b] + base_overhead[w][b])

    Then groups by ``(workload_id, backend)`` and aggregates with the
    chosen reducer (``"mean"`` or ``"median"``). The model's
    ``contention_factor[w][b]`` is overwritten for every cell that has
    at least one observation; cells without observations are left at
    1.0 with ``"default_no_data"`` provenance. This makes
    contention-data gaps visible to audits rather than papering over
    them with a single global factor.

    Note on counter-intuitive factors: in the QRB5165 closed-loop trace
    the only contended cell with measurements is yolov8n on DSP, and
    its ratio comes out to ~0.79 — under contention yolov8n runs
    *faster* than solo. Likely cause: 12× dronet runs on CPU, leaving
    DSP fully available for yolov8n with possible cache-warmth
    benefits; we report the number honestly because the closed-loop
    measurement is what it is.

    Args:
        model: A v3 calibration model whose ``overhead_us`` is already
            seeded by :func:`bootstrap_from_solo_measurements`.
        rounds_data: List of dicts with at least keys ``workload_id``,
            ``backend``, ``measured_us``. ``predicted_us`` is read but
            not load-bearing (we recompute the base prediction from
            ``chain_sum + overhead_us``).
        raw_cost_matrix: Same cost matrix used to bootstrap the model;
            used to recompute ``chain_sum``.
        aggregator: ``"mean"`` (default) or ``"median"``. Mean is more
            sensitive to round 2's outlier; median is more robust on
            small N. We default to mean for transparency on N=4.

    Returns:
        A new ``CalibrationModel`` with updated ``contention_factor``
        and ``contention_provenance`` cells.
    """

    if aggregator not in ("mean", "median"):
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")

    grouped: dict[str, dict[str, list[float]]] = {}
    for entry in rounds_data:
        w = str(entry["workload_id"])
        b = str(entry["backend"])
        measured_us = float(entry["measured_us"])
        chain_us, _covered = _per_op_sum_us(raw_cost_matrix, w, b)
        base_overhead = float(model.overhead_us.get(w, {}).get(b, 0.0))
        base_pred = chain_us + base_overhead
        if base_pred <= 0.0:
            log.warning(
                "calibration_contention_skip_zero_base",
                workload=w,
                backend=b,
                chain_us=chain_us,
                base_overhead=base_overhead,
            )
            continue
        ratio = measured_us / base_pred
        grouped.setdefault(w, {}).setdefault(b, []).append(ratio)

    new_contention: dict[str, dict[str, float]] = {
        wid: dict(per_b) for wid, per_b in model.contention_factor.items()
    }
    new_provenance: dict[str, dict[str, str]] = {
        wid: dict(per_b) for wid, per_b in model.contention_provenance.items()
    }
    for w, per_b in grouped.items():
        for b, ratios in per_b.items():
            if aggregator == "mean":
                value = float(statistics.fmean(ratios))
            else:
                value = float(statistics.median(ratios))
            new_contention.setdefault(w, {})[b] = value
            new_provenance.setdefault(w, {})[b] = PROVENANCE_MEASURED
            log.info(
                "calibration_contention_fit",
                workload=w,
                backend=b,
                aggregator=aggregator,
                n_observations=len(ratios),
                contention_factor=value,
            )

    return CalibrationModel(
        schema_version=model.schema_version,
        target_id=model.target_id,
        overhead_us={wid: dict(per_b) for wid, per_b in model.overhead_us.items()},
        contention_factor=new_contention,
        history=model.history,
        created_at=model.created_at,
        contention_provenance=new_provenance,
        overhead_us_warm={wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()},
        contention_factor_warm={
            wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()
        },
        contention_provenance_warm={
            wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()
        },
    )


def update_from_measurement(
    model: CalibrationModel,
    measurement: MeasurementRecord,
    ema_alpha: float = 0.5,
) -> CalibrationModel:
    """Apply one EMA update keyed on solo vs contended execution.

    The v3 model represents predicted latency as
    ``(per_op_sum + overhead[w][b]) * contention[w][b]``. Each EMA call
    fits exactly one of the two terms — the one consistent with the
    measurement's ``concurrent_workloads`` regime — and leaves the other
    untouched. Folding both into a single update would conflate the
    additive and multiplicative components (see the dry-run trial that
    revealed this bug).

    * **Solo** (``measurement.concurrent_workloads == ()``):
      ``target_overhead = measured_us - per_op_sum_us``, then
      ``new_overhead = old_overhead + alpha * (target_overhead - old_overhead)``.
      Contention is left unchanged.

    * **Contended** (non-empty tuple):
      ``target_contention = measured_us / (per_op_sum_us + overhead[w][b])``,
      then EMA on contention. Overhead is left unchanged. The set of
      concurrent workloads is recorded in the history entry but does
      not currently key into a per-concurrent-set contention dict;
      that's a follow-up when we have data for more than one
      combination.

    Args:
        model: Current calibration state.
        measurement: One (workload, backend) observation.
        ema_alpha: EMA smoothing factor in ``[0, 1]``. ``0.5`` matches
            the existing closed-loop default.

    Returns:
        A new CalibrationModel with the matching term updated for
        ``(measurement.workload_id, measurement.backend)`` and one
        appended history entry.
    """

    if not 0.0 <= ema_alpha <= 1.0:
        raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")

    w = measurement.workload_id
    b = measurement.backend
    is_solo = len(measurement.concurrent_workloads) == 0
    mode = techniques_to_mode(measurement.deployment_techniques)
    is_warm = mode == DEPLOYMENT_MODE_WARM

    if is_warm:
        overhead_src = model.overhead_us_warm
        contention_src = model.contention_factor_warm
    else:
        overhead_src = model.overhead_us
        contention_src = model.contention_factor

    old_overhead = float(overhead_src.get(w, {}).get(b, 0.0))
    old_contention = float(contention_src.get(w, {}).get(b, 1.0))

    new_overhead = old_overhead
    new_contention = old_contention
    if is_solo:
        target_overhead = measurement.measured_us - measurement.per_op_sum_us
        new_overhead = old_overhead + ema_alpha * (target_overhead - old_overhead)
        new_overhead = max(0.0, new_overhead)
    else:
        base = measurement.per_op_sum_us + old_overhead
        if base > 0.0:
            target_contention = measurement.measured_us / base
            new_contention = old_contention + ema_alpha * (target_contention - old_contention)
            new_contention = max(0.0, new_contention)
        else:
            log.warning(
                "calibration_update_contention_skip_zero_base",
                workload_id=w,
                backend=b,
                per_op_sum_us=measurement.per_op_sum_us,
                old_overhead_us=old_overhead,
                deployment_mode=mode,
            )

    cold_overhead = {wid: dict(per_b) for wid, per_b in model.overhead_us.items()}
    cold_contention = {wid: dict(per_b) for wid, per_b in model.contention_factor.items()}
    cold_provenance = {wid: dict(per_b) for wid, per_b in model.contention_provenance.items()}
    warm_overhead = {wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()}
    warm_contention = {wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()}
    warm_provenance = {wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()}

    if is_warm:
        warm_overhead.setdefault(w, {})[b] = new_overhead
        warm_contention.setdefault(w, {})[b] = new_contention
        if not is_solo:
            warm_provenance.setdefault(w, {})[b] = PROVENANCE_MEASURED
    else:
        cold_overhead.setdefault(w, {})[b] = new_overhead
        cold_contention.setdefault(w, {})[b] = new_contention
        if not is_solo:
            cold_provenance.setdefault(w, {})[b] = PROVENANCE_MEASURED

    round_idx = len(model.history) + 1
    entry = CalibrationRound(
        round=round_idx,
        workload_id=w,
        backend=b,
        predicted_us=measurement.predicted_us,
        measured_us=measurement.measured_us,
        per_op_sum_us=measurement.per_op_sum_us,
        delta_overhead_us=new_overhead - old_overhead,
        timestamp=_utc_now_iso(),
    )

    log.info(
        "calibration_update",
        target_id=model.target_id,
        workload_id=w,
        backend=b,
        regime="solo" if is_solo else "contended",
        deployment_mode=mode,
        old_overhead_us=old_overhead,
        new_overhead_us=new_overhead,
        old_contention=old_contention,
        new_contention=new_contention,
        ema_alpha=ema_alpha,
    )
    return CalibrationModel(
        schema_version=model.schema_version,
        target_id=model.target_id,
        overhead_us=cold_overhead,
        contention_factor=cold_contention,
        history=model.history + (entry,),
        created_at=model.created_at,
        contention_provenance=cold_provenance,
        overhead_us_warm=warm_overhead,
        contention_factor_warm=warm_contention,
        contention_provenance_warm=warm_provenance,
    )


def compose_predicted_makespan_us(
    *,
    model: CalibrationModel,
    workload_id: str,
    per_lane_busy_us: dict[str, float],
    concurrent_workloads: tuple[str, ...] = (),
    deployment_mode: str = DEPLOYMENT_MODE_COLD,
) -> float:
    """Combine per-lane raw busy time with the v3 two-term calibration.

    The v3 model says ``predicted = (per_op_sum + overhead) * contention``
    *per lane*. The solver gives us ``per_lane_busy_us`` already (sum of
    chunk durations on each lane it landed). This function applies the
    overhead and contention per-lane, then returns the makespan as
    ``max`` over lanes — which is what an unsynchronised, lane-parallel
    execution actually finishes at.

    Solo mode (``concurrent_workloads == ()``) forces contention = 1.0
    regardless of the stored factor, because the stored factor was
    bootstrapped from contended runs and applying it to a solo
    schedule would silently mis-predict (the original Bug 1).

    Args:
        model: Calibration model.
        workload_id: The workload whose schedule we're scoring.
        per_lane_busy_us: ``{backend: sum_of_chunk_durations}`` for every
            backend the schedule uses.
        concurrent_workloads: Other workloads expected to run alongside.
            Empty tuple = solo.

    Returns:
        Predicted makespan (us). ``0.0`` if no lanes were used.
    """

    if not per_lane_busy_us:
        return 0.0
    overhead_map, contention_map, _ = _select_mode_dicts(model, deployment_mode)
    overhead = overhead_map.get(workload_id, {})
    contention = contention_map.get(workload_id, {})
    is_solo = len(concurrent_workloads) == 0
    finish_per_lane: list[float] = []
    for backend, busy in per_lane_busy_us.items():
        ovh = float(overhead.get(backend, 0.0))
        ct = 1.0 if is_solo else float(contention.get(backend, 1.0))
        finish_per_lane.append((busy + ovh) * ct)
    return max(finish_per_lane) if finish_per_lane else 0.0


def overhead_for(model: CalibrationModel, workload_id: str) -> dict[str, float]:
    """Return the per-backend overhead dict for ``workload_id`` (empty if absent)."""

    return dict(model.overhead_us.get(workload_id, {}))


def contention_for(model: CalibrationModel, workload_id: str) -> dict[str, float]:
    """Return the per-backend contention factors for ``workload_id`` (empty if absent)."""

    return dict(model.contention_factor.get(workload_id, {}))


def apply(
    model: CalibrationModel,
    raw_cost_matrix: dict,
    workload_id: str | None = None,
    *,
    deployment_mode: str = DEPLOYMENT_MODE_COLD,
) -> dict:
    """Project the v3 two-term calibration onto a cost matrix view.

    The returned per-op cost matrix is **unchanged** — neither overhead
    nor contention is folded into per-op cells, because doing so would
    double-count overhead every time the scheduler sums per-op costs
    across a partition. Instead, the per-(workload, backend) overhead
    and contention factors for ``workload_id`` are exposed under top-
    level keys::

        ``_calibration_overhead_us[backend]``         — base overhead (us)
        ``_calibration_contention_factor[backend]``   — multiplier (unitless)

    so the scheduler can compute, once per partition::

        partition_cost = (Σ per_op + overhead[b]) * contention[b]

    The ``_per_workload`` variants expose the full cross-workload maps
    for callers that want to plan multiple workloads in one pass.

    Backwards compatibility: callers that do not pass ``workload_id``
    get a zero-overhead, unit-contention dict and a
    ``DeprecationWarning`` — this is safer than picking some arbitrary
    workload's calibration.

    Args:
        model: Calibration state to apply.
        raw_cost_matrix: ``qnn_cost_matrix_v1``-shaped dict.
        workload_id: Workload whose calibration should be injected
            under the flat ``_calibration_*`` keys. If omitted, zeros
            are emitted with a deprecation warning.

    Returns:
        A new dict; the input is not mutated.
    """

    out: dict[str, Any] = {}
    for key, value in raw_cost_matrix.items():
        if key == "_meta":
            out[key] = value
            continue
        if not isinstance(value, dict):
            out[key] = value
            continue
        # v3 leaves per-op costs untouched; calibration is applied once
        # per partition by the scheduler using the top-level keys below.
        scaled_ops: dict[str, Any] = {}
        for op_id, costs in value.items():
            if not isinstance(costs, dict):
                scaled_ops[op_id] = costs
                continue
            scaled_ops[op_id] = {b: float(us) for b, us in costs.items()}
        out[key] = scaled_ops

    overhead_map, contention_map, _ = _select_mode_dicts(model, deployment_mode)

    per_workload_overhead: dict[str, dict[str, float]] = {
        wid: dict(per_b) for wid, per_b in overhead_map.items()
    }
    per_workload_contention: dict[str, dict[str, float]] = {
        wid: dict(per_b) for wid, per_b in contention_map.items()
    }
    out["_calibration_overhead_us_per_workload"] = per_workload_overhead
    out["_calibration_contention_factor_per_workload"] = per_workload_contention
    out["_calibration_deployment_mode"] = deployment_mode

    if workload_id is None:
        warnings.warn(
            "calibration.apply() called without workload_id under schema "
            f"{CALIBRATION_SCHEMA_VERSION}; emitting zero per-backend "
            "overhead and unit contention. Pass workload_id to inject "
            "the correct constants.",
            DeprecationWarning,
            stacklevel=2,
        )
        backends: set[str] = set()
        for per_b in overhead_map.values():
            backends.update(per_b.keys())
        for per_b in contention_map.values():
            backends.update(per_b.keys())
        out["_calibration_overhead_us"] = {b: 0.0 for b in sorted(backends)}
        out["_calibration_contention_factor"] = {b: 1.0 for b in sorted(backends)}
    else:
        out["_calibration_overhead_us"] = dict(overhead_map.get(workload_id, {}))
        out["_calibration_contention_factor"] = dict(
            contention_map.get(workload_id, {})
        )
    return out


def _round_to_dict(r: CalibrationRound) -> dict[str, Any]:
    return asdict(r)


def _dict_to_round(d: dict[str, Any]) -> CalibrationRound:
    return CalibrationRound(**d)


def _coerce_per_backend_float_map(
    raw: dict[str, Any],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for wid, per_b in raw.items():
        if not isinstance(per_b, dict):
            continue
        out[str(wid)] = {str(k): float(v) for k, v in per_b.items()}
    return out


def _coerce_per_backend_str_map(
    raw: dict[str, Any],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for wid, per_b in raw.items():
        if not isinstance(per_b, dict):
            continue
        out[str(wid)] = {str(k): str(v) for k, v in per_b.items()}
    return out


def save(model: CalibrationModel, path: Path) -> None:
    """Persist a calibration model as JSON (indent=2 for stable diffs)."""

    payload: dict[str, Any] = {
        "schema_version": model.schema_version,
        "target_id": model.target_id,
        "overhead_us": {
            wid: dict(per_b) for wid, per_b in model.overhead_us.items()
        },
        "contention_factor": {
            wid: dict(per_b) for wid, per_b in model.contention_factor.items()
        },
        "contention_provenance": {
            wid: dict(per_b) for wid, per_b in model.contention_provenance.items()
        },
        "overhead_us_warm": {
            wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()
        },
        "contention_factor_warm": {
            wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()
        },
        "contention_provenance_warm": {
            wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()
        },
        "history": [_round_to_dict(r) for r in model.history],
        "created_at": model.created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("calibration_saved", path=str(path), target_id=model.target_id)


def load(path: Path) -> CalibrationModel:
    """Load a calibration model from JSON.

    Raises:
        CalibrationSchemaMismatchError: If ``schema_version`` does not
            match :data:`CALIBRATION_SCHEMA_VERSION`. v1/v2/v3 files
            cannot be auto-upgraded — re-run
            :func:`bootstrap_from_solo_measurements`,
            :func:`bootstrap_warm_from_csv_traces` /
            :func:`bootstrap_warm_from_measurements`, then
            :func:`bootstrap_contention_from_closed_loop` to regenerate
            with the deployment-mode axis populated.
    """

    payload = json.loads(Path(path).read_text())
    version = payload.get("schema_version")
    if version != CALIBRATION_SCHEMA_VERSION:
        if version in LEGACY_SCHEMA_VERSIONS:
            raise CalibrationSchemaMismatchError(
                f"calibration schema {version!r} at {path} is legacy; "
                f"v4 adds a deployment-mode axis (cold_start vs warm_loop). "
                f"Re-run bootstrap_from_solo_measurements() + "
                f"bootstrap_warm_from_csv_traces() (or "
                f"bootstrap_warm_from_measurements()) + "
                f"bootstrap_contention_from_closed_loop() against the "
                f"profiled cost matrix + qnn_e2e/measurements.json + "
                f"qnn_warm/measurements.json (or realtime_qnn CSVs) + "
                f"qnn_closed_loop rounds to regenerate this file."
            )
        raise CalibrationSchemaMismatchError(
            f"calibration schema mismatch at {path}: expected "
            f"{CALIBRATION_SCHEMA_VERSION}, got {version!r}"
        )
    raw_overhead = payload["overhead_us"]
    overhead: dict[str, dict[str, float]] = {}
    for wid, per_b in raw_overhead.items():
        if not isinstance(per_b, dict):
            raise CalibrationSchemaMismatchError(
                f"overhead_us[{wid!r}] must be a per-backend dict in v4, "
                f"got {type(per_b).__name__}"
            )
        overhead[str(wid)] = {str(k): float(v) for k, v in per_b.items()}
    raw_contention = payload["contention_factor"]
    contention: dict[str, dict[str, float]] = {}
    for wid, per_b in raw_contention.items():
        if not isinstance(per_b, dict):
            raise CalibrationSchemaMismatchError(
                f"contention_factor[{wid!r}] must be a per-backend dict in v4, "
                f"got {type(per_b).__name__} (v2 used a flat per-backend "
                "dict; this file is likely v2 — rerun bootstrap)."
            )
        contention[str(wid)] = {str(k): float(v) for k, v in per_b.items()}
    provenance = _coerce_per_backend_str_map(payload.get("contention_provenance", {}))
    overhead_warm = _coerce_per_backend_float_map(payload.get("overhead_us_warm", {}))
    contention_warm = _coerce_per_backend_float_map(payload.get("contention_factor_warm", {}))
    provenance_warm = _coerce_per_backend_str_map(payload.get("contention_provenance_warm", {}))
    history = tuple(_dict_to_round(r) for r in payload.get("history", []))
    return CalibrationModel(
        schema_version=version,
        target_id=str(payload["target_id"]),
        overhead_us=overhead,
        contention_factor=contention,
        history=history,
        created_at=str(payload["created_at"]),
        contention_provenance=provenance,
        overhead_us_warm=overhead_warm,
        contention_factor_warm=contention_warm,
        contention_provenance_warm=provenance_warm,
    )
