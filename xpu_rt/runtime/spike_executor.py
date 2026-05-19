"""Spike + Gemmini/Saturn-OPU substrate for the XpuRtExecutor protocol.

Consumes an XPU-RT bundle directory (``payload.mlir``,
``kernel_contracts/*.yaml``, ``generated_kernels/<provider>/...``) and
runs the matmul regions on :mod:`xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv.CRiscvEvaluator`.
``CRiscvEvaluator`` already handles the heavy lifting (riscv64
cross-compile, spike subprocess, parsing ``cycles=`` /
``mismatches=N/M`` from harness stdout). This module is a thin
adapter that:

  1. Walks ``kernel_contracts/*.yaml`` in the bundle.
  2. For each region with a generated C kernel, materializes the
     contract as a :class:`xpu_rt.kernels.provider.KernelContract`
     and asks ``CRiscvEvaluator`` to evaluate it.
  3. Aggregates the per-region results into a
     :class:`~xpu_rt.runtime.measured_cost.MeasuredCost` and writes
     it to ``<bundle_dir>/measured_cost.json``.
  4. When a ``run_dir`` is supplied, also writes
     ``<run_dir>/02_graph_analysis/kernel_execution/kernel_execution_report.json``
     so :mod:`xpu_rt.promotion.gates` advances the
     ``verified_kernel`` rung from the same evidence.

Scope today (Phase 1 MVP):

  * Op families: ``matmul`` only — that's the only family
    ``CRiscvEvaluator`` ships with a harness for. Other regions are
    marked ``skipped`` with a reason; they do not turn the executor
    into a failure.
  * Dtypes: int8 × int8 → int32 — the matmul harness's static
    flavour. f32 / f16 / bf16 matmuls land as ``skipped`` with a
    reason rather than a synthetic int8 reinterpretation. The intent
    is honest evidence, not coverage padding.
  * Substrates: ``gemmini`` (RoCC custom-3) and
    ``saturn_opu_v128`` (RVV 1.0 + zvl128b) — both already wired in
    ``CRiscvEvaluator`` via ``target_id``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
    CRiscvEvaluator,
    _check_toolchain,
    _ToolchainMissing,
)
from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse
from xpu_rt.kernels.provider import KernelContract as ProviderContract
from xpu_rt.runtime.errors import AdapterUnavailableError
from xpu_rt.runtime.measured_cost import (
    MeasuredCost,
    MeasurementSample,
    aggregate_correctness,
)

logger = logging.getLogger(__name__)


_MATMUL_FAMILIES: frozenset[str] = frozenset({
    "matmul",
    "mm",
    "gemm",
    "bmm",
    "linear",
    "linalg.matmul",
    "linalg_matmul",
    "aten_matmul",
    "aten_mm",
    "aten_addmm",
    "aten_linear",
})


_INT8_DTYPES: frozenset[str] = frozenset({"i8", "int8"})


@dataclass(frozen=True)
class _ContractEntry:
    """One ``kernel_contracts/*.yaml`` parsed into the fields the
    executor needs to bridge to ``ProviderContract``."""

    path: Path
    op_name: str
    op_family: str
    region_id: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]


@dataclass
class SpikeExecutor:
    """Run an XPU-RT bundle on Spike + (Gemmini | Saturn/OPU).

    Args:
        target_id: ``"gemmini"`` (default) or any id starting with
            ``"saturn"`` / ``"opu"``. Plumbs straight through to
            ``CRiscvEvaluator.target_id``.
        require_gemmini_extension: When True, fails over to
            ``skipped`` for matmul regions if the active ``spike``
            binary lacks the Gemmini extension. Set False for tests
            that exercise the RVV-only path.
        timeout_s: Per-region cross-compile + spike wall-clock cap.
        keep_workdir: Keep CRiscvEvaluator's tmp dir for postmortem.
    """

    target_id: str = "gemmini"
    require_gemmini_extension: bool = True
    timeout_s: int = 60
    keep_workdir: bool = False
    name: str = field(init=False)

    def __post_init__(self) -> None:
        tid = self.target_id.lower()
        if tid.startswith("saturn") or tid.startswith("opu"):
            object.__setattr__(self, "name", f"spike_{tid}")
        else:
            object.__setattr__(self, "name", "spike_gemmini")

    # ---- protocol ---------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        try:
            _check_toolchain()
        except _ToolchainMissing as exc:
            return False, str(exc)
        return True, ""

    def execute(
        self,
        bundle_dir: Path,
        *,
        run_dir: Path | None = None,
        sample_inputs: tuple[Any, ...] | None = None,
    ) -> MeasuredCost:
        bundle_dir = Path(bundle_dir)
        if not bundle_dir.is_dir():
            raise FileNotFoundError(
                f"SpikeExecutor.execute: bundle_dir does not exist: {bundle_dir}"
            )

        available, reason = self.is_available()
        if not available:
            raise AdapterUnavailableError(self.name, reason)

        contracts = _load_contracts(bundle_dir)
        kernel_sources = _index_generated_kernels(bundle_dir)

        samples: list[MeasurementSample] = []
        for entry in contracts:
            samples.append(self._run_one(entry, kernel_sources))

        cycles_total = _sum_or_none(s.cycles for s in samples)
        latency_total = _sum_or_none(s.latency_us_p50 for s in samples)
        correctness = aggregate_correctness(samples)

        notes = ""
        if not contracts:
            notes = "no kernel_contracts/*.yaml found in bundle"
        elif not any(s.correctness == "pass" for s in samples):
            notes = (
                "no region produced a passing run — see samples[*].extras "
                "for per-region reason"
            )

        measured = MeasuredCost(
            executor=self.name,
            hardware_key=self.target_id,
            target_id=self.target_id,
            cycles_total=cycles_total,
            latency_us_p50_total=latency_total,
            correctness_vs_eager=correctness,
            samples=tuple(samples),
            run_id=bundle_dir.name,
            notes=notes,
        )

        measured.write_json(bundle_dir / "measured_cost.json")
        if run_dir is not None:
            _emit_kernel_execution_report(Path(run_dir), measured)

        return measured

    # ---- internal ---------------------------------------------------------

    def _run_one(
        self,
        entry: _ContractEntry,
        kernel_sources: dict[str, Path],
    ) -> MeasurementSample:
        if entry.op_family not in _MATMUL_FAMILIES:
            return MeasurementSample(
                region_id=entry.region_id,
                op_family=entry.op_family,
                correctness="skipped",
                extras={
                    "reason": "op_family_not_supported",
                    "supported": sorted(_MATMUL_FAMILIES),
                },
            )

        dtype_hint = tuple(d.lower() for d in entry.dtypes) or ()
        if dtype_hint and not _INT8_DTYPES.intersection(dtype_hint):
            return MeasurementSample(
                region_id=entry.region_id,
                op_family=entry.op_family,
                correctness="skipped",
                extras={
                    "reason": "dtype_not_supported",
                    "supported": sorted(_INT8_DTYPES),
                    "dtypes": list(dtype_hint),
                },
            )

        kernel_path = _find_kernel_source(entry, kernel_sources)
        if kernel_path is None:
            return MeasurementSample(
                region_id=entry.region_id,
                op_family=entry.op_family,
                correctness="skipped",
                extras={
                    "reason": "no_generated_kernel",
                    "looked_for": list(_kernel_source_keys(entry)),
                },
            )

        kernel_code = kernel_path.read_text()
        # CRiscvEvaluator hard-codes the matmul harness to int8 ×
        # int8 → int32, so we override the dtype here when the
        # contract only listed f32 but is otherwise a valid matmul
        # shape. The point of this executor is to give the int8
        # Gemmini path *something* measurable; honest non-int8
        # contracts already bail above.
        contract = ProviderContract(
            region_id=entry.region_id or entry.op_name,
            op_family="matmul",
            input_shapes=entry.input_shapes,
            output_shapes=entry.output_shapes,
            dtypes=("i8",) if not dtype_hint else entry.dtypes,
            target_name=self.target_id,
            hardware_key=self.target_id,
        )
        evaluator = CRiscvEvaluator(
            contract=contract,
            target_id=self.target_id,
            require_gemmini_extension=self.require_gemmini_extension,
            timeout_s=self.timeout_s,
            keep_workdir=self.keep_workdir,
        )
        candidate = ProposeResponse(kernel_code=kernel_code, action="bundle_kernel")
        try:
            report = evaluator.evaluate(candidate)
        except NotImplementedError as exc:
            return MeasurementSample(
                region_id=entry.region_id,
                op_family=entry.op_family,
                correctness="skipped",
                extras={"reason": "harness_unsupported", "detail": str(exc)},
            )

        total = _total_output_elements(entry.output_shapes)
        meta = dict(report.metadata or {})
        mismatches = _as_int(meta.get("mismatches"))
        if report.correct:
            outcome = "pass"
        elif "reason" in meta and meta["reason"] in {
            "compile_failed",
            "spike_timeout",
            "toolchain_missing",
            "no_gemmini_extension",
            "harness_unsupported_op_family",
        }:
            outcome = "error"
        else:
            outcome = "fail"

        extras: dict[str, Any] = {
            "kernel_source": str(kernel_path.relative_to(kernel_path.parents[0]))
            if kernel_path.parents[2:]
            else str(kernel_path),
            "score": report.score,
            "diff_summary": report.diff_summary,
        }
        if report.compile_log:
            extras["compile_log_tail"] = report.compile_log[-500:]
        if report.runtime_log:
            extras["runtime_log_tail"] = report.runtime_log[-500:]
        for k, v in meta.items():
            extras.setdefault(k, v)

        return MeasurementSample(
            region_id=entry.region_id,
            op_family=entry.op_family,
            cycles=report.cycles,
            correctness=outcome,
            mismatches=mismatches,
            total_elements=total,
            extras=extras,
        )


# --------------------------------------------------------------------------- #
# Bundle introspection helpers
# --------------------------------------------------------------------------- #


def _load_contracts(bundle_dir: Path) -> list[_ContractEntry]:
    contracts_dir = bundle_dir / "kernel_contracts"
    if not contracts_dir.is_dir():
        return []

    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:  # pragma: no cover — declared dep.
        raise

    entries: list[_ContractEntry] = []
    for path in sorted(contracts_dir.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning(
                "spike_executor: skipping malformed contract %s: %s", path, exc
            )
            continue
        op_name = str(doc.get("op_name", "") or "")
        meta = doc.get("metadata") or {}
        region_id = str(meta.get("region_id") or "")
        input_shapes = _as_shape_tuple(doc.get("input_shapes") or ())
        output_shapes = _as_shape_tuple(doc.get("output_shapes") or ())
        dtypes = tuple(str(d) for d in (doc.get("supported_dtypes") or ()))
        entries.append(
            _ContractEntry(
                path=path,
                op_name=op_name,
                op_family=_infer_op_family(op_name),
                region_id=region_id,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                dtypes=dtypes,
            )
        )
    return entries


def _index_generated_kernels(bundle_dir: Path) -> dict[str, Path]:
    """Build a name → file map for everything under ``generated_kernels/``.

    Bundle layout:
      ``generated_kernels/<provider>/<region_id>_<op_name>.<ext>`` for
      single-file kernels, or
      ``generated_kernels/<provider>/<region_id>_<op_name>/<files>``
      for multi-file kernels — we point at the primary file listed in
      ``index.json``.

    Returns a map keyed by lowercased ``<region_id>_<op_name>`` and
    by ``<op_name>`` (with and without provider prefix) so the
    contract → kernel join is permissive.
    """
    gk_dir = bundle_dir / "generated_kernels"
    out: dict[str, Path] = {}
    if not gk_dir.is_dir():
        return out

    index_path = gk_dir / "index.json"
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []
        if isinstance(entries, list):
            for rec in entries:
                if not isinstance(rec, dict):
                    continue
                rel = rec.get("path")
                if not isinstance(rel, str):
                    continue
                file_path = gk_dir / rel
                if not file_path.is_file():
                    continue
                stems = _kernel_index_keys(rec)
                for stem in stems:
                    out.setdefault(stem.lower(), file_path)

    # Fallback: glob every C/cu/cpp source so we catch kernels not in
    # the index (some pipelines emit kernels without updating index.json).
    for path in gk_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".c", ".cu", ".cpp", ".cxx"}:
            continue
        out.setdefault(path.stem.lower(), path)

    return out


def _kernel_index_keys(rec: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    region = str(rec.get("region_id") or "")
    op = str(rec.get("op_name") or "")
    if region and op:
        keys.append(f"{region}_{op}")
    if op:
        keys.append(op)
    path = str(rec.get("path") or "")
    if path:
        keys.append(Path(path).stem)
    return keys


def _find_kernel_source(
    entry: _ContractEntry,
    kernel_sources: dict[str, Path],
) -> Path | None:
    for key in _kernel_source_keys(entry):
        path = kernel_sources.get(key.lower())
        if path is not None:
            return path
    return None


def _kernel_source_keys(entry: _ContractEntry) -> tuple[str, ...]:
    safe_op = _safe_stem(entry.op_name)
    safe_region = _safe_stem(entry.region_id)
    keys: list[str] = []
    if safe_region:
        keys.append(f"{safe_region}_{safe_op}")
    keys.append(safe_op)
    return tuple(keys)


def _safe_stem(raw: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw)


def _infer_op_family(op_name: str) -> str:
    name = (op_name or "").lower()
    if not name:
        return ""
    if name in _MATMUL_FAMILIES:
        return name
    # Strip common ATen / linalg prefixes + suffixes.
    base = name
    for prefix in ("aten_", "aten.", "linalg_", "linalg."):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for suffix in ("_default", "_tensor", "_scalar"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # Drop numeric suffixes appended by bundle_emit for collision
    # avoidance (e.g. ``linalg_matmul_4`` → ``matmul``).
    while base and base[-1].isdigit():
        base = base[:-1]
    base = base.rstrip("_")
    if base in _MATMUL_FAMILIES:
        return base
    return base


def _as_shape_tuple(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[tuple[int, ...]] = []
    for shape in value:
        if isinstance(shape, (list, tuple)):
            try:
                out.append(tuple(int(d) for d in shape))
            except (TypeError, ValueError):
                continue
    return tuple(out)


def _total_output_elements(shapes: tuple[tuple[int, ...], ...]) -> int | None:
    if not shapes:
        return None
    total = 0
    for shape in shapes:
        prod = 1
        for d in shape:
            if d <= 0:
                return None
            prod *= int(d)
        total += prod
    return total


def _sum_or_none(values: Any) -> int | None:
    collected: list[int] = []
    for v in values:
        if v is None:
            return None
        try:
            collected.append(int(v))
        except (TypeError, ValueError):
            return None
    if not collected:
        return None
    return sum(collected)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Promotion-gate evidence emission
# --------------------------------------------------------------------------- #


def _emit_kernel_execution_report(run_dir: Path, measured: MeasuredCost) -> Path:
    """Write the report :mod:`xpu_rt.promotion.gates` consumes.

    The gate at ``_check_verified_kernel`` reads
    ``02_graph_analysis/kernel_execution/kernel_execution_report.json``
    and pivots on ``status`` / ``overall``. We write that file so a
    successful Spike run unlocks the ``verified_kernel`` rung from the
    same evidence the bundle's ``measured_cost.json`` records.
    """
    out_dir = run_dir / "02_graph_analysis" / "kernel_execution"
    out_dir.mkdir(parents=True, exist_ok=True)

    status_map = {"pass": "pass", "fail": "fail", "error": "fail", "skipped": "skipped"}
    payload: dict[str, Any] = {
        "schema_version": "kernel_execution_report_v1",
        "executor": measured.executor,
        "target_id": measured.target_id,
        "hardware_key": measured.hardware_key,
        "status": status_map.get(measured.correctness_vs_eager, "skipped"),
        "overall": status_map.get(measured.correctness_vs_eager, "skipped"),
        "cycles_total": measured.cycles_total,
        "latency_us_p50_total": measured.latency_us_p50_total,
        "samples": [s.as_dict() for s in measured.samples],
        "run_id": measured.run_id,
        "notes": measured.notes,
    }
    report_path = out_dir / "kernel_execution_report.json"
    report_path.write_text(json.dumps(payload, indent=2))
    return report_path


__all__ = ["SpikeExecutor"]
