"""End-to-end glue differential (, paper-facing).

Phase C drive the emitted plan executor with golden
inputs, compare its output against the eager PyTorch reference
(torch.matmul), and emit
``06_glue_emit/glue_differential_report.json``. Wires into
downstream-retry: a failing differential triggers the typed retry
surface so the outer agent reconsiders the candidate.

This is the paper-facing milestone. Once green:

  CompGen emits per-workload glue from a validated execution plan.
  The generated executor calls verified shape-specialized kernels,
  checks plan invariants at launch, and passes end-to-end
  differential testing against the original model.

The differential uses 's layered check:

  - For declared bit_equality: tiled output must equal eager output
    bit-for-bit (max_abs_error == 0 AND max_rel_error == 0).
  - For declared tolerance_eps: ``|sim - eager| <= 4 * K * eps *
    max|A| * max|B|`` (Higham's bound, derived per-case).

Kernel callables:

  wires the *eager-fallback* kernel callable for each region —
  ``torch.matmul(A, B)`` for set_tile_params on a matmul. + widens
  to compiled kernels via the cffi-C reference + Triton template
  paths. The eager-fallback is honest because the verifier
  already passed on the contract (shape, dtype, layout, accumulator),
  and the differential we're applying here is exactly the
  layered check the contract claims.

  When the operator passes a custom ``kernel_resolver`` (+ wires
  this), it overrides the eager fallback per region.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


_REPORT_SCHEMA_VERSION = "glue_differential_report_v1"
_DEFAULT_NUM_CASES = 4


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return None


def _import_executor(executor_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"_glue_diff_executor_{executor_path.parent.name}",
        executor_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _eager_fallback_kernel_for_region(
    *, contract: dict[str, Any],
) -> Callable[..., Any]:
    """Build the eager kernel callable for one region from its
    contract. Today: matmul archetype only. Returns
    ``torch.matmul(A, B)`` regardless of tile sizes (documents
    that the differential check anchors to the contract's claimed
    refinement, not to the tile-K reference).
    """
    archetype = contract.get("archetype", "")
    if archetype != "compute_tiled":
        # Other archetypes are deferred to .x.
        def _not_yet_supported(*args, **kwargs):
            raise RuntimeError(
                f"glue_differential: archetype {archetype!r} is not yet "
                f"supported by the eager-fallback kernel resolver"
            )
        return _not_yet_supported

    def _eager_matmul(*args, **kwargs):
        # Args 0 and 1 are A and B per the contract's io.inputs order.
        import torch
        if len(args) < 2:
            raise RuntimeError(
                f"glue_differential: matmul kernel needs ≥2 inputs, got "
                f"{len(args)}"
            )
        return torch.matmul(args[0], args[1])
    return _eager_matmul


def _tiled_kernel_for_region(
    *, contract: dict[str, Any],
) -> Callable[..., Any]:
    """Build a TILED kernel callable that uses the
    ``_tiled_matmul_eval`` simulator with the contract's tile sizes.

    This is what the differential check is most useful with: tiled vs
    eager. For K_iters=1 it's bit-equal; for K_iters>1 it's within
    Higham's bound.
    """
    archetype = contract.get("archetype", "")
    if archetype != "compute_tiled":
        return _eager_fallback_kernel_for_region(contract=contract)

    # Pull tile sizes from contract.io.attributes.
    attrs = {a["name"]: a["value"] for a in (contract.get("io") or {}).get("attributes") or []}
    tile_M = int(attrs.get("tile_M", 0) or 0)
    tile_N = int(attrs.get("tile_N", 0) or 0)
    tile_K = int(attrs.get("tile_K", 0) or 0)
    if tile_M <= 0 or tile_N <= 0 or tile_K <= 0:
        return _eager_fallback_kernel_for_region(contract=contract)

    from compgen.graph_compilation.real_transform_differential import (
        _tiled_matmul_eval,
    )

    def _tiled(*args, **kwargs):
        if len(args) < 2:
            raise RuntimeError(
                f"glue_differential: tiled-matmul kernel needs ≥2 inputs, "
                f"got {len(args)}"
            )
        return _tiled_matmul_eval(
            args[0], args[1],
            tile_M=tile_M, tile_N=tile_N, tile_K=tile_K,
        )
    return _tiled


def _generate_cases_from_contract(
    *, contract: dict[str, Any], num_cases: int = _DEFAULT_NUM_CASES,
) -> list[tuple[Any, ...]]:
    """Generate inputs matching the contract's shapes + dtypes."""
    import torch
    cases: list[tuple[Any, ...]] = []
    for seed in range(num_cases):
        torch.manual_seed(seed)
        case: list[Any] = []
        for t in (contract.get("io") or {}).get("inputs") or []:
            dims = list(t["shape"]["dims"])
            dtype_name = t["dtype_class"][0]
            torch_dtype = {
                "f32": torch.float32, "fp32": torch.float32,
                "f16": torch.float16, "fp16": torch.float16,
                "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                "f64": torch.float64,
            }.get(dtype_name, torch.float32)
            case.append(torch.randn(*dims, dtype=torch_dtype))
        cases.append(tuple(case))
    return cases


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    region_id: str
    status: str  # "pass" | "fail"
    max_abs_error: float
    max_rel_error: float
    higham_bound: float
    declared_refinement: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "region_id": self.region_id,
            "status": self.status,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "higham_bound": self.higham_bound,
            "declared_refinement": self.declared_refinement,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GlueDifferentialResult:
    out_dir: Path
    report_path: Path
    status: str  # "pass" | "fail" | "skipped"
    refinement_status: str  # discharged_bit_equality | discharged_tolerance_eps | fail_*
    cases_passed: int
    cases_total: int
    failure_summary: str = ""
    case_records: tuple[CaseRecord, ...] = field(default_factory=tuple)


def run_glue_differential(
    run_dir: Path,
    *,
    num_cases: int = _DEFAULT_NUM_CASES,
    kernel_resolver: Callable[[dict[str, Any]], Callable[..., Any]] | None = None,
) -> GlueDifferentialResult:
    """Drive the emitted executor with synthesized cases and
    compare its output against eager torch.matmul. Apply 's
    layered check. Emit the report.

    ``kernel_resolver`` is per-region: given the contract dict
    for one region, returns the kernel callable to use. Defaults to
    the tiled matmul evaluator (so bit_equality holds when
    K_iters=1 for the contract's tile, and tolerance_eps within
    Higham's bound otherwise).
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "06_glue_emit"
    report_path = out_dir / "glue_differential_report.json"
    executor_path = out_dir / "generated_plan_executor.py"
    if not executor_path.exists():
        body = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "status": "skipped",
            "refinement_status": "skipped",
            "failure_summary": (
                f"M-47 emitted executor not found at "
                f"{executor_path.relative_to(run_dir)}; run "
                f"--stop-after glue-emit first"
            ),
            "cases_total": 0, "cases_passed": 0, "case_records": [],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return GlueDifferentialResult(
            out_dir=out_dir, report_path=report_path,
            status="skipped",
            refinement_status="skipped",
            cases_passed=0, cases_total=0,
            failure_summary=body["failure_summary"],
        )

    # Load plan + bindings.
    plan_path_yaml = run_dir / "05_execution_plan" / "execution_plan.yaml"
    plan_path_json = run_dir / "05_execution_plan" / "execution_plan.json"
    plan_path = plan_path_yaml if plan_path_yaml.exists() else plan_path_json
    if not plan_path.exists():
        body = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "status": "skipped",
            "refinement_status": "skipped",
            "failure_summary": "M-46 execution plan not found",
            "cases_total": 0, "cases_passed": 0, "case_records": [],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return GlueDifferentialResult(
            out_dir=out_dir, report_path=report_path,
            status="skipped", refinement_status="skipped",
            cases_passed=0, cases_total=0,
            failure_summary=body["failure_summary"],
        )
    if plan_path.suffix == ".yaml":
        try:
            import yaml  # type: ignore[import-untyped]
            plan_dict = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
        except ImportError:
            plan_dict = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan_dict = _read_json(plan_path)

    bindings = plan_dict.get("region_kernel_bindings") or []
    if not bindings:
        body = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "status": "skipped",
            "refinement_status": "skipped",
            "failure_summary": (
                "no region_kernel_bindings in plan; submit a provider "
                "response (M-43) and re-emit (M-46) before differential"
            ),
            "cases_total": 0, "cases_passed": 0, "case_records": [],
        }
        report_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return GlueDifferentialResult(
            out_dir=out_dir, report_path=report_path,
            status="skipped", refinement_status="skipped",
            cases_passed=0, cases_total=0,
            failure_summary=body["failure_summary"],
        )

    # Resolve contracts + kernel callables per region.
    contracts_by_region: dict[str, dict[str, Any]] = {}
    kernels: dict[str, Callable[..., Any]] = {}
    region_order: list[str] = []
    for b in bindings:
        region_id = b["region_id"]
        contract_path = run_dir / "04_kernel_codegen" / "contracts"
        candidate = contract_path / f"{region_id}.{b['contract_hash']}.json"
        if not candidate.exists():
            continue
        contract = _read_json(candidate)
        contracts_by_region[region_id] = contract
        if kernel_resolver is not None:
            kernels[region_id] = kernel_resolver(contract)
        else:
            kernels[region_id] = _tiled_kernel_for_region(contract=contract)
        region_order.append(region_id)

    # Every binding referenced a contract file that doesn't exist on
    # disk. This happens on synthetic runs that only exercise gap
    # discovery (the bindings are emitted but the commit step is
    # skipped). Treat the same way as the "no bindings" branch above:
    # write a typed ``skipped`` report and return — never crash on
    # ``region_order[0]``.
    if not region_order:
        body = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "status": "skipped",
            "refinement_status": "skipped",
            "failure_summary": (
                "no contract files resolved on disk for any binding; "
                "commit a provider response (M-43) so "
                "04_kernel_codegen/contracts/<region>.<hash>.json exists "
                "before running the differential"
            ),
            "cases_total": 0, "cases_passed": 0, "case_records": [],
        }
        report_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return GlueDifferentialResult(
            out_dir=out_dir, report_path=report_path,
            status="skipped", refinement_status="skipped",
            cases_passed=0, cases_total=0,
            failure_summary=body["failure_summary"],
        )

    # Import the emitted executor.
    module = _import_executor(executor_path)

    # For each case, drive the executor and compare against eager.
    import torch
    from compgen.graph_compilation.real_transform_differential import (
        matmul_higham_bound,
    )
    from compgen.runtime.glue import CpuRuntimeAdapter

    case_records: list[CaseRecord] = []
    cases_passed = 0
    overall_status = "pass"
    overall_refinement = "pending"
    failure_summary = ""

    # The differential operates on the FIRST region's contract today
    # (ships single-region; multi-region differential lands in
    # .x with proper IO routing).
    primary_region = region_order[0]
    primary_contract = contracts_by_region[primary_region]
    primary_attrs = {
        a["name"]: a["value"]
        for a in (primary_contract.get("io") or {}).get("attributes") or []
    }
    declared_refinement = str(
        primary_attrs.get("declared_refinement") or "unknown"
    )

    cases = _generate_cases_from_contract(
        contract=primary_contract, num_cases=num_cases,
    )

    for idx, case_inputs in enumerate(cases):
        case_id = f"case_{idx:03d}"
        # Build io dict in contract input order.
        io = {}
        for i, t_in in enumerate(case_inputs):
            io[f"arg_{i}"] = t_in
        try:
            # Run via the emitted executor.
            adapter = CpuRuntimeAdapter()
            tiled_out = module.compgen_run(io, kernels, runtime=adapter)
        except Exception as exc:  # noqa: BLE001 — surface as fail
            case_records.append(CaseRecord(
                case_id=case_id, region_id=primary_region,
                status="fail",
                max_abs_error=float("inf"), max_rel_error=float("inf"),
                higham_bound=0.0,
                declared_refinement=declared_refinement,
                reason=f"executor raised: {type(exc).__name__}: {exc}",
            ))
            overall_status = "fail"
            failure_summary = f"executor raised on {case_id}: {exc}"
            continue

        # Eager reference.
        eager_out = torch.matmul(case_inputs[0], case_inputs[1])
        diff = (tiled_out - eager_out).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = eager_out.abs().clamp(min=1e-30)
        max_rel = (
            float((diff / denom).max().item()) if diff.numel() else 0.0
        )
        bound = matmul_higham_bound(case_inputs[0], case_inputs[1])

        if declared_refinement == "bit_equality":
            ok = (max_abs == 0.0 and max_rel == 0.0)
            reason = (
                "" if ok else
                f"bit_equality requires exact equality but observed "
                f"max_abs={max_abs:.3e}"
            )
        else:
            ok = max_abs <= bound
            reason = (
                "" if ok else
                f"deviation past Higham bound: max_abs={max_abs:.3e} > "
                f"bound={bound:.3e}"
            )
        case_records.append(CaseRecord(
            case_id=case_id, region_id=primary_region,
            status="pass" if ok else "fail",
            max_abs_error=max_abs, max_rel_error=max_rel,
            higham_bound=bound,
            declared_refinement=declared_refinement,
            reason=reason,
        ))
        if ok:
            cases_passed += 1
        else:
            overall_status = "fail"
            failure_summary = failure_summary or reason

    if overall_status == "pass":
        if declared_refinement == "bit_equality":
            overall_refinement = "discharged_bit_equality"
        elif declared_refinement == "tolerance_eps":
            overall_refinement = "discharged_tolerance_eps"
        else:
            overall_refinement = "discharged_unknown"
    else:
        overall_refinement = "fail_refinement_mismatch"

    body = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utcnow(),
        "status": overall_status,
        "refinement_status": overall_refinement,
        "declared_refinement": declared_refinement,
        "failure_summary": failure_summary,
        "cases_total": len(cases),
        "cases_passed": cases_passed,
        "primary_region": primary_region,
        "case_records": [c.to_dict() for c in case_records],
        "evidence": {
            "executor_path": str(executor_path.relative_to(run_dir)),
            "plan_path": str(plan_path.relative_to(run_dir)),
            "contracts": {
                rid: f"04_kernel_codegen/contracts/{rid}.{contracts_by_region[rid].get('metadata', {}).get('source_candidate_id', '')}.json"
                for rid in region_order
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return GlueDifferentialResult(
        out_dir=out_dir, report_path=report_path,
        status=overall_status,
        refinement_status=overall_refinement,
        cases_passed=cases_passed, cases_total=len(cases),
        failure_summary=failure_summary,
        case_records=tuple(case_records),
    )
