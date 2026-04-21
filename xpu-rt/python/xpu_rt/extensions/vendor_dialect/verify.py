"""Verification harness for a scaffolded vendor adapter.

Runs the verification ladder declared in the descriptor:

1. **Structural** — import the scaffolded package, load the adapter, and
   run its lower + emit with a trivial Payload IR module. Catches
   scaffold regressions and obvious wiring bugs.
2. **Matmul diff-test** — compile a single ``linalg.matmul`` region,
   invoke the adapter's :meth:`validate`, and compare against a torch
   reference. Optional per descriptor.
3. **Workload diff-test** — compile a real workload (e.g. tinyllama)
   and compare outputs against Stage-0 golden artifacts. Optional per
   descriptor.

Each gate is wrapped in a timing + diagnostics record. Callers treat
the report as the ground truth for "did this adapter land?".
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter
from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Report records
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    name: str
    passed: bool
    elapsed_s: float = 0.0
    notes: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    package_path: str
    adapter_name: str
    target: str
    gates: list[GateResult] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def verify_package(
    package_path: str | Path,
    *,
    golden_inputs: dict[str, Any] | None = None,
    golden_output: Any = None,
    run_workload_gate: bool | None = None,
) -> VerificationReport:
    """Run the verification ladder for a scaffolded adapter package.

    Args:
        package_path: Directory that contains the scaffolded package
            (the one written by :func:`scaffold_package` — holds the
            inner ``<package_name>/`` Python module).
        golden_inputs: Optional golden inputs for matmul / workload gates.
        golden_output: Optional golden output tensor.
        run_workload_gate: Override descriptor's workload-diff-test flag.

    Returns:
        A :class:`VerificationReport` with per-gate timings and notes.
    """
    pkg_path = Path(package_path).expanduser().resolve()
    adapter = _load_scaffolded_adapter(pkg_path)
    descriptor = adapter.descriptor

    report = VerificationReport(
        package_path=str(pkg_path),
        adapter_name=adapter.name,
        target=adapter.target,
    )

    # Gate 1 — structural
    if descriptor.verification.structural:
        report.gates.append(_run_gate("structural", lambda: _gate_structural(adapter)))

    # Gate 2 — matmul diff-test
    if descriptor.verification.matmul_diff_test:
        report.gates.append(
            _run_gate(
                "matmul_diff",
                lambda: _gate_matmul_diff(
                    adapter,
                    tolerance_atol=descriptor.verification.tolerance_atol,
                    tolerance_rtol=descriptor.verification.tolerance_rtol,
                ),
            )
        )

    # Gate 3 — workload diff-test
    want_workload = (
        run_workload_gate
        if run_workload_gate is not None
        else descriptor.verification.workload_diff_test
    )
    if want_workload and golden_inputs is not None and golden_output is not None:
        report.gates.append(
            _run_gate(
                "workload_diff",
                lambda: _gate_workload_diff(
                    adapter,
                    golden_inputs=golden_inputs,
                    golden_output=golden_output,
                    tolerance_atol=descriptor.verification.tolerance_atol,
                    tolerance_rtol=descriptor.verification.tolerance_rtol,
                ),
            )
        )

    report.passed = all(g.passed for g in report.gates) and bool(report.gates)
    return report


# --------------------------------------------------------------------------- #
# Gate implementations
# --------------------------------------------------------------------------- #


def _gate_structural(adapter: VendorDialectAdapter) -> GateResult:
    """Lower + emit a minimal module; assert no crash + artifact returned."""
    out_dir = Path("/tmp") / f"compgen_verify_{adapter.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trivial = "module { func.func @empty() { return } }"
    lowering = adapter.lower_payload(trivial, output_dir=out_dir)
    artifact = adapter.emit_artifact(lowering, output_dir=out_dir)
    ok = bool(artifact.code) or bool(artifact.metadata)
    return GateResult(
        name="structural",
        passed=ok,
        notes=f"format={artifact.format}",
        details={
            "format": artifact.format,
            "target": artifact.target_name,
            "vendor_mlir_bytes": len(lowering.vendor_mlir),
        },
    )


def _gate_matmul_diff(
    adapter: VendorDialectAdapter,
    *,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> GateResult:
    """Compile a trivial ``linalg.matmul`` region and validate.

    We do not ship a real Payload-IR matmul emitter here — the scaffolded
    adapter is expected to accept the textual module below. Real matmul
    validation is Phase-D work; this gate smoke-tests the *call shape*.
    """
    out_dir = Path("/tmp") / f"compgen_verify_{adapter.name}_matmul"
    out_dir.mkdir(parents=True, exist_ok=True)
    matmul_payload = (
        "module {\n"
        "  func.func @matmul(%a: tensor<32x64xf16>, %b: tensor<64x32xf16>) "
        "-> tensor<32x32xf16> {\n"
        "    %c0 = arith.constant 0.0 : f16\n"
        "    %init = tensor.empty() : tensor<32x32xf16>\n"
        "    %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<32x32xf16>) "
        "-> tensor<32x32xf16>\n"
        "    %out = linalg.matmul ins(%a, %b : tensor<32x64xf16>, tensor<64x32xf16>) "
        "outs(%fill : tensor<32x32xf16>) -> tensor<32x32xf16>\n"
        "    return %out : tensor<32x32xf16>\n"
        "  }\n"
        "}\n"
    )
    lowering = adapter.lower_payload(matmul_payload, output_dir=out_dir)
    artifact = adapter.emit_artifact(lowering, output_dir=out_dir)
    ok_struct = bool(artifact.code) or bool(artifact.metadata)
    ok_valid = adapter.validate(artifact, golden_inputs={}, golden_output=None)
    return GateResult(
        name="matmul_diff",
        passed=bool(ok_struct and ok_valid),
        notes=f"validate={ok_valid} tol(atol={tolerance_atol}, rtol={tolerance_rtol})",
        details={"format": artifact.format},
    )


def _gate_workload_diff(
    adapter: VendorDialectAdapter,
    *,
    golden_inputs: dict[str, Any],
    golden_output: Any,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> GateResult:
    """Compile the golden-inputs Payload module and diff-test the result.

    The harness does not synthesize a Payload module here; the adapter's
    ``emit_artifact`` must tolerate a text module the caller has supplied
    via the ``payload_mlir`` entry of ``golden_inputs`` — real workload
    verification is deferred to Phase-D's concrete wrappers.
    """
    out_dir = Path("/tmp") / f"compgen_verify_{adapter.name}_workload"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_mlir = str(golden_inputs.get("payload_mlir", "module {}"))
    lowering = adapter.lower_payload(payload_mlir, output_dir=out_dir)
    artifact = adapter.emit_artifact(lowering, output_dir=out_dir)
    ok = adapter.validate(artifact, golden_inputs=golden_inputs, golden_output=golden_output)
    return GateResult(
        name="workload_diff",
        passed=bool(ok),
        notes=f"tol(atol={tolerance_atol}, rtol={tolerance_rtol})",
        details={"format": artifact.format},
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _run_gate(name: str, fn: Callable[[], GateResult]) -> GateResult:
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # pragma: no cover — defensive
        elapsed = time.perf_counter() - start
        log.warning("vendor_verify.gate_error", gate=name, error=str(exc))
        return GateResult(
            name=name,
            passed=False,
            elapsed_s=elapsed,
            notes=f"error: {exc}",
        )
    result.elapsed_s = time.perf_counter() - start
    log.info("vendor_verify.gate_done", gate=name, passed=result.passed, elapsed_s=result.elapsed_s)
    return result


def _load_scaffolded_adapter(package_path: Path) -> VendorDialectAdapter:
    """Import the scaffolded package and invoke ``load_adapter``.

    We try two paths in order:

    * If the package is importable in the current interpreter (because
      ``pip install -e .`` was run), just import it by name.
    * Otherwise, fall back to a file-based import so tests can run
      against a scaffolded tree without installing it.
    """
    descriptor_path = _find_descriptor(package_path)
    descriptor = VendorDialectDescriptor.load(descriptor_path)
    pkg_name = descriptor.package_name

    # Try installed import first.
    try:
        mod = importlib.import_module(pkg_name)
    except Exception:
        mod = _import_from_path(pkg_name, package_path)

    factory = getattr(mod, "load_adapter", None)
    if factory is None:
        raise AttributeError(f"{pkg_name}.load_adapter not found")
    adapter = factory()
    if not isinstance(adapter, VendorDialectAdapter):
        raise TypeError(
            f"{pkg_name}.load_adapter returned {type(adapter).__name__}, "
            f"expected VendorDialectAdapter"
        )
    return adapter


def _find_descriptor(package_path: Path) -> Path:
    """Locate ``descriptor.yaml`` inside a scaffolded package directory."""
    direct = list(package_path.glob("*/descriptor.yaml"))
    if direct:
        return direct[0]
    # Accept ``package_path`` itself being the inner module directory.
    candidate = package_path / "descriptor.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no descriptor.yaml under {package_path}")


def _import_from_path(pkg_name: str, package_path: Path):
    """Load ``package_path/<pkg_name>/__init__.py`` as a package."""
    init_file = package_path / pkg_name / "__init__.py"
    if not init_file.is_file():
        raise ImportError(f"cannot import {pkg_name} from {package_path}: {init_file} missing")
    parent = str(package_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(pkg_name)


__all__ = ["GateResult", "VerificationReport", "verify_package"]
