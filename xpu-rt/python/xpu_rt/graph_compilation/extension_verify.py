"""Verify a filled extension against its locked reference.

Two checks per workspace:

1. **Locked-files audit** — every file listed in
   ``extension_contract.locked_files`` must hash to the value recorded
   in ``locked_files_sha256``. If the agent edited anything outside
   ``extension.py``, verify fails.

2. **Differential test** — generate ``num_random_inputs`` random
   tensors with shapes matching the gap_record's ``shape_signature``,
   call ``reference.reference(*inputs)`` and ``extension.extension(*inputs)``,
   compare with ``rtol`` / ``atol`` from the contract. Track
   ``max_abs_error`` and ``max_rel_error``.

The verifier writes ``results/verification.json`` and returns a
``VerifyResult`` so the caller (gap_closure) can decide whether to
register the extension.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from compgen.graph_compilation.hashing import sha256_file


@dataclass
class VerifyResult:
    status: str  # "pass" | "fail"
    extension_id: str
    max_abs_error: float
    max_rel_error: float
    num_inputs: int
    locked_audit_status: str  # "pass" | "fail"
    locked_audit_violations: list[str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "extension_verification_v1",
            "status": self.status,
            "extension_id": self.extension_id,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "num_inputs": self.num_inputs,
            "locked_audit_status": self.locked_audit_status,
            "locked_audit_violations": list(self.locked_audit_violations),
            "detail": self.detail,
            "verified_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# --------------------------------------------------------------------------- #
# Locked-files audit
# --------------------------------------------------------------------------- #


def _audit_locked_files(workspace: Path, contract: dict[str, Any]) -> tuple[str, list[str]]:
    violations: list[str] = []
    for rel, declared_sha in contract.get("locked_files_sha256", {}).items():
        path = workspace / rel
        if not path.exists():
            violations.append(f"missing locked file: {rel}")
            continue
        actual = sha256_file(path)
        if actual != declared_sha:
            violations.append(
                f"locked file modified: {rel} declared={declared_sha[:12]} actual={actual[:12]}"
            )
    return ("pass" if not violations else "fail", violations)


# --------------------------------------------------------------------------- #
# Tensor sampler
# --------------------------------------------------------------------------- #


_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.bool": torch.bool,
}


def _build_inputs(
    shape_sig: dict[str, Any], dtype_sig: dict[str, Any], generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Build a single random input tuple from the gap's signatures.

    Static shapes only for now — dynamic dims (recorded as strings)
    are coerced to 1.
    """
    in_shapes = shape_sig.get("inputs", [])
    in_dtypes = dtype_sig.get("inputs", [])
    inputs: list[torch.Tensor] = []
    for i, raw_shape in enumerate(in_shapes):
        shape = [int(s) if isinstance(s, int) else 1 for s in raw_shape]
        dtype_str = in_dtypes[i] if i < len(in_dtypes) else "torch.float32"
        dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
        if dtype.is_floating_point:
            t = torch.randn(shape, generator=generator, dtype=dtype)
        elif dtype == torch.bool:
            t = torch.rand(shape, generator=generator) > 0.5
        else:
            # Integer dtype: small range so it doesn't overflow downstream ops.
            t = torch.randint(0, 8, shape, generator=generator, dtype=dtype)
        inputs.append(t)
    return tuple(inputs)


# --------------------------------------------------------------------------- #
# Loader for the workspace's reference + extension functions
# --------------------------------------------------------------------------- #


def _load_callable(path: Path, module_name: str, attr: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, attr):
        raise AttributeError(f"{path} has no attribute {attr!r}")
    return getattr(module, attr)


# --------------------------------------------------------------------------- #
# Differential check
# --------------------------------------------------------------------------- #


def _diff_max(actual: Any, expected: Any) -> tuple[float, float]:
    """Element-wise (max_abs, max_rel) between two outputs."""
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        max_abs, max_rel = 0.0, 0.0
        for a, e in zip(actual, expected):
            ma, mr = _diff_max(a, e)
            max_abs = max(max_abs, ma)
            max_rel = max(max_rel, mr)
        return max_abs, max_rel
    if not (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)):
        # Non-tensor outputs: only treat exact equality as 0; otherwise inf.
        return (0.0, 0.0) if actual == expected else (float("inf"), float("inf"))
    if actual.shape != expected.shape:
        return (float("inf"), float("inf"))
    if not actual.is_floating_point() or not expected.is_floating_point():
        eq = (actual == expected).all().item()
        return (0.0, 0.0) if eq else (float("inf"), float("inf"))
    diff = (actual.detach() - expected.detach()).abs()
    if diff.numel() == 0:
        return 0.0, 0.0
    max_abs = float(diff.max().item())
    denom = expected.detach().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_verify(workspace: Path) -> VerifyResult:
    """Run the locked-files audit + differential test for ``workspace``."""
    workspace = Path(workspace).resolve()
    contract = json.loads((workspace / "extension_contract.json").read_text(encoding="utf-8"))
    gap = json.loads((workspace / "gap_record.json").read_text(encoding="utf-8"))
    extension_id = contract["extension_id"]
    verif = contract.get("verification", {})

    # 1. Locked-files audit (cheap, do it first).
    audit_status, violations = _audit_locked_files(workspace, contract)
    if audit_status == "fail":
        result = VerifyResult(
            status="fail",
            extension_id=extension_id,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            num_inputs=0,
            locked_audit_status="fail",
            locked_audit_violations=violations,
            detail=f"locked-files audit failed: {violations}",
        )
        _write_result(workspace, result)
        return result

    # 2. Load reference + extension.
    try:
        reference = _load_callable(
            workspace / "reference.py", f"crg_ext_ref_{extension_id}", "reference"
        )
        extension = _load_callable(
            workspace / "extension.py", f"crg_ext_impl_{extension_id}", "extension"
        )
    except Exception as exc:
        result = VerifyResult(
            status="fail",
            extension_id=extension_id,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            num_inputs=0,
            locked_audit_status=audit_status,
            locked_audit_violations=[],
            detail=f"failed to load extension/reference: {type(exc).__name__}: {exc}",
        )
        _write_result(workspace, result)
        return result

    # 3. Differential trials.
    n_trials = int(verif.get("num_random_inputs", 100))
    rtol = float(verif.get("rtol", 1e-5))
    atol = float(verif.get("atol", 1e-5))
    seed = int(verif.get("seed", 0))
    gen = torch.Generator()
    gen.manual_seed(seed)

    max_abs, max_rel = 0.0, 0.0
    failures = 0
    last_error: str | None = None
    for _ in range(n_trials):
        inputs = _build_inputs(
            gap.get("shape_signature", {}), gap.get("dtype_signature", {}), gen
        )
        try:
            with torch.no_grad():
                ref_out = reference(*inputs)
                ext_out = extension(*inputs)
        except Exception as exc:
            failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        ma, mr = _diff_max(ext_out, ref_out)
        max_abs = max(max_abs, ma)
        max_rel = max(max_rel, mr)

    if failures:
        result = VerifyResult(
            status="fail",
            extension_id=extension_id,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            num_inputs=n_trials - failures,
            locked_audit_status=audit_status,
            locked_audit_violations=[],
            detail=f"{failures}/{n_trials} trials raised; last={last_error}",
        )
    elif max_abs > atol or max_rel > rtol:
        result = VerifyResult(
            status="fail",
            extension_id=extension_id,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            num_inputs=n_trials,
            locked_audit_status=audit_status,
            locked_audit_violations=[],
            detail=f"tolerance exceeded: max_abs={max_abs} > atol={atol} OR max_rel={max_rel} > rtol={rtol}",
        )
    else:
        result = VerifyResult(
            status="pass",
            extension_id=extension_id,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            num_inputs=n_trials,
            locked_audit_status=audit_status,
            locked_audit_violations=[],
            detail=f"{n_trials}/{n_trials} trials within tolerance",
        )
    _write_result(workspace, result)
    return result


def _write_result(workspace: Path, result: VerifyResult) -> None:
    out = workspace / "results" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_manifest_post_verify(workspace, result)


def _update_manifest_post_verify(workspace: Path, result: VerifyResult) -> None:
    """Patch ``manifest.yaml`` after verification.

    The agent edits ``extension.py`` only; the verifier owns the
    ``status`` / ``last_verified_*`` fields. Drafts go to ``verified``
    on pass, ``rejected`` on fail; once ``registered`` we never roll
    back the status here.
    """
    import yaml  # local import — keeps top-level import surface small

    manifest_path = workspace / "manifest.yaml"
    if not manifest_path.exists():
        return
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw["last_verified_at_utc"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw["last_verified_status"] = result.status
    current = raw.get("status", "draft")
    if current != "registered":
        raw["status"] = "verified" if result.status == "pass" else "rejected"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Spec 05: --out report pack
# --------------------------------------------------------------------------- #


def emit_extension_reports(*, workspace: Path, out_dir: Path, verify_result: VerifyResult) -> None:
    """Write the four spec-required reports under ``out_dir``.

    - ``extension_verify.json`` — top-level verify result (mirrors what verify
      already wrote into the workspace's ``results/verification.json``,
      with ``contract_hash`` and ``extension_source_hash`` added)
    - ``differential_report.json`` — frozen-case-by-case diff numbers
    - ``locked_files_audit.json`` — sha256 audit per locked file
    - ``source_hashes.json`` — hash of every workspace file (the
      provenance log a downstream registry/observer can pin against)
    """
    workspace = Path(workspace).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    contract_path = workspace / "extension_contract.json"
    extension_path = workspace / "extension.py"
    contract_obj = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(contract_path)
    extension_source_hash = sha256_file(extension_path)

    # 1. extension_verify.json
    verify_obj = dict(verify_result.to_dict())
    verify_obj["contract_hash"] = "sha256:" + contract_hash
    verify_obj["extension_source_hash"] = "sha256:" + extension_source_hash
    verify_obj["extension_path"] = str(workspace)
    (out_dir / "extension_verify.json").write_text(
        json.dumps(verify_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 2. differential_report.json — re-run the frozen cases and record per-case diffs.
    differential = _build_differential_report(workspace, contract_obj)
    (out_dir / "differential_report.json").write_text(
        json.dumps(differential, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 3. locked_files_audit.json — sha256 of every locked file, declared vs actual.
    locked_audit = {
        "schema_version": "extension_locked_audit_v1",
        "extension_id": contract_obj["extension_id"],
        "status": verify_result.locked_audit_status,
        "violations": list(verify_result.locked_audit_violations),
        "files": [],
    }
    for rel, declared in contract_obj.get("locked_files_sha256", {}).items():
        path = workspace / rel
        actual = sha256_file(path) if path.exists() else None
        locked_audit["files"].append(
            {
                "path": rel,
                "declared_sha256": declared,
                "actual_sha256": actual,
                "matches": actual == declared,
                "exists": actual is not None,
            }
        )
    (out_dir / "locked_files_audit.json").write_text(
        json.dumps(locked_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # 4. source_hashes.json — hash every file in the workspace (locked + editable).
    file_hashes: list[dict[str, Any]] = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(workspace).as_posix()
        file_hashes.append(
            {
                "path": rel,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "role": _classify_role(rel, contract_obj),
            }
        )
    source_hashes = {
        "schema_version": "extension_source_hashes_v1",
        "extension_id": contract_obj["extension_id"],
        "extension_path": str(workspace),
        "contract_hash": "sha256:" + contract_hash,
        "files": file_hashes,
    }
    (out_dir / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_differential_report(workspace: Path, contract_obj: dict[str, Any]) -> dict[str, Any]:
    """Per-case max_abs / max_rel for every frozen (input, expected) pair."""
    extension_path = workspace / "extension.py"
    extension = _load_callable(
        extension_path, f"crg_diff_report_{workspace.name}", "extension"
    )
    cases = contract_obj.get("verification", {}).get("frozen_cases", {}).get("cases", [])
    rtol = float(contract_obj.get("verification", {}).get("rtol", 1e-5))
    atol = float(contract_obj.get("verification", {}).get("atol", 1e-5))

    case_reports: list[dict[str, Any]] = []
    overall_max_abs, overall_max_rel = 0.0, 0.0
    for case in cases:
        inputs = torch.load(workspace / case["inputs"], weights_only=False)
        expected = torch.load(workspace / case["expected"], weights_only=False)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        try:
            with torch.no_grad():
                actual = extension(*inputs)
        except Exception as exc:
            case_reports.append(
                {
                    "index": case["index"],
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "max_abs_error": float("inf"),
                    "max_rel_error": float("inf"),
                }
            )
            overall_max_abs = float("inf")
            overall_max_rel = float("inf")
            continue
        ma, mr = _diff_max(actual, expected)
        case_reports.append(
            {
                "index": case["index"],
                "status": "pass" if (ma <= atol and mr <= rtol) else "fail",
                "max_abs_error": ma,
                "max_rel_error": mr,
            }
        )
        overall_max_abs = max(overall_max_abs, ma)
        overall_max_rel = max(overall_max_rel, mr)

    return {
        "schema_version": "extension_differential_report_v1",
        "extension_id": contract_obj["extension_id"],
        "rtol": rtol,
        "atol": atol,
        "frozen_case_count": len(case_reports),
        "max_abs_error": overall_max_abs,
        "max_rel_error": overall_max_rel,
        "status": "pass" if (overall_max_abs <= atol and overall_max_rel <= rtol) else "fail",
        "cases": case_reports,
    }


def _classify_role(rel_path: str, contract_obj: dict[str, Any]) -> str:
    locked = set(contract_obj.get("locked_files", []))
    fillable = set(contract_obj.get("fillable_files", []))
    if rel_path in locked:
        return "locked"
    if rel_path in fillable:
        return "editable"
    if rel_path.startswith("results/"):
        return "verifier_output"
    if rel_path == "extension_contract.json":
        return "contract"
    return "other"


# (``_load_callable`` and ``_diff_max`` are already defined above and
# reused by ``_build_differential_report``.)
