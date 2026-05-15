"""JSON / CSV report writers for the admission probe and suite runner."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import torch

from compgen.model_admission.schemas import (
    AdmissionReport,
    AdmissionStatus,
    DynamoCaptureReport,
    EagerReport,
    ExportReport,
    FxReport,
    SuiteSummary,
    SuiteSummaryRow,
    TorchCompileReport,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_environment(path: Path) -> None:
    payload = {
        "schema_version": "admission_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    write_json(path, payload)


def write_admission(
    out_dir: Path,
    admission: AdmissionReport,
    eager: EagerReport,
    fx: FxReport,
    export: ExportReport,
    dynamo: DynamoCaptureReport,
    compile_rep: TorchCompileReport,
) -> None:
    write_json(out_dir / "admission_report.json", admission.to_dict())
    write_json(out_dir / "eager_report.json", eager.to_dict())
    write_json(out_dir / "fx_report.json", fx.to_dict())
    write_json(out_dir / "export_report.json", export.to_dict())
    write_json(out_dir / "dynamo_report.json", dynamo.to_dict())
    write_json(out_dir / "torch_compile_report.json", compile_rep.to_dict())


def write_suite_summary(out_dir: Path, summary: SuiteSummary) -> None:
    """Emit the four artifacts the user spec requires.

    - ``admission_summary.json``
    - ``admission_summary.csv``
    - ``availability_matrix.csv``
    - ``torch_compile_matrix.csv``
    - ``artifact_matrix.csv``
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "admission_summary.json", summary.to_dict())

    cols = [
        "model_id",
        "slice_id",
        "family",
        "support_mode",
        "blocking",
        "source_verified",
        "weights_available",
        "dependency_status",
        "eager_status",
        "fx_status",
        "export_status",
        "dynamo_status",
        "torch_compile_status",
        "graph_break_count",
        "compile_time_s",
        "recommended_next_step",
    ]
    _write_csv(out_dir / "admission_summary.csv", cols, summary.rows)

    _write_csv(
        out_dir / "availability_matrix.csv",
        ["model_id", "slice_id", "family", "support_mode", "blocking", "weights_available", "dependency_status"],
        summary.rows,
    )

    _write_csv(
        out_dir / "torch_compile_matrix.csv",
        ["model_id", "slice_id", "torch_compile_status", "graph_break_count", "compile_time_s"],
        summary.rows,
    )

    _write_csv(
        out_dir / "artifact_matrix.csv",
        ["model_id", "slice_id", "eager_status", "fx_status", "export_status", "dynamo_status", "torch_compile_status"],
        summary.rows,
    )


def _write_csv(path: Path, columns: list[str], rows: list[SuiteSummaryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


def aggregate_summary(suite_path: Path, out_dir: Path, rows: list[SuiteSummaryRow]) -> SuiteSummary:
    available = sum(1 for r in rows if r.dependency_status == AdmissionStatus.AVAILABLE.value)
    available_slice = sum(1 for r in rows if r.dependency_status == AdmissionStatus.AVAILABLE_SLICE_ONLY.value)
    unavailable = sum(1 for r in rows if r.dependency_status.startswith("unavailable_"))
    failed = sum(1 for r in rows if r.dependency_status.startswith("failed_"))
    return SuiteSummary(
        suite_path=str(suite_path),
        out_dir=str(out_dir),
        rows=list(rows),
        total=len(rows),
        available=available,
        available_slice_only=available_slice,
        unavailable=unavailable,
        failed=failed,
    )


__all__ = [
    "aggregate_summary",
    "write_admission",
    "write_environment",
    "write_json",
    "write_suite_summary",
]
