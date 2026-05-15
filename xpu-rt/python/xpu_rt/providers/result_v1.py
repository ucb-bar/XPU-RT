"""``provider_result_v1`` typed schema.

The v1 schema replaces the loose
:class:`compgen.kernels.provider.ProviderResult` for new adapters
and audits. Concretely::

    {
      "schema_version": "provider_result_v1",
      "task_id": "kcodegen_0007",
      "provider_id": "cffi_c",
      "target_id": "host_cpu",
      "contract_hash": "abc123",
      "status": "generated",            # closed enum
      "artifacts": {
        "source":         "<path>",
        "metadata":       "<path>",
        "compile_log":    "<path>"
      },
      "claims": {
        "estimated_latency_us": null,
        "supports_dispatch": ["sync"],
        "registers": null,
        "shared_memory_bytes": null
      },
      "contract_feedback": [],
      "detail": ""
    }

``status`` is closed-enum:

* ``generated``         — artifacts emitted; verifier will decide correctness.
* ``contract_rejected`` — provider declined the contract (e.g. wrong op family).
* ``blocked``           — environment / toolchain prereq missing.
* ``error``             — adapter raised; ``detail`` carries the message.

Critically, ``status="generated"`` is **not** a certification. The
verifier downstream emits the
:class:`compgen.kernels.kernel_certificate.KernelCertificate`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from compgen.kernels.provider import ProviderResult as LegacyProviderResult

SCHEMA_VERSION: Final[str] = "provider_result_v1"

STATUSES: Final[tuple[str, ...]] = (
    "generated",
    "contract_rejected",
    "blocked",
    "error",
)


class ProviderResultV1Error(ValueError):
    """A v1 result body violated the schema."""


@dataclass(frozen=True)
class ProviderResultV1:
    """The v1 result shape every adapter must return."""

    schema_version: str
    task_id: str
    provider_id: str
    target_id: str
    contract_hash: str
    status: str
    artifacts: dict[str, str] = field(default_factory=dict)
    claims: dict[str, Any] = field(default_factory=dict)
    contract_feedback: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProviderResultV1Error(
                f"schema_version={self.schema_version!r} must be {SCHEMA_VERSION!r}"
            )
        if self.status not in STATUSES:
            raise ProviderResultV1Error(
                f"status={self.status!r} must be one of {STATUSES}"
            )
        if self.status == "generated":
            if "source" not in self.artifacts:
                raise ProviderResultV1Error(
                    "status=generated requires artifacts['source']"
                )
        if self.status in {"blocked", "error", "contract_rejected"}:
            if not self.detail:
                raise ProviderResultV1Error(
                    f"status={self.status!r} requires a non-empty detail"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "contract_hash": self.contract_hash,
            "status": self.status,
            "artifacts": dict(self.artifacts),
            "claims": dict(self.claims),
            "contract_feedback": [dict(f) for f in self.contract_feedback],
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "ProviderResultV1":
        return cls(
            schema_version=str(body.get("schema_version", SCHEMA_VERSION)),
            task_id=str(body["task_id"]),
            provider_id=str(body["provider_id"]),
            target_id=str(body.get("target_id", "")),
            contract_hash=str(body.get("contract_hash", "")),
            status=str(body["status"]),
            artifacts=dict(body.get("artifacts", {}) or {}),
            claims=dict(body.get("claims", {}) or {}),
            contract_feedback=list(body.get("contract_feedback", []) or []),
            detail=str(body.get("detail", "")),
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


def legacy_to_v1(
    legacy: LegacyProviderResult,
    *,
    task_id: str,
    provider_id: str,
    target_id: str,
    contract_hash: str,
    artifact_dir: str | Path | None = None,
) -> ProviderResultV1:
    """Translate a legacy :class:`ProviderResult` into v1 shape.

    The legacy result carries ``found`` / ``correct`` / ``kernel_code``
    / ``latency_us`` / ``metadata``. Mapping:

    * ``found=False`` + non-empty metadata['reason'] → ``status="blocked"``
      with ``detail`` from the reason.
    * ``found=True`` → ``status="generated"`` and kernel source is
      written under ``artifact_dir/kernel.<ext>`` when ``artifact_dir``
      is supplied; the path is recorded in ``artifacts['source']``.
      When no artifact_dir is supplied the source is **inlined** in
      ``claims['inline_source']`` so callers can still inspect it.
    """

    import math

    if not legacy.found:
        detail = (
            (legacy.metadata or {}).get("reason")
            or (legacy.metadata or {}).get("stderr_tail")
            or "legacy provider returned found=False"
        )
        return ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id=task_id,
            provider_id=provider_id,
            target_id=target_id,
            contract_hash=contract_hash,
            status="blocked",
            detail=str(detail)[:1024],
            claims={
                "iterations_used": int(legacy.iterations_used),
                "total_candidates": int(legacy.total_candidates),
                "legacy_metadata": dict(legacy.metadata or {}),
            },
        )

    ext = {"python": "py", "triton": "py", "c": "c", "cuda": "cu"}.get(
        (legacy.language or "").lower(), "txt"
    )
    artifacts: dict[str, str] = {}
    claims: dict[str, Any] = {
        "estimated_latency_us": (
            None
            if not math.isfinite(legacy.latency_us)
            else float(legacy.latency_us)
        ),
        "iterations_used": int(legacy.iterations_used),
        "total_candidates": int(legacy.total_candidates),
        "speedup_vs_baseline": (
            float(legacy.speedup) if math.isfinite(legacy.speedup) else None
        ),
        "language": legacy.language,
        "emit_mode": legacy.emit_mode,
        "legacy_metadata": dict(legacy.metadata or {}),
    }

    if artifact_dir is not None and legacy.kernel_code:
        ad = Path(artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        source_path = ad / f"kernel.{ext}"
        source_path.write_text(legacy.kernel_code)
        artifacts["source"] = str(source_path)
        meta_path = ad / "kernel_metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "language": legacy.language,
                    "emit_mode": legacy.emit_mode,
                    "iterations_used": legacy.iterations_used,
                    "total_candidates": legacy.total_candidates,
                    "metadata": dict(legacy.metadata or {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        artifacts["metadata"] = str(meta_path)
    else:
        artifacts["source"] = ""
        claims["inline_source"] = legacy.kernel_code or ""

    return ProviderResultV1(
        schema_version=SCHEMA_VERSION,
        task_id=task_id,
        provider_id=provider_id,
        target_id=target_id,
        contract_hash=contract_hash,
        status="generated",
        artifacts=artifacts,
        claims=claims,
        contract_feedback=[
            {
                "kind": getattr(f, "kind", ""),
                "message": getattr(f, "message", ""),
            }
            for f in (legacy.contract_feedback or [])
        ],
    )


def load_result_v1(path: str | Path) -> ProviderResultV1:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProviderResultV1.from_dict(body)
