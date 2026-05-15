"""ExtensionManifest + ExtensionTask schemas.

A ``compgen_extension.yaml`` manifest declares a bundle of cards
provided by a user / agent-authored extension. Loading is purely
structural — no filesystem side effects beyond reading the YAML.
Each card in ``provides`` round-trips through its schema and
inherits the integration-level discipline.

``extension_task_v1`` is the artifact emitted by the unsupported-op
flow. defines the schema; wires the emission and
the MCP commit/resume tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.extensions.errors import ExtensionManifestError, ExtensionTaskError
from compgen.pass_tools.pass_tool_types import PassToolCard
from compgen.providers.provider_types import ProviderCard
from compgen.targets.target_types import TargetCard

MANIFEST_SCHEMA_VERSION: Final[str] = "compgen_extension_v1"
EXTENSION_TASK_SCHEMA_VERSION: Final[str] = "extension_task_v1"

ALLOWED_EXTENSION_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {
        "kernel_provider",
        "dialect_provider",
        "pass_tool",
        "kernel_template",
        "contract_rule",
    }
)

EXTENSION_TASK_REASONS: Final[frozenset[str]] = frozenset(
    {
        "unsupported_op",
        "provider_gap",
        "missing_dialect_lowering",
        "missing_pass",
    }
)


@dataclass(frozen=True)
class ExtensionSecurity:
    sandbox_required: bool
    allowed_write_root: str


@dataclass(frozen=True)
class ExtensionManifest:
    schema_version: str
    extension_id: str
    version: str
    author: str
    description: str
    targets: tuple[TargetCard, ...]
    kernel_providers: tuple[ProviderCard, ...]
    dialect_providers: tuple[DialectProviderCard, ...]
    pass_tools: tuple[PassToolCard, ...]
    security: ExtensionSecurity
    required_env: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    required_python_imports: tuple[str, ...] = ()
    required_verification_checks: tuple[str, ...] = ()
    paper_claimable: bool = False
    source_path: Path | None = field(default=None, compare=False)

    @classmethod
    def from_dict(
        cls, body: dict[str, Any], *, source: Path | None = None
    ) -> "ExtensionManifest":
        if not isinstance(body, dict):
            raise ExtensionManifestError(
                f"manifest root must be a mapping (source={source})"
            )
        schema_version = str(body.get("schema_version", ""))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ExtensionManifestError(
                f"manifest schema_version={schema_version!r} must be "
                f"{MANIFEST_SCHEMA_VERSION!r} (source={source})"
            )
        extension = body.get("extension") or {}
        if not extension or "id" not in extension:
            raise ExtensionManifestError(
                f"manifest 'extension.id' is required (source={source})"
            )
        extension_id = str(extension["id"])
        version = str(extension.get("version", "0.0.0"))
        author = str(extension.get("author", ""))
        description = str(extension.get("description", ""))

        provides = body.get("provides") or {}
        targets = tuple(
            TargetCard.from_dict(t, source=source)
            for t in provides.get("targets", [])
        )
        kernel_providers = tuple(
            ProviderCard.from_dict(p, source=source)
            for p in provides.get("kernel_providers", [])
        )
        dialect_providers = tuple(
            DialectProviderCard.from_dict(d, source=source)
            for d in provides.get("dialect_providers", [])
        )
        pass_tools = tuple(
            PassToolCard.from_dict(pt, source=source)
            for pt in provides.get("pass_tools", [])
        )

        security_raw = body.get("security") or {}
        sandbox_required = bool(security_raw.get("sandbox_required", True))
        allowed_write_root = str(security_raw.get("allowed_write_root", "")).strip()
        if sandbox_required and not allowed_write_root:
            raise ExtensionManifestError(
                f"extension {extension_id!r} has sandbox_required=true but "
                f"security.allowed_write_root is empty (source={source})"
            )
        security = ExtensionSecurity(
            sandbox_required=sandbox_required,
            allowed_write_root=allowed_write_root,
        )

        probes = body.get("probes") or {}
        verification = body.get("verification") or {}
        return cls(
            schema_version=schema_version,
            extension_id=extension_id,
            version=version,
            author=author,
            description=description,
            targets=targets,
            kernel_providers=kernel_providers,
            dialect_providers=dialect_providers,
            pass_tools=pass_tools,
            security=security,
            required_env=tuple(probes.get("required_env", ())),
            required_commands=tuple(probes.get("commands", ())),
            required_python_imports=tuple(probes.get("python_imports", ())),
            required_verification_checks=tuple(
                verification.get("required_checks", ())
            ),
            paper_claimable=bool(body.get("paper_claimable", False)),
            source_path=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extension": {
                "id": self.extension_id,
                "version": self.version,
                "author": self.author,
                "description": self.description,
            },
            "provides": {
                "targets": [t.to_dict() for t in self.targets],
                "kernel_providers": [p.to_dict() for p in self.kernel_providers],
                "dialect_providers": [d.to_dict() for d in self.dialect_providers],
                "pass_tools": [pt.to_dict() for pt in self.pass_tools],
            },
            "probes": {
                "required_env": list(self.required_env),
                "commands": list(self.required_commands),
                "python_imports": list(self.required_python_imports),
            },
            "security": {
                "sandbox_required": self.security.sandbox_required,
                "allowed_write_root": self.security.allowed_write_root,
            },
            "verification": {
                "required_checks": list(self.required_verification_checks),
            },
            "paper_claimable": self.paper_claimable,
        }


def load_manifest(path: str | Path) -> ExtensionManifest:
    p = Path(path)
    body = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ExtensionManifestError(
            f"manifest at {p} must be a YAML mapping; got {type(body).__name__}"
        )
    return ExtensionManifest.from_dict(body, source=p)


# ---------------------------------------------------------------------------
# extension_task_v1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtensionTask:
    schema_version: str
    task_id: str
    reason: str
    op: str
    region_id: str
    contract_hash: str
    allowed_extension_types: tuple[str, ...]
    allowed_outputs: tuple[str, ...]
    forbidden: tuple[str, ...]
    verification_required: tuple[str, ...]
    output_dir: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "ExtensionTask":
        if not isinstance(body, dict):
            raise ExtensionTaskError("extension task body must be a mapping")
        sv = str(body.get("schema_version", ""))
        if sv != EXTENSION_TASK_SCHEMA_VERSION:
            raise ExtensionTaskError(
                f"extension task schema_version={sv!r} must be "
                f"{EXTENSION_TASK_SCHEMA_VERSION!r}"
            )
        required = ("task_id", "reason", "op", "region_id", "contract_hash", "output_dir")
        missing = [k for k in required if k not in body]
        if missing:
            raise ExtensionTaskError(
                f"extension task missing required fields: {missing}"
            )
        reason = str(body["reason"])
        if reason not in EXTENSION_TASK_REASONS:
            raise ExtensionTaskError(
                f"extension task reason={reason!r} must be one of "
                f"{sorted(EXTENSION_TASK_REASONS)}"
            )
        allowed_types = tuple(body.get("allowed_extension_types", ()))
        for kind in allowed_types:
            if kind not in ALLOWED_EXTENSION_TASK_TYPES:
                raise ExtensionTaskError(
                    f"extension task allowed_extension_types contains {kind!r} which "
                    f"is not in {sorted(ALLOWED_EXTENSION_TASK_TYPES)}"
                )
        known = set(required) | {
            "schema_version",
            "allowed_extension_types",
            "allowed_outputs",
            "forbidden",
            "verification_required",
        }
        extra = {k: v for k, v in body.items() if k not in known}
        return cls(
            schema_version=sv,
            task_id=str(body["task_id"]),
            reason=reason,
            op=str(body["op"]),
            region_id=str(body["region_id"]),
            contract_hash=str(body["contract_hash"]),
            allowed_extension_types=allowed_types,
            allowed_outputs=tuple(body.get("allowed_outputs", ())),
            forbidden=tuple(body.get("forbidden", ())),
            verification_required=tuple(body.get("verification_required", ())),
            output_dir=str(body["output_dir"]),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "reason": self.reason,
            "op": self.op,
            "region_id": self.region_id,
            "contract_hash": self.contract_hash,
            "allowed_extension_types": list(self.allowed_extension_types),
            "allowed_outputs": list(self.allowed_outputs),
            "forbidden": list(self.forbidden),
            "verification_required": list(self.verification_required),
            "output_dir": self.output_dir,
        }
        out.update(self.extra)
        return out

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


def load_extension_task(path: str | Path) -> ExtensionTask:
    p = Path(path)
    body = json.loads(p.read_text(encoding="utf-8"))
    return ExtensionTask.from_dict(body)
