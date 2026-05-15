"""Typed schemas for ProviderCard, ProviderProbeResult, and the
five-level integration ladder.

The legacy kernel-only types ``BidPreview`` / ``ProviderResult`` /
``KernelProvider`` live in :mod:`compgen.kernels.provider` and stay
there — this module is the *generalized* card-driven layer that wraps
them. Re-exports happen through :mod:`compgen.providers`; importing
the legacy types directly from ``compgen.kernels.provider`` remains the
canonical path for kernel-codegen code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

INTEGRATION_LEVELS: Final[tuple[str, ...]] = (
    "card_only",
    "probe",
    "generate",
    "verify",
    "promote",
)

PAPER_CLAIMABLE_LEVELS: Final[frozenset[str]] = frozenset(
    {"verify", "promote"}
)

PROBE_STATUSES: Final[tuple[str, ...]] = (
    "available",
    "blocked",
    "unsupported",
    "probe_error",
    "not_installed",
)

BLOCKED_REASONS: Final[tuple[str, ...]] = (
    "env_missing",
    "python_package_missing",
    "command_missing",
    "hardware_unavailable",
    "sdk_missing",
    "license_missing",
    "version_mismatch",
    "unsupported_platform",
    "unsupported_contract_kind",
    "probe_exception",
)


# Type aliases — string-literal narrowing is enforced at construction.
IntegrationLevel = str
ProviderProbeStatus = str
BlockedReason = str


class ProviderCardError(ValueError):
    """A ProviderCard YAML body violated the schema."""


class ProviderProbeError(ValueError):
    """A ProviderProbeResult was constructed with an invalid status / reason."""


@dataclass(frozen=True)
class ProviderCard:
    """Static declaration of a kernel / dialect provider.

    Cards live under ``python/compgen/providers/cards/*.yaml`` (or in
    a user extension manifest) and are loaded once at registry build
    time. A card is **not** evidence of support — only certified
    artifacts are.
    """

    schema_version: str
    provider_id: str
    integration_level: IntegrationLevel
    target_families: tuple[str, ...]
    contract_kinds: tuple[str, ...]
    emits: tuple[str, ...]
    entrypoint: str
    paper_claimable: bool = False
    required_env: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    required_python_imports: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(cls, body: dict[str, Any], *, source: Path | None = None) -> "ProviderCard":
        try:
            schema_version = str(body["schema_version"])
            provider_id = str(body["provider_id"])
            integration_level = str(body["integration_level"])
        except KeyError as exc:
            raise ProviderCardError(
                f"provider card missing required field {exc.args[0]!r} (source={source})"
            ) from exc
        if integration_level not in INTEGRATION_LEVELS:
            raise ProviderCardError(
                f"provider {provider_id!r} integration_level={integration_level!r} "
                f"must be one of {INTEGRATION_LEVELS} (source={source})"
            )
        paper_claimable = bool(body.get("paper_claimable", False))
        if paper_claimable and integration_level not in PAPER_CLAIMABLE_LEVELS:
            raise ProviderCardError(
                f"provider {provider_id!r} has paper_claimable=true at "
                f"integration_level={integration_level!r} — only "
                f"{sorted(PAPER_CLAIMABLE_LEVELS)} may set paper_claimable=true "
                f"(source={source})"
            )
        return cls(
            schema_version=schema_version,
            provider_id=provider_id,
            integration_level=integration_level,
            target_families=tuple(body.get("target_families", ())),
            contract_kinds=tuple(body.get("contract_kinds", ())),
            emits=tuple(body.get("emits", ())),
            entrypoint=str(body.get("entrypoint", "")),
            paper_claimable=paper_claimable,
            required_env=tuple(body.get("required_env", ())),
            required_commands=tuple(body.get("required_commands", ())),
            required_python_imports=tuple(body.get("required_python_imports", ())),
            description=str(body.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "integration_level": self.integration_level,
            "target_families": list(self.target_families),
            "contract_kinds": list(self.contract_kinds),
            "emits": list(self.emits),
            "entrypoint": self.entrypoint,
            "paper_claimable": self.paper_claimable,
            "required_env": list(self.required_env),
            "required_commands": list(self.required_commands),
            "required_python_imports": list(self.required_python_imports),
            "description": self.description,
        }


@dataclass(frozen=True)
class ProviderProbeResult:
    """Typed result of running a provider's probe.

    Hard rule 5: missing SDKs / hardware / licenses / packages always
    produce a typed ``blocked`` status with a typed ``blocked_reason``.
    A raw exception in ``probe()`` produces ``status=probe_error``
    with ``blocked_reason=probe_exception`` and the exception string
    in ``detail``. No crash, no silent disappearance.
    """

    schema_version: str
    provider_id: str
    status: ProviderProbeStatus
    blocked_reason: BlockedReason | None = None
    version: str = ""
    supports: tuple[str, ...] = ()
    detail: str = ""
    paper_claimable: bool = False
    required_env: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROBE_STATUSES:
            raise ProviderProbeError(
                f"provider {self.provider_id!r} status={self.status!r} "
                f"must be one of {PROBE_STATUSES}"
            )
        if self.status in {"blocked", "unsupported", "probe_error", "not_installed"}:
            if self.blocked_reason is None:
                raise ProviderProbeError(
                    f"provider {self.provider_id!r} status={self.status!r} "
                    f"requires a typed blocked_reason"
                )
            if self.blocked_reason not in BLOCKED_REASONS:
                raise ProviderProbeError(
                    f"provider {self.provider_id!r} blocked_reason="
                    f"{self.blocked_reason!r} must be one of {BLOCKED_REASONS}"
                )
        if self.status == "available" and self.blocked_reason is not None:
            raise ProviderProbeError(
                f"provider {self.provider_id!r} status=available must not "
                f"carry blocked_reason={self.blocked_reason!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "version": self.version,
            "supports": list(self.supports),
            "detail": self.detail,
            "paper_claimable": self.paper_claimable,
            "required_env": list(self.required_env),
            "required_commands": list(self.required_commands),
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "ProviderProbeResult":
        reason = body.get("blocked_reason")
        return cls(
            schema_version=str(body["schema_version"]),
            provider_id=str(body["provider_id"]),
            status=str(body["status"]),
            blocked_reason=str(reason) if reason is not None else None,
            version=str(body.get("version", "")),
            supports=tuple(body.get("supports", ())),
            detail=str(body.get("detail", "")),
            paper_claimable=bool(body.get("paper_claimable", False)),
            required_env=tuple(body.get("required_env", ())),
            required_commands=tuple(body.get("required_commands", ())),
        )
