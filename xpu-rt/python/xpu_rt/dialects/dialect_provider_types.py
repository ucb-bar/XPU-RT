"""DialectProviderCard schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.providers.provider_types import (
    INTEGRATION_LEVELS,
    PAPER_CLAIMABLE_LEVELS,
)


class DialectProviderCardError(ValueError):
    """A DialectProviderCard YAML body violated the schema."""


@dataclass(frozen=True)
class DialectProviderCard:
    schema_version: str
    dialect_provider_id: str
    dialect_name: str
    integration_level: str
    consumes: tuple[str, ...]
    emits: tuple[str, ...]
    entrypoint: str
    paper_claimable: bool = False
    required_env: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def from_dict(
        cls, body: dict[str, Any], *, source: Path | None = None
    ) -> "DialectProviderCard":
        try:
            schema_version = str(body["schema_version"])
            dialect_provider_id = str(body["dialect_provider_id"])
            dialect_name = str(body["dialect_name"])
            integration_level = str(body["integration_level"])
        except KeyError as exc:
            raise DialectProviderCardError(
                f"dialect provider card missing required field {exc.args[0]!r} "
                f"(source={source})"
            ) from exc
        if integration_level not in INTEGRATION_LEVELS:
            raise DialectProviderCardError(
                f"dialect provider {dialect_provider_id!r} integration_level="
                f"{integration_level!r} must be one of {INTEGRATION_LEVELS} "
                f"(source={source})"
            )
        paper_claimable = bool(body.get("paper_claimable", False))
        if paper_claimable and integration_level not in PAPER_CLAIMABLE_LEVELS:
            raise DialectProviderCardError(
                f"dialect provider {dialect_provider_id!r} has "
                f"paper_claimable=true at integration_level={integration_level!r} — "
                f"only {sorted(PAPER_CLAIMABLE_LEVELS)} may set paper_claimable=true "
                f"(source={source})"
            )
        return cls(
            schema_version=schema_version,
            dialect_provider_id=dialect_provider_id,
            dialect_name=dialect_name,
            integration_level=integration_level,
            consumes=tuple(body.get("consumes", ())),
            emits=tuple(body.get("emits", ())),
            entrypoint=str(body.get("entrypoint", "")),
            paper_claimable=paper_claimable,
            required_env=tuple(body.get("required_env", ())),
            description=str(body.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dialect_provider_id": self.dialect_provider_id,
            "dialect_name": self.dialect_name,
            "integration_level": self.integration_level,
            "consumes": list(self.consumes),
            "emits": list(self.emits),
            "entrypoint": self.entrypoint,
            "paper_claimable": self.paper_claimable,
            "required_env": list(self.required_env),
            "description": self.description,
        }
