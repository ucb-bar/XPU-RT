"""Runtime evaluation of promoted synthesized guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from compgen.semantic.synthesis.guard_lang import eval_guard
from compgen.semantic.synthesis.registry import GuardRegistry


@dataclass(frozen=True)
class GuardVerdict:
    """Runtime outcome for one promoted guard artifact."""

    allow: bool
    guard_key: str
    fragments_evaluated: int
    failed_fragment_index: int | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class GuardRuntime:
    """Evaluate promoted guards against a concrete fact environment."""

    def __init__(self, registry: GuardRegistry) -> None:
        self.registry = registry

    def evaluate(self, guard_key: str, env: Mapping[str, Any]) -> GuardVerdict:
        artifact = self.registry.get(guard_key)
        for index, fragment in enumerate(artifact.fragments):
            if not eval_guard(fragment, env):
                return GuardVerdict(
                    allow=False,
                    guard_key=guard_key,
                    fragments_evaluated=len(artifact.fragments),
                    failed_fragment_index=index,
                    reason="fragment_rejected",
                    details={"proof_status": artifact.proof_status, "transform_family": artifact.transform_family},
                )
        return GuardVerdict(
            allow=True,
            guard_key=guard_key,
            fragments_evaluated=len(artifact.fragments),
            reason="guard_matched",
            details={"proof_status": artifact.proof_status, "transform_family": artifact.transform_family},
        )

    def evaluate_many(self, guard_keys: Sequence[str], env: Mapping[str, Any]) -> list[GuardVerdict]:
        return [self.evaluate(guard_key, env) for guard_key in guard_keys]


__all__ = ["GuardRuntime", "GuardVerdict"]
