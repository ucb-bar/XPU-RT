"""Escalating provider router — Claude Code default, autocomp on miss.

Reads ``KernelContractV3.selection.providers`` (or a configured order),
tries each provider in turn, escalates on:

* **correctness miss**  : provider's result fails ``correct``-flag check
  or downstream verification gate.
* **performance miss**  : provider's ``latency_us`` exceeds
  ``perf_target_us`` by more than ``perf_slack_factor`` (default 2×).
* **explicit no-find**  : provider returns ``found=False``.

The router does *not* call the verification gate itself — the caller
passes in a callable that runs the gate against the candidate kernel.
That keeps the router orthogonal to which gates exist (TV, differential,
profile_budget …).

Cost-efficiency intent: every kernel routes through Claude Code first
(~$0.05 / 0$ in-session), and only escalates to autocomp (~$13-20) on the
~5% of kernels where one-shot can't match the perf budget. See the
contract-cost analysis in ``docs/architecture/kernel_contracts.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compgen.kernels.provider import (
    KernelContract,
    KernelProvider,
    ProviderResult,
    SearchBudget,
)


class EscalationReason(Enum):
    NONE = "none"                   # accepted on first try
    NOT_FOUND = "not_found"          # provider returned found=False
    CORRECTNESS = "correctness"      # gate rejected as wrong
    PERFORMANCE = "performance"      # exceeded perf_target_us × slack
    PROVIDER_ERROR = "provider_error"  # provider raised


@dataclass(frozen=True)
class RouteOutcome:
    """What the router decided for one contract."""

    result: ProviderResult
    chosen_provider: str
    escalation_path: tuple[str, ...]  # provider names tried, in order
    final_reason: EscalationReason
    perf_target_us: float | None = None


# A gate callable: takes (contract, ProviderResult) and returns
# ``(passed, reason_str)``. ``passed=False`` triggers escalation.
GateCallable = Callable[[KernelContract, ProviderResult], tuple[bool, str]]


def _accept_all_gate(_contract: KernelContract, result: ProviderResult) -> tuple[bool, str]:
    """Default gate: accept whatever the provider returned (only the
    found-flag triggers escalation in this mode)."""
    return result.found, "" if result.found else "provider returned found=False"


@dataclass
class EscalatingProviderRouter:
    """Try providers in order; escalate on gate failure or perf miss.

    Attributes:
        providers: Ordered list. First = default (Claude Code), later =
            escalation tiers (autocomp last).
        gate: Optional correctness gate. If None, only ``result.found``
            is used to decide escalation.
        perf_slack_factor: Allowable slowdown vs ``perf_target_us``
            before escalating. Default 2.0 = "if it's >2× the target,
            escalate". ``None`` disables perf-based escalation.
    """

    providers: list[KernelProvider]
    gate: GateCallable | None = None
    perf_slack_factor: float | None = 2.0

    def route(
        self,
        contract: KernelContract,
        budget: SearchBudget,
        *,
        perf_target_us: float | None = None,
    ) -> RouteOutcome:
        """Walk the provider ladder, returning the first acceptable result.

        Order of checks per provider:
            1. ``provider.accepts_contract`` — skip if False.
            2. ``provider.search`` — capture exceptions as PROVIDER_ERROR.
            3. ``found`` flag — escalate if False.
            4. Caller gate — escalate on (False, reason).
            5. Perf budget — escalate if latency_us > perf_target_us × slack.

        The first provider whose result clears all checks wins. If every
        provider escalates, the last result is returned with the final
        escalation reason in ``RouteOutcome.final_reason``.
        """
        gate = self.gate or _accept_all_gate
        path: list[str] = []
        last_result = ProviderResult()
        last_provider = ""
        last_reason = EscalationReason.NOT_FOUND

        for provider in self.providers:
            if not provider.accepts_contract(contract):
                continue
            path.append(provider.name)
            last_provider = provider.name

            try:
                result = provider.search(contract, budget)
            except Exception as exc:  # noqa: BLE001
                last_result = ProviderResult(
                    found=False,
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
                last_reason = EscalationReason.PROVIDER_ERROR
                continue

            last_result = result

            if not result.found:
                last_reason = EscalationReason.NOT_FOUND
                continue

            passed, gate_reason = gate(contract, result)
            if not passed:
                last_reason = EscalationReason.CORRECTNESS
                # Tag the gate reason on the result for the next tier.
                last_result = ProviderResult(
                    **{**result.__dict__,
                       "metadata": {**result.metadata, "gate_reason": gate_reason}},
                )
                continue

            if (
                perf_target_us is not None
                and self.perf_slack_factor is not None
                and result.latency_us > 0
                and result.latency_us > perf_target_us * self.perf_slack_factor
            ):
                last_reason = EscalationReason.PERFORMANCE
                continue

            return RouteOutcome(
                result=result,
                chosen_provider=provider.name,
                escalation_path=tuple(path),
                final_reason=EscalationReason.NONE,
                perf_target_us=perf_target_us,
            )

        # Every provider escalated; return the last one's result so the
        # caller can decide what to do (return-as-is / hand-write / fail).
        return RouteOutcome(
            result=last_result,
            chosen_provider=last_provider,
            escalation_path=tuple(path),
            final_reason=last_reason,
            perf_target_us=perf_target_us,
        )


__all__ = [
    "EscalatingProviderRouter",
    "EscalationReason",
    "GateCallable",
    "RouteOutcome",
]
