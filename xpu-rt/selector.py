"""Deterministic epoch-level candidate selector.

Chooses which precomputed schedule candidate to run in the next epoch from
observed freshness risk. Deliberately rule-based and stateless-per-decision
(all state is explicit in :class:`SelectorState`) so a decision can be replayed
and audited, and so the same code can later drive a runtime hot-swap rather than
only an offline replay.

    risk = observed_max_input_age / freshness_window

CAUSALITY IS THE LOAD-BEARING PROPERTY
--------------------------------------
The decision for epoch k is a function of observations from epochs < k only.
`decide()` takes the PREVIOUS epoch's observed age and returns the candidate for
the NEXT epoch. Feeding it the age observed during epoch k would make the
selector clairvoyant, and its advantage over any static candidate would be an
artifact of that. This is the single easiest way to accidentally fabricate a
positive result here, so the signature makes the lag explicit and
`SelectorLogRow` records which epoch each observation came from.

A consequence worth stating in any report: the selector CANNOT prevent the
staleness it reacts to. It observes a violation and changes the next epoch. Its
value, if any, is in how quickly it escalates and how little utility it gives up
once the danger passes -- not in achieving zero stale outputs.

Escalation is asymmetric, which is a safety choice rather than a tuning one:

  * escalate DIRECTLY to the highest level whose entry threshold the observed
    risk exceeds -- danger gets the strongest available response immediately;
  * de-escalate ONE level per decision -- recovery is released gradually, so a
    single quiet epoch cannot drop protection all the way to nominal.

Three independent brakes prevent chattering. They can bind simultaneously and
each is recorded separately in the log so a "no switch" outcome is never
ambiguous:

  * hysteresis     -- exit threshold is strictly below entry threshold
  * min_residency  -- epochs that must elapse since ENTERING the current level
  * cooldown       -- epochs that must elapse since the LAST switch of any kind
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# Reasons a decision did or did not switch. Recorded verbatim in selector_log.csv.
HELD_NO_TRIGGER = "held_no_trigger"
HELD_HYSTERESIS = "held_hysteresis"
HELD_MIN_RESIDENCY = "held_min_residency"
HELD_COOLDOWN = "held_cooldown"
HELD_AT_CEILING = "held_at_max_protection"  # saturated AND still above threshold
ESCALATED = "escalated"
DEESCALATED = "deescalated"
BOOTSTRAP = "bootstrap_no_observation"

SWITCH_REASONS = (ESCALATED, DEESCALATED)


@dataclass(frozen=True)
class CandidateLevel:
    """One rung of the protection ladder.

    `entry_risk` is the risk at or above which this level becomes the target.
    `exit_risk` is the risk strictly below which the selector may step DOWN out
    of it. exit_risk < entry_risk is the hysteresis band and is enforced at
    construction, because equal thresholds oscillate every epoch at the
    boundary.
    """

    candidate_id: str
    protection_level: int
    entry_risk: float
    exit_risk: float
    intent: str = ""

    def __post_init__(self) -> None:
        if self.exit_risk >= self.entry_risk and self.protection_level > 0:
            raise ValueError(
                f"{self.candidate_id}: exit_risk {self.exit_risk} must be "
                f"strictly below entry_risk {self.entry_risk}; equal thresholds "
                f"oscillate at the boundary"
            )


@dataclass
class SelectorConfig:
    levels: Tuple[CandidateLevel, ...]
    min_residency_epochs: int = 2
    cooldown_epochs: int = 1

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("selector needs at least one candidate level")
        levels = sorted(self.levels, key=lambda c: c.protection_level)
        if [c.protection_level for c in levels] != list(range(len(levels))):
            raise ValueError(
                f"protection_level must be 0..N-1 with no gaps or duplicates, "
                f"got {[c.protection_level for c in levels]}"
            )
        # Entry thresholds must increase with protection, or a higher level
        # could never be the argmax target and would be unreachable.
        for lo, hi in zip(levels, levels[1:]):
            if hi.entry_risk <= lo.entry_risk:
                raise ValueError(
                    f"{hi.candidate_id} entry_risk {hi.entry_risk} must exceed "
                    f"{lo.candidate_id}'s {lo.entry_risk}, else it is unreachable"
                )
        if self.min_residency_epochs < 0 or self.cooldown_epochs < 0:
            raise ValueError("residency and cooldown must be non-negative")
        self.levels = tuple(levels)

    @property
    def max_level(self) -> int:
        return self.levels[-1].protection_level

    def by_level(self, level: int) -> CandidateLevel:
        return self.levels[level]


@dataclass
class SelectorState:
    current_level: int = 0
    epochs_at_current_level: int = 0
    epochs_since_switch: int = 10**6  # unconstrained before the first switch
    switch_count: int = 0


@dataclass
class SelectorLogRow:
    epoch: int
    observation_from_epoch: Optional[int]
    observed_max_age: Optional[float]
    freshness_window: float
    risk: Optional[float]
    level_before: int
    level_after: int
    candidate_before: str
    candidate_after: str
    target_level: int
    reason: str
    switched: bool
    switch_count: int
    epochs_at_level_before: int
    epochs_since_switch_before: int
    blocked_by: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


CSV_COLUMNS: Tuple[str, ...] = tuple(SelectorLogRow.__annotations__)


class Selector:
    """Replayable epoch-level selector.

    Usage is one `decide()` per epoch boundary, in order. The returned
    candidate_id is what should run in `epoch`, chosen from the observation
    supplied for an EARLIER epoch.
    """

    def __init__(self, config: SelectorConfig,
                 state: Optional[SelectorState] = None) -> None:
        self.config = config
        self.state = state or SelectorState()
        self.log: List[SelectorLogRow] = []

    def _target_level(self, risk: float) -> int:
        """Highest level whose entry threshold the risk meets."""
        target = 0
        for lvl in self.config.levels:
            if risk >= lvl.entry_risk:
                target = lvl.protection_level
        return target

    def decide(
        self,
        epoch: int,
        observed_max_age: Optional[float],
        freshness_window: float,
        *,
        observation_from_epoch: Optional[int] = None,
    ) -> str:
        """Pick the candidate for `epoch` from an age observed BEFORE it.

        `observed_max_age` is None for the first epoch, when nothing has been
        observed yet -- the selector must start somewhere, and it starts at the
        nominal level. That bootstrap epoch is unprotected by construction and
        is labelled BOOTSTRAP so it is never mistaken for a decision.
        """
        if freshness_window <= 0:
            raise ValueError(f"freshness_window must be positive, got {freshness_window}")
        if observation_from_epoch is not None and observation_from_epoch >= epoch:
            # The whole point: a decision may not see its own epoch.
            raise ValueError(
                f"observation_from_epoch {observation_from_epoch} must precede "
                f"the epoch being decided ({epoch}); a selector that reads the "
                f"epoch it is choosing for is an oracle, not a policy"
            )

        st = self.state
        before_level = st.current_level
        before_at_level = st.epochs_at_current_level
        before_since_switch = st.epochs_since_switch
        cfg = self.config

        risk = None if observed_max_age is None else observed_max_age / freshness_window

        blocked: List[str] = []
        if risk is None:
            target = before_level
            reason = BOOTSTRAP
        else:
            target = self._target_level(risk)
            cur = cfg.by_level(before_level)

            if target > before_level:
                reason = ESCALATED
            elif target < before_level:
                # Hysteresis: only step down once risk clears the CURRENT
                # level's exit threshold, which sits below its entry threshold.
                if risk >= cur.exit_risk:
                    reason = HELD_HYSTERESIS
                    blocked.append(HELD_HYSTERESIS)
                    target = before_level
                else:
                    target = before_level - 1  # one rung at a time
                    reason = DEESCALATED
            else:
                reason = HELD_NO_TRIGGER

            # Brakes apply to switches only, and never block ESCALATION away
            # from an unsafe state... except that they do, deliberately: a
            # selector that can escalate every epoch chatters just as badly as
            # one that de-escalates every epoch. Both directions are rate
            # limited, and the report must say so, because it bounds how fast
            # protection can arrive.
            if reason in SWITCH_REASONS:
                if before_at_level < cfg.min_residency_epochs:
                    blocked.append(HELD_MIN_RESIDENCY)
                if before_since_switch < cfg.cooldown_epochs:
                    blocked.append(HELD_COOLDOWN)
                if blocked:
                    # Report the FIRST brake that bound, and every brake that
                    # bound in `blocked_by`, so "no switch" is never ambiguous
                    # between "nothing triggered" and "triggered but held".
                    reason = blocked[0]
                    target = before_level

            if target == before_level and reason in SWITCH_REASONS:
                reason = HELD_NO_TRIGGER  # defensive; brakes already handle this

            # Saturation. Sitting at the top of the ladder while risk is STILL
            # at or above that level's entry threshold is operationally distinct
            # from "nothing triggered": it means the strongest candidate
            # available is not enough. Reported separately because a run that
            # spends most of its epochs here has not demonstrated that adaptive
            # selection works -- it has demonstrated that the candidate set is
            # inadequate, which is a different conclusion with a different fix.
            if (before_level == cfg.max_level
                    and target == before_level
                    and risk >= cfg.by_level(cfg.max_level).entry_risk):
                reason = HELD_AT_CEILING

        switched = target != before_level
        st.current_level = target
        if switched:
            st.switch_count += 1
            st.epochs_at_current_level = 0
            st.epochs_since_switch = 0
        else:
            st.epochs_at_current_level = before_at_level + 1
            st.epochs_since_switch = min(before_since_switch + 1, 10**6)

        self.log.append(SelectorLogRow(
            epoch=epoch,
            observation_from_epoch=observation_from_epoch,
            observed_max_age=observed_max_age,
            freshness_window=freshness_window,
            risk=risk,
            level_before=before_level,
            level_after=target,
            candidate_before=cfg.by_level(before_level).candidate_id,
            candidate_after=cfg.by_level(target).candidate_id,
            target_level=self._target_level(risk) if risk is not None else before_level,
            reason=reason,
            switched=switched,
            switch_count=st.switch_count,
            epochs_at_level_before=before_at_level,
            epochs_since_switch_before=(
                before_since_switch if before_since_switch < 10**6 else -1
            ),
            blocked_by="|".join(blocked),
        ))
        return cfg.by_level(target).candidate_id

    # --- reporting -------------------------------------------------------

    def rows(self) -> List[Dict[str, object]]:
        return [r.as_dict() for r in self.log]

    def summary(self) -> Dict[str, object]:
        """Aggregates Phase 11 asks for: switch count, residency distribution,
        and how often each brake bound."""
        if not self.log:
            return {"n_epochs": 0, "switch_count": 0}
        time_in: Dict[str, int] = {}
        for r in self.log:
            time_in[r.candidate_after] = time_in.get(r.candidate_after, 0) + 1
        blocked_counts: Dict[str, int] = {}
        for r in self.log:
            for b in filter(None, r.blocked_by.split("|")):
                blocked_counts[b] = blocked_counts.get(b, 0) + 1
        switch_epochs = [r.epoch for r in self.log if r.switched]
        return {
            "n_epochs": len(self.log),
            "switch_count": self.state.switch_count,
            "switches_per_epoch": self.state.switch_count / len(self.log),
            "switch_epochs": switch_epochs,
            "epochs_in_candidate": time_in,
            "fraction_in_candidate": {
                k: v / len(self.log) for k, v in time_in.items()
            },
            "blocked_by_counts": blocked_counts,
            "final_level": self.state.current_level,
            "escalations": sum(1 for r in self.log if r.reason == ESCALATED),
            "deescalations": sum(1 for r in self.log if r.reason == DEESCALATED),
        }


def replay(
    config: SelectorConfig,
    observed_ages_by_epoch: Sequence[Optional[float]],
    freshness_window: float,
    *,
    lag: int = 1,
) -> Selector:
    """Replay a whole run: epoch k is decided from the age observed at k-lag.

    `lag` defaults to 1 -- one epoch of telemetry delay, the minimum physically
    realisable value. It is a parameter rather than a constant so the cost of a
    slower telemetry path can be measured instead of assumed.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1 epoch; {lag} would let the selector "
                         f"see the epoch it is deciding")
    sel = Selector(config)
    for epoch in range(len(observed_ages_by_epoch)):
        src = epoch - lag
        if src < 0:
            sel.decide(epoch, None, freshness_window, observation_from_epoch=None)
        else:
            sel.decide(epoch, observed_ages_by_epoch[src], freshness_window,
                       observation_from_epoch=src)
    return sel
