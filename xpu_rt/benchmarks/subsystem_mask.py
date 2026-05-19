"""Typed mask for prove-or-kill subsystem ablations.

`SubsystemMask` is a frozen dataclass of bool fields, one per
subsystem and one per agent-routable compilation pass. The
foundation step (this module + `subsystem_ablation.py`) defines the
mask, plumbs it through `run_graph_compilation` via the manifest,
and emits a typed diff between two runs.

Per-subsystem "off" behavior — the deterministic fallback each
subsystem must short-circuit to when its bit is `False` — is wired
phase by phase in follow-up PRs (Phase 1: kernels + agent; Phase 2:
eqsat + memory; Phase 3: capture + ir + analysis). Subsystems that
have not yet been wired raise `SubsystemMaskUnwiredError` if their
flag is flipped off, so a stale ablation run can't silently report
"no signal" because the off-path was never implemented.

The mask is intentionally non-hierarchical: every flag is a leaf.
A "subsystem-level" off (e.g. all of `agent.decisions.*`) is
expressed by flipping every leaf to `False` via
`SubsystemMask.with_subsystem_off("agent.decisions")`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, fields
from typing import Any, ClassVar


class SubsystemMaskError(ValueError):
    """Base for mask-related typed errors."""


class SubsystemMaskUnknownFlagError(SubsystemMaskError):
    """Caller asked for a flag name that is not on the mask."""


class SubsystemMaskUnwiredError(SubsystemMaskError):
    """A flag was flipped off but its off-path is not yet implemented.

    Raised at the subsystem entry point. The error message names the
    flag and the phase that owns wiring it. This is the contract that
    prevents silent "no signal" results in the ablation harness.
    """


@dataclass(frozen=True)
class SubsystemMask:
    """Boolean mask over xpu_rt subsystems and agent-routable passes.

    Field naming convention: `<subsystem>__<leaf>` (double underscore
    so the canonical dotted name `subsystem.leaf` survives a round
    trip through field-name attribute access). The public string
    name uses dots: `kernels.codegen_fallback`.
    """

    # --- Phase 1: kernels + agent ------------------------------------------ #
    kernels__codegen_fallback: bool = True
    kernels__contract_v3: bool = True
    agent__decisions__tiling: bool = True
    agent__decisions__fusion: bool = True
    agent__decisions__encoding: bool = True
    agent__decisions__codegen_backend: bool = True
    agent__hw_aware_dispatch: bool = True

    # --- Phase 2: eqsat + memory ------------------------------------------- #
    eqsat__pipeline: bool = True
    eqsat__cost_model: bool = True
    eqsat__rules__fusion: bool = True
    eqsat__rules__fission: bool = True
    eqsat__rules__scheduling: bool = True
    memory__knowledge: bool = True
    memory__kernel_db: bool = True
    memory__embeddings: bool = True
    memory__calibration: bool = True

    # --- Phase 3: capture + ir + analysis ---------------------------------- #
    capture__dynamo_baseline: bool = True
    capture__inductor_harvest: bool = True
    capture__torchao_pipeline: bool = True
    ir__recipe: bool = True
    ir__event: bool = True
    analysis__pass_pool: bool = True

    # Phase that owns wiring each flag. Subsystem entry points consult
    # this map (via `is_wired`) before applying the mask.
    _PHASE_OF_FLAG: ClassVar[dict[str, str]] = {
        # Phase 1
        "kernels.codegen_fallback": "phase1",
        "kernels.contract_v3": "phase1",
        "agent.decisions.tiling": "phase1",
        "agent.decisions.fusion": "phase1",
        "agent.decisions.encoding": "phase1",
        "agent.decisions.codegen_backend": "phase1",
        "agent.hw_aware_dispatch": "phase1",
        # Phase 2
        "eqsat.pipeline": "phase2",
        "eqsat.cost_model": "phase2",
        "eqsat.rules.fusion": "phase2",
        "eqsat.rules.fission": "phase2",
        "eqsat.rules.scheduling": "phase2",
        "memory.knowledge": "phase2",
        "memory.kernel_db": "phase2",
        "memory.embeddings": "phase2",
        "memory.calibration": "phase2",
        # Phase 3
        "capture.dynamo_baseline": "phase3",
        "capture.inductor_harvest": "phase3",
        "capture.torchao_pipeline": "phase3",
        "ir.recipe": "phase3",
        "ir.event": "phase3",
        "analysis.pass_pool": "phase3",
    }

    # Flags whose deterministic off-path has shipped. Updated as
    # each subsystem's wiring lands. Read by `check_wired`. Wiring
    # is tracked per-flag (not per-phase) so flipping one Phase-1
    # flag doesn't falsely claim its siblings are wired.
    #
    # Each entry must point at a single read site in the codebase
    # that consults `active_mask_from_env()` and short-circuits to a
    # deterministic fallback. The current wired set:
    #
    # - `agent.decisions.fusion`
    #     → xpu_rt.graph_compilation.action_space.build_action_space
    #       (skips `_gen_fusion`; no fuse_producer_consumer
    #       candidates surface, so no agent or greedy selector can
    #       pick fusion).
    # - `agent.decisions.tiling`
    #     → same site (skips `_gen_tiling`; no set_tile_params
    #       candidates surface; selectors fall back to whatever the
    #       codegen backend's default tile is).
    # - `memory.knowledge`
    #     → xpu_rt.graph_compilation.recipe_planning.run_recipe_planning
    #       (forces promoted_ids to empty so greedy's warm-cache tier
    #       never fires; isolates the promoted-recipe library's
    #       delivered value).
    # - `eqsat.pipeline`
    #     → xpu_rt.api.compile_model (skips run_eqsat_pass, synthesizes
    #       a zero-effect EqSatResult). NOTE: as of commit 4eab92c4112a,
    #       eqsat is non-load-bearing in compile_model — its rewritten
    #       module is dropped on the floor (no `.module` field on
    #       EqSatResult; downstream pipeline uses the pre-eqsat module).
    #       The flag is wired for completeness; toggling it cannot
    #       affect runtime latency until eqsat output is propagated.
    # - `kernels.codegen_fallback`
    #     → xpu_rt.api.compile_model (skips run_provider_fallback when
    #       the auto-codegen stage didn't ship a native kernel).
    #       Lives in the compile_model path only — not reachable from
    #       run_graph_compilation. Expected null on host_cpu since no
    #       real Triton/Exo provider is registered.
    _WIRED_FLAGS: ClassVar[frozenset[str]] = frozenset({
        "agent.decisions.fusion",
        "agent.decisions.tiling",
        "memory.knowledge",
        "eqsat.pipeline",
        "kernels.codegen_fallback",
    })

    @staticmethod
    def _field_name(dotted: str) -> str:
        return dotted.replace(".", "__")

    @staticmethod
    def _dotted_name(field_name: str) -> str:
        return field_name.replace("__", ".")

    @classmethod
    def all_on(cls) -> SubsystemMask:
        return cls()

    @classmethod
    def all_off(cls) -> SubsystemMask:
        return cls(**{f.name: False for f in fields(cls)})

    @classmethod
    def from_disable_list(cls, names: list[str] | tuple[str, ...]) -> SubsystemMask:
        """Build a mask with the named flags flipped to False.

        Names use dotted form: `kernels.codegen_fallback`. A name may
        also be a subsystem prefix (`agent.decisions`) — every leaf
        whose dotted name starts with `<prefix>.` is flipped off.
        """
        known = cls.flag_names()
        overrides: dict[str, bool] = {}
        for raw in names:
            name = raw.strip()
            if not name:
                continue
            if name in known:
                overrides[cls._field_name(name)] = False
                continue
            prefix = name + "."
            matched = [n for n in known if n.startswith(prefix)]
            if not matched:
                raise SubsystemMaskUnknownFlagError(
                    f"unknown subsystem flag or prefix: {name!r}. "
                    f"Known flags: {sorted(known)}"
                )
            for m in matched:
                overrides[cls._field_name(m)] = False
        return cls(**overrides)

    @classmethod
    def flag_names(cls) -> tuple[str, ...]:
        """Dotted names of every leaf flag, sorted."""
        return tuple(sorted(cls._dotted_name(f.name) for f in fields(cls)))

    def get(self, dotted: str) -> bool:
        if dotted not in self._PHASE_OF_FLAG:
            raise SubsystemMaskUnknownFlagError(
                f"unknown subsystem flag: {dotted!r}"
            )
        return bool(getattr(self, self._field_name(dotted)))

    def with_subsystem_off(self, prefix: str) -> SubsystemMask:
        """Return a copy with every leaf under `prefix` flipped off."""
        return type(self).from_disable_list([prefix]) if prefix in self.flag_names() else (
            dataclasses.replace(self, **{
                self._field_name(n): False
                for n in self.flag_names() if n.startswith(prefix + ".")
            })
        )

    def disabled_flags(self) -> tuple[str, ...]:
        return tuple(
            self._dotted_name(f.name)
            for f in fields(self)
            if getattr(self, f.name) is False
        )

    def to_dict(self) -> dict[str, bool]:
        return {self._dotted_name(f.name): bool(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SubsystemMask:
        kwargs: dict[str, bool] = {}
        for k, v in payload.items():
            field_name = cls._field_name(k)
            if field_name not in {f.name for f in fields(cls)}:
                raise SubsystemMaskUnknownFlagError(
                    f"unknown subsystem flag: {k!r}"
                )
            kwargs[field_name] = bool(v)
        return cls(**kwargs)

    def is_wired(self, dotted: str) -> bool:
        """Has the off-path for this flag been implemented yet?"""
        if dotted not in self._PHASE_OF_FLAG:
            raise SubsystemMaskUnknownFlagError(
                f"unknown subsystem flag: {dotted!r}"
            )
        return dotted in self._WIRED_FLAGS

    def check_wired(self, dotted: str) -> None:
        """Call from a subsystem entry point when the flag is off.

        Raises `SubsystemMaskUnwiredError` if the flag's off-path
        has not yet shipped. The harness foundation PR ships no
        off-paths — every off-flag raises until its wiring PR adds
        the flag name to ``_WIRED_FLAGS``.
        """
        if self.get(dotted):
            return
        if not self.is_wired(dotted):
            phase = self._PHASE_OF_FLAG[dotted]
            raise SubsystemMaskUnwiredError(
                f"subsystem flag {dotted!r} flipped off, but its "
                f"off-path has not been wired yet (owner: {phase}). "
                f"Either run with {dotted}=True, or land the wiring "
                f"PR that adds {dotted!r} to SubsystemMask._WIRED_FLAGS "
                f"and implements the deterministic fallback."
            )


# --------------------------------------------------------------------------- #
# Active-mask resolution
# --------------------------------------------------------------------------- #


_ACTIVE_MASK_ENV = "XPU_RT_SUBSYSTEM_MASK"


def active_mask_from_env() -> SubsystemMask | None:
    """Read a comma-separated disable list from the env var.

    Returns None when the env var is unset or empty. Subsystem entry
    points should prefer an explicit `mask` parameter; this hook
    exists so legacy callers can disable subsystems without touching
    the signature of every intermediate function.
    """
    raw = os.environ.get(_ACTIVE_MASK_ENV, "").strip()
    if not raw:
        return None
    disable_list = [p.strip() for p in raw.split(",") if p.strip()]
    return SubsystemMask.from_disable_list(disable_list)
