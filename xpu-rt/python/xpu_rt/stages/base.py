"""Core abstractions for the compilation stages framework.

Each compilation stage defines:
  - Input/output contracts (IR invariants)
  - Shared passes (target-agnostic transformations)
  - A plugin slot for target-specific behavior (what the LLM generates)
  - Verification (contract enforcement)
  - A REQUIREMENTS.md path for LLM generation context

Target-specific behavior is injected via ``TargetStagePlugin`` (composition),
not by subclassing.  This is critical because the LLM generates plugins at
runtime — it cannot subclass an ABC across a process boundary.

Inspired by IREE's interface pattern: define interfaces, let targets implement
them, call interfaces not target code.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# IR Invariants — the atoms of contract checking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IRInvariant:
    """A single verifiable invariant on the IR.

    Attributes:
        name: Human-readable invariant name.
        description: What must be true.
        required_ops: Op types that MUST be present (empty = no requirement).
        forbidden_ops: Op types that MUST NOT be present (empty = no restriction).
        custom_check: Optional callable ``(ModuleOp) -> bool`` for complex checks.
    """

    name: str
    description: str
    required_ops: frozenset[str] = frozenset()
    forbidden_ops: frozenset[str] = frozenset()
    custom_check: Callable[[ModuleOp], bool] | None = None


# ---------------------------------------------------------------------------
# Stage Contract — what a stage promises
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageContract:
    """Input/output contract for a compilation stage.

    Attributes:
        stage_name: Which stage this contract belongs to.
        preconditions: IR invariants that MUST hold before the stage runs.
        postconditions: IR invariants that MUST hold after the stage runs.
        preserved_invariants: Invariants that the stage must NOT violate.
        required_target_info: Keys in TargetProfile this stage needs.
    """

    stage_name: str
    preconditions: list[IRInvariant] = field(default_factory=list)
    postconditions: list[IRInvariant] = field(default_factory=list)
    preserved_invariants: list[IRInvariant] = field(default_factory=list)
    required_target_info: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage Result — what a stage produces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """Result of executing one compilation stage.

    Attributes:
        stage_name: Which stage produced this result.
        module: The IR after this stage (None if stage doesn't modify IR).
        passed: Whether stage completed successfully.
        contract_violations: List of contract violations (empty = pass).
        diagnostics: Informational messages.
        artifacts: Non-IR outputs (YAML plans, kernel files, etc.).
        metrics: Performance/cost metrics from this stage.
    """

    stage_name: str
    module: ModuleOp | None = None
    passed: bool = True
    contract_violations: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Target Stage Plugin — what the LLM generates
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetStagePlugin(Protocol):
    """Protocol for target-specific stage implementations.

    This is what the LLM GENERATES per (target, stage).  The shared stage
    infrastructure calls these methods; the target plugin provides the
    target-specific logic.
    """

    @property
    def target_name(self) -> str: ...

    @property
    def stage_name(self) -> str: ...

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        """Configure the plugin with target information."""
        ...

    def transform(self, module: ModuleOp) -> ModuleOp:
        """Apply target-specific transformations."""
        ...

    def get_artifacts(self) -> dict[str, Any]:
        """Return any non-IR artifacts this plugin produced."""
        ...


# ---------------------------------------------------------------------------
# Compilation Stage — the ABC
# ---------------------------------------------------------------------------


class CompilationStage(abc.ABC):
    """Abstract base class for all compilation stages.

    A CompilationStage owns:
      1. A contract (preconditions / postconditions)
      2. Shared passes (target-agnostic)
      3. A slot for target-specific passes (filled by a TargetStagePlugin)
      4. Verification (contract enforcement)
      5. A REQUIREMENTS.md path for LLM generation
      6. An optional ``llm_phase`` tag (1..7) declaring which pipeline
         phase from ``proposed_compgen_architecture.md`` the stage
         belongs to. Enforces L-XLA-6: Phase 2 stages never read
         runtime state, Phase 5 stages never rewrite Recipe-IR graph
         structure, etc. Opt-in — stages that haven't been classified
         yet leave ``llm_phase = None``.

    Subclasses implement the concrete stage logic.  Target-specific behavior
    is injected via ``TargetStagePlugin``, NOT by subclassing.
    """

    #: Optional phase tag. 1 = normalization, 2 = semantic global opt,
    #: 3 = placement/layout, 4 = kernel contract emission,
    #: 5 = runtime contract, 6 = verify/promote, 7 = package.
    #: Override in subclasses that belong to a specific phase.
    llm_phase: int | None = None

    def __init__(self) -> None:
        self._plugin: TargetStagePlugin | None = None

    # -- Identity --

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stage name (e.g., 'encoding', 'dispatch')."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """One-line description of what this stage does."""

    # -- Contracts --

    @abc.abstractmethod
    def input_contract(self) -> StageContract:
        """Define what IR invariants must hold BEFORE this stage."""

    @abc.abstractmethod
    def output_contract(self) -> StageContract:
        """Define what IR invariants must hold AFTER this stage."""

    # -- Passes --

    @abc.abstractmethod
    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Apply target-agnostic transformations."""

    def target_specific_passes(
        self,
        module: ModuleOp,
        target: TargetProfile,
        capabilities: CapabilitySpec,
    ) -> ModuleOp:
        """Apply target-specific transformations via the plugin.

        If no plugin is registered, returns module unchanged (graceful degradation).
        """
        if self._plugin is not None:
            self._plugin.configure(target, capabilities)
            return self._plugin.transform(module)
        return module

    def get_shared_artifacts(self) -> dict[str, Any]:
        """Return artefacts produced by ``shared_passes``.

        Override in subclasses that write on-disk side-products (e.g.
        :class:`BundleStage` writes ``payload.mlir`` + ``manifest.json``
        and exposes the bundle directory here). Merged into
        ``StageResult.artifacts`` alongside plugin artefacts in
        :meth:`run`.
        """
        return {}

    # -- Execution (template method) --

    def run(
        self,
        module: ModuleOp,
        target: TargetProfile,
        capabilities: CapabilitySpec,
        config: dict[str, Any] | None = None,
    ) -> StageResult:
        """Execute this stage: verify input → shared → plugin → verify output.

        This is NOT abstract — the template method pattern handles sequencing.
        """
        # Delayed import: the trace package depends on ``xdsl`` already
        # being loaded, which transitively pulls this module.
        import time as _stage_time

        from compgen.trace import StagePublisher, get_ir_dump_writer

        diagnostics: list[str] = []
        violations: list[str] = []
        dumper = get_ir_dump_writer()
        stage_t0 = _stage_time.time()
        shared_t0 = shared_t1 = plugin_t0 = plugin_t1 = 0.0

        with StagePublisher.span(
            payload={
                "stage": self.name,
                "target": getattr(target, "name", ""),
                "llm_phase": self.llm_phase,
            }
        ) as span_id:
            if dumper is not None:
                dumper.dump(name=f"stage_{self.name}", phase="entry", module=module, trace_event_id=span_id or "")

            # 1. Verify input contract
            input_violations = self.verify_contract(module, self.input_contract())
            if input_violations:
                return StageResult(
                    stage_name=self.name,
                    module=module,
                    passed=False,
                    contract_violations=[f"INPUT: {v}" for v in input_violations],
                )

            # 2. Run shared passes
            shared_t0 = _stage_time.time()
            try:
                module = self.shared_passes(module, target)
                diagnostics.append(f"{self.name}: shared passes complete")
            except Exception as e:
                return StageResult(
                    stage_name=self.name,
                    module=module,
                    passed=False,
                    contract_violations=[f"shared_passes failed: {e}"],
                )
            shared_t1 = _stage_time.time()
            shared_ms = (shared_t1 - shared_t0) * 1000.0
            if dumper is not None:
                dumper.dump(
                    name=f"stage_{self.name}_shared",
                    phase="after",
                    module=module,
                    trace_event_id=span_id or "",
                    duration_ms=shared_ms,
                )

            # 3. Run target-specific passes
            plugin_t0 = _stage_time.time()
            try:
                module = self.target_specific_passes(module, target, capabilities)
                if self._plugin is not None:
                    diagnostics.append(f"{self.name}: plugin '{self._plugin.target_name}' complete")
            except Exception as e:
                return StageResult(
                    stage_name=self.name,
                    module=module,
                    passed=False,
                    contract_violations=[f"target_specific_passes failed: {e}"],
                )
            plugin_t1 = _stage_time.time()
            plugin_ms = (plugin_t1 - plugin_t0) * 1000.0
            if dumper is not None:
                dumper.dump(
                    name=f"stage_{self.name}_plugin",
                    phase="after",
                    module=module,
                    trace_event_id=span_id or "",
                    duration_ms=plugin_ms,
                )

            # 4. Verify output contract
            output_violations = self.verify_contract(module, self.output_contract())
            violations.extend(f"OUTPUT: {v}" for v in output_violations)

            # 5. Collect artifacts from shared passes + plugin.
            # Shared artefacts win where names collide (plugins can
            # augment, but the stage's core on-disk outputs are
            # authoritative).
            artifacts: dict[str, Any] = {}
            if self._plugin is not None:
                artifacts = dict(self._plugin.get_artifacts())
            shared = self.get_shared_artifacts()
            if shared:
                artifacts.update(shared)

            stage_ms = (_stage_time.time() - stage_t0) * 1000.0
            if dumper is not None:
                dumper.dump(
                    name=f"stage_{self.name}",
                    phase="exit",
                    module=module,
                    trace_event_id=span_id or "",
                    duration_ms=stage_ms,
                    meta={
                        "shared_ms": round(shared_ms, 3),
                        "plugin_ms": round(plugin_ms, 3),
                    },
                )

            return StageResult(
                stage_name=self.name,
                module=module,
                passed=len(violations) == 0,
                contract_violations=violations,
                diagnostics=diagnostics,
                artifacts=artifacts,
            )

    # -- Verification --

    def verify_contract(self, module: ModuleOp, contract: StageContract) -> list[str]:
        """Check all invariants in a contract. Returns violation messages."""
        violations: list[str] = []
        all_invariants = contract.preconditions + contract.postconditions + contract.preserved_invariants

        for inv in all_invariants:
            # Check required ops
            if inv.required_ops:
                op_names = {op.name for op in module.walk()}
                missing = inv.required_ops - op_names
                if missing:
                    violations.append(f"{inv.name}: missing required ops {missing}")

            # Check forbidden ops
            if inv.forbidden_ops:
                op_names = {op.name for op in module.walk()}
                found = inv.forbidden_ops & op_names
                if found:
                    violations.append(f"{inv.name}: forbidden ops present {found}")

            # Check custom predicate
            if inv.custom_check is not None:
                try:
                    if not inv.custom_check(module):
                        violations.append(f"{inv.name}: custom check failed")
                except Exception as e:
                    violations.append(f"{inv.name}: custom check error: {e}")

        return violations

    # -- Plugin management --

    def register_plugin(self, plugin: TargetStagePlugin) -> None:
        """Register a target-specific plugin for this stage."""
        if plugin.stage_name != self.name:
            raise ValueError(f"Plugin stage_name '{plugin.stage_name}' does not match stage '{self.name}'")
        self._plugin = plugin
        log.debug("stage.plugin_registered", stage=self.name, target=plugin.target_name)

    @property
    def has_plugin(self) -> bool:
        """Whether a target-specific plugin is registered."""
        return self._plugin is not None

    # -- LLM Generation --

    @abc.abstractmethod
    def requirements_doc_path(self) -> Path:
        """Path to the REQUIREMENTS.md for this stage."""
