"""Action taxonomy + step-result types for CompilerEnv.

Every environment action is one of these frozen dataclasses.  They carry
no behavior — dispatch happens inside CompilerEnv.step().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.agent.env.observations import Observation

# ============================================================================
# Actions: what the agent can do (validated before execution)
# ============================================================================


@dataclass(frozen=True)
class Action:
    """Base for all agent actions."""

    action_type: str
    region_id: str = ""


@dataclass(frozen=True)
class TileAction(Action):
    """Tile a matmul/linalg op."""

    action_type: str = "tile"
    tile_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class FuseAction(Action):
    """Fuse two adjacent regions."""

    action_type: str = "fuse"
    target_region_id: str = ""       # second region to fuse with


@dataclass(frozen=True)
class AssignDeviceAction(Action):
    """Place a region on a device."""

    action_type: str = "assign_device"
    device_index: int = 0


@dataclass(frozen=True)
class SetDtypeAction(Action):
    """Change dtype of a region's computation."""

    action_type: str = "set_dtype"
    dtype: str = "f16"


@dataclass(frozen=True)
class InsertCopyAction(Action):
    """Insert a cross-device copy."""

    action_type: str = "insert_copy"
    target_region_id: str = ""
    async_: bool = True


@dataclass(frozen=True)
class NoopAction(Action):
    """Do nothing (pass this turn)."""

    action_type: str = "noop"


@dataclass(frozen=True)
class GeneralizeAction(Action):
    """Generalize a named linalg op (matmul→generic). Exposes computation body."""

    action_type: str = "generalize"


@dataclass(frozen=True)
class ApplyPassAction(Action):
    """Apply a registered xDSL compiler pass."""

    action_type: str = "apply_pass"
    pass_name: str = ""
    pass_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointAction(Action):
    """Save current state for speculative exploration."""

    action_type: str = "checkpoint"


@dataclass(frozen=True)
class RollbackAction(Action):
    """Restore to last checkpoint."""

    action_type: str = "rollback"


@dataclass(frozen=True)
class InspectAction(Action):
    """Request detailed info about a region (selective attention)."""

    action_type: str = "inspect"


@dataclass(frozen=True)
class AnalyzeAction(Action):
    """Run network analysis — detect patterns, bottlenecks, kernel opportunities."""

    action_type: str = "analyze"


@dataclass(frozen=True)
class SearchKernelAction(Action):
    """Search for an optimized kernel for a pattern cluster via Autocomp."""

    action_type: str = "search_kernel"
    cluster_id: str = ""
    budget: int = 10


@dataclass(frozen=True)
class GeneratePassAction(Action):
    """Ask the LLM to generate a new xDSL compiler pass."""

    action_type: str = "generate_pass"
    description: str = ""
    target_pattern: str = ""
    expected_effect: str = ""


@dataclass(frozen=True)
class SolveAction(Action):
    """Run solver for heterogeneous placement and/or scheduling."""

    action_type: str = "solve"
    solve_type: str = "placement"  # "placement", "schedule", "both"
    timeout_ms: int = 10000


@dataclass(frozen=True)
class BenchmarkAction(Action):
    """Run real hardware benchmark and get ground-truth measurements."""

    action_type: str = "benchmark"
    device: str = "cuda"             # "cpu" or "cuda"
    mode: str = "eager"              # "eager", "compiled"
    num_iterations: int = 100


@dataclass(frozen=True)
class CalibrateAction(Action):
    """Calibrate cost model from benchmark results."""

    action_type: str = "calibrate"


@dataclass(frozen=True)
class DiscoverOpsAction(Action):
    """Scan FX graph for unknown ops and generate support."""

    action_type: str = "discover_ops"
    auto_generate: bool = False      # if True, use LLM to generate decompositions


@dataclass(frozen=True)
class CompileAndRunAction(Action):
    """Compile with agent decisions via torch.compile and benchmark."""

    action_type: str = "compile_and_run"
    device: str = "cuda"
    num_iterations: int = 50


@dataclass(frozen=True)
class EqSatAction(Action):
    """Run equality saturation on current IR.

    Explores equivalent computational forms and extracts the cheapest
    via a global cost model. Rules are selected by category.
    """

    action_type: str = "eqsat"
    rule_categories: tuple[str, ...] = ("algebraic",)
    max_iterations: int = 10
    segment_threshold: int = 200
    blackbox_overrides: dict[str, str] = field(default_factory=dict)
    segment_threshold_override: int | None = None


@dataclass(frozen=True)
class ProposeRuleAction(Action):
    """Propose a new rewrite rule for the e-graph.

    The rule is a Python RewritePattern (generated by LLM or user).
    It is validated before being added to the rule registry.
    """

    action_type: str = "propose_rule"
    rule_code: str = ""
    category: str = "llm_generated"


@dataclass(frozen=True)
class InspectEGraphAction(Action):
    """Inspect the current e-graph state.

    Returns a summary: e-class count, e-node count, ambiguous regions,
    extraction forks, and rule statistics.
    """

    action_type: str = "inspect_egraph"


@dataclass(frozen=True)
class SetExtractionObjectiveAction(Action):
    """Adjust extraction cost model weights.

    Lets the agent tune what the extractor optimizes for.
    """

    action_type: str = "set_extraction_objective"
    fusion_weight: float = 1.0
    transfer_weight: float = 1.0
    backend_match_weight: float = 1.0


@dataclass(frozen=True)
class ConfigureProfilingAction(Action):
    """Configure profiling for the current target.

    The LLM selects instrumentation level, counters, and custom hooks
    based on detected bottlenecks and target capabilities.
    """

    action_type: str = "configure_profiling"
    instrumentation_level: str = "op_level"  # "none", "op_level", "tile_level", "full"
    counters: list[str] = field(default_factory=list)
    custom_hooks: dict[str, str] = field(default_factory=dict)
    analysis_focus: str = "latency"


@dataclass(frozen=True)
class ConfigureDispatchAction(Action):
    """Configure dispatch strategy for heterogeneous execution.

    The LLM selects the dispatch strategy and transport parameters
    based on topology and workload characteristics.
    """

    action_type: str = "configure_dispatch"
    strategy: str = "bulk_sync"  # "bulk_sync", "pipeline", "wavefront", "streaming"
    transport_overrides: dict[str, str] = field(default_factory=dict)
    thread_config: dict[str, int] = field(default_factory=dict)
    double_buffer: bool = False


@dataclass(frozen=True)
class GenerateRuntimeHooksAction(Action):
    """Generate target-specific C/Zephyr hook code.

    The LLM generates instrumentation code tailored to the target
    hardware, verified by compilation check.
    """

    action_type: str = "generate_runtime_hooks"
    hook_type: str = "profiling"  # "profiling", "dispatch", "transport", "dma"
    target_language: str = "c"  # "c", "zephyr_c"
    hook_code: dict[str, str] = field(default_factory=dict)  # hook_point → C code


@dataclass(frozen=True)
class RequestVerificationAction(Action):
    """Request formal verification for a region.

    The agent explicitly asks the system to run translation validation
    or differential testing on a specific region. This is how the agent
    allocates verification budget.
    """

    action_type: str = "request_verification"
    level: str = "translation_validation"  # "translation_validation", "differential", "both"


@dataclass(frozen=True)
class RequestSemanticsAction(Action):
    """Request LLM to generate semantics for an op type.

    When the agent encounters ops without defined semantics, it can
    request the LLM to generate an OperationSemantics definition.
    This expands the verifiable frontier over time.
    """

    action_type: str = "request_semantics"
    op_type: str = ""  # e.g. "linalg.matmul"


@dataclass(frozen=True)
class RequestTransferAnalysisAction(Action):
    """Request a verified transfer analysis for a region.

    The agent requests a specific kind of dataflow analysis. The LLM
    generates the transfer functions, the system verifies them for
    soundness, and verified facts flow into the observation.
    """

    action_type: str = "request_transfer_analysis"
    analysis_type: str = ""  # "tile_divisibility", "local_mem_fit", "contiguous_layout"


@dataclass(frozen=True)
class GenerateXDSLDialectAction(Action):
    """Generate xDSL dialect scaffolding for an extension pack."""

    action_type: str = "generate_xdsl_dialect"
    pack_name: str = ""
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerateLLVMPatchAction(Action):
    """Generate LLVM fork patch scaffolding for an extension pack."""

    action_type: str = "generate_llvm_patch"
    pack_name: str = ""
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalAction:
    """An action that the system has validated as legal, with estimated cost.

    The agent picks from a list of these — it never submits raw actions
    that haven't been pre-validated.
    """

    action: Action
    estimated_cost_delta_us: float   # negative = improvement
    estimated_cost_after_us: float
    reason: str                      # why this action is legal/recommended
    risk: str                        # "safe", "moderate", "risky"
    rank: int                        # system's ranking (1 = best predicted)


# ============================================================================
# StepResult: what comes back after an action
# ============================================================================


@dataclass(frozen=True)
class StepResult:
    """Result of taking one action in the environment."""

    observation: Observation
    reward: float                    # positive = improvement
    done: bool                       # episode over?
    info: StepInfo


@dataclass(frozen=True)
class StepInfo:
    """Detailed info about what happened."""

    action_applied: bool
    verification_passed: bool
    cost_before_us: float
    cost_after_us: float
    improvement_pct: float
    error: str                       # empty if no error
    diagnostics: tuple[str, ...]     # warnings, notes
