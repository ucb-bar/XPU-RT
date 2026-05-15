"""Merlin-style hardware specification schema.

Goes beyond TargetProfile (what a device HAS) to describe how a device
EXECUTES: ISA exposure, execution model, tile geometry, memory model,
numeric contracts, runtime contracts, verification surface, patches.

The HardwareSpec is the document hardware engineers author.  A TargetProfile
is extracted from it for backward compatibility with the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---- Section 0: Platform & Deployment ----


@dataclass(frozen=True)
class PlatformSpec:
    """Platform and deployment environment.

    Attributes:
        vendor: Hardware vendor name.
        family: Target family identifier.
        chip_name: Chip/SoC name.
        host_arch: Host architecture (e.g., "riscv64", "x86_64").
        toolchain: Compiler toolchain identifier.
        sdk: SDK name (optional).
        sdk_version: SDK version string (optional).
        deployment_model: Deployment model. One of: ``linux_userspace``,
            ``zephyr_rtos``, ``bare_metal``, ``firmware``, ``linux_embedded``.
    """

    vendor: str
    family: str
    chip_name: str
    host_arch: str = "riscv64"
    toolchain: str = "llvm-18"
    sdk: str = ""
    sdk_version: str = ""
    deployment_model: str = "linux_userspace"


# ---- Section 1: Execution Model (primary classifier) ----


class ExecutionModel(Enum):
    """The execution model is the primary classifier for target families."""

    SIMD_VECTOR = "simd_vector"
    DECOUPLED_MATRIX = "decoupled_matrix"
    ROCC_COPROCESSOR = "rocc_coprocessor"
    TEXT_ISA_NPU = "text_isa_npu"
    SIMT_GPU = "simt_gpu"
    DATAFLOW = "dataflow"
    FIRMWARE_DRIVEN = "firmware_driven"


@dataclass(frozen=True)
class ExecutionModelSpec:
    """Execution model description."""

    model: ExecutionModel
    thread_model: str = "single_thread"
    dispatch_model: str = "synchronous"
    parallelism: str = "data_parallel"
    control_flow: str = "host_driven"
    has_scoreboard: bool = False
    has_hardware_scheduler: bool = False


# ---- Section 2: ISA & Instruction Exposure ----


@dataclass(frozen=True)
class ISAExtension:
    """A single ISA extension."""

    name: str
    version: str = "1.0"
    description: str = ""


@dataclass(frozen=True)
class ISASpec:
    """ISA and instruction exposure."""

    base_isa: str
    extensions: list[ISAExtension] = field(default_factory=list)
    custom_instructions: dict[str, str] = field(default_factory=dict)
    instruction_encoding: str = "standard"
    compiler_intrinsics: bool = True
    inline_asm_supported: bool = True


# ---- Section 3: Native Operation Families ----


@dataclass(frozen=True)
class NativeOpFamily:
    """A family of natively supported operations."""

    name: str
    ops: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    fallback: str = "decompose"


@dataclass(frozen=True)
class NativeOpsSpec:
    """Native operation families."""

    families: list[NativeOpFamily] = field(default_factory=list)
    custom_ops: dict[str, str] = field(default_factory=dict)
    decomposition_rules: dict[str, str] = field(default_factory=dict)


# ---- Section 4: Tensor/Matrix Engine Geometry ----


@dataclass(frozen=True)
class TileGeometry:
    """Geometry of a single tile/block."""

    name: str
    dimensions: list[int] = field(default_factory=list)
    element_bits: int = 16
    layout: str = "row_major"
    alignment_bytes: int = 16


@dataclass(frozen=True)
class EngineGeometrySpec:
    """Tensor/matrix engine geometry."""

    tiles: list[TileGeometry] = field(default_factory=list)
    systolic_array_dim: list[int] = field(default_factory=list)
    vector_length_bits: int = 0
    max_warp_size: int = 0
    register_file_entries: int = 0


# ---- Section 5: Memory & Layout Model ----


@dataclass(frozen=True)
class AddressSpace:
    """A distinct address space on the device."""

    name: str
    id: int = 0
    size_bytes: int = 0
    access: str = "read_write"
    dma_accessible: bool = True


@dataclass(frozen=True)
class MemoryModelSpec:
    """Memory and layout model."""

    address_spaces: list[AddressSpace] = field(default_factory=list)
    coherence: str = "coherent"
    dma_model: str = "none"
    preferred_layouts: list[str] = field(default_factory=list)
    alignment_constraints: dict[str, int] = field(default_factory=dict)
    double_buffering: bool = False
    max_outstanding_dma: int = 1


# ---- Section 6: Datatype & Numeric Contract ----


@dataclass(frozen=True)
class DtypeSupport:
    """Support level for a specific dtype."""

    name: str
    native: bool = True
    accumulator_dtype: str = ""
    rounding_mode: str = "rne"


@dataclass(frozen=True)
class NumericContractSpec:
    """Datatype and numeric contract."""

    supported_dtypes: list[DtypeSupport] = field(default_factory=list)
    mixed_precision_pairs: list[tuple[str, str]] = field(default_factory=list)
    denormal_handling: str = "ieee"
    nan_handling: str = "ieee"
    max_ulp_error: dict[str, float] = field(default_factory=dict)


# ---- Section 7: Runtime Contract ----


@dataclass(frozen=True)
class RuntimeMathSpec:
    """What math runtime the target makes available to emitted kernels.

    Providers branch on these flags to decide whether they can emit
    ``__builtin_expf`` (libm), one of the target's intrinsics, or a
    polynomial approximation (or reject the contract entirely).

    Attributes:
        has_libm: ``true`` when libc's ``math.h`` symbols (``expf``,
            ``sqrtf``, ``sinf``, …) are linkable from emitted kernels.
            False on bare-metal targets that ship no libm.
        has_libc: ``true`` when libc symbols beyond ``math.h``
            (``printf``, ``memcpy``, …) are linkable.
        intrinsics: List of available custom intrinsics by name
            (e.g. ``["mu_fexp", "mu_fnexp"]`` on Muon). A provider
            checks ``"mu_fexp" in contract.runtime.intrinsics`` to
            decide if it can lower softmax via the f16 exp intrinsic.
    """

    has_libm: bool = False
    has_libc: bool = False
    intrinsics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeContractSpec:
    """Runtime contract.

    Attributes:
        math: Runtime math capabilities — see :class:`RuntimeMathSpec`.
            Providers consult this to choose between libm calls,
            target intrinsics, or polynomial fallbacks.
    """

    calling_convention: str = "c_abi"
    kernel_launch: str = "function_call"
    synchronization: str = "barrier"
    memory_allocation: str = "static"
    error_handling: str = "return_codes"
    requires_runtime_library: str = ""
    workspace_allocation: str = "caller"
    math: RuntimeMathSpec = field(default_factory=RuntimeMathSpec)


# ---- Section 7b: Profiling Capabilities ----


@dataclass(frozen=True)
class ProfilerBackend:
    """A profiler backend available on this target.

    Describes a single profiling tool or mechanism that the hardware exposes.
    The agentic LLM reads these to decide which counters to enable and can
    generate target-specific hook code for ``integration == "embedded"``.

    Attributes:
        name: Identifier for this profiler (e.g., ``"nsight_systems"``,
            ``"perf"``, ``"zephyr_trace"``, ``"etm"``, ``"custom_pmu"``).
        kind: Category of profiler.  One of ``"host_sampling"``,
            ``"device_trace"``, ``"pmu_counters"``, ``"hardware_trace"``.
        counters: Available PMU counter names (e.g.,
            ``["cycles", "instructions", "cache_misses"]``).
        tile_level: Whether this profiler can instrument at tile granularity.
        integration: How CompGen should integrate with this profiler.
            ``"external"`` — launch wrapper (e.g., ``nsys profile``).
            ``"embedded"`` — codegen emits instrumentation hooks.
            ``"sdk"`` — call profiler SDK API at runtime.
        setup_command: Shell command to launch the profiler externally
            (only used when ``integration == "external"``).
        sdk_header: C header to include for SDK integration
            (e.g., ``"nvToolsExt.h"``, ``"zephyr/tracing/tracing.h"``).
        requires_root: Whether the profiler needs elevated privileges.
    """

    name: str
    kind: str = "pmu_counters"
    counters: list[str] = field(default_factory=list)
    tile_level: bool = False
    integration: str = "external"
    setup_command: str = ""
    sdk_header: str = ""
    requires_root: bool = False


@dataclass(frozen=True)
class ProfilingSpec:
    """Profiling capabilities for this target.

    Declares what the hardware CAN expose for performance analysis.
    The scaffold provides the adapter protocol; the agentic LLM reads this
    spec to select counters, generate hooks, and configure analysis.

    Attributes:
        backends: Available profiler backends on this target.
        default_backend: Name of the preferred backend for general use.
        tile_profiling: Whether tile-level instrumentation is possible.
        memory_traffic_counters: Whether memory bandwidth counters exist.
        power_monitoring: Whether power consumption can be measured.
        thermal_monitoring: Whether thermal data is accessible.
        custom_hooks: LLM-generated hook code snippets, keyed by hook point
            name (e.g., ``{"pre_dispatch": "...c code..."}``,
            ``{"post_dma": "...c code..."}``.  Populated by the agentic
            loop via ``GenerateRuntimeHooksAction``.
    """

    backends: list[ProfilerBackend] = field(default_factory=list)
    default_backend: str = ""
    tile_profiling: bool = False
    memory_traffic_counters: bool = False
    power_monitoring: bool = False
    thermal_monitoring: bool = False
    custom_hooks: dict[str, str] = field(default_factory=dict)


# ---- Section 7c: Topology ----


class DeploymentTopology(Enum):
    """How devices are organized across the system."""

    SINGLE_DEVICE = "single_device"
    MULTI_DEVICE = "multi_device"
    MULTI_DOMAIN_SOC = "multi_domain_soc"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True)
class TopologyNode:
    """A compute node in the system topology.

    A node is a unit of independent execution — a host CPU, an RTOS domain
    on an SoC, or a remote machine in a cluster.  Each node contains one
    or more devices.

    Attributes:
        name: Unique node identifier (e.g., ``"host"``, ``"npu_domain"``,
            ``"worker-0"``).
        device_indices: Indices into ``TargetProfile.devices`` that belong
            to this node.
        role: Role in the system.  One of ``"host"``, ``"accelerator"``,
            ``"worker"``, ``"coordinator"``.
        runtime_env: Execution environment.  One of ``"linux_userspace"``,
            ``"zephyr_rtos"``, ``"bare_metal"``, ``"firmware"``.
        properties: Target-specific node properties (stack sizes, thread
            priorities, etc.) that the LLM can read and override.
    """

    name: str
    device_indices: list[int] = field(default_factory=list)
    role: str = "worker"
    runtime_env: str = "linux_userspace"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyLink:
    """A communication link between two topology nodes.

    Attributes:
        src_node: Name of the source node.
        dst_node: Name of the destination node.
        transport: Transport mechanism.  One of ``"local"``,
            ``"shared_memory"``, ``"zephyr_ipc"``, ``"dma"``,
            ``"network"``, ``"pcie"``, ``"custom"``.
        bandwidth_gbps: Link bandwidth in GB/s.
        latency_us: One-way latency in microseconds.
        bidirectional: Whether the link supports full-duplex.
        properties: Transport-specific configuration (buffer sizes,
            queue depths, etc.) that the LLM can tune.
    """

    src_node: str
    dst_node: str
    transport: str = "local"
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    bidirectional: bool = True
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologySpec:
    """System topology — how nodes and devices are interconnected.

    This is the target-agnostic scaffold.  A single-host GPU is one node
    with N devices.  A heterogeneous SoC is one node per domain (host CPU,
    NPU, DSP) connected by IPC links.  A cluster is K nodes connected by
    network links.  The planner and executor see the same abstraction.

    Attributes:
        deployment: The deployment topology class.
        nodes: Compute nodes in the system.
        links: Communication links between nodes.
    """

    deployment: DeploymentTopology = DeploymentTopology.SINGLE_DEVICE
    nodes: list[TopologyNode] = field(default_factory=list)
    links: list[TopologyLink] = field(default_factory=list)


# ---- Section 8: Verification Surface ----


@dataclass(frozen=True)
class VerificationSurfaceSpec:
    """Verification surface.

    Attributes:
        has_simulator: Whether a simulator is available for this target.
        simulator_command: Shell command (or templated command — see
            :func:`compgen.mcp.tools.embedded._simulator_command`) that
            launches the simulator on a compiled ELF.
        build_command: Optional shell command run before the simulator
            when ``simulator_run(execute=True)``. When empty (the default),
            the helper falls back to its Zephyr ``west build`` flow if a
            Zephyr root + sample directory are reachable, and otherwise
            skips the build step entirely. Set this to a target-specific
            command (e.g. ``"make -C $CHIPYARD/sims/vcs"``) when the
            simulator_command itself doesn't perform the build.
        sim_backend: Simulator backend name. One of ``"vcs"``,
            ``"verilator"``, ``"firesim"``. Plugged into
            :func:`compgen.mcp.tools.embedded._simulator_command`'s
            substitution as ``{sim_backend}`` so a single
            ``simulator_command`` template can resolve to any of
            ``sims/vcs/``, ``sims/verilator/``, ``sims/firesim/``.
            Defaults to ``"verilator"`` (the open-source default for
            Chipyard targets).
        has_emulator: Whether a software emulator is available.
        golden_model: Reference model for output comparison.
        max_acceptable_ulp: Maximum acceptable ULP error.
        performance_counters: Performance counter names exposed by sim.
        trace_support: Trace support kind.
        formal_model: Whether a formal model is available.
    """

    has_simulator: bool = False
    simulator_command: str = ""
    build_command: str = ""
    sim_backend: str = "verilator"
    has_emulator: bool = False
    golden_model: str = "pytorch_cpu"
    max_acceptable_ulp: float = 1.0
    performance_counters: list[str] = field(default_factory=list)
    trace_support: str = "none"
    formal_model: bool = False


# ---- Section 9: Patch Requirements ----


@dataclass(frozen=True)
class PatchRequirement:
    """A single patch requirement."""

    component: str
    description: str
    priority: str = "required"
    estimated_effort: str = "medium"


@dataclass(frozen=True)
class PatchSpec:
    """Patch requirements for CompGen to support this target."""

    requirements: list[PatchRequirement] = field(default_factory=list)
    new_dialects_needed: list[str] = field(default_factory=list)
    new_stages_needed: list[str] = field(default_factory=list)
    existing_backend_integration: str = ""


# ---- Top-level HardwareSpec ----


@dataclass(frozen=True)
class HardwareSpec:
    """Complete Merlin-style hardware specification.

    This is the document hardware engineers author.  It embeds enough
    information to extract a TargetProfile for backward compatibility.
    """

    name: str
    schema_version: str = "2.0"
    platform: PlatformSpec = field(
        default_factory=lambda: PlatformSpec(vendor="unknown", family="unknown", chip_name="unknown")
    )
    execution_model: ExecutionModelSpec = field(
        default_factory=lambda: ExecutionModelSpec(model=ExecutionModel.SIMD_VECTOR)
    )
    isa: ISASpec = field(default_factory=lambda: ISASpec(base_isa="unknown"))
    native_ops: NativeOpsSpec = field(default_factory=NativeOpsSpec)
    engine_geometry: EngineGeometrySpec = field(default_factory=EngineGeometrySpec)
    memory_model: MemoryModelSpec = field(default_factory=MemoryModelSpec)
    numeric_contract: NumericContractSpec = field(default_factory=NumericContractSpec)
    runtime_contract: RuntimeContractSpec = field(default_factory=RuntimeContractSpec)
    profiling: ProfilingSpec = field(default_factory=ProfilingSpec)
    topology: TopologySpec = field(default_factory=TopologySpec)
    verification_surface: VerificationSurfaceSpec = field(default_factory=VerificationSurfaceSpec)
    patches: PatchSpec = field(default_factory=PatchSpec)
    constraints: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
