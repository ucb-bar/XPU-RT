"""SoC/Zephyr runtime code generation from hardware specs.

Given a HardwareSpec, generates either a Zephyr RTOS project or
bare-metal C runtime code.  Selection is based on
``spec.platform.deployment_model``:

  - ``"zephyr"``  → full Zephyr project (prj.conf, DTS overlay,
    CMakeLists.txt, main.c with k_thread/k_sem)
  - ``"bare_metal"`` → static arena allocator, polling-loop main,
    linker script with memory regions, optional DMA ops
  - other values  → ValueError

All output is generated via Jinja2 templates stored under
``runtime_templates/{zephyr,bare_metal}/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader

from compgen.targetgen.hardware_spec import AddressSpace, HardwareSpec, MemoryModelSpec

log = structlog.get_logger()

_TEMPLATE_ROOT = Path(__file__).parent / "runtime_templates"


# ---- Helper dataclasses for template contexts ----


@dataclass(frozen=True)
class ThreadSpec:
    """Description of a Zephyr thread to create."""

    name: str
    stack_size: int = 4096
    priority: int = 5
    description: str = ""
    body: str = "/* TODO: generated dispatch logic */"
    wait_sem: str = ""
    post_sem: str = ""


@dataclass(frozen=True)
class SemaphoreSpec:
    """Description of a Zephyr semaphore."""

    name: str
    initial_count: int = 0
    limit: int = 1


@dataclass(frozen=True)
class ArenaSpec:
    """Description of a static memory arena for bare-metal allocation."""

    name: str
    size_bytes: int
    alignment: int = 16
    section: str = ""


@dataclass(frozen=True)
class AddressSpaceOverlay:
    """Address space info for DTS overlay generation."""

    name: str
    base_address: int
    size_bytes: int
    dma_accessible: bool = True


@dataclass(frozen=True)
class MemoryRegion:
    """Memory region for linker script generation."""

    name: str
    origin: int
    length: int
    flags: str = "rwx"
    alignment: int = 16
    is_arena: bool = False


@dataclass
class SocCodegenResult:
    """Result of SoC code generation."""

    output_dir: Path
    generated_files: list[str] = field(default_factory=list)


# ---- Jinja2 environment ----


def _get_env(subdir: str) -> Environment:
    """Create a Jinja2 environment for the given template subdirectory."""
    template_dir = _TEMPLATE_ROOT / subdir
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ---- Shared helpers ----

_BASE_ADDRESS = 0x8000_0000
_REGION_ALIGN = 0x1000_0000  # 256 MB boundary


def _has_dma(spec: HardwareSpec) -> bool:
    """Check whether the spec declares a DMA model."""
    return spec.memory_model.dma_model != "none"


def _assign_base_addresses(
    address_spaces: list[AddressSpace],
) -> list[tuple[AddressSpace, int]]:
    """Pair each address space with an assigned base address."""
    result: list[tuple[AddressSpace, int]] = []
    base = _BASE_ADDRESS
    for space in address_spaces:
        result.append((space, base))
        base += ((space.size_bytes + _REGION_ALIGN - 1) // _REGION_ALIGN) * _REGION_ALIGN
    return result


# ---- Context builders ----


def _infer_threads(
    spec: HardwareSpec,
    ipc_context: dict[str, Any] | None = None,
    instrumented: bool = False,
) -> list[ThreadSpec]:
    """Infer Zephyr threads from hardware spec.

    Args:
        spec: Hardware specification.
        ipc_context: Optional ZephyrIPCTransport codegen context for
            generating real IPC dispatch code.
        instrumented: Whether to emit tracing instrumentation.
    """
    threads: list[ThreadSpec] = []
    ipc = ipc_context or {}

    # Build real dispatch body
    trace_begin = "SYS_TRACE_IDLE();\n        " if instrumented else ""
    trace_end = "\n        /* trace: dispatch done */" if instrumented else ""

    if ipc.get("use_ipc_service"):
        dispatch_body = (
            f"{trace_begin}"
            f"/* Wait for work via IPC service endpoint */\n"
            f"        struct {ipc.get('endpoint_name', 'compgen_ep')}_msg msg;\n"
            f"        int rc = ipc_service_recv(&ep, &msg, sizeof(msg), K_FOREVER);\n"
            f"        if (rc == 0) {{\n"
            f"            kernel_fn_t fn = kernel_lookup(msg.kernel_name);\n"
            f"            if (fn) fn(msg.args, msg.args_size);\n"
            f"        }}{trace_end}"
        )
    elif ipc.get("mechanism") == "k_pipe":
        dispatch_body = (
            f"{trace_begin}"
            f"/* Read command from k_pipe */\n"
            f"        uint8_t cmd_buf[{ipc.get('pipe_size', 4096)}];\n"
            f"        size_t bytes_read = 0;\n"
            f"        k_pipe_get(&cmd_pipe, cmd_buf, sizeof(cmd_buf),\n"
            f"                   &bytes_read, 1, K_FOREVER);\n"
            f"        if (bytes_read > 0) {{\n"
            f"            dispatch_command(cmd_buf, bytes_read);\n"
            f"        }}{trace_end}"
        )
    else:
        # Default: k_msgq based dispatch
        dispatch_body = (
            f"{trace_begin}"
            f"/* Wait for dispatch command via k_msgq */\n"
            f"        struct dispatch_msg msg;\n"
            f"        if (k_msgq_get(&dispatch_msgq, &msg, K_FOREVER) == 0) {{\n"
            f"            kernel_fn_t fn = kernel_lookup(msg.kernel_id);\n"
            f"            if (fn) {{\n"
            f"                fn(msg.args, msg.args_size);\n"
            f"            }}\n"
            f"        }}{trace_end}"
        )

    threads.append(
        ThreadSpec(
            name="dispatch",
            stack_size=ipc.get("stack_size", 8192),
            priority=ipc.get("thread_priority", 5),
            description="Main dispatch thread — receives and executes kernel commands",
            body=dispatch_body,
            wait_sem="dispatch_ready",
            post_sem="dispatch_done",
        )
    )

    if _has_dma(spec):
        dma_body = (
            f"{trace_begin}"
            f"/* Wait for DMA request */\n"
            f"        struct dma_request req;\n"
            f"        if (k_msgq_get(&dma_msgq, &req, K_FOREVER) == 0) {{\n"
            f"            struct dma_config cfg = {{0}};\n"
            f"            struct dma_block_config blk = {{0}};\n"
            f"            blk.source_address = req.src_addr;\n"
            f"            blk.dest_address = req.dst_addr;\n"
            f"            blk.block_size = req.size;\n"
            f"            cfg.head_block = &blk;\n"
            f"            cfg.block_count = 1;\n"
            f"            cfg.channel_direction = MEMORY_TO_MEMORY;\n"
            f"            dma_config(dma_dev, req.channel, &cfg);\n"
            f"            dma_start(dma_dev, req.channel);\n"
            f"        }}{trace_end}"
        )
        threads.append(
            ThreadSpec(
                name="dma_handler",
                stack_size=4096,
                priority=3,
                description="DMA transfer management — configures and starts DMA channels",
                body=dma_body,
                wait_sem="dma_request",
                post_sem="dma_complete",
            )
        )

    # Profiling collector thread (when instrumented)
    if instrumented:
        threads.append(
            ThreadSpec(
                name="trace_collector",
                stack_size=4096,
                priority=10,  # low priority
                description="Collects and flushes trace data",
                body=("/* Periodically flush trace buffer */\n        k_msleep(100);\n        cg_trace_flush(NULL);"),
                wait_sem="",
                post_sem="",
            )
        )

    return threads


def _infer_semaphores(spec: HardwareSpec) -> list[SemaphoreSpec]:
    """Infer Zephyr semaphores from hardware spec."""
    sems: list[SemaphoreSpec] = [
        SemaphoreSpec(name="dispatch_ready", initial_count=0, limit=1),
        SemaphoreSpec(name="dispatch_done", initial_count=0, limit=1),
    ]

    if _has_dma(spec):
        sems.extend(
            [
                SemaphoreSpec(name="dma_request", initial_count=0, limit=1),
                SemaphoreSpec(name="dma_complete", initial_count=0, limit=1),
            ]
        )

    return sems


def _address_space_overlays(spec: HardwareSpec) -> list[AddressSpaceOverlay]:
    """Build DTS overlay entries from address spaces."""
    return [
        AddressSpaceOverlay(
            name=space.name,
            base_address=base,
            size_bytes=space.size_bytes,
            dma_accessible=space.dma_accessible,
        )
        for space, base in _assign_base_addresses(spec.memory_model.address_spaces)
    ]


def _arenas_from_memory_model(memory_model: MemoryModelSpec) -> list[ArenaSpec]:
    """Build arena specs from address spaces."""
    arenas: list[ArenaSpec] = []
    for space in memory_model.address_spaces:
        alignment = 16
        # Use larger alignment for DMA-accessible regions
        if space.dma_accessible:
            alignment = 64
        arenas.append(
            ArenaSpec(
                name=space.name,
                size_bytes=space.size_bytes,
                alignment=alignment,
                section=f".{space.name}",
            )
        )
    return arenas


_EXECUTABLE_REGIONS = frozenset(("dram", "flash", "rom"))


def _memory_regions_from_spec(spec: HardwareSpec) -> list[MemoryRegion]:
    """Build linker memory regions from address spaces."""
    regions: list[MemoryRegion] = []
    for space, base in _assign_base_addresses(spec.memory_model.address_spaces):
        name_lower = space.name.lower()
        regions.append(
            MemoryRegion(
                name=space.name,
                origin=base,
                length=space.size_bytes,
                flags="rwx" if name_lower in _EXECUTABLE_REGIONS else "rw",
                alignment=64 if space.dma_accessible else 16,
                is_arena=name_lower not in _EXECUTABLE_REGIONS,
            )
        )
    return regions


# ---- Zephyr project generation ----


def generate_zephyr_project(
    spec: HardwareSpec,
    output_dir: str | Path,
    *,
    ipc_context: dict[str, Any] | None = None,
    instrumented: bool = False,
    instrumentation_kconfig: dict[str, str] | None = None,
) -> SocCodegenResult:
    """Generate a complete Zephyr RTOS project from a hardware spec.

    Args:
        spec: The hardware specification to generate from.
        output_dir: Directory to write generated files into.
        ipc_context: Optional ZephyrIPCTransport codegen context for
            real IPC dispatch code generation.
        instrumented: Whether to emit tracing/profiling instrumentation.
        instrumentation_kconfig: Extra Kconfig overrides from
            InstrumentationConfig.zephyr_kconfig().

    Returns:
        SocCodegenResult with list of generated file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src_dir = out / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    env = _get_env("zephyr")
    generated: list[str] = []

    threads = _infer_threads(spec, ipc_context=ipc_context, instrumented=instrumented)
    semaphores = _infer_semaphores(spec)
    address_spaces = _address_space_overlays(spec)

    family = spec.platform.family
    deployment_model = spec.platform.deployment_model

    # Compute heap size as sum of smaller address spaces (not DRAM)
    non_dram = [s for s in spec.memory_model.address_spaces if s.name.lower() != "dram"]
    heap_size = sum(s.size_bytes for s in non_dram) if non_dram else 65536

    # Merge instrumentation Kconfig overrides
    extra_config: dict[str, str] = {}
    if instrumentation_kconfig:
        extra_config.update(instrumentation_kconfig)

    # 1. prj.conf
    tmpl = env.get_template("prj.conf.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        deployment_model=deployment_model,
        main_stack_size=8192,
        heap_pool_size=heap_size,
        workqueue_stack_size=4096,
        dma_model=spec.memory_model.dma_model,
        has_semaphores=bool(semaphores),
        extra_config=extra_config,
    )
    (out / "prj.conf").write_text(content)
    generated.append("prj.conf")

    # 2. app.overlay
    tmpl = env.get_template("app.overlay.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        address_spaces=address_spaces,
    )
    (out / "app.overlay").write_text(content)
    generated.append("app.overlay")

    # 3. CMakeLists.txt
    tmpl = env.get_template("CMakeLists.txt.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        project_name=spec.name.replace("-", "_"),
        extra_sources=[],
        include_dirs=[],
    )
    (out / "CMakeLists.txt").write_text(content)
    generated.append("CMakeLists.txt")

    # 4. main.c
    tmpl = env.get_template("main.c.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        deployment_model=deployment_model,
        dispatch_model=spec.execution_model.dispatch_model,
        thread_model=spec.execution_model.thread_model,
        threads=threads,
        semaphores=semaphores,
        dispatch_body="/* poll for work */",
        poll_interval_ms=10,
    )
    (src_dir / "main.c").write_text(content)
    generated.append("src/main.c")

    log.info(
        "soc_codegen.zephyr_generated",
        target=spec.name,
        output_dir=str(out),
        files=generated,
    )

    return SocCodegenResult(output_dir=out, generated_files=generated)


# ---- Arena allocator generation ----


def generate_arena_allocator(
    memory_model: MemoryModelSpec,
    output_dir: str | Path,
    *,
    target_name: str = "unknown",
    family: str = "unknown",
) -> SocCodegenResult:
    """Generate a static arena allocator from a memory model.

    Args:
        memory_model: The memory model spec with address spaces.
        output_dir: Directory to write arena_alloc.c into.
        target_name: Name for header comments.
        family: Family for header comments.

    Returns:
        SocCodegenResult with list of generated file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = _get_env("bare_metal")
    arenas = _arenas_from_memory_model(memory_model)

    tmpl = env.get_template("arena_alloc.c.template")
    content = tmpl.render(
        target_name=target_name,
        family=family,
        arenas=arenas,
    )
    (out / "arena_alloc.c").write_text(content)

    log.info("soc_codegen.arena_generated", target=target_name, arenas=len(arenas))
    return SocCodegenResult(output_dir=out, generated_files=["arena_alloc.c"])


# ---- DMA ops generation ----


def generate_dma_ops(
    memory_model: MemoryModelSpec,
    output_dir: str | Path,
    *,
    target_name: str = "unknown",
    family: str = "unknown",
    dma_base_addr: int = 0x1000_0000,
) -> SocCodegenResult:
    """Generate DMA transfer code from a memory model.

    Args:
        memory_model: The memory model spec with DMA configuration.
        output_dir: Directory to write dma_ops.c into.
        target_name: Name for header comments.
        family: Family for header comments.
        dma_base_addr: Base address for DMA registers.

    Returns:
        SocCodegenResult with list of generated file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if memory_model.dma_model == "none":
        log.info("soc_codegen.no_dma", target=target_name)
        return SocCodegenResult(output_dir=out, generated_files=[])

    env = _get_env("bare_metal")
    tmpl = env.get_template("dma_ops.c.template")
    content = tmpl.render(
        target_name=target_name,
        family=family,
        dma_model=memory_model.dma_model,
        max_outstanding_dma=memory_model.max_outstanding_dma,
        dma_base_addr=dma_base_addr,
        dma_control_addr=dma_base_addr + 0x100,
        dma_status_addr=dma_base_addr + 0x104,
    )
    (out / "dma_ops.c").write_text(content)

    log.info(
        "soc_codegen.dma_generated",
        target=target_name,
        dma_model=memory_model.dma_model,
    )
    return SocCodegenResult(output_dir=out, generated_files=["dma_ops.c"])


# ---- Bare-metal runtime generation ----


def generate_bare_metal_runtime(
    spec: HardwareSpec,
    output_dir: str | Path,
) -> SocCodegenResult:
    """Generate a complete bare-metal runtime from a hardware spec.

    Produces: arena_alloc.c, main.c, linker.ld, and optionally dma_ops.c.

    Args:
        spec: The hardware specification to generate from.
        output_dir: Directory to write generated files into.

    Returns:
        SocCodegenResult with list of generated file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = _get_env("bare_metal")
    generated: list[str] = []
    family = spec.platform.family
    arenas = _arenas_from_memory_model(spec.memory_model)
    regions = _memory_regions_from_spec(spec)

    # 1. Arena allocator
    result = generate_arena_allocator(
        spec.memory_model,
        out,
        target_name=spec.name,
        family=family,
    )
    generated.extend(result.generated_files)

    # 2. main.c
    sync_mechanism = spec.runtime_contract.synchronization
    tmpl = env.get_template("main.c.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        dispatch_model=spec.execution_model.dispatch_model,
        sync_mechanism=sync_mechanism,
        arenas=arenas,
        has_dma=_has_dma(spec),
        hw_init_body="/* hardware-specific initialization */",
        dispatch_body="/* execute next work item */",
    )
    (out / "main.c").write_text(content)
    generated.append("main.c")

    # 3. linker.ld
    # Determine default regions for text/data/bss/stack
    # Use the largest region (typically DRAM) for text/data
    largest = max(regions, key=lambda r: r.length) if regions else None
    default_region = largest.name.upper() if largest else "DRAM"

    tmpl = env.get_template("linker.ld.template")
    content = tmpl.render(
        target_name=spec.name,
        family=family,
        memory_regions=regions,
        text_region=default_region,
        data_region=default_region,
        bss_region=default_region,
        stack_region=default_region,
        stack_size=0x4000,
    )
    (out / "linker.ld").write_text(content)
    generated.append("linker.ld")

    # 4. DMA ops (if applicable)
    if _has_dma(spec):
        dma_result = generate_dma_ops(
            spec.memory_model,
            out,
            target_name=spec.name,
            family=family,
        )
        generated.extend(dma_result.generated_files)

    log.info(
        "soc_codegen.bare_metal_generated",
        target=spec.name,
        output_dir=str(out),
        files=generated,
    )

    return SocCodegenResult(output_dir=out, generated_files=generated)


# ---- Top-level dispatcher ----


def generate_soc_runtime(
    spec: HardwareSpec,
    output_dir: str | Path,
) -> SocCodegenResult:
    """Generate SoC runtime code, selecting strategy from deployment_model.

    Args:
        spec: The hardware specification.
        output_dir: Where to write generated files.

    Returns:
        SocCodegenResult.

    Raises:
        ValueError: If deployment_model is not ``"zephyr"`` or ``"bare_metal"``.
    """
    deployment = spec.platform.deployment_model

    if deployment == "zephyr":
        return generate_zephyr_project(spec, output_dir)
    if deployment == "bare_metal":
        return generate_bare_metal_runtime(spec, output_dir)

    msg = f"Unsupported deployment_model={deployment!r} for SoC codegen. Expected 'zephyr' or 'bare_metal'."
    raise ValueError(msg)
