"""C HAL driver code generation from a HardwareSpec.

Maps hardware-spec fields to concrete C allocation, dispatch, and
synchronisation strategies, then renders them through simple string
templates (no Jinja dependency).

Allocation strategies (from ``memory_model``):
    malloc          — standard heap (RVV CPU, vendor matrix)
    scratchpad      — offset-based scratchpad management (RoCC)
    firmware_managed— firmware-side allocation (NPU)
    device_alloc    — device-side allocator (GPU)

Launch mechanisms (from ``runtime_contract.kernel_launch``):
    function_call   — direct C function call
    inline_intrinsic— compiler intrinsic wrapper
    rocc_instruction— custom RoCC instruction encoding
    mailbox         — firmware mailbox command
    command_queue   — GPU-style command queue

Sync mechanisms (from ``runtime_contract.synchronization``):
    none, barrier, fence, fence_dma, event, polling
"""

from __future__ import annotations

from pathlib import Path

import structlog

from compgen.targetgen.hardware_spec import ExecutionModel, HardwareSpec

log = structlog.get_logger()

_TEMPLATE_DIR = Path(__file__).parent / "runtime_templates"

# ---------------------------------------------------------------------------
# Strategy helpers — map spec fields to strategy strings
# ---------------------------------------------------------------------------


def _infer_alloc_strategy(spec: HardwareSpec) -> str:
    """Determine buffer allocation strategy from memory model."""
    space_names = {s.name.lower() for s in spec.memory_model.address_spaces}
    mem_alloc = spec.runtime_contract.memory_allocation.lower()

    if "firmware" in mem_alloc:
        return "firmware_managed"
    if "scratchpad" in space_names:
        return "scratchpad"
    if mem_alloc == "dynamic" or spec.runtime_contract.kernel_launch == "command_queue":
        return "device_alloc"
    return "malloc"


def _infer_launch_mechanism(spec: HardwareSpec) -> str:
    """Determine kernel launch mechanism from runtime contract."""
    launch = spec.runtime_contract.kernel_launch.lower()
    if launch == "command_queue":
        return "command_queue"
    if launch == "mailbox":
        return "mailbox"
    if spec.execution_model.model == ExecutionModel.ROCC_COPROCESSOR:
        return "rocc_instruction"
    if spec.execution_model.model == ExecutionModel.DECOUPLED_MATRIX and spec.isa.compiler_intrinsics:
        return "inline_intrinsic"
    return "function_call"


def _infer_sync_mechanism(spec: HardwareSpec) -> str:
    """Determine synchronisation mechanism from runtime contract."""
    sync = spec.runtime_contract.synchronization.lower()
    if sync == "polling":
        return "polling"
    if sync == "event":
        return "event"
    if sync == "fence":
        # RoCC with DMA gets fence+DMA flavour
        if spec.memory_model.dma_model != "none":
            return "fence_dma"
        return "fence"
    if sync == "barrier":
        return "barrier"
    return "none"


# ---------------------------------------------------------------------------
# Per-strategy code fragments
# ---------------------------------------------------------------------------

# -- Allocation ------------------------------------------------------------

_ALLOC_MALLOC = {
    "alloc_includes": "#include <stdlib.h>",
    "alloc_globals": "",
    "alloc_body": "    return malloc(size_bytes);",
    "free_body": "    free(ptr);",
}

_ALLOC_SCRATCHPAD = {
    "alloc_includes": "",
    "alloc_globals": (
        "static uint8_t scratchpad[262144];  /* sized from spec */\nstatic size_t  scratchpad_offset = 0;"
    ),
    "alloc_body": (
        "    void *p = &scratchpad[scratchpad_offset];\n"
        "    scratchpad_offset += (size_bytes + 15u) & ~(size_t)15u;\n"
        "    return p;"
    ),
    "free_body": "    (void)ptr;  /* scratchpad: bulk-reset only */",
}

_ALLOC_FIRMWARE = {
    "alloc_includes": "",
    "alloc_globals": (
        "/* Firmware manages allocation via mailbox commands. */\n"
        "extern void *npu_firmware_alloc(size_t size);\n"
        "extern void  npu_firmware_free(void *ptr);"
    ),
    "alloc_body": "    return npu_firmware_alloc(size_bytes);",
    "free_body": "    npu_firmware_free(ptr);",
}

_ALLOC_DEVICE = {
    "alloc_includes": "",
    "alloc_globals": ("extern void *device_malloc(size_t size);\nextern void  device_free(void *ptr);"),
    "alloc_body": "    return device_malloc(size_bytes);",
    "free_body": "    device_free(ptr);",
}

_ALLOC_MAP: dict[str, dict[str, str]] = {
    "malloc": _ALLOC_MALLOC,
    "scratchpad": _ALLOC_SCRATCHPAD,
    "firmware_managed": _ALLOC_FIRMWARE,
    "device_alloc": _ALLOC_DEVICE,
}

# -- Dispatch / launch -----------------------------------------------------

_LAUNCH_FUNCTION_CALL = {
    "dispatch_includes": "",
    "dispatch_globals": (
        "typedef int (*kernel_fn_t)(const void *, size_t);\nextern kernel_fn_t kernel_lookup(const char *name);"
    ),
    "dispatch_body": (
        "    kernel_fn_t fn = kernel_lookup(kernel_name);\n    if (!fn) return -1;\n    return fn(args, args_size);"
    ),
}

_LAUNCH_INLINE_INTRINSIC = {
    "dispatch_includes": "",
    "dispatch_globals": (
        "typedef int (*kernel_fn_t)(const void *, size_t);\n"
        "extern kernel_fn_t kernel_lookup(const char *name);\n"
        "\n"
        "/* Vendor intrinsic wrappers are linked at build time. */"
    ),
    "dispatch_body": (
        "    kernel_fn_t fn = kernel_lookup(kernel_name);\n"
        "    if (!fn) return -1;\n"
        '    __asm__ volatile("fence" ::: "memory");\n'
        "    return fn(args, args_size);"
    ),
}

_LAUNCH_ROCC = {
    "dispatch_includes": "",
    "dispatch_globals": (
        "#define ROCC_INSTRUCTION(x, rs1, rs2, funct) \\\n"
        '    __asm__ volatile(".insn r CUSTOM_0, 0x7, " #funct ", x0, %0, %1" \\\n'
        '                     :: "r"(rs1), "r"(rs2))\n'
        "\n"
        "typedef int (*kernel_fn_t)(const void *, size_t);\n"
        "extern kernel_fn_t kernel_lookup(const char *name);"
    ),
    "dispatch_body": (
        "    kernel_fn_t fn = kernel_lookup(kernel_name);\n    if (!fn) return -1;\n    return fn(args, args_size);"
    ),
}

_LAUNCH_MAILBOX = {
    "dispatch_includes": "",
    "dispatch_globals": (
        "typedef struct { uint32_t cmd; uint32_t payload_addr; uint32_t payload_size; } mailbox_msg_t;\n"
        "extern int mailbox_send(const mailbox_msg_t *msg);\n"
        "extern int mailbox_recv(mailbox_msg_t *msg);"
    ),
    "dispatch_body": (
        "    mailbox_msg_t msg;\n"
        "    msg.cmd = 0x01;  /* DISPATCH */\n"
        "    msg.payload_addr = (uint32_t)(uintptr_t)args;\n"
        "    msg.payload_size = (uint32_t)args_size;\n"
        "    return mailbox_send(&msg);"
    ),
}

_LAUNCH_COMMAND_QUEUE = {
    "dispatch_includes": "",
    "dispatch_globals": (
        "typedef struct command_queue command_queue_t;\n"
        "extern command_queue_t *default_queue;\n"
        "extern int queue_enqueue(command_queue_t *q, const char *name,\n"
        "                         const void *args, size_t args_size);\n"
        "extern int queue_flush(command_queue_t *q);"
    ),
    "dispatch_body": (
        "    int rc = queue_enqueue(default_queue, kernel_name, args, args_size);\n"
        "    if (rc != 0) return rc;\n"
        "    return queue_flush(default_queue);"
    ),
}

_LAUNCH_MAP: dict[str, dict[str, str]] = {
    "function_call": _LAUNCH_FUNCTION_CALL,
    "inline_intrinsic": _LAUNCH_INLINE_INTRINSIC,
    "rocc_instruction": _LAUNCH_ROCC,
    "mailbox": _LAUNCH_MAILBOX,
    "command_queue": _LAUNCH_COMMAND_QUEUE,
}

# -- Synchronisation -------------------------------------------------------

_SYNC_NONE = "    return 0;  /* no synchronization required */"

_SYNC_BARRIER = "    __sync_synchronize();\n    return 0;"

_SYNC_FENCE = '    __asm__ volatile("fence" ::: "memory");\n    return 0;'

_SYNC_FENCE_DMA = (
    '    __asm__ volatile("fence" ::: "memory");\n'
    "    /* Wait for outstanding DMA to complete. */\n"
    "    extern int dma_wait_all(void);\n"
    "    return dma_wait_all();"
)

_SYNC_EVENT = "    extern int event_synchronize(void);\n    return event_synchronize();"

_SYNC_POLLING = (
    "    extern volatile uint32_t *status_reg;\n    while (*status_reg & 0x1u) { /* spin */ }\n    return 0;"
)

_SYNC_MAP: dict[str, str] = {
    "none": _SYNC_NONE,
    "barrier": _SYNC_BARRIER,
    "fence": _SYNC_FENCE,
    "fence_dma": _SYNC_FENCE_DMA,
    "event": _SYNC_EVENT,
    "polling": _SYNC_POLLING,
}

# -- Driver init/shutdown helpers ------------------------------------------

_INIT_BODIES: dict[str, str] = {
    "malloc": "    /* No special init for malloc-based targets. */\n    return 0;",
    "scratchpad": "    scratchpad_offset = 0;\n    return 0;",
    "firmware_managed": ("    extern int npu_firmware_init(void);\n    return npu_firmware_init();"),
    "device_alloc": ("    extern int device_runtime_init(void);\n    return device_runtime_init();"),
}

_SHUTDOWN_BODIES: dict[str, str] = {
    "malloc": "    /* Nothing to tear down. */\n    return 0;",
    "scratchpad": "    scratchpad_offset = 0;\n    return 0;",
    "firmware_managed": ("    extern int npu_firmware_shutdown(void);\n    return npu_firmware_shutdown();"),
    "device_alloc": ("    extern int device_runtime_shutdown(void);\n    return device_runtime_shutdown();"),
}

_EXTRA_INCLUDES: dict[str, str] = {
    "malloc": "#include <stdlib.h>",
    "scratchpad": "",
    "firmware_managed": "",
    "device_alloc": "",
}


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _read_template(name: str) -> str:
    """Read a template file from the runtime_templates directory."""
    path = _TEMPLATE_DIR / name
    return path.read_text()


def _render(template_text: str, substitutions: dict[str, str]) -> str:
    """Render a template using Python str.format_map."""
    return template_text.format_map(substitutions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hal_driver(spec: HardwareSpec, output_dir: Path) -> list[Path]:
    """Generate C HAL driver source files from a HardwareSpec.

    Args:
        spec: The hardware specification to generate from.
        output_dir: Directory where generated .c files are written.

    Returns:
        List of paths to the generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    alloc_strategy = _infer_alloc_strategy(spec)
    launch_mechanism = _infer_launch_mechanism(spec)
    sync_mechanism = _infer_sync_mechanism(spec)

    log.info(
        "hal_codegen.strategies",
        target=spec.name,
        alloc=alloc_strategy,
        launch=launch_mechanism,
        sync=sync_mechanism,
    )

    family = spec.platform.family

    # ---- hal_allocator.c ----
    alloc_frags = _ALLOC_MAP[alloc_strategy]
    alloc_text = _render(
        _read_template("hal_allocator.c.template"),
        {
            "target_name": spec.name,
            "alloc_strategy": alloc_strategy,
            **alloc_frags,
        },
    )
    alloc_path = output_dir / "hal_allocator.c"
    alloc_path.write_text(alloc_text)

    # ---- hal_dispatch.c ----
    launch_frags = _LAUNCH_MAP[launch_mechanism]
    sync_body = _SYNC_MAP[sync_mechanism]
    dispatch_text = _render(
        _read_template("hal_dispatch.c.template"),
        {
            "target_name": spec.name,
            "launch_mechanism": launch_mechanism,
            "sync_mechanism": sync_mechanism,
            "dispatch_body": launch_frags["dispatch_body"],
            "dispatch_includes": launch_frags["dispatch_includes"],
            "dispatch_globals": launch_frags["dispatch_globals"],
            "sync_body": sync_body,
        },
    )
    dispatch_path = output_dir / "hal_dispatch.c"
    dispatch_path.write_text(dispatch_text)

    # ---- hal_driver.c ----
    driver_text = _render(
        _read_template("hal_driver.c.template"),
        {
            "target_name": spec.name,
            "family": family,
            "dispatch_model": spec.execution_model.dispatch_model,
            "extra_includes": _EXTRA_INCLUDES.get(alloc_strategy, ""),
            "init_body": _INIT_BODIES.get(alloc_strategy, "    /* init */"),
            "shutdown_body": _SHUTDOWN_BODIES.get(alloc_strategy, "    /* shutdown */"),
        },
    )
    driver_path = output_dir / "hal_driver.c"
    driver_path.write_text(driver_text)

    generated = [driver_path, alloc_path, dispatch_path]
    log.info("hal_codegen.done", files=[str(p) for p in generated])
    return generated
