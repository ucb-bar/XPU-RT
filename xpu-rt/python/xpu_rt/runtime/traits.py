"""Device capability traits — the runtime's view of what a target can do.

``DeviceTraits`` is the capability struct the runtime + kernel
providers + dispatch strategies query **instead** of branching on
``vendor == "cuda"``.  It's pure Python, pure data, derived from the
:class:`~compgen.targets.schema.TargetProfile` at device-open time.

Design rationale: per the runtime-HAL plan
(``~/.claude/plans/graceful-marinating-crab.md``), IREE's experience
shows that unifying GPU drivers behind a single vtable is infeasible —
command-buffer formats and semaphore implementations vary too much
across CUDA / HIP / Vulkan / Metal.  The clean alternative is a
**capability-traits + per-vendor driver** pattern: drivers stay
specialised, common logic queries traits, target-specific fast paths
branch on what the hardware actually supports.

Example::

    traits = DeviceTraits.from_target_profile(profile)
    if traits.supports_event_tensors:
        # paper's megakernel path (atomics + persistent kernels)
        plan = build_megakernel_plan(ir, traits)
    elif traits.supports_graph_capture:
        # CUDA-graph-style path
        plan = build_graph_capture_plan(ir, traits)
    else:
        # one-kernel-at-a-time fallback
        plan = build_iterative_plan(ir, traits)

Fields marked ``(derived)`` are computed from primitive fields in
:meth:`DeviceTraits.from_target_profile`; callers can pass overrides
via ``profile.metadata`` if the heuristic is wrong for a given chip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compgen.targets.schema import TargetProfile


# Keys forwarded from ``TargetProfile.metadata`` into
# ``DeviceTraits.metadata``. These are the Blackwell / Hopper / generic
# probe outputs the cost model + emitter consume. Listed here so the
# heuristic's surface is auditable and the probe + YAML stay in sync.
_FORWARDED_METADATA_KEYS = frozenset(
    {
        # Compute capability (probe).
        "compute_capability_major",
        "compute_capability_minor",
        "sm_count",
        # Architectural Blackwell / Hopper features.
        "supports_tma",
        "supports_clusters",
        "supports_fp8",
        "supports_fp4",
        "supports_ondevice_scheduler",
        # Roofline-cost inputs (used by kernels.cost.roofline).
        "peak_flops_per_s",
        "peak_bandwidth_bps",
        "peak_bandwidth_level",
        # Misc Blackwell-specific / topology.
        "interconnect_topology",  # "nvlink" | "pcie" | "shared_memory"
        "interconnect_bandwidth_gbps",
    }
)


# Normalised device classes. Map from ``DeviceSpec.device_type`` strings.
_DEVICE_CLASS_MAP = {
    "gpu": "gpu",
    "cpu": "cpu",
    "accelerator": "accel",
    "accel": "accel",
    "npu": "npu",
    "tpu": "accel",
    "dsp": "dsp",
    "fpga": "fpga",
}

# Vendors we know have native timeline-semaphore primitives
# (cuEvent / hipEvent / VkSemaphore).
_TIMELINE_SEMAPHORE_VENDORS = frozenset({"nvidia", "amd", "apple", "intel"})

# Vendors with GPU-family command-buffer support
# (CUDA stream, hipGraph, Vulkan command buffer, Metal command buffer).
_GPU_COMMAND_BUFFER_VENDORS = frozenset({"nvidia", "amd", "apple", "intel"})


@dataclass(frozen=True)
class DeviceTraits:
    """Runtime capability struct for a single device.

    All fields are pure booleans / integers / small tuples — cheap to
    compare, serialisable, easy to pickle into trace events.

    Attributes:
        device_class: Normalised class — ``"gpu"``, ``"cpu"``, ``"npu"``,
            ``"accel"``, ``"fpga"``, ``"dsp"``, or ``"unknown"``.
        vendor: Vendor string from the underlying ``DeviceSpec.vendor``.
            Lowercased for comparison convenience.
    """

    # --- identity ---------------------------------------------------------
    device_class: str
    vendor: str

    # --- core capabilities ------------------------------------------------
    has_native_timeline_semaphores: bool
    has_global_atomics: bool
    has_shared_memory_atomics: bool
    supports_persistent_kernels: bool
    supports_cooperative_launch: bool
    supports_command_buffers: bool
    supports_graph_capture: bool

    # --- memory model -----------------------------------------------------
    memory_spaces: tuple[str, ...]
    max_device_memory_bytes: int
    supports_host_pinned: bool
    supports_peer_access: bool

    # --- parallelism ------------------------------------------------------
    max_concurrent_queues: int
    max_workgroup_size: int

    # --- platform ---------------------------------------------------------
    is_bare_metal: bool
    has_rtos_support: bool
    #: Static memory budget. ``0`` means unbounded (host-OS path).
    runtime_memory_budget_bytes: int

    # --- paper-specific (derived) ----------------------------------------
    #: True iff the device can host the Event Tensor megakernel pattern.
    #: Needs atomics + persistent kernels.
    supports_event_tensors: bool
    #: True iff the device can accept symbolic task-grid launches
    #: (persistent kernel with a per-SM queue).
    supports_task_grid: bool

    # --- extra features from the profile ---------------------------------
    #: Raw feature strings from ``DeviceSpec.features``. Use for
    #: vendor-specific features that don't have a first-class trait
    #: yet (e.g. ``"tf32"``, ``"bf16_tensor_cores"``, ``"cp_async"``).
    features: frozenset[str] = field(default_factory=frozenset)

    #: Open-ended metadata dict populated from
    #: ``TargetProfile.metadata`` (filtered to ``_FORWARDED_METADATA_KEYS``)
    #: and from the live device probe in
    #: :meth:`compgen.runtime.native.device.Device.probe_traits`.
    #: Used by :mod:`compgen.kernels.cost.roofline` (peak rates) and
    #: by Phase-5's CUDA emitter (Blackwell-specific lowering choices —
    #: TMA / clusters / FP8). Keys with first-class booleans elsewhere
    #: are repeated here for ergonomics, e.g. ``supports_clusters``.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_target_profile(
        cls,
        profile: TargetProfile,
        *,
        device_index: int = 0,
    ) -> DeviceTraits:
        """Derive traits for one device in a ``TargetProfile``.

        Uses a best-effort heuristic based on ``device_type``,
        ``vendor``, and ``features``. Callers can override any derived
        field by setting ``profile.metadata["device_traits_overrides"]``
        (a dict of field-name → value applied after the heuristic).

        Args:
            profile: The target profile to derive from.
            device_index: Which device in ``profile.devices`` to use
                (default 0 — the primary device).

        Returns:
            Fully populated ``DeviceTraits``.
        """
        # Graceful empty profile: emit a minimal CPU trait.
        if not profile.devices:
            return cls._cpu_fallback()

        device = profile.devices[device_index]
        device_class = _DEVICE_CLASS_MAP.get(device.device_type.lower(), "unknown")
        vendor = (device.vendor or "unknown").lower()
        features = frozenset(device.features)
        meta = profile.metadata or {}

        is_gpu = device_class == "gpu"
        is_cpu = device_class == "cpu"
        is_accel = device_class in ("accel", "npu", "dsp", "fpga")

        # --- timeline semaphores -----------------------------------------
        # GPU vendors have native timeline-semaphore primitives; everyone
        # else emulates via atomics + condvars.
        has_native_timeline_semaphores = is_gpu and vendor in _TIMELINE_SEMAPHORE_VENDORS

        # --- atomics ----------------------------------------------------
        # Atomics are the universal floor: required for timeline
        # semaphore emulation, needed for the paper's event tensors.
        has_global_atomics = True if not is_accel else ("atomics" in features)
        has_shared_memory_atomics = is_gpu  # CPU has no shared mem; accels variable.

        # --- kernels ----------------------------------------------------
        # "Persistent kernel" = a launch that returns only after a
        # sentinel is set. Works on anything with a thread-of-execution.
        supports_persistent_kernels = is_gpu or is_cpu or ("persistent_kernels" in features)
        supports_cooperative_launch = is_gpu and ("cooperative_launch" in features)

        # --- command buffers / graph capture ----------------------------
        supports_command_buffers = is_gpu and vendor in _GPU_COMMAND_BUFFER_VENDORS
        supports_graph_capture = supports_command_buffers and ("graph_capture" in features or vendor == "nvidia")

        # --- memory -----------------------------------------------------
        memory_spaces = tuple(ml.name for ml in device.memory_hierarchy) or ("host",)
        max_device_memory_bytes = sum(ml.size_bytes for ml in device.memory_hierarchy)
        supports_host_pinned = is_gpu
        supports_peer_access = is_gpu and len(profile.devices) > 1

        # --- parallelism ------------------------------------------------
        # CPU: one queue per physical core; GPU: 32 streams baseline; accel: vendor-specific.
        if is_cpu:
            max_concurrent_queues = os.cpu_count() or 1
        elif is_gpu:
            max_concurrent_queues = int(meta.get("max_streams", 32))
        else:
            max_concurrent_queues = int(meta.get("max_concurrent_queues", 1))
        max_workgroup_size = int(meta.get("max_workgroup_size", 1024 if is_gpu else 1))

        # --- platform ---------------------------------------------------
        is_bare_metal = bool(meta.get("is_bare_metal", False)) or "bare_metal" in features
        has_rtos_support = bool(meta.get("has_rtos_support", False)) or "zephyr" in features or "freertos" in features
        runtime_memory_budget_bytes = int(meta.get("runtime_memory_budget_bytes", 0))

        # --- derived capabilities ---------------------------------------
        supports_event_tensors = has_global_atomics and supports_persistent_kernels
        supports_task_grid = supports_persistent_kernels

        # Forward whitelisted profile metadata into the traits' metadata
        # dict. The CUDA probe (Phase-4 ``cg_rt_cuda_probe_device``)
        # populates the rest at device-open time.
        forwarded_meta: dict[str, Any] = {k: meta[k] for k in _FORWARDED_METADATA_KEYS if k in meta}

        # Derive Blackwell-class boolean features when the profile
        # declares the compute capability. Probe-supplied values (set
        # later via :meth:`with_probe`) take precedence.
        cc_major = forwarded_meta.get("compute_capability_major")
        if isinstance(cc_major, (int, float)):
            cc_major = int(cc_major)
            forwarded_meta.setdefault("supports_tma", cc_major >= 9)
            forwarded_meta.setdefault("supports_clusters", cc_major >= 9)
            forwarded_meta.setdefault("supports_fp8", cc_major >= 9)
            forwarded_meta.setdefault("supports_fp4", cc_major >= 10)
            forwarded_meta.setdefault("supports_ondevice_scheduler", cc_major >= 9)

        traits = cls(
            device_class=device_class,
            vendor=vendor,
            has_native_timeline_semaphores=has_native_timeline_semaphores,
            has_global_atomics=has_global_atomics,
            has_shared_memory_atomics=has_shared_memory_atomics,
            supports_persistent_kernels=supports_persistent_kernels,
            supports_cooperative_launch=supports_cooperative_launch,
            supports_command_buffers=supports_command_buffers,
            supports_graph_capture=supports_graph_capture,
            memory_spaces=memory_spaces,
            max_device_memory_bytes=max_device_memory_bytes,
            supports_host_pinned=supports_host_pinned,
            supports_peer_access=supports_peer_access,
            max_concurrent_queues=max_concurrent_queues,
            max_workgroup_size=max_workgroup_size,
            is_bare_metal=is_bare_metal,
            has_rtos_support=has_rtos_support,
            runtime_memory_budget_bytes=runtime_memory_budget_bytes,
            supports_event_tensors=supports_event_tensors,
            supports_task_grid=supports_task_grid,
            features=features,
            metadata=forwarded_meta,
        )

        # Apply overrides from profile metadata (escape hatch for
        # specific chips where the heuristic is wrong).
        overrides = meta.get("device_traits_overrides")
        if isinstance(overrides, dict) and overrides:
            from dataclasses import replace

            allowed = {f.name for f in traits.__dataclass_fields__.values()}
            filtered = {k: v for k, v in overrides.items() if k in allowed}
            if filtered:
                traits = replace(traits, **filtered)

        return traits

    @classmethod
    def _cpu_fallback(cls) -> DeviceTraits:
        """Minimal CPU traits for empty-profile fallback."""
        return cls(
            device_class="cpu",
            vendor="unknown",
            has_native_timeline_semaphores=False,
            has_global_atomics=True,
            has_shared_memory_atomics=False,
            supports_persistent_kernels=True,
            supports_cooperative_launch=False,
            supports_command_buffers=False,
            supports_graph_capture=False,
            memory_spaces=("host",),
            max_device_memory_bytes=0,
            supports_host_pinned=False,
            supports_peer_access=False,
            max_concurrent_queues=os.cpu_count() or 1,
            max_workgroup_size=1,
            is_bare_metal=False,
            has_rtos_support=False,
            runtime_memory_budget_bytes=0,
            supports_event_tensors=True,  # atomics + persistent = yes, CPU-emulated
            supports_task_grid=True,
            features=frozenset(),
            metadata={},
        )

    def with_probe(self, probe: dict[str, Any]) -> DeviceTraits:
        """Return a copy of these traits with the live-probe values
        merged into ``metadata``.

        Probe-supplied values WIN over profile-derived ones — when
        the YAML says ``sm_count: 132`` (B200) but the live card
        reports 188, we trust the card. The probe dict is the output
        of :meth:`compgen.runtime.native.device.Device.probe_traits`
        and is JSON-serialisable.

        Top-level boolean fields whose canonical home is on the
        :class:`DeviceTraits` dataclass (``supports_persistent_kernels``,
        ``supports_cooperative_launch``) are also updated when the
        probe reports a value for them.
        """
        from dataclasses import replace

        merged = dict(self.metadata)
        merged.update(probe)

        # Fields with first-class dataclass slots — keep both surfaces in
        # sync. Probe values override.
        replacements: dict[str, Any] = {"metadata": merged}
        if "supports_persistent_kernels" in probe:
            replacements["supports_persistent_kernels"] = bool(probe["supports_persistent_kernels"])
        if "supports_cooperative_launch" in probe:
            replacements["supports_cooperative_launch"] = bool(probe["supports_cooperative_launch"])
        if "has_global_atomics" in probe:
            replacements["has_global_atomics"] = bool(probe["has_global_atomics"])
        # Re-derive the two paper-specific composites since their
        # primaries may have changed.
        new_atomics = replacements.get("has_global_atomics", self.has_global_atomics)
        new_persistent = replacements.get("supports_persistent_kernels", self.supports_persistent_kernels)
        replacements["supports_event_tensors"] = bool(new_atomics) and bool(new_persistent)
        replacements["supports_task_grid"] = bool(new_persistent)

        return replace(self, **replacements)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (for trace events + bundles)."""
        return {
            "device_class": self.device_class,
            "vendor": self.vendor,
            "has_native_timeline_semaphores": self.has_native_timeline_semaphores,
            "has_global_atomics": self.has_global_atomics,
            "has_shared_memory_atomics": self.has_shared_memory_atomics,
            "supports_persistent_kernels": self.supports_persistent_kernels,
            "supports_cooperative_launch": self.supports_cooperative_launch,
            "supports_command_buffers": self.supports_command_buffers,
            "supports_graph_capture": self.supports_graph_capture,
            "memory_spaces": list(self.memory_spaces),
            "max_device_memory_bytes": self.max_device_memory_bytes,
            "supports_host_pinned": self.supports_host_pinned,
            "supports_peer_access": self.supports_peer_access,
            "max_concurrent_queues": self.max_concurrent_queues,
            "max_workgroup_size": self.max_workgroup_size,
            "is_bare_metal": self.is_bare_metal,
            "has_rtos_support": self.has_rtos_support,
            "runtime_memory_budget_bytes": self.runtime_memory_budget_bytes,
            "supports_event_tensors": self.supports_event_tensors,
            "supports_task_grid": self.supports_task_grid,
            "features": sorted(self.features),
            "metadata": dict(self.metadata),
        }


__all__ = ["DeviceTraits"]
