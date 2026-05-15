"""Wave 1.18 — target registry tests.

Pin the discovery + introspection contract every agent uses to
explore the target hierarchy. CPU-only; doesn't load any vendor
adapters.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _clean_registry():
    from compgen.targets.registry import reset

    reset()
    yield
    reset()


class TestRegistrationPaths:
    def test_register_in_tree(self) -> None:
        from compgen.targets.registry import register_target, registry

        pkg = register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="paper-faithful Blackwell hardware",
        )
        assert pkg.target_id == "gpu.nvidia.blackwell"
        assert pkg.registration_path == "in_tree"
        assert registry().get("gpu.nvidia.blackwell") is pkg

    def test_register_vendor_common(self) -> None:
        """Empty arch → vendor-common entry."""
        from compgen.targets.registry import register_target, registry

        pkg = register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="",
        )
        assert pkg.target_id == "gpu.nvidia"
        assert registry().get("gpu.nvidia") is pkg

    def test_arch_fallback_to_vendor_common(self) -> None:
        """Looking up an arch we haven't registered should fall back
        to the vendor-common entry. That's how the dispatch path
        finds shared NVIDIA code when only one arch is registered."""
        from compgen.targets.registry import register_target, registry

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="",
            rationale="vendor-common nvidia",
        )
        # No `gpu.nvidia.blackwell` registered yet.
        fallback = registry().get("gpu.nvidia.blackwell")
        assert fallback is not None
        assert fallback.target_id == "gpu.nvidia"

    def test_re_registering_replaces(self) -> None:
        """Agent override: re-registering same target_id replaces.
        Lets the MCP-driven path override an in-tree default at
        session scope."""
        from compgen.targets.registry import register_target, registry

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            registration_path="in_tree",
        )
        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            registration_path="mcp",
        )
        pkg = registry().get("gpu.nvidia.blackwell")
        assert pkg is not None
        assert pkg.registration_path == "mcp"


class TestNavigation:
    """The agent uses these to introspect the hierarchy without
    knowing the layout in advance."""

    def _register_a_few(self) -> None:
        from compgen.targets.registry import register_target

        register_target(target_class="gpu", vendor="nvidia", arch="blackwell")
        register_target(target_class="gpu", vendor="nvidia", arch="hopper")
        register_target(target_class="gpu", vendor="amd", arch="cdna3")
        register_target(target_class="cpu", vendor="x86", arch="avx512")

    def test_classes(self) -> None:
        from compgen.targets.registry import registry

        self._register_a_few()
        assert registry().classes() == ("cpu", "gpu")

    def test_vendors_under_class(self) -> None:
        from compgen.targets.registry import registry

        self._register_a_few()
        assert registry().vendors("gpu") == ("amd", "nvidia")
        assert registry().vendors("cpu") == ("x86",)

    def test_arches_under_vendor(self) -> None:
        from compgen.targets.registry import registry

        self._register_a_few()
        assert registry().arches("gpu", "nvidia") == ("blackwell", "hopper")

    def test_tree_nested_view(self) -> None:
        """Nested-dict view for the agent's "show me everything"
        query."""
        from compgen.targets.registry import registry

        self._register_a_few()
        tree = registry().tree()
        assert tree == {
            "cpu": {"x86": ["avx512"]},
            "gpu": {"amd": ["cdna3"], "nvidia": ["blackwell", "hopper"]},
        }


class TestFiltering:
    def test_find_by_predicate(self) -> None:
        from compgen.targets.registry import register_target, registry

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            metadata={"supports_tensor_cores": True, "compute_capability": 100},
        )
        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="ampere",
            metadata={"supports_tensor_cores": True, "compute_capability": 80},
        )
        register_target(
            target_class="cpu",
            vendor="x86",
            arch="avx512",
            metadata={"supports_tensor_cores": False},
        )

        # Find all tensor-core-capable targets.
        with_tc = registry().find(lambda p: p.metadata.get("supports_tensor_cores", False))
        assert len(with_tc) == 2
        assert all(p.target_class == "gpu" for p in with_tc)

    def test_describe_returns_audit_payload(self) -> None:
        """Agent's "tell me about target X" query — same shape as
        BackendChoice.to_dict() for composability."""
        from compgen.targets.registry import register_target, registry

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="cuBLASDx + cu13 NVRTC + cluster-launch",
            metadata={"sm_count": 132},
        )
        d = registry().describe("gpu.nvidia.blackwell")
        # Required audit keys.
        for key in (
            "target_id",
            "target_class",
            "vendor",
            "arch",
            "rationale",
            "registration_path",
            "metadata",
            "adapters",
        ):
            assert key in d
        assert d["target_id"] == "gpu.nvidia.blackwell"
        assert d["rationale"].startswith("cuBLASDx")
        assert d["metadata"]["sm_count"] == 132

    def test_describe_unknown_returns_empty(self) -> None:
        """Unknown targets return an empty dict — never raises so
        the agent can ask freely without try/except."""
        from compgen.targets.registry import registry

        assert registry().describe("gpu.tenstorrent.gridx") == {}


class TestInTreeAutoRegistration:
    """Wave 1.11/1.12 — importing ``compgen.targets`` auto-registers
    every in-tree target package. The agent doesn't have to know
    what to import; it just lists the registry."""

    def test_importing_targets_populates_registry(self) -> None:
        # Reset, then re-import compgen.targets which triggers
        # registration of all in-tree leaves.
        from compgen.targets.registry import reset

        reset()

        import compgen.targets as targets_mod

        # Trigger the side-effect by re-executing __init__'s
        # registration helper.
        targets_mod._register_in_tree()

        from compgen.targets.registry import registry

        reg = registry()
        # Both target classes registered.
        assert "gpu" in reg.classes()
        assert "cpu" in reg.classes()
        # NVIDIA arches registered as separate leaves.
        nv_arches = reg.arches("gpu", "nvidia")
        assert "blackwell" in nv_arches
        assert "hopper" in nv_arches
        assert "ampere" in nv_arches
        # Vendor-common entry exists too.
        assert reg.get("gpu.nvidia") is not None

    def test_blackwell_metadata_pinned(self) -> None:
        """Bridge #095/108-validated values surface in the audit
        query — agents trust these for routing decisions."""
        from compgen.targets.registry import reset

        reset()
        import compgen.targets as targets_mod

        targets_mod._register_in_tree()

        from compgen.targets.registry import registry

        bw = registry().get("gpu.nvidia.blackwell")
        assert bw is not None
        m = bw.metadata
        assert m["supports_clusters"] is True
        assert m["supports_tensor_cores"] is True
        assert m["supports_tcgen05_mma"] is True
        assert m["supports_cu13_nvrtc"] is True
        assert m["default_tile_shape"] == [64, 64, 16]
        assert m["preferred_precision"] == "bf16_fp32"
        assert m["cublasdx_sm_tag"] == 1000


class TestEntryPointDiscovery:
    """``discover_entry_points`` is best-effort; missing entry
    points or load failures don't break the registry."""

    def test_no_entry_points_returns_zero(self) -> None:
        from compgen.targets.registry import discover_entry_points

        # No third-party packages declare 'compgen.targets' in this
        # CI env → returns 0, doesn't raise.
        n = discover_entry_points(group="compgen.targets")
        assert n == 0


class TestProtocolsImportable:
    """Wave 1.10 — Protocols are importable from the class-level
    contracts modules. The universal compile path imports from
    here, never from vendor-specific modules."""

    def test_gpu_contracts(self) -> None:
        from compgen.targets.gpu.contracts import (
            DEFAULT_SCHEDULING_OVERHEAD_US,
            Device,
            EventTimer,
            GpuBodyEmitter,
            GpuCostModel,
            GpuProbe,
            GpuRuntime,
        )

        assert isinstance(DEFAULT_SCHEDULING_OVERHEAD_US, float)
        assert all(
            P is not None
            for P in (
                GpuProbe,
                GpuBodyEmitter,
                GpuRuntime,
                GpuCostModel,
                Device,
                EventTimer,
            )
        )

    def test_cpu_contracts(self) -> None:
        from compgen.targets.cpu.contracts import CpuBodyEmitter, CpuRuntime

        assert CpuBodyEmitter is not None
        assert CpuRuntime is not None

    def test_tpu_contracts(self) -> None:
        from compgen.targets.tpu.contracts import (
            TpuBodyEmitter,
            TpuRuntime,
            TpuTopology,
        )

        assert all(P is not None for P in (TpuBodyEmitter, TpuRuntime, TpuTopology))


class TestProtocolStructuralCheck:
    """Pin the runtime-checkable contract: a class with the right
    methods satisfies the Protocol via ``isinstance``."""

    def test_gpu_probe_runtime_check(self) -> None:
        from compgen.targets.gpu.contracts import GpuProbe

        class _MyProbe:
            def is_available(self) -> bool:
                return True

            def device_arch(self) -> str:
                return "sm_100"

            def supports_clusters(self) -> bool:
                return True

            def supports_tensor_cores(self) -> bool:
                return True

            def library_paths(self) -> dict[str, str | None]:
                return {}

            def vendor_extras(self) -> dict[str, Any]:
                return {}

        assert isinstance(_MyProbe(), GpuProbe)

    def test_cpu_runtime_runtime_check(self) -> None:
        from compgen.targets.cpu.contracts import CpuRuntime

        class _MyCpuRt:
            def compile_source(self, **kwargs: Any) -> Any:
                return None

            def dispatch(self, **kwargs: Any) -> None:
                return None

        assert isinstance(_MyCpuRt(), CpuRuntime)
