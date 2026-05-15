"""Tests for SupportPlan runtime extensions and planner DmaOp/MemoryPlan extensions."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.runtime.planner import (
    DmaOp,
    ExecutionPlan,
    MemoryPlan,
)
from compgen.targetgen.classify import classify_hardware
from compgen.targetgen.load import load_hardware_spec
from compgen.targetgen.plan import generate_support_plan

EXEMPLAR_DIR = Path(__file__).parent / "exemplars"


# ---------- SupportPlan runtime fields from exemplars ----------


class TestSupportPlanRuntimeFields:
    """Verify that SupportPlan gets runtime_template, threading_model,
    and memory_strategy from the deployment_model in each exemplar."""

    def _plan_for(self, yaml_name: str) -> tuple[str, str, str]:
        spec = load_hardware_spec(EXEMPLAR_DIR / yaml_name)
        classification = classify_hardware(spec)
        plan = generate_support_plan(spec, classification)
        return plan.runtime_template, plan.threading_model, plan.memory_strategy

    def test_rvv_cpu_linux_userspace(self) -> None:
        rt, tm, ms = self._plan_for("test_rvv_cpu.yaml")
        assert rt == "linux_userspace"
        assert tm == "pthreads"
        assert ms == "dynamic"

    def test_matrix_ext_defaults_to_linux(self) -> None:
        # test_matrix_ext.yaml has no explicit deployment_model → defaults to linux_userspace
        rt, tm, ms = self._plan_for("test_matrix_ext.yaml")
        assert rt == "linux_userspace"
        assert tm == "pthreads"
        assert ms == "dynamic"

    def test_rocc_accel_bare_metal(self) -> None:
        rt, tm, ms = self._plan_for("test_rocc_accel.yaml")
        assert rt == "bare_metal"
        assert tm == "polling"
        assert ms == "static"

    def test_npu_text_isa_firmware(self) -> None:
        rt, tm, ms = self._plan_for("test_npu_text_isa.yaml")
        assert rt == "firmware"
        assert tm == "none"
        assert ms == "firmware"

    def test_gpu_simt_defaults_to_linux(self) -> None:
        # test_gpu_simt.yaml has no explicit deployment_model → defaults to linux_userspace
        rt, tm, ms = self._plan_for("test_gpu_simt.yaml")
        assert rt == "linux_userspace"
        assert tm == "pthreads"
        assert ms == "dynamic"

    def test_all_exemplars_have_runtime_fields(self) -> None:
        """Every exemplar must produce valid runtime fields."""
        valid_templates = {"linux_userspace", "zephyr_rtos", "bare_metal", "firmware", "linux_embedded"}
        valid_threading = {"pthreads", "k_thread", "polling", "none"}
        valid_memory = {"dynamic", "static", "firmware"}

        for yaml_file in sorted(EXEMPLAR_DIR.glob("*.yaml")):
            spec = load_hardware_spec(yaml_file)
            classification = classify_hardware(spec)
            plan = generate_support_plan(spec, classification)
            assert plan.runtime_template in valid_templates, (
                f"{yaml_file.name}: invalid runtime_template={plan.runtime_template}"
            )
            assert plan.threading_model in valid_threading, (
                f"{yaml_file.name}: invalid threading_model={plan.threading_model}"
            )
            assert plan.memory_strategy in valid_memory, (
                f"{yaml_file.name}: invalid memory_strategy={plan.memory_strategy}"
            )


# ---------- DmaOp ----------


class TestDmaOp:
    def test_construction_minimal(self) -> None:
        op = DmaOp(tensor_name="x", src_space="dram", dst_space="scratchpad")
        assert op.tensor_name == "x"
        assert op.src_space == "dram"
        assert op.dst_space == "scratchpad"
        assert op.src_offset == 0
        assert op.dst_offset == 0
        assert op.size_bytes == 0
        assert op.stride_pattern == "contiguous"
        assert op.async_ is True

    def test_construction_full(self) -> None:
        op = DmaOp(
            tensor_name="weights",
            src_space="dram",
            dst_space="scratchpad",
            src_offset=1024,
            dst_offset=0,
            size_bytes=4096,
            stride_pattern="2d_strided",
            async_=False,
        )
        assert op.size_bytes == 4096
        assert op.stride_pattern == "2d_strided"
        assert op.async_ is False

    def test_frozen(self) -> None:
        op = DmaOp(tensor_name="x", src_space="dram", dst_space="scratchpad")
        with pytest.raises(AttributeError):
            op.tensor_name = "y"  # type: ignore[misc]

    def test_serialization_in_execution_plan(self) -> None:
        dma = DmaOp(
            tensor_name="act",
            src_space="dram",
            dst_space="local_memory",
            src_offset=256,
            dst_offset=0,
            size_bytes=2048,
            stride_pattern="nd_strided",
            async_=False,
        )
        plan = ExecutionPlan(dma_ops=[dma])
        d = plan.to_dict()
        assert len(d["dma_ops"]) == 1
        entry = d["dma_ops"][0]
        assert entry["tensor"] == "act"
        assert entry["src_space"] == "dram"
        assert entry["dst_space"] == "local_memory"
        assert entry["src_offset"] == 256
        assert entry["dst_offset"] == 0
        assert entry["bytes"] == 2048
        assert entry["stride_pattern"] == "nd_strided"
        assert entry["async"] is False


# ---------- MemoryPlan extensions ----------


class TestMemoryPlanExtensions:
    def test_defaults(self) -> None:
        mp = MemoryPlan(device_index=0)
        assert mp.address_space == "global"
        assert mp.physical_offset == 0

    def test_with_address_space(self) -> None:
        mp = MemoryPlan(
            device_index=0,
            peak_bytes=65536,
            address_space="scratchpad",
            physical_offset=0x8000_0000,
        )
        assert mp.address_space == "scratchpad"
        assert mp.physical_offset == 0x8000_0000

    def test_serialization_includes_new_fields(self) -> None:
        mp = MemoryPlan(
            device_index=1,
            peak_bytes=1024,
            address_space="local_memory",
            physical_offset=4096,
        )
        plan = ExecutionPlan(memory_plans=[mp])
        d = plan.to_dict()
        entry = d["memory_plans"][0]
        assert entry["address_space"] == "local_memory"
        assert entry["physical_offset"] == 4096

    def test_frozen(self) -> None:
        mp = MemoryPlan(device_index=0, address_space="sram")
        with pytest.raises(AttributeError):
            mp.address_space = "dram"  # type: ignore[misc]


# ---------- ExecutionPlan dma_ops default ----------


class TestExecutionPlanDmaOps:
    def test_default_empty(self) -> None:
        plan = ExecutionPlan()
        assert plan.dma_ops == []

    def test_to_dict_empty_dma_ops(self) -> None:
        plan = ExecutionPlan()
        d = plan.to_dict()
        assert d["dma_ops"] == []

    def test_to_dict_with_multiple_dma_ops(self) -> None:
        ops = [
            DmaOp(tensor_name="a", src_space="dram", dst_space="sp"),
            DmaOp(tensor_name="b", src_space="sp", dst_space="dram", size_bytes=512),
        ]
        plan = ExecutionPlan(dma_ops=ops)
        d = plan.to_dict()
        assert len(d["dma_ops"]) == 2
        assert d["dma_ops"][1]["bytes"] == 512
