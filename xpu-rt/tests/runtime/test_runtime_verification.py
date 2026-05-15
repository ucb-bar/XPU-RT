"""runtime-as-verification-target tests (§12 Dream 6).

Coverage:

happy path: emit + emit both pass refinement, ABI, and
  budget;
- plan-refinement negative controls: corrupt the emit to drop a
  dispatch / add an extra dispatch / reorder dispatches → each
  raises ``RuntimeRefinementError`` with the right ``kind``;
- ABI-conformance negative controls: inject ``cudaMalloc(...)``,
  ``hipLaunchKernel(...)``, or ``vkCmdDispatch(...)`` calls into the
  emit → each raises ``AbiConformanceError`` naming the symbol;
- resource-budget negative controls: bloat ``push_constants`` past
  256 bytes or ``bindings`` past 32 slots → raises
  ``ResourceBudgetError``;
- aggregator: ``run_runtime_verification`` writes
  ``runtime_verification_report.json`` even on failure, then raises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from compgen.runtime.errors import (
    AbiConformanceError,
    ResourceBudgetError,
    RuntimeRefinementError,
)
from compgen.runtime.execution_plan import (
    DependencyEdge,
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)
from compgen.runtime.glue_emit import (
    emit_c11_baremetal_executor,
    emit_cpp_host_executor,
)
from compgen.runtime.verification import (
    check_abi_conformance,
    check_plan_refinement,
    check_resource_budget,
    run_runtime_verification,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _emit_certs(run_dir: Path, bindings: list[RegionKernelBinding]) -> None:
    for b in bindings:
        cert_path = run_dir / b.certificate_path
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(json.dumps({
            "schema_version": "kernel_certificate_v1",
            "contract_hash": b.contract_hash,
            "task_id": "t", "region_id": b.region_id, "candidate_id": "c",
            "accepted_at_utc": "x", "artifact_hashes": {},
            "artifact_paths": {}, "verifier_report_path": "",
            "verifier_report_hash": "", "claims": {},
        }))


def _write_plan(run_dir: Path, plan: ExecutionPlan) -> None:
    plan_dir = run_dir / "05_execution_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore[import-untyped]
        (plan_dir / "execution_plan.yaml").write_text(
            yaml.safe_dump(plan.to_dict(), sort_keys=True), encoding="utf-8",
        )
    except ImportError:
        (plan_dir / "execution_plan.json").write_text(
            json.dumps(plan.to_dict(), sort_keys=True), encoding="utf-8",
        )


def _make_emit(tmp_path: Path, regions: tuple[str, ...] = ("r0", "r1")):
    bindings = [
        RegionKernelBinding(
            region_id=r, contract_hash=f"h{i}",
            certificate_path=f"04_kernel_codegen/certificates/h{i}.json",
        )
        for i, r in enumerate(regions)
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _emit_certs(run_dir, bindings)
    edges = [
        DependencyEdge(from_region=regions[i], to_region=regions[i + 1])
        for i in range(len(regions) - 1)
    ]
    plan = ExecutionPlan(
        workload="test", target="host_cpu",
        resources=[Resource(id="q", kind="compute", device="host_cpu")],
        region_placement=[
            RegionPlacement(region_id=r, device="host_cpu", queue="q")
            for r in regions
        ],
        dependency_edges=edges,
        region_kernel_bindings=bindings,
    )
    plan.validate()
    _write_plan(run_dir, plan)
    return emit_c11_baremetal_executor(run_dir)


# --------------------------------------------------------------------------- #
# Happy paths                                                                 #
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_c11_emit_passes_all_three_gates(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        report = run_runtime_verification(result.out_dir)
        assert report.overall == "pass", report.to_dict()
        assert report.refinement.overall == "pass"
        assert report.abi.overall == "pass"
        assert report.budget.overall == "pass"

    def test_runtime_report_persisted(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        report = run_runtime_verification(result.out_dir)
        assert report.report_path is not None and report.report_path.exists()
        body = json.loads(report.report_path.read_text())
        assert body["schema_version"] == "runtime_verification_report_v1"
        assert body["overall"] == "pass"

    def test_cpp_emit_also_passes(self, tmp_path: Path) -> None:
        # Build a CUDA plan (cpp_host) instead.
        bindings = [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ]
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        _emit_certs(run_dir, bindings)
        plan = ExecutionPlan(
            workload="test", target="cuda",
            resources=[Resource(id="q", kind="compute", device="cuda")],
            region_placement=[
                RegionPlacement(region_id="r0", device="cuda", queue="q"),
            ],
            region_kernel_bindings=bindings,
        )
        plan.validate()
        _write_plan(run_dir, plan)
        # Emit the cpp variant only (skip the c11 emit so the gate
        # picks the cpp manifest).
        emit_cpp_host_executor(run_dir)
        report = run_runtime_verification(run_dir / "06_glue_emit")
        assert report.overall == "pass"


# --------------------------------------------------------------------------- #
# Plan-refinement negative controls                                           #
# --------------------------------------------------------------------------- #


class TestPlanRefinementNegativeControls:
    def test_dropping_dispatch_raises_count_or_missing(
        self, tmp_path: Path,
    ) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        # Drop the second dispatch by replacing it with a no-op.
        corrupted = re.sub(
            r"rt_status = cg_rt_command_buffer_dispatch\(\s*"
            r"command_buffer,\s*compgen_kernel_r1[^;]*;",
            "rt_status = 0; /* dropped r1 */",
            src,
            count=1,
        )
        assert corrupted != src
        result.executor_path.write_text(corrupted)
        with pytest.raises(RuntimeRefinementError) as exc:
            check_plan_refinement(result.out_dir)
        assert exc.value.kind in ("count_mismatch", "missing_dispatch")

    def test_extra_dispatch_raises_unknown(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        # Append an extra dispatch to a phantom kernel.
        injected = src.replace(
            "cleanup:",
            "    cg_rt_command_buffer_dispatch(command_buffer, "
            "compgen_kernel_phantom, push_constants, "
            "sizeof(push_constants), bindings, n_bindings);\n"
            "cleanup:",
            1,
        )
        assert injected != src
        result.executor_path.write_text(injected)
        with pytest.raises(RuntimeRefinementError) as exc:
            check_plan_refinement(result.out_dir)
        assert exc.value.kind in ("count_mismatch", "unknown_dispatch")

    def test_reordered_dispatch_raises_order_mismatch(
        self, tmp_path: Path,
    ) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        # Swap the two dispatch identifiers so r1 dispatches before r0.
        corrupted = src.replace(
            "compgen_kernel_r0", "_TMP_KER_r0_TMP_",
        ).replace(
            "compgen_kernel_r1", "compgen_kernel_r0",
        ).replace(
            "_TMP_KER_r0_TMP_", "compgen_kernel_r1",
        )
        assert corrupted != src
        result.executor_path.write_text(corrupted)
        with pytest.raises(RuntimeRefinementError) as exc:
            check_plan_refinement(result.out_dir)
        assert exc.value.kind == "order_mismatch"


# --------------------------------------------------------------------------- #
# ABI-conformance negative controls                                           #
# --------------------------------------------------------------------------- #


class TestAbiConformanceNegativeControls:
    def test_cuda_malloc_injection_raises(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        injected = src.replace(
            "rt_status = cg_rt_instance_create",
            "    cudaMalloc(NULL, 0);\n    rt_status = cg_rt_instance_create",
            1,
        )
        assert injected != src
        result.executor_path.write_text(injected)
        with pytest.raises(AbiConformanceError) as exc:
            check_abi_conformance(result.executor_path)
        assert "cudaMalloc" in exc.value.symbols

    def test_hip_launch_injection_raises(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        injected = src.replace(
            "rt_status = cg_rt_instance_create",
            "    hipLaunchKernel(NULL);\n    rt_status = cg_rt_instance_create",
            1,
        )
        result.executor_path.write_text(injected)
        with pytest.raises(AbiConformanceError) as exc:
            check_abi_conformance(result.executor_path)
        assert "hipLaunchKernel" in exc.value.symbols

    def test_vulkan_injection_raises(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        injected = src.replace(
            "rt_status = cg_rt_instance_create",
            "    vkCmdDispatch(NULL, 0, 0, 0);\n    "
            "rt_status = cg_rt_instance_create",
            1,
        )
        result.executor_path.write_text(injected)
        with pytest.raises(AbiConformanceError) as exc:
            check_abi_conformance(result.executor_path)
        assert "vkCmdDispatch" in exc.value.symbols


# --------------------------------------------------------------------------- #
# Resource-budget negative controls                                           #
# --------------------------------------------------------------------------- #


class TestResourceBudgetNegativeControls:
    def test_push_constants_overcommit_raises(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        # Inflate the push-constants array past the cap.
        corrupted = src.replace(
            "uint8_t            push_constants[256]",
            "uint8_t            push_constants[1024]",
            1,
        )
        assert corrupted != src
        result.executor_path.write_text(corrupted)
        with pytest.raises(ResourceBudgetError) as exc:
            check_resource_budget(result.out_dir)
        assert exc.value.resource == "push_constants_bytes"
        assert exc.value.observed > exc.value.declared

    def test_binding_slots_overcommit_raises(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        corrupted = src.replace(
            "cg_rt_buffer_t    *bindings[32]",
            "cg_rt_buffer_t    *bindings[256]",
            1,
        )
        assert corrupted != src
        result.executor_path.write_text(corrupted)
        with pytest.raises(ResourceBudgetError) as exc:
            check_resource_budget(result.out_dir)
        assert exc.value.resource == "binding_slots"


# --------------------------------------------------------------------------- #
# Aggregator                                                                  #
# --------------------------------------------------------------------------- #


class TestAggregator:
    def test_report_written_even_on_failure(self, tmp_path: Path) -> None:
        result = _make_emit(tmp_path)
        src = result.executor_path.read_text()
        # Drop r1.
        corrupted = re.sub(
            r"rt_status = cg_rt_command_buffer_dispatch\(\s*"
            r"command_buffer,\s*compgen_kernel_r1[^;]*;",
            "rt_status = 0;",
            src, count=1,
        )
        result.executor_path.write_text(corrupted)
        with pytest.raises(RuntimeRefinementError):
            run_runtime_verification(result.out_dir)
        report_path = result.out_dir / "runtime_verification_report.json"
        assert report_path.exists(), (
            "M-91 must persist the report before raising — operator "
            "needs the JSON on disk to triage"
        )
        body = json.loads(report_path.read_text())
        assert body["overall"] == "fail"
        # And: the refinement section names the kind.
        kinds = {f["kind"] for f in body["refinement"]["failures"]}
        assert (kinds & {"count_mismatch", "missing_dispatch"})


# --------------------------------------------------------------------------- #
# Pure happy-path on C++ emit                                                 #
# --------------------------------------------------------------------------- #


class TestCppEmitGates:
    def test_cpp_emit_abi_conformance_passes(self, tmp_path: Path) -> None:
        bindings = [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ]
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        _emit_certs(run_dir, bindings)
        plan = ExecutionPlan(
            workload="t", target="cuda",
            resources=[Resource(id="q", kind="compute", device="cuda")],
            region_placement=[
                RegionPlacement(region_id="r0", device="cuda", queue="q"),
            ],
            region_kernel_bindings=bindings,
        )
        plan.validate()
        _write_plan(run_dir, plan)
        result = emit_cpp_host_executor(run_dir)
        report = check_abi_conformance(result.executor_path)
        assert report.overall == "pass", report.to_dict()
